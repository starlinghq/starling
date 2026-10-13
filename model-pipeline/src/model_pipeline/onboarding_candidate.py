"""Shared source and environment verification for proposed model candidates."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import replace
from pathlib import Path, PurePosixPath

from jsonschema import Draft202012Validator

from model_pipeline import fixtures
from model_pipeline.onboarding_e5 import _environment_identity, _locked_version_matches
from model_pipeline.onboarding_runner import PipelineError, Stage

SCHEMA = Path(__file__).resolve().parents[2] / "schemas/onboarding-candidate.schema.json"
EXCLUDED = {"bin", "obj", "__pycache__", ".cache", ".git", ".venv", "artifacts"}


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for name, value in pairs:
        if name in result:
            raise PipelineError(f"duplicate JSON field: {name}")
        result[name] = value
    return result


def _relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value or str(path) != value:
        raise PipelineError(f"unsafe candidate path: {value!r}")
    return path


def write_json(path: Path, value: object) -> None:
    """Write deterministic, strictly finite JSON evidence."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def repository_input(root: Path, relative: str) -> Path:
    """Return a declared repository file, refusing traversal and escapes."""
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise PipelineError("unsafe repository input")
    result = root / path
    if not result.is_file() or not result.resolve().is_relative_to(root.resolve()):
        raise PipelineError(f"missing or escaping repository input: {relative}")
    return result


def guard_output(out: Path, sources: list[Path] | tuple[Path, ...], message: str) -> None:
    """Create a stage output only when it neither contains nor sits inside a read-only source."""
    output = out.resolve()
    for source in sources:
        resolved = source.resolve()
        if output.is_relative_to(resolved) or resolved.is_relative_to(output):
            raise PipelineError(f"{message}: {source}")
    out.mkdir(parents=True, exist_ok=True)


def load_candidate(path: Path) -> dict:
    try:
        if path.stat().st_size > 1024 * 1024:
            raise PipelineError("candidate contract exceeds 1 MiB")
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        problems = list(Draft202012Validator(schema).iter_errors(data))
    except (OSError, ValueError, RecursionError) as exc:
        raise PipelineError(f"cannot read candidate contract: {exc}") from exc
    if problems:
        raise PipelineError(f"invalid candidate contract: {problems[0].message}")
    paths = [record["path"] for record in data["origin"]["files"]]
    if len(paths) != len(set(paths)):
        raise PipelineError("candidate source lock has duplicate file paths")
    for value in paths:
        _relative(value)
    return data


def _source(snapshot: Path, relative: str) -> Path:
    path = snapshot / _relative(relative)
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(snapshot.resolve()) or not resolved.is_file():
            raise PipelineError(f"candidate source escapes snapshot or is not a file: {relative}")
        if any(parent.is_symlink() for parent in path.parents if parent.is_relative_to(snapshot)):
            raise PipelineError(f"candidate source has a symlink directory: {relative}")
        return resolved
    except (OSError, RuntimeError) as exc:
        raise PipelineError(f"missing candidate source: {relative}: {exc}") from exc


def verify_assets(candidate: dict, snapshot: Path, out: Path) -> dict:
    records = []
    for record in candidate["origin"]["files"]:
        path = _source(snapshot, record["path"])
        actual_size, digest = path.stat().st_size, fixtures.sha256_file(path)
        if (actual_size, digest) != (record["size"], record["sha256"]):
            raise PipelineError(
                f"candidate asset mismatch: {record['path']}: expected {record['size']} bytes "
                f"sha256 {record['sha256']}, found {actual_size} bytes sha256 {digest}"
            )
        records.append({**record, "verification": "pinned-size-and-sha256"})
    lock = {"profile": candidate["id"], "origin": candidate["origin"], "assets": records}
    write_json(out / "source-lock.json", lock)
    return {"source_files_verified": len(records), "reference_status": "proposed"}


def source_reader(stage: Stage, candidate: dict, snapshot: Path) -> Stage:
    """Recheck pins before raw source reads and detect lasting changes during execution."""
    original = stage.execute

    def checked(out: Path, dependencies: dict) -> dict:
        verify_assets(candidate, snapshot, out)
        result = original(out, dependencies)
        return {**result, "original_source_pins": "verified-before-read"}

    return replace(stage, inputs={**stage.inputs, "original_snapshot": snapshot}, execute=checked)


def environment_identity(reference_python: Path, root: Path, timeout: float) -> dict:
    if not reference_python.is_file():
        raise PipelineError(f"reference Python is missing: {reference_python}")
    return _environment_identity(reference_python, shutil.which("dotnet") or "dotnet", root, timeout)


def code_inputs(root: Path, *, files: tuple[str, ...] = (), directories: tuple[str, ...] = ()) -> dict[str, Path]:
    paths = {
        root / "model-pipeline/src/model_pipeline/onboarding_candidate.py",
        root / "model-pipeline/src/model_pipeline/onboarding_runner.py",
        root / "model-pipeline/src/model_pipeline/onboarding_e5.py",
        root / "model-pipeline/src/model_pipeline/fixtures.py",
        root / "model-pipeline/schemas/onboarding-candidate.schema.json",
        root / "model-pipeline/pyproject.toml",
    }
    for relative in files:
        path = root / _relative(relative)
        if not path.is_file():
            raise PipelineError(f"declared code input is missing: {relative}")
        paths.add(path)
    for relative in directories:
        directory = root / _relative(relative)
        if not directory.is_dir():
            raise PipelineError(f"declared source directory is missing: {relative}")
        for path in directory.rglob("*"):
            if set(path.relative_to(directory).parts) & EXCLUDED:
                continue
            if path.is_file():
                paths.add(path)
    for path in paths:
        if not path.resolve().is_relative_to(root.resolve()):
            raise PipelineError(f"code input escapes repository: {path}")
    return {str(path.relative_to(root)): path for path in sorted(paths)}


def _requirements(path: Path) -> dict[str, str]:
    required = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", line)
        if match:
            name = re.sub(r"[-_.]+", "-", match[1]).lower()
            if name in required:
                raise PipelineError(f"reference requirements have a duplicate normalized package: {name}")
            required[name] = match[2]
        elif line.strip() and not line.lstrip().startswith(("#", "--")):
            raise PipelineError("reference environment must use exact version requirements")
    if not required:
        raise PipelineError("reference requirements lock has no pinned packages")
    return required


def candidate_stages(
    recipe: dict,
    *,
    repo_root: Path,
    snapshot: Path,
    reference_python: Path,
    threads: int,
    timeout: float,
    source_files: tuple[str, ...] = (),
    source_directories: tuple[str, ...] = (),
) -> tuple[dict, list[Stage], dict[str, Path]]:
    root = repo_root.resolve()
    snapshot = snapshot.resolve(strict=True)
    python = Path(os.path.abspath(reference_python))
    if not snapshot.is_dir() or type(threads) is not int or not 1 <= threads <= 2 or timeout <= 0:
        raise PipelineError("candidate runs require a snapshot directory, one/two threads, and positive timeout")
    profile = root / _relative(recipe["manifest"])
    requirements = root / _relative(recipe["reference_environment"])
    corpus = root / _relative(recipe["reference_corpus"])
    for path in (profile, requirements, corpus):
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise PipelineError(f"missing or escaping repository input: {path}")
    candidate = load_candidate(profile)
    expected_packages = _requirements(requirements)
    runtime_package = "onnxruntime" if candidate["runtime"]["backend"] == "onnx" else "onnxruntime-genai"
    if expected_packages.get(runtime_package) != candidate["runtime"]["version"]:
        raise PipelineError(f"reference lock must pin {runtime_package} to the candidate runtime version")
    identity = environment_identity(python, root, timeout)
    shared = code_inputs(root, files=source_files, directories=source_directories)
    shared.update({"candidate": profile, "reference_requirements": requirements})
    sources = [
        snapshot,
        profile,
        requirements,
        corpus,
        root / "model-pipeline/protected",
    ]
    sources.extend(_source(snapshot, record["path"]) for record in candidate["origin"]["files"])

    def guard(out: Path) -> None:
        guard_output(out, sources, "candidate output overlaps read-only input")

    def assets(out: Path, _deps: dict) -> dict:
        guard(out)
        return verify_assets(candidate, snapshot, out)

    def environment(out: Path, _deps: dict) -> dict:
        guard(out)
        if identity["python"]["status"] != "available" or identity["dotnet"]["status"] != "available":
            raise PipelineError("reference Python and pinned .NET SDK must be available")
        installed = {
            re.sub(r"[-_.]+", "-", name).lower(): version
            for name, version in identity["python"]["identity"]["packages"]
        }
        mismatches = [
            f"{name}: expected {version}, installed {installed.get(name, 'missing')}"
            for name, version in expected_packages.items()
            if not _locked_version_matches(installed.get(name), version)
        ]
        if mismatches:
            raise PipelineError("reference environment differs from lock: " + "; ".join(mismatches))
        write_json(out / "environment.json", identity)
        return {"reference_environment": "verified", "runtime": candidate["runtime"]}

    parameters = {"candidate": candidate, "threads": threads, "timeout": timeout, "environment": identity}
    return (
        candidate,
        [
            Stage("assets", (), {"snapshot": snapshot, **shared}, parameters, assets),
            Stage("environment", ("assets",), shared, parameters, environment),
        ],
        shared,
    )
