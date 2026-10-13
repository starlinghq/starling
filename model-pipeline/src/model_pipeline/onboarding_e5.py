"""E5 CPU onboarding: proposals, protected fidelity, and a reviewable evidence bundle.

This plugin wraps the existing runners. Fresh reference files never become goldens, and
only the protected evaluator can provide G1/G2 results. No stage changes registry status.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any

from model_pipeline import evaluate, fixtures, manifest, protection
from model_pipeline.onboarding_runner import PipelineError, Stage, run_command
from model_pipeline.tokenizer_package import PINS as TOKENIZER_PINS
from model_pipeline.tokenizer_package import tokenizer_version

PLUGIN = "e5-onnx-cpu-v1"
# E5 owns its distribution stage; the shared candidate distribution must not be appended.
SHARED_DISTRIBUTION = False
PROFILE = "multilingual-e5-small/onnx-fp32-cpu"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
REFERENCE = "model-pipeline/reference/encoder/run_e5.py"
EXPORT = "model-pipeline/reference/encoder/stage_e5_export.py"
FIDELITY = "ai-dotnet/tools/Starlings.Fidelity.Tool/Starlings.Fidelity.Tool.csproj"
RELEASE = "tools/release-drill-ai.sh"
PACKAGES = ("torch", "transformers", "sentence-transformers", "tokenizers", "onnxruntime", "numpy", "sentencepiece")
ORT_VERSION = "1.30.0"
E5_GRAPH = {
    "inputs": [
        {"name": name, "type": "int64", "shape": ["batch_size", "sequence_length"]}
        for name in ("input_ids", "attention_mask", "token_type_ids")
    ],
    "outputs": [{"name": "last_hidden_state", "type": "float32", "shape": ["batch_size", "sequence_length", 384]}],
}

# The legacy reference runner has no thread argument. A local bootstrap constrains its
# existing PyTorch and ORT sessions without changing the runner or the model implementation.
BOUNDED_RUNNER = """import runpy, sys
import torch
import onnxruntime as ort
threads = int(sys.argv[1])
script = sys.argv[2]
torch.set_num_threads(threads)
torch.set_num_interop_threads(1)
original = ort.InferenceSession
def bounded_session(path, sess_options=None, *args, **kwargs):
    options = sess_options or ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return original(path, options, *args, **kwargs)
ort.InferenceSession = bounded_session
sys.argv = sys.argv[2:]
runpy.run_path(script, run_name="__main__")
"""


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _input(root: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or "\\" in relative:
        raise PipelineError(f"repository input must be relative without traversal: {relative!r}")
    path = root / rel
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise PipelineError(f"repository input is missing or escapes repository: {relative!r}")
    return path.resolve()


def _profile(recipe: dict, root: Path) -> tuple[Path, dict]:
    if recipe.get("plugin") != PLUGIN or recipe.get("recipe_version") != 1:
        raise PipelineError(f"unsupported plugin or recipe version; use recipe_version 1 and {PLUGIN}")
    path = _input(root, recipe["manifest"])
    problems = manifest.validate_file(path)
    if problems:
        raise PipelineError("invalid model manifest: " + "; ".join(str(p) for p in problems))
    man = manifest.load(path)
    defaults = man["behavior"].get("defaults", {})
    tokenizer = man["processing"]["tokenizer"]
    if (
        man["id"] != PROFILE
        or man["origin"]["weights"]["revision"] != REVISION
        or man["origin"]["code"]["revision"] != REVISION
        or man["execution"]["precision"] != "fp32"
        or man["execution"]["backend"]["name"] != "onnxruntime"
        or man["execution"]["backend"]["version"] != ORT_VERSION
        or man["execution"]["backend"].get("binding") != "Microsoft.ML.OnnxRuntime"
        or man["execution"]["variant"] != "reference"
        or man["execution"].get("graph") != E5_GRAPH
        or man["execution"].get("provider_options", {}) != {}
        or man["behavior"]["tasks"] != ["embedding"]
        or man["behavior"]["max_input_tokens"] != 512
        or man["behavior"]["offsets_unit"] != "none"
        or man["behavior"]["truncation"] != "truncate-right"
        or defaults.get("pooling") != "mean over attention mask"
        or defaults.get("normalize") != "l2"
        or defaults.get("dimension") != 384
        or defaults.get("input_prefix")
        != ('Callers add "query: " or "passage: " themselves (model card). The profile does not insert prefixes.')
        or tokenizer["package"] != "Starlings.Tokenizers"
        or tokenizer["profile_id"] != "xlmr-unigram-e5"
        or tokenizer.get("algorithm") != "unigram"
        or tokenizer["backend_tokenizer_dependency"] is not False
        or man["qualification"]["reference_runner"] != REFERENCE
        or man["qualification"]["fixtures"] != f"model-pipeline/protected/{PROFILE}"
        or man["qualification"]["tolerances"] != f"model-pipeline/protected/{PROFILE}/tolerances.yaml"
    ):
        raise PipelineError(
            f"unsupported profile {man['id']!r}: {PLUGIN} implements only pinned {PROFILE} at {REVISION}; "
            "inspect capabilities and implement a separate plugin for another profile"
        )
    _verify_dotnet_ort_contract(root, man["execution"]["backend"]["version"])
    _verify_tokenizer_version(root, tokenizer["version"])
    return path, man


def _verify_tokenizer_version(root: Path, version: str) -> None:
    """The manifest's tokenizer version must be the Starlings.Tokenizers version ai-dotnet pins exactly."""
    if not (root / TOKENIZER_PINS).exists():
        return  # Small manifest-only repositories can still be inspected without SDK/source.
    try:
        actual = tokenizer_version(root)
    except (OSError, ET.ParseError) as exc:
        raise PipelineError(f"cannot verify tokenizer package version: {exc}") from exc
    if actual != version:
        raise PipelineError(f"unsupported tokenizer version: manifest declares {version}, ai-dotnet pins {actual}")


def _verify_dotnet_ort_contract(root: Path, version: str) -> None:
    """The declaration must describe the centrally pinned packages the fixed tool builds."""
    props = root / "ai-dotnet/Directory.Packages.props"
    if not props.exists() and not (root / FIDELITY).exists():
        return  # Small manifest-only repositories can still be inspected without SDK/source.
    expected = {"Microsoft.ML.OnnxRuntime": f"[{version}]", "Microsoft.ML.OnnxRuntime.Managed": f"[{version}]"}
    try:
        declared = {
            node.attrib.get("Include"): node.attrib.get("Version") for node in ET.parse(props).iter("PackageVersion")
        }
        for package, pin in expected.items():
            if declared.get(package) != pin:
                raise PipelineError(
                    f"unsupported profile backend: {package} must be centrally pinned to {pin}; "
                    f"found {declared.get(package)!r}"
                )
        locks = (
            (root / FIDELITY).parent / "packages.lock.json",
            root / "ai-dotnet/src/Starlings.Runtime.Onnx/packages.lock.json",
        )
        for lock in locks:
            if not lock.exists():
                continue
            targets = json.loads(lock.read_text(encoding="utf-8"))["dependencies"]
            for target, dependencies in targets.items():
                for package in expected:
                    entry = dependencies.get(package)
                    if entry is not None and entry.get("resolved") != version:
                        raise PipelineError(
                            f"unsupported profile backend: {lock.relative_to(root)} {target} "
                            f"resolves {package} to {entry.get('resolved')!r}, expected {version}"
                        )
    except (OSError, ET.ParseError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PipelineError(f"cannot verify declared ONNX Runtime package contract: {exc}") from exc


def describe_plan(recipe: dict, *, repo_root: Path) -> dict:
    """Describe the plugin without probing Python/.NET or requiring a model snapshot."""
    root = repo_root.resolve()
    _, man = _profile(recipe, root)
    return {
        "plugin": PLUGIN,
        "recipe": recipe["id"],
        "profile": man["id"],
        "revision": man["origin"]["weights"]["revision"],
        "asset_count": len(man["origin"]["files"]),
        "stages": ["assets", "environment", "reference", "export", "dotnet", "evaluate", "review"],
        "optional_stages": ["distribution"],
        "required_gates": ["G1", "G2"],
        "reference_status": "proposed; requires separate protected-fixture adoption review",
        "release_status": "incomplete; this pipeline does not approve releases or change support status",
        "dependencies": {"python_packages": list(PACKAGES), "dotnet_sdk": "global.json"},
    }


def _probe(argv: list[str], *, root: Path, timeout: float) -> dict:
    """Small read-only toolchain probe; failures become actionable stage failures."""
    with tempfile.TemporaryDirectory(prefix="onboarding-probe-") as temporary:
        log = Path(temporary) / "probe.log"
        try:
            run_command(argv, cwd=root, log_path=log, timeout=min(timeout, 60))
        except PipelineError as exc:
            return {"status": "unavailable", "error": str(exc), "argv": argv}
        if log.stat().st_size > 1024 * 1024:
            return {"status": "unavailable", "error": "toolchain probe output exceeds 1 MiB", "argv": argv}
        # run_command writes one command-header line before the combined process output.
        output = log.read_text(encoding="utf-8").partition("\n")[2].strip()
        return {"status": "available", "stdout": output, "argv": argv}


def _environment_identity(python: Path, dotnet: str, root: Path, timeout: float) -> dict:
    script = (
        "import importlib.metadata as m,json,platform,sys; "
        "print(json.dumps({'python':sys.version,'executable':sys.executable,'prefix':sys.prefix,"
        "'platform':platform.platform(),'packages':sorted((d.metadata['Name'],d.version) "
        "for d in m.distributions() if d.metadata['Name'])},sort_keys=True))"
    )
    py = _probe([str(python), "-c", script], root=root, timeout=timeout)
    if py["status"] == "available":
        try:
            py["identity"] = json.loads(py.pop("stdout"))
        except (json.JSONDecodeError, KeyError) as exc:
            py = {"status": "unavailable", "error": f"invalid Python environment probe: {exc}"}
    dotnet_info = _probe([dotnet, "--info"], root=root, timeout=timeout)
    dotnet_sdk = _probe([dotnet, "--version"], root=root, timeout=timeout)
    if dotnet_sdk["status"] != "available":
        dotnet_info = {**dotnet_info, "status": "unavailable", "error": dotnet_sdk.get("error"), "sdk": dotnet_sdk}
    else:
        dotnet_info["sdk"] = dotnet_sdk
    return {
        "python": py,
        "dotnet": dotnet_info,
        "host": {"platform": platform.platform(), "machine": platform.node(), "cpu_count": os.cpu_count()},
        "orchestrator_python": sys.version,
        "orchestrator_packages": fixtures.environment(["model-pipeline", "pyyaml", "jsonschema"]),
    }


def _code_inputs(root: Path, stage: str) -> dict[str, Path]:
    """Fingerprint each runner's source/dependencies, excluding generated build trees."""
    package = root / "model-pipeline/src/model_pipeline"
    paths = {
        root / "model-pipeline/pyproject.toml",
        root / "model-pipeline/schemas/model-manifest.schema.json",
        *(
            package / name
            for name in ("__init__.py", "manifest.py", "fixtures.py", "onboarding_e5.py", "onboarding_runner.py")
        ),
    }
    if stage in {"export", "evaluate", "review"}:
        paths.update({package / "results.py", root / "model-pipeline/schemas/stage-results.schema.json"})
    if stage in {"evaluate", "review"}:
        paths.update(package / name for name in ("evaluate.py", "evaluate_extraction.py", "protection.py"))
    if stage in {"dotnet", "environment", "distribution"}:
        paths.update(
            {
                root / "global.json",
                root / "Directory.Build.props",
                root / "Directory.Build.targets",
                root / "NuGet.Config",
                root / "nuget.config",
            }
        )
    if stage == "distribution":
        for directory in ("ai-dotnet/src", "ai-dotnet/samples", "tools/release-drill"):
            for path in (root / directory).rglob("*"):
                if set(path.relative_to(root / directory).parts) & {"bin", "obj", "__pycache__", "artifacts"}:
                    continue
                if path.is_file():
                    paths.add(path)
        paths.update(path for path in (root / "ai-dotnet").glob("Directory.*") if path.is_file())
        paths.add(root / "model-pipeline/registry/gliner2-base-v1/onnx-fp32-cpu/manifest.yaml")
    if stage == "dotnet":
        pending = [root / FIDELITY]
        projects = set()
        while pending:
            project = pending.pop().resolve()
            if project in projects or not project.is_file():
                continue
            if not project.is_relative_to(root):
                raise PipelineError(f"project dependency escapes repository: {project}")
            projects.add(project)
            pending.extend(
                (project.parent / ref.attrib["Include"]).resolve()
                for ref in ET.parse(project).iter("ProjectReference")
                if "Include" in ref.attrib
            )
            for path in project.parent.rglob("*"):
                if set(path.relative_to(project.parent).parts) & {"bin", "obj"}:
                    continue
                if path.is_file() and (
                    path.suffix in {".cs", ".csproj", ".props", ".targets"} or path.name == "packages.lock.json"
                ):
                    paths.add(path)
            for parent in (project.parent, *project.parent.parents):
                if not parent.is_relative_to(root):
                    break
                paths.update(
                    parent / name
                    for name in (
                        "Directory.Build.props",
                        "Directory.Build.targets",
                        "Directory.Packages.props",
                        "NuGet.Config",
                        "nuget.config",
                    )
                )
    return {str(p.relative_to(root)): p for p in sorted(paths) if p.is_file()}


def _locked_version_matches(actual: str | None, expected: str) -> bool:
    # PEP 440: a requirement without a local version also accepts its local builds.
    # The lock uses canonical numeric versions; no general version parser is needed.
    return actual == expected or (actual is not None and "+" not in expected and actual.split("+", 1)[0] == expected)


def _assert_committed_code(root: Path, paths: set[Path], timeout: float) -> None:
    relative = sorted(str(path.relative_to(root)) for path in paths)
    for argv in (
        ["git", "ls-files", "--error-unmatch", "--", *relative],
        ["git", "diff", "--quiet", "HEAD", "--", *relative],
    ):
        if _probe(argv, root=root, timeout=timeout)["status"] != "available":
            raise PipelineError(
                "distribution requires relevant pipeline, reference, and .NET code to match "
                "committed HEAD; commit the candidate implementation before the clean-consumer drill"
            )


def build_stages(
    recipe: dict,
    *,
    repo_root: Path,
    snapshot: Path,
    reference_python: Path,
    threads: int = 2,
    timeout: float = 1800,
    distribution: bool = False,
) -> list[Stage]:
    """Build deterministic stages; toolchain probes participate in every cache identity."""
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 64:
        raise PipelineError("threads must be an integer from 1 through 64")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 86400:
        raise PipelineError("timeout must be positive and at most 86400 seconds")
    root, snapshot = repo_root.resolve(), snapshot.resolve()
    # Preserve the venv path: resolving bin/python's symlink would select the base interpreter.
    python = Path(os.path.abspath(reference_python))
    man_path, man = _profile(recipe, root)
    corpus = _input(root, recipe["reference_corpus"])
    requirements = _input(root, recipe["reference_environment"])
    protected = protection.resolve_protected(root, man["qualification"]["fixtures"], kind="fixtures")
    if not snapshot.is_dir():
        raise PipelineError(f"model snapshot directory is missing: {snapshot}; acquire the pinned assets first")
    if snapshot.is_relative_to(root / "model-pipeline/protected"):
        raise PipelineError("model snapshots cannot be protected evidence directories")
    dotnet = shutil.which("dotnet") or "dotnet"
    identity = _environment_identity(python, dotnet, root, timeout)
    repository_identity = fixtures.git_head(root)
    head = repository_identity.removesuffix("-dirty") if repository_identity else None
    base, base_detail = protection.resolve_base(root, "auto")
    common = {
        "plugin": PLUGIN,
        "recipe": recipe,
        "threads": threads,
        "timeout": timeout,
        "environment": identity,
    }
    distribution_inputs = set(_code_inputs(root, "dotnet").values()) | set(_code_inputs(root, "evaluate").values())
    distribution_inputs.update(_code_inputs(root, "distribution").values())
    distribution_inputs.update({man_path, corpus, requirements, root / REFERENCE, root / EXPORT, root / RELEASE})
    distribution_inputs.update(path for path in protected.rglob("*") if path.is_file())
    if distribution:
        _assert_committed_code(root, distribution_inputs, timeout)
    snapshot_links = {
        f"snapshot-link:{path.relative_to(snapshot)}": path
        for path in snapshot.rglob("*")
        if path.is_symlink() and path.is_file()
    }
    env = {
        "PYTHONPATH": str(root / "model-pipeline/src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
        "TOKENIZERS_PARALLELISM": "false",
        "DOTNET_PROCESSOR_COUNT": str(threads),
        "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }

    def guard(out: Path) -> None:
        output = out.resolve()
        for source in (
            snapshot,
            root / "model-pipeline/protected",
            *snapshot_links.values(),
        ):
            source = source.resolve()
            if output.is_relative_to(source) or source.is_relative_to(output):
                raise PipelineError(f"stage output overlaps read-only source/evidence: {output} and {source}")
        out.mkdir(parents=True, exist_ok=True)

    def stage(name: str, dependencies: tuple[str, ...], inputs: dict, execute, **parameters) -> Stage:
        return Stage(name, dependencies, {**_code_inputs(root, name), **inputs}, {**common, **parameters}, execute)

    def assets(out: Path, deps: dict) -> dict:
        guard(out)
        records = []
        declared = set()
        for record in man["origin"]["files"]:
            rel = PurePosixPath(record["path"])
            if rel.is_absolute() or ".." in rel.parts or "\\" in record["path"]:
                raise PipelineError(f"unsafe manifest asset path: {record['path']!r}")
            path = snapshot / rel
            if not path.is_file() or any(
                parent.is_symlink() for parent in path.parents if parent != snapshot and parent.is_relative_to(snapshot)
            ):
                raise PipelineError(f"missing asset or symlink directory in snapshot: {record['path']}")
            size, digest = path.stat().st_size, fixtures.sha256_file(path)
            if size != record["size"] or digest != record["sha256"]:
                raise PipelineError(
                    f"asset mismatch for {record['path']}: expected size={record['size']} sha256={record['sha256']}; "
                    f"actual size={size} sha256={digest}; acquire the exact pinned snapshot"
                )
            records.append({**record, "verification": "manifest-size-and-sha256"})
            declared.add(record["path"])
        # Ancillary snapshot files may be read by transformers. Record their bytes, without
        # implying the manifest pins them. The complete snapshot is also a cache input.
        ancillary = []
        for path in sorted(snapshot.rglob("*")):
            if path.is_symlink() and not path.is_file():
                raise PipelineError(f"ancillary asset is a broken or directory symlink: {path.relative_to(snapshot)}")
            if not path.is_file() or str(path.relative_to(snapshot)) in declared:
                continue
            ancillary.append(
                {
                    "path": str(path.relative_to(snapshot)),
                    "size": path.stat().st_size,
                    "sha256": fixtures.sha256_file(path),
                    "verification": "observed-only; not manifest-pinned",
                }
            )
        lock = {"profile": man["id"], "origin": man["origin"], "assets": records, "ancillary_assets": ancillary}
        _json(out / "source-lock.json", lock)
        return {"manifest_assets_verified": len(records), "ancillary_assets_observed": len(ancillary)}

    def environment(out: Path, deps: dict) -> dict:
        guard(out)
        _json(out / "environment.json", identity)
        for tool in ("python", "dotnet"):
            if identity[tool]["status"] != "available":
                raise PipelineError(
                    f"missing dependency {tool}: {identity[tool].get('error')}; select a reference "
                    "Python environment and install the SDK pinned by global.json"
                )
        installed = {re.sub(r"[-_.]+", "-", n).lower(): v for n, v in identity["python"]["identity"]["packages"]}
        missing = [name for name in PACKAGES if name not in installed]
        if missing:
            raise PipelineError(f"reference Python is missing packages: {', '.join(missing)}; install {requirements}")
        expected = {
            re.sub(r"[-_.]+", "-", match[1]).lower(): match[2]
            for line in requirements.read_text(encoding="utf-8").splitlines()
            if (match := re.match(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)", line))
        }
        mismatches = [
            f"{name}: expected {version}, installed {installed.get(name, 'missing')}"
            for name, version in expected.items()
            if not _locked_version_matches(installed.get(name), version)
        ]
        if mismatches:
            raise PipelineError("reference environment differs from the recipe lock: " + "; ".join(mismatches))
        return {"toolchain_verified": True, "packages": {name: installed[name] for name in PACKAGES}}

    def python_run(out: Path, script: Path, args: list[str]) -> dict:
        guard(out)
        bootstrap = out / "bounded-runner.py"
        bootstrap.write_text(BOUNDED_RUNNER, encoding="utf-8")
        return run_command(
            [str(python), str(bootstrap), str(threads), str(script), *args],
            cwd=root,
            log_path=out / "command.log",
            env=env,
            timeout=timeout,
        )

    def reference(out: Path, deps: dict) -> dict:
        result = python_run(
            out,
            root / REFERENCE,
            [
                "--model-dir",
                str(snapshot),
                "--revision",
                REVISION,
                "--corpus",
                str(corpus),
                "--out",
                str(out),
            ],
        )
        _json(
            out / "proposal.json",
            {
                "status": "proposed",
                "automatically_adopted": False,
                "fixture_review": "Separate maintainer review is required for adoption.",
            },
        )
        return {**result, "reference_status": "proposed", "automatically_adopted": False}

    def export(out: Path, deps: dict) -> dict:
        return python_run(
            out,
            root / EXPORT,
            [
                "--model-dir",
                str(snapshot),
                "--fixtures",
                str(protected),
                "--profile",
                PROFILE,
                "--out",
                str(out),
                "--intra-op-threads",
                str(threads),
            ],
        )

    def dotnet_run(out: Path, deps: dict) -> dict:
        guard(out)
        with tempfile.TemporaryDirectory(prefix="dotnet-build-", dir=out) as build:
            return run_command(
                [
                    dotnet,
                    "run",
                    "--project",
                    str(root / FIDELITY),
                    "-c",
                    "Release",
                    "--no-launch-profile",
                    "--disable-build-servers",
                    "--artifacts-path",
                    build,
                    "-p:RestoreLockedMode=true",
                    "--",
                    "e5",
                    "--snapshot",
                    str(snapshot),
                    "--fixtures",
                    str(protected),
                    "--profile",
                    PROFILE,
                    "--out",
                    str(out),
                    "--hidden",
                    "--threads",
                    str(threads),
                ],
                cwd=root,
                log_path=out / "command.log",
                env=env,
                timeout=timeout,
            )

    def evaluation(out: Path, deps: dict) -> dict:
        guard(out)
        collected = out / "results"
        collected.mkdir()
        # Sidecar paths are relative to the result JSON; keep them alongside copied results.
        for name in ("export", "dotnet"):
            for path in deps[name].iterdir():
                if path.name.endswith(".results.json") or path.suffix == ".f32":
                    shutil.copyfile(path, collected / path.name)
        result = evaluate.evaluate(
            man_path, collected, repo_root=root, protected_base="auto", require_gates=("G1", "G2")
        )
        evaluate.write_report(result, out)
        if result.exit_code or any(result.report["gates"][g]["status"] != "passed" for g in ("G1", "G2")):
            raise PipelineError(f"protected fidelity failed (exit {result.exit_code}); see {out / 'report.json'}")
        return {"gates": result.report["gates"], "report": "report.json", "protected_base": base}

    def distribution_run(out: Path, deps: dict) -> dict:
        guard(out)
        if head is None:
            raise PipelineError("clean-consumer distribution drill requires a committed Git HEAD")
        _assert_committed_code(root, distribution_inputs, timeout)
        # The existing drill writes artifacts relative to its repository. Run the committed
        # script in a temporary clean clone so it cannot mutate shared worktree artifacts.
        with tempfile.TemporaryDirectory(prefix="committed-drill-", dir=out) as temporary:
            clone = Path(temporary) / "source"
            run_command(
                ["git", "clone", "--quiet", "--no-hardlinks", str(root), str(clone)],
                cwd=root,
                log_path=out / "clone.log",
                timeout=timeout,
            )
            run_command(
                ["git", "-c", "advice.detachedHead=false", "checkout", "--quiet", "--detach", head],
                cwd=clone,
                log_path=out / "checkout.log",
                timeout=timeout,
            )
            drill_env = {
                k: v for k, v in env.items() if k not in {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONPATH"}
            }
            result = run_command(
                ["bash", str(clone / RELEASE), "--seed-model-from", str(snapshot)],
                cwd=clone,
                log_path=out / "command.log",
                env=drill_env,
                timeout=timeout,
            )
            shutil.copytree(clone / "artifacts/release-drill-ai", out / "artifacts")
        record = {
            "status": "passed",
            "scope": "E5 real-model clean consumer on current host",
            "candidate_commit": head,
            "mode": "seed-model-from",
            "command": result,
        }
        _json(out / "distribution.json", record)
        return record

    def review(out: Path, deps: dict) -> dict:
        guard(out)
        report = json.loads((deps["evaluate"] / "report.json").read_text(encoding="utf-8"))
        for gate in ("G1", "G2"):
            if report["gates"][gate]["status"] != "passed":
                raise PipelineError(f"review requires passed protected gate {gate}")
        dist = (
            json.loads((deps["distribution"] / "distribution.json").read_text(encoding="utf-8"))
            if distribution
            else {"status": "not-run", "reason": "enable distribution for the committed clean-consumer drill"}
        )
        data = {
            "format": "starlings.onboarding-review",
            "format_version": 1,
            "recipe": recipe,
            "profile": PROFILE,
            "candidate_commit": head,
            "reference": {"status": "proposed", "automatically_adopted": False},
            "gates": report["gates"],
            "protected_evidence": report["protected_evidence"],
            "fidelity_inputs": report["inputs"],
            "distribution": dist,
            "release_qualification": {
                "status": "incomplete",
                "not_qualified_by_pipeline": ["G0", "G3", "G4", "G5", "G7"],
            },
            "capabilities": man["capabilities"],
            "support_status_changed": False,
            "source_lock": json.loads((deps["assets"] / "source-lock.json").read_text(encoding="utf-8")),
            "environment": identity,
        }
        _json(out / "review.json", data)
        shutil.copyfile(deps["evaluate"] / "report.json", out / "fidelity-report.json")
        shutil.copyfile(deps["evaluate"] / "report.md", out / "fidelity-report.md")
        (out / "review.md").write_text(
            f"# Onboarding review: {PROFILE}\n\n"
            f"Candidate commit: `{head}`. Upstream revision: `{REVISION}`.\n\n"
            "Protected evaluator: **G1 passed; G2 passed**. "
            "See fidelity-report.md and review.json for exact evidence.\n\n"
            "Fresh reference outputs are **proposals**. Adoption requires separate maintainer review; "
            "no goldens were promoted.\n\n"
            f"Clean-consumer distribution drill: **{dist['status']}** (committed code, real model when run).\n\n"
            "Release qualification is **incomplete**. "
            "This run does not qualify G0, G3, G4, G5, or G7, approve a release, "
            "or promote support status. Declared capabilities retain their manifest statuses.\n",
            encoding="utf-8",
        )
        return {"gates": report["gates"], "release_qualification": "incomplete", "reference_status": "proposed"}

    stages = [
        stage("assets", (), {"manifest": man_path, "snapshot": snapshot, **snapshot_links}, assets),
        stage("environment", ("assets",), {"requirements": requirements}, environment),
        stage(
            "reference",
            ("assets", "environment"),
            {"runner": root / REFERENCE, "corpus": corpus},
            reference,
            bootstrap=BOUNDED_RUNNER,
        ),
        stage(
            "export",
            ("reference", "assets", "environment"),
            {"runner": root / EXPORT, "fixtures": protected},
            export,
            bootstrap=BOUNDED_RUNNER,
        ),
        stage("dotnet", ("assets", "environment"), {"fixtures": protected}, dotnet_run),
        stage(
            "evaluate",
            ("reference", "export", "dotnet"),
            {"manifest": man_path, "fixtures": protected},
            evaluation,
            protected_base=base,
            protected_base_detail=base_detail,
        ),
    ]
    if distribution:
        stages.append(
            stage(
                "distribution",
                ("evaluate", "assets", "environment"),
                {"script": root / RELEASE},
                distribution_run,
                candidate_commit=head,
            )
        )
    stages.append(
        stage(
            "review",
            ("assets", "reference", "evaluate", *(("distribution",) if distribution else ())),
            {"manifest": man_path},
            review,
            candidate_commit=head,
            distribution=distribution,
        )
    )
    return stages
