"""Small, content-verified stage runner for local onboarding recipes.

Callbacks are trusted repository code, not a sandbox. They must write only into
the supplied output directory, and declare code/environment dependencies as
inputs or parameters. Attempts are immutable after completion; retries retain
the failed attempt and its command logs for investigation.
"""

from __future__ import annotations

import fcntl
import hashlib
import inspect
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class PipelineError(RuntimeError):
    """Invalid pipeline configuration or a bounded command failure."""

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


@dataclass(frozen=True)
class Stage:
    name: str
    dependencies: Sequence[str]
    inputs: Mapping[str, Path]
    parameters: Any
    execute: Callable[[Path, dict[str, Path]], Any]


def _json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PipelineError(f"Value must be JSON-compatible: {exc}") from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(stream.fileno())
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise PipelineError(f"File changed while computing its digest: {path}")
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _no_symlink(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    for part in [*reversed(absolute.parents), absolute]:
        if part.is_symlink():
            raise PipelineError(f"Output path cannot contain a symlink: {part}")
    return absolute


def _tree(path: Path, *, output: bool = False) -> dict:
    """Hash every entry; only Python bytecode caches are excluded from inputs.

    Build directories are deliberately included: callers should declare precise
    source files rather than trees containing mutable bin/obj/build products.
    Input symlinks (e.g. immutable Hugging Face snapshots) follow regular files
    and record both their target and bytes. Directory symlinks are rejected.
    """
    if not path.exists():
        raise PipelineError(f"Missing {'output' if output else 'input'}: {path}")
    entries = {}

    def visit(item: Path, relative: str) -> None:
        mode = item.lstat().st_mode
        link = None
        if stat.S_ISLNK(mode):
            if output or not item.is_file():
                raise PipelineError(f"Unsafe symlink in {'output' if output else 'input'}: {item}")
            link = os.readlink(item)
            mode = item.stat().st_mode
        if stat.S_ISREG(mode):
            if output and item.stat().st_nlink != 1:
                raise PipelineError(f"Output file cannot be a hard link: {item}")
            entry = {"type": "file", "size": item.stat().st_size, "sha256": _file_sha(item)}
            if link is not None:
                entry["symlink_target"] = link
            entries[relative] = entry
        elif stat.S_ISDIR(mode):
            entries[relative] = {"type": "directory"}
            for child in sorted(item.iterdir()):
                if not output and (child.name == "__pycache__" or child.suffix in {".pyc", ".pyo"}):
                    continue
                visit(child, f"{relative}/{child.name}" if relative != "." else child.name)
        else:
            raise PipelineError(f"Only regular files and directories are allowed: {item}")

    visit(path, ".")
    if output and not any(entry["type"] == "file" for entry in entries.values()):
        raise PipelineError(f"Stage produced no evidence files: {path}")
    return entries


def _ordered(stages: Sequence[Stage]) -> list[Stage]:
    names = {}
    for stage in stages:
        if (
            not isinstance(stage, Stage)
            or not isinstance(stage.name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", stage.name)
        ):
            raise PipelineError("Stage names must be nonempty letters/digits/underscores/hyphens")
        if stage.name in names:
            raise PipelineError(f"Duplicate stage name: {stage.name}")
        if not callable(stage.execute):
            raise PipelineError(f"Stage {stage.name} has no executable callback")
        if (
            not isinstance(stage.dependencies, Sequence)
            or isinstance(stage.dependencies, str)
            or any(not isinstance(name, str) for name in stage.dependencies)
            or len(set(stage.dependencies)) != len(stage.dependencies)
        ):
            raise PipelineError(f"Stage {stage.name} dependencies must be a sequence of unique names")
        if not isinstance(stage.inputs, Mapping) or any(not isinstance(key, str) for key in stage.inputs):
            raise PipelineError(f"Stage {stage.name} inputs must map logical names to paths")
        if any(not isinstance(path, (str, os.PathLike)) for path in stage.inputs.values()):
            raise PipelineError(f"Stage {stage.name} inputs must be filesystem paths")
        _json(stage.parameters)
        names[stage.name] = stage
    if not names:
        raise PipelineError("Pipeline must contain at least one stage")
    result, visiting, visited = [], set(), set()

    def visit(name: str) -> None:
        if name not in names:
            raise PipelineError(f"Unknown stage dependency: {name}")
        if name in visiting:
            raise PipelineError(f"Dependency cycle includes stage: {name}")
        if name in visited:
            return
        visiting.add(name)
        for dependency in names[name].dependencies:
            visit(dependency)
        visiting.remove(name)
        visited.add(name)
        result.append(names[name])

    for name in names:
        visit(name)
    return result


def plan_stages(stages: Sequence[Stage]) -> list[dict]:
    """Validate and describe dependency order without reading model assets."""
    return [
        {
            "name": stage.name,
            "dependencies": list(stage.dependencies),
            "inputs": {key: str(path) for key, path in stage.inputs.items()},
            "parameters": stage.parameters,
        }
        for stage in _ordered(stages)
    ]


def _code_identity(stage: Stage) -> dict:
    try:
        source = inspect.getsource(stage.execute)
        module_file = inspect.getsourcefile(stage.execute)
    except (OSError, TypeError) as exc:
        raise PipelineError(f"Stage {stage.name} callback must have inspectable repository source") from exc
    if not module_file or not Path(module_file).is_file():
        raise PipelineError(f"Stage {stage.name} callback must have a source file")
    return {
        "engine_sha256": _file_sha(Path(__file__)),
        "callback_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "callback_module": str(Path(module_file).resolve()),
        "callback_module_sha256": _file_sha(Path(module_file)),
    }


def _atomic_json(path: Path, value: Any) -> None:
    _no_symlink(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _validate_run_dir(run_dir: Path, stages: Sequence[Stage]) -> Path:
    run_dir = _no_symlink(run_dir)
    repository = Path(__file__).resolve().parents[3]
    forbidden = [repository / "model-pipeline/protected"]
    for stage in stages:
        for path in stage.inputs.values():
            forbidden.extend([Path(path).absolute(), Path(path).resolve()])
    for path in forbidden:
        if _overlap(run_dir, path):
            raise PipelineError(f"Run directory overlaps an input or protected evidence: {path}")
    return run_dir


def _failure(exc: BaseException) -> dict:
    result = {"type": type(exc).__name__, "message": str(exc)}
    if isinstance(exc, PipelineError):
        result["details"] = exc.details
    return result


def _cache_valid(previous: Any, fingerprint: str, run_dir: Path, stage: Stage) -> bool:
    if not isinstance(previous, dict) or previous.get("status") != "completed":
        return False
    if previous.get("fingerprint") != fingerprint or not isinstance(previous.get("outputs"), dict):
        return False
    try:
        if (
            previous.get("name") != stage.name
            or previous.get("dependencies") != list(stage.dependencies)
            or _sha(previous["identity"]) != fingerprint
        ):
            return False
        output = _no_symlink(Path(previous["output_dir"]))
        parent = run_dir / "stages" / stage.name
        if output.name != "output" or output.parent.parent != parent:
            return False
        _json(previous["metadata"])
        if previous.get("metadata_sha256") != _sha(previous["metadata"]):
            return False
        return previous["outputs"] == _tree(output, output=True)
    except (KeyError, OSError, PipelineError, TypeError, ValueError):
        return False


def verify_run_outputs(record: dict, run_dir: Path) -> None:
    """Verify local completion and exact output integrity before exporting evidence.

    This validates a local run record, not a signed attestation. It deliberately
    does not re-read source inputs: a completed run can be reviewed after source
    code moves on. Output files, metadata, and dependency identities must agree
    with the recorded fingerprints and digest maps.
    """
    run_dir = _no_symlink(Path(run_dir))
    if (
        not isinstance(record, dict)
        or record.get("run_version") != 1
        or record.get("status") != "completed"
        or not isinstance(record.get("stages"), dict)
        or not record["stages"]
    ):
        raise PipelineError("Only a completed, version-1 run with evidence stages can be verified")
    try:
        _json(record)
        _no_symlink(run_dir / "run.json")
        if record["run_dir"] != str(run_dir):
            raise PipelineError("Run record directory differs from the requested run directory")
        for name, entry in record["stages"].items():
            if (
                not isinstance(name, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name)
                or not isinstance(entry, dict)
                or entry.get("name") != name
                or entry.get("status") != "completed"
            ):
                raise PipelineError(f"Invalid or incomplete stage record: {name!r}")
            fingerprint = entry["fingerprint"]
            if (
                not isinstance(fingerprint, str)
                or not re.fullmatch(r"[a-f0-9]{64}", fingerprint)
                or _sha(entry["identity"]) != fingerprint
                or _sha(entry["metadata"]) != entry["metadata_sha256"]
            ):
                raise PipelineError(f"Stage {name} has corrupt fingerprint or metadata")
            dependencies = entry["dependencies"]
            if (
                not isinstance(dependencies, list)
                or any(not isinstance(dependency, str) for dependency in dependencies)
                or len(set(dependencies)) != len(dependencies)
                or name in dependencies
                or any(dependency not in record["stages"] for dependency in dependencies)
            ):
                raise PipelineError(f"Stage {name} has invalid dependencies")
            expected_dependencies = {
                dependency: {
                    "fingerprint": record["stages"][dependency]["fingerprint"],
                    "outputs": record["stages"][dependency]["outputs"],
                }
                for dependency in dependencies
            }
            if entry["identity"]["name"] != name or entry["identity"]["dependencies"] != expected_dependencies:
                raise PipelineError(f"Stage {name} dependency identity differs from the run")
            output = _no_symlink(Path(entry["output_dir"]))
            if (
                output.name != "output"
                or not re.fullmatch(r"[a-f0-9]{32}", output.parent.name)
                or output.parent.parent != run_dir / "stages" / name
                or not output.is_dir()
            ):
                raise PipelineError(f"Stage {name} output is not a run-local attempt directory")
            if not isinstance(entry["outputs"], dict) or _tree(output, output=True) != entry["outputs"]:
                raise PipelineError(f"Stage {name} output evidence is missing, changed, or has extra files")
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise PipelineError(f"Invalid run record or missing output: {exc}") from exc


def run_pipeline(stages: Sequence[Stage], run_dir: Path, *, provenance: Any, resume: bool = False) -> dict:
    """Execute a validated DAG, recording failures and reusing verified evidence.

    A stage can resume only when *all* of its dependencies resumed; a dependency
    rerun invalidates dependents even if it reproduces identical output bytes.
    KeyboardInterrupt is recorded and re-raised so command-line cancellation
    keeps its normal semantics. Hard-killed running stages are retried on resume.
    """
    stages = _ordered(stages)
    _json(provenance)
    run_dir = _validate_run_dir(Path(run_dir), stages)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = _no_symlink(run_dir / ".pipeline.lock")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(lock_fd).st_mode):
            raise PipelineError("Run directory lock must be a regular file")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PipelineError(f"Another writer holds the run directory lock: {run_dir}") from exc
        return _run_locked(stages, run_dir, provenance=provenance, resume=resume)
    finally:
        os.close(lock_fd)


def _run_locked(stages: Sequence[Stage], run_dir: Path, *, provenance: Any, resume: bool) -> dict:
    record_path = _no_symlink(run_dir / "run.json")
    previous = {}
    if record_path.exists():
        if not resume:
            raise PipelineError("Run record already exists; choose another directory or enable resume")
        try:
            previous = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PipelineError(f"Cannot read existing run record: {exc}") from exc
        if (
            not isinstance(previous, dict)
            or previous.get("run_version") != 1
            or not isinstance(previous.get("stages"), dict)
        ):
            raise PipelineError("Existing run record has an invalid structure")
        _atomic_json(run_dir / "history" / f"{uuid.uuid4().hex}.json", previous)
    started = time.monotonic()
    record = {
        "run_version": 1,
        "run_id": uuid.uuid4().hex,
        "run_dir": str(run_dir),
        "status": "running",
        "started_at": _now(),
        "provenance": provenance,
        "stages": {},
    }
    _atomic_json(record_path, record)
    for stage in stages:
        stage_started = time.monotonic()
        entry = {"name": stage.name, "dependencies": list(stage.dependencies), "started_at": _now()}
        record["stages"][stage.name] = entry
        if any(record["stages"][name]["status"] != "completed" for name in stage.dependencies):
            entry.update(
                status="blocked",
                cache_status="not-run",
                failure={"message": "Dependency failed"},
                ended_at=_now(),
                duration_seconds=0,
            )
            _atomic_json(record_path, record)
            continue
        print(f"Onboarding stage {stage.name}: checking inputs", file=sys.stderr, flush=True)
        try:
            inputs = {
                key: {"path": str(Path(path).absolute()), "entries": _tree(Path(path))}
                for key, path in stage.inputs.items()
            }
            dependencies = {
                name: {
                    "fingerprint": record["stages"][name]["fingerprint"],
                    "outputs": record["stages"][name]["outputs"],
                }
                for name in stage.dependencies
            }
            identity = {
                "inputs": inputs,
                "parameters": stage.parameters,
                "provenance": provenance,
                "code": _code_identity(stage),
                "dependencies": dependencies,
                "name": stage.name,
            }
            fingerprint = _sha(identity)
            entry.update(fingerprint=fingerprint, identity=identity)
            cached = previous.get("stages", {}).get(stage.name)
            dependencies_cached = all(record["stages"][name]["cache_status"] == "cached" for name in stage.dependencies)
            if resume and dependencies_cached and _cache_valid(cached, fingerprint, run_dir, stage):
                for key, path in stage.inputs.items():
                    if _tree(Path(path)) != inputs[key]["entries"]:
                        raise PipelineError(f"Stage {stage.name} input changed during cache verification: {key}")
                entry.update(cached)
                entry.update(
                    cache_status="cached", resumed_at=_now(), resume_duration_seconds=time.monotonic() - stage_started
                )
                print(f"Onboarding stage {stage.name}: cached", file=sys.stderr, flush=True)
            else:
                output = _no_symlink(run_dir / "stages" / stage.name / uuid.uuid4().hex / "output")
                output.mkdir(parents=True)
                entry.update(status="running", cache_status="executed", output_dir=str(output))
                entry["log_path"] = str(output.parent / "stage.log")
                Path(entry["log_path"]).write_text(f"Started {stage.name} at {entry['started_at']}\n", encoding="utf-8")
                _atomic_json(record_path, record)
                print(f"Onboarding stage {stage.name}: executing", file=sys.stderr, flush=True)
                metadata = stage.execute(
                    output, {name: Path(record["stages"][name]["output_dir"]) for name in stage.dependencies}
                )
                _json(metadata)
                outputs = _tree(output, output=True)
                # A callback may not rewrite the dependencies that justify its result.
                for name in stage.dependencies:
                    dependency = record["stages"][name]
                    if _tree(Path(dependency["output_dir"]), output=True) != dependency["outputs"]:
                        raise PipelineError(f"Stage {stage.name} changed dependency output: {name}")
                for key, path in stage.inputs.items():
                    if _tree(Path(path)) != inputs[key]["entries"]:
                        raise PipelineError(f"Stage {stage.name} input changed during execution: {key}")
                entry.update(status="completed", metadata=metadata, metadata_sha256=_sha(metadata), outputs=outputs)
                print(f"Onboarding stage {stage.name}: completed", file=sys.stderr, flush=True)
                with Path(entry["log_path"]).open("a", encoding="utf-8") as stream:
                    stream.write(f"Completed {stage.name} at {_now()}\n")
        except (Exception, KeyboardInterrupt) as exc:
            entry.update(status="failed", cache_status="executed", failure=_failure(exc))
            print(f"Onboarding stage {stage.name}: failed: {exc}", file=sys.stderr, flush=True)
            if "log_path" in entry:
                with Path(entry["log_path"]).open("a", encoding="utf-8") as stream:
                    stream.write(f"Failed: {type(exc).__name__}: {exc}\n")
            if isinstance(exc, KeyboardInterrupt):
                entry.update(ended_at=_now(), duration_seconds=time.monotonic() - stage_started)
                record.update(status="interrupted", ended_at=_now(), duration_seconds=time.monotonic() - started)
                _atomic_json(record_path, record)
                raise
        if entry.get("cache_status") != "cached":
            entry.update(ended_at=_now(), duration_seconds=time.monotonic() - stage_started)
        _atomic_json(record_path, record)
    record.update(
        status="completed" if all(e["status"] == "completed" for e in record["stages"].values()) else "failed",
        ended_at=_now(),
        duration_seconds=time.monotonic() - started,
    )
    _atomic_json(record_path, record)
    return record


def _terminate(process: subprocess.Popen) -> None:
    """Stop the complete POSIX process group, including grandchildren."""
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=1)
    # The group leader may have exited while a child ignored SIGTERM.
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=5)


def run_command(
    argv: Sequence[str | os.PathLike],
    *,
    cwd: Path,
    log_path: Path,
    env: Mapping[str, str] | None = None,
    timeout: float = 1800,
) -> dict:
    """Run an argv without a shell, streaming combined output to a durable log."""
    if (
        not isinstance(argv, Sequence)
        or isinstance(argv, (str, bytes))
        or not argv
        or not all(isinstance(arg, (str, os.PathLike)) for arg in argv)
    ):
        raise PipelineError("Command must be a nonempty argv sequence, never shell text")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise PipelineError("Command timeout must be a positive finite number")
    argv = [os.fspath(arg) for arg in argv]
    if any(not isinstance(arg, str) or "\0" in arg for arg in argv):
        raise PipelineError("Command arguments must be text without NUL bytes")
    if env is not None and (
        not isinstance(env, Mapping)
        or any(
            not isinstance(key, str) or not isinstance(value, str) or "\0" in key or "\0" in value or "=" in key
            for key, value in env.items()
        )
    ):
        raise PipelineError("Command environment must map valid text names to text values")
    log_path = _no_symlink(Path(log_path))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    command_env = os.environ.copy()
    if env is not None:
        command_env.update(env)
    details = {"argv": argv, "cwd": str(Path(cwd).absolute()), "log_path": str(log_path), "timeout_seconds": timeout}
    with log_path.open("ab") as log:
        log.write(("Command: " + _json(argv) + "\n").encode())
        log.flush()
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=command_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                shell=False,
                start_new_session=True,
            )
        except OSError as exc:
            details.update(failure=str(exc), duration_seconds=time.monotonic() - started)
            raise PipelineError(f"Cannot start command {argv[0]}: {exc}", details=details) from exc
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _terminate(process)
            details.update(timed_out=True, returncode=process.returncode, duration_seconds=time.monotonic() - started)
            log.write(b"\nCommand timed out; process group terminated.\n")
            raise PipelineError(f"Command timed out after {timeout} seconds: {argv[0]}", details=details) from exc
        except BaseException:
            _terminate(process)
            raise
        details.update(returncode=returncode, duration_seconds=time.monotonic() - started)
        if returncode != 0:
            # Also stop surviving children of a failed command.
            _terminate(process)
            raise PipelineError(f"Command failed with exit code {returncode}: {argv[0]}", details=details)
        # A successful leader must not leave unbounded background descendants.
        _terminate(process)
    return details
