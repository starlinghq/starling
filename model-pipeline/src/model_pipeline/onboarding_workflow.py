"""Run independent model recipes with a reviewable aggregate status."""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
import uuid
from pathlib import Path

from model_pipeline import fixtures
from model_pipeline.onboarding_plugins import build_stages, get_plugin
from model_pipeline.onboarding_recipe import RecipeError, _repository_file, load_recipe
from model_pipeline.onboarding_runner import PipelineError, run_pipeline

# Names the workflow itself writes beside the per-model run directories.
RESERVED_MODEL_IDS = frozenset({"history", "workflow.json"})


def _read(path: Path) -> dict:
    if path.stat().st_size > 1024 * 1024:
        raise PipelineError("workflow inputs exceed the 1 MiB limit")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PipelineError("workflow inputs must be objects")
    return data


def load_workflow(path: Path, bindings_path: Path, root: Path) -> tuple[dict, dict, list[tuple[dict, Path, dict]]]:
    definition, bindings = _read(path), _read(bindings_path)
    if (
        set(definition) != {"workflow_version", "id", "models"}
        or type(definition["workflow_version"]) is not int
        or definition["workflow_version"] != 1
        or not isinstance(definition["id"], str)
        or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,127}", definition["id"])
        or not isinstance(definition["models"], list)
        or not definition["models"]
    ):
        raise PipelineError("invalid workflow definition")
    records, names = [], set()
    for model in definition["models"]:
        if (
            not isinstance(model, dict)
            or set(model) != {"id", "recipe"}
            or not isinstance(model["id"], str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,127}", model["id"])
            or model["id"] in names
        ):
            raise PipelineError("workflow model IDs must be unique safe directory names")
        if model["id"] in RESERVED_MODEL_IDS:
            raise PipelineError(f"workflow model ID {model['id']} is reserved for workflow output")
        names.add(model["id"])
        recipe_path = _repository_file(model["recipe"], root.resolve(), "workflow recipe")
        recipe = load_recipe(recipe_path, root)
        record = bindings.get(model["id"])
        if (
            not isinstance(record, dict)
            or set(record) != {"snapshot", "reference_python"}
            or any(not isinstance(value, str) or not Path(value).is_absolute() for value in record.values())
        ):
            raise PipelineError(f"workflow binding {model['id']} needs absolute snapshot/reference_python paths")
        records.append((model, recipe_path, recipe))
    if names != set(bindings):
        raise PipelineError("workflow and binding model IDs differ")
    snapshots = [Path(bindings[name]["snapshot"]).resolve() for name in names]
    if len(snapshots) != len(set(snapshots)):
        raise PipelineError("different workflow models must have distinct source snapshots")
    return definition, bindings, records


def _save(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_workflow(
    path: Path,
    bindings_path: Path,
    *,
    root: Path,
    output: Path,
    threads: int = 2,
    timeout: float = 1800,
    resume: bool = False,
    distribution: bool = False,
) -> dict:
    from model_pipeline.onboarding_cli import _output_path

    definition, bindings, records = load_workflow(path, bindings_path, root)
    output = _output_path(
        output,
        root,
        inputs=(
            path,
            bindings_path,
            *(Path(record[key]) for record in bindings.values() for key in ("snapshot", "reference_python")),
            *(recipe_path for _model, recipe_path, _recipe in records),
            *(
                root / recipe[key]
                for _model, _path, recipe in records
                for key in ("manifest", "reference_corpus", "reference_environment")
            ),
        ),
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".workflow.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PipelineError("another writer is running this workflow") from exc
        summary = {
            "format": "starlings.onboarding-workflow",
            "format_version": 1,
            "workflow": definition,
            "workflow_sha256": fixtures.sha256_file(path),
            "bindings_sha256": fixtures.sha256_file(bindings_path),
            "status": "running",
            "models": {},
            "qualification": "per-model evidence only; no automatic reference adoption or release approval",
        }
        destination = output / "workflow.json"
        if destination.exists():
            history = output / "history"
            history.mkdir(exist_ok=True)
            _save(history / f"{uuid.uuid4().hex}.json", _read(destination))
        _save(destination, summary)
        try:
            for model, recipe_path, recipe in records:
                identifier = model["id"]
                binding = bindings[identifier]
                run_dir = output / identifier
                try:
                    stages = build_stages(
                        recipe,
                        plugin=get_plugin(recipe),
                        recipe_path=recipe_path,
                        repo_root=root,
                        snapshot=Path(binding["snapshot"]),
                        reference_python=Path(binding["reference_python"]),
                        threads=threads,
                        timeout=timeout,
                        distribution=distribution,
                    )
                    result = run_pipeline(
                        stages,
                        run_dir,
                        provenance={"recipe": recipe, "recipe_path": str(recipe_path), "repo_root": str(root)},
                        resume=resume,
                    )
                    summary["models"][identifier] = {
                        "status": result["status"],
                        "run_record": str(run_dir / "run.json"),
                        "stages": {
                            name: {key: stage[key] for key in ("status", "cache_status", "failure") if key in stage}
                            for name, stage in result["stages"].items()
                        },
                    }
                except (PipelineError, RecipeError, ValueError, OSError, KeyError) as exc:
                    summary["models"][identifier] = {"status": "failed", "failure": str(exc)}
                _save(destination, summary)
        except KeyboardInterrupt:
            summary["status"] = "interrupted"
            _save(destination, summary)
            raise
        summary["status"] = (
            "completed" if all(record["status"] == "completed" for record in summary["models"].values()) else "failed"
        )
        _save(destination, summary)
        return summary
