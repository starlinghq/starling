from __future__ import annotations

import fcntl
import json
from pathlib import Path

import pytest

from model_pipeline import onboarding_workflow as workflow
from model_pipeline.onboarding_runner import PipelineError, Stage


def setup(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("manifest", "corpus", "requirements"):
        (root / name).write_text("placeholder")
    definitions, bindings = [], {}
    for name in ("first", "second"):
        (root / f"{name}.yaml").write_text(
            f"recipe_version: 1\nid: {name}\nplugin: e5-onnx-cpu-v1\n"
            "manifest: manifest\nreference_corpus: corpus\nreference_environment: requirements\n"
        )
        snapshot = tmp_path / name
        snapshot.mkdir()
        (snapshot / "weight").write_text(name)
        definitions.append({"id": name, "recipe": f"{name}.yaml"})
        bindings[name] = {"snapshot": str(snapshot), "reference_python": str(tmp_path / "python")}
    definition = root / "workflow.json"
    definition.write_text(json.dumps({"workflow_version": 1, "id": "test", "models": definitions}))
    binding_file = root / "bindings.json"
    binding_file.write_text(json.dumps(bindings))
    return root, definition, binding_file, root / "artifacts/run"


def fake_plugin(counts: dict, *, fail: str | None = None):
    class Plugin:
        @staticmethod
        def build_stages(recipe: dict, **options):
            name = recipe["id"]

            def execute(out: Path, _deps):
                counts[name] = counts.get(name, 0) + 1
                if name == fail:
                    raise PipelineError("deliberate model failure")
                (out / "evidence.json").write_text(json.dumps({"model": name}))
                return {"status": "development-parity"}

            return [Stage("evaluate", (), {"snapshot": options["snapshot"]}, {"model": name}, execute)]

    return Plugin


def test_workflow_retains_failure_and_executes_independent_models(tmp_path, monkeypatch):
    root, definition, bindings, output = setup(tmp_path)
    counts = {}
    monkeypatch.setattr(workflow, "get_plugin", lambda _recipe: fake_plugin(counts, fail="first"))
    result = workflow.run_workflow(definition, bindings, root=root, output=output)
    assert result["status"] == "failed"
    assert result["models"]["first"]["status"] == "failed"
    assert result["models"]["second"]["status"] == "completed"
    assert counts == {"first": 1, "second": 1}
    assert json.loads((output / "workflow.json").read_text())["status"] == "failed"


def test_workflow_resume_repairs_only_corrupted_model(tmp_path, monkeypatch):
    root, definition, bindings, output = setup(tmp_path)
    counts = {}
    monkeypatch.setattr(workflow, "get_plugin", lambda _recipe: fake_plugin(counts))
    assert workflow.run_workflow(definition, bindings, root=root, output=output)["status"] == "completed"
    cached = workflow.run_workflow(definition, bindings, root=root, output=output, resume=True)
    assert all(model["stages"]["evaluate"]["cache_status"] == "cached" for model in cached["models"].values())
    assert counts == {"first": 1, "second": 1}
    run = json.loads((output / "first/run.json").read_text())
    evidence = Path(run["stages"]["evaluate"]["output_dir"]) / "evidence.json"
    evidence.write_text("corrupt")
    repaired = workflow.run_workflow(definition, bindings, root=root, output=output, resume=True)
    assert repaired["status"] == "completed"
    assert repaired["models"]["second"]["stages"]["evaluate"]["cache_status"] == "cached"
    assert counts == {"first": 2, "second": 1}


def test_workflow_requires_exact_bindings_and_safe_ids(tmp_path):
    root, definition, bindings, _output = setup(tmp_path)
    data = json.loads(bindings.read_text())
    data["extra"] = data["first"]
    bindings.write_text(json.dumps(data))
    with pytest.raises(PipelineError, match="IDs differ"):
        workflow.load_workflow(definition, bindings, root)
    del data["extra"]
    bindings.write_text(json.dumps(data))
    specification = json.loads(definition.read_text())
    specification["models"][0]["id"] = "../escape"
    definition.write_text(json.dumps(specification))
    with pytest.raises(PipelineError, match="safe directory"):
        workflow.load_workflow(definition, bindings, root)


@pytest.mark.parametrize("reserved", sorted(workflow.RESERVED_MODEL_IDS))
def test_workflow_rejects_ids_that_collide_with_workflow_output(tmp_path, reserved):
    root, definition, bindings, _output = setup(tmp_path)
    specification = json.loads(definition.read_text())
    specification["models"][0]["id"] = reserved
    definition.write_text(json.dumps(specification))
    data = json.loads(bindings.read_text())
    data[reserved] = data.pop("first")
    bindings.write_text(json.dumps(data))
    with pytest.raises(PipelineError, match="reserved"):
        workflow.load_workflow(definition, bindings, root)


def test_workflow_rejects_another_writer(tmp_path):
    root, definition, bindings, output = setup(tmp_path)
    output.mkdir(parents=True)
    with (output / ".workflow.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(PipelineError, match="another writer"):
            workflow.run_workflow(definition, bindings, root=root, output=output)


def test_workflow_refuses_to_overwrite_a_recipe_input(tmp_path):
    root, definition, bindings, output = setup(tmp_path)
    output.mkdir(parents=True)
    source = output / "workflow.json"
    source.write_text("immutable manifest")
    recipe = root / "first.yaml"
    recipe.write_text(recipe.read_text().replace("manifest: manifest", "manifest: artifacts/run/workflow.json"))
    with pytest.raises(ValueError, match="overlaps an input"):
        workflow.run_workflow(definition, bindings, root=root, output=output)
    assert source.read_text() == "immutable manifest"


def test_workflow_preserves_failed_summary_on_resume(tmp_path, monkeypatch):
    root, definition, bindings, output = setup(tmp_path)
    monkeypatch.setattr(workflow, "get_plugin", lambda _recipe: fake_plugin({}, fail="first"))
    assert workflow.run_workflow(definition, bindings, root=root, output=output)["status"] == "failed"
    monkeypatch.setattr(workflow, "get_plugin", lambda _recipe: fake_plugin({}))
    assert workflow.run_workflow(definition, bindings, root=root, output=output, resume=True)["status"] == "completed"
    histories = list((output / "history").glob("*.json"))
    assert len(histories) == 1
    assert json.loads(histories[0].read_text())["models"]["first"]["status"] == "failed"
