from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_pipeline import onboarding_e5, onboarding_runner
from model_pipeline.cli import main
from model_pipeline.onboarding_runner import Stage

REPO = Path(__file__).resolve().parents[2]
RECIPE = REPO / "model-pipeline/recipes/multilingual-e5-small-cpu.yaml"


def _write_stage(output_dir: Path, _dependencies: dict[str, Path]) -> dict[str, str]:
    (output_dir / "review.md").write_text("review evidence\n", encoding="utf-8")
    return {"evidence": "review.md"}


def _run_record(tmp_path: Path) -> tuple[Path, dict]:
    run_dir = tmp_path / "run"
    result = onboarding_runner.run_pipeline(
        [Stage("review", (), {}, {"kind": "test"}, _write_stage)],
        run_dir,
        provenance={"repo_root": str(REPO), "recipe": "test"},
    )
    assert result["status"] == "completed"
    return run_dir, result


def _write_record(run_dir: Path, record: dict) -> None:
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")


def test_inspect_cli_emits_synthetic_snapshot_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps({"model_type": "fixture", "hidden_size": 16}), encoding="utf-8")
    (snapshot / "tokenizer.json").write_text(
        json.dumps({"version": "1.0", "model": {"type": "BPE", "vocab": {}, "merges": []}}), encoding="utf-8"
    )

    assert main(["inspect", str(snapshot)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["model_config"]["model_type"] == "fixture"
    assert report["tokenizer"]["components"]
    assert report["plugin_route"]["status"] == "unknown"


def test_plan_cli_uses_shipped_recipe_and_makes_no_release_claim(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plan", str(RECIPE), "--repo-root", str(REPO)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["plugin"] == "e5-onnx-cpu-v1"
    assert plan["required_gates"] == ["G1", "G2"]
    assert "incomplete" in plan["release_status"]
    assert plan["reference_status"].startswith("proposed")


@pytest.mark.parametrize(
    ("recipe", "plugin"),
    [
        ("multilingual-e5-small-cpu.yaml", "e5-onnx-cpu-v1"),
        ("minilm-cpu.yaml", "bert-onnx-cpu-v1"),
        ("qwen3-cpu.yaml", "qwen-genai-cpu-v1"),
    ],
)
def test_plan_cli_lists_distribution_once_for_every_plugin(
    capsys: pytest.CaptureFixture[str], recipe: str, plugin: str
) -> None:
    assert main(["plan", str(REPO / "model-pipeline/recipes" / recipe), "--repo-root", str(REPO)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["plugin"] == plugin
    assert plan["optional_stages"].count("distribution") == 1
    assert "distribution" not in plan["stages"]


@pytest.mark.parametrize(("result_status", "expected_exit"), [("completed", 0), ("failed", 1)])
def test_run_cli_dispatches_stage_build_and_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    result_status: str,
    expected_exit: int,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    output = tmp_path / "run"
    calls: dict[str, object] = {}

    def build(recipe: dict, **kwargs: object) -> list[Stage]:
        calls["recipe"] = recipe
        calls["build_kwargs"] = kwargs
        return []

    def run(stages: list[Stage], run_dir: Path, **kwargs: object) -> dict:
        run_dir.mkdir(parents=True)
        calls["stages"] = stages
        calls["run_dir"] = run_dir
        calls["run_kwargs"] = kwargs
        return {
            "status": result_status,
            "run_id": "abc123",
            "stages": {"reference": {"status": "failed", "failure": {"message": "fixture failure"}}}
            if result_status == "failed"
            else {"reference": {"status": "completed"}},
        }

    monkeypatch.setattr(onboarding_e5, "build_stages", build)
    monkeypatch.setattr(onboarding_runner, "run_pipeline", run)
    args = [
        "run",
        str(RECIPE),
        "--repo-root",
        str(REPO),
        "--snapshot",
        str(snapshot),
        "--reference-python",
        str(tmp_path / "python"),
        "--out",
        str(output),
        "--threads",
        "3",
        "--timeout",
        "19",
        "--resume",
        "--distribution",
    ]
    assert main(args) == expected_exit
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == result_status
    kwargs = calls["build_kwargs"]
    assert kwargs["threads"] == 3
    assert kwargs["timeout"] == 19
    assert kwargs["distribution"] is True
    assert calls["run_kwargs"]["resume"] is True
    assert calls["run_dir"] == output.resolve()
    if result_status == "failed":
        task = output / "failure-task-abc123.json"
        assert task.is_file()
        assert json.loads(task.read_text(encoding="utf-8"))["failures"][0]["stage"] == "reference"


@pytest.mark.parametrize("output_kind", ["inside_snapshot", "protected"])
def test_run_rejects_output_overlapping_snapshot_or_protected_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_kind: str,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    called = False

    def build(*_args: object, **_kwargs: object) -> list[Stage]:
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(onboarding_e5, "build_stages", build)
    output = snapshot / "run" if output_kind == "inside_snapshot" else REPO / "model-pipeline/protected/new-run"
    result = main(
        [
            "run",
            str(RECIPE),
            "--repo-root",
            str(REPO),
            "--snapshot",
            str(snapshot),
            "--reference-python",
            str(tmp_path / "python"),
            "--out",
            str(output),
        ]
    )
    assert result == 2
    assert not called


def test_propose_exports_verified_run_artifacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir, result = _run_record(tmp_path)
    output = tmp_path / "proposal"
    assert main(["propose", str(run_dir), "--out", str(output)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "proposed"
    assert (output / "stages/review/review.md").read_text(encoding="utf-8") == "review evidence\n"
    assert json.loads((output / "run.json").read_text(encoding="utf-8"))["run_id"] == result["run_id"]
    assert (output / "README.md").is_file()


def test_propose_rejects_failed_run(tmp_path: Path) -> None:
    run_dir, record = _run_record(tmp_path)
    record["status"] = "failed"
    _write_record(run_dir, record)
    assert main(["propose", str(run_dir), "--out", str(tmp_path / "proposal")]) == 2


@pytest.mark.parametrize("tamper", ["digest", "traversal", "outside_output", "symlink"])
def test_propose_rejects_corrupt_or_unsafe_artifacts(tmp_path: Path, tamper: str) -> None:
    run_dir, record = _run_record(tmp_path)
    stage = record["stages"]["review"]
    output = Path(stage["output_dir"])
    if tamper == "digest":
        stage["outputs"]["review.md"]["sha256"] = "0" * 64
    elif tamper == "traversal":
        stage["outputs"]["../escaped"] = {"type": "file", "size": 1, "sha256": "0" * 64}
    elif tamper == "outside_output":
        stage["output_dir"] = str(tmp_path / "outside" / "output")
    else:
        artifact = output / "review.md"
        artifact.unlink()
        artifact.symlink_to(tmp_path / "outside.txt")
        (tmp_path / "outside.txt").write_text("outside", encoding="utf-8")
    _write_record(run_dir, record)

    assert main(["propose", str(run_dir), "--out", str(tmp_path / "proposal")]) == 2
