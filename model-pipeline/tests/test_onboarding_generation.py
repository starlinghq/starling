from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from model_pipeline.evaluate_generation_candidate import POLICY
from model_pipeline.onboarding_generation import (
    CONFIG_CONTRACT,
    EXPECTED_BEHAVIOR,
    EXPECTED_RUNTIME,
    PROFILES,
    REQUIRED_EVIDENCE,
    describe_plan,
)
from model_pipeline.onboarding_runner import PipelineError


def profile(tmp_path, repo="Qwen/Qwen3-0.6B"):
    expected = PROFILES[repo]
    candidate = {
        "format_version": 1,
        "id": expected["id"],
        "task": "generation",
        "origin": {
            "repo": repo,
            "revision": expected["revision"],
            "files": [{"path": name, "size": 1, "sha256": "a" * 64} for name in ("config.json", "tokenizer.json")],
        },
        "runtime": EXPECTED_RUNTIME,
        "behavior": EXPECTED_BEHAVIOR[repo],
        "evaluation": {"status": "proposed", "policy": {**POLICY, "required_evidence": REQUIRED_EVIDENCE}},
    }
    recipe = {"recipe_version": 1, "id": "test", "plugin": "qwen-genai-cpu-v1", "manifest": "profile.json"}
    (tmp_path / "profile.json").write_text(json.dumps(candidate))
    return recipe, candidate


@pytest.mark.parametrize("repo", list(PROFILES))
def test_admits_only_pinned_text_generation_profiles(tmp_path, repo):
    recipe, candidate = profile(tmp_path, repo)
    report = describe_plan(recipe, repo_root=tmp_path)
    assert report["profile"] == candidate["id"]
    assert report["release_status"].startswith("unapproved")
    assert "bundle" in report["stages"]


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("runtime", "precision", "fp16"),
        ("behavior", "vocab_size", 248320),
        ("behavior", "enable_thinking_default", True),
        ("origin", "revision", "a" * 40),
    ],
)
def test_rejects_altered_contract(tmp_path, section, key, value):
    recipe, candidate = profile(tmp_path)
    candidate = copy.deepcopy(candidate)
    candidate[section][key] = value
    (tmp_path / "profile.json").write_text(json.dumps(candidate))
    with pytest.raises(PipelineError):
        describe_plan(recipe, repo_root=tmp_path)


def test_plan_rejects_duplicate_original_assets(tmp_path):
    recipe, candidate = profile(tmp_path)
    candidate["origin"]["files"][1] = candidate["origin"]["files"][0]
    (tmp_path / "profile.json").write_text(json.dumps(candidate))
    with pytest.raises(PipelineError, match="duplicate"):
        describe_plan(recipe, repo_root=tmp_path)


def builder(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "reference/qwen-chat"
    monkeypatch.syspath_prepend(str(directory))
    specification = importlib.util.spec_from_file_location("qwen_candidate_builder", directory / "build_bundle.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_flat_float32_overlay_changes_only_dtype_fields(monkeypatch):
    module = builder(monkeypatch)
    original = {"model_type": "qwen3", "torch_dtype": "bfloat16", "head_dim": 128, "hidden_size": 1024}
    before = copy.deepcopy(original)
    result = module.float32_config(original)
    assert original == before
    assert result == {**before, "dtype": "float32", "torch_dtype": "float32"}
    with pytest.raises(ValueError, match="flat"):
        module.float32_config({"model_type": "qwen3_5", "text_config": {}})


def test_verified_copy_rejects_corruption_and_never_changes_source(tmp_path, monkeypatch):
    module = builder(monkeypatch)
    snapshot = tmp_path / "source"
    snapshot.mkdir()
    config = {"model_type": "qwen3", "torch_dtype": "bfloat16", **CONFIG_CONTRACT["qwen3"]}
    contents = {"config.json": json.dumps(config).encode(), "tokenizer.json": b"{}", "tokenizer_config.json": b"{}"}
    for name, raw in contents.items():
        (snapshot / name).write_bytes(raw)
    candidate = {
        "origin": {
            "files": [
                {"path": name, "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                for name, raw in contents.items()
            ]
        }
    }
    module.verified_copy(candidate, snapshot, tmp_path / "scratch")
    assert (snapshot / "config.json").read_bytes() == contents["config.json"]
    assert json.loads((tmp_path / "scratch/config.json").read_text())["torch_dtype"] == "float32"
    (snapshot / "tokenizer.json").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="asset mismatch"):
        module.verified_copy(candidate, snapshot, tmp_path / "corrupt-scratch")


def test_dotnet_stage_publishes_into_transient_artifacts_and_keeps_only_evidence(tmp_path, monkeypatch):
    import sys

    from model_pipeline import onboarding_candidate, onboarding_generation
    from model_pipeline.onboarding_runner import Stage

    root, snapshot = tmp_path / "repo", tmp_path / "snapshot"
    root.mkdir()
    snapshot.mkdir()
    recipe, _candidate = profile(root)
    (root / "corpus.jsonl").write_text("")
    recipe["reference_corpus"] = "corpus.jsonl"
    declared = {}

    def code_inputs(_root, *, files=(), directories=()):
        declared["files"], declared["directories"] = files, directories
        return {}

    monkeypatch.setattr(
        onboarding_candidate,
        "candidate_stages",
        lambda *args, **kwargs: (
            None,
            [Stage(name, (), {}, {}, lambda out, deps: {}) for name in ("assets", "environment")],
            {},
        ),
    )
    monkeypatch.setattr(onboarding_candidate, "code_inputs", code_inputs)
    monkeypatch.setattr(onboarding_candidate, "source_reader", lambda stage, *args: stage)
    commands = []

    def command(argv, **kwargs):
        commands.append(argv)
        kwargs["log_path"].write_text("okay")
        if argv[1] == "publish":
            published = Path(argv[argv.index("--output") + 1])
            published.mkdir(parents=True)
            (published / "Starlings.Generation.Fidelity.dll").write_bytes(b"fixture")
            (published / "libonnxruntime-genai.so").write_bytes(b"native")
        return {"returncode": 0}

    monkeypatch.setattr(onboarding_generation, "run_command", command)
    graph = {
        stage.name: stage
        for stage in onboarding_generation.build_stages(
            recipe, repo_root=root, snapshot=snapshot, reference_python=Path(sys.executable), threads=2, timeout=11
        )
    }
    assert "ai-dotnet/tools/Starlings.Generation.Fidelity" in declared["directories"]
    out = tmp_path / "dotnet"
    result = graph["dotnet"].execute(out, {"bundle": tmp_path / "bundle"})
    publish, run = commands
    assert publish[1] == "publish" and "--disable-build-servers" in publish and "-m:1" in publish
    assert "-p:RestoreLockedMode=true" in publish
    artifacts = Path(publish[publish.index("--artifacts-path") + 1])
    assert artifacts.is_relative_to(out) and not artifacts.exists()
    assert Path(publish[publish.index("--output") + 1]).is_relative_to(artifacts)
    assert run[1].endswith("Starlings.Generation.Fidelity.dll")
    assert result["build"] == {"returncode": 0}
    assert sorted(path.name for path in out.iterdir()) == ["build.log", "command.log"]

    evaluate, assets, review_out = tmp_path / "evaluate", tmp_path / "assets", tmp_path / "review"
    evaluate.mkdir()
    assets.mkdir()
    review_out.mkdir()
    (evaluate / "report.json").write_text(json.dumps({"status": "passed", "cases": 1, "failures": []}))
    (assets / "source-lock.json").write_text(json.dumps({"profile": _candidate["id"]}))
    graph["review"].execute(review_out, {"evaluate": evaluate, "assets": assets})
    review = json.loads((review_out / "review.json").read_text())
    for key in (
        "candidate_commit",
        "source_lock",
        "support_status",
        "release_qualification",
        "protected_fixture_adoption",
        "distribution",
    ):
        assert key in review
    assert review["development_parity"]["status"] == "passed"
    assert review["distribution"] == "not-run" and review["release_qualification"] == "unapproved"
    assert "candidate_commit" in graph["review"].parameters
    assert (review_out / "review.md").is_file() and (review_out / "report.json").is_file()
