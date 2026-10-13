"""Tamper attempts against protected thresholds (P-007): every one is refused or makes the run non-evidence."""

from __future__ import annotations

import json
import os

import pytest
import yaml
from synthetic_profile import PROTECTED, TOLERANCES, SyntheticRepo, git

from model_pipeline import evaluate, protection
from model_pipeline.cli import main


@pytest.fixture
def repo(tmp_path):
    r = SyntheticRepo(tmp_path / "repo")
    r.tokenizer()
    r.export()
    r.dotnet()
    return r


def cli(repo: SyntheticRepo, *extra: str, base: str = "HEAD") -> int:
    out = repo.root.parent / "out"
    return main(
        ["evaluate", str(repo.manifest), "--results", str(repo.results), "--out", str(out), "--protected-base", base]
        + list(extra)
    )


@pytest.mark.parametrize("option", ["--tolerances", "--tolerance", "--threshold", "--bound"])
def test_cli_refuses_a_tolerance_override(repo, option, capsys):
    with pytest.raises(SystemExit) as exc:
        cli(repo, option, "loose.yaml")
    assert exc.value.code == evaluate.EXIT_PROTECTED
    assert "refused" in capsys.readouterr().err


def test_results_file_cannot_carry_thresholds(repo):
    path = repo.results / "dotnet.results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["environment"]["tolerances"] = {"embedding_max_abs": 1.0}
    path.write_text(json.dumps(data), encoding="utf-8")
    assert cli(repo) == evaluate.EXIT_PROTECTED


def test_uncommitted_tolerance_edit_is_detected(repo):
    tol = repo.protected / "tolerances.yaml"
    loose = {**TOLERANCES, "float": {**TOLERANCES["float"], "embedding_max_abs": 1.0}}
    tol.write_text(yaml.safe_dump(loose), encoding="utf-8")
    ev = evaluate.evaluate(repo.manifest, repo.results, repo_root=repo.root, protected_base="HEAD")
    assert ev.exit_code == evaluate.EXIT_PROTECTED
    assert ev.report["protected_evidence"]["status"] == "changed"
    assert ev.report["protected_evidence"]["changed"] == [f"{PROTECTED}/tolerances.yaml"]
    assert {g["status"] for g in ev.report["gates"].values()} == {"not-run"}


def test_committed_tolerance_edit_on_a_branch_is_detected(repo):
    base = git(repo.root, "rev-parse", "HEAD").strip()
    git(repo.root, "checkout", "-q", "-b", "implementation-job")
    tol = repo.protected / "tolerances.yaml"
    tol.write_text(tol.read_text(encoding="utf-8").replace("1.0e-05", "1.0e-03"), encoding="utf-8")
    git(repo.root, "commit", "-q", "-am", "loosen")
    assert cli(repo, base=base) == evaluate.EXIT_PROTECTED


def test_rewritten_golden_is_detected(repo):
    emb = repo.protected / "embeddings.jsonl"
    emb.write_text(emb.read_text(encoding="utf-8").replace('"embedding":[', '"embedding":[0.5,', 1), encoding="utf-8")
    # The rewritten golden is inconsistent anyway; with a consistent rewrite Git still catches it.
    assert cli(repo) in (evaluate.EXIT_PROTECTED, evaluate.EXIT_INPUT)
    v = protection.verify_unchanged(repo.root, [emb], "HEAD")
    assert v.status == "changed"


def test_untracked_file_in_protected_tree_is_detected(repo):
    (repo.protected / "extra.jsonl").write_text("{}\n", encoding="utf-8")
    v = protection.verify_unchanged(repo.root, [repo.protected], "HEAD")
    assert v.status == "changed"
    assert v.changed == [f"{PROTECTED}/extra.jsonl"]


def test_tolerances_outside_protected_tree_are_refused(repo):
    loose = repo.root / "model-pipeline" / "tolerances.yaml"
    loose.write_text(yaml.safe_dump(TOLERANCES), encoding="utf-8")
    repo.write_manifest(lambda m: m["qualification"].update(tolerances="model-pipeline/tolerances.yaml"))
    assert cli(repo) == evaluate.EXIT_PROTECTED


def test_dotdot_escape_is_refused(repo):
    repo.write_manifest(
        lambda m: m["qualification"].update(tolerances=f"{PROTECTED}/../../../registry/tolerances.yaml")
    )
    assert cli(repo) == evaluate.EXIT_PROTECTED


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_symlink_out_of_protected_tree_is_refused(repo):
    outside = repo.root / "loose.yaml"
    outside.write_text(yaml.safe_dump(TOLERANCES), encoding="utf-8")
    link = repo.protected / "tolerances.loose.yaml"
    link.symlink_to(outside)
    repo.write_manifest(lambda m: m["qualification"].update(tolerances=f"{PROTECTED}/tolerances.loose.yaml"))
    assert cli(repo) == evaluate.EXIT_PROTECTED


def test_tolerances_of_another_profile_are_refused(tmp_path):
    repo = SyntheticRepo(tmp_path / "repo", {**TOLERANCES, "profile": "other/profile"})
    repo.tokenizer()
    assert cli(repo) == evaluate.EXIT_PROTECTED


def test_non_tolerance_file_is_refused(repo):
    repo.write_manifest(lambda m: m["qualification"].update(tolerances=f"{PROTECTED}/embeddings.jsonl"))
    assert cli(repo) == evaluate.EXIT_PROTECTED


# -- pull request rule ----------------------------------------------------------------------------


def test_protected_change_needs_the_label():
    paths = ["model-pipeline/protected/x/tolerances.yaml"]
    assert not protection.check_changes(paths, []).ok
    assert protection.check_changes(paths, [protection.PROTECTED_LABEL]).ok


@pytest.mark.parametrize(
    "path",
    [
        "model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/embeddings.jsonl",
        "ai-dotnet/tests/tolerances.json",
        "tolerances.yaml",
    ],
)
def test_protected_paths(path):
    assert protection.is_protected(path)


def test_threshold_change_cannot_ride_along_with_implementation():
    paths = ["model-pipeline/protected/x/tolerances.yaml", "ai-dotnet/src/Starlings.Models.E5/E5Encoder.cs"]
    result = protection.check_changes(paths, [protection.PROTECTED_LABEL])
    assert not result.ok
    assert result.implementation == ["ai-dotnet/src/Starlings.Models.E5/E5Encoder.cs"]


def test_implementation_only_change_passes():
    assert protection.check_changes(["ai-dotnet/src/a.cs", "model-pipeline/src/model_pipeline/cli.py"], []).ok


def test_check_protected_cli(capsys):
    files = "model-pipeline/protected/a/tolerances.yaml\nREADME.md"
    assert main(["check-protected", "--files", files, "--labels-json", "[]"]) == 1
    labels = json.dumps([{"name": "protected-change"}])
    assert main(["check-protected", "--files", files, "--labels-json", labels]) == 0
