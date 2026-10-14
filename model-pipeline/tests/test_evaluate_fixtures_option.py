"""`model-pipeline evaluate --fixtures`: an explicit protected fixtures directory (P-030 release sets)."""

from __future__ import annotations

import json
import shutil

import pytest
from synthetic_profile import PROTECTED, SyntheticRepo, git

from model_pipeline import evaluate
from model_pipeline.cli import main

RELEASE = "model-pipeline/protected/synthetic-encoder/release/set-v1"


@pytest.fixture
def repo(tmp_path):
    return SyntheticRepo(tmp_path / "repo")


def run(repo: SyntheticRepo, **kw) -> evaluate.Evaluation:
    return evaluate.evaluate(repo.manifest, repo.results, repo_root=repo.root, protected_base="HEAD", **kw)


def all_stages(repo: SyntheticRepo) -> None:
    repo.tokenizer()
    repo.export()
    repo.dotnet()


def copy_set(repo: SyntheticRepo, rel: str, edit=None, commit: bool = True) -> str:
    target = repo.root / rel
    target.mkdir(parents=True)
    for name in ("tokenizer.jsonl", "embeddings.jsonl"):
        shutil.copyfile(repo.protected / name, target / name)
    if edit:
        edit(target)
    if commit:
        git(repo.root, "add", "-A")
        git(repo.root, "commit", "-q", "-m", "release set")
    return rel


def cli(repo: SyntheticRepo, out, *extra: str) -> int:
    args = ["evaluate", str(repo.manifest), "--results", str(repo.results), "--out", str(out)]
    return main([*args, "--protected-base", "HEAD", "--repo-root", str(repo.root), *extra])


def test_default_is_the_manifest_directory(repo):
    all_stages(repo)
    ev = run(repo)
    assert ev.exit_code == evaluate.EXIT_OK
    assert ev.report["inputs"]["fixtures_directory"] == {"path": PROTECTED, "source": "manifest"}
    assert all(f["path"].startswith(PROTECTED + "/") for f in ev.report["inputs"]["fixtures"])
    assert f"Fixtures: `{PROTECTED}` (from the manifest)." in evaluate.markdown(ev.report)


def test_reads_the_given_protected_directory(repo):
    copy_set(repo, RELEASE)
    all_stages(repo)
    ev = run(repo, require_gates=("G1", "G2"), fixtures_dir_override=RELEASE)
    assert ev.exit_code == evaluate.EXIT_OK, ev.report["checks"]
    assert ev.report["inputs"]["fixtures_directory"] == {"path": RELEASE, "source": "--fixtures"}
    assert [f["path"] for f in ev.report["inputs"]["fixtures"]] == [
        f"{RELEASE}/tokenizer.jsonl",
        f"{RELEASE}/embeddings.jsonl",
    ]
    assert ev.report["protected_evidence"]["status"] == "verified"
    assert ev.report["gates"]["G1"]["status"] == "passed"
    assert ev.report["gates"]["G2"]["status"] == "passed"
    # Tolerances still come from the manifest, never from the fixtures directory.
    assert ev.report["inputs"]["tolerances"]["path"] == f"{PROTECTED}/tolerances.yaml"
    assert f"Fixtures: `{RELEASE}` (from the --fixtures option)." in evaluate.markdown(ev.report)


def test_compares_with_the_given_set(repo):
    def shift(target):
        rows = [json.loads(line) for line in (target / "embeddings.jsonl").read_text(encoding="utf-8").splitlines()]
        rows[0]["embedding"][0] += 1e-3
        text = "".join(json.dumps(r) + "\n" for r in rows)
        (target / "embeddings.jsonl").write_text(text, encoding="utf-8", newline="\n")

    copy_set(repo, RELEASE, edit=shift)
    all_stages(repo)
    assert run(repo).exit_code == evaluate.EXIT_OK
    ev = run(repo, fixtures_dir_override=RELEASE)
    assert ev.exit_code == evaluate.EXIT_FAILED
    assert ev.report["gates"]["G2"]["status"] == "failed"


def test_uncommitted_set_is_not_gate_evidence(repo):
    copy_set(repo, RELEASE, commit=False)
    all_stages(repo)
    ev = run(repo, fixtures_dir_override=RELEASE)
    assert ev.exit_code == evaluate.EXIT_PROTECTED
    assert ev.report["gates"]["G1"]["status"] == "not-run"
    assert ev.report["gates"]["G2"]["status"] == "not-run"


@pytest.mark.parametrize("path", ["results", "model-pipeline/registry", "model-pipeline/protected/../registry"])
def test_outside_protected_is_refused(repo, tmp_path, path):
    all_stages(repo)
    assert cli(repo, tmp_path / "o", "--fixtures", path) == evaluate.EXIT_PROTECTED
    assert not (tmp_path / "o").exists()


def test_copy_outside_protected_is_refused(repo, tmp_path):
    copy_set(repo, "model-pipeline/release-copy")
    all_stages(repo)
    assert cli(repo, tmp_path / "o", "--fixtures", "model-pipeline/release-copy") == evaluate.EXIT_PROTECTED


def test_absolute_path_is_refused(repo, tmp_path):
    all_stages(repo)
    assert cli(repo, tmp_path / "o", "--fixtures", str(repo.protected)) == evaluate.EXIT_PROTECTED


def test_symlink_is_refused(repo, tmp_path):
    link = repo.root / "model-pipeline/protected/synthetic-encoder/link"
    link.symlink_to(repo.protected, target_is_directory=True)
    all_stages(repo)
    rel = "model-pipeline/protected/synthetic-encoder/link"
    assert cli(repo, tmp_path / "o", "--fixtures", rel) == evaluate.EXIT_PROTECTED


def test_missing_directory_is_refused(repo, tmp_path):
    all_stages(repo)
    assert cli(repo, tmp_path / "o", "--fixtures", RELEASE) == evaluate.EXIT_PROTECTED


def test_file_instead_of_directory_is_bad_input(repo, tmp_path):
    all_stages(repo)
    assert cli(repo, tmp_path / "o", "--fixtures", f"{PROTECTED}/tokenizer.jsonl") == evaluate.EXIT_INPUT


def test_with_g3_is_bad_input(repo, tmp_path):
    code = cli(repo, tmp_path / "o", "--g3", "retrieval", "--dataset", "x", "--fixtures", RELEASE)
    assert code == evaluate.EXIT_INPUT
