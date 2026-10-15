"""`model-pipeline evaluate --reference-embeddings`: an embeddings.jsonl kept outside Git, bound by digest."""

from __future__ import annotations

import json
import shutil

import pytest
from synthetic_profile import SyntheticRepo, git

from model_pipeline import evaluate, fixtures
from model_pipeline.cli import main

RELEASE = "model-pipeline/protected/synthetic-encoder/release/set-v1"


@pytest.fixture
def repo(tmp_path):
    return SyntheticRepo(tmp_path / "repo")


def release_set(repo: SyntheticRepo, external, reference: dict | None, commit: bool = True):
    """A release set that keeps embeddings.jsonl outside Git (at `external`), with `reference` as reference.json."""
    target = repo.root / RELEASE
    target.mkdir(parents=True)
    shutil.copyfile(repo.protected / "tokenizer.jsonl", target / "tokenizer.jsonl")
    external.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repo.protected / "embeddings.jsonl", external)
    if reference is not None:
        (target / "reference.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    if commit:
        git(repo.root, "add", "-A")
        git(repo.root, "commit", "-q", "-m", "release set")
    repo.tokenizer()
    repo.export()
    repo.dotnet()
    return target


def binding(repo: SyntheticRepo, **over) -> dict:
    sha = fixtures.sha256_file(repo.protected / "embeddings.jsonl")
    return {"rows": 3, "embeddings": {"file": "embeddings.jsonl", "sha256": sha, "rows": 3, "committed": False} | over}


def run(repo: SyntheticRepo, external, **kw) -> evaluate.Evaluation:
    return evaluate.evaluate(
        repo.manifest,
        repo.results,
        repo_root=repo.root,
        protected_base="HEAD",
        fixtures_dir_override=RELEASE,
        reference_embeddings=external,
        **kw,
    )


def cli(repo: SyntheticRepo, out, external) -> int:
    args = ["evaluate", str(repo.manifest), "--results", str(repo.results), "--out", str(out)]
    extra = ["--protected-base", "HEAD", "--repo-root", str(repo.root), "--fixtures", RELEASE]
    return main([*args, *extra, "--reference-embeddings", str(external), "--require-gates", "G1,G2"])


def test_matching_digest_is_gate_evidence(repo, tmp_path):
    external = tmp_path / "outside" / "embeddings.jsonl"
    release_set(repo, external, binding(repo))
    ev = run(repo, external, require_gates=("G1", "G2"))
    assert ev.exit_code == evaluate.EXIT_OK, ev.report["checks"]
    assert ev.report["gates"]["G2"]["status"] == "passed"
    assert ev.report["protected_evidence"]["status"] == "verified"
    # The protected reference.json that pins the digest is among the Git-verified files, the external file is not.
    assert [f["path"] for f in ev.report["inputs"]["fixtures"]] == [
        f"{RELEASE}/tokenizer.jsonl",
        f"{RELEASE}/reference.json",
    ]
    ref = ev.report["inputs"]["reference_embeddings"]
    assert ref["path"] == str(external.resolve())
    assert ref["sha256"] == fixtures.sha256_file(external)
    assert ref["bound_by"] == f"{RELEASE}/reference.json embeddings.sha256"
    assert "Reference embeddings:" in evaluate.markdown(ev.report)
    assert cli(repo, tmp_path / "o", external) == evaluate.EXIT_OK


def test_the_external_file_is_the_reference(repo, tmp_path):
    # A digest-matching file is compared like a committed one: a perturbed candidate still fails.
    external = tmp_path / "outside" / "embeddings.jsonl"
    release_set(repo, external, binding(repo))
    repo.dotnet(noise=1e-3)
    ev = run(repo, external)
    assert ev.exit_code == evaluate.EXIT_FAILED
    assert ev.report["gates"]["G2"]["status"] == "failed"


def test_mismatch_is_refused(repo, tmp_path):
    external = tmp_path / "outside" / "embeddings.jsonl"
    release_set(repo, external, binding(repo))
    rows = [json.loads(line) for line in external.read_text(encoding="utf-8").splitlines()]
    rows[0]["embedding"][0] += 1e-7
    external.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    with pytest.raises(evaluate.protection.ProtectedEvidenceError, match="SHA-256"):
        run(repo, external)
    assert cli(repo, tmp_path / "o", external) == evaluate.EXIT_PROTECTED
    assert not (tmp_path / "o").exists()


@pytest.mark.parametrize(
    "reference",
    [
        None,
        {"rows": 3},
        {"rows": 3, "embeddings": {"file": "embeddings.jsonl", "rows": 3}},
        {"rows": 3, "embeddings": {"file": "embeddings.jsonl", "sha256": None}},
    ],
    ids=["no-reference-json", "no-embeddings-key", "no-sha256", "null-sha256"],
)
def test_missing_key_is_refused(repo, tmp_path, reference):
    external = tmp_path / "outside" / "embeddings.jsonl"
    release_set(repo, external, reference)
    assert cli(repo, tmp_path / "o", external) == evaluate.EXIT_PROTECTED
    assert not (tmp_path / "o").exists()


def test_edited_digest_is_not_gate_evidence(repo, tmp_path):
    # Changing the recorded digest in the work tree to match another file makes the protected evidence "changed".
    external = tmp_path / "outside" / "embeddings.jsonl"
    target = release_set(repo, external, binding(repo, sha256="0" * 64))
    rows = [json.loads(line) for line in external.read_text(encoding="utf-8").splitlines()]
    rows[0]["embedding"][0] += 1e-3
    external.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    edited = binding(repo, sha256=fixtures.sha256_file(external))
    (target / "reference.json").write_text(json.dumps(edited, indent=2) + "\n", encoding="utf-8")
    assert cli(repo, tmp_path / "o", external) == evaluate.EXIT_PROTECTED


@pytest.mark.parametrize("where", ["results/embeddings.jsonl", "model-pipeline/registry/embeddings.jsonl"])
def test_inside_repository_outside_protected_is_refused(repo, tmp_path, where):
    external = repo.root / where
    release_set(repo, external, binding(repo), commit=False)
    git(repo.root, "add", "-A")
    git(repo.root, "commit", "-q", "-m", "release set")
    with pytest.raises(evaluate.protection.ProtectedEvidenceError, match="inside the repository"):
        run(repo, external)
    assert cli(repo, tmp_path / "o", external) == evaluate.EXIT_PROTECTED


def test_symlink_into_repository_is_refused(repo, tmp_path):
    inside = repo.root / "results" / "embeddings.jsonl"
    release_set(repo, inside, binding(repo))
    link = tmp_path / "outside" / "embeddings.jsonl"
    link.parent.mkdir(parents=True)
    link.symlink_to(inside)
    assert cli(repo, tmp_path / "o", link) == evaluate.EXIT_PROTECTED


def test_set_with_committed_embeddings_is_refused(repo, tmp_path):
    external = tmp_path / "outside" / "embeddings.jsonl"
    target = release_set(repo, external, binding(repo), commit=False)
    shutil.copyfile(external, target / "embeddings.jsonl")
    git(repo.root, "add", "-A")
    git(repo.root, "commit", "-q", "-m", "release set")
    assert cli(repo, tmp_path / "o", external) == evaluate.EXIT_PROTECTED


def test_with_g3_or_tokenizer_fixtures_is_bad_input(repo, tmp_path):
    external = tmp_path / "embeddings.jsonl"
    base = ["evaluate", str(repo.manifest), "--results", str(repo.results), "--out", str(tmp_path / "o")]
    ref = ["--reference-embeddings", str(external)]
    assert main([*base, "--g3", "retrieval", "--dataset", "x", *ref]) == evaluate.EXIT_INPUT
    assert main([*base, "--tokenizer-fixtures", "x.jsonl", *ref]) == evaluate.EXIT_INPUT
