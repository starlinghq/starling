"""Embedding fixtures with prompt rows (EmbeddingGemma 2) and .NET stages without a text path."""

from __future__ import annotations

import json

import pytest
from synthetic_profile import PROMPTED, SyntheticRepo

from model_pipeline import evaluate, fixtures


@pytest.fixture
def repo(tmp_path):
    return SyntheticRepo(tmp_path / "repo", prompted=True)


def run(repo: SyntheticRepo, **kw) -> evaluate.Evaluation:
    return evaluate.evaluate(repo.manifest, repo.results, repo_root=repo.root, protected_base="HEAD", **kw)


def check(ev: evaluate.Evaluation, check_id: str) -> dict:
    return next(c for c in ev.report["checks"] if c["id"] == check_id)


def test_prompt_rows_are_cases_of_their_own(repo):
    fx = evaluate.load_fixtures(repo.protected)
    assert set(PROMPTED) <= set(fx.case_ids)
    assert fx.ids["ascii@query"] == PROMPTED["ascii@query"]
    assert fx.attention_mask["ascii@query"] == [1] * len(PROMPTED["ascii@query"])
    assert fx.ids["ascii"] != fx.ids["ascii@query"]


def test_all_stages_pass_with_prompt_rows(repo):
    repo.tokenizer()
    repo.export()
    repo.dotnet()
    ev = run(repo, require_gates=("G1", "G2"))
    assert ev.exit_code == evaluate.EXIT_OK, ev.report["checks"]
    assert check(ev, "dotnet.embedding")["metrics"]["cases"] == 5


def test_a_missing_prompt_case_fails(repo):
    repo.tokenizer()
    repo.export()
    repo.dotnet(drop="ascii@query")
    ev = run(repo)
    assert ev.exit_code == evaluate.EXIT_FAILED
    assert "ascii@query" in check(ev, "dotnet.embedding")["detail"]


def test_duplicate_rows_are_rejected(repo):
    path = repo.protected / "embeddings.jsonl"
    rows = fixtures.read_jsonl(path)
    fixtures.write_jsonl(path, [*rows, rows[-1]])
    with pytest.raises(evaluate.EvaluationError, match="appears twice"):
        evaluate.load_fixtures(repo.protected)


def test_dotnet_without_a_text_path_leaves_g2_not_run_and_needs_no_tokenizer_asset(repo):
    repo.tokenizer()
    repo.export()
    path = repo.dotnet()
    data = json.loads(path.read_text(encoding="utf-8"))
    for outputs in data["cases"].values():
        del outputs["text_embedding"]
    del data["batches"]["text"]
    data["assets"] = [a for a in data["assets"] if a["path"] != "tokenizer.json"]
    path.write_text(json.dumps(data), encoding="utf-8")

    ev = run(repo, require_gates=("G2",))
    assert ev.exit_code == evaluate.EXIT_GATE_NOT_PASSED
    assert check(ev, "dotnet.identity")["status"] == "passed"
    assert check(ev, "dotnet.text_embedding")["status"] == "not-run"
    assert check(ev, "dotnet.batch_evidence.text_embedding")["status"] == "not-run"
    assert check(ev, "dotnet.embedding")["status"] == "passed"
    assert ev.report["stages"]["dotnet-ids"] == "passed"
    assert ev.report["stages"]["dotnet-text"] == "incomplete"
    assert ev.report["gates"]["G2"]["status"] == "not-run"
    assert not [c for c in ev.report["checks"] if c["status"] in ("failed", "invalid")]


def test_a_text_path_still_requires_the_tokenizer_asset(repo):
    repo.tokenizer()
    repo.export()
    path = repo.dotnet()
    data = json.loads(path.read_text(encoding="utf-8"))
    data["assets"] = [a for a in data["assets"] if a["path"] != "tokenizer.json"]
    path.write_text(json.dumps(data), encoding="utf-8")
    ev = run(repo)
    assert check(ev, "dotnet.identity")["status"] == "invalid"
