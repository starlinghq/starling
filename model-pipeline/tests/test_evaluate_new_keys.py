"""The float.min_cosine and float.token_embeddings_relative keys, and report-only hidden states."""

from __future__ import annotations

import copy
import json

import pytest
from synthetic_profile import DIM, HIDDEN, TOLERANCES, SyntheticRepo

from model_pipeline import evaluate, results


def tolerances(**float_keys) -> dict:
    tol = copy.deepcopy(TOLERANCES)
    tol["float"].update(float_keys)
    return tol


def run(repo: SyntheticRepo, **kw) -> evaluate.Evaluation:
    return evaluate.evaluate(repo.manifest, repo.results, repo_root=repo.root, protected_base="HEAD", **kw)


def check(ev: evaluate.Evaluation, check_id: str) -> dict:
    return next(c for c in ev.report["checks"] if c["id"] == check_id)


def write_token_embeddings(repo: SyntheticRepo, ref: dict, cand: dict) -> None:
    """Token-embedding tensors in the export results, from flat per-case lists (DIM wide rows)."""
    path = repo.results / f"export{results.RESULTS_SUFFIX}"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["tensors"]["token_embeddings"] = results.write_tensor(
        repo.results, "export-tokens.f32", [(c, (len(cand[c]) // DIM, DIM), cand[c]) for c in repo.cases]
    )
    data["tensors"]["reference_token_embeddings"] = results.write_tensor(
        repo.results, "export-reference-tokens.f32", [(c, (len(ref[c]) // DIM, DIM), ref[c]) for c in repo.cases]
    )
    path.write_text(json.dumps(data), encoding="utf-8")


def add_token_embeddings(repo: SyntheticRepo, scale: float = 1000.0, noise: float = 0.0) -> None:
    """Reference scale * hidden; the candidate adds `noise` to the first value of each case."""
    ref = {c: [scale * v for v in HIDDEN[c]] for c in repo.cases}
    cand = {c: [v + (noise if j == 0 else 0.0) for j, v in enumerate(ref[c])] for c in repo.cases}
    write_token_embeddings(repo, ref, cand)


def large_and_near_zero(repo: SyntheticRepo, noise: float, at: int) -> None:
    """Each case: value 0 is 4000, the rest near zero; the candidate adds `noise` at index `at`."""
    ref = {c: [4000.0 if j == 0 else 1e-3 for j in range(len(HIDDEN[c]))] for c in repo.cases}
    cand = {c: [v + (noise if j == at else 0.0) for j, v in enumerate(ref[c])] for c in repo.cases}
    write_token_embeddings(repo, ref, cand)


def all_stages(repo: SyntheticRepo, **kw) -> None:
    repo.tokenizer()
    repo.export(**kw)
    repo.dotnet()


def test_unimplemented_keys_are_now_evaluated(tmp_path):
    repo = SyntheticRepo(tmp_path / "r", tolerances(min_cosine=0.9999999, token_embeddings_relative=1e-4))
    all_stages(repo)
    add_token_embeddings(repo)
    ev = run(repo, require_gates=("G1", "G2"))
    assert ev.exit_code == evaluate.EXIT_OK, [c for c in ev.report["checks"] if c["status"] != "passed"]
    assert not [c for c in ev.report["checks"] if c["id"].startswith("tolerances.")]
    for cid in ("export.embedding.min_cosine", "dotnet.embedding.min_cosine", "dotnet.text_embedding.min_cosine"):
        c = check(ev, cid)
        assert c["status"] == "passed"
        assert c["bound"] == 0.9999999
        assert c["metrics"]["min_cosine"] >= 0.9999999
    assert check(ev, "dotnet.batch.fixture.vs_reference.min_cosine")["status"] == "passed"
    tokens = check(ev, "export.token_embeddings")
    assert tokens["status"] == "passed"
    assert tokens["metrics"]["max_relative"] == 0.0


def test_min_cosine_fails_on_a_rotated_embedding(tmp_path):
    # 2e-3 on one component of a unit vector: max abs bound 1e-2 holds, min cosine 0.9999999 does not.
    tol = tolerances(min_cosine=0.9999999, embedding_max_abs=1e-2, padded_batch_vs_single_embedding_max_abs=1e-2)
    repo = SyntheticRepo(tmp_path / "r", tol)
    all_stages(repo, emb_noise=2e-3)
    ev = run(repo)
    assert check(ev, "export.embedding")["status"] == "passed"
    c = check(ev, "export.embedding.min_cosine")
    assert c["status"] == "failed"
    assert "min cosine" in c["detail"]
    assert ev.report["gates"]["G2"]["status"] == "failed"


def test_token_embeddings_relative_bound_scales_with_magnitude(tmp_path):
    repo = SyntheticRepo(tmp_path / "r", tolerances(token_embeddings_relative=1e-4))
    all_stages(repo)
    # Every case's largest reference value is about 1000: 0.03 absolute is about 3e-5 per case.
    add_token_embeddings(repo, noise=0.03)
    assert check(run(repo), "export.token_embeddings")["status"] == "passed"
    add_token_embeddings(repo, noise=0.5)
    c = check(run(repo), "export.token_embeddings")
    assert c["status"] == "failed"
    assert c["metrics"]["max_relative"] == pytest.approx(5e-4, rel=2e-2)
    assert c["metrics"]["max_abs"] == pytest.approx(0.5, rel=1e-3)
    assert "per case" in c["detail"]


def test_token_embeddings_are_judged_per_case_not_per_value(tmp_path):
    # 0.2 on a near-zero value next to a 4000 in the same case: 0.2 per value, 5e-5 per case. Per case binds.
    repo = SyntheticRepo(tmp_path / "r", tolerances(token_embeddings_relative=1e-4))
    all_stages(repo)
    large_and_near_zero(repo, noise=0.2, at=1)
    ev = run(repo, require_gates=("G1", "G2"))
    c = check(ev, "export.token_embeddings")
    assert c["status"] == "passed", c
    assert c["metrics"]["max_relative"] == pytest.approx(0.2 / 4000, rel=1e-3)
    # Report only: the per-value figure is visible but does not gate.
    assert c["metrics"]["per_value_max_relative"] == pytest.approx(0.2, rel=1e-3)
    assert c["metrics"]["per_value_worst_case"] in repo.cases
    assert "report only" in c["detail"]
    assert ev.report["gates"]["G2"]["status"] == "passed"
    assert ev.exit_code == evaluate.EXIT_OK


def test_token_embeddings_fail_when_a_case_exceeds_the_bound(tmp_path):
    # 0.5 against a case maximum of 4000 is 1.25e-4 per case, above 1e-4.
    repo = SyntheticRepo(tmp_path / "r", tolerances(token_embeddings_relative=1e-4))
    all_stages(repo)
    large_and_near_zero(repo, noise=0.5, at=0)
    ev = run(repo)
    c = check(ev, "export.token_embeddings")
    assert c["status"] == "failed"
    assert c["metrics"]["max_relative"] == pytest.approx(1.25e-4, rel=1e-3)
    assert c["metrics"]["worst_case"] in repo.cases
    assert "> 0.0001" in c["detail"]
    assert ev.report["gates"]["G2"]["status"] == "failed"


@pytest.mark.parametrize(("noise", "status"), [(5e-5, "passed"), (2e-4, "failed")])
def test_token_embeddings_with_zero_references_use_max_one(tmp_path, noise, status):
    # All-zero references: the denominator is max(1, 0) = 1, so the bound is absolute.
    repo = SyntheticRepo(tmp_path / "r", tolerances(token_embeddings_relative=1e-4))
    all_stages(repo)
    ref = {c: [0.0] * len(HIDDEN[c]) for c in repo.cases}
    cand = {c: [noise] + [0.0] * (len(HIDDEN[c]) - 1) for c in repo.cases}
    write_token_embeddings(repo, ref, cand)
    c = check(run(repo), "export.token_embeddings")
    assert c["status"] == status
    assert c["metrics"]["max_relative"] == pytest.approx(noise, rel=1e-3)


def test_token_embeddings_with_an_empty_case_count_as_zero_error(tmp_path):
    repo = SyntheticRepo(tmp_path / "r", tolerances(token_embeddings_relative=1e-4))
    all_stages(repo)
    empty = next(iter(repo.cases))
    ref = {c: ([] if c == empty else [1000.0 * v for v in HIDDEN[c]]) for c in repo.cases}
    write_token_embeddings(repo, ref, {c: list(v) for c, v in ref.items()})
    c = check(run(repo), "export.token_embeddings")
    assert c["status"] == "passed"
    assert c["metrics"]["max_relative"] == 0.0


def test_token_embeddings_without_tensors_are_not_run(tmp_path):
    repo = SyntheticRepo(tmp_path / "r", tolerances(token_embeddings_relative=1e-4))
    all_stages(repo)
    ev = run(repo, require_gates=("G2",))
    assert check(ev, "export.token_embeddings")["status"] == "not-run"
    assert ev.report["gates"]["G2"]["status"] == "not-run"


def test_hidden_state_without_a_protected_key_is_report_only(tmp_path):
    tol = tolerances()
    del tol["float"]["last_hidden_state_max_abs"]
    repo = SyntheticRepo(tmp_path / "r", tol)
    all_stages(repo, hidden_noise=1.0)
    ev = run(repo, require_gates=("G1", "G2"))
    hidden = check(ev, "export.last_hidden_state")
    assert hidden["status"] == "not-applicable"
    assert hidden["metrics"]["max_abs"] == pytest.approx(1.0)
    assert ev.exit_code == evaluate.EXIT_OK
