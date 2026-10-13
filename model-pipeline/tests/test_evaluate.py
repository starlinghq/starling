"""Staged fidelity evaluator: stage attribution on synthetic data (P-007)."""

from __future__ import annotations

import json

import pytest
from synthetic_profile import IDS, SyntheticRepo

from model_pipeline import evaluate
from model_pipeline.cli import main


@pytest.fixture
def repo(tmp_path):
    return SyntheticRepo(tmp_path / "repo")


def run(repo: SyntheticRepo, base: str = "HEAD", **kw) -> evaluate.Evaluation:
    return evaluate.evaluate(repo.manifest, repo.results, repo_root=repo.root, protected_base=base, **kw)


def check(ev: evaluate.Evaluation, check_id: str) -> dict:
    return next(c for c in ev.report["checks"] if c["id"] == check_id)


def taxonomies(ev: evaluate.Evaluation) -> dict[str, str]:
    return {f["stage"]: f["taxonomy"] for f in ev.report["findings"]}


def all_stages(repo: SyntheticRepo, **dotnet) -> None:
    repo.tokenizer()
    repo.export()
    repo.dotnet(**dotnet)


def test_all_stages_pass(repo):
    all_stages(repo)
    ev = run(repo, require_gates=("G1", "G2"))
    assert ev.exit_code == evaluate.EXIT_OK, ev.report["checks"]
    assert ev.report["first_divergence"] is None
    assert ev.report["findings"] == []
    assert ev.report["gates"]["G1"]["status"] == "passed"
    assert ev.report["gates"]["G2"]["status"] == "passed"
    assert ev.report["protected_evidence"]["status"] == "verified"
    assert set(ev.report["stages"].values()) == {"passed"}
    # Offsets are bound in the file but the embedding task has none: explicit, not silently skipped.
    assert check(ev, "tokenizer.offsets")["status"] == "not-applicable"
    emb = check(ev, "dotnet.embedding")
    assert emb["bound"] == 1e-5
    assert {"max_abs", "mean_abs", "p99_abs", "min_cosine"} <= set(emb["metrics"])


def test_export_hidden_state_failure_is_attributed_to_export(repo):
    repo.tokenizer()
    repo.export(hidden_noise=1e-3)
    repo.dotnet()
    ev = run(repo)
    assert ev.exit_code == evaluate.EXIT_FAILED
    assert ev.report["first_divergence"] == "export"
    assert check(ev, "export.last_hidden_state")["status"] == "failed"
    assert ev.report["stages"]["dotnet-ids"] == "passed"
    assert taxonomies(ev) == {"export": "export/operator or precision"}
    assert ev.report["gates"]["G1"]["status"] == "passed"
    assert ev.report["gates"]["G2"]["status"] == "failed"


def test_dotnet_following_a_failed_export_is_not_blamed_on_dotnet(repo):
    repo.tokenizer()
    repo.export(emb_noise=1e-4)
    repo.dotnet(export_noise=1e-4)  # C# reproduces Python ORT exactly
    ev = run(repo)
    assert ev.report["first_divergence"] == "export"
    assert check(ev, "dotnet.embedding.vs_python")["status"] == "passed"
    assert check(ev, "dotnet.embedding")["status"] == "failed"
    assert taxonomies(ev)["dotnet-ids"] == "consequence of export"
    assert taxonomies(ev)["dotnet-text"] == "consequence of export"


def test_dotnet_deviation_from_python_ort_is_a_dotnet_fault(repo):
    # Within the reference bound (1e-5) but beyond the same-backend bound (1e-6): still caught.
    all_stages(repo, noise=5e-6)
    ev = run(repo)
    assert ev.exit_code == evaluate.EXIT_FAILED
    assert ev.report["first_divergence"] == "dotnet-ids"
    assert ev.report["stages"]["export"] == "passed"
    assert check(ev, "dotnet.embedding")["status"] == "passed"
    assert check(ev, "dotnet.embedding.vs_python")["status"] == "failed"
    assert taxonomies(ev)["dotnet-ids"] == ".NET runtime binding, preprocessing or pooling"


def test_tokenizer_failure_comes_first_and_explains_the_text_path(repo):
    bad = list(IDS["long"])
    bad[2] = 4
    repo.tokenizer(ids={"long": bad})
    repo.export()
    repo.dotnet(text_noise=1e-3)
    ev = run(repo)
    assert ev.report["first_divergence"] == "tokenizer"
    tok = check(ev, "tokenizer.input_ids")
    assert tok["metrics"]["mismatched_cases"] == ["long"]
    assert tok["metrics"]["first_mismatch"]["position"] == 2
    assert check(ev, "tokenizer.attention_mask")["status"] == "passed"
    assert taxonomies(ev) == {"tokenizer": "tokenizer", "dotnet-text": "consequence of tokenizer"}
    assert ev.report["gates"]["G1"]["status"] == "failed"


def test_text_path_failure_with_exact_tokenizer_is_orchestration(repo):
    all_stages(repo, text_noise=1e-3)
    ev = run(repo)
    assert ev.report["first_divergence"] == "dotnet-text"
    assert ev.report["stages"]["dotnet-ids"] == "passed"
    assert taxonomies(ev) == {"dotnet-text": "processor"}


def test_batch_only_failure_is_batching(repo):
    all_stages(repo, batch_noise=1e-4)
    ev = run(repo)
    assert ev.report["first_divergence"] == "dotnet-ids"
    assert check(ev, "dotnet.batch.fixture.vs_single")["status"] == "failed"
    assert taxonomies(ev)["dotnet-ids"] == ".NET batching"


def test_export_batch_only_failure(repo):
    repo.tokenizer()
    repo.export(batch_noise=1e-4)
    repo.dotnet()
    ev = run(repo)
    assert ev.report["first_divergence"] == "export"
    assert "padded batches" in ev.report["findings"][0]["finding"]


def test_missing_case_fails_rather_than_shrinking_the_set(repo):
    all_stages(repo, drop="long")
    ev = run(repo)
    detail = check(ev, "dotnet.embedding")["detail"]
    assert "missing cases ['long']" in detail
    assert ev.exit_code == evaluate.EXIT_FAILED


def test_missing_tokenizer_case_fails(repo):
    repo.tokenizer(drop="empty")
    repo.export()
    repo.dotnet()
    ev = run(repo)
    assert check(ev, "tokenizer.input_ids")["status"] == "failed"


def test_nan_fails(repo):
    all_stages(repo, nan=True)
    ev = run(repo)
    assert check(ev, "dotnet.finite")["status"] == "failed"
    assert "long/embedding" in check(ev, "dotnet.finite")["detail"]


def test_without_export_stage_dotnet_failure_is_unattributed(repo):
    repo.tokenizer()
    repo.dotnet(noise=1e-4)
    ev = run(repo)
    assert check(ev, "dotnet.embedding.vs_python")["status"] == "not-run"
    assert ev.report["stages"]["export"] == "incomplete"
    assert taxonomies(ev)["dotnet-ids"] == "unattributed"


def test_missing_stages_leave_gates_not_run(repo):
    repo.tokenizer()
    ev = run(repo)
    assert ev.exit_code == evaluate.EXIT_OK
    assert ev.report["gates"]["G1"]["status"] == "passed"
    assert ev.report["gates"]["G2"]["status"] == "not-run"
    assert run(repo, require_gates=("G2",)).exit_code == evaluate.EXIT_GATE_NOT_PASSED


def test_different_machines_do_not_get_the_same_machine_bound(repo):
    repo.tokenizer()
    repo.export(machine="box-a")
    repo.dotnet(machine="box-b")
    ev = run(repo)
    assert check(ev, "dotnet.embedding.vs_python")["status"] == "not-run"
    assert ev.report["gates"]["G2"]["status"] == "not-run"


def test_wrong_model_digest_makes_the_stage_invalid(repo):
    repo.tokenizer()
    repo.export(model_sha="f" * 64)
    repo.dotnet()
    ev = run(repo)
    assert check(ev, "export.identity")["status"] == "invalid"
    assert taxonomies(ev)["export"] == "packaging/harness"


def test_candidate_fed_other_ids_is_invalid(repo):
    all_stages(repo, ids={"ascii": [0, 2]})
    ev = run(repo)
    assert check(ev, "dotnet.input_ids")["status"] == "invalid"
    assert ev.report["stages"]["dotnet-ids"] == "failed"


def test_provider_options_must_match_the_profile(tmp_path):
    from synthetic_profile import TOLERANCES

    repo = SyntheticRepo(tmp_path / "repo", {**TOLERANCES, "provider_options": {"use_tf32": 0}})
    repo.tokenizer()
    repo.export(options={"use_tf32": "0"})
    repo.dotnet(options={"use_tf32": "1"})
    ev = run(repo)
    assert check(ev, "export.provider_options")["status"] == "passed"
    assert check(ev, "dotnet.provider_options")["status"] == "failed"


def test_unknown_tolerance_key_blocks_the_gate(tmp_path):
    from synthetic_profile import TOLERANCES

    tol = json.loads(json.dumps(TOLERANCES))
    tol["float"]["logits_max_abs"] = 1e-3
    repo = SyntheticRepo(tmp_path / "repo", tol)
    all_stages(repo)
    ev = run(repo)
    assert check(ev, "tolerances.float.logits_max_abs")["status"] == "not-run"
    assert ev.report["gates"]["G2"]["status"] == "not-run"


def test_unverified_protected_evidence_cannot_pass_a_gate(repo):
    all_stages(repo)
    ev = run(repo, base="none")
    assert ev.exit_code == evaluate.EXIT_OK
    assert ev.report["protected_evidence"]["status"] == "unverified"
    assert ev.report["gates"]["G1"]["status"] == "not-run"


def test_cli_writes_reports(repo, tmp_path):
    all_stages(repo, noise=5e-6)
    out = tmp_path / "out"
    code = main(
        ["evaluate", str(repo.manifest), "--results", str(repo.results), "--out", str(out), "--protected-base", "HEAD"]
    )
    assert code == evaluate.EXIT_FAILED
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["first_divergence"] == "dotnet-ids"
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "First divergence: **dotnet-ids**" in md
    assert "csharp_vs_python_same_machine_and_provider.embedding_max_abs" in md


def test_tensor_sidecar_digest_is_checked(repo, tmp_path):
    all_stages(repo)
    sidecar = repo.results / "export-hidden.f32"
    data = bytearray(sidecar.read_bytes())
    data[0] ^= 1
    sidecar.write_bytes(bytes(data))
    code = main(["evaluate", str(repo.manifest), "--results", str(repo.results), "--out", str(tmp_path / "o")])
    assert code == evaluate.EXIT_INPUT


def test_error_stats_match_numpy_conventions():
    stats = evaluate.ErrorStats()
    stats.add("a", [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 0.0], cosine=False)
    s = stats.summary()
    assert s["max_abs"] == 4.0
    assert s["mean_abs"] == 1.0
    assert s["p99_abs"] == pytest.approx(0 + (4.0 - 0) * 0.97)  # numpy.quantile(..., 0.99), linear


# -- regressions: batch evidence per output path, and incomplete case sets (review of PR #12) -------------


def edit_results(repo: SyntheticRepo, stage: str, change) -> None:
    path = repo.results / f"{stage}.results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def assert_not_passed(ev: evaluate.Evaluation, gate: str = "G2") -> None:
    assert ev.exit_code != evaluate.EXIT_OK
    assert ev.report["gates"][gate]["status"] != "passed"


def test_missing_text_batches_block_g2(repo):
    all_stages(repo)
    edit_results(repo, "dotnet", lambda d: d["batches"].pop("text"))
    ev = run(repo, require_gates=("G1", "G2"))
    assert_not_passed(ev)
    assert check(ev, "dotnet.batch_evidence.text_embedding")["status"] == "failed"
    assert check(ev, "dotnet.batch_evidence.embedding")["status"] == "passed"
    assert ev.report["first_divergence"] == "dotnet-text"
    assert [(f["stage"], f["taxonomy"]) for f in ev.report["findings"]] == [("dotnet-text", "missing evidence")]


def test_single_row_dotnet_batches_block_g2(repo):
    all_stages(repo)

    def single_rows(d):
        for b in d["batches"].values():
            b["case_ids"], b["values"] = b["case_ids"][:1], b["values"][:1]

    edit_results(repo, "dotnet", single_rows)
    ev = run(repo)
    assert_not_passed(ev)
    assert check(ev, "dotnet.batch_evidence.embedding")["status"] == "failed"
    assert check(ev, "dotnet.batch_evidence.text_embedding")["status"] == "failed"
    assert ev.report["stages"]["export"] == "passed"


def test_batches_without_padding_block_g2(repo):
    all_stages(repo)

    def same_length(d):
        row = d["cases"]["ascii"]["embedding"]
        d["batches"]["fixture"].update(case_ids=["ascii", "ascii"], values=[row, row])

    edit_results(repo, "export", same_length)
    ev = run(repo)
    assert_not_passed(ev)
    evidence = check(ev, "export.batch_evidence.embedding")
    assert evidence["status"] == "failed"
    assert "never pad" in evidence["detail"]
    assert ev.report["first_divergence"] == "export"


def test_missing_export_case_is_attributed_to_export(repo, tmp_path):
    all_stages(repo)
    edit_results(repo, "export", lambda d: d["cases"].pop("ascii"))
    out = tmp_path / "out"
    code = main(["evaluate", str(repo.manifest), "--results", str(repo.results), "--out", str(out)])
    assert code == evaluate.EXIT_FAILED
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["first_divergence"] == "export"
    vs_py = next(c for c in report["checks"] if c["id"] == "dotnet.embedding.vs_python")
    assert vs_py["status"] == "not-run"
    assert "export results lack cases ['ascii']" in vs_py["detail"]
    assert {f["stage"] for f in report["findings"]} == {"export"}
    assert report["gates"]["G2"]["status"] == "failed"


def test_missing_dotnet_case_is_attributed_to_dotnet(repo):
    all_stages(repo, drop="ascii")
    ev = run(repo)
    assert ev.exit_code == evaluate.EXIT_FAILED
    assert ev.report["first_divergence"] == "dotnet-ids"
    assert ev.report["stages"]["export"] == "passed"
    stages = {f["stage"] for f in ev.report["findings"]}
    assert "dotnet-ids" in stages and stages <= {"dotnet-ids", "dotnet-text"}
    assert all(f["taxonomy"] == "missing evidence" for f in ev.report["findings"])
