"""Staged fidelity evaluator, task entity-extraction, on synthetic data (P-015)."""

from __future__ import annotations

import json

import pytest
import yaml
from synthetic_extraction import (
    BY_ID,
    PROFILE,
    PROTECTED,
    TOLERANCES,
    SyntheticExtractionRepo,
    entity_result,
    git,
    logit,
)

from model_pipeline import evaluate, evaluate_extraction, results
from model_pipeline.cli import main
from model_pipeline.evaluate_extraction import f32, sigmoid32


@pytest.fixture
def repo(tmp_path):
    return SyntheticExtractionRepo(tmp_path / "repo")


def run(repo, base: str = "HEAD", **kw) -> evaluate.Evaluation:
    return evaluate.evaluate(repo.manifest, repo.results, repo_root=repo.root, protected_base=base, **kw)


def check(ev: evaluate.Evaluation, check_id: str) -> dict:
    return next(c for c in ev.report["checks"] if c["id"] == check_id)


def taxonomies(ev: evaluate.Evaluation) -> dict[str, str]:
    return {f["stage"]: f["taxonomy"] for f in ev.report["findings"]}


def all_stages(repo, **dotnet) -> None:
    repo.tokenizer()
    repo.export()
    repo.dotnet(**dotnet)


def kinds(ev: evaluate.Evaluation, check_id: str | None = None) -> list[tuple[str, str]]:
    return [(x["case"], x["kind"]) for x in ev.report["entity_exceptions"] if check_id in (None, x["check"])]


# -- the happy path and the report ----------------------------------------------------------------------


def test_all_stages_pass(repo):
    all_stages(repo)
    ev = run(repo, require_gates=("G1", "G2"))
    assert ev.exit_code == evaluate.EXIT_OK, [c for c in ev.report["checks"] if c["status"] != "passed"]
    assert ev.report["task"] == "entity-extraction"
    assert ev.report["format_version"] == 2
    assert ev.report["gates"]["G1"]["status"] == "passed"
    assert ev.report["gates"]["G2"]["status"] == "passed"
    assert ev.report["first_divergence"] is None
    assert ev.report["entity_exceptions"] == []
    assert set(ev.report["stages"].values()) == {"passed"}
    assert check(ev, "export.span_logits")["metrics"]["max_rel"] == 0.0
    # No same-machine key in the file: the C# vs Python comparison is reported, never binding.
    assert check(ev, "dotnet.span_logits.vs_python")["status"] == "not-applicable"
    md = evaluate.markdown(ev.report)
    assert "Allowed entity exceptions" in md and "Max rel" in md


def test_cli_writes_extraction_reports(repo, tmp_path):
    all_stages(repo)
    out = tmp_path / "out"
    code = main(["evaluate", str(repo.manifest), "--results", str(repo.results), "--out", str(out)])
    assert code == evaluate.EXIT_OK
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["task"] == "entity-extraction"
    assert "(entity-extraction)" in (out / "report.md").read_text(encoding="utf-8")


# -- no tolerances file: measured, never qualified -----------------------------------------------------


def test_missing_tolerances_file_reports_not_run_and_never_passes(tmp_path):
    repo = SyntheticExtractionRepo(tmp_path / "repo", tolerances=None)
    all_stages(repo)
    ev = run(repo)
    assert ev.exit_code == evaluate.EXIT_GATE_NOT_PASSED
    assert ev.report["gates"]["G1"]["status"] == "not-run"
    assert ev.report["gates"]["G2"]["status"] == "not-run"
    assert "no protected tolerances file" in ev.report["gates"]["G2"]["reason"]
    assert ev.report["manifest_gates"] == {}
    assert ev.report["inputs"]["tolerances"] == {
        "path": f"{PROTECTED}/tolerances.yaml",
        "sha256": None,
        "missing": True,
    }
    assert set(ev.report["expected_tolerance_keys"]) == evaluate_extraction.KNOWN_KEYS
    # Metrics are still measured, but nothing bound-dependent passes.
    span = check(ev, "export.span_logits")
    assert span["status"] == "not-run" and span["metrics"]["max_rel"] == 0.0
    assert check(ev, "tokenizer.words")["status"] == "not-run"
    assert check(ev, "export.entities")["status"] == "not-run"
    assert "No protected tolerances file" in evaluate.markdown(ev.report)
    assert run(repo, require_gates=("G1",)).exit_code == evaluate.EXIT_GATE_NOT_PASSED


def test_missing_tolerances_file_cannot_be_replaced_by_one_in_the_results(tmp_path):
    repo = SyntheticExtractionRepo(tmp_path / "repo", tolerances=None)
    all_stages(repo)
    (repo.results / "tolerances.yaml").write_text(yaml.safe_dump(TOLERANCES), encoding="utf-8")
    (repo.manifest.parent / "tolerances.yaml").write_text(yaml.safe_dump(TOLERANCES), encoding="utf-8")
    ev = run(repo)
    assert ev.report["gates"]["G2"]["status"] == "not-run"
    assert ev.exit_code == evaluate.EXIT_GATE_NOT_PASSED


def test_missing_tolerances_for_an_embedding_profile_also_never_pass(tmp_path):
    from synthetic_profile import SyntheticRepo

    repo = SyntheticRepo(tmp_path / "repo")
    repo.tokenizer()
    repo.export()
    repo.dotnet()
    (repo.protected / "tolerances.yaml").unlink()
    ev = evaluate.evaluate(repo.manifest, repo.results, repo_root=repo.root, protected_base="none")
    assert ev.report["gates"]["G2"]["status"] == "not-run"
    assert ev.exit_code == evaluate.EXIT_GATE_NOT_PASSED


def test_deleting_the_tolerances_file_is_a_protected_change(repo):
    all_stages(repo)
    (repo.protected / "tolerances.yaml").unlink()
    ev = run(repo)
    assert ev.exit_code == evaluate.EXIT_PROTECTED
    assert ev.report["protected_evidence"]["changed"] == [f"{PROTECTED}/tolerances.yaml"]


# -- tamper attempts --------------------------------------------------------------------------------------


@pytest.mark.parametrize("option", ["--tolerances", "--threshold", "--bound"])
def test_cli_refuses_tolerance_overrides(repo, option, capsys):
    all_stages(repo)
    with pytest.raises(SystemExit) as exc:
        main(["evaluate", str(repo.manifest), "--results", str(repo.results), "--out", "x", option, "1"])
    assert exc.value.code == evaluate.EXIT_PROTECTED
    assert "refused" in capsys.readouterr().err


def test_results_carrying_a_threshold_are_refused(repo, tmp_path):
    all_stages(repo)
    path = repo.results / "dotnet.results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["cases"]["basic"]["text_entities"]["threshold"] = 0.4
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(results.ToleranceOverrideError):
        run(repo)
    code = main(["evaluate", str(repo.manifest), "--results", str(repo.results), "--out", str(tmp_path / "o")])
    assert code == evaluate.EXIT_PROTECTED


def test_loosened_entity_bound_on_a_branch_is_detected(repo):
    all_stages(repo, text_entities={"basic": entity_result("basic", lambda items: [items[1], items[0], items[2]])})
    tol = json.loads(json.dumps(TOLERANCES))
    tol["entities"]["order_swap_max_confidence_delta"] = 1.0
    (repo.protected / "tolerances.yaml").write_text(yaml.safe_dump(tol), encoding="utf-8")
    git(repo.root, "commit", "-qam", "loosen")
    ev = run(repo, base="HEAD~1")
    assert ev.exit_code == evaluate.EXIT_PROTECTED
    assert ev.report["gates"]["G2"]["status"] == "not-run"


def test_tolerances_of_another_profile_are_refused(tmp_path):
    from model_pipeline import protection

    tol = dict(TOLERANCES, profile="other/profile")
    repo = SyntheticExtractionRepo(tmp_path / "repo", tolerances=tol)
    all_stages(repo)
    with pytest.raises(protection.ProtectedEvidenceError):
        run(repo)


def test_unknown_tolerance_key_blocks_the_gate(tmp_path):
    tol = json.loads(json.dumps(TOLERANCES))
    tol["entities"]["relation_identity"] = "exact"
    repo = SyntheticExtractionRepo(tmp_path / "repo", tolerances=tol)
    all_stages(repo)
    ev = run(repo)
    assert check(ev, "tolerances.entities.relation_identity")["status"] == "not-run"
    assert ev.report["gates"]["G2"]["status"] == "not-run"


def test_a_non_exact_discrete_value_is_not_run(tmp_path):
    tol = json.loads(json.dumps(TOLERANCES))
    tol["discrete"]["words"] = "case-insensitive"
    repo = SyntheticExtractionRepo(tmp_path / "repo", tolerances=tol)
    all_stages(repo)
    ev = run(repo)
    assert check(ev, "tokenizer.words")["status"] == "not-run"
    assert ev.report["gates"]["G1"]["status"] == "not-run"


# -- the entity rule: order ------------------------------------------------------------------------------


def swap_basic_persons(items):
    return [items[1], items[0], items[2]]


def test_near_tie_swap_is_allowed_and_listed(repo):
    all_stages(repo, text_entities={"basic": entity_result("basic", swap_basic_persons)})
    ev = run(repo, require_gates=("G1", "G2"))
    assert ev.exit_code == evaluate.EXIT_OK
    assert kinds(ev, "dotnet.text_entities") == [("basic", "order-swap")]
    # The text batch carries the same rows, so its comparison with the reference lists the swap too.
    assert kinds(ev, "dotnet.batch.text.vs_reference") == [("basic", "order-swap")]
    x = ev.report["entity_exceptions"][0]
    assert (x["first"]["text"], x["second"]["text"]) == ("Bob", "Ann")
    assert x["delta"] <= 2e-5
    assert "1 allowed exception" in check(ev, "dotnet.text_entities")["detail"]
    assert "order-swap" in evaluate.markdown(ev.report)


def test_swap_beyond_the_bound_fails(tmp_path):
    tol = json.loads(json.dumps(TOLERANCES))
    tol["entities"]["order_swap_max_confidence_delta"] = 1e-6  # Ann and Bob differ by 1.5e-5
    repo = SyntheticExtractionRepo(tmp_path / "repo", tolerances=tol)
    all_stages(repo, text_entities={"basic": entity_result("basic", swap_basic_persons)})
    ev = run(repo)
    text = check(ev, "dotnet.text_entities")
    assert text["status"] == "failed"
    assert "order differs" in text["detail"]
    assert ev.report["entity_exceptions"] == []


def test_exact_order_policy_allows_no_swap_but_still_compares_equal_lists(tmp_path):
    tol = json.loads(json.dumps(TOLERANCES))
    tol["entities"]["order_swap_max_confidence_delta"] = 0.0
    repo = SyntheticExtractionRepo(tmp_path / "repo", tolerances=tol)
    all_stages(repo)
    assert run(repo).exit_code == evaluate.EXIT_OK
    repo.dotnet(text_entities={"basic": entity_result("basic", swap_basic_persons)})
    assert check(run(repo), "dotnet.text_entities")["status"] == "failed"


def test_items_out_of_label_order_fail(repo):
    all_stages(repo, text_entities={"basic": entity_result("basic", lambda items: [items[2], items[0], items[1]])})
    ev = run(repo)
    assert "schema label order" in check(ev, "dotnet.text_entities")["detail"]


def test_duplicate_items_fail(repo):
    all_stages(repo, text_entities={"basic": entity_result("basic", lambda items: [items[0], *items])})
    assert "duplicate" in check(run(repo), "dotnet.text_entities")["detail"]


# -- the entity rule: threshold flips --------------------------------------------------------------------


def add_in_location(items):
    items.append({"label": "location", "text": "in", "start": 12, "end": 14, "confidence": 0.50000})


def test_near_threshold_addition_is_allowed(repo):
    all_stages(repo, text_entities={"basic": entity_result("basic", add_in_location)})
    ev = run(repo)
    assert ev.exit_code == evaluate.EXIT_OK
    [x] = [x for x in ev.report["entity_exceptions"] if x["check"] == "dotnet.text_entities"]
    assert (x["kind"], x["direction"], x["text"]) == ("threshold-flip", "added", "in")
    assert x["margin"] <= 2e-5


def test_near_threshold_drop_is_allowed(repo):
    all_stages(repo, text_entities={"near": entity_result("near", lambda items: items[:1])})
    ev = run(repo)
    assert ev.exit_code == evaluate.EXIT_OK
    [x] = [x for x in ev.report["entity_exceptions"] if x["check"] == "dotnet.text_entities"]
    assert (x["kind"], x["direction"], x["text"]) == ("threshold-flip", "dropped", "Eve")


def test_far_from_threshold_addition_fails(repo):
    def add_met(items):
        items.insert(2, {"label": "person", "text": "met", "start": 4, "end": 7, "confidence": 0.6})

    all_stages(repo, text_entities={"basic": entity_result("basic", add_met)})
    detail = check(run(repo), "dotnet.text_entities")["detail"]
    assert "'met'" in detail and "from the threshold" in detail


def test_far_from_threshold_drop_fails(repo):
    all_stages(repo, text_entities={"near": entity_result("near", lambda items: items[1:])})
    ev = run(repo)
    assert check(ev, "dotnet.text_entities")["status"] == "failed"
    assert "Kim" in check(ev, "dotnet.text_entities")["detail"]


def test_flip_margin_exactly_at_the_bound_is_allowed(tmp_path):
    margin = abs(sigmoid32(logit(0.49999)) - f32(0.5))
    tol = json.loads(json.dumps(TOLERANCES))
    tol["entities"]["threshold_flip_max_score_margin"] = margin
    repo = SyntheticExtractionRepo(tmp_path / "repo", tolerances=tol)
    all_stages(repo, text_entities={"basic": entity_result("basic", add_in_location)})
    assert run(repo).exit_code == evaluate.EXIT_OK
    tol["entities"]["threshold_flip_max_score_margin"] = margin * 0.999
    repo2 = SyntheticExtractionRepo(tmp_path / "repo2", tolerances=tol)
    all_stages(repo2, text_entities={"basic": entity_result("basic", add_in_location)})
    assert check(run(repo2), "dotnet.text_entities")["status"] == "failed"


def test_added_item_misplaced_against_reference_scores_is_an_order_violation(repo):
    # "in" (score 0.49999) placed before Rome (0.97): a flip is allowed, the order is not.
    def add_first(items):
        items.insert(2, {"label": "location", "text": "in", "start": 12, "end": 14, "confidence": 0.5})

    all_stages(repo, text_entities={"basic": entity_result("basic", add_first)})
    detail = check(run(repo), "dotnet.text_entities")["detail"]
    assert "order differs" in detail


def test_span_wider_than_max_width_fails(repo):
    def wide(items):
        items.append({"label": "location", "text": "Ann met Bob in", "start": 0, "end": 14, "confidence": 0.5})

    all_stages(repo, text_entities={"basic": entity_result("basic", wide)})
    assert "no word span" in check(run(repo), "dotnet.text_entities")["detail"]


# -- gate, offsets, empty results ------------------------------------------------------------------------


def test_gate_decision_mismatch_fails_and_is_attributed_to_export(repo):
    closed = [5.0, 2.0] + [-3.0] * 18
    repo.tokenizer()
    repo.export(
        count={"basic": closed},
        entities={"basic": {"gate_open": False, "entities": []}},
    )
    repo.dotnet()
    ev = run(repo)
    assert check(ev, "export.gate_decision")["metrics"]["mismatched_cases"] == ["basic"]
    assert "gate decision differs" in check(ev, "export.entities")["detail"]
    assert ev.report["first_divergence"] == "export"
    assert taxonomies(ev)["export"] == "export/operator or precision"


def test_entity_gate_inconsistent_with_own_count_logits_fails(repo):
    all_stages(repo, entities={"gate-closed": {"gate_open": True, "entities": []}})
    ev = run(repo)
    gate = check(ev, "dotnet.gate_decision")
    assert gate["metrics"]["entities_gate_inconsistent_with_count_logits"] == ["gate-closed"]
    assert gate["status"] == "failed"


def test_entity_offsets_beyond_text_length_fail(repo):
    def beyond(items):
        items[2] = dict(items[2], end=99)

    all_stages(repo, text_entities={"basic": entity_result("basic", beyond)})
    ev = run(repo)
    assert "outside the normalized text (length 20)" in check(ev, "dotnet.text_entities")["detail"]
    assert taxonomies(ev)["dotnet-text"] == "processor"


def test_entity_surface_must_match_the_offsets(repo):
    all_stages(
        repo, text_entities={"basic": entity_result("basic", lambda items: [dict(items[0], text="Anne"), *items[1:]])}
    )
    assert "not normalized_text[start:end].strip()" in check(run(repo), "dotnet.text_entities")["detail"]


def test_processor_character_maps_beyond_text_length_fail(repo):
    repo.tokenizer(edits={"empty": {"end_map": [2]}})
    repo.export()
    repo.dotnet()
    ev = run(repo)
    assert check(ev, "tokenizer.character_maps.well_formed")["status"] == "failed"
    assert check(ev, "tokenizer.end_map")["metrics"]["mismatched_cases"] == ["empty"]
    assert taxonomies(ev)["tokenizer"] == "processor"
    assert ev.report["gates"]["G1"]["status"] == "failed"


def test_empty_results_compare_exactly(repo):
    # A result in the empty case or the closed-gate case is a violation, not an exception.
    all_stages(
        repo,
        text_entities={
            "empty": {
                "gate_open": True,
                "entities": [{"label": "person", "text": ".", "start": 0, "end": 1, "confidence": 0.9}],
            }
        },
    )
    assert "reference score" in check(run(repo), "dotnet.text_entities")["detail"]
    repo.dotnet(
        text_entities={
            "gate-closed": {
                "gate_open": False,
                "entities": [{"label": "person", "text": "Nothing", "start": 0, "end": 7, "confidence": 0.9}],
            }
        }
    )
    assert "gate closed but 1 entities" in check(run(repo), "dotnet.text_entities")["detail"]


def test_empty_and_closed_gate_cases_pass_when_empty(repo):
    all_stages(repo)
    ev = run(repo)
    assert check(ev, "dotnet.text_entities")["metrics"]["exact_cases"] == 4


# -- numeric bounds --------------------------------------------------------------------------------------


def test_span_logits_bound_is_relative(repo):
    # 5e-5 relative: on |logit| = 10 that is 5e-4 absolute, still within 1e-4 * max(1, |ref|).
    repo.tokenizer()
    repo.export(rel_noise=5e-5)
    repo.dotnet(rel_noise=5e-5)
    ev = run(repo)
    span = check(ev, "export.span_logits")
    assert span["status"] == "passed"
    assert span["metrics"]["max_abs"] > 1e-4 and span["metrics"]["max_rel"] <= 1e-4


def test_small_logit_absolute_error_beyond_the_relative_bound_fails(repo):
    repo.tokenizer()
    repo.export(logit_noise=3e-4)
    repo.dotnet(logit_noise=3e-4)
    ev = run(repo)
    assert check(ev, "export.span_logits")["status"] == "failed"
    assert ev.report["first_divergence"] == "export"
    assert taxonomies(ev)["dotnet-ids"] == "consequence of export"


def test_hidden_state_failure_is_export(repo):
    repo.tokenizer()
    repo.export(hidden_noise=1e-3)
    repo.dotnet()
    ev = run(repo)
    assert check(ev, "export.last_hidden_state")["status"] == "failed"
    assert taxonomies(ev) == {"export": "export/operator or precision"}


def test_dotnet_vs_python_bound_when_set(tmp_path):
    tol = json.loads(json.dumps(TOLERANCES))
    tol["csharp_vs_python_same_machine_and_provider"] = {"span_logits_relative": 1e-6}
    repo = SyntheticExtractionRepo(tmp_path / "repo", tolerances=tol)
    repo.tokenizer()
    repo.export()
    repo.dotnet(rel_noise=5e-6)  # within the reference bound, beyond the same-machine bound
    ev = run(repo)
    assert check(ev, "dotnet.span_logits")["status"] == "passed"
    assert check(ev, "dotnet.span_logits.vs_python")["status"] == "failed"
    assert taxonomies(ev)["dotnet-ids"] == ".NET runtime binding or preprocessing"


def test_dotnet_decoder_fault_is_dotnet_decoding(repo):
    all_stages(repo, entities={"near": entity_result("near", lambda items: items[1:])})
    ev = run(repo)
    assert ev.report["first_divergence"] == "dotnet-ids"
    assert taxonomies(ev)["dotnet-ids"] == ".NET decoding"


def test_export_entities_beyond_exceptions_with_good_logits_is_a_boundary_finding(repo):
    repo.tokenizer()
    repo.export(entities={"near": entity_result("near", lambda items: items[1:])})
    repo.dotnet()
    ev = run(repo)
    assert taxonomies(ev)["export"] == "threshold or ordering boundary"


def test_text_path_failure_with_exact_processor_is_orchestration(repo):
    all_stages(repo, text_entities={"near": entity_result("near", lambda items: items[1:])})
    ev = run(repo)
    assert ev.report["first_divergence"] == "dotnet-text"
    assert taxonomies(ev) == {"dotnet-text": "processor"}


def test_processor_word_failure_explains_the_text_path(repo):
    repo.tokenizer(edits={"basic": {"words": ["ann", "met", "bob", "in", "rome"]}})
    repo.export()
    repo.dotnet(text_entities={"near": entity_result("near", lambda items: items[1:])})
    ev = run(repo)
    assert ev.report["first_divergence"] == "tokenizer"
    assert taxonomies(ev) == {"tokenizer": "processor", "dotnet-text": "consequence of processor"}


def test_token_id_only_failure_is_tokenizer(repo):
    ids = list(BY_ID["basic"]["input_ids"])
    ids[-1] += 1
    repo.tokenizer(edits={"basic": {"input_ids": ids, "attention_mask": [1] * len(ids)}})
    repo.export()
    repo.dotnet()
    assert taxonomies(run(repo)) == {"tokenizer": "tokenizer"}


def test_nan_fails(repo):
    repo.tokenizer()
    repo.export()
    repo.dotnet()
    path = repo.results / "dotnet.results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["cases"]["basic"]["span_logits"][0] = float("nan")
    path.write_text(json.dumps(data), encoding="utf-8")
    ev = run(repo)
    assert "basic/span_logits" in check(ev, "dotnet.finite")["detail"]


# -- batches and missing evidence ---------------------------------------------------------------------------


def edit_results(repo, stage: str, change) -> None:
    path = repo.results / f"{stage}.results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_missing_text_batch_blocks_g2(repo):
    all_stages(repo)
    edit_results(repo, "dotnet", lambda d: d["batches"].pop("text"))
    ev = run(repo, require_gates=("G2",))
    assert ev.exit_code == evaluate.EXIT_FAILED
    assert check(ev, "dotnet.batch_evidence.text_entities")["status"] == "failed"
    assert [(f["stage"], f["taxonomy"]) for f in ev.report["findings"]] == [("dotnet-text", "missing evidence")]


def test_batches_without_padding_block_g2(repo):
    all_stages(repo)

    def same_length(d):
        row = d["cases"]["basic"]["span_logits"]
        d["batches"]["padded"].update(case_ids=["basic", "basic"], values=[row, row])

    edit_results(repo, "export", same_length)
    ev = run(repo)
    assert "never pad" in check(ev, "export.batch_evidence.span_logits")["detail"]
    assert ev.report["gates"]["G2"]["status"] == "failed"


def test_single_row_batch_blocks_g2(repo):
    all_stages(repo)

    def single(d):
        for b in d["batches"].values():
            b["case_ids"], b["values"] = b["case_ids"][:1], b["values"][:1]

    edit_results(repo, "dotnet", single)
    ev = run(repo)
    assert check(ev, "dotnet.batch_evidence.span_logits")["status"] == "failed"
    assert check(ev, "dotnet.batch_evidence.text_entities")["status"] == "failed"


def test_batch_only_failure_is_batching(repo):
    all_stages(repo, batch_noise=1e-2)
    ev = run(repo)
    assert check(ev, "dotnet.batch.padded.vs_single")["status"] == "failed"
    assert taxonomies(ev)["dotnet-ids"] == ".NET batching"


def test_text_batch_entities_compare_with_the_rule(repo):
    all_stages(repo)

    def swap(d):
        rows = d["batches"]["text"]["values"]
        i = d["batches"]["text"]["case_ids"].index("basic")
        rows[i]["entities"] = swap_basic_persons(rows[i]["entities"])

    edit_results(repo, "dotnet", swap)
    ev = run(repo)
    assert ev.exit_code == evaluate.EXIT_OK
    assert set(kinds(ev)) == {("basic", "order-swap")}
    assert {x["check"] for x in ev.report["entity_exceptions"]} == {
        "dotnet.batch.text.vs_single",
        "dotnet.batch.text.vs_reference",
    }


def test_missing_export_case_is_missing_evidence_in_export(repo):
    repo.tokenizer()
    repo.export(drop="near")
    repo.dotnet()
    ev = run(repo)
    assert ev.report["first_divergence"] == "export"
    assert {f["stage"] for f in ev.report["findings"]} == {"export"}
    assert all(f["taxonomy"] == "missing evidence" for f in ev.report["findings"])
    assert check(ev, "export.span_logits")["status"] == "failed"
    assert ev.report["stages"]["dotnet-ids"] == "passed"


def test_missing_dotnet_text_case_is_missing_evidence_in_dotnet_text(repo):
    all_stages(repo, drop_text="empty")
    ev = run(repo)
    assert ev.report["first_divergence"] == "dotnet-text"
    assert ev.report["stages"]["dotnet-ids"] == "passed"
    assert {(f["stage"], f["taxonomy"]) for f in ev.report["findings"]} == {("dotnet-text", "missing evidence")}
    assert "missing cases ['empty']" in check(ev, "dotnet.text_entities")["detail"]


def test_missing_processor_case_fails_stage_1(repo):
    repo.tokenizer(drop="gate-closed")
    repo.export()
    repo.dotnet()
    ev = run(repo)
    assert ev.report["first_divergence"] == "tokenizer"
    assert {f["taxonomy"] for f in ev.report["findings"]} == {"missing evidence"}


def test_missing_stages_leave_gates_not_run(repo):
    repo.tokenizer()
    ev = run(repo)
    assert ev.report["gates"]["G1"]["status"] == "passed"
    assert ev.report["gates"]["G2"]["status"] == "not-run"


# -- identity and input validity ------------------------------------------------------------------------------


def test_manifest_without_a_graph_makes_runtime_stages_invalid(tmp_path):
    repo = SyntheticExtractionRepo(tmp_path / "repo", graph=False)
    repo.tokenizer()
    repo.export(assets=[])
    repo.dotnet()
    ev = run(repo)
    assert "pins no ONNX graph" in check(ev, "export.identity")["detail"]
    assert check(ev, "export.identity")["status"] == "invalid"
    assert ev.report["gates"]["G2"]["status"] == "failed"
    assert check(ev, "tokenizer.identity")["status"] == "passed"


def test_wrong_graph_data_digest_is_invalid(repo):
    repo.tokenizer()
    repo.export(assets=[{"path": "onnx/model.onnx", "sha256": "0" * 63 + "1"}, repo.GRAPH_ASSETS[1]])
    repo.dotnet()
    assert check(run(repo), "export.identity")["status"] == "invalid"


def test_stage_fed_other_inputs_is_invalid(repo):
    all_stages(repo)
    edit_results(repo, "export", lambda d: d["cases"]["basic"].update(schema_index=[0, 1, 2]))
    ev = run(repo)
    assert check(ev, "export.inputs")["status"] == "invalid"
    assert taxonomies(ev)["export"] == "packaging/harness"


def test_version_1_results_are_refused_for_an_extraction_profile(repo):
    repo.tokenizer()
    edit_results(repo, "tokenizer", lambda d: (d.pop("task"), d["cases"].clear(), d.update(format_version=1)))
    with pytest.raises(evaluate.EvaluationError, match="task 'embedding'"):
        run(repo)


def test_offsets_unit_other_than_scalars_is_refused(repo):
    all_stages(repo)
    repo.write_manifest(lambda m: m["behavior"].update(offsets_unit="utf16-code-units"))
    with pytest.raises(evaluate.EvaluationError, match="offsets_unit"):
        run(repo, base="none")


# -- schema ---------------------------------------------------------------------------------------------------


def schema_errors(doc: dict) -> list[str]:
    return [e.message for e in results._validator().iter_errors(doc)]


def base_doc(**kw) -> dict:
    doc = {
        "format": results.FORMAT,
        "format_version": 1,
        "profile": PROFILE,
        "stage": "export",
        "candidate": "x",
        "producer": {"name": "x"},
        "cases": {"a": {"embedding": [0.5]}},
    }
    doc.update(kw)
    return doc


def test_schema_version_1_stays_numeric_and_taskless():
    assert schema_errors(base_doc()) == []
    assert schema_errors(base_doc(task="embedding"))
    assert schema_errors(base_doc(cases={"a": {"normalized_text": "x"}}))


def test_schema_version_2_requires_a_task_and_types_entities():
    assert schema_errors(base_doc(format_version=2))
    entity = {"gate_open": True, "entities": [{"label": "p", "text": "A", "start": 0, "end": 1, "confidence": 1.0}]}
    ok = base_doc(format_version=2, task="entity-extraction", cases={"a": {"words": ["a"], "entities": entity}})
    assert schema_errors(ok) == []
    bad = json.loads(json.dumps(ok))
    del bad["cases"]["a"]["entities"]["entities"][0]["confidence"]
    assert schema_errors(bad)
    bad = json.loads(json.dumps(ok))
    bad["cases"]["a"]["entities"]["entities"][0]["start"] = -1
    assert schema_errors(bad)


def test_new_results_writes_version_1_without_a_task():
    doc = results.new_results(
        profile=PROFILE, stage="export", candidate="x", producer={"name": "x"}, environment={}, assets=[]
    )
    assert doc["format_version"] == 1 and "task" not in doc
    doc = results.new_results(
        profile=PROFILE,
        stage="export",
        candidate="x",
        producer={"name": "x"},
        environment={},
        assets=[],
        task="entity-extraction",
    )
    assert doc["format_version"] == 2 and doc["task"] == "entity-extraction"


# -- arithmetic ------------------------------------------------------------------------------------------------


def test_sigmoid32_matches_float32_values():
    assert sigmoid32(0.0) == 0.5
    assert sigmoid32(20.0) == 1.0  # saturates in float32, as in GLiNER2's output
    assert sigmoid32(-1200.0) == 0.0
    assert sigmoid32(-20.0) == pytest.approx(2.0611537e-09, rel=1e-6)
    assert f32(0.1) != 0.1 and abs(f32(0.1) - 0.1) < 1e-8


def test_tolerance_key_names_match_the_documented_proposal():
    keys = set(evaluate_extraction.EXPECTED_KEYS)
    flat = set(evaluate.flatten(TOLERANCES))
    assert flat <= keys
    assert {"float.span_logits_relative", "entities.threshold_flip_max_score_margin", "discrete.gate_decision"} <= keys
