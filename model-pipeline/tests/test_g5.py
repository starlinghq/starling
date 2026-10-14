"""G5 budget checks (model_pipeline.g5): profile mapping, rule directions, missing data and the protected budgets."""

import hashlib
import json
from pathlib import Path

import pytest

from model_pipeline import benchmarks, cli, g5

REPO = Path(__file__).resolve().parents[2]
EG2_WORKLOADS = "ai-dotnet/benchmarks/workloads/embeddinggemma2-v1.json"


def run(run_id, model, language="csharp", impl="library", provider="cpu", threads=6, **extra):
    record = {
        "run_id": run_id,
        "model": model,
        "mode": "warm",
        "started_utc": "2026-10-14T00:00:00Z",
        "harness": {"name": f"{language}-{impl}", "language": language, "impl": impl, "version": "1"},
        "settings": {"provider": provider, "intra_op_num_threads": threads, **extra.pop("settings", {})},
        "environment": {"ort_version": "1.30.0"},
        "artifacts": {
            "model_sha256": "a" * 64,
            "workload_sha256": "b" * 64,
            "workload_file": None,
            "model_path": extra.pop("model_path", "/m/model.onnx"),
            "git_dirty": False,
        },
        "quality": {"pass": True},
    }
    record["artifacts"].update(extra.pop("artifacts", {}))
    record.update(extra)
    return record


def series(run_id, workload, ms, path="text", mode="warm", stages=None, metrics=None, ok=True, concurrency=1):
    s = {
        "run_id": run_id,
        "mode": mode,
        "path": path,
        "workload": workload,
        "concurrency": concurrency,
        "batch": 1,
        "n": len(ms),
        "ms": list(ms),
        "stages_ms": stages or {},
        "quality": {"pass": ok},
    }
    if metrics:
        s["metrics"] = metrics
    return s


def results(runs, all_series):
    return benchmarks.RawResults(runs={r["run_id"]: r for r in runs}, series=all_series)


def verdict(res, budgets, name):
    (v,) = g5.evaluate(res, budgets, only={name}, resamples=50)
    return v


def three_runs(prefix, model, language, impl, workload, value, **kw):
    """Three runs of one side, each with a series of 20 samples at ``value``."""
    runs = [run(f"{prefix}{i}", model, language, impl, **kw) for i in range(3)]
    return runs, [series(r["run_id"], workload, [value] * 20) for r in runs]


# ----------------------------------------------------------------------------------------------- profiles


def test_profile_mapping_by_model_provider_and_path():
    assert g5.profile_of(run("x", "multilingual-e5-small", provider="cuda"), {"path": "ids"}) == (
        "multilingual-e5-small/onnx-fp32-cuda"
    )
    eg2 = run("x", "embeddinggemma-2")
    assert g5.profile_of(eg2, {"path": "text"}) == "embeddinggemma-2/onnx-fp32-cpu"
    assert g5.profile_of(eg2, {"path": "image"}) == "embeddinggemma-2/image-fp32-cpu"
    assert g5.profile_of(run("x", "gliner2-base-v1"), {"path": "text"}) == "gliner2-base-v1/onnx-fp32-cpu"
    # Qwen: the Python runner records the variant; the C# harness's model path names the bundle directory.
    py = run("x", "qwen3.5-0.8b", "python", "python-genai", provider="cuda", settings={"variant": "fp16x"})
    assert g5.profile_of(py, {"path": "ids"}) == "qwen3.5-0.8b/ortgenai-fp16x-cuda"
    cs = run("x", "qwen3.5-0.8b", model_path="/snap/ortgenai-fp32x/model.onnx")
    assert g5.profile_of(cs, {"path": "ids"}) == "qwen3.5-0.8b/ortgenai-fp32x-cpu"
    assert g5.profile_of(run("x", "unknown-model"), {"path": "text"}) is None


def test_sides_python_by_language_not_impl():
    assert g5.side_of(run("x", "m", "python", "python-genai")) == "python"
    assert g5.side_of(run("x", "m", "csharp", "library")) == "csharp"
    assert g5.side_of(run("x", "m", "csharp", "direct")) == "direct"


# -------------------------------------------------------------------------------------------------- rules


def test_warm_ratio_passes_at_the_budget_and_fails_above():
    budgets = {"gliner2-base-v1/onnx-fp32-cpu": {"comparison": "x", "warm_median_ratio_max": 1.10}}
    name = "gliner2-base-v1/onnx-fp32-cpu"
    py_runs, py_series = three_runs("p", "gliner2-base-v1", "python", "python-ort", "short", 10.0)
    for value, expected in ((11.0, "pass"), (11.2, "fail")):
        cs_runs, cs_series = three_runs("c", "gliner2-base-v1", "csharp", "library", "short", value)
        v = verdict(results(cs_runs + py_runs, cs_series + py_series), budgets, name)
        assert v["verdict"] == expected
        (check,) = v["rules"]["warm_median_ratio_max"]["checks"]
        assert check["value"] == pytest.approx(value / 10.0)
        assert check["runs"] == [3, 3]
        assert v["qualifying"] == []


def test_decode_tokens_per_second_is_a_minimum_and_ttft_a_maximum():
    model, name = "qwen3.5-0.8b", "qwen3.5-0.8b/ortgenai-fp32x-cpu"
    budgets = {name: {"decode_tokens_per_second_ratio_min": 0.95, "time_to_first_token_ratio_max": 1.10}}
    kw = {"model_path": "/s/ortgenai-fp32x/model.onnx"}
    runs = [run("c", model, **kw), run("p", model, "python", "python-genai", settings={"variant": "fp32x"}, **kw)]

    def s(rid, tps, ttft):
        metrics = {"decode_tokens_per_s": [tps] * 5, "ttft_ms": [ttft] * 5}
        return series(rid, "short-decode64", [1.0] * 5, path="ids", metrics=metrics)

    v = verdict(results(runs, [s("c", 9.6, 100.0), s("p", 10.0, 100.0)]), budgets, name)
    assert v["rules"]["decode_tokens_per_second_ratio_min"]["status"] == "pass"  # 0.96 >= 0.95
    assert v["rules"]["time_to_first_token_ratio_max"]["status"] == "pass"
    v = verdict(results(runs, [s("c", 9.4, 120.0), s("p", 10.0, 100.0)]), budgets, name)
    assert v["rules"]["decode_tokens_per_second_ratio_min"]["status"] == "fail"  # 0.94 < 0.95
    assert v["rules"]["time_to_first_token_ratio_max"]["status"] == "fail"  # 1.2 > 1.1
    assert v["verdict"] == "fail"


def test_inference_ratio_uses_the_inference_stage_not_the_total():
    name = "gliner2-base-v1/onnx-fp32-cpu"
    budgets = {name: {"inference_ratio_max": 1.10, "prepare_plus_decode_share_of_inference_max": 0.05}}
    runs = [run("c", "gliner2-base-v1"), run("p", "gliner2-base-v1", "python", "python-ort")]
    cs = series("c", "short", [30.0] * 4, stages={"prepare": [0.2] * 4, "inference": [10.0] * 4, "decode": [0.2] * 4})
    py = series("p", "short", [10.0] * 4, stages={"inference": [10.0] * 4, "run": [9.0] * 4})
    v = verdict(results(runs, [cs, py]), budgets, name)
    assert v["rules"]["inference_ratio_max"]["checks"][0]["value"] == pytest.approx(1.0)
    assert v["rules"]["prepare_plus_decode_share_of_inference_max"]["checks"][0]["value"] == pytest.approx(0.04)
    assert v["verdict"] == "pass"
    cs["stages_ms"]["decode"] = [0.4] * 4  # (0.2 + 0.4) / 10 = 0.06 > 0.05
    v = verdict(results(runs, [cs, py]), budgets, name)
    assert v["rules"]["prepare_plus_decode_share_of_inference_max"]["status"] == "fail"


def test_a_workload_without_its_python_partner_is_no_data_not_a_pass():
    name = "gliner2-base-v1/onnx-fp32-cpu"
    budgets = {name: {"warm_median_ratio_max": 1.10}}
    runs = [run("c", "gliner2-base-v1"), run("p", "gliner2-base-v1", "python", "python-ort")]
    all_series = [series("c", "a", [1.0] * 3), series("p", "a", [1.0] * 3), series("c", "b", [1.0] * 3)]
    v = verdict(results(runs, all_series), budgets, name)
    assert v["rules"]["warm_median_ratio_max"]["status"] == "incomplete"
    assert v["verdict"] == "incomplete"
    missing = [c for c in v["rules"]["warm_median_ratio_max"]["checks"] if c["pass"] is None]
    assert [c["workload"] for c in missing] == ["warm/text/b/c1"] and missing[0]["missing"] == "Python"
    only_cs = verdict(results(runs[:1], [series("c", "a", [1.0])]), budgets, name)
    assert only_cs["verdict"] == "no data" or only_cs["rules"]["warm_median_ratio_max"]["status"] == "no data"


def test_cold_and_adapter_overhead_rules_for_e5():
    name = "multilingual-e5-small/onnx-fp32-cpu"
    budgets = {name: {"cold_start_ratio_max": 1.0, "adapter_overhead_max": g5.ADAPTER_OVERHEAD}}
    model = "multilingual-e5-small"
    runs = [run("c", model), run("d", model, impl="direct"), run("p", model, "python", "python-ort")]
    all_series = [
        series("c", "short-b1", [5.9] * 5, path="ids"),
        series("d", "short-b1", [5.0] * 5, path="ids"),  # 0.9 ms over direct: within max(1 ms, 0.5 ms)
        series("c", "cold-start", [900.0] * 10, mode="cold"),
        series("p", "cold-start", [1000.0] * 10, mode="cold"),
    ]
    v = verdict(results(runs, all_series), budgets, name)
    assert v["rules"]["adapter_overhead_max"]["checks"][0]["limit_ms"] == pytest.approx(1.0)
    assert v["rules"]["adapter_overhead_max"]["status"] == "pass"
    assert v["rules"]["cold_start_ratio_max"]["checks"][0]["value"] == pytest.approx(0.9)
    assert v["verdict"] == "pass"
    all_series[0]["ms"] = [6.2] * 5  # 1.2 ms over direct
    v = verdict(results(runs, all_series), budgets, name)
    assert v["rules"]["adapter_overhead_max"]["status"] == "fail"
    with pytest.raises(ValueError, match="not the implemented rule"):
        g5.evaluate(results(runs, all_series), {name: {"adapter_overhead_max": "p95"}}, resamples=10)


def test_preprocess_share_maps_samples_to_rows_through_the_workload_file():
    name = "embeddinggemma-2/image-fp32-cpu"
    budgets = {name: {"preprocess_share_of_embed_max": 0.05}}
    sha = hashlib.sha256((REPO / EG2_WORKLOADS).read_bytes()).hexdigest()
    cs = run("c", "embeddinggemma-2", artifacts={"workload_file": EG2_WORKLOADS, "workload_sha256": sha})
    # preprocess-only: requests photo-chelsea, photo-coffee, size-huge-4000x3000 (sample i -> request i % 3).
    pre = series("c", "preprocess-only", [1.0] * 6, path="image", stages={"preprocess": [1.0, 2.0, 50.0] * 2})
    # photo-b1: requests photo-chelsea, photo-coffee, photo-camera; huge-b1: size-huge-4000x3000.
    photo = series("c", "photo-b1", [1.0] * 6, path="image", stages={"embed": [100.0, 100.0, 7.0] * 2})
    huge = series("c", "huge-b1", [1.0] * 2, path="image", stages={"embed": [500.0] * 2})
    v = verdict(results([cs], [pre, photo, huge]), budgets, name)
    checks = {c["workload"].split(":")[1]: c for c in v["rules"]["preprocess_share_of_embed_max"]["checks"]}
    assert checks["photo-chelsea"]["value"] == pytest.approx(0.01)
    assert checks["photo-coffee"]["value"] == pytest.approx(0.02)
    assert checks["size-huge-4000x3000"]["value"] == pytest.approx(0.10)
    assert v["rules"]["preprocess_share_of_embed_max"]["status"] == "fail"
    v = verdict(results([cs], [pre, photo]), budgets, name)  # no huge-b1: that row has no denominator
    assert checks and v["rules"]["preprocess_share_of_embed_max"]["status"] == "incomplete"
    cs["artifacts"]["workload_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 differs"):
        verdict(results([cs], [pre, photo]), budgets, name)


def test_quality_failures_and_mismatched_sides_fail_the_profile():
    name = "gliner2-base-v1/onnx-fp32-cpu"
    budgets = {name: {"warm_median_ratio_max": 1.10}}
    runs = [run("c", "gliner2-base-v1"), run("p", "gliner2-base-v1", "python", "python-ort")]
    all_series = [series("c", "a", [1.0] * 3), series("p", "a", [1.0] * 3), series("p", "b", [1.0], ok=False)]
    v = verdict(results(runs, all_series), budgets, name)
    assert v["verdict"] == "fail" and any("quality failed" in p for p in v["problems"])
    runs[1]["settings"]["intra_op_num_threads"] = 4
    v = verdict(results(runs, all_series[:2]), budgets, name)
    assert v["verdict"] == "fail" and any("intra_op_num_threads" in p for p in v["problems"])
    runs[1]["settings"]["intra_op_num_threads"] = 6
    runs[1]["environment"]["ort_version"] = "1.31.0"
    v = verdict(results(runs, all_series[:2]), budgets, name)
    assert any("ort version" in p for p in v["problems"])


def test_unknown_budget_keys_are_errors():
    runs = [run("c", "gliner2-base-v1")]
    with pytest.raises(ValueError, match="no implemented rule"):
        g5.evaluate(results(runs, [series("c", "a", [1.0])]), {"gliner2-base-v1/onnx-fp32-cpu": {"p99_max": 2}})


# ------------------------------------------------------------------------------------- protected budgets


def test_every_key_of_the_protected_budgets_has_a_rule():
    budgets, comments = g5.load_budgets()
    empty = {"runs": {}, "groups": {}}
    for profile, keys in budgets.items():
        for key, budget in keys.items():
            if key not in g5.DESCRIPTIVE_KEYS:
                assert g5.evaluate_rule(key, budget, empty, 10, 1) == [], (profile, key)
    assert "upper 95% bound" in comments[("multilingual-e5-small/onnx-fp32-cpu", "warm_median_ratio_max")]


def test_cli_exit_code_and_json(tmp_path, capsys):
    raw = tmp_path / "raw.jsonl"
    raw.write_text("", encoding="utf-8")
    out = tmp_path / "g5.json"
    assert cli.main(["bench", "g5", str(raw), "--json", str(out)]) == 1  # no data is never a pass
    assert json.loads(out.read_text()) == []
