import json
import random

from model_pipeline import benchmarks
from model_pipeline.cli import main

SHA = "0" * 64


def run_record(run_id: str, language: str, impl: str, provider: str = "cpu", mode: str = "warm") -> dict:
    load = {"loadavg": [0.1, 0.2, 0.3], "gpus": None}
    return {
        "schema": "starlings-bench-raw/1",
        "kind": "run",
        "run_id": run_id,
        "started_utc": "2026-10-09T00:00:00Z",
        "mode": mode,
        "model": "multilingual-e5-small",
        "harness": {"name": "test", "language": language, "impl": impl, "version": "1"},
        "artifacts": {
            "model_sha256": SHA,
            "tokenizer_sha256": SHA,
            "workload_sha256": SHA,
            "fixture_sha256": {"embeddings.jsonl": SHA},
            "git_commit": "abc",
            "git_dirty": False,
            "packages": {"onnxruntime": "1.30.0"},
        },
        "environment": {
            "cpu_model": "test cpu",
            "cpu_flags": ["avx2"],
            "physical_cores": 6,
            "logical_cpus": 12,
            "ram_bytes": 1,
            "os": "linux",
            "kernel": "7",
            "runtime": "x",
            "ort_version": "1.30.0",
            "ort_available_providers": ["CPUExecutionProvider"],
            "gpu": None,
            "cpu_governor": None,
            "load_before": load,
            "load_after": load,
        },
        "settings": {
            "provider": provider,
            "intra_op_num_threads": 6,
            "inter_op_num_threads": 1,
            "execution_mode": "sequential",
            "graph_optimization_level": "all",
            "provider_options": {},
        },
        "quality": {
            "reference": "embeddings.jsonl",
            "embedding_max_abs_tolerance": 1e-5,
            "cases": 46,
            "pass": True,
            "ids_single": {"max_abs": 2e-7, "min_cosine": 1.0, "finite": True},
            "text_single": {"max_abs": 2e-7, "min_cosine": 1.0, "finite": True},
        },
    }


def series_record(run_id: str, ms: list[float], workload: str = "short-b1", stages: bool = True) -> dict:
    return {
        "schema": "starlings-bench-raw/1",
        "kind": "series",
        "run_id": run_id,
        "mode": "warm",
        "path": "ids",
        "workload": workload,
        "input": "short",
        "batch": 1,
        "concurrency": 1,
        "warmup_calls": 10,
        "n": len(ms),
        "wall_ms": sum(ms),
        "ms": ms,
        "stages_ms": {"run": [m * 0.9 for m in ms]} if stages else {},
        "quality": {"max_abs": 2e-7, "min_cosine": 1.0, "finite": True},
    }


def write(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_quantile_matches_linear_interpolation():
    values = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert benchmarks.quantile(values, 0.5) == 3.0
    assert benchmarks.quantile(values, 0.0) == 1.0
    assert benchmarks.quantile(values, 1.0) == 5.0
    assert benchmarks.quantile(values, 0.25) == 2.0
    assert abs(benchmarks.quantile([1.0, 2.0], 0.95) - 1.95) < 1e-12


def test_tails_are_refused_for_small_samples():
    small = [float(i) for i in range(999)]
    assert benchmarks.tail(small, 0.99, benchmarks.P99_MIN_SAMPLES) is None
    assert benchmarks.tail(small + [999.0], 0.99, benchmarks.P99_MIN_SAMPLES) is not None
    assert benchmarks.tail(small[:19], 0.95, benchmarks.TAIL_MIN_SAMPLES) is None


def test_bootstrap_is_deterministic_and_brackets_the_median():
    rng = random.Random(7)
    runs = [[rng.gauss(10, 1) for _ in range(200)] for _ in range(3)]
    lo, hi = benchmarks.bootstrap(runs, benchmarks.median, resamples=300, seed=3)
    assert (lo, hi) == benchmarks.bootstrap(runs, benchmarks.median, resamples=300, seed=3)
    pooled = [x for r in runs for x in r]
    assert lo <= benchmarks.median(pooled) <= hi
    assert hi - lo < 1.0


def test_run_to_run_variation_widens_the_interval():
    same = [[10.0 + 0.01 * i for i in range(100)] for _ in range(3)]
    shifted = [[base + 0.01 * i for i in range(100)] for base in (9.0, 10.0, 11.0)]
    lo_same, hi_same = benchmarks.bootstrap(same, benchmarks.median, resamples=300)
    lo_shift, hi_shift = benchmarks.bootstrap(shifted, benchmarks.median, resamples=300)
    assert hi_shift - lo_shift > 3 * (hi_same - lo_same)


def test_load_validates_and_reports_problems(tmp_path):
    good = tmp_path / "good.jsonl"
    write(good, [run_record("r1", "python", "python-ort"), series_record("r1", [1.0, 2.0, 3.0])])
    results, problems = benchmarks.load([good])
    assert problems == []
    assert len(results.runs) == 1 and len(results.series) == 1

    bad_series = series_record("r1", [1.0, 2.0])
    bad_series["n"] = 3
    orphan = series_record("missing", [1.0])
    no_artifacts = run_record("r2", "csharp", "library")
    del no_artifacts["artifacts"]
    bad = tmp_path / "bad.jsonl"
    write(bad, [run_record("r1", "python", "python-ort"), bad_series, orphan, no_artifacts])
    _, problems = benchmarks.load([bad])
    assert any("n=3 but 2 samples" in p for p in problems)
    assert any("unknown run missing" in p for p in problems)
    assert any("artifacts" in p for p in problems)


def test_summary_compares_csharp_with_python_and_checks_the_overhead_budget(tmp_path):
    rng = random.Random(1)
    records = []
    for i in range(2):
        for language, impl, center in (
            ("csharp", "library", 10.2),
            ("csharp", "direct", 10.0),
            ("python", "python-ort", 11.0),
        ):
            run_id = f"{impl}-{i}"
            records.append(run_record(run_id, language, impl))
            records.append(series_record(run_id, [rng.gauss(center, 0.1) for _ in range(100)]))
    path = tmp_path / "raw.jsonl"
    write(path, records)
    results, problems = benchmarks.load([path])
    assert problems == []
    summary = benchmarks.summarize(results, resamples=200)

    (row,) = summary["csharp_vs_python"]
    assert row["a"] == "library" and row["b"] == "python-ort"
    assert row["n_a"] == 200 and row["runs_a"] == 2
    assert 0.9 < row["ratio"] < 0.95
    assert row["ratio_ci95"][0] <= row["ratio"] <= row["ratio_ci95"][1]

    (overhead,) = summary["adapter_overhead_p95"]
    assert overhead["budget_ms"] == max(1.0, 0.1 * overhead["b_ms"])
    assert overhead["verdict"] == "met"

    group = next(g for g in summary["groups"] if g["impl"] == "python-ort")
    assert group["p99_ms"] is None  # 200 samples: p99 not reported
    assert group["p95_ms"] is not None
    text = benchmarks.markdown(summary)
    assert "C# (library) vs Python" in text
    assert "| met |" in text


def test_overhead_verdicts():
    row = {"b_ms": 50.0, "delta_ci95_ms": [0.5, 4.0]}
    assert benchmarks.overhead_budget(row)["budget_ms"] == 5.0
    assert benchmarks.overhead_budget(row)["verdict"] == "met"
    assert benchmarks.overhead_budget({"b_ms": 50.0, "delta_ci95_ms": [4.0, 6.0]})["verdict"] == "inconclusive"
    assert benchmarks.overhead_budget({"b_ms": 50.0, "delta_ci95_ms": [5.5, 6.0]})["verdict"] == "not met"


def test_cli_summarize(tmp_path, capsys):
    path = tmp_path / "raw.jsonl"
    write(path, [run_record("r1", "python", "python-ort"), series_record("r1", [1.0] * 30)])
    out_json = tmp_path / "summary.json"
    assert main(["bench", "validate", str(path)]) == 0
    assert main(["bench", "summarize", str(path), "--resamples", "50", "--json", str(out_json)]) == 0
    assert "All series" in capsys.readouterr().out
    assert json.loads(out_json.read_text())["groups"][0]["n"] == 30


def test_summary_excludes_series_that_failed_quality(tmp_path):
    good = series_record("r1", [1.0] * 30)
    bad = series_record("r1", [0.5] * 30, workload="long512-b32")
    bad["quality"]["max_abs"] = 3e-3  # outside the run's 1e-5 tolerance
    path = tmp_path / "raw.jsonl"
    records = [run_record("r1", "csharp", "library"), good, bad]
    records += [run_record("r2", "python", "python-ort"), series_record("r2", [1.1] * 30)]
    write(path, records)
    results, problems = benchmarks.load([path])
    assert problems == []
    summary = benchmarks.summarize(results, resamples=50)

    (failure,) = summary["quality_failures"]
    assert failure["workload"] == "long512-b32"
    assert failure["reason"] == "series outside tolerance"
    assert all(g["workload"] != "long512-b32" for g in summary["groups"])
    assert "Quality failures: excluded" in benchmarks.markdown(summary)


def test_summary_excludes_every_series_of_a_run_whose_quality_failed(tmp_path):
    run = run_record("r1", "csharp", "library")
    run["quality"]["pass"] = False
    path = tmp_path / "raw.jsonl"
    write(path, [run, series_record("r1", [1.0] * 30)])
    results, _ = benchmarks.load([path])
    summary = benchmarks.summarize(results, resamples=50)
    assert summary["groups"] == []
    assert summary["quality_failures"][0]["reason"] == "run quality failed"


def generation_records(run_id: str, passed: bool = True) -> list[dict]:
    """A model-harness run (no embedding tolerance) with one generation series carrying metrics and its own verdict."""
    run = run_record(run_id, "csharp", "library")
    run["model"] = "qwen3.5-0.8b"
    run["profile"] = "ortgenai-fp32x-cpu"
    run["quality"] = {"reference": "generation.jsonl", "cases": 22, "pass": True}
    series = series_record(run_id, [100.0 + i for i in range(20)], workload="short-decode64")
    series["stages_ms"] = {"template": [0.0] * 20, "prefill": [10.0] * 20, "decode": [90.0] * 20}
    series["metrics"] = {"ttft_ms": [12.0] * 20, "decode_tokens_per_s": [40.0 + i for i in range(20)]}
    series["quality"] = {"check": "greedy ids", "generations": 5, "equal": 5 if passed else 4, "pass": passed}
    return [run, series]


def test_model_harness_records_validate_and_summarize_metrics(tmp_path):
    path = tmp_path / "raw.jsonl"
    write(path, generation_records("g1"))
    results, problems = benchmarks.load([path])
    assert problems == []
    summary = benchmarks.summarize(results, resamples=50)
    (group,) = summary["groups"]
    assert group["quality_max_abs"] is None
    assert group["metric_median"]["ttft_ms"] == 12.0
    assert group["metric_median"]["decode_tokens_per_s"] == 49.5
    text = benchmarks.markdown(summary)
    assert "Per-sample metrics (medians)" in text
    assert "22 cases, True" in text


def test_model_harness_series_failing_its_own_check_is_excluded(tmp_path):
    path = tmp_path / "raw.jsonl"
    write(path, generation_records("g1", passed=False))
    results, problems = benchmarks.load([path])
    assert problems == []
    summary = benchmarks.summarize(results, resamples=50)
    assert summary["groups"] == []
    (failure,) = summary["quality_failures"]
    assert failure["reason"] == "series failed its own check"
    assert failure["max_abs"] is None
    assert "Quality failures: excluded" in benchmarks.markdown(summary)


def test_metric_arrays_must_match_the_sample_count(tmp_path):
    run, series = generation_records("g1")
    series["metrics"]["ttft_ms"] = [1.0]
    path = tmp_path / "raw.jsonl"
    write(path, [run, series])
    _, problems = benchmarks.load([path])
    assert any("metric ttft_ms has 1 values" in p for p in problems)


def test_series_quality_needs_embedding_errors_or_a_verdict(tmp_path):
    run, series = generation_records("g1")
    series["quality"] = {"check": "greedy ids"}
    path = tmp_path / "raw.jsonl"
    write(path, [run, series])
    _, problems = benchmarks.load([path])
    assert problems


def test_runs_of_different_models_are_not_pooled(tmp_path):
    a = generation_records("g1")
    b = generation_records("g2")
    b[0]["model"] = "gliner2-base-v1"
    path = tmp_path / "raw.jsonl"
    write(path, a + b)
    results, _ = benchmarks.load([path])
    assert len(benchmarks.summarize(results, resamples=50)["groups"]) == 2
