"""The Python benchmark runner's quality gate (reference/encoder/bench_e5_matched.py, P-008)."""

import importlib.util
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "reference" / "encoder" / "bench_e5_matched.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("bench_e5_matched", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # stdlib imports only; NumPy and ONNX Runtime load lazily
    return module


def series(workload: str, path: str, max_abs: float, finite: bool = True) -> dict:
    return {
        "mode": "warm",
        "path": path,
        "workload": workload,
        "concurrency": 1,
        "quality": {"max_abs": max_abs, "min_cosine": 1.0, "finite": finite},
    }


def test_one_failing_series_fails_the_run_even_when_the_sweep_passed():
    runner = load_runner()
    quality = {"pass": True}
    all_series = [
        series("short-b1", "ids", 2e-7),
        series("long512-b32", "text", 3e-3),
        series("short-b8", "ids", 1e-7),
    ]
    assert runner.apply_quality_gate(quality, all_series, 1e-5) is False
    assert quality["pass"] is False
    assert quality["preliminary_pass"] is True
    assert quality["series_pass"] is False
    assert quality["failed_series"] == ["warm/text/long512-b32/c1"]
    assert [s["quality"]["pass"] for s in all_series] == [True, False, True]


def test_passing_series_keep_the_run_passing_and_non_finite_fails():
    runner = load_runner()
    quality = {"pass": True}
    assert runner.apply_quality_gate(quality, [series("short-b1", "ids", 2e-7)], 1e-5) is True
    assert quality["failed_series"] == []
    assert runner.apply_quality_gate({"pass": True}, [series("short-b8", "ids", 0.0, finite=False)], 1e-5) is False
    assert runner.apply_quality_gate({"pass": False}, [series("short-b1", "ids", 1e-7)], 1e-5) is False
