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


def load_core():
    spec = importlib.util.spec_from_file_location("matched_core", RUNNER.parents[1] / "bench" / "matched_core.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # standard library only at import time
    return module


def test_native_library_digests_cover_onnxruntime_and_genai_shared_objects_only(tmp_path):
    core = load_core()
    ort = tmp_path / "libonnxruntime.so.1.30.0"
    ort.write_bytes(b"abc")
    maps = "\n".join(
        [
            f"7f00-7f01 r--p 00000000 103:02 4 {ort}",
            f"7f01-7f02 r-xp 00001000 103:02 4 {ort}",
            "7f02-7f03 r--p 00000000 103:02 5 /v/onnxruntime_genai/libonnxruntime-genai.so",
            "7f03-7f04 r--p 00000000 103:02 6 /v/capi/onnxruntime_pybind11_state.cpython-313-x86_64-linux-gnu.so",
            "7f04-7f05 rw-s 00000000 103:02 7 /h/.cache/Microsoft/DeveloperTools/.onnxruntime/onnxruntime-genai.db-shm",
            "7f05-7f06 r--p 00000000 103:02 8 /usr/lib/libcudart.so.13",
            "7f06-7f07 rw-p 00000000 00:00 0 [heap]",
            "7f07-7f08 rw-p 00000000 00:00 0",
        ]
    )
    assert core.onnxruntime_library_paths(maps) == sorted(
        [
            str(ort),
            "/v/onnxruntime_genai/libonnxruntime-genai.so",
            "/v/capi/onnxruntime_pybind11_state.cpython-313-x86_64-linux-gnu.so",
        ]
    )
    assert core.file_sha256(str(ort)) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert core.file_sha256(str(tmp_path / "missing.so")) is None
    digests = core.native_library_digests()
    assert digests == "unavailable" or isinstance(digests, list)
