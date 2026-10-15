"""The Qwen3.5 Python runner's greedy-ids rule per profile (reference/bench/bench_qwen35_matched.py, G5)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

BENCH = Path(__file__).resolve().parents[1] / "reference" / "bench"
PROTECTED = Path(__file__).resolve().parents[1] / "protected" / "qwen3.5-0.8b"


def load_runner():
    sys.path.insert(0, str(BENCH))
    try:
        spec = importlib.util.spec_from_file_location("bench_qwen35_matched", BENCH / "bench_qwen35_matched.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # stdlib imports only; GenAI loads lazily
        return module
    finally:
        sys.path.remove(str(BENCH))


def test_fp16x_profile_allows_divergence_at_a_near_tie_only():
    runner = load_runner()
    assert runner.near_tie_margin(PROTECTED / "ortgenai-fp32x-cpu" / "tolerances.yaml") is None
    assert runner.near_tie_margin(PROTECTED / "ortgenai-fp32x-cuda" / "tolerances.yaml") is None
    assert runner.near_tie_margin(PROTECTED / "ortgenai-fp16x-cuda" / "tolerances.yaml") == 0.05

    case = {"id": "c", "prompt_ids": [1, 2], "output_ids": [5, 6, 7, 8], "step_margin": [0.9, 0.01, 0.9, 0.2]}

    def output(tokens):
        return {"case": case, "max_new": 4, "prompt_ids": [1, 2], "tokens": tokens, "stop": None}

    near_tie = output([5, 9, 9, 9])  # diverges at step 1, reference margin 0.01
    clear = output([5, 6, 9, 9])  # diverges at step 2, reference margin 0.9
    exact = SimpleNamespace(near_tie=None, tolerance_file=PROTECTED / "ortgenai-fp32x-cuda" / "tolerances.yaml")
    exact.greedy_rule = lambda: runner.QwenRun.greedy_rule(exact)
    fp16 = SimpleNamespace(near_tie=0.05, tolerance_file=PROTECTED / "ortgenai-fp16x-cuda" / "tolerances.yaml")
    fp16.greedy_rule = lambda: runner.QwenRun.greedy_rule(fp16)

    assert runner.QwenRun.judge(exact, [(0, near_tie)])["pass"] is False
    q = runner.QwenRun.judge(fp16, [(0, near_tie)])
    assert q["pass"] is True and q["diverged_at_near_tie"] == 1
    q = runner.QwenRun.judge(fp16, [(0, clear)])
    assert q["pass"] is False and "diverges at step 2" in q["failures"][0]
