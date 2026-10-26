"""Summarize the alternating long-license prefill timing runs (8 new tokens each, one process per run).

usage: python3 -I timing_summary.py <runs dir>
"""

import json
import statistics
import sys
from pathlib import Path

runs = Path(sys.argv[1])
base = runs / "timing-1-one-call"
base_ids = json.loads((base / "results.jsonl").read_text().splitlines()[0])[
    "output_ids"
]
base_step0 = (base / "step0" / "long-license.f32").read_bytes()
by_mode = {}
for d in sorted(runs.glob("timing-*")):
    r = json.loads((d / "results.jsonl").read_text().splitlines()[0])
    mode = d.name.split("-", 2)[2]
    same_ids = r["output_ids"] == base_ids
    same_step0 = (d / "step0" / "long-license.f32").read_bytes() == base_step0
    by_mode.setdefault(mode, []).append(r["prefill_ms"])
    print(
        f"{d.name:24} prefill_ms {r['prefill_ms']:9.1f}  ids_equal_one_call {same_ids}  step0_bytes_equal_one_call {same_step0}"
    )
for mode, xs in by_mode.items():
    print(
        f"{mode:12} n={len(xs)} median_ms {statistics.median(xs):9.1f} min_ms {min(xs):9.1f} max_ms {max(xs):9.1f}"
    )
one = statistics.median(by_mode["one-call"])
for mode, xs in by_mode.items():
    print(f"{mode:12} median ratio to one-call {statistics.median(xs) / one:.3f}")
