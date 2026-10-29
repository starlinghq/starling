"""Compare two Starlings.Qwen35.Fidelity parity output directories case by case.

usage: python3 -I compare_runs.py <baseline dir> <candidate dir> <fixture generation.jsonl>
Prints, per case: prompt tokens, whether the generated ids and finish reason are equal, whether the step-0
log-softmax files are byte-identical, their max abs difference, and both prefill times.
"""

import hashlib
import json
import struct
import sys
from pathlib import Path


def rows(path):
    return {
        r["id"]: r
        for r in (
            json.loads(line)
            for line in Path(path).read_text().splitlines()
            if line.strip()
        )
    }


def floats(path):
    data = path.read_bytes()
    return struct.unpack(f"<{len(data) // 4}f", data)


def env(d):
    p = d / "environment.json"
    return (
        json.loads(p.read_text()).get("prefill_chunk_tokens")
        if p.exists()
        else "(environment.json not written yet)"
    )


def main():
    base, cand, fixture = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    a, b, fx = rows(base / "results.jsonl"), rows(cand / "results.jsonl"), rows(fixture)
    print(f"baseline  {base.name}: prefill_chunk_tokens={env(base)}")
    print(f"candidate {cand.name}: prefill_chunk_tokens={env(cand)}")
    hdr = f"{'case':22} {'prompt':>6} {'ids_eq':>6} {'finish_eq':>9} {'ref_exact':>9} {'step0_bytes_eq':>14} {'step0_max_abs':>13} {'prefill_ms_a':>12} {'prefill_ms_b':>12}"
    print(hdr)
    all_ids = all_step0 = True
    worst = 0.0
    for cid, ra in a.items():
        rb = b[cid]
        ids_eq = ra["output_ids"] == rb["output_ids"]
        fin_eq = ra["finish_reason"] == rb["finish_reason"]
        ref = fx[cid]["output_ids"]
        ref_exact = rb["output_ids"] == ref
        fa, fb = base / "step0" / f"{cid}.f32", cand / "step0" / f"{cid}.f32"
        bytes_eq = (
            hashlib.sha256(fa.read_bytes()).digest()
            == hashlib.sha256(fb.read_bytes()).digest()
        )
        diff = (
            0.0
            if bytes_eq
            else max(abs(x - y) for x, y in zip(floats(fa), floats(fb), strict=True))
        )
        worst = max(worst, diff)
        all_ids &= ids_eq and fin_eq
        all_step0 &= bytes_eq
        n = len(ra.get("prompt_ids") or fx[cid]["prompt_ids"])
        print(
            f"{cid:22} {n:6d} {ids_eq!s:>6} {fin_eq!s:>9} {ref_exact!s:>9} {bytes_eq!s:>14} {diff:13.3e} "
            f"{ra.get('prefill_ms', float('nan')):12.1f} {rb.get('prefill_ms', float('nan')):12.1f}"
        )
    print(
        f"cases {len(a)}; ids and finish equal on all: {all_ids}; step0 byte-identical on all: {all_step0}; max abs step0 difference {worst:.3e}"
    )


main()
