"""Committed subset of the EmbeddingGemma 2 frame-sampling cases (C# unit tests, run in CI).

`model-pipeline/reference/embeddinggemma2/make_eg2_sampling_cases.py --out sampling-cases.json` (seed 20261011) calls
upstream's `EmbeddingGemma2VideoProcessor.sample_frames` (transformers 5.19.0; fps 1, max_frames 32, overflow
"uniform") on 20,849 cases (5.3 MB, not committed; SHA-256 below). This script keeps every structured case (the 399
metadata-less frame counts and the 450 boundary cases around 1 s and 32/33 sampled frames at 15 frame rates) and
every 40th seeded random case, and writes `video-sampling-cases.json` next to it. The C# tests check
`EmbeddingGemma2VideoSampling.SampleIndices` on this subset in CI and on the full file when
`STARLING_EMBEDDINGGEMMA2_SAMPLING_CASES` points at it.

Usage:
    python make_video_sampling_cases.py <sampling-cases.json>
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

FULL_SHA256 = "b9ef083bff2fb37aec9e2d66bd1d1bf9ea7e2f54b68ef64f20415361a362113a"
STRUCTURED = 399 + 15 * 15 * 2
EVERY = 40


def main() -> int:
    source = Path(sys.argv[1])
    data = source.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != FULL_SHA256:
        raise SystemExit(f"{source}: SHA-256 {digest} is not the spike's sampling-cases.json ({FULL_SHA256})")
    cases = json.loads(data)
    subset = cases[:STRUCTURED] + cases[STRUCTURED::EVERY]
    out = {
        "source": "make_eg2_sampling_cases.py (seed 20261011), transformers 5.19.0 EmbeddingGemma2VideoProcessor.sample_frames",
        "source_sha256": FULL_SHA256,
        "source_cases": len(cases),
        "selection": f"cases[:{STRUCTURED}] (structured) + cases[{STRUCTURED}::{EVERY}] (seeded random)",
        "cases": subset,
    }
    target = Path(__file__).with_name("video-sampling-cases.json")
    target.write_text(json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{len(subset)} of {len(cases)} cases -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
