"""Frame-sampling cases for the EmbeddingGemma 2 video path: (total frames, fps, duration) -> upstream indices.

Calls `EmbeddingGemma2VideoProcessor.sample_frames` (transformers 5.19.0) with the checkpoint's settings (fps 1,
max_frames 32, overflow "uniform") on a seeded mix of common and random frame rates, durations that are exact,
slightly short or long of the frame count, the 32/33-frame boundary, sub-second clips, and metadata-less inputs
(fps and duration None: every frame kept, then the 32-frame cap). A C# port checks itself against the output.

Usage:
    python make_eg2_sampling_cases.py --out <cases.json> [--count 20000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers.models.embedding_gemma2.video_processing_embedding_gemma2 import EmbeddingGemma2VideoProcessor
from transformers.video_utils import VideoMetadata

SEED = 20261011
RATES = [1, 5, 10, 12, 15, 23.976023976023978, 24, 25, 29.97002997002997, 30, 48, 50, 59.94005994005994, 60, 120]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20000)
    args = parser.parse_args()
    vp = EmbeddingGemma2VideoProcessor(max_soft_tokens=140)
    g = np.random.default_rng(SEED)
    cases = []

    def add(total: int, fps, duration) -> None:
        md = VideoMetadata(total_num_frames=total, fps=fps, duration=duration)
        idx = vp.sample_frames(md, fps=1, max_frames=32, overflow_strategy="uniform").tolist()
        cases.append({"total": total, "fps": fps, "duration": duration, "expected": idx})

    for n in range(1, 400):  # metadata-less: arange then the 32-frame cap
        add(n, None, None)
    for fps in RATES:  # boundaries around 1 s and 32/33 sampled frames
        for seconds in (0.04, 0.5, 0.999, 1.0, 1.001, 1.9, 2.0, 31.0, 31.999, 32.0, 32.5, 33.0, 33.01, 64.0, 600.0):
            total = max(1, int(round(seconds * fps)))
            add(total, fps, seconds)
            add(total, fps, total / fps)
    for _ in range(args.count):
        fps = float(g.choice(RATES)) if g.random() < 0.7 else float(g.uniform(0.25, 240))
        total = int(g.integers(1, 20000))
        jitter = float(g.choice([0.0, 0.0, g.uniform(-1, 1), g.uniform(-0.01, 0.01)]))
        duration = max(0.0, total / fps + jitter)
        add(total, fps, duration)
    args.out.write_text(json.dumps(cases) + "\n", encoding="utf-8")
    print(len(cases), "cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
