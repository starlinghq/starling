"""Compare a C# video-prototype run (`embed` mode) with the reference (EmbeddingGemma 2 video-path spike).

Reads `<run>/run.json` (per row: owned-tokenizer ids equal, per video frame/resized/pixel/position digests equal,
timings) and `<run>/embeddings.f32` (768 float32 per row, in run.json order) and reports per row and worst: embedding
max abs and cosine (float64) against `video-embeddings.jsonl`.

Usage:
    python compare_embeddinggemma2_video_csharp.py --reference <dir> --run <dir> --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ref = {r["id"]: r for r in map(json.loads, (args.reference / "video-embeddings.jsonl").read_text().splitlines())}
    run = json.loads((args.run / "run.json").read_text())
    embs = np.fromfile(args.run / "embeddings.f32", np.float32).reshape(-1, 768)
    rows, worst = {}, {"embedding_max_abs": 0.0, "min_cosine": 1.0}
    flags = {
        "ids_equal": 0,
        "frames_equal": 0,
        "resized_equal": 0,
        "pixels_equal": 0,
        "positions_equal": 0,
        "videos": 0,
    }
    for k, (rid, rec) in enumerate(run["rows"].items()):
        e = embs[k].astype(np.float64)
        r = np.array(ref[rid]["embedding"], np.float64)
        d = float(np.abs(e - r).max())
        cos = float(e @ r / np.linalg.norm(e) / np.linalg.norm(r))
        rows[rid] = {"embedding_max_abs": d, "cosine": cos} | {
            k2: rec[k2] for k2 in ("tokens", "ids_equal_owned_tokenizer", "videos")
        }
        worst["embedding_max_abs"] = max(worst["embedding_max_abs"], d)
        worst["min_cosine"] = min(worst["min_cosine"], cos)
        flags["ids_equal"] += rec["ids_equal_owned_tokenizer"]
        for v in rec["videos"]:
            flags["videos"] += 1
            flags["frames_equal"] += v["frames_uint8_equal"]
            flags["resized_equal"] += v["resized_uint8_equal"]
            flags["pixels_equal"] += v["pixel_values_equal"]
            flags["positions_equal"] += v["positions_equal"]
    result = {
        "run": {k: v for k, v in run.items() if k != "rows"},
        "rows": len(rows),
        "counts": flags,
        "worst": worst,
        "per_row": rows,
    }
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_row"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
