"""Embedding impact of other video decoders (EmbeddingGemma 2 video-path spike). CPU, fp32, PyTorch reference model.

For each decoder directory written by `decode_eg2_videos.py` (`<decode dir>/<name>/<video id>.npy`: frames at the
reference's sampled indices), every video whose frames differ from the reference is embedded through
sentence-transformers as a pre-decoded frame array without metadata (at most 32 frames, so upstream keeps them all and
processes them exactly as the sampled frames of the file) and compared with the reference embedding of the file.
A control decoder whose frames equal the reference (`--control`) checks that the array input reproduces the file's
embedding exactly.

Usage:
    python compare_eg2_video_decoders.py --model-dir <snapshot> --reference <dir> --decode <dir> --out <json>
        --decoders pyav opencv numpy-nearest numpy-bilinear [--control torchcodec-ffmpeg8]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--decode", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decoders", nargs="+", required=True)
    parser.add_argument("--control")
    parser.add_argument(
        "--control-videos", nargs="*", default=["h264-bars-640x360-30fps-5s", "vp9-odd-321x179-30fps-3s"]
    )
    args = parser.parse_args()
    rows = {
        r["id"]: r
        for r in map(json.loads, (args.reference / "video-embeddings.jsonl").read_text().splitlines())
        if r["kind"] == "video" and r["input_form"] == "path"
    }
    st = SentenceTransformer(str(args.model_dir), device="cpu", model_kwargs={"dtype": torch.float32})
    st.eval()

    def embed(frames: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return st.encode([frames], batch_size=1, convert_to_numpy=True)[0]

    result: dict = {}
    jobs = [(d, None) for d in args.decoders] + ([(args.control, args.control_videos)] if args.control else [])
    for name, only in jobs:
        summary = json.loads((args.decode / f"{name}.json").read_text())["videos"]
        entry: dict = {"videos": {}}
        for vid, info in summary.items():
            if only is not None and vid not in only:
                continue
            if "error" in info or info.get("max_level_diff") is None:
                entry["videos"][vid] = {"skipped": info.get("error") or "shape differs (rotation not applied)"}
                continue
            if info["equal"] and only is None:
                entry["videos"][vid] = {"frames_equal": True, "embedding_max_abs": 0.0, "cosine": 1.0}
                continue
            emb = embed(np.load(args.decode / name / f"{vid}.npy"))
            ref = np.array(rows[vid]["embedding"], np.float32)
            d = float(np.abs(emb - ref).max())
            cos = float(emb.astype(np.float64) @ ref / np.linalg.norm(emb) / np.linalg.norm(ref))
            entry["videos"][vid] = {
                "frames_equal": info["equal"],
                "max_level_diff": info["max_level_diff"],
                "mean_level_diff": info["mean_level_diff"],
                "embedding_max_abs": d,
                "cosine": cos,
            }
            print(name, vid, info["max_level_diff"], f"{d:.2e}", f"{cos:.7f}", flush=True)
        vals = [v for v in entry["videos"].values() if "embedding_max_abs" in v]
        entry["worst_embedding_max_abs"] = max((v["embedding_max_abs"] for v in vals), default=None)
        entry["min_cosine"] = min((v["cosine"] for v in vals), default=None)
        entry["videos_with_different_frames"] = sum(1 for v in vals if not v["frames_equal"])
        result[name] = entry
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: {x: y for x, y in v.items() if x != "videos"} for k, v in result.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
