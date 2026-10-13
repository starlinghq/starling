"""ONNX Runtime parity for the EmbeddingGemma 2 video path (video-path spike), on the released image-path graphs.

The released vision graph (`embeddinggemma2-vision-fp32.onnx`, any multiple of 9 patches) and multimodal text graph
(`embeddinggemma2-text-mm-fp32.onnx`, `multimodal_embeds` and `multimodal_mask` inputs) are used unchanged. Inputs
come from the reference (`run_embeddinggemma2_video.py --tensors`): the sampled frames' resized uint8 pixels are
rescaled and patched as upstream (`eg2_video.preprocess`, checked against the reference digests), the frames go
through the vision graph, and the soft tokens are scattered at the `<|video|>` positions (and `<|image|>` for the one
row with an image, whose pixels come from the image processor restatement).

Per row: soft tokens (row-relative), token embeddings (row-relative), embedding max abs and cosine. Vision layouts:
frames padded to 1,260 patches in batches of `--frame-batch` (upstream pads every frame to 1,260), unpadded frames
(`9 * count` patches) in the same batches, and one frame at a time; the text graph also runs padded batches of rows
of different lengths.

Usage:
    python parity_embeddinggemma2_video_ort.py --vision <graph> --text <graph> --reference <dir> --tensors <dir>
        --out <json> [--provider cpu|cuda] [--threads N] [--tf32 0|1] [--frame-batch 8]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import eg2_video
import numpy as np
import onnxruntime as ort
import torch

IMAGE_TOKEN = 258880
VIDEO_TOKEN = eg2_video.VIDEO_TOKEN


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def session(path: Path, provider: str, threads: int, tf32: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    if provider == "cuda":
        providers = [("CUDAExecutionProvider", {"device_id": 0, "use_tf32": tf32}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), so, providers=providers)


def pool(tok: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask[..., None].astype(np.float32)
    mean = (tok * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
    return mean / np.clip(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12, None)


def worst(acc: dict, key: str, value: float, mode: str = "max") -> None:
    acc[key] = value if key not in acc else (max(acc[key], value) if mode == "max" else min(acc[key], value))


def image_patches(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """The image processor (280-token budget) restated, for the still in `mm-image-and-video`."""
    from torchvision.transforms.v2 import functional as tvF
    from transformers.models.gemma4.image_processing_gemma4 import get_aspect_ratio_preserving_size

    chw = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    h, w = chw.shape[-2:]
    th, tw = get_aspect_ratio_preserving_size(h, w, 16, 2520, 3)
    if (th, tw) != (h, w):
        chw = tvF.resize(chw, [th, tw], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
    x = chw * eg2_video.RESCALE
    ph, pw = th // 16, tw // 16
    pv = x.reshape(3, ph, 16, pw, 16).permute(1, 3, 2, 4, 0).reshape(ph * pw, -1).numpy()
    gx, gy = np.meshgrid(np.arange(pw), np.arange(ph), indexing="xy")
    pos = np.stack([gx, gy], -1).reshape(-1, 2).astype(np.int64)
    return pv, pos, ph * pw // 9


def run_vision(vis, pv: np.ndarray, pos: np.ndarray, count: int, batch: int, pad: bool) -> np.ndarray:
    """Frames [T, P, 768] through the vision graph in batches; returns soft tokens [T, count, 512]."""
    out = []
    rows = eg2_video.MAX_PATCHES if pad else 9 * count
    for s in range(0, pv.shape[0], batch):
        p = np.zeros((min(batch, pv.shape[0] - s), rows, 768), np.float32)
        q = np.full((p.shape[0], rows, 2), -1, np.int64)
        p[:, : 9 * count] = pv[s : s + batch, : 9 * count]
        q[:, : 9 * count] = pos[s : s + batch, : 9 * count]
        soft, _, valid = vis.run(None, {"pixel_values": p, "pixel_position_ids": q})
        if not (valid[:, :count].all() and valid.sum() == p.shape[0] * count):
            raise SystemExit("soft-token mask is not the first `count` slots")
        out.append(soft[:, :count])
    return np.concatenate(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--tensors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provider", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--tf32", type=int, default=0)
    parser.add_argument("--frame-batch", type=int, default=8)
    parser.add_argument("--layouts", default="padded,unpadded,single", help="vision layouts to compare")
    parser.add_argument("--rows", nargs="*", help="only these row ids")
    args = parser.parse_args()
    layouts = args.layouts.split(",")

    t0 = time.time()
    vis = session(args.vision, args.provider, args.threads, args.tf32)
    txt = session(args.text, args.provider, args.threads, args.tf32)
    load_s = time.time() - t0
    rows = [json.loads(x) for x in (args.reference / "video-embeddings.jsonl").read_text().splitlines() if x.strip()]
    if args.rows:
        rows = [r for r in rows if r["id"] in args.rows]

    per_row, agg, finite = {}, {}, True
    mm_by_row: dict[str, np.ndarray] = {}
    emb_by_row: dict[str, np.ndarray] = {}
    timing = {"vision_s": 0.0, "text_s": 0.0, "frames": 0}
    for r in rows:
        z = np.load(args.tensors / f"{r['id']}.npz")
        ids = np.array(r["input_ids"], np.int64)[None]
        mm = np.zeros((1, ids.shape[1], 512), np.float32)
        entry: dict = {"kind": r["kind"], "tokens": int(ids.shape[1]), "videos": []}
        video_soft = []
        for k, v in enumerate(r["videos"]):
            frames = torch.from_numpy(z[f"video{k}_frames_uint8"]).permute(0, 3, 1, 2).contiguous()
            pre = eg2_video.preprocess(frames)
            count = pre["count"]
            valid = 9 * count
            pv = pre["pixel_values"].numpy()
            pos = pre["positions"].numpy()
            if sha(pv[:, :valid]) != v["digests"]["pixel_values"] or sha(pos[:, :valid]) != v["digests"]["positions"]:
                raise SystemExit(f"{r['id']}: restated pixels differ from the reference digests")
            ref_soft = z[f"video{k}_soft_tokens"].reshape(len(pv), count, 512)
            ve: dict = {"video": v["video"], "frames": len(pv), "soft_tokens_per_frame": count}
            results = {}
            for layout in layouts:
                t = time.time()
                if layout == "single":
                    s = run_vision(vis, pv, pos, count, 1, pad=True)
                else:
                    s = run_vision(vis, pv, pos, count, args.frame_batch, pad=layout == "padded")
                if layout == layouts[0]:
                    timing["vision_s"] += time.time() - t
                    timing["frames"] += len(pv)
                finite &= bool(np.isfinite(s).all())
                results[layout] = s
                d = float(np.abs(s - ref_soft).max())
                ve[f"soft_tokens_{layout}_max_abs"] = d
                ve[f"soft_tokens_{layout}_row_rel"] = d / float(np.abs(ref_soft).max())
                worst(agg, f"soft_tokens_{layout}_row_rel", ve[f"soft_tokens_{layout}_row_rel"])
            for layout in layouts[1:]:
                d = float(np.abs(results[layout] - results[layouts[0]]).max())
                ve[f"{layout}_vs_{layouts[0]}_max_abs"] = d
                worst(agg, f"soft_tokens_{layout}_vs_{layouts[0]}_max_abs", d)
            video_soft.append(results[layouts[0]].reshape(-1, 512))
            entry["videos"].append(ve)
        image_soft = None
        if r["image"] is not None:
            pv_i, pos_i, n_i = image_patches(z["image0_rgb"])
            pv_p = np.zeros((1, 2520, 768), np.float32)
            pos_p = np.full((1, 2520, 2), -1, np.int64)
            pv_p[0, : 9 * n_i], pos_p[0, : 9 * n_i] = pv_i, pos_i
            soft, _, _ = vis.run(None, {"pixel_values": pv_p, "pixel_position_ids": pos_p})
            image_soft = soft[0, :n_i]
        vmask = ids[0] == VIDEO_TOKEN
        imask = ids[0] == IMAGE_TOKEN
        if video_soft:
            mm[0, vmask] = np.concatenate(video_soft)
        if image_soft is not None:
            mm[0, imask] = image_soft
        mmask = (vmask | imask)[None]
        mm_by_row[r["id"]] = mm
        att = np.ones_like(ids)
        t = time.time()
        _, tok = txt.run(
            None, {"input_ids": ids, "attention_mask": att, "multimodal_embeds": mm, "multimodal_mask": mmask}
        )
        timing["text_s"] += time.time() - t
        finite &= bool(np.isfinite(tok).all())
        ref_tok = z["token_embeddings"]
        emb = pool(tok, att)[0]
        emb_by_row[r["id"]] = emb
        ref = np.array(r["embedding"], np.float32)
        d = float(np.abs(emb - ref).max())
        cos = float(np.dot(emb.astype(np.float64), ref) / np.linalg.norm(emb) / np.linalg.norm(ref))
        entry.update(
            {
                "token_embeddings_row_rel": float(np.abs(tok[0] - ref_tok).max() / np.abs(ref_tok).max()),
                "embedding_max_abs": d,
                "cosine": cos,
            }
        )
        per_row[r["id"]] = entry
        worst(agg, "token_embeddings_row_rel", entry["token_embeddings_row_rel"])
        worst(agg, "embedding_max_abs", d)
        worst(agg, "min_cosine", cos, "min")
        print(r["id"], entry["tokens"], f"{d:.2e}", f"{cos:.9f}", flush=True)

    # Text graph: padded batches of rows of different lengths (short rows only; long rows cost memory).
    batch_rows = [r for r in rows if r["tokens"] <= 1300]
    tb = 0.0
    for s in range(0, len(batch_rows), 4):
        chunk = batch_rows[s : s + 4]
        width = max(r["tokens"] for r in chunk)
        ids = np.zeros((len(chunk), width), np.int64)
        att = np.zeros_like(ids)
        mm = np.zeros((len(chunk), width, 512), np.float32)
        mmask = np.zeros((len(chunk), width), bool)
        for b, r in enumerate(chunk):
            n = r["tokens"]
            ids[b, :n] = r["input_ids"]
            att[b, :n] = 1
            mm[b, :n] = mm_by_row[r["id"]][0]
            mmask[b, :n] = (ids[b, :n] == VIDEO_TOKEN) | (ids[b, :n] == IMAGE_TOKEN)
        _, tok = txt.run(
            None, {"input_ids": ids, "attention_mask": att, "multimodal_embeds": mm, "multimodal_mask": mmask}
        )
        finite &= bool(np.isfinite(tok).all())
        embs = pool(tok, att)
        for b, r in enumerate(chunk):
            tb = max(tb, float(np.abs(embs[b] - emb_by_row[r["id"]]).max()))
    agg["text_batch4_vs_single_max_abs"] = tb
    result = {
        "provider": args.provider,
        "provider_options": {"use_tf32": args.tf32} if args.provider == "cuda" else {},
        "threads": args.threads,
        "frame_batch": args.frame_batch,
        "onnxruntime": ort.__version__,
        "vision_graph": args.vision.name,
        "text_graph": args.text.name,
        "session_load_s": round(load_s, 1),
        "timing_single_rows": {k: round(v, 1) for k, v in timing.items()},
        "rows": len(rows),
        "worst": agg,
        "nan_or_inf": not finite,
        "per_row": per_row,
    }
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "per_row"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
