"""ONNX Runtime parity for the EmbeddingGemma 2 image path (image-path spike).

Feeds the reference's own preprocessed tensors (`run_embeddinggemma2_image.py --tensors`) to the two graphs from
`export_embeddinggemma2_vision_onnx.py` and compares every stage with the PyTorch fp32 CPU reference:

- vision graph, per image: `pooled` (768, before `embed_vision`) and `soft_tokens` (512), padded to 2,520 patches as
  upstream and unpadded (`9 * count` patches);
- multimodal text graph, per row: `token_embeddings` (768) with the soft tokens scattered at the `<|image|>` ids
  (relative to `max(1, |ref|)`, as for the text path), then mean pooling and L2 normalization in NumPy float32
  against the reference embedding (max abs, cosine);
- batch independence: all images through the vision graph in batches (`--vision-batch`, default 8; each padded to
  2,520) against single
  images; padded batches of rows of different lengths through the text graph against single rows;
- NaN/Inf in every output, padded positions included.

Usage:
    python parity_embeddinggemma2_image_ort.py --vision <graph> --text <graph> --reference <dir> --tensors <dir>
        --out <json> [--provider cpu|cuda] [--threads N] [--tf32 0|1]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

IMAGE_TOKEN = 258880
MAX_PATCHES = 2520


def session(path: Path, provider: str, threads: int, tf32: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    if provider == "cuda":
        providers = [("CUDAExecutionProvider", {"device_id": 0, "use_tf32": tf32}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), so, providers=providers)


def pad_patches(pv: np.ndarray, pos: np.ndarray, rows: int) -> tuple[np.ndarray, np.ndarray]:
    n = pv.shape[0]
    pv_p = np.zeros((rows, pv.shape[1]), np.float32)
    pos_p = np.full((rows, 2), -1, np.int64)
    pv_p[:n], pos_p[:n] = pv, pos
    return pv_p, pos_p


def pool(tok: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask[..., None].astype(np.float32)
    mean = (tok * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
    return mean / np.clip(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12, None)


def worst(acc: dict, key: str, value: float, mode: str = "max") -> None:
    if key not in acc:
        acc[key] = value
    else:
        acc[key] = max(acc[key], value) if mode == "max" else min(acc[key], value)


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
    parser.add_argument(
        "--vision-batch", type=int, default=8, help="images per vision batch; 0 or 1 skips the check (GPU memory)"
    )
    args = parser.parse_args()

    t0 = time.time()
    vis = session(args.vision, args.provider, args.threads, args.tf32)
    txt = session(args.text, args.provider, args.threads, args.tf32)
    load_s = time.time() - t0
    rows = [json.loads(x) for x in (args.reference / "image-embeddings.jsonl").read_text().splitlines() if x.strip()]

    per_row, agg = {}, {}
    finite = True
    soft_by_row: dict[str, list[np.ndarray]] = {}
    emb_by_row: dict[str, np.ndarray] = {}
    timing = {"vision_s": 0.0, "text_s": 0.0}
    for r in rows:
        z = np.load(args.tensors / f"{r['id']}.npz")
        entry: dict = {}
        softs = []
        k = 0
        while f"image{k}_pixel_values" in z:
            pv, pos = z[f"image{k}_pixel_values"], z[f"image{k}_positions"]
            n = pv.shape[0] // 9
            pv_p, pos_p = pad_patches(pv, pos, MAX_PATCHES)
            t = time.time()
            soft, pooled, valid = vis.run(None, {"pixel_values": pv_p[None], "pixel_position_ids": pos_p[None]})
            timing["vision_s"] += time.time() - t
            soft_u, pooled_u, valid_u = vis.run(None, {"pixel_values": pv[None], "pixel_position_ids": pos[None]})
            finite &= bool(np.isfinite(soft).all() and np.isfinite(pooled).all())
            if not (valid[0, :n].all() and valid[0].sum() == n and valid_u[0].all()):
                raise SystemExit(f"{r['id']}: soft-token mask is not the first {n} slots")
            ref_soft, ref_pooled = z[f"image{k}_soft_tokens"], z[f"image{k}_pooled"]
            d_soft = float(np.abs(soft[0, :n] - ref_soft).max())
            d_pooled = float(np.abs(pooled[0, :n] - ref_pooled).max())
            d_pooled_rel = float((np.abs(pooled[0, :n] - ref_pooled) / np.maximum(1, np.abs(ref_pooled))).max())
            d_pooled_row = d_pooled / float(np.abs(ref_pooled).max())
            d_soft_row = d_soft / float(np.abs(ref_soft).max())
            d_unpad = float(np.abs(soft_u[0] - soft[0, :n]).max())
            entry.setdefault("images", []).append(
                {
                    "soft_tokens_max_abs": d_soft,
                    "soft_tokens_row_rel": d_soft_row,
                    "pooled_max_abs": d_pooled,
                    "pooled_rel_max": d_pooled_rel,
                    "pooled_row_rel": d_pooled_row,
                    "unpadded_vs_padded_soft_max_abs": d_unpad,
                    "pooled_ref_max_abs_value": float(np.abs(ref_pooled).max()),
                }
            )
            worst(agg, "soft_tokens_max_abs", d_soft)
            worst(agg, "pooled_max_abs", d_pooled)
            worst(agg, "pooled_rel_max", d_pooled_rel)
            worst(agg, "pooled_row_rel", d_pooled_row)
            worst(agg, "soft_tokens_row_rel", d_soft_row)
            worst(agg, "unpadded_vs_padded_soft_max_abs", d_unpad)
            softs.append(soft[0, :n])
            k += 1
        soft_by_row[r["id"]] = softs
        ids = np.array(r["input_ids"], np.int64)[None]
        mm = np.zeros((1, ids.shape[1], 512), np.float32)
        mm_mask = ids == IMAGE_TOKEN
        if softs:
            mm[0, mm_mask[0]] = np.concatenate(softs)
        att = np.ones_like(ids)
        t = time.time()
        _, tok = txt.run(
            None, {"input_ids": ids, "attention_mask": att, "multimodal_embeds": mm, "multimodal_mask": mm_mask}
        )
        timing["text_s"] += time.time() - t
        finite &= bool(np.isfinite(tok).all())
        ref_tok = z["token_embeddings"]
        d_tok_rel = float((np.abs(tok[0] - ref_tok) / np.maximum(1, np.abs(ref_tok))).max())
        d_tok_row = float(np.abs(tok[0] - ref_tok).max() / np.abs(ref_tok).max())
        emb = pool(tok, att)[0]
        emb_by_row[r["id"]] = emb
        ref = np.array(r["embedding"], np.float32)
        d = float(np.abs(emb - ref).max())
        cos = float(np.dot(emb.astype(np.float64), ref) / np.linalg.norm(emb) / np.linalg.norm(ref))
        entry.update(
            {
                "kind": r["kind"],
                "tokens": int(ids.shape[1]),
                "token_embeddings_rel_max": d_tok_rel,
                "token_embeddings_row_rel": d_tok_row,
                "token_embeddings_max_abs": float(np.abs(tok[0] - ref_tok).max()),
                "embedding_max_abs": d,
                "cosine": cos,
            }
        )
        per_row[r["id"]] = entry
        worst(agg, "token_embeddings_rel_max", d_tok_rel)
        worst(agg, "token_embeddings_row_rel", d_tok_row)
        worst(agg, "embedding_max_abs", d)
        worst(agg, "min_cosine", cos, "min")
        if r["kind"] != "text":
            worst(agg, "image_rows_embedding_max_abs", d)

    # Batch independence: vision in batches (images of different sizes, each padded to 2,520).
    image_rows = [r for r in rows if r["kind"] == "image"]
    vb = 0.0
    vbs = args.vision_batch
    for s in range(0, len(image_rows) if vbs > 1 else 0, max(vbs, 1)):
        chunk = image_rows[s : s + vbs]
        pvs, poss, ns = [], [], []
        for r in chunk:
            z = np.load(args.tensors / f"{r['id']}.npz")
            pv_p, pos_p = pad_patches(z["image0_pixel_values"], z["image0_positions"], MAX_PATCHES)
            pvs.append(pv_p)
            poss.append(pos_p)
            ns.append(z["image0_pixel_values"].shape[0] // 9)
        soft, _, _ = vis.run(None, {"pixel_values": np.stack(pvs), "pixel_position_ids": np.stack(poss)})
        finite &= bool(np.isfinite(soft).all())
        for b, r in enumerate(chunk):
            vb = max(vb, float(np.abs(soft[b, : ns[b]] - soft_by_row[r["id"]][0]).max()))
    agg["vision_batch_vs_single_soft_max_abs"] = vb if vbs > 1 else "not run"
    agg["vision_batch_size"] = vbs

    # Text graph: padded batches of rows of different lengths (interleaved, image-only and text rows).
    tb = 0.0
    batch_rows = [r for r in rows if r["kind"] in ("interleaved", "text")] + image_rows[:6]
    for s in range(0, len(batch_rows), 6):
        chunk = batch_rows[s : s + 6]
        width = max(len(r["input_ids"]) for r in chunk)
        ids = np.zeros((len(chunk), width), np.int64)
        att = np.zeros_like(ids)
        mm = np.zeros((len(chunk), width, 512), np.float32)
        for b, r in enumerate(chunk):
            n = len(r["input_ids"])
            ids[b, :n] = r["input_ids"]
            att[b, :n] = 1
            if soft_by_row[r["id"]]:
                mm[b, :n][ids[b, :n] == IMAGE_TOKEN] = np.concatenate(soft_by_row[r["id"]])
        mm_mask = ids == IMAGE_TOKEN
        _, tok = txt.run(
            None, {"input_ids": ids, "attention_mask": att, "multimodal_embeds": mm, "multimodal_mask": mm_mask}
        )
        finite &= bool(np.isfinite(tok).all())
        embs = pool(tok, att)
        for b, r in enumerate(chunk):
            single = np.array(r["embedding"], np.float32)
            d = float(np.abs(embs[b] - single).max())
            tb = max(tb, float(np.abs(embs[b] - emb_by_row[r["id"]]).max()))
            per_row[r["id"]]["batched_embedding_vs_reference_max_abs"] = d
            worst(agg, "text_batch6_vs_reference_max_abs", d)
    agg["text_batch6_vs_single_max_abs"] = tb
    result = {
        "provider": args.provider,
        "provider_options": {"use_tf32": args.tf32} if args.provider == "cuda" else {},
        "threads": args.threads,
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
