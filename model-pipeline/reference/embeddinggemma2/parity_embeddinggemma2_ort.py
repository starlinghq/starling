"""ONNX Runtime parity for the exported EmbeddingGemma 2 text graph (admission spike).

Reference: the sentence-transformers embeddings in `embeddings.jsonl` (run_embeddinggemma2.py,
PyTorch fp32 CPU). Hidden states and token embeddings are recomputed with `eg2_text.TextPath`
(which matches transformers' `EmbeddingGemma2TextModel` bit for bit), and its pooled output is
checked against the stored embeddings first. Candidates pool the graph's `token_embeddings` in
NumPy float32 (mean over the mask, L2 normalization). Every run uses the fixture `input_ids`.

Checks: per-row hidden state (512) and token embeddings (768) over valid positions, final
embedding max/mean/p99 abs and min cosine (float64); padded batches against single rows, for the
corpus rows and for synthetic lengths around the 512-token sliding window (512 to 2,049 tokens);
NaN/Inf over every output element including padded positions; with `--provider cuda`, node
placement from an ORT profile. `--dump` writes the ORT embeddings for the C# comparison.

Usage:
    python parity_embeddinggemma2_ort.py --model-dir <snapshot> --graph <onnx> --reference <dir> --out <json>
        [--provider cpu|cuda] [--threads N] [--cuda-tf32 0|1] [--dump <npz>]
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from eg2_text import TextPath, load_text_model, pool

from model_pipeline import fixtures

SYNTHETIC_LENGTHS = [511, 512, 513, 514, 515, 600, 1024, 1025, 1026, 1027, 1100, 1537, 2049]
PAD_ID = 0


def stats(a: np.ndarray) -> dict:
    a = np.abs(a).ravel()
    return {"max_abs": float(a.max()), "mean_abs": float(a.mean()), "p99_abs": float(np.quantile(a, 0.99))}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def pad(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    w = max(len(r) for r in rows)
    ids = np.array([r + [PAD_ID] * (w - len(r)) for r in rows], dtype=np.int64)
    mask = np.array([[1] * len(r) + [0] * (w - len(r)) for r in rows], dtype=np.int64)
    return ids, mask


def session(graph: Path, provider: str, threads: int, tf32: int, profile: bool) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    if profile:
        so.enable_profiling = True
        so.profile_file_prefix = str(Path(tempfile.mkdtemp()) / "eg2")
    if provider == "cuda":
        providers = [("CUDAExecutionProvider", {"device_id": 0, "use_tf32": tf32}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(graph), so, providers=providers)


def placement(sess: ort.InferenceSession) -> dict:
    path = sess.end_profiling()
    events = json.loads(Path(path).read_text())
    nodes: dict[str, str] = {}
    for ev in events:
        args = ev.get("args", {})
        if ev.get("cat") == "Node" and "provider" in args:
            nodes[ev["name"].removesuffix("_kernel_time")] = args["provider"]
    counts: dict[str, int] = {}
    for p in nodes.values():
        counts[p] = counts.get(p, 0) + 1
    cpu_ops: dict[str, int] = {}
    for ev in events:
        args = ev.get("args", {})
        if (
            ev.get("cat") == "Node"
            and args.get("provider") == "CPUExecutionProvider"
            and ev["name"].endswith("_kernel_time")
        ):
            cpu_ops[args.get("op_name", "?")] = cpu_ops.get(args.get("op_name", "?"), 0) + 1
    return {"nodes_by_provider": counts, "cpu_kernel_calls_all_profiled_runs": cpu_ops}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provider", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--cuda-tf32", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--dump", type=Path)
    args = parser.parse_args()

    torch.manual_seed(0)
    rows = fixtures.read_jsonl(args.reference / "embeddings.jsonl")
    tp = TextPath(load_text_model(args.model_dir)).eval()
    sess = session(args.graph, args.provider, args.threads, args.cuda_tf32, profile=args.provider == "cuda")

    per_row, hid_d, tok_d, emb_d, tok_rel = [], [], [], [], []
    ref_pool_check = []
    ort_embs, nonfinite = [], []
    t_ort = 0.0
    first = True
    for r in rows:
        ids, mask = pad([r["input_ids"]])
        with torch.inference_mode():
            th, te = (t.numpy() for t in tp(torch.from_numpy(ids), torch.from_numpy(mask)))
        ref = np.array(r["embedding"], dtype=np.float32)
        ref_pool_check.append(float(np.abs(pool(te, mask)[0] - ref).max()))
        t0 = time.perf_counter()
        oh, oe = sess.run(None, {"input_ids": ids, "attention_mask": mask})
        t_ort += time.perf_counter() - t0
        if first and args.provider == "cuda":
            first = False
        emb = pool(oe, mask)[0]
        ort_embs.append(emb)
        if not (np.isfinite(oh).all() and np.isfinite(oe).all()):
            nonfinite.append(f"{r['id']}/{r['prompt']}")
        hid_d.append(np.abs(oh - th).ravel())
        tok_d.append(np.abs(oe - te).ravel())
        tok_rel.append(float(np.abs(oe - te).max() / max(1.0, float(np.abs(te).max()))))
        emb_d.append(np.abs(emb - ref))
        per_row.append(
            {
                "id": r["id"],
                "prompt": r["prompt"],
                "tokens": len(r["input_ids"]),
                "hidden_max_abs": float(hid_d[-1].max()),
                "token_embeddings_max_abs": float(tok_d[-1].max()),
                "token_embeddings_ref_max_abs_value": float(np.abs(te).max()),
                "embedding_max_abs": float(emb_d[-1].max()),
                "cosine": cosine(emb, ref),
            }
        )
    place = placement(sess) if args.provider == "cuda" else None

    # Padded batches of corpus rows (file order) against ORT single rows and against the reference.
    singles = np.stack(ort_embs)
    refs = np.stack([np.array(r["embedding"], dtype=np.float32) for r in rows])
    batched = []
    for s in range(0, len(rows), args.batch):
        ids, mask = pad([r["input_ids"] for r in rows[s : s + args.batch]])
        oh, oe = sess.run(None, {"input_ids": ids, "attention_mask": mask})
        if not (np.isfinite(oh).all() and np.isfinite(oe).all()):
            nonfinite.append(f"batch@{s}")
        batched.append(pool(oe, mask))
    batched = np.concatenate(batched)

    # Synthetic lengths around the sliding window: prefixes of the longest corpus row, each padded
    # in one batch with a 12-token row, compared with its own single run and with PyTorch.
    longest = max(rows, key=lambda r: len(r["input_ids"]))["input_ids"]
    short = next(r for r in rows if r["id"] == "ascii-sentence" and r["prompt"] == "none")["input_ids"]
    synth = []
    for n in SYNTHETIC_LENGTHS:
        seq = longest[: n - 1] + [1]
        ids1, mask1 = pad([seq])
        ids2, mask2 = pad([seq, short])
        _, s_e = sess.run(None, {"input_ids": ids1, "attention_mask": mask1})
        b_h, b_e = sess.run(None, {"input_ids": ids2, "attention_mask": mask2})
        _, sh_e = sess.run(None, {"input_ids": pad([short])[0], "attention_mask": pad([short])[1]})
        with torch.inference_mode():
            _, t_e = (t.numpy() for t in tp(torch.from_numpy(ids1), torch.from_numpy(mask1)))
            _, tb_e = (t.numpy() for t in tp(torch.from_numpy(ids2), torch.from_numpy(mask2)))
        finite = bool(np.isfinite(b_h).all() and np.isfinite(b_e).all())
        if not finite:
            nonfinite.append(f"synthetic-{n}")
        e_single, e_batch, e_torch = pool(s_e, mask1)[0], pool(b_e, mask2), pool(t_e, mask1)[0]
        synth.append(
            {
                "tokens": n,
                "ort_batch_vs_ort_single_max_abs": float(np.abs(e_batch[0] - e_single).max()),
                "ort_batch_short_row_vs_single_max_abs": float(
                    np.abs(e_batch[1] - pool(sh_e, pad([short])[1])[0]).max()
                ),
                "ort_single_vs_torch_max_abs": float(np.abs(e_single - e_torch).max()),
                "ort_single_vs_torch_cosine": cosine(e_single, e_torch),
                "torch_batch_vs_torch_single_max_abs": float(np.abs(pool(tb_e, mask2)[0] - e_torch).max()),
                "finite_including_padding": finite,
            }
        )

    all_h, all_t, all_e = np.concatenate(hid_d), np.concatenate(tok_d), np.concatenate(emb_d)
    result = {
        "graph": args.graph.name,
        "graph_sha256": fixtures.sha256_file(args.graph),
        "provider": args.provider,
        "provider_options": {"use_tf32": args.cuda_tf32} if args.provider == "cuda" else None,
        "providers_in_session": sess.get_providers(),
        "intra_op_num_threads": args.threads,
        "rows": len(rows),
        "reference": "embeddings.jsonl (sentence-transformers, PyTorch fp32 CPU); hidden states and token embeddings "
        "recomputed with eg2_text.TextPath on the same ids",
        "reference_recompute_pooled_vs_embeddings_jsonl_max_abs": max(ref_pool_check),
        "last_hidden_state_512": stats(all_h),
        "token_embeddings_768": stats(all_t),
        "token_embeddings_max_abs_relative_to_max_1_ref": max(tok_rel),
        "embedding": {**stats(all_e), "min_cosine": min(p["cosine"] for p in per_row)},
        "padded_batches": {
            "batch_size": args.batch,
            "ort_batch_vs_ort_single": stats(batched - singles),
            "ort_batch_vs_reference": stats(batched - refs),
        },
        "synthetic_window_lengths": synth,
        "nan_or_inf": nonfinite,
        "placement": place,
        "ort_seconds_single_rows": round(t_ort, 2),
        "worst_rows_embedding": sorted(per_row, key=lambda p: -p["embedding_max_abs"])[:5],
        "per_row": per_row,
    }
    fixtures.write_json(args.out, result)
    if args.dump:
        np.savez(args.dump, embeddings=singles, batched=batched)
    brief = {
        k: result[k]
        for k in ("last_hidden_state_512", "token_embeddings_768", "embedding", "padded_batches", "nan_or_inf")
    }
    brief["placement"] = place
    brief["synthetic_max"] = {
        k: max(s[k] for s in synth)
        for k in (
            "ort_batch_vs_ort_single_max_abs",
            "ort_single_vs_torch_max_abs",
            "torch_batch_vs_torch_single_max_abs",
        )
    }
    print(json.dumps(brief, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
