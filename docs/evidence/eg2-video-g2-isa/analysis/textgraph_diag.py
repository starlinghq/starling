"""Run the EmbeddingGemma 2 multimodal text graph on the reference soft tokens (Python ORT) and compare with the
reference token embeddings: isolates the text graph from decoding, preprocessing and the vision graph.

usage: textgraph_diag.py <text-mm.onnx> <video-embeddings.jsonl> <refs dir> <out dir> [--rows a,b] [--threads n]
       [--opt all|extended|basic|disable] [--dump-cast-free]
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort


def sidecar(stage_dir, name, rid):
    meta = json.load(open(os.path.join(stage_dir, "dotnet.results.json"), encoding="utf-8"))["tensors"][name]
    data = np.fromfile(os.path.join(stage_dir, meta["file"]), dtype="<f4")
    off = 0
    parts = []
    for c in meta["cases"]:
        n = int(np.prod(c["shape"]))
        if c["id"] == rid or c["id"].startswith(rid + "/"):
            parts.append(data[off : off + n].reshape(c["shape"]))
        off += n
    assert off == data.size, (off, data.size)
    assert parts, (name, rid)
    return np.concatenate(parts, axis=0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("jsonl")
    p.add_argument("refs")
    p.add_argument("out")
    p.add_argument("--rows", default="commons-video-04")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--opt", default="all")
    p.add_argument("--disable", default="", help="comma-separated optimizer names to disable")
    p.add_argument("--save", action="store_true")
    p.add_argument("--soft-from", default="", help="stage results dir: use its soft_tokens instead of the reference's")
    p.add_argument("--compare-te", default="", help="stage results dir: also compare with its token_embeddings")
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rows = {}
    for line in open(a.jsonl, encoding="utf-8"):
        r = json.loads(line)
        rows[r["id"]] = r

    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = False
    so.graph_optimization_level = {
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    }[a.opt]
    disabled = [x for x in a.disable.split(",") if x]
    t = time.time()
    sess = ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"], disabled_optimizers=disabled or None)
    print(f"session {time.time() - t:.1f} s; ort {ort.__version__}; opt {a.opt}; disabled {disabled}")
    os.makedirs(a.out, exist_ok=True)
    summary = {}
    for rid in a.rows.split(","):
        r = rows[rid]
        ids = np.array(r["input_ids"], dtype=np.int64)
        ref = np.load(os.path.join(a.refs, rid + ".npz"))
        if a.soft_from:
            soft = sidecar(a.soft_from, "soft_tokens", rid)
        else:
            soft = np.concatenate([ref[k] for k in sorted(ref.files) if k.endswith("_soft_tokens")], axis=0)
        if a.noise:
            rng = np.random.default_rng(a.seed)
            soft = (soft + rng.normal(0, a.noise * np.abs(soft).max(), soft.shape)).astype(np.float32)
        vals, counts = np.unique(ids, return_counts=True)
        vid = int(vals[np.argmax(counts)])
        mask = ids == vid
        assert mask.sum() == soft.shape[0], (mask.sum(), soft.shape)
        emb = np.zeros((1, ids.size, 512), dtype=np.float32)
        emb[0, mask] = soft
        feeds = {
            "input_ids": ids[None],
            "attention_mask": np.ones((1, ids.size), dtype=np.int64),
            "multimodal_embeds": emb,
            "multimodal_mask": mask[None],
        }
        t = time.time()
        out = sess.run(["token_embeddings"], feeds)[0][0]
        dt = time.time() - t
        refte = ref["token_embeddings"]
        d = np.abs(out.astype(np.float64) - refte)
        rel = float(d.max() / np.abs(refte).max())
        worst = np.unravel_index(int(d.argmax()), d.shape)
        per_tok = d.max(axis=1) / np.abs(refte).max()
        summary[rid] = {
            "row_relative": rel,
            "max_abs": float(d.max()),
            "ref_max_abs": float(np.abs(refte).max()),
            "worst_token": int(worst[0]),
            "worst_dim": int(worst[1]),
            "tokens_over_1e-3": int((per_tok > 1e-3).sum()),
            "mean_abs": float(d.mean()),
            "out_sha256": hashlib.sha256(out.tobytes()).hexdigest()[:16],
            "seconds": round(dt, 1),
        }
        if a.compare_te:
            te = sidecar(a.compare_te, "token_embeddings", rid)
            summary[rid]["vs_stage_te_max_abs"] = float(np.abs(te - out).max())
            dd = np.abs(te.astype(np.float64) - refte)
            summary[rid]["stage_te_row_relative"] = float(dd.max() / np.abs(refte).max())
        print(rid, json.dumps(summary[rid]))
        if a.save:
            np.save(os.path.join(a.out, rid + ".token_embeddings.npy"), out)
    json.dump(summary, open(os.path.join(a.out, "textgraph.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
