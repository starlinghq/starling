"""Kev-0.8B spike, step 2 fixture subset (P-027; model-pipeline/admissions/kev.md "Spike plan").

Runs upstream Kev at the pinned code revision in fp32 (CPU by default, `--device cuda` for the CUDA kernels Kev binds),
through its own DecisionModel internals:

- `forward` layout: each question is one causal row, state + branch, recomputing the state (what kev.benchmark
  scores and what every published number used).
- `probs` layout: the state runs once and each branch continues from a copy of its cache (the serving path).

For each record it stores token ids, positions, readout offsets, fp32 hidden states at every <decide> and </opt>
index, logits and probabilities for both layouts. Output is an .npz per record plus an index JSON. These are spike
evidence under .models/, not protected fixtures: hidden states of 64 records are too large for Git.

Usage (Kev environment, HF_HUB_OFFLINE=1 with the base in the HF cache, TORCH_FORCE_WEIGHTS_ONLY_LOAD=1):
    python kev_fixtures.py --checkpoint <kev-0.8b dir> --suite <kev>/evals/v7/decision-v7 --count 64 --out <dir>
    (from the Kev checkout, so suite paths resolve)
    python kev_fixtures.py --checkpoint <dir> --long-docs <kev>/evals/documents-v1 --out <dir>
    python kev_fixtures.py --checkpoint <dir> --long-docs <kev>/evals/documents-v1 --long-tokens 32768 \
        --layouts probs --device cuda --out <dir>   (one synthetic state of that many tokens; GPU memory recorded)

On CUDA the script applies LocalPredictor's fp32-exact switches (no TF32, SDPA math kernel) and, for rows longer than
kev.model.ROW_PASS_TOKENS, kev.predictors.long_row_kernels (fp32 memory-efficient SDPA), as kev.benchmark does.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from kev.checkpoint import Checkpoint, LoadOptions
from kev.data import materialize
from kev.model import (
    MAX_TRAIN_STATE_8K,
    ROW_PASS_TOKENS,
    SERVE_MAX_BRANCH,
    SERVE_MAX_BRANCH_8K,
    SERVE_MAX_STATE,
    SERVE_MAX_STATE_8K,
    rows_of,
)
from kev.suite import CONTEXT, load_split, read_manifest


def sync(dev) -> None:
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize()


def kernels(model, n_tokens: int):
    """kev.benchmark's choice: long rows on CUDA run under SDPA's memory-efficient kernels (fp32 kept)."""
    if str(model.device).startswith("cuda") and n_tokens > ROW_PASS_TOKENS:
        from kev.predictors import long_row_kernels

        return long_row_kernels()
    return contextlib.nullcontext()


def readouts(model, enc, layouts=("forward", "probs")):
    """Both layouts for one encoded record. Returns dict of numpy arrays and per-question lists. A layout left out of
    `layouts` is filled with the other one's arrays and marked in out["computed"]."""
    state_ids, state_pos, rows = rows_of(enc)
    dev = model.device
    out = {"forward": {"hidden": [], "logits": [], "probs": []}, "probs": {"hidden": [], "logits": [], "probs": []}}
    out["computed"] = list(layouts)
    longest = len(state_ids) + max(len(r["ids"]) for r in rows)
    sync(dev)
    t0 = time.perf_counter()
    full_rows = [(state_ids + r["ids"], state_pos + r["pos"]) for r in rows] if "forward" in layouts else []
    with kernels(model, longest):
        hs = model._rows_hidden(full_rows) if full_rows else []
    for h, r in zip(hs, rows, strict=bool(hs)):
        d = len(state_ids) + r["decide"]
        opts = [len(state_ids) + o for o in r["opts"]]
        sel = h[torch.tensor([d, *opts], device=dev)]
        z = model.head(sel[0], sel[1:])
        out["forward"]["hidden"].append(sel.cpu().numpy())
        out["forward"]["logits"].append(z.cpu().numpy())
        out["forward"]["probs"].append(F.softmax(z, -1).cpu().numpy())
    sync(dev)
    t1 = time.perf_counter()
    hs = []
    if "probs" in layouts:
        with kernels(model, longest):
            n_state, cache, _ = model.prefix(enc)
            sync(dev)
            out["state_seconds"] = time.perf_counter() - t1
            hs = model._rows_hidden([(r["ids"], r["pos"]) for r in rows], cache=cache, prefix_len=n_state)
    for h, r in zip(hs, rows, strict=bool(hs)):
        sel = h[torch.tensor([r["decide"], *r["opts"]], device=dev)]
        z = model.head(sel[0], sel[1:])
        out["probs"]["hidden"].append(sel.cpu().numpy())
        out["probs"]["logits"].append(z.cpu().numpy())
        out["probs"]["probs"].append(F.softmax(z, -1).cpu().numpy())
    sync(dev)
    t2 = time.perf_counter()
    for a, b in (("forward", "probs"), ("probs", "forward")):
        if a not in layouts:
            out[a] = out[b]
    out["seconds"] = {"forward": t1 - t0, "probs": t2 - t1}
    return out


def save(path: Path, enc, rec_id: str, res: dict, extra: dict | None = None) -> dict:
    state_ids, state_pos, rows = rows_of(enc)
    arrays = {"state_ids": np.array(state_ids, dtype=np.int64), "state_pos": np.array(state_pos, dtype=np.int64)}
    for q, r in enumerate(rows):
        arrays[f"q{q}_ids"] = np.array(r["ids"], dtype=np.int64)
        arrays[f"q{q}_pos"] = np.array(r["pos"], dtype=np.int64)
        arrays[f"q{q}_readout"] = np.array([r["decide"], *r["opts"]], dtype=np.int64)  # offsets within the branch
        for layout in ("forward", "probs"):
            for kind in ("hidden", "logits", "probs"):
                arrays[f"q{q}_{layout}_{kind}"] = res[layout][kind][q].astype(np.float32)
    np.savez_compressed(path, **arrays)
    fwd = np.concatenate(res["forward"]["probs"])
    cached = np.concatenate(res["probs"]["probs"])
    return {
        "id": rec_id,
        "file": path.name,
        "state_tokens": len(state_ids),
        "questions": len(rows),
        "branch_tokens": [len(r["ids"]) for r in rows],
        "options": [len(r["opts"]) for r in rows],
        "forward_vs_probs_max_abs": float(np.abs(fwd - cached).max()),
        "seconds": res["seconds"],
        "layouts_computed": res["computed"],
        **({"state_seconds": res["state_seconds"]} if "state_seconds" in res else {}),
        **(extra or {}),
    }


def long_doc_records(tok, suite: Path, long_tokens: int = 0) -> list[tuple[str, dict, dict]]:
    """Two development records truncated to about 2,048 state tokens, and two synthetic states of 7,552 tokens
    built by concatenating development documents (no development state is that long; the longest is about 5,000).
    With long_tokens, instead one synthetic state of that many tokens (documents joined and repeated as needed) with
    the questions of the longest record."""
    recs = [json.loads(line) for line in (suite / "development.jsonl").read_text(encoding="utf-8").split("\n") if line]

    def ntok(text):
        return len(tok(text, add_special_tokens=False).input_ids)

    def cut(text, n):
        ids = tok(text, add_special_tokens=False).input_ids[:n]
        return tok.decode(ids)

    sized = sorted(((ntok(r["state"]), i) for i, r in enumerate(recs) if isinstance(r["state"], str)), reverse=True)
    out = []
    if long_tokens:
        parts, used, total = [], [], 0
        while total < long_tokens:
            for n, j in sized:
                parts.append(recs[j]["state"])
                used.append(j)
                total += n + 2
                if total >= long_tokens:
                    break
        i = sized[0][1]
        r = dict(recs[i])
        r["state"] = cut("\n\n".join(parts), long_tokens - 1)
        source = f"synthetic: {len(used)} documents-v1 development states joined, cut to {long_tokens - 1} tokens"
        return [(f"documents-v1-dev-{i}-{long_tokens}-synthetic", r, {"source": f"{source}; questions of {i}"})]
    for _, i in sized[:2]:
        r = dict(recs[i])
        r["state"] = cut(r["state"], 2047)
        out.append(
            (f"documents-v1-dev-{i}-2k", r, {"source": f"documents-v1 development line {i}, cut to 2047 tokens"})
        )
    for k, (_, i) in enumerate(sized[2:4]):
        parts, used = [], []
        for _, j in sized[k * 7 :]:
            parts.append(recs[j]["state"])
            used.append(j)
            if ntok("\n\n".join(parts)) >= MAX_TRAIN_STATE_8K:
                break
        r = dict(recs[i])
        # The 8k serving context caps the whole row (state + branch) at 8,192 tokens, so the state uses upstream's
        # 8k training state size and leaves the rest for the question.
        r["state"] = cut("\n\n".join(parts), MAX_TRAIN_STATE_8K - 1)
        out.append(
            (
                f"documents-v1-dev-{i}-7k5-synthetic",
                r,
                {
                    "source": f"synthetic: documents-v1 development states {used} joined, "
                    f"cut to {MAX_TRAIN_STATE_8K - 1} tokens; "
                    f"questions of {i}"
                },
            )
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--suite")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--long-docs")
    parser.add_argument("--long-tokens", type=int, default=0, help="one synthetic state of this many tokens")
    parser.add_argument("--layouts", default="forward,probs", help="comma list: forward, probs")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    layouts = tuple(args.layouts.split(","))

    torch.manual_seed(0)
    if args.device == "cpu":  # on CUDA kev.benchmark runs without it (cuBLAS and the fla kernels would refuse)
        torch.use_deterministic_algorithms(True)
    if os.environ.get("TORCH_FORCE_WEIGHTS_ONLY_LOAD") != "1":
        raise SystemExit(
            "set TORCH_FORCE_WEIGHTS_ONLY_LOAD=1: head.pt is a pickle and must load with weights_only=True"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda":
        # LocalPredictor's fp32-exact switches (kev/predictors.py)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    tok, model = Checkpoint(args.checkpoint).load(args.device, LoadOptions())
    index = []
    with torch.no_grad():
        if args.suite:
            # The suite's admitted context, as kev.benchmark uses it.
            ctx = read_manifest(args.suite).get("context", CONTEXT)
            records = load_split(args.suite, "development")
            # Evenly spaced over the partition so every source appears; the suite is grouped by source.
            picks = [round(i * (len(records) - 1) / max(args.count - 1, 1)) for i in range(args.count)]
            for n in picks:
                rec = records[n]
                enc = model.encode(
                    tok, materialize(rec), max_state=ctx["max_state"], max_branch=ctx["max_branch"], strict=True
                )
                res = readouts(model, enc, layouts)
                rid = rec["_meta"]["id"]
                index.append(save(args.out / f"r{n:04d}.npz", enc, rid, res, {"suite_index": n}))
                print(n, rid, index[-1]["state_tokens"], index[-1]["forward_vs_probs_max_abs"], flush=True)
        if args.long_docs:
            long_ctx = (
                (SERVE_MAX_STATE, SERVE_MAX_BRANCH) if args.long_tokens else (SERVE_MAX_STATE_8K, SERVE_MAX_BRANCH_8K)
            )
            for rid, rec, extra in long_doc_records(tok, Path(args.long_docs), args.long_tokens):
                enc = model.encode(tok, materialize(rec), max_state=long_ctx[0], max_branch=long_ctx[1], strict=True)
                if args.device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                res = readouts(model, enc, layouts)
                extra["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                if args.device == "cuda":
                    extra["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
                    extra["cuda_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
                extra["finite"] = bool(
                    all(np.isfinite(a).all() for lay in ("forward", "probs") for a in res[lay]["hidden"])
                )
                index.append(save(args.out / f"{rid}.npz", enc, rid, res, extra))
                print(rid, index[-1]["state_tokens"], index[-1]["seconds"], flush=True)
    name = "index-long.json" if args.long_docs and not args.suite else "index.json"
    if args.long_tokens:
        name = f"index-long-{args.long_tokens}.json"
    (args.out / name).write_text(json.dumps(index, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
