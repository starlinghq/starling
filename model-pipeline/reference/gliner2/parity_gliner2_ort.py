"""ONNX Runtime parity for the GLiNER2 export spike (backlog P-014).

Feeds the reference inputs recorded by `run_gliner2_extraction.py` (the `.npz` tensors) to each exported
graph and compares stage by stage with the PyTorch fp32 CPU reference:

    last_hidden_state   max / mean / p99 abs over real tokens
    count_logits        max abs; gate (argmax > 0) equality
    span_logits         max / mean / p99 abs over valid spans (start + width < words)
    scores              max abs of sigmoid over valid spans
    entities            decoded with `gliner2_entity.decode_entities` (the public output contract);
                        entity-set equality, confidence max abs, and every threshold flip with its margin

Graphs: `A` (one graph, ids to logits), `B+H` (backbone graph, then heads graph). Also padded batches
(cases grouped by label count; route A needs one label count per batch) against single runs, NaN/Inf,
repeatability, and per-provider node placement from an ORT profile.

Usage:
    python parity_gliner2_ort.py --onnx-dir <dir> --reference <ref dir> --out <json> \
        [--provider cpu|cuda] [--threads 1,12] [--cuda-tf32 0] [--graphs A-legacy,A-dynamo,B-legacy+H-legacy,...]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from gliner2_entity import decode_entities, sigmoid32

from model_pipeline import fixtures


def stats(diff: np.ndarray) -> dict:
    d = np.abs(diff.astype(np.float64)).reshape(-1)
    if d.size == 0:
        return {"max": 0.0, "mean": 0.0, "p99": 0.0, "n": 0}
    return {"max": float(d.max()), "mean": float(d.mean()), "p99": float(np.percentile(d, 99)), "n": int(d.size)}


def merge(acc: dict | None, s: dict) -> dict:
    """Running max of max/p99 and n-weighted mean."""
    if acc is None:
        return dict(s)
    n = acc["n"] + s["n"]
    return {
        "max": max(acc["max"], s["max"]),
        "p99": max(acc["p99"], s["p99"]),
        "mean": (acc["mean"] * acc["n"] + s["mean"] * s["n"]) / n if n else 0.0,
        "n": n,
    }


def entity_keys(result: dict) -> list:
    return [(k, [(e["text"], e["start"], e["end"]) for e in v]) for k, v in result.get("entities", {}).items()]


def make_session(path: Path, provider: str, threads: int, tf32: int, profile_dir: str | None = None):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profile_dir:
        so.enable_profiling = True
        so.profile_file_prefix = os.path.join(profile_dir, path.stem)
    if provider == "cuda":
        providers = [
            ("CUDAExecutionProvider", {"device_id": 0, "use_tf32": tf32, "arena_extend_strategy": "kSameAsRequested"}),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), sess_options=so, providers=providers)


class Graph:
    """One route: either a single session (A) or backbone + heads sessions (B+H)."""

    def __init__(self, name: str, onnx_dir: Path, provider: str, threads: int, tf32: int, profile_dir=None):
        self.name = name
        parts = name.split("+")
        self.sessions = [
            make_session(onnx_dir / f"gliner2-{p}.onnx", provider, threads, tf32, profile_dir) for p in parts
        ]
        self.split = len(parts) == 2

    def run(self, ids, mask, word_index, schema_index):
        if not self.split:
            count, logits, hidden = self.sessions[0].run(
                None, {"input_ids": ids, "attention_mask": mask, "word_index": word_index, "schema_index": schema_index}
            )
            return hidden, count, logits
        (hidden,) = self.sessions[0].run(None, {"input_ids": ids, "attention_mask": mask})
        b = np.arange(hidden.shape[0])[:, None]
        words = np.ascontiguousarray(hidden[b, word_index])
        schema = np.ascontiguousarray(hidden[b, schema_index])
        count, logits = self.sessions[1].run(None, {"words": words, "schema": schema})
        return hidden, count, logits

    def placement(self) -> list[dict]:
        out = []
        for s in self.sessions:
            path = s.end_profiling()
            events = json.loads(Path(path).read_text())
            nodes: dict[str, dict] = {}
            for e in events:
                if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"):
                    a = e.get("args", {})
                    nodes[e["name"]] = {"provider": a.get("provider"), "op": a.get("op_name")}
            by_provider: dict[str, int] = {}
            cpu_ops: dict[str, int] = {}
            for n in nodes.values():
                by_provider[n["provider"]] = by_provider.get(n["provider"], 0) + 1
                if n["provider"] == "CPUExecutionProvider":
                    cpu_ops[n["op"]] = cpu_ops.get(n["op"], 0) + 1
            out.append(
                {
                    "providers": s.get_providers(),
                    "provider_options": {k: dict(v) for k, v in s.get_provider_options().items()},
                    "nodes_by_provider": by_provider,
                    "cpu_node_ops": dict(sorted(cpu_ops.items())),
                }
            )
        return out


def load_reference(ref_dir: Path) -> list[tuple[dict, dict]]:
    rows = fixtures.read_jsonl(ref_dir / "extraction.jsonl")
    return [(r, dict(np.load(ref_dir / "tensors" / f"{r['id']}.npz"))) for r in rows]


def compare_case(row: dict, ref: dict, hidden, count, logits) -> dict:
    n = len(row["start_map"])
    k = row["span_logits_shape"][2]
    m = len(row["labels"])
    s = int(ref["attention_mask"].sum())
    valid = (np.arange(n)[:, None] + np.arange(k)[None, :]) < n
    ref_logits = ref["span_logits"]
    cand_logits = logits[:m, :n]
    lg = np.stack([cand_logits[i][valid] - ref_logits[i][valid] for i in range(m)])
    sc = np.stack([sigmoid32(cand_logits[i][valid]) - sigmoid32(ref_logits[i][valid]) for i in range(m)])
    decoded = decode_entities(
        count_logits=count,
        span_logits=cand_logits,
        labels=row["labels"],
        text=row["normalized_text"],
        start_map=row["start_map"],
        end_map=row["end_map"],
        threshold=row["threshold"],
    )
    same = entity_keys(decoded) == entity_keys(row["output"])
    flips = []
    if not same:
        thr = np.float32(row["threshold"])
        ref_s = sigmoid32(ref_logits)
        cand_s = sigmoid32(cand_logits)
        for i in range(m):
            for st, w in zip(*np.nonzero(valid), strict=True):
                a, b = ref_s[i, st, w] >= thr, cand_s[i, st, w] >= thr
                if a != b:
                    flips.append(
                        {
                            "label": row["labels"][i],
                            "start": int(st),
                            "width": int(w),
                            "reference_score": float(ref_s[i, st, w]),
                            "candidate_score": float(cand_s[i, st, w]),
                            "threshold": float(thr),
                        }
                    )
    conf = None
    if same:
        conf = 0.0
        for lab, ents in decoded["entities"].items():
            for x, y in zip(ents, row["output"]["entities"][lab], strict=True):
                conf = max(conf, abs(x["confidence"] - y["confidence"]))
    return {
        "id": row["id"],
        "hidden": stats(hidden[:s] - ref["last_hidden_state"][0, :s]),
        "count_logits": stats(count - ref["count_logits"]),
        "gate_equal": bool((np.argmax(count) > 0) == (np.argmax(ref["count_logits"]) > 0)),
        "span_logits": stats(lg),
        "scores": stats(sc),
        "entities_equal": same,
        "entity_sets_equal": [(k, sorted(v)) for k, v in entity_keys(decoded)]
        == [(k, sorted(v)) for k, v in entity_keys(row["output"])],
        "confidence_max_abs": conf,
        "flips": flips,
        "nan_or_inf": bool(
            not (np.isfinite(hidden).all() and np.isfinite(count).all() and np.isfinite(cand_logits).all())
        ),
    }


def run_singles(graph: Graph, cases) -> tuple[list[dict], dict]:
    rows, agg, times = [], {}, []
    for row, ref in cases:
        t0 = time.perf_counter()
        hidden, count, logits = graph.run(
            ref["input_ids"], ref["attention_mask"], ref["word_index"], ref["schema_index"]
        )
        times.append(time.perf_counter() - t0)
        r = compare_case(row, ref, hidden[0], count[0], logits[0])
        rows.append(r)
        for key in ("hidden", "count_logits", "span_logits", "scores"):
            agg[key] = merge(agg.get(key), r[key])
    summary = {
        **{k: agg[k] for k in ("hidden", "count_logits", "span_logits", "scores")},
        "cases": len(rows),
        "entities_equal": sum(r["entities_equal"] for r in rows),
        "cases_with_different_entities": [r["id"] for r in rows if not r["entities_equal"]],
        "cases_with_order_only_differences": [
            r["id"] for r in rows if r["entity_sets_equal"] and not r["entities_equal"]
        ],
        "flips": [dict(f, id=r["id"]) for r in rows for f in r["flips"]],
        "gate_mismatches": [r["id"] for r in rows if not r["gate_equal"]],
        "confidence_max_abs": max((r["confidence_max_abs"] or 0.0) for r in rows),
        "nan_or_inf": [r["id"] for r in rows if r["nan_or_inf"]],
        "worst_span_logits_case": max(rows, key=lambda r: r["span_logits"]["max"])["id"],
        "worst_hidden_case": max(rows, key=lambda r: r["hidden"]["max"])["id"],
        "seconds_total_single": round(sum(times), 3),
    }
    return rows, summary


def run_batches(graph: Graph, cases, singles: dict, batch_size: int, max_tokens: int = 0) -> dict:
    """Padded batches of cases with the same label count, against the same graph's single runs."""
    groups: dict[int, list] = {}
    for row, ref in (c for c in cases if not max_tokens or c[1]["input_ids"].shape[1] <= max_tokens):
        groups.setdefault(len(row["labels"]), []).append((row, ref))
    batch_rows, hid, lg = [], None, None
    for _m, members in sorted(groups.items()):
        for i in range(0, len(members), batch_size):
            chunk = members[i : i + batch_size]
            if len(chunk) < 2:
                continue
            s = max(r["input_ids"].shape[1] for _, r in chunk)
            w = max(r["word_index"].shape[1] for _, r in chunk)
            b = len(chunk)
            ids = np.zeros((b, s), np.int64)
            mask = np.zeros((b, s), np.int64)
            wi = np.zeros((b, w), np.int64)
            si = np.stack([r["schema_index"][0] for _, r in chunk])
            for j, (_, r) in enumerate(chunk):
                ids[j, : r["input_ids"].shape[1]] = r["input_ids"][0]
                mask[j, : r["input_ids"].shape[1]] = r["attention_mask"][0]
                wi[j, : r["word_index"].shape[1]] = r["word_index"][0]
            hidden, count, logits = graph.run(ids, mask, wi, si)
            for j, (row, ref) in enumerate(chunk):
                one = singles[row["id"]]
                sl = ref["input_ids"].shape[1]
                n = len(row["start_map"])
                k = row["span_logits_shape"][2]
                valid = (np.arange(n)[:, None] + np.arange(k)[None, :]) < n
                m = len(row["labels"])
                h = stats(hidden[j, :sl] - one["hidden"])
                lgs = stats(np.stack([logits[j, q, :n][valid] - one["logits"][q, :n][valid] for q in range(m)]))
                ref_cmp = compare_case(row, ref, hidden[j, :sl], count[j], logits[j])
                hid, lg = merge(hid, h), merge(lg, lgs)
                batch_rows.append(
                    {
                        "id": row["id"],
                        "batch": [r["id"] for r, _ in chunk],
                        "hidden_vs_single": h,
                        "span_logits_vs_single": lgs,
                        "entities_equal_reference": ref_cmp["entities_equal"],
                    }
                )
    return {
        "batch_size": batch_size,
        "grouping": "cases with the same label count, corpus order",
        "max_tokens": max_tokens or None,
        "rows_compared": len(batch_rows),
        "hidden_vs_single": hid,
        "span_logits_vs_single": lg,
        "cases_with_different_entities_vs_reference": [
            r["id"] for r in batch_rows if not r["entities_equal_reference"]
        ],
        "rows": batch_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", default="1,12")
    parser.add_argument("--cuda-tf32", default="0", help="comma list of use_tf32 values for the CUDA EP")
    parser.add_argument("--graphs", default="A-legacy,A-dynamo,B-legacy+H-legacy,B-dynamo+H-dynamo")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--batch-max-tokens", type=int, default=0, help="leave longer cases out of the batch check (GPU memory)"
    )
    parser.add_argument("--per-case", action="store_true", help="include per-case rows")
    args = parser.parse_args()

    cases = load_reference(args.reference)
    results = []
    settings = (
        [(int(t), 0) for t in args.threads.split(",")]
        if args.provider == "cpu"
        else [(1, int(v)) for v in args.cuda_tf32.split(",")]
    )
    for name in args.graphs.split(","):
        for threads, tf32 in settings:
            label = f"{name} {args.provider} " + (
                f"threads={threads}" if args.provider == "cpu" else f"use_tf32={tf32}"
            )
            prof = tempfile.mkdtemp(prefix="g2prof")
            t0 = time.time()
            graph = Graph(name, args.onnx_dir, args.provider, threads, tf32, profile_dir=prof)
            load_s = time.time() - t0
            rows, summary = run_singles(graph, cases)
            placement = graph.placement()
            del graph
            gc.collect()
            # Fresh sessions (profiling off) for repeatability and batches.
            graph = Graph(name, args.onnx_dir, args.provider, threads, tf32)
            singles = {}
            repeat = None
            for row, ref in cases:
                hidden, count, logits = graph.run(
                    ref["input_ids"], ref["attention_mask"], ref["word_index"], ref["schema_index"]
                )
                singles[row["id"]] = {"hidden": hidden[0], "logits": logits[0]}
                prev = next(r for r in rows if r["id"] == row["id"])
                again = compare_case(row, ref, hidden[0], count[0], logits[0])
                d = abs(again["span_logits"]["max"] - prev["span_logits"]["max"])
                repeat = max(repeat or 0.0, d)
            batches = run_batches(graph, cases, singles, args.batch_size, args.batch_max_tokens)
            entry = {
                "graph": name,
                "provider": args.provider,
                "threads": threads,
                "use_tf32": tf32 if args.provider == "cuda" else None,
                "session_load_seconds": round(load_s, 2),
                "summary": summary,
                "repeatability_span_logits_max_delta": repeat,
                "batch": {k: v for k, v in batches.items() if k != "rows"},
                "placement": placement,
            }
            if args.per_case:
                entry["per_case"] = rows
                entry["batch_rows"] = batches["rows"]
            results.append(entry)
            del graph
            gc.collect()
            s = summary
            hid, lg = s["hidden"], s["span_logits"]
            print(
                f"{label:44s} hidden max {hid['max']:.2e} p99 {hid['p99']:.2e} | span logits max "
                f"{lg['max']:.2e} mean {lg['mean']:.2e} p99 {lg['p99']:.2e} | "
                f"scores max {s['scores']['max']:.2e} | entities {s['entities_equal']}/{s['cases']} "
                f"flips {len(s['flips'])} | batch logits max {batches['span_logits_vs_single']['max']:.2e} "
                f"| nan {len(s['nan_or_inf'])} | {s['seconds_total_single']}s"
            )
    fixtures.write_json(
        args.out,
        {
            "reference": str(args.reference.name),
            "onnxruntime": ort.__version__,
            "environment": fixtures.environment(["onnxruntime", "onnxruntime-gpu", "numpy", "torch"]),
            "results": results,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
