"""Browser spike (docs/m0/browser-wasm-spike.md): e5-small embedding drift of each graph variant and backend.

Reference: the protected fp32 PyTorch (sentence-transformers) embeddings,
`model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/embeddings.jsonl` (46 cases, fixture input ids). The
inputs are those ids, so tokenization is out of the comparison. Pooling: mean over the mask, L2 normalization.

Subcommands:

- `native`: Python ONNX Runtime (CPUExecutionProvider) on each graph (`model-<variant>.onnx`), batch 1 per case, plus
  the padded short-b8 batch, plus warm latency (median over --iterations) for passage34-b1, short-b8, long512-b1.
  Separates quantization drift from browser-backend drift.
- `browser`: reads `run_spike.py e5` result files and computes the same statistics from the embeddings they carry;
  with --native, also each browser run against the native run of the same graph (backend drift alone).

Statistics per (variant, backend): max_abs and mean_abs over all 46 x 384 values, min_cosine over cases, the
worst case, and the padded batch-of-8 against the same backend's single-row embeddings (max_abs).

Usage:
    python drift_e5_browser.py native --fixtures <protected dir> --graphs <dir> --variants fp32 fp16 int8-dynamic \
        --out native.json
    python drift_e5_browser.py browser --fixtures <protected dir> --native native.json --out browser.json \
        result1.json result2.json ...
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np


def load_reference(fixtures: Path) -> list[dict]:
    lines = (fixtures / "embeddings.jsonl").read_text(encoding="utf-8").split("\n")
    return [json.loads(line) for line in lines if line.strip()]


def pooled(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean over the attention mask then L2 normalization, accumulated in float64 (as the C# library)."""
    m = mask[..., None].astype(np.float64)
    summed = (hidden.astype(np.float64) * m).sum(axis=1)
    mean = summed / np.maximum(m.sum(axis=1), 1e-9)
    norm = np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
    return (mean / norm).astype(np.float32)


def stats(reference: list[dict], embeddings: dict[str, np.ndarray]) -> dict:
    diffs, cosines = [], {}
    for row in reference:
        ref = np.asarray(row["embedding"], dtype=np.float64)
        got = np.asarray(embeddings[row["id"]], dtype=np.float64)
        diffs.append(np.abs(ref - got))
        cosines[row["id"]] = float(ref @ got / (np.linalg.norm(ref) * np.linalg.norm(got)))
    flat = np.concatenate(diffs)
    worst = min(cosines, key=cosines.get)
    return {
        "cases": len(reference),
        "max_abs": float(flat.max()),
        "mean_abs": float(flat.mean()),
        "p99_abs": float(np.percentile(flat, 99)),
        "min_cosine": cosines[worst],
        "min_cosine_case": worst,
        "nan_or_inf": bool(not np.isfinite(flat).all()),
    }


def short_batch(reference: list[dict], size: int = 8) -> list[dict]:
    return [r for r in reference if len(r["input_ids"]) < 512][:size]


def pack(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    seq = max(len(r) for r in rows)
    ids = np.ones((len(rows), seq), dtype=np.int64)
    mask = np.zeros((len(rows), seq), dtype=np.int64)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = r
        mask[i, : len(r)] = 1
    return ids, mask


def native(args: argparse.Namespace) -> None:
    import onnxruntime as ort

    reference = load_reference(args.fixtures)
    by_id = {r["id"]: r for r in reference}
    out = {
        "candidate": "Python ONNX Runtime CPUExecutionProvider",
        "onnxruntime": ort.__version__,
        "machine": platform.processor() or platform.machine(),
        "threads": args.threads,
        "variants": {},
    }
    for variant in args.variants:
        path = args.graphs / f"model-{variant}.onnx"
        options = ort.SessionOptions()
        if args.threads:
            options.intra_op_num_threads = args.threads
        t0 = time.perf_counter()
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        load_ms = (time.perf_counter() - t0) * 1000

        def run(rows: list[list[int]], s=session) -> np.ndarray:
            ids, mask = pack(rows)
            hidden = s.run(None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": np.zeros_like(ids)})[0]
            return pooled(hidden, mask)

        embeddings = {r["id"]: run([r["input_ids"]])[0] for r in reference}
        batch_rows = short_batch(reference)
        batch = run([r["input_ids"] for r in batch_rows])
        batch_vs_single = max(float(np.abs(batch[i] - embeddings[r["id"]]).max()) for i, r in enumerate(batch_rows))
        latency = {}
        for name, rows, n in (
            ("passage34-b1", [by_id["e5-passage-prefix"]["input_ids"]], args.iterations),
            ("short-b8", [r["input_ids"] for r in batch_rows], args.iterations),
            ("long512-b1", [by_id["long-english"]["input_ids"]], max(3, args.iterations // 4)),
        ):
            for _ in range(3):
                run(rows)
            samples = []
            for _ in range(n):
                t0 = time.perf_counter()
                run(rows)
                samples.append((time.perf_counter() - t0) * 1000)
            latency[name] = {"median_ms": statistics.median(samples), "samples_ms": samples}
        out["variants"][variant] = {
            "graph": path.name,
            "bytes": path.stat().st_size,
            "load_ms": load_ms,
            "drift_vs_reference": stats(reference, embeddings),
            "batch8_vs_single_max_abs": batch_vs_single,
            "latency": latency,
            "embeddings": {k: [float(x) for x in v] for k, v in embeddings.items()},
        }
        d = out["variants"][variant]["drift_vs_reference"]
        print(f"{variant:>13} max_abs {d['max_abs']:.3g} min_cos {d['min_cosine']:.6f} ({d['min_cosine_case']})")
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


def browser(args: argparse.Namespace) -> None:
    reference = load_reference(args.fixtures)
    native_runs = json.loads(args.native.read_text(encoding="utf-8"))["variants"] if args.native else {}
    out = []
    for path in args.results:
        result = json.loads(path.read_text(encoding="utf-8"))
        e5 = result.get("e5", {})
        entry = {
            "file": path.name,
            "label": result.get("label"),
            "variant": result.get("variant"),
            "backend": result.get("backend"),
            "threads": e5.get("load", {}).get("num_threads"),
            "load": {k: v for k, v in e5.get("load", {}).items() if k not in ("adapter_features",)},
        }
        if "embeddings" in e5:
            embeddings = {r["id"]: np.asarray(r["embedding"], dtype=np.float32) for r in e5["embeddings"]}
            entry["drift_vs_reference"] = stats(reference, embeddings)
            native_run = native_runs.get(entry["variant"])
            if native_run:
                # Backend drift: the same graph in the browser against native Python ONNX Runtime (CPU EP).
                entry["vs_native_ort"] = stats(
                    [{"id": k, "embedding": v} for k, v in native_run["embeddings"].items()], embeddings
                )
            entry["batch8_vs_single_max_abs"] = max(
                float(np.abs(np.asarray(r["embedding"], dtype=np.float32) - embeddings[r["id"]]).max())
                for r in e5["batch8"]
            )
            entry["first_run_ms"] = e5.get("first_run_ms")
            entry["latency_js_median_ms"] = {
                k: statistics.median(v["samples_ms"]) for k, v in e5.get("latency_js", {}).items()
            }
            entry["latency_e2e_median_ms"] = {k: statistics.median(v) for k, v in e5.get("latency_e2e", {}).items()}
            entry["memory"] = e5.get("memory")
        entry["peak_rss_tree_bytes"] = result.get("peak_rss_tree_bytes")
        entry["peak_rss_renderer_bytes"] = result.get("peak_rss_renderer_bytes")
        out.append(entry)
        d = entry.get("drift_vs_reference")
        summary = f"max_abs {d['max_abs']:.3g} min_cos {d['min_cosine']:.6f}" if d else entry["load"].get("error")
        print(f"{entry['variant']:>13} {entry['backend']:>6}  {summary}")
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    n = sub.add_parser("native")
    n.add_argument("--fixtures", type=Path, required=True)
    n.add_argument("--graphs", type=Path, required=True)
    n.add_argument("--variants", nargs="+", default=["fp32", "fp16", "int8-dynamic"])
    n.add_argument("--threads", type=int, default=0)
    n.add_argument("--iterations", type=int, default=20)
    n.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("browser")
    b.add_argument("--fixtures", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--native", type=Path, help="`native` output, for the browser-vs-native comparison")
    b.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args()
    native(args) if args.command == "native" else browser(args)


if __name__ == "__main__":
    main()
