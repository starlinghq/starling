"""Stage 2 (export) results for the e5 browser profiles: Python ONNX Runtime (CPU) on the browser profile's graph.

The browser profiles (`multilingual-e5-small/onnx-{fp32,int8embed}-{wasm,webgpu}`) have no hidden-state export stage:
their protected tolerances bind no `float.last_hidden_state_max_abs`. Their stage 2 is the same graph the browser runs
(`model-fp32.onnx` = the Hub's `onnx/model.onnx`, or `model-int8-embed.onnx` from `quantize_e5_browser.py`) on native
ONNX Runtime, batch 1 per protected fixture case, plus two padded batches: the protected `tokenizer-batch.json` and the
first 8 cases shorter than 512 tokens (the conformance app's `batch8`). Pooling is the library's: mean over the mask
and L2 normalization, accumulated in float64. The evaluator compares this stage with the PyTorch reference (stage 2
bounds) and with the browser stage (`browser_vs_native_same_graph`).

Writes `export.results.json` in the stage results format. It applies no thresholds.

Usage (environment: requirements-browser.txt, plus model-pipeline/src on PYTHONPATH):
    PYTHONPATH=model-pipeline/src python stage_e5_browser_native.py --graph <model-int8-embed.onnx> \
        --graph-path onnx/model-int8-embed.onnx \
        --fixtures model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu \
        --profile multilingual-e5-small/onnx-int8embed-wasm --out <results dir>
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import onnxruntime as ort

from model_pipeline import fixtures, results

PAD_ID = 1
SHORT_BATCH = 8
MAX_TOKENS = 512
PACKAGES = ["onnxruntime", "numpy"]


def pooled(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean over the attention mask then L2 normalization, accumulated in float64 (as the C# library)."""
    m = mask[..., None].astype(np.float64)
    mean = (hidden.astype(np.float64) * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)
    return (mean / np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def padded(rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    width = max(len(r) for r in rows)
    ids = np.array([r + [PAD_ID] * (width - len(r)) for r in rows], dtype=np.int64)
    mask = np.array([[1] * len(r) + [0] * (width - len(r)) for r in rows], dtype=np.int64)
    return ids, mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", type=Path, required=True, help="the browser profile's graph file")
    parser.add_argument("--graph-path", required=True, help="the graph's path as the manifest names it")
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--intra-op-threads", type=int, default=0, help="0 keeps the ONNX Runtime default")
    args = parser.parse_args()

    cases = fixtures.read_jsonl(args.fixtures / "embeddings.jsonl")
    batch_fx = json.loads((args.fixtures / "tokenizer-batch.json").read_text(encoding="utf-8"))
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.intra_op_threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(args.graph), options, providers=["CPUExecutionProvider"])
    names = {i.name for i in session.get_inputs()}

    def run(ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        return pooled(session.run(["last_hidden_state"], feeds)[0], mask)

    env = fixtures.environment(PACKAGES) | {
        "machine": platform.node(),
        "provider": "cpu",
        "provider_options": {},
        "onnxruntime_providers": session.get_providers(),
        "intra_op_threads": args.intra_op_threads,
    }
    out = results.new_results(
        profile=args.profile,
        stage="export",
        candidate=f"Python onnxruntime {ort.__version__} CPUExecutionProvider on {args.graph_path} (the browser "
        "profile's graph), float64 pooling; no hidden-state reference (not bound for browser profiles)",
        producer={
            "name": "model-pipeline/reference/encoder/stage_e5_browser_native.py",
            "repository_commit": fixtures.git_head(Path(__file__).parent),
        },
        environment=env,
        assets=[{"path": args.graph_path, "sha256": fixtures.sha256_file(args.graph)}],
    )
    for case in cases:
        ids = np.array([case["input_ids"]], dtype=np.int64)
        out["cases"][case["id"]] = {
            "input_ids": case["input_ids"],
            "embedding": fixtures.float32_list(run(ids, np.ones_like(ids))[0]),
        }
    rows = batch_fx["rows"]
    ids = np.array([r["ids"] for r in rows], dtype=np.int64)
    mask = np.array([r["attention_mask"] for r in rows], dtype=np.int64)
    out["batches"]["fixture-padded"] = {
        "output": "embedding",
        "case_ids": [r["id"] for r in rows],
        "values": [fixtures.float32_list(v) for v in run(ids, mask)],
    }
    short = [c for c in cases if len(c["input_ids"]) < MAX_TOKENS][:SHORT_BATCH]
    ids, mask = padded([c["input_ids"] for c in short])
    out["batches"]["short-b8"] = {
        "output": "embedding",
        "case_ids": [c["id"] for c in short],
        "values": [fixtures.float32_list(v) for v in run(ids, mask)],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    fixtures.write_json(args.out / "export.results.json", out)
    print(f"wrote {args.out / 'export.results.json'} ({len(cases)} cases, 2 padded batches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
