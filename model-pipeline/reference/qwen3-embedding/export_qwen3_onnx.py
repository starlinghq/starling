"""Export Qwen3-Embedding-0.6B or Qwen3-Reranker-0.6B (`qwen3_graph.Qwen3Graph`) to ONNX, fp32, CPU.

Exporter: torch.onnx.export(..., dynamo=True), opset 20. Only if the dynamo export raises does the script fall back to
the TorchScript exporter (opset 18); `export.json` records which one produced the graph. Inputs `input_ids` and
`attention_mask` (int64, [batch, sequence], dynamic batch, sequence up to 8192); outputs `last_hidden_state`
(embedding) or `yes_no_logits` and `last_hidden_state` (reranker). Weights go to external data next to the graph
(`model.onnx` + `model.onnx.data`).

After the export the script checks the graph with `onnx.checker` and runs it once in ONNX Runtime CPU against PyTorch
on random ids (a 9- and a 600-token batch of two, the second row right-padded); `export.json` records the outcome,
node counts, sizes, digests and tool versions. `release_qwen3_onnx.py` turns the export into the hosted files.

Usage:
    python export_qwen3_onnx.py --model embedding|reranker --model-dir <snapshot> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import transformers
from qwen3_graph import EOS_TOKEN_ID, MAX_SEQUENCE, Qwen3Graph, load_backbone, output_names

DYNAMO_OPSET = 20
TORCHSCRIPT_OPSET = 18


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sample(seq: int, batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seq)
    ids = torch.randint(0, 151643, (batch, seq), generator=g)
    mask = torch.ones_like(ids)
    if batch > 1:
        mask[1, seq // 2 :] = 0
        ids[1, seq // 2 :] = EOS_TOKEN_ID
    return ids, mask


def export(graph: Qwen3Graph, exporter: str, path: Path) -> None:
    ids, mask = sample(17)
    names = {"input_names": ["input_ids", "attention_mask"], "output_names": output_names(graph.kind)}
    if exporter == "dynamo":
        batch, seq = torch.export.Dim("batch"), torch.export.Dim("sequence", max=MAX_SEQUENCE)
        torch.onnx.export(
            graph,
            (ids, mask),
            str(path),
            dynamo=True,
            opset_version=DYNAMO_OPSET,
            external_data=True,
            dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
            **names,
        )
    else:
        axes = {n: {0: "batch", 1: "sequence"} for n in names["input_names"] + names["output_names"]}
        torch.onnx.export(
            graph,
            (ids, mask),
            str(path),
            dynamo=False,
            opset_version=TORCHSCRIPT_OPSET,
            external_data=True,
            dynamic_axes=axes,
            **names,
        )


def check(graph: Qwen3Graph, path: Path, threads: int) -> dict:
    onnx.checker.check_model(str(path), full_check=False)
    model = onnx.load(str(path), load_external_data=False)
    ops: dict[str, int] = {}
    for n in model.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    runs = {}
    for seq in (9, 600):
        ids, mask = sample(seq)
        with torch.inference_mode():
            ref = graph(ids, mask)
        ref = [t.numpy() for t in (ref if isinstance(ref, tuple) else (ref,))]
        got = sess.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})
        valid = mask.numpy().astype(bool)
        runs[str(seq)] = {
            f"{name}_max_abs_valid": float(np.abs(g - r)[valid].max())
            for name, g, r in zip(output_names(graph.kind), got, ref, strict=True)
        }
        runs[str(seq)]["finite"] = bool(all(np.isfinite(g).all() for g in got))
    data = sorted(p for p in path.parent.iterdir() if p != path and p.name.startswith(path.name))
    return {
        "opset": [{"domain": o.domain, "version": o.version} for o in model.opset_import],
        "nodes": len(model.graph.node),
        "op_types": dict(sorted(ops.items())),
        "inputs": [i.name for i in model.graph.input],
        "outputs": [o.name for o in model.graph.output],
        "graph_bytes": path.stat().st_size,
        "graph_sha256": sha256(path),
        "external_data": [{"path": p.name, "size": p.stat().st_size} for p in data],
        "ort_cpu_vs_torch_random_ids": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["embedding", "reranker"])
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4, help="ONNX Runtime threads for the smoke check")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "model.onnx"
    if any(p.name.startswith(path.name) for p in args.out.iterdir()):
        print(f"{args.out} already holds an export; use an empty directory")
        return 2
    torch.manual_seed(0)
    graph = Qwen3Graph(load_backbone(args.model_dir), args.model).eval()
    record = {
        "model": args.model,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "onnx": onnx.__version__,
        "onnxscript": __import__("onnxscript").__version__,
        "onnxruntime": ort.__version__,
        "attempts": [],
    }
    for exporter in ("dynamo", "torchscript"):
        t0 = time.time()
        try:
            with torch.inference_mode(False), torch.no_grad():
                export(graph, exporter, path)
            entry = {"exporter": exporter, "status": "exported", "seconds": round(time.time() - t0, 1)}
            entry.update(check(graph, path, args.threads))
        except Exception as exc:  # recorded as evidence; the fallback and the stop rule are applied below
            entry = {
                "exporter": exporter,
                "status": "failed",
                "seconds": round(time.time() - t0, 1),
                "error": f"{type(exc).__name__}: {exc}"[:4000],
                "traceback_tail": traceback.format_exc()[-4000:],
            }
            for p in args.out.iterdir():
                if p.name.startswith(path.name):
                    p.unlink()
        record["attempts"].append(entry)
        print(exporter, json.dumps({k: v for k, v in entry.items() if k not in ("op_types", "traceback_tail")})[:2000])
        if entry["status"] == "exported":
            record["exporter"] = exporter
            record["opset"] = DYNAMO_OPSET if exporter == "dynamo" else TORCHSCRIPT_OPSET
            break
    (args.out / "export.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return 0 if "exporter" in record else 1


if __name__ == "__main__":
    raise SystemExit(main())
