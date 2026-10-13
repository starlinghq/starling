"""Export the EmbeddingGemma 2 text path (`eg2_text.TextPath`) to ONNX (admission spike).

Exporters: `dynamo` (torch.onnx.export(..., dynamo=True), the first choice) and `legacy`
(TorchScript). Inputs `input_ids` and `attention_mask` (int64, [batch, sequence]), outputs
`last_hidden_state` ([batch, sequence, 512]) and `token_embeddings` ([batch, sequence, 768]).
Weights go to external data next to the graph. After each export the script checks the graph
with `onnx.checker` and runs it once in ONNX Runtime CPU against PyTorch on a short and a
600-token input; `export.json` records the outcome, node counts, sizes and digests.

Usage:
    python export_embeddinggemma2_onnx.py --model-dir <snapshot> --out <dir>
        [--exporters dynamo,legacy] [--attn sdpa|eager]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from eg2_text import TextPath, load_text_model


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sample(seq: int, batch: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seq)
    ids = torch.randint(3, 200000, (batch, seq), generator=g)
    ids[:, 0] = 2
    mask = torch.ones_like(ids)
    mask[1, seq // 2 :] = 0
    ids[1, seq // 2 :] = 0
    return ids, mask


def export(model: TextPath, exporter: str, path: Path) -> None:
    ids, mask = sample(17)
    names = {"input_names": ["input_ids", "attention_mask"], "output_names": ["last_hidden_state", "token_embeddings"]}
    if exporter == "dynamo":
        batch, seq = torch.export.Dim("batch"), torch.export.Dim("sequence", max=8192)
        torch.onnx.export(
            model,
            (ids, mask),
            str(path),
            dynamo=True,
            external_data=True,
            dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
            **names,
        )
    else:
        torch.onnx.export(
            model,
            (ids, mask),
            str(path),
            dynamo=False,
            opset_version=18,
            external_data=True,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "last_hidden_state": {0: "batch", 1: "sequence"},
                "token_embeddings": {0: "batch", 1: "sequence"},
            },
            **names,
        )


def check(model: TextPath, path: Path) -> dict:
    onnx.checker.check_model(str(path), full_check=False)
    graph = onnx.load(str(path), load_external_data=False)
    ops: dict[str, int] = {}
    for n in graph.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    runs = {}
    for seq in (9, 600):
        ids, mask = sample(seq)
        with torch.inference_mode():
            ref_h, ref_e = (t.numpy() for t in model(ids, mask))
        got_h, got_e = sess.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})
        valid = mask.numpy().astype(bool)
        runs[str(seq)] = {
            "hidden_max_abs_valid": float(np.abs(got_h - ref_h)[valid].max()),
            "token_embeddings_max_abs_valid": float(np.abs(got_e - ref_e)[valid].max()),
            "finite": bool(np.isfinite(got_h).all() and np.isfinite(got_e).all()),
        }
    data = [p for p in path.parent.iterdir() if p.name.startswith(path.name) and p != path] + [
        p for p in path.parent.iterdir() if p.suffix == ".data" and p.stem.startswith(path.stem)
    ]
    return {
        "opset": [{"domain": o.domain, "version": o.version} for o in graph.opset_import],
        "nodes": len(graph.graph.node),
        "op_types": dict(sorted(ops.items())),
        "graph_bytes": path.stat().st_size,
        "graph_sha256": sha256(path),
        "external_data": sorted({p.name: p.stat().st_size for p in data}.items()),
        "ort_cpu_vs_torch_random_ids": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exporters", default="dynamo,legacy")
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = TextPath(load_text_model(args.model_dir, args.attn)).eval()
    results = {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__, "attn": args.attn}
    for exporter in args.exporters.split(","):
        path = args.out / f"eg2-text-{exporter}-{args.attn}.onnx"
        t0 = time.time()
        try:
            with torch.inference_mode(False), torch.no_grad():
                export(model, exporter, path)
            entry = {"status": "exported", "seconds": round(time.time() - t0, 1)}
            entry.update(check(model, path))
        except Exception as exc:  # recorded as evidence; the stop rule is applied by the operator
            entry = {
                "status": "failed",
                "seconds": round(time.time() - t0, 1),
                "error": f"{type(exc).__name__}: {exc}"[:4000],
                "traceback_tail": traceback.format_exc()[-4000:],
            }
        results[f"{exporter}-{args.attn}"] = entry
        print(exporter, json.dumps({k: v for k, v in entry.items() if k not in ("op_types", "traceback_tail")})[:1500])
    out = args.out / f"export-{args.attn}.json"
    out.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
