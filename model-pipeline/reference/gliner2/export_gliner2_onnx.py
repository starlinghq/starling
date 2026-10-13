"""Export spike for GLiNER2 entity extraction (backlog P-014). CPU, fp32.

Routes (see `gliner2_entity.py` for the modules):

    A  full model: input_ids, attention_mask, word_index, schema_index
                   -> count_logits [B, 20], span_logits [B, M, W, 8], last_hidden_state [B, S, 768]
    B  backbone:   input_ids, attention_mask -> last_hidden_state [B, S, 768]
    H  heads only: words [B, W, 768], schema [B, 1+M, 768] -> count_logits, span_logits
                   (B + H is the split route; B alone needs the heads ported to C#)

Each route is exported with the TorchScript exporter (`dynamo=False`) and with the dynamo exporter
(`dynamo=True`). Failures are recorded, not hidden: the summary lists every attempt with its error.

Usage:
    python export_gliner2_onnx.py --model-dir <snapshot> --out <dir> [--routes A,B,H] [--exporters legacy,dynamo]
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import onnx
import torch
from gliner2 import GLiNER2
from gliner2_entity import Backbone, EntityHeads, EntityModel

from model_pipeline import fixtures

OPSET = 18


def sample_inputs(route: str):
    """Small dummy inputs with distinct batch, sequence, word and label sizes (dynamic axes must not alias)."""
    g = torch.Generator().manual_seed(0)
    b, s, w, m = 2, 23, 9, 3
    ids = torch.randint(5, 1000, (b, s), generator=g)
    mask = torch.ones(b, s, dtype=torch.long)
    if route == "A":
        word_index = torch.arange(10, 10 + w).unsqueeze(0).expand(b, -1).contiguous()
        schema_index = torch.tensor([[1, 4, 6, 8]] * b)
        return (ids, mask, word_index, schema_index)
    if route == "B":
        return (ids, mask)
    words = torch.randn(b, w, 768, generator=g)
    schema = torch.randn(b, 1 + m, 768, generator=g)
    return (words, schema)


ROUTES = {
    "A": {
        "module": EntityModel,
        "inputs": ["input_ids", "attention_mask", "word_index", "schema_index"],
        "outputs": ["count_logits", "span_logits", "last_hidden_state"],
        "dynamic_axes": {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "word_index": {0: "batch", 1: "words"},
            "schema_index": {0: "batch", 1: "schema_slots"},
            "count_logits": {0: "batch"},
            "span_logits": {0: "batch", 1: "labels", 2: "words"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
    },
    "B": {
        "module": Backbone,
        "inputs": ["input_ids", "attention_mask"],
        "outputs": ["last_hidden_state"],
        "dynamic_axes": {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
    },
    "H": {
        "module": EntityHeads,
        "inputs": ["words", "schema"],
        "outputs": ["count_logits", "span_logits"],
        "dynamic_axes": {
            "words": {0: "batch", 1: "words"},
            "schema": {0: "batch", 1: "schema_slots"},
            "count_logits": {0: "batch"},
            "span_logits": {0: "batch", 1: "labels", 2: "words"},
        },
    },
}


class WithHidden(torch.nn.Module):
    """Route A with the backbone output exposed for stage diagnosis."""

    def __init__(self, extractor, explicit_attention: bool = False):
        super().__init__()
        self.inner = EntityModel(extractor, explicit_attention)

    def forward(self, input_ids, attention_mask, word_index, schema_index):
        hidden = self.inner.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        d = hidden.shape[-1]
        words = torch.gather(hidden, 1, word_index.unsqueeze(-1).expand(-1, -1, d))
        schema = torch.gather(hidden, 1, schema_index.unsqueeze(-1).expand(-1, -1, d))
        count_logits, span_logits = self.inner.heads(words, schema)
        return count_logits, span_logits, hidden


def dynamic_shapes(route: str):
    spec = ROUTES[route]["dynamic_axes"]
    dims: dict[str, torch.export.Dim] = {}

    def dim(name):
        if name not in dims:
            dims[name] = torch.export.Dim(name, min=1, max=8192)
        return dims[name]

    return tuple({ax: dim(n) for ax, n in spec[i].items()} for i in ROUTES[route]["inputs"])


def export_one(extractor, route: str, exporter: str, out: Path, explicit: bool) -> dict:
    spec = ROUTES[route]
    if route == "A":
        module = WithHidden(extractor, explicit)
    elif route == "H":
        module = EntityHeads(extractor, explicit)
    else:
        module = Backbone(extractor)
    module = module.eval()
    args = sample_inputs(route)
    path = out / f"gliner2-{route}-{exporter}.onnx"
    rec = {"route": route, "exporter": exporter, "path": path.name, "opset": OPSET, "explicit_attention": explicit}
    t0 = time.time()
    try:
        with torch.inference_mode(False), torch.no_grad():
            if exporter == "legacy":
                torch.onnx.export(
                    module,
                    args,
                    str(path),
                    input_names=spec["inputs"],
                    output_names=spec["outputs"],
                    dynamic_axes=spec["dynamic_axes"],
                    opset_version=OPSET,
                    do_constant_folding=True,
                    dynamo=False,
                )
            else:
                program = torch.onnx.export(
                    module,
                    args,
                    input_names=spec["inputs"],
                    output_names=spec["outputs"],
                    dynamic_shapes=dict(zip(spec["inputs"], dynamic_shapes(route), strict=True)),
                    opset_version=OPSET,
                    dynamo=True,
                    optimize=True,
                    external_data=True,
                )
                program.save(str(path), external_data=True)
        rec["seconds"] = round(time.time() - t0, 1)
        model = onnx.load(str(path), load_external_data=False)
        ops: dict[str, int] = {}
        for node in model.graph.node:
            ops[node.op_type] = ops.get(node.op_type, 0) + 1
        rec.update(
            status="exported",
            nodes=len(model.graph.node),
            op_types=dict(sorted(ops.items())),
            files=[fixtures.file_record(p, out) for p in sorted(out.glob(f"{path.name}*"))],
            inputs=[
                (i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in model.graph.input
            ],
            outputs=[
                (o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]) for o in model.graph.output
            ],
        )
        try:
            onnx.checker.check_model(str(path))
            rec["onnx_checker"] = "ok"
        except Exception as exc:  # noqa: BLE001 - recorded
            rec["onnx_checker"] = f"{type(exc).__name__}: {str(exc)[:500]}"
    except Exception as exc:  # noqa: BLE001 - every failure is evidence
        rec.update(
            status="failed",
            seconds=round(time.time() - t0, 1),
            error=f"{type(exc).__name__}: {str(exc)[:2000]}",
            traceback_tail=traceback.format_exc()[-3000:],
        )
    return rec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--routes", default="A,B,H")
    parser.add_argument("--exporters", default="legacy,dynamo")
    parser.add_argument("--keep-mha-fastpath", action="store_true", help="reproduce the unrepaired export failure")
    parser.add_argument(
        "--explicit-attention",
        action="store_true",
        help="repair 2 (TorchScript exporter): explicit encoder layers in count_embed "
        "(gliner2_entity.encoder_layer_explicit)",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    extractor = GLiNER2.from_pretrained(str(args.model_dir))
    extractor.eval()
    # Repair 1 for the TorchScript exporter: in eval mode nn.TransformerEncoderLayer (count_embed's
    # DownscaledTransformer) takes the fused fast path `aten::_transformer_encoder_layer_fwd`, which has no
    # ONNX symbolic. The slow path computes the same function with standard operators.
    if not args.keep_mha_fastpath:
        torch.backends.mha.set_fastpath_enabled(False)
    results = []
    for route in args.routes.split(","):
        for exporter in args.exporters.split(","):
            rec = export_one(extractor, route, exporter, args.out, args.explicit_attention)
            print(
                json.dumps(
                    {
                        k: rec.get(k)
                        for k in ("route", "exporter", "status", "seconds", "nodes", "onnx_checker", "error")
                    }
                )[:1500]
            )
            results.append(rec)
    summary = {
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "environment": fixtures.environment(
            ["torch", "onnx", "onnxscript", "onnx-ir", "onnxruntime", "transformers", "gliner2"]
        ),
        "attempts": results,
    }
    existing = args.out / "export-summary.json"
    if existing.exists():
        old = json.loads(existing.read_text())
        keep = [
            a
            for a in old.get("attempts", [])
            if (a["route"], a["exporter"]) not in {(r["route"], r["exporter"]) for r in results}
        ]
        summary["attempts"] = keep + results
    fixtures.write_json(existing, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
