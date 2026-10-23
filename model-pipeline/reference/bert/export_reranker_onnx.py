"""Export an XLM-R cross-encoder (AutoModelForSequenceClassification) to an fp32 ONNX graph.

Used once to build Starlinghq/bge-reranker-v2-m3-onnx from BAAI/bge-reranker-v2-m3; not part of the
onboarding stages. Inputs are input_ids and attention_mask (int64, dynamic batch and sequence); the
single output is the raw relevance logit, ``logits`` [batch, 1]. Weights go to one external data file
(``model.onnx_data``) because the fp32 graph exceeds the 2 GB protobuf limit.

    python export_reranker_onnx.py --snapshot <upstream snapshot> --out <dir>/onnx
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import onnx
import torch
from transformers import AutoModelForSequenceClassification

OPSET = 17


class Logits(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True).logits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.snapshot), dtype=torch.float32, attn_implementation="eager"
    ).eval()
    if model.config.num_labels != 1:
        raise ValueError("expected a single-logit cross-encoder")
    wrapper = Logits(model).eval()
    ids = torch.tensor([[0, 2367, 83, 6, 85407, 32, 2, 2, 1274, 2], [0, 2367, 83, 2, 2, 1274, 2, 1, 1, 1]])
    mask = (torch.arange(ids.shape[1])[None, :] < torch.tensor([[10], [7]])).long()
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out) as scratch, torch.inference_mode():
        staged = Path(scratch) / "model.onnx"
        torch.onnx.export(
            wrapper,
            (ids, mask),
            str(staged),
            dynamo=False,
            opset_version=OPSET,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            do_constant_folding=True,
        )
        graph = onnx.load(str(staged), load_external_data=True)
        onnx.save_model(
            graph,
            str(args.out / "model.onnx"),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.onnx_data",
            size_threshold=1024,
        )
    onnx.checker.check_model(str(args.out / "model.onnx"))
    print(f"wrote {args.out / 'model.onnx'} (opset {OPSET}, external data model.onnx_data)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
