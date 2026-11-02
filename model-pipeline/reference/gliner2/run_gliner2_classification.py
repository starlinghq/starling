"""GLiNER2 text-classification reference run, CPU fp32 (issue #183, step 1).

For each case of a classification corpus (id, task, labels, multi_label, cls_threshold, text), runs the pinned GLiNER2
public API `GLiNER2.classify_text(text, {task: {"labels": ..., "multi_label": ..., "cls_threshold": ...}},
include_confidence=True)` and records:

- the public output;
- the model input GLiNER2 builds: input ids (captured at the encoder) and the schema routing positions (`[P]`, then
  one `[L]` per label, in label order);
- the classifier logits (`classifier(embs[1:]).squeeze(-1)`, captured at the module) and the probabilities GLiNER2
  derives from them (softmax over the labels for single-label, sigmoid for multi-label; temperature 1).

With `--onnx`, the released route A graph (`gliner2-entities-fp32.onnx`, the profile's locked graph, which exposes
`last_hidden_state`) is run on the same ids with ONNX Runtime CPU; the `[L]` rows of its hidden state go through the
classifier head restated in NumPy from the checkpoint's `classifier.*` tensors (`Linear 768->1536, ReLU,
Linear 1536->1`). That is the route the .NET package takes, so the summary's `ort_vs_torch` errors bound what a
faithful port can reach.

Writes `classification.jsonl` (one row per case) and `environment.json`.

Usage:
    python run_gliner2_classification.py --model-dir <snapshot f9634218> --corpus <jsonl> --out <dir> \
        [--onnx <gliner2-entities-fp32.onnx>] [--threads 1]
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import gliner2
import numpy as np
import torch
from gliner2 import GLiNER2
from run_gliner2_tokenizer import ASSETS, gliner2_source_revision

from model_pipeline import fixtures

PACKAGES = ["gliner2", "transformers", "tokenizers", "sentencepiece", "torch", "numpy", "safetensors", "onnxruntime"]
HEAD_KEYS = ("classifier.0.weight", "classifier.0.bias", "classifier.2.weight", "classifier.2.bias")


def f32(values) -> list[float]:
    """float32 values as the shortest decimal that round-trips to the same float32."""
    return [float(str(np.float32(v))) for v in np.asarray(values, dtype=np.float32).reshape(-1)]


def probabilities(logits: torch.Tensor, multi_label: bool) -> torch.Tensor:
    """`_extract_classification_result` with class_act "auto" and temperature 1."""
    return torch.sigmoid(logits) if multi_label else torch.softmax(logits, dim=-1)


def load_head(model_dir: Path) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file

    tensors = load_file(str(model_dir / "model.safetensors"))
    return {k: tensors[k].astype(np.float32) for k in HEAD_KEYS}


def head_numpy(head: dict[str, np.ndarray], embs: np.ndarray) -> np.ndarray:
    h = np.maximum(embs @ head["classifier.0.weight"].T + head["classifier.0.bias"], 0.0)
    return (h @ head["classifier.2.weight"].T + head["classifier.2.bias"]).reshape(-1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    extractor = GLiNER2.from_pretrained(str(args.model_dir))
    extractor.eval()
    tok = extractor.processor.tokenizer
    p_id, l_id, sep_id = (tok.convert_tokens_to_ids(t) for t in ("[P]", "[L]", "[SEP_TEXT]"))

    captured: dict[str, torch.Tensor] = {}

    def on_encoder(_module, a, kwargs):
        captured["input_ids"] = (kwargs.get("input_ids") if "input_ids" in kwargs else a[0]).detach().clone()

    def on_classifier(_module, _inputs, output):
        captured["logits"] = output.detach().clone().squeeze(-1)

    extractor.encoder.register_forward_pre_hook(on_encoder, with_kwargs=True)
    extractor.classifier.register_forward_hook(on_classifier)

    session = None
    head = None
    if args.onnx:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = args.threads
        session = ort.InferenceSession(str(args.onnx), opts, providers=["CPUExecutionProvider"])
        head = load_head(args.model_dir)

    rows = []
    worst = {"logits_max_abs": 0.0, "logits_relative": 0.0, "probs_max_abs": 0.0, "decisions_equal": 0}
    for case in fixtures.read_jsonl(args.corpus):
        captured.clear()
        tasks = {
            case["task"]: {
                "labels": case["labels"],
                "multi_label": case["multi_label"],
                "cls_threshold": case["cls_threshold"],
            }
        }
        with torch.no_grad():
            result = extractor.classify_text(case["text"], tasks, include_confidence=True)
        ids = captured["input_ids"]
        if ids.shape[0] != 1:
            raise RuntimeError(f"{case['id']}: expected one sequence, got {tuple(ids.shape)}")
        ids_list = ids[0].tolist()
        schema_positions = [i for i, t in enumerate(ids_list) if t in (p_id, l_id)]
        if len(schema_positions) != 1 + len(case["labels"]) or ids_list[schema_positions[0]] != p_id:
            raise RuntimeError(f"{case['id']}: schema positions {schema_positions} do not match the labels")
        logits = captured["logits"].reshape(-1)
        probs = probabilities(logits, case["multi_label"])
        row = {
            "id": case["id"],
            "task": case["task"],
            "labels": case["labels"],
            "multi_label": case["multi_label"],
            "cls_threshold": case["cls_threshold"],
            "text": case["text"],
            "input_ids": ids_list,
            "schema_positions": schema_positions,
            "logits": f32(logits.numpy()),
            "probabilities": f32(probs.numpy()),
            "output": result,
        }
        if session is not None:
            sep = ids_list.index(sep_id)
            feeds = {
                "input_ids": ids.numpy().astype(np.int64),
                "attention_mask": np.ones_like(ids.numpy(), dtype=np.int64),
                "word_index": np.array([[sep + 1]], dtype=np.int64),
                "schema_index": np.array([schema_positions], dtype=np.int64),
            }
            hidden = session.run(["last_hidden_state"], feeds)[0][0]
            ort_logits = head_numpy(head, hidden[schema_positions[1:]])
            ort_probs = probabilities(torch.from_numpy(ort_logits), case["multi_label"]).numpy()
            d = np.abs(ort_logits - logits.numpy())
            rel = d / np.maximum(1.0, np.abs(logits.numpy()))
            worst["logits_max_abs"] = max(worst["logits_max_abs"], float(d.max()))
            worst["logits_relative"] = max(worst["logits_relative"], float(rel.max()))
            worst["probs_max_abs"] = max(worst["probs_max_abs"], float(np.abs(ort_probs - probs.numpy()).max()))
            worst["decisions_equal"] += int(int(np.argmax(ort_logits)) == int(torch.argmax(logits)))
            row["ort_logits"] = f32(ort_logits)
        rows.append(row)
        print(case["id"], result)

    args.out.mkdir(parents=True, exist_ok=True)
    fixtures.write_jsonl(args.out / "classification.jsonl", rows)
    env = {
        "fixture_format_version": 1,
        "runner": "model-pipeline/reference/gliner2/run_gliner2_classification.py",
        "model": {"repo": "fastino/gliner2-base-v1", "revision": "f9634218e53580c56edf0de97ca1a7d3f1c2354e"},
        "gliner2_source": {
            "url": "https://github.com/fastino-ai/GLiNER2",
            "revision": gliner2_source_revision(),
            "version": gliner2.__version__ if hasattr(gliner2, "__version__") else None,
        },
        "entry_point": (
            'GLiNER2.classify_text(text, {task: {"labels": labels, "multi_label": multi_label, '
            '"cls_threshold": cls_threshold}}, include_confidence=True)'
        ),
        "precision": "fp32",
        "device": "cpu",
        "torch_threads": args.threads,
        "corpus": fixtures.file_record(args.corpus),
        "assets": [fixtures.file_record(args.model_dir / a, args.model_dir) for a in [*ASSETS, "model.safetensors"]],
        "environment": fixtures.environment(PACKAGES),
    }
    if session is not None:
        env["onnx_graph"] = {
            "path": "onnx/gliner2-entities-fp32.onnx",
            "sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
        }
        env["ort_vs_torch"] = worst | {"cases": len(rows)}
        print("ort_vs_torch", env["ort_vs_torch"])
    fixtures.write_json(args.out / "environment.json", env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
