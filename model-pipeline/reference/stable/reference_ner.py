"""Reference NER scores for the proposed GLiNER2 G3 sets: the upstream GLiNER2 package on PyTorch, CPU fp32.

Runs ``GLiNER2.extract(text, Schema().entities(labels), threshold=0.5, include_confidence=True,
include_spans=True)`` per gold row (the public API at its default operating point, which the .NET producer
``Starlings.Fidelity.Tool ner`` mirrors with ``Gliner2EntityExtractor.Extract(text, labels)``), writes predictions
in the R5 format and scores them with ``model_pipeline.evaluate_task_quality`` (span P/R/F1, exact and partial,
micro and macro).

Environment: ``reference/gliner2/requirements-export.txt`` (GLiNER2 at the pinned source commit).

Usage:
    python reference_ner.py --model-dir <gliner2 snapshot> --gold <gold.jsonl> --predictions <out.jsonl> \\
        --scores <out.json> [--threads N]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from gliner2 import GLiNER2
from gliner2.inference.schema import Schema

from model_pipeline import evaluate_task_quality as tq
from model_pipeline import fixtures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    model = GLiNER2.from_pretrained(str(args.model_dir))
    model.eval()
    gold = fixtures.read_jsonl(args.gold)
    preds, offset_mismatch = [], 0
    t0 = time.perf_counter()
    for row in gold:
        result = model.extract(
            row["text"], Schema().entities(row["labels"]), threshold=0.5, include_confidence=True, include_spans=True
        )
        ents = []
        for label, items in result.get("entities", {}).items():
            for e in items:
                if row["text"][e["start"] : e["end"]].strip() != e["text"].strip():
                    offset_mismatch += 1
                ents.append(
                    {
                        "start": e["start"],
                        "end": e["end"],
                        "label": label,
                        "text": e["text"],
                        "confidence": float(e["confidence"]),
                    }
                )
        preds.append({"id": row["id"], "entities": ents})
    elapsed = time.perf_counter() - t0
    fixtures.write_jsonl(args.predictions, preds)
    data = tq.load_ner_set(args.gold)
    metrics = tq.ner_metrics(data, tq.load_predictions(args.predictions, data))
    scores = {
        "dataset_sha256": tq.dataset_digest(data.files),
        "predictions_sha256": fixtures.sha256_file(args.predictions),
        "offset_text_mismatches": offset_mismatch,
        "metrics": metrics,
        "extract_seconds": round(elapsed, 1),
        "environment": fixtures.environment(["torch", "transformers", "tokenizers", "gliner2"]),
    }
    fixtures.write_json(args.scores, scores)
    summary = {k: v for k, v in metrics["metrics"].items() if k.endswith(".f1") and ".label." not in k}
    print(json.dumps({"summary": summary, "seconds": round(elapsed, 1), "offset_mismatch": offset_mismatch}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
