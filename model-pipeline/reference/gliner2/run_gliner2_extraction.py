"""GLiNER2 entity-extraction reference run, CPU fp32, with stage tensors (backlog P-014).

For each case of the extraction corpus (text, labels, threshold, optional descriptions and word cap
`max_len`), runs the pinned GLiNER2 public API and records:

- the public output of `GLiNER2.extract(text, schema, threshold, include_confidence=True,
  include_spans=True, max_len=...)` and of the plain call (no confidence, no spans), which
  deduplicates repeated surfaces;
- the model inputs GLiNER2 builds (`ExtractorCollator`, inference mode): normalized text, words with
  character maps, schema tokens, input ids, word and schema routing positions;
- the stage tensors at the boundaries a port must reproduce: backbone last hidden state, word and
  schema vectors, count logits and predicted count, span representations, label projections,
  span logits (count row 0, the row GLiNER2 reads for entities);
- the same logits from the exportable restatement in `gliner2_entity.py` (`EntityModel`) and its
  pure-Python decoder, checked against the public output.

Writes `extraction.jsonl` (one row per case, committed as proposed fixtures) and, with
`--tensors-out`, one `.npz` per case (large; not committed). `batch.json` compares padded batches
of the public API with single-case calls.

Usage:
    python run_gliner2_extraction.py --model-dir <snapshot> --revision <sha> \
        --corpus <jsonl> --out <dir> [--tensors-out <dir>] [--threads N] [--categories c1,c2]

`--categories` keeps only the corpus rows of those categories, in corpus order. The P-030 release fixture uses the
Stable proposal's extraction release corpus (`extraction-release-v1.jsonl`, the first 30 rows of each NER subset)
with `--categories wnut17`: the WNUT-17 sentences only (CC BY 4.0), not the Few-NERD ones (CC BY-SA 4.0).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import gliner2
import numpy as np
import torch
from gliner2 import GLiNER2
from gliner2.inference.schema import Schema
from gliner2.training.trainer import ExtractorCollator
from gliner2_entity import EntityModel, decode_entities, threshold_margins
from run_gliner2_tokenizer import ASSETS, gliner2_source_revision

from model_pipeline import fixtures

PACKAGES = ["gliner2", "transformers", "tokenizers", "sentencepiece", "torch", "numpy", "safetensors"]


def build_schema(case: dict) -> Schema:
    if case.get("descriptions"):
        spec = {label: case["descriptions"].get(label, {}) for label in case["labels"]}
        return Schema().entities(spec)
    return Schema().entities(case["labels"])


def f32(values) -> list[float]:
    """float32 values as the shortest decimal that round-trips to the same float32."""
    return [float(str(np.float32(v))) for v in np.asarray(values, dtype=np.float32).reshape(-1)]


def entity_keys(result: dict) -> list:
    """Entity identity without confidences: label, then (text, start, end) in output order."""
    return [(k, [(e["text"], e["start"], e["end"]) for e in v]) for k, v in result.get("entities", {}).items()]


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def run_case(model, case: dict) -> tuple[dict, dict]:
    schema = build_schema(case)
    thr = float(case.get("threshold", 0.5))
    max_len = case.get("max_len")
    public = model.extract(
        case["text"], schema, threshold=thr, include_confidence=True, include_spans=True, max_len=max_len
    )
    plain = model.extract(case["text"], schema, threshold=thr, max_len=max_len)

    collator = ExtractorCollator(model.processor, is_training=False, max_len=max_len, architecture="span")
    batch = collator([(case["text"], schema)])
    assert len(batch.schema_tokens_list[0]) == 1, "one entity schema per case"
    schema_tokens = batch.schema_tokens_list[0][0]
    labels = [schema_tokens[j + 1] for j in range(len(schema_tokens) - 1) if schema_tokens[j] == "[E]"]
    word_pos = batch.text_word_indices[0, : batch.text_word_counts[0]]
    schema_pos = torch.tensor(batch.schema_special_indices[0][0], dtype=torch.long)

    with torch.inference_mode():
        hidden = model.encoder(input_ids=batch.input_ids, attention_mask=batch.attention_mask).last_hidden_state
        token_embs, schema_embs = model.processor.extract_embeddings_from_batch(hidden, batch.input_ids, batch)
        words = token_embs[0]
        embs = torch.stack(schema_embs[0][0])
        count_logits = model.count_pred(embs[0].unsqueeze(0))[0]
        pred_count = int(count_logits.argmax().item())
        span_info = model.compute_span_rep_batched([words])[0]
        struct = model.count_embed(embs[1:], max(pred_count, 1))
        logits_all = torch.einsum("lkd,bpd->bplk", span_info["span_rep"], struct)
        span_logits = logits_all[0]  # [M, W, K]

        wrapper = EntityModel(model).eval()
        w_count, w_logits = wrapper(
            batch.input_ids, batch.attention_mask, word_pos.unsqueeze(0), schema_pos.unsqueeze(0)
        )
    w_count, w_logits = w_count[0].numpy(), w_logits[0].numpy()
    span_logits_np = span_logits.numpy()

    normalized = batch.original_texts[0]
    starts, ends = batch.start_mappings[0], batch.end_mappings[0]
    decode_args = {"labels": labels, "text": normalized, "start_map": starts, "end_map": ends, "threshold": thr}
    decoded = decode_entities(count_logits=count_logits.numpy(), span_logits=span_logits_np, **decode_args)
    decoded_wrapper = decode_entities(count_logits=w_count, span_logits=w_logits, **decode_args)
    n = len(starts)
    k = model.max_width
    valid = (np.arange(n)[:, None] + np.arange(k)[None, :]) < n
    tensors = {
        "input_ids": batch.input_ids.numpy(),
        "attention_mask": batch.attention_mask.numpy(),
        "word_index": word_pos.numpy()[None, :],
        "schema_index": schema_pos.numpy()[None, :],
        "last_hidden_state": hidden.numpy(),
        "words": words.numpy(),
        "schema_vectors": embs.numpy(),
        "count_logits": count_logits.numpy(),
        "span_rep": span_info["span_rep"].numpy(),
        "struct_proj": struct[0].numpy(),
        "span_logits": span_logits_np,
    }
    row = {
        "id": case["id"],
        "category": case["category"],
        "text": case["text"],
        "labels_requested": case["labels"],
        "descriptions": case.get("descriptions"),
        "threshold": thr,
        "max_len": max_len,
        "normalized_text": normalized,
        "words": batch.text_tokens[0],
        "start_map": starts,
        "end_map": ends,
        "schema_tokens": schema_tokens,
        "labels": labels,
        "input_ids": batch.input_ids[0].tolist(),
        "word_index": word_pos.tolist(),
        "schema_index": schema_pos.tolist(),
        "sequence_length": int(batch.input_ids.shape[1]),
        "count_logits": f32(count_logits.numpy()),
        "pred_count": pred_count,
        "gate_open": pred_count > 0,
        "span_logits_shape": list(span_logits_np.shape),
        # Valid region only, row-major over (label, start word, width), invalid spans dropped.
        "span_logits_valid": f32(np.stack([span_logits_np[m][valid] for m in range(len(labels))])),
        "threshold_margins": threshold_margins(span_logits_np, n, thr, k),
        "output": public,
        "output_plain": plain,
        "checks": {
            "wrapper_count_logits_max_abs": float(np.abs(w_count - count_logits.numpy()).max()),
            "wrapper_span_logits_max_abs_valid": float(
                max(
                    (np.abs(w_logits[m][valid] - span_logits_np[m][valid]).max() for m in range(len(labels))),
                    default=0.0,
                )
            ),
            "decoder_matches_public": decoded == public,
            "wrapper_entities_match_public": entity_keys(decoded_wrapper) == entity_keys(public),
        },
        "tensor_sha256": {name: digest(a) for name, a in tensors.items()},
        "last_hidden_state_stats": {
            "shape": list(hidden.shape),
            "max_abs": float(hidden.abs().max()),
            "mean_abs": float(hidden.abs().mean()),
        },
    }
    return row, tensors


def batch_check(model, cases: list[dict], batch_size: int) -> dict:
    """Public API in padded batches (per-case schemas, threshold 0.5, no word cap) against single calls."""
    picked = [c for c in cases if float(c.get("threshold", 0.5)) == 0.5 and c.get("max_len") is None]
    schemas = [build_schema(c) for c in picked]
    batched = model.batch_extract(
        [c["text"] for c in picked],
        schemas,
        batch_size=batch_size,
        threshold=0.5,
        include_confidence=True,
        include_spans=True,
    )
    singles = [
        model.extract(c["text"], s, threshold=0.5, include_confidence=True, include_spans=True)
        for c, s in zip(picked, schemas, strict=True)
    ]
    rows, max_conf = [], 0.0
    for c, b, s in zip(picked, batched, singles, strict=True):
        same_spans = entity_keys(b) == entity_keys(s)
        diff = _conf_diff(b, s) if same_spans else None
        if diff is not None:
            max_conf = max(max_conf, diff)
        rows.append({"id": c["id"], "same_entities": same_spans, "confidence_max_abs": diff})
    return {
        "batch_size": batch_size,
        "cases": len(picked),
        "cases_with_different_entities": [r["id"] for r in rows if not r["same_entities"]],
        "confidence_max_abs": max_conf,
        "rows": rows,
    }


def _conf_diff(a: dict, b: dict) -> float:
    d = 0.0
    for k, v in a.get("entities", {}).items():
        for x, y in zip(v, b["entities"][k], strict=True):
            d = max(d, abs(x["confidence"] - y["confidence"]))
    return d


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tensors-out", type=Path)
    parser.add_argument("--threads", type=int, default=0, help="torch intra-op threads (0: torch default)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--categories", help="comma-separated corpus categories to run (default: every row)")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    model = GLiNER2.from_pretrained(str(args.model_dir))
    model.eval()
    cases = fixtures.load_corpus(args.corpus)
    categories = args.categories.split(",") if args.categories else None
    if categories:
        unknown = set(categories) - {c["category"] for c in cases}
        if unknown:
            raise SystemExit(f"{args.corpus}: no rows of categories {sorted(unknown)}")
        cases = [c for c in cases if c["category"] in categories]

    rows = []
    for case in cases:
        row, tensors = run_case(model, case)
        rows.append(row)
        if args.tensors_out:
            args.tensors_out.mkdir(parents=True, exist_ok=True)
            np.savez(args.tensors_out / f"{case['id']}.npz", **tensors)
        c = row["checks"]
        print(
            f"{case['id']:28s} S={row['sequence_length']:5d} W={len(row['words']):5d} M={len(row['labels'])} "
            f"gate={int(row['gate_open'])} decoder_ok={c['decoder_matches_public']} "
            f"wrapper={c['wrapper_span_logits_max_abs_valid']:.2e} "
            f"min_margin={row['threshold_margins'].get('min_margin')}"
        )
    fixtures.write_jsonl(args.out / "extraction.jsonl", rows)
    fixtures.write_json(args.out / "batch.json", batch_check(model, cases, args.batch_size))
    fixtures.write_json(
        args.out / "environment.json",
        {
            "fixture_format_version": fixtures.FIXTURE_FORMAT_VERSION,
            "runner": "model-pipeline/reference/gliner2/run_gliner2_extraction.py",
            "runner_repository_commit": fixtures.git_head(Path(__file__).parent),
            "model": {"repo": "fastino/gliner2-base-v1", "revision": args.revision},
            "gliner2_source": {
                "url": "https://github.com/fastino-ai/GLiNER2",
                "revision": gliner2_source_revision(),
                "version": gliner2.__version__,
            },
            "entry_point": "GLiNER2.extract(text, Schema().entities(labels), threshold, include_confidence=True, "
            "include_spans=True, max_len=max_len); default overlap policy (span checkpoint: confidence-first greedy)",
            "precision": "fp32",
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "assets": [
                fixtures.file_record(args.model_dir / a, args.model_dir) for a in [*ASSETS, "model.safetensors"]
            ],
            "corpus": fixtures.file_record(args.corpus),
            **({"corpus_categories": categories, "cases": len(rows)} if categories else {}),
            "offset_units": {
                "start_map, end_map, output start/end": "Unicode scalar indices into normalized_text, exclusive end",
                "normalized_text": "text with '.' appended unless it ends in . ! or ? (empty text becomes '.')",
            },
            "environment": fixtures.environment(PACKAGES),
        },
    )
    bad = [r["id"] for r in rows if not r["checks"]["decoder_matches_public"]]
    print(f"cases={len(rows)} decoder_mismatches={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
