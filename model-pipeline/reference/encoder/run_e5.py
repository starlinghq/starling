"""Reference runner for intfloat/multilingual-e5-small (backlog P-003). CPU, fp32.

Produces, from a local snapshot pinned by Hub commit SHA:

- tokenizer.jsonl: per corpus case, the `tokenizers` encoding of the pinned `tokenizer.json`
  (ids, tokens, type ids, masks, offsets in three declared units, decode), untruncated and
  truncated to 512 tokens.
- tokenizer-batch.json: one right-padded batch and a row-by-row equivalence check.
- embeddings.jsonl: sentence-transformers embeddings (the reference), one per case.
- parity.json: sentence-transformers (PyTorch) against Python ONNX Runtime on `onnx/model.onnx`,
  same token ids, for last hidden states and pooled embeddings.
- environment.json: interpreter, packages, CPU, asset digests.

Usage:
    python run_e5.py --model-dir <snapshot> --revision <sha> --corpus <jsonl> --out <dir>
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer

from model_pipeline import fixtures

MAX_LENGTH = 512
PAD_ID = 1
PAD_TOKEN = "<pad>"
PACKAGES = ["torch", "transformers", "sentence-transformers", "tokenizers", "onnxruntime", "numpy", "sentencepiece"]
ASSETS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
    "config.json",
    "modules.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
    "model.safetensors",
    "onnx/model.onnx",
    "onnx/tokenizer.json",
]
BATCH_IDS = ["empty", "ascii-sentence", "emoji-zwj", "chinese", "accents-decomposed", "e5-query-prefix"]


def encoding_record(text: str, enc, pair: str | None) -> dict:
    record = {
        "ids": enc.ids,
        "tokens": enc.tokens,
        "type_ids": enc.type_ids,
        "attention_mask": enc.attention_mask,
        "special_tokens_mask": enc.special_tokens_mask,
    }
    if pair is None:
        record["offsets"] = fixtures.all_offset_units(text, enc.offsets)
    else:
        # Offsets of a pair index into the sequence the token came from (sequence_ids).
        seq = enc.sequence_ids
        record["sequence_ids"] = seq
        per_unit = {unit: [] for unit in fixtures.OFFSET_UNITS}
        for (s, e), sid in zip(enc.offsets, seq, strict=True):
            source = pair if sid == 1 else text
            for unit in fixtures.OFFSET_UNITS:
                per_unit[unit].append(fixtures.convert_offsets(source, [(s, e)], unit)[0])
        record["offsets"] = per_unit
    return record


def tokenizer_fixtures(tok: Tokenizer, cases: list[dict]) -> list[dict]:
    truncating = copy.deepcopy(tok)
    truncating.enable_truncation(MAX_LENGTH)
    rows = []
    for case in cases:
        text, pair = case["text"], case.get("pair")
        enc = tok.encode(text, pair) if pair is not None else tok.encode(text)
        row = {
            "id": case["id"],
            "category": case["category"],
            "text": text,
            "encoding": encoding_record(text, enc, pair),
            "decode": {
                "skip_special_tokens": tok.decode(enc.ids, skip_special_tokens=True),
                "keep_special_tokens": tok.decode(enc.ids, skip_special_tokens=False),
            },
        }
        if pair is not None:
            row["pair"] = pair
        if len(enc.ids) > MAX_LENGTH:
            tenc = truncating.encode(text, pair) if pair is not None else truncating.encode(text)
            row["truncated"] = {
                "max_length": MAX_LENGTH,
                "strategy": "longest_first",
                "direction": "right",
                "encoding": encoding_record(text, tenc, pair),
            }
        rows.append(row)
    return rows


def batch_fixture(tok: Tokenizer, cases_by_id: dict[str, dict]) -> dict:
    padded = copy.deepcopy(tok)
    padded.enable_padding(pad_id=PAD_ID, pad_token=PAD_TOKEN, direction="right")
    texts = [cases_by_id[i]["text"] for i in BATCH_IDS]
    batch = padded.encode_batch(texts)
    rows, equivalent = [], True
    for case_id, text, enc in zip(BATCH_IDS, texts, batch, strict=True):
        single = tok.encode(text)
        n = len(single.ids)
        same = (
            enc.ids[:n] == single.ids
            and all(i == PAD_ID for i in enc.ids[n:])
            and enc.attention_mask == [1] * n + [0] * (len(enc.ids) - n)
        )
        equivalent &= same
        rows.append({"id": case_id, "ids": enc.ids, "attention_mask": enc.attention_mask, "matches_single": same})
    return {
        "padding": {"side": "right", "pad_id": PAD_ID, "strategy": "longest"},
        "rows": rows,
        "equivalent": equivalent,
    }


def st_ids(model: SentenceTransformer, text: str) -> list[int]:
    features = model.tokenize([text])
    return features["input_ids"][0].tolist()


def pooled(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean pooling over the attention mask, then L2 normalization, in float32 (ST Pooling + Normalize)."""
    m = mask[..., None].astype(np.float32)
    summed = (hidden * m).sum(axis=1)
    counts = np.clip(m.sum(axis=1), 1e-9, None)
    mean = summed / counts
    norms = np.linalg.norm(mean, axis=1, keepdims=True)
    return mean / np.clip(norms, 1e-12, None)


def stats(diff: np.ndarray) -> dict:
    a = np.abs(diff).ravel()
    return {
        "max_abs": float(a.max()) if a.size else 0.0,
        "mean_abs": float(a.mean()) if a.size else 0.0,
        "p99_abs": float(np.quantile(a, 0.99)) if a.size else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--batch-ids",
        help="comma-separated corpus ids for the padded-batch check (default: the development corpus ids)",
    )
    args = parser.parse_args()
    if args.batch_ids:
        BATCH_IDS[:] = args.batch_ids.split(",")

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    cases = fixtures.load_corpus(args.corpus)
    by_id = {c["id"]: c for c in cases}
    tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))

    # 1. Tokenizer fixtures from the pinned tokenizer.json.
    tok_rows = tokenizer_fixtures(tok, cases)
    fixtures.write_jsonl(args.out / "tokenizer.jsonl", tok_rows)
    fixtures.write_json(args.out / "tokenizer-batch.json", batch_fixture(tok, by_id))

    # 2. sentence-transformers reference (fp32, CPU).
    model = SentenceTransformer(str(args.model_dir), device="cpu")
    model.eval()
    assert model.max_seq_length == MAX_LENGTH, model.max_seq_length
    session = ort.InferenceSession(str(args.model_dir / "onnx" / "model.onnx"), providers=["CPUExecutionProvider"])

    single_cases = [c for c in cases if "pair" not in c]
    tokenizer_mismatches = []
    emb_rows = []
    hidden_diffs, emb_diffs, numpy_pool_vs_st, e2e_diffs = [], [], [], []
    per_case = []
    for case in single_cases:
        text = case["text"]
        row = next(r for r in tok_rows if r["id"] == case["id"])
        # The declared profile: ids from the pinned tokenizer.json, truncated to 512.
        ids = row.get("truncated", row)["encoding"]["ids"]
        input_ids = np.array([ids], dtype=np.int64)
        mask = np.ones_like(input_ids)
        types = np.zeros_like(input_ids)
        features = {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(mask),
            "token_type_ids": torch.from_numpy(types),
        }
        with torch.inference_mode():
            out = model(features)
        st_emb = out["sentence_embedding"][0].numpy()
        pt_hidden = out["token_embeddings"].numpy()

        # sentence-transformers end to end, with its own tokenizer class (cross-check only).
        e2e_ids = st_ids(model, text)
        e2e_emb = model.encode([text], batch_size=1, convert_to_numpy=True)[0]
        if e2e_ids != ids:
            tokenizer_mismatches.append({"id": case["id"], "tokenizer_json": ids, "sentence_transformers": e2e_ids})
        else:
            e2e_diffs.append(float(np.abs(e2e_emb - st_emb).max()))

        ort_hidden = session.run(
            ["last_hidden_state"], {"input_ids": input_ids, "attention_mask": mask, "token_type_ids": types}
        )[0]
        ort_emb = pooled(ort_hidden, mask)[0]
        pt_emb = pooled(pt_hidden, mask)[0]
        h = stats(ort_hidden - pt_hidden)
        e = stats(ort_emb - st_emb)
        hidden_diffs.append(np.abs(ort_hidden - pt_hidden).ravel())
        emb_diffs.append(np.abs(ort_emb - st_emb))
        numpy_pool_vs_st.append(float(np.abs(pt_emb - st_emb).max()))
        per_case.append(
            {
                "id": case["id"],
                "tokens": len(ids),
                "hidden_max_abs": h["max_abs"],
                "embedding_max_abs": e["max_abs"],
                "cosine_ort_st": float(np.dot(ort_emb, st_emb) / (np.linalg.norm(ort_emb) * np.linalg.norm(st_emb))),
                "nan_or_inf": bool(not np.isfinite(ort_hidden).all() or not np.isfinite(st_emb).all()),
            }
        )
        emb_rows.append(
            {
                "id": case["id"],
                "input_ids": ids,
                "embedding": fixtures.float32_list(st_emb),
                "norm": float(np.linalg.norm(st_emb)),
            }
        )
    fixtures.write_jsonl(args.out / "embeddings.jsonl", emb_rows)

    # 3. Batch equivalence: the same ids padded into batches of 16 through the ST modules.
    singles = np.stack([np.array(r["embedding"], dtype=np.float32) for r in emb_rows])
    batched_parts = []
    for start in range(0, len(emb_rows), 16):
        chunk = [r["input_ids"] for r in emb_rows[start : start + 16]]
        w = max(len(x) for x in chunk)
        ids_t = torch.tensor([x + [PAD_ID] * (w - len(x)) for x in chunk])
        mask_t = torch.tensor([[1] * len(x) + [0] * (w - len(x)) for x in chunk])
        with torch.inference_mode():
            batched_parts.append(
                model({"input_ids": ids_t, "attention_mask": mask_t, "token_type_ids": torch.zeros_like(ids_t)})[
                    "sentence_embedding"
                ].numpy()
            )
    batched = np.concatenate(batched_parts)
    # ORT batch on the padded tokenizer output for the batch fixture rows.
    batch_fx = fixtures.read_jsonl(args.out / "tokenizer.jsonl")
    padded_ids = [next(r for r in batch_fx if r["id"] == i)["encoding"]["ids"] for i in BATCH_IDS]
    width = max(len(x) for x in padded_ids)
    b_ids = np.array([x + [PAD_ID] * (width - len(x)) for x in padded_ids], dtype=np.int64)
    b_mask = np.array([[1] * len(x) + [0] * (width - len(x)) for x in padded_ids], dtype=np.int64)
    b_hidden = session.run(
        ["last_hidden_state"], {"input_ids": b_ids, "attention_mask": b_mask, "token_type_ids": np.zeros_like(b_ids)}
    )[0]
    b_emb = pooled(b_hidden, b_mask)
    ref_rows = np.stack([singles[[c["id"] for c in single_cases].index(i)] for i in BATCH_IDS])

    all_hidden = np.concatenate(hidden_diffs)
    all_emb = np.concatenate(emb_diffs)
    parity = {
        "reference": "sentence-transformers modules (Transformer, Pooling, Normalize; PyTorch fp32 CPU), "
        "one row per call",
        "candidate": "Python ONNX Runtime, CPUExecutionProvider, onnx/model.onnx from the same snapshot, fp32",
        "inputs": "identical token ids from the pinned tokenizer.json (tokenizers library), truncated to 512; "
        "token_type_ids all 0",
        "cases": len(single_cases),
        "last_hidden_state_ort_vs_torch": {
            "max_abs": float(all_hidden.max()),
            "mean_abs": float(all_hidden.mean()),
            "p99_abs": float(np.quantile(all_hidden, 0.99)),
        },
        "embedding_ort_vs_st": {
            "max_abs": float(all_emb.max()),
            "mean_abs": float(all_emb.mean()),
            "p99_abs": float(np.quantile(all_emb, 0.99)),
            "min_cosine": min(p["cosine_ort_st"] for p in per_case),
        },
        "embedding_numpy_pooling_on_torch_hidden_vs_st_max_abs": max(numpy_pool_vs_st),
        "st_padded_batch16_vs_single": stats(batched - singles),
        "st_encode_end_to_end_vs_reference_max_abs_where_ids_agree": max(e2e_diffs),
        "ort_padded_batch_vs_st_batch1": stats(b_emb - ref_rows),
        "st_tokenizer_differs_from_tokenizer_json": tokenizer_mismatches,
        "nan_or_inf_cases": [p["id"] for p in per_case if p["nan_or_inf"]],
        "per_case": per_case,
    }
    fixtures.write_json(args.out / "parity.json", parity)

    env = fixtures.environment(PACKAGES)
    env["torch_threads"] = torch.get_num_threads()
    env["onnxruntime_providers"] = session.get_providers()
    fixtures.write_json(
        args.out / "environment.json",
        {
            "fixture_format_version": fixtures.FIXTURE_FORMAT_VERSION,
            "runner": "model-pipeline/reference/encoder/run_e5.py",
            "runner_repository_commit": fixtures.git_head(Path(__file__).parent),
            "model": {"repo": "intfloat/multilingual-e5-small", "revision": args.revision},
            "assets": [fixtures.file_record(args.model_dir / a, args.model_dir) for a in ASSETS],
            "corpus": fixtures.file_record(args.corpus),
            "offset_units": {
                "source": "tokenizers Python binding, indices into the Python str (Unicode scalar values)",
                "recorded": list(fixtures.OFFSET_UNITS),
                "end": "exclusive",
                "special_tokens": "(0, 0)",
            },
            "environment": env,
        },
    )
    print(
        f"cases={len(cases)} hidden_max={parity['last_hidden_state_ort_vs_torch']['max_abs']:.3g} "
        f"emb_max={parity['embedding_ort_vs_st']['max_abs']:.3g} st_tok_diff={len(tokenizer_mismatches)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
