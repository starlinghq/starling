"""Reference runner for google/embeddinggemma-2, text path only (EmbeddingGemma 2 admission spike). CPU, fp32.

Produces, from a local snapshot pinned by Hub commit SHA:

- tokenizer.jsonl: per corpus case, the `tokenizers` encoding of the pinned `tokenizer.json`
  (ids, tokens, type ids, masks, offsets in three declared units, decode). No truncation: the
  model card's context window is 8,192 tokens and no corpus case reaches it.
- tokenizer-batch.json: one right-padded batch (pad id 0) and a row-by-row equivalence check.
- tokenizer-implementations.json: transformers' live tokenizer (`AutoTokenizer`, the class
  `tokenizer_config.json` names) against `tokenizer.json`, field by field and per case, and
  sentence-transformers' own preprocessing (chat-template rendering) against the
  `tokenizer.json` ids of `prompt + text`. Only differences are listed.
- embeddings.jsonl: sentence-transformers embeddings (the reference), one row per single-text
  case and prompt (none, `query`, `document`), with the ids they were computed from.
- reference.json: stability of the reference itself (padded batches, sdpa against eager
  attention, text-only loading, thread count, end-to-end `encode`).
- environment.json: interpreter, packages, CPU, asset digests.

The model's `config.json` declares bfloat16 and transformers 5 loads that dtype by default; the
reference passes `dtype=float32` explicitly (bf16 weights upcast exactly to fp32).

Usage:
    python run_embeddinggemma2.py --model-dir <snapshot> --revision <sha> --corpus <jsonl> --out <dir>
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from model_pipeline import fixtures

REPO = "google/embeddinggemma-2"
CONTEXT = 8192
PAD_ID = 0
PAD_TOKEN = "<pad>"
# Prompts from config_sentence_transformers.json at the pinned revision (prompt names `query` and `document`).
PROMPTS = {"none": "", "query": "task: search result | query: ", "document": "title: none | text: "}
PACKAGES = [
    "torch",
    "transformers",
    "sentence-transformers",
    "tokenizers",
    "onnxruntime",
    "onnx",
    "onnxscript",
    "numpy",
    "safetensors",
    "pillow",
    "torchvision",
]
ASSETS = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "chat_template.jinja",
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
    "2_Normalize/config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "model.safetensors",
]
BATCH_IDS = ["empty", "ascii-sentence", "emoji-zwj", "chinese", "accents-decomposed", "long-mixed"]


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
    rows = []
    for case in cases:
        text, pair = case["text"], case.get("pair")
        enc = tok.encode(text, pair) if pair is not None else tok.encode(text)
        if len(enc.ids) > CONTEXT:
            raise SystemExit(f"{case['id']}: {len(enc.ids)} tokens exceeds the {CONTEXT}-token context")
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


def json_diff(a, b, path="") -> list[str]:
    """Paths where two parsed JSON values differ (vocabulary and merges compared as whole values)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only in {'live' if k in a else 'file'}")
            elif k in ("vocab", "merges"):
                if a[k] != b[k]:
                    out.append(f"{path}/{k}: differs")
            else:
                out.extend(json_diff(a[k], b[k], f"{path}/{k}"))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out.extend(json_diff(x, y, f"{path}[{i}]"))
        return out
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def implementations(model_dir: Path, tok: Tokenizer, model: SentenceTransformer, cases: list[dict]) -> dict:
    live = AutoTokenizer.from_pretrained(str(model_dir))
    live_json = json.loads(live.backend_tokenizer.to_str())
    file_json = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    pipeline_diff = json_diff(live_json, file_json)
    live_ids_diff = []
    for case in cases:
        text, pair = case["text"], case.get("pair")
        ref = tok.encode(text, pair).ids if pair is not None else tok.encode(text).ids
        got = live(text, pair)["input_ids"] if pair is not None else live(text)["input_ids"]
        if got != ref:
            live_ids_diff.append({"id": case["id"], "tokenizer_json": ref, "transformers": got})
    st_diff = []
    for case in cases:
        if "pair" in case:
            continue
        for name, prompt in PROMPTS.items():
            ref = tok.encode(prompt + case["text"]).ids
            kwargs = {"prompt": prompt} if prompt else {}
            got = model.preprocess([case["text"]], **kwargs)["input_ids"][0].tolist()
            if got != ref:
                st_diff.append({"id": case["id"], "prompt": name, "tokenizer_json": ref, "sentence_transformers": got})
    return {
        "reference": "tokenizers library reading tokenizer.json as is",
        "transformers_class": type(live).__name__,
        "transformers_pipeline_vs_tokenizer_json": pipeline_diff,
        "transformers_ids_differ": live_ids_diff,
        "sentence_transformers_preprocess": "SentenceTransformer.preprocess (message modality: "
        "chat_template.jinja renders the prompt as a system message followed by the text, "
        "then the processor tokenizes with special tokens)",
        "sentence_transformers_ids_differ_from_tokenizer_json_of_prompt_plus_text": st_diff,
    }


def forward(model: SentenceTransformer, id_rows: list[list[int]]) -> np.ndarray:
    width = max(len(r) for r in id_rows)
    ids = torch.tensor([r + [PAD_ID] * (width - len(r)) for r in id_rows])
    mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in id_rows])
    with torch.inference_mode():
        return model({"input_ids": ids, "attention_mask": mask})["sentence_embedding"].numpy()


def stats(diff: np.ndarray) -> dict:
    a = np.abs(diff).ravel()
    return {"max_abs": float(a.max()), "mean_abs": float(a.mean()), "p99_abs": float(np.quantile(a, 0.99))}


def load(model_dir: Path, attn: str | None = None, text_only: bool = False) -> SentenceTransformer:
    model_kwargs = {"dtype": torch.float32}
    if attn:
        model_kwargs["attn_implementation"] = attn
    config_kwargs = {"vision_config": None, "audio_config": None} if text_only else None
    model = SentenceTransformer(str(model_dir), device="cpu", model_kwargs=model_kwargs, config_kwargs=config_kwargs)
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    # The release set's reference (protected/embeddinggemma-2/release/onnx-fp32-release/SOURCE.md) was produced with
    # this option (runner commit 4b90bc9a, environment.json); it changes only the padded-batch check's rows.
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

    # 2. sentence-transformers reference (fp32, CPU, default sdpa attention, full checkpoint as upstream loads it).
    model = load(args.model_dir)
    inner = model[0].model
    assert next(inner.parameters()).dtype == torch.float32
    fixtures.write_json(args.out / "tokenizer-implementations.json", implementations(args.model_dir, tok, model, cases))

    single_cases = [c for c in cases if "pair" not in c]
    emb_rows, e2e = [], []
    for case in single_cases:
        for name, prompt in PROMPTS.items():
            ids = tok.encode(prompt + case["text"]).ids
            emb = forward(model, [ids])[0]
            kwargs = {"prompt_name": name} if prompt else {}
            end_to_end = model.encode([case["text"]], batch_size=1, convert_to_numpy=True, **kwargs)[0]
            e2e.append(float(np.abs(end_to_end - emb).max()))
            emb_rows.append(
                {
                    "id": case["id"],
                    "prompt": name,
                    "prompt_text": prompt,
                    "text": case["text"],
                    "input_ids": ids,
                    "embedding": fixtures.float32_list(emb),
                    "norm": float(np.linalg.norm(emb)),
                }
            )
    fixtures.write_jsonl(args.out / "embeddings.jsonl", emb_rows)

    # 3. Reference stability.
    singles = np.stack([np.array(r["embedding"], dtype=np.float32) for r in emb_rows])
    all_ids = [r["input_ids"] for r in emb_rows]
    batched = np.concatenate([forward(model, all_ids[s : s + 16]) for s in range(0, len(all_ids), 16)])
    default_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    one_thread = np.stack([forward(model, [ids])[0] for ids in all_ids[::3]])
    torch.set_num_threads(default_threads)
    del model
    eager = load(args.model_dir, attn="eager")
    eager_rows = np.stack([forward(eager, [ids])[0] for ids in all_ids])
    del eager
    text_only = load(args.model_dir, text_only=True)
    text_only_rows = np.stack([forward(text_only, [ids])[0] for ids in all_ids[::3]])
    del text_only
    nonfinite = [r["id"] for r in emb_rows if not np.isfinite(np.array(r["embedding"])).all()]
    reference = {
        "reference": "sentence-transformers modules (Transformer, Pooling, Normalize) on the tokenizer.json ids of "
        "prompt + text, PyTorch fp32 CPU, sdpa attention, one row per call, full checkpoint",
        "rows": len(emb_rows),
        "prompts": PROMPTS,
        "max_tokens": max(len(x) for x in all_ids),
        "padded_batch16_vs_single": stats(batched - singles),
        "eager_attention_vs_sdpa": stats(eager_rows - singles),
        "text_only_load_vs_full_every_third_row": stats(text_only_rows - singles[::3]),
        "one_thread_vs_default_every_third_row": stats(one_thread - singles[::3]),
        "default_threads": default_threads,
        "encode_end_to_end_vs_reference_max_abs": max(e2e),
        "nan_or_inf_rows": nonfinite,
    }
    fixtures.write_json(args.out / "reference.json", reference)

    env = fixtures.environment(PACKAGES)
    env["torch_threads"] = default_threads
    fixtures.write_json(
        args.out / "environment.json",
        {
            "fixture_format_version": fixtures.FIXTURE_FORMAT_VERSION,
            "runner": "model-pipeline/reference/embeddinggemma2/run_embeddinggemma2.py",
            "runner_repository_commit": fixtures.git_head(Path(__file__).parent),
            "model": {"repo": REPO, "revision": args.revision},
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
    print(json.dumps({k: v for k, v in reference.items() if k not in ("prompts", "reference")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
