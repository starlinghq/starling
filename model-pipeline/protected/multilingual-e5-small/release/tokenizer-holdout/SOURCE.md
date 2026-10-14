# Proposed: P-030 tokenizer holdout, multilingual-e5-small

Status: **proposed**. Not accepted and not under `model-pipeline/protected/`. Intended destination if accepted: `model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/release/` (the tokenizer and encoder release set; the CUDA and browser profiles reuse its rows).

Tokenizer profile: XLM-R SentencePiece unigram (`intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3`).

## What it is

`corpus.jsonl` has 232 rows in the corpus format of `model-pipeline/reference/corpus/` (`id`, `category`, `text`): 208 MASSIVE utterances (4 per locale, 52 locales), 8 HumanEval prompts (Python code) and 16 OpenAssistant assistant replies (Markdown, lists, code, numbers; the longest exceed 512 tokens, so the truncation path is covered). No text was written by an agent: every row is copied verbatim from a pinned public dataset, and the selection is a seeded procedure, not a choice per row. The four profiles' holdouts are disjoint (one seeded shuffle per stratum, dealt round-robin). Rows that appear in any development corpus are excluded.

`holdout.json` records the sources, revisions, file SHA-256 digests, seed, counts and six suggested padded-batch ids (`--batch-ids`, the runners' `BATCH_IDS`).

## Sources and licenses (verified 2026-10-14 at the pinned revision)

| Source | Revision | File | License (evidence) |
| --- | --- | --- | --- |
| `AmazonScience/massive` (Hub Parquet conversion) | `ed58ac423a2f4121720918bf5301577edce4ffd3` | `all_1.1/test/0000.parquet` | CC-BY-4.0 (card front matter; repository `LICENSE`) |
| `openai/openai_humaneval` | `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544` | `openai_humaneval/test-00000-of-00001.parquet` | MIT (card front matter) |
| `OpenAssistant/oasst1` | `fdf72ae0827c1cda404aff25b6603abec9e3399b` | `data/validation-…134b8fd0c89408b6.parquet` | Apache-2.0 (card front matter; repository `LICENSE`) |

All three allow redistributing the excerpt in this repository with attribution. File digests: `model-pipeline/reference/stable/sources.lock.json`. Accepting the set needs a `THIRD-PARTY-NOTICES.md` entry for the three sources (draft in `docs/plan/stable-proposal.md`).

## Reproduce

```sh
python -I model-pipeline/reference/stable/fetch_sources.py --cache <cache> --only massive,humaneval,oasst1
python -I model-pipeline/reference/stable/select_tokenizer_holdouts.py --cache <cache> --out model-pipeline/proposed/stable
```

The output is byte-identical for the same seed (20261014) and pinned files.

## Fixtures

`../onnx-fp32-cpu-release/` holds the runner's output on this corpus (tokenizer.jsonl, tokenizer-batch.json, embeddings.jsonl, parity.json, environment.json), generated for this proposal.

```sh
cd model-pipeline/reference/encoder
<encoder venv>/bin/python run_e5.py --model-dir <e5 snapshot> \
  --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 \
  --corpus ../../proposed/stable/multilingual-e5-small/tokenizer-holdout/corpus.jsonl \
  --out ../../proposed/stable/multilingual-e5-small/onnx-fp32-cpu-release --batch-ids "$BATCH_IDS"
```

## Decisions for the reviewer

1. Accept these texts as the P-030 tokenizer holdout for this profile (plan 10.1: licensed public data, not agent-written).
2. Whether the holdout is also the text-encoder release set (it is for e5 and EmbeddingGemma 2 here: the runner embeds every row).
3. Rotate: once a failure on this set has been debugged against it, treat it as development data and draw a new set with another seed.
