# Proposed: P-030 tokenizer holdout, qwen3.5-0.8b

Status: **proposed**. Not accepted and not under `model-pipeline/protected/`. Intended destination if accepted: `model-pipeline/protected/qwen-bpe/chat/tokenizer/release/`.

Tokenizer profile: Qwen byte-level BPE, chat profile `qwen35-bpe-chat` (`Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`, asset `transformers-5.18.0/tokenizer.json`).

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

`../tokenizer-release/` holds the runner's output on this corpus (tokenizer.jsonl, live-pipeline.json, environment.json), generated for this proposal: 232 cases, 0 mismatches between the `tokenizers` pipeline and the transformers cross-check.

```sh
cd model-pipeline/reference/qwen-bpe
<qwen-chat venv>/bin/python run_qwen_bpe_tokenizer.py --profile chat \
  --snapshot <Qwen_Qwen3.5-0.8B@2fc06364> --asset <snapshot>/transformers-5.18.0/tokenizer.json \
  --corpus ../../proposed/stable/qwen3.5-0.8b/tokenizer-holdout/corpus.jsonl --out ../../proposed/stable/qwen3.5-0.8b/tokenizer-release
```

## Decisions for the reviewer

1. Accept these texts as the P-030 tokenizer holdout for this profile (plan 10.1: licensed public data, not agent-written).
2. Whether the holdout is also the text-encoder release set (it is for e5 and EmbeddingGemma 2 here: the runner embeds every row).
3. Rotate: once a failure on this set has been debugged against it, treat it as development data and draw a new set with another seed.
