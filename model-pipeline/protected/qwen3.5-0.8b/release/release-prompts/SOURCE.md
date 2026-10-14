# Proposed: P-030 generation release prompts, Qwen3.5-0.8B

Status: **proposed**. Not accepted and not under `model-pipeline/protected/`. Intended destination if accepted: `model-pipeline/protected/qwen3.5-0.8b/release/` (prompts plus the reference fixture built from them), used by `model-pipeline evaluate` for G1/G2 of the three `ortgenai-*` profiles instead of the agent-written `prompts-v1.jsonl`.

## What it is

`prompts-release-v1.jsonl`: 32 single-turn user prompts in the format of `reference/generation/prompts-v1.jsonl` (`id`, `category`, `messages`, `max_new_tokens` 64; the template's default thinking mode). The texts are OpenAssistant oasst1 root prompts, copied verbatim; none was written by an agent.

| Source | Revision | File | License (evidence) |
| --- | --- | --- | --- |
| `OpenAssistant/oasst1` | `fdf72ae0827c1cda404aff25b6603abec9e3399b` | `data/validation-00000-of-00001-134b8fd0c89408b6.parquet` (SHA-256 in `reference/stable/sources.lock.json`) | Apache-2.0 (card front matter; repository `LICENSE`) |

Selection (`reference/stable/select_generation.py`, seed 20261014): root prompts (`role` prompter, no parent), `review_result` true, not deleted, not synthetic, no `pii` label above 0, detoxify toxicity below 0.1, 20 to 1,000 characters; sorted by message id, one seeded shuffle, first 32. The draw is 19 English and 13 Spanish prompts (language in `category`). Multilingual prompt coverage beyond these two languages stays with the template corpus (P-029); a larger draw would add other languages. `selection.json` records the provenance.

## Reference fixture: commands

```sh
cd model-pipeline/reference/generation
<generation venv>/bin/python run_reference.py --model-dir <Qwen/Qwen3.5-0.8B@2fc06364 snapshot> \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 \
  --prompts ../../proposed/stable/qwen3.5-0.8b/release-prompts/prompts-release-v1.jsonl --out <ref run> --device cpu
<generation venv>/bin/python make_generation_fixture.py --reference <ref run> \
  --prompts ../../proposed/stable/qwen3.5-0.8b/release-prompts/prompts-release-v1.jsonl --out <fixture dir>
```

Generated for this proposal: `reference/` (`generation.jsonl`, `environment.json`; format `starlings.generation-fixture` version 1), from a CPU run of about 5 minutes (transformers 5.18.0, torch 2.14.1 CPU, fp32; the Qwen3.5 linear-attention layers ran on transformers' reference PyTorch kernels because `flash-linear-attention` is not installed). Per prompt it keeps the rendered template, prompt ids, greedy output ids, per-step margins and the step-0 top 64; the full step-0 vectors (1 MB each) stay outside Git, as for the development fixture.

All 32 outputs stop at `max_new_tokens` (finish reason `length`). The smallest top-2 logit margin over all greedy steps is 5.7e-4, and six prompts have a step below 0.01: the set exercises the near-tie rule of the fp16x CUDA tolerance, which is what a release set should do.

## Decisions for the reviewer

1. Accept the 32 prompts as the Qwen P-030 release prompts.
2. Keep `max_new_tokens` at 64 (development prompts use 128; 64 keeps a CPU run to minutes while still crossing several near-tie steps).
3. The greedy-ids tolerance for fp16x CUDA (`equal_until_first_divergence_at_margin_below_<m>`) applies unchanged; near-tie divergences are listed, not hidden.
