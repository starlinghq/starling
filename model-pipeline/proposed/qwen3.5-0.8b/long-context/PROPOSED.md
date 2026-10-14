# Proposed fixture: long-context generation (prompts-long-v1)

Status: **proposed**. Not accepted, and not under `model-pipeline/protected/`. If the maintainer accepts it, the intended destination is `model-pipeline/protected/qwen3.5-0.8b/long-context/`, with the files byte-identical. `CpuModelTests.LongContextGreedyAndLogitsEqualTheReference` reads the protected copy when it exists and this proposal otherwise.

This fixture covers prompts above the 1,666-token prompt of `prompts-v1.jsonl`, for the capability `long_context.over_1666_tokens` of `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17`. It uses the same format as the accepted `reference/generation.jsonl` (`starlings.generation-fixture` version 1).

## Reference source

1. `model-pipeline/reference/generation/make_long_prompts.py` wrote `prompts-long-v1.jsonl` (SHA-256 `bb1eb0b0b34f4c4c005ad69ef19c47e610eaac756169c5b9d1eb9d5b2a3d4619`). The file has three single-message prompts of 4,112, 8,213 and 16,391 tokens. Each is a prefix of the repository's documentation at `fb27658`, followed by one question; the sources are listed in the script's invocation in `docs/evidence/r8-qwen35-long-context/README.md`. Each prompt asks for 32 new tokens.
2. `model-pipeline/reference/generation/run_reference.py --device cpu --threads 6`, in the environment of `requirements.txt` (transformers 5.18.0, torch 2.14.1+cu130, fp32, TF32 off, greedy), on the pinned snapshot. The snapshot digests in `environment.json` are equal to those of the accepted P-016 reference.
3. `model-pipeline/reference/generation/make_generation_fixture.py`, unchanged, produced `generation.jsonl` and `environment.json`.

The reference run was killed (SIGKILL, under machine-wide memory pressure) on a fourth prompt of 32,844 tokens, after it had written the three results kept here. Because the runner writes its `environment.json` only at the end, that file was rebuilt with the runner's own fields: packages read from the same environment, and snapshot digests recomputed. The `note` field records this. The 32,844-token prompt was removed from the prompt file, and no reference exists for it.

## Contents

| File | Contents |
| --- | --- |
| `generation.jsonl` | 3 cases with the fields of the accepted fixture: rendering, prompt ids, 32 greedy output ids (all three end at the length limit), output text, per-step chosen log-probability and top-1 minus top-2 margin, and the 64 most likely step-0 log-probabilities. |
| `environment.json` | Provenance (repository commit `fb27658`, with the tree dirty from the R8 working changes), the prompt digest, the step-0 full-vector digests, and the reference environment. |

## Result at the time of the proposal

fp32x on CPU: exact greedy ids on all three cases, and the step-0 and top-5 log-probability differences are within the protected bounds. Cached versus uncached logits are within `2e-4` at 4,112 and 8,213 tokens but **reach 4.1e-4 at 16,391 tokens**; see `docs/evidence/r8-qwen35-long-context/README.md`. The CPU test checks the 4,112-token case only, so the test suite stays within CPU time.

## Decisions for the reviewer

1. Accept the prompts and the fixture as development data. The prompts were written by an agent from repository documentation.
2. Decide what to do with the cached-versus-uncached bound at long context: keep `2e-4` and record long context as a known exception, or set a length-dependent bound. The proposal does not change `tolerances.yaml`.
