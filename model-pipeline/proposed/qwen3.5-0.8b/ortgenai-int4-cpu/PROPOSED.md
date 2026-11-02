# Proposed fixture: int4 CPU reference (prompts-v1 through the Python binding)

Status: **proposed**. It is not accepted and not under `model-pipeline/protected/`. If the maintainer accepts it, the intended destination is `model-pipeline/protected/qwen3.5-0.8b/ortgenai-int4-cpu/`, with these files byte-identical, next to a `tolerances.yaml` and a `task-quality.yaml` that the maintainer writes (issue #163, step 2). `Int4CpuModelTests` reads the protected copy when it exists and this proposal otherwise.

The fixture is the reference of the Experimental profile `qwen3.5-0.8b/ortgenai-int4-cpu`: `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17`, built by the ONNX Runtime GenAI 0.17.1 builder with `-p int4 -e cpu` from the float32 config overlay. It uses the format of the accepted `reference/generation.jsonl` (`starlings.generation-fixture` version 1).

## Reference source

1. `model-pipeline/reference/generation/build_ortgenai_bundle.py --variant int4` built the graph. Digests: `model.onnx.data` `709e0879…f9745` (541,523,968 bytes, byte-identical to the M0 int4 CPU build), `model.onnx` `a41f341f…39cf7`, `genai_config.json` `e7e1957b…a81ba`. Two builds gave the same digests.
2. `model-pipeline/reference/generation/run_genai_reference.py run --threads 6`, in the environment of `requirements.txt` (onnxruntime-genai 0.17.1 and onnxruntime 1.30.0 from pip, Python 3.12.14), on the CPU provider. It fed the prompt ids of the accepted `model-pipeline/protected/qwen3.5-0.8b/reference/generation.jsonl`. Runtime settings are those of the C# runtime: `pad_token_id` 248319, greedy, 1,024-token prefill calls, `max_length` = prompt + `max_new_tokens`.
3. `run_genai_reference.py annotate` re-rendered all 22 prompts with the snapshot's transformers 5.18.0 tokenizer and chat template. Renderings and ids equal the accepted fixture on all 22. It also decoded `output_text`.
4. `make_generation_fixture.py`, unchanged, wrote `generation.jsonl` and `environment.json`. The repository commit recorded there is the branch's work-in-progress commit with the tree dirty: the fixture files themselves were not yet committed. One field was edited by hand afterwards: `reference_run.prompt_ids_from` named the absolute path of a temporary worktree and now names the repository-relative path of the same file (its `prompt_ids_from_sha256` is unchanged).

Three `run`s at 6 threads gave identical output ids, log-probabilities, margins and step-0 vectors (byte for byte) on all 22 prompts; only timings and the annotation-only `output_text` differ. A 4-thread run, identical on the 11 prompts it completed, is in the evidence README.

## Contents

| File | Contents |
| --- | --- |
| `generation.jsonl` | 22 cases with the fields of the accepted fixture: prompt, rendering and prompt ids (equal to the accepted fixture's), greedy output ids including the stop token (13 end at the length limit, 9 at a stop token), finish reason, output text, per-step chosen log-probability and top-1 minus top-2 margin, and the 64 most likely step-0 log-probabilities. |
| `environment.json` | Provenance, the step-0 full-vector digests (the vectors stay outside Git), and the reference run: graph digests, overlays, threads, packages, and the annotation step's snapshot digests. |

## Result at the time of the proposal

The .NET runtime (`Starlings.Qwen35.Fidelity --variant int4`, 6 threads) produced 22/22 identical greedy sequences, and its step-0 log-probabilities are bit-identical over the full vocabulary. Cached versus uncached decoding differs by up to 0.456 in log-probability on int4, with equal tokens. See `docs/evidence/qwen35-int4-cpu-2026-11-01/README.md`.

## Decisions for the reviewer

1. Accept a quantized profile's reference that is the same graph run by the upstream Python binding, instead of the transformers fp32 reference. Against fp32, int4 matches 1 of 22 sequences, so it cannot be judged on that reference.
2. Set the int4 bounds. Candidate values and the observations behind them are in the evidence README, section 6. No tolerance file is proposed here. The cached-versus-uncached bound needs an int4-specific value, because fp32x's 2e-4 does not hold for this graph.
3. Decide the task-quality floor. GSM8K was not measured on int4 in this step.
4. The prompt set is the accepted development set (prompts-v1). This is not a release set.
