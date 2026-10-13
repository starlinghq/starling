# Accepted fixtures

Status: **accepted** as development fixtures by Mark on 2026-10-10 (PR #32); binding tolerances per profile in `../ortgenai-*/tolerances.yaml`. Development fixtures for the pilot LLM `Qwen/Qwen3.5-0.8B` at `2fc06364715b967f1860aea9cf38778875588b17` (backlog P-016), shared by the profiles `qwen3.5-0.8b/ortgenai-fp32x-cpu`, `ortgenai-fp32x-cuda` and `ortgenai-fp16x-cuda` (`model-pipeline/registry/qwen3.5-0.8b/`). Not a release evaluation set: the prompt set was written by an agent for the M0 generation-backend check; the protected release set is P-030.

Generated in two steps, both at repository commit `b8b8cbccf4ad55275725b18dce36dcdf9d03ada0` (clean tree):

1. `model-pipeline/reference/generation/run_reference.py --device cpu --threads 8` on `model-pipeline/reference/generation/prompts-v1.jsonl` (22 prompts, SHA-256 `7b236f0d013ce1057f075f7d2d941dde836618ba0f848d6a123e0f86c4b928d2`), in the environment of `model-pipeline/reference/generation/requirements.txt` (torch 2.14.1+cu130, transformers 5.18.0, tokenizers 0.23.2, Python 3.12.14): `Qwen3_5ForCausalLM` in fp32 on CPU, TF32 off, greedy (`do_sample=False`), stop ids 248046 and 248044, `max_new_tokens` per prompt (128), the prompt rendered with the checkpoint's chat template (`apply_chat_template`, `add_generation_prompt=True`, `chat_template_kwargs` from the prompt) and tokenized by the tokenizer transformers builds. The run is bit-identical to the M0 reference of `docs/m0/generation-backend-check.md` (same output ids, renderings and prompt ids on all 22 prompts; step-0 log-softmax vectors equal to the last bit).
2. `model-pipeline/reference/generation/make_generation_fixture.py`, which copies the prompt, the rendering, prompt and output ids, finish reason, decoded text, per-step chosen log-probability and top-1 minus top-2 logit margin, and the 64 most likely step-0 log-probabilities into `generation.jsonl` (fixture format `starlings.generation-fixture` version 1). The full step-0 vectors (22 x 248,320 float32) stay outside Git; `environment.json` records their SHA-256.

| File | Contents |
| --- | --- |
| `generation.jsonl` | 22 cases: `id`, `category`, `messages`, `tools`, `chat_template_kwargs`, `max_new_tokens`, `rendered`, `prompt_ids`, `output_ids` (including the stop token when one was generated), `finish_reason` (`eos` or `length`), `output_text` (`skip_special_tokens=False`), `step_logprob`, `step_margin`, `step0_top` (64 `[id, log-probability]` pairs), `step0_vocab`, `step0_sha256`. |
| `environment.json` | Fixture format and provenance (repository commit, prompts digest, step-0 vector digests) and the reference run's environment (packages, device, threads, snapshot file digests). |

Byte-identical to the copy the C# tests read until this directory exists (`ai-dotnet/tests/Starlings.Models.Qwen35.Tests/TestData/generation-v1/generation.jsonl`).

What the fixture covers (plan 10.1, LLM list): the prompt template (21 text prompts rendered by the owned C# renderer, and the tools prompt that it rejects), tokenizer ids at the backend boundary (22 renderings through the owned tokenizer), greedy token ids up to the stop token or 128 tokens (12 prompts end at the length limit, 10 at `<|im_end|>`), step-0 logits (top 64 here, full vectors by digest), EOS and stop handling (`<|endoftext|>` and `<|im_end|>` typed by the user, broken special-token forms, `<think>` and `<tool_call>` in user text), multi-turn, a system prompt, `enable_thinking=true`, multilingual text with combining marks and emoji, code, and one 1,666-token prompt. Cached versus uncached decoding, streaming reassembly, the max-token boundary, cancellation and concurrency are checked against these cases by the C# runner (`ai-dotnet/tools/Starlings.Qwen35.Fidelity`); they need no further reference data.

Decisions for the reviewer:

1. The reference is transformers 5.18.0 fp32 on CPU with the checkpoint's own chat template (the runtime of `docs/m0/generation-backend-check.md`), and its tokenizer is the pipeline transformers builds (`Qwen3_5Tokenizer`), equal to the `qwen35-bpe-chat` profile asset.
2. Step-0 logits are kept as the top 64 log-probabilities per prompt. A full-vocabulary bound (the C# runner measures it against the full vectors) needs those vectors from a rerun of step 1 or from the maintainer's copy; their digests are recorded.
3. Tolerances are not proposed here: candidate values with the observations behind them are in `docs/evidence/p016-qwen35/README.md` ("Proposed tolerances"). The maintainer sets `tolerances.yaml` per profile.
4. The prompt set (`prompts-v1.jsonl`) is agent-written development data and is accepted as such.
