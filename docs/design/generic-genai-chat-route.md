# A model-agnostic GenAI chat route: design note

Status: **proposed**, 2026-11-01 (#109; the text part of #110 is the second intended consumer). Nothing in this note changes product code, a capability status, a manifest or a tolerance. It is input for a maintainer decision, not a support claim.

## Problem

`microsoft/Phi-4-mini-instruct` (MIT, 3.8 B, `Phi3ForCausalLM`, `model_type` `phi3`) fits the ONNX Runtime GenAI chat pattern that Qwen already uses, but no route admits it:

- The only GenAI onboarding route, `qwen-genai-cpu-v1`, accepts `qwen2` and `qwen3` only (`model-pipeline/ONBOARDING.md`). Its plugin, `model-pipeline/src/model_pipeline/onboarding_generation.py`, hard-codes `PROFILES`, `CONFIG_CONTRACT` and `EXPECTED_BEHAVIOR` per Qwen repository, including Qwen token ids (151643 to 151645) and Qwen's pad overlay (151935). Its reference, bundle builder and GenAI backend live under `model-pipeline/reference/qwen-chat/`.
- On the .NET side the GenAI chat package is Qwen-typed: `Starlings.Models.QwenChat` (template, parser, checkpoint) and `Starlings.Models.QwenChat.OnnxGenAI` (assets, profile, loader). The shared pieces already exist: `Starlings.Generation.Abstractions` (`IChatTemplate`, chat types, stop matching, incremental detokenizer) and `Starlings.Generation.OnnxGenAI` (`GenAIModel`, `GenAISequence`, `TextGenerator`).
- The owned tokenizer cannot load Phi-4-mini's `tokenizer.json` today (next section).

Each new chat family would otherwise need its own copy of the Qwen route. This note proposes the shape of one route that Phi-4-mini, and later the text half of `google/gemma-3-4b-it` (#110), can share.

## Constraints

- **Tokenizer, in starlinghq/tokenizers first.** Phi-4-mini ships an o200k-vocabulary byte-level BPE (`vocab_size` 200064 in `config.json`). Its pre-tokenizer is a `Sequence` of a `Split` with the o200k regex and `ByteLevel` with `use_regex` false. Two parts of that are outside the supported set in starlinghq/tokenizers `docs/supported-components.md` (as last read; confirm against the 0.3.4 release before starting):
  1. The `Split` regex is neither of the two Qwen patterns nor the Llama 3 pattern (added in 0.3.3). It uses the classes `\p{Lu}`, `\p{Lt}`, `\p{Lm}`, `\p{Lo}`, `\p{Ll}`, which the hand-written scanner does not have (its tables are `\p{L}`, `\p{N}`, `\p{M}`), and it is declared with `behavior: Removed, invert: true`, where the Qwen patterns are accepted only as `Isolated, invert: false`. Because the regex's last alternative is `\s+`, every character is matched, so the two forms may split identically; the tokenizer repo has to prove that, not assume it.
  2. The chat special tokens (`<|assistant|>`, `<|end|>`, `<|user|>`, `<|system|>`, `<|tool|>`, ...) have `rstrip: true`. `rstrip` true is listed as unsupported.
  So the profile (working name `phi4-bpe-chat`) needs a new scanner pattern with generated class tables, `rstrip` support on added tokens, conformance fixtures and the randomized differential run, then a tokenizers release and a pin raise here (`ai-dotnet/Directory.Packages.props`, currently `[0.3.4]`, plus every manifest's `processing.tokenizer.version`) in one change.
- **No remote code.** `config.json` carries an `auto_map` to `modeling_phi3.Phi3ForCausalLM` and an `AutoTokenizer` entry pointing at another repository. The reference must load the built-in transformers `Phi3ForCausalLM` (and `PreTrainedTokenizerFast` from the pinned `tokenizer.json` in the Phi repo) with `trust_remote_code` off, and the route must refuse a candidate whose architecture the pinned transformers does not provide.
- **Architecture details the parity checks must cover.** `partial_rotary_factor` 0.75, `rope_scaling` type `longrope` with short and long factors switching at `original_max_position_embeddings` 4096, tied embeddings, GQA 24/8. The first profile caps prompt plus output below 4096 tokens so only the short factors are exercised; long-context behavior stays `untested` until a separate run covers it.
- **Export.** The bundle comes from the ORT GenAI model builder at the pinned version (0.17.1), run on the pinned snapshot, not from a downloaded ONNX file, unless that file is first compared against the Python reference.
- **Compute.** fp32 is about 15 GB of weights. CPU parity runs fit the qualification machine; GPU runs need the budget tracked in #210, use only the RTX 4060 Ti, and support Experimental status only.
- **Bounded authority.** Tolerances, goldens and maturity are maintainer-owned. A new route reuses the generation evaluator's checks (`REQUIRED_EVIDENCE`: prompt text and ids, initial logits, output ids, finish reason, streaming, usage, cancellation, cache, isolation); it does not weaken them per family.

## Options

**A. A Phi-specific route** (`phi-genai-cpu-v1`) copied from the Qwen plugin. Fastest for one model, but the next family (#110) repeats the copy, and fixes drift between copies.

**B. Generalize `qwen-genai-cpu-v1` in place.** Keeps one plugin but changes the identity of a route that already has evidence and published profiles; every existing recipe's stage keys would move.

**C. A new data-driven route, `genai-chat-cpu-v1`, beside the Qwen route.** A route contract table `model-pipeline/routes/genai-chat-cpu-v1/contracts.json`, read by both the plugin and the .NET package, as `bert-onnx-cpu-v1/contracts.json` already is. One entry per upstream repository: `architecture`, `model_type`, a config contract (widths, layers, heads, KV heads, head dim, rope type, partial rotary factor, max positions), `vocab_size`, EOS ids, BOS policy, pad overlay (id and reason, or none), `tokenizer_profile`, template id and its upstream SHA-256, and the explicit status of tools, thinking and multimodal input. The per-family code that remains is the chat-template port (an `IChatTemplate` implementation, as `QwenChatTemplate`, `Lfm3ChatTemplate` and `CloverChatTemplate` already are) and its template corpus. Qwen profiles move to the new route only when the maintainer decides; until then both routes exist.

## Recommendation

Option C. It matches how the BERT route already removed per-model allow-lists from code, it keeps the Qwen route's evidence and stage keys stable, and it makes a second family (#110 text) a table entry plus a template port rather than a new plugin.

On the .NET side, add a model-definition package for Phi (`Starlings.Models.Phi4`: checkpoint identity, template port, tool-call parser if tools are admitted) and either a thin `Starlings.Models.Phi4.OnnxGenAI` loader in the QwenChat style or, preferably, a generic GenAI chat loader that reads the same contract table. The choice between those two is the first open question for the maintainer; the route design does not depend on it.

Order: tokenizer profile first (it is the long pole and lives in another repository), then the route table and plugin, then the template port, then the export and parity run.

## Checklist (one PR each)

1. starlinghq/tokenizers: issue, then PR adding the o200k split pattern (class tables generated from the reference, as the Qwen tables are), `Removed`/`invert: true` equivalence evidence, `rstrip` on added tokens, and the `phi4-bpe-chat` profile with conformance fixtures and a randomized differential run. Release.
2. This repo: raise the `Starlings.Tokenizers` pin and every manifest's `processing.tokenizer.version` in one change.
3. `model-pipeline`: `routes/genai-chat-cpu-v1/contracts.json`, its schema and loader with the same refusal rules as the BERT table, and the `inspect` rule for `phi3` / `Phi3ForCausalLM`. No candidate yet.
4. `model-pipeline`: route-generic reference runner and bundle builder (built-in transformers architectures only, `trust_remote_code` off), reusing `reference/generation/`; the generic plugin registered in `onboarding_plugins.py` with orchestration tests.
5. `Starlings.Models.Phi4`: checkpoint identity and the C# port of the pinned chat template with a template corpus test against the Python rendering. Tools `unsupported` unless this PR also ports and tests the tool format.
6. GenAI loader (generic or Phi-specific, per the maintainer's choice) and the candidate profile, recipe and asset lock for `microsoft/Phi-4-mini-instruct` pinned by full commit SHA.
7. Onboarding run on CPU (fp32, prompts under 4096 tokens), evidence under `docs/evidence/`, profile Experimental. The maintainer adds tolerances and the registry manifest.
8. Later, separately: the #110 text entry on the same table; long-context (`longrope` long factors) evidence; GPU runs under #210.
