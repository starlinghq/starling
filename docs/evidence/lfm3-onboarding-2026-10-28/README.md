# LFM3-1.7B on ONNX Runtime, CPU fp32: development evidence

Development evidence for `Starlings.Models.Lfm3` (candidate `lfm3-1.7b/onnx-fp32-cpu`, model request #217), produced on 2026-10-28 on Linux x64 (Intel Xeon at 2.80 GHz, 4 vCPUs, 15 GB RAM) in a cloud container with no GPU. It is not release qualification: no gate ran, no fixture is protected, and nothing here is binding.

## Inputs

- Graph: `LiquidAI/LFM3-1.7B-ONNX` at `c680759afee1ce8310c90d6c621f4558200dee77`, `onnx/model.onnx` and its four data files (fp32, 7.18 GB). Every file was downloaded at that revision and its SHA-256 checked against `Lfm3ModelAssets.Fp32Lock` (the Hub's LFS digests).
- Checkpoint: `LiquidAI/LFM3-1.7B` at `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`: `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja`, the files the template and tokenizer stages read.
- Corpus: `model-pipeline/corpora/five-models/chat.jsonl` (14 cases, the one the Qwen chat candidates use) and `model-pipeline/corpora/lfm3/chat.jsonl` (5 cases: thinking enabled, digit runs, `preserve_thinking`, French, a 64-token answer). 19 cases, 1 to 64 new tokens each, prompts of 11 to 368 tokens.
- Reference environment: `model-pipeline/reference/lfm3/requirements.txt` (hash-locked; Python 3.12, transformers 5.21.0, tokenizers 0.23.3, onnxruntime 1.30.0, torch 2.14.1+cpu).
- Owned tokenizer: `Starlings.Tokenizers` built from starlinghq/tokenizers#18 (branch `feat/lfm3-bpe-chat`, packed locally as `0.3.4-lfm3local`). The released `0.3.2` this repository pins rejects the LFM3 pre-tokenizer regex, so the two tokenizer-dependent checks below ran against that local build.

## Results

| Check | Compared with | Result |
| --- | --- | --- |
| Template text (`Lfm3ChatTemplateTests`, in CI) | transformers 5.21.0 `apply_chat_template` on the checkpoint's template (`run_lfm3.py template`) | 19/19 renderings equal |
| Prompt ids (`Lfm3ParityTests.PromptIdsMatchTheReferenceTokenizer`) | the transformers tokenizer (`TokenizersBackend`), `add_special_tokens=False` | 19/19 equal, including the literal `<|endoftext|>`, `<|im_start|>` and `<|im_end|>` in user text |
| Greedy ids and step-0 logits (`Lfm3ParityTests.GreedyIdsAndStep0LogitsMatchOnnxRuntimeInPython`) | an independent greedy loop over the same graph on ONNX Runtime 1.30.0 in Python (`run_lfm3.py onnx`) | 19/19 cases equal ids (313 generated tokens, 8 cases ending at `<|im_end|>`, 11 at the token limit); step-0 logits max abs difference 0 on every case (`dotnet-parity.txt`) |

The .NET run took 95 s for the 19 cases (the test's wall time, model load included); the Python run took 146 s. These are not benchmarks.

Sample completions (Python run, decoded): greeting "Hello! How can I help you today?"; arithmetic "2 + 2 = 4."; multilingual "La rivière est large."; thinking-digits starts "We need to add 12345 and 67890. That's straightforward: 12345 + 67890."

## Not done

- **PyTorch reference.** `run_lfm3.py reference` runs the checkpoint's own `modeling_lfm3.py` (transformers `trust_remote_code`) in fp32. It was not run here: executing that downloaded code needs the maintainer's approval. Until it runs, the export's fidelity to the checkpoint is unmeasured, and the candidate's `reference_*` bounds are unchecked.
- **fp16 and CUDA.** `Lfm3ModelAssets.Fp16Lock` and `Lfm3Precision.Fp16` exist for the RTX 3090 profile, but no GPU was available; `generation.fp16` and `generation.cuda` are Untested.
- **Long context.** Prompts stop at 368 tokens; the sliding-window layers (window 256) are exercised only just past their window.

## Files

| File | Contents |
| --- | --- |
| `onnx-generated-ids.jsonl` | Greedy ids per case from `run_lfm3.py onnx` (the step-0 logits, 512 KB per case, are not committed) |
| `dotnet-parity.txt` | Per-case diagnostics of the .NET parity test |
| `environment-onnx.json`, `environment-template.json` | Runner environments: pins, template and tokenizer digests, corpus digests, package versions |

Commands (paths shortened):

```sh
python -I model-pipeline/reference/lfm3/run_lfm3.py template --checkpoint <LFM3-1.7B snapshot> \
    --corpus model-pipeline/corpora/five-models/chat.jsonl --corpus model-pipeline/corpora/lfm3/chat.jsonl --out <out>
python -I model-pipeline/reference/lfm3/run_lfm3.py onnx --checkpoint <LFM3-1.7B snapshot> \
    --onnx-model <LFM3-1.7B-ONNX snapshot>/onnx/model.onnx \
    --corpus model-pipeline/corpora/five-models/chat.jsonl --corpus model-pipeline/corpora/lfm3/chat.jsonl --out <out> --threads 4
STARLING_LFM3_SNAPSHOT=<LFM3-1.7B-ONNX snapshot> STARLING_LFM3_REFERENCE=<out> \
    dotnet test --project ai-dotnet/tests/Starlings.Models.Lfm3.Tests -- --filter-class '*Lfm3ParityTests'
```
