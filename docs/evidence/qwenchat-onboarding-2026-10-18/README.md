# QwenChat onboarding (Qwen3-0.6B, Qwen2.5-0.5B-Instruct), CPU fp32

This is development evidence for `Starlings.Models.QwenChat`, produced on 2026-10-18 on Linux x64 (Intel i7-5820K, AVX2, two native threads). It covers the onboarding route `qwen-genai-cpu-v1` for both profiles. The development stages were last run at implementation commit `7eaa9c5` (pinned bundle lock). The clean-consumer distribution ran at `bbe92d7`, which differs only by the distribution-stage fix. It supports the proposed Experimental admission (`model-pipeline/admissions/qwen-chat.md`). The references and the comparison policy are proposed, not adopted into `protected/`, and no gate (G0 to G7) was run.

## Results

| Profile | Rows | Prompt text and ids | Greedy ids, finish reason | Step-0 logits vs transformers, worst max / mean (bound 5e-3 / 5e-4) | .NET vs Python GenAI | Clean consumer |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen3-0.6b/ortgenai-fp32-cpu` | 14 | exact | exact | 1.28e-4 / 3.67e-5 | identical on every row | passed (offline, 174 s) |
| `qwen2.5-0.5b-instruct/ortgenai-fp32-cpu` | 14 | exact | exact | 7.02e-5 / 1.15e-5 | identical on every row | passed (offline, 129 s) |

The corpus is `model-pipeline/corpora/five-models/chat.jsonl`: 14 conversations with 16 new tokens each, and one with a 1-token limit. It covers a greeting, arithmetic, a system message, multiple turns, assistant reasoning history, whitespace, empty content, Chinese, other Unicode, a literal `<|endoftext|>`, role look-alikes and a 1,662-character prompt.

The .NET fidelity tool (`ai-dotnet/tools/Starlings.Generation.Fidelity`) also checked these on every row:
- streaming equals the non-streamed result (text and ids, with one final update);
- 7-token chunked prefill equals a single prefill;
- a repeated request after an unrelated one is identical;
- usage counts are correct;
- cancellation works before the prefill and between decode steps;
- an extra stop token id and a stop string end generation as documented.

All of these passed.

Concurrent requests, long prompts near the context length, sampling and other platforms were not tested.

## Bundle reproducibility

`build_bundle.py` (ONNX Runtime GenAI 0.17.1 builder, source commit `83de55a1`, wheel hash-locked in `model-pipeline/reference/generation/requirements.txt`) was run on the verified original snapshots twice per profile: once by the pipeline's `bundle` stage and once by hand. Every file was byte-identical between the two builds. `QwenChatProfile.BundleFiles` and the registry manifests pin those digests. `QwenChatModel.Load` refuses any bundle whose lock or files differ from them.

| Profile | `model.onnx.data` SHA-256 | Size (bytes) |
| --- | --- | --- |
| Qwen3-0.6B | `2f800ce53e4c864ff5c96e99f869dc4ca0973f0c64c1f1860eea302093b3d7c4` | 2,405,171,200 |
| Qwen2.5-0.5B-Instruct | `4269c6a8f794c9db3f1304c18ead66ca8aa6eb67e57b8bbe1b815cdbdecd7f99` | 1,984,561,152 |

The full locks are in `qwen3/bundle-lock.json` and `qwen25/bundle-lock.json`. Reproducibility was shown on one machine with the locked builder wheel only.

## Distribution

The `--distribution` stage:
1. packed `Starlings.Models.QwenChat` and its dependencies from a clean clone of the committed source;
2. restored an external consumer from those packages only (Starlings bytes checked against the packed ones, `Starlings.Tokenizers` against nuget.org);
3. removed the checkout and package caches;
4. ran the consumer with `unshare -rn` (no network interfaces) on the pinned bundle.

The consumer reproduced the development parity (`consumer-parity.json`, `distribution.json`). This is development evidence of package consumption, not G6.

The first distribution attempt failed to compile the consumer (`CS0103: ToolMain`). Since #37 the fidelity tools link `ai-dotnet/tools/Shared/ToolMain.cs`, which the stage did not copy. `bbe92d7` fixes the stage. The parity stages were unaffected.

## Timings

These are observed on a shared workstation, not a benchmark. A full uncached run took 9 minutes for Qwen3 and 11 minutes for Qwen2.5, without distribution. That breaks down as about 3 minutes for the transformers reference, 1 for the bundle, 3 for Python GenAI and 3 to 4 for the .NET tool. The distribution stage added about 6 and 5 minutes respectively.

## Environment

- Reference: Python 3.12 with torch 2.14.1+cu130 and transformers 5.18.0, selecting CPU.
- Backend: onnxruntime-genai 0.17.1, from the hash-locked `reference/generation/requirements.txt`.
- .NET SDK 10.0.100.

Each profile's `environment.json` has the full record.

## Files

For each of `qwen3/` and `qwen25/`:
- `review.md`, `review.json`: the pipeline's review;
- `report.json`: the evaluator's per-row comparisons;
- `distribution.json`, `consumer-parity.json`, `consumer-offline.log`: the clean-consumer drill;
- `bundle-lock.json`: the built bundle's digests;
- `environment.json`.

Weights, graphs and logits are not committed. The complete runs and verified exports (`model-pipeline propose`) are under the ignored `artifacts/onboarding/`.

## Reproduce

From the repository root, with the environments and snapshots described in `model-pipeline/FIVE_MODELS.md`. Each snapshot must hold every file in its candidate contract.

```sh
.models/venv-generation/bin/model-pipeline run model-pipeline/recipes/qwen3-cpu.yaml \
  --snapshot /path/to/snapshots/qwen3 \
  --reference-python .models/venv-generation/bin/python \
  --out artifacts/onboarding/qwen3 --threads 2 --timeout 3600 --distribution
.models/venv-generation/bin/model-pipeline propose artifacts/onboarding/qwen3 \
  --out artifacts/onboarding/reviews/qwen3
```

For Qwen2.5, use `qwen25-cpu.yaml` and the `qwen25` paths. Qwen CPU profiles accept only `--threads 1` or `2`.
