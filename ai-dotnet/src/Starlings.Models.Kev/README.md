# Starlings.Models.Kev

Kev 1.0 decision models in-process on ONNX Runtime: one document (the *state*) and typed questions about it in, a calibrated probability per option out, in one prefill pass without generating text. Two checkpoints, both **Experimental**:

| Profile | Checkpoint | Base | Bundle |
| --- | --- | --- | --- |
| `kev-0.8b/onnx-fp32-cpu` | [`jaredpalmer/kev-0.8b`](https://huggingface.co/jaredpalmer/kev-0.8b) at `9a45d25eb2ab761841196625383fa1dff0e56c1e` | `Qwen/Qwen3.5-0.8B-Base` at `dc7cdfe2…` | 3.1 GB |
| `kev-4b/onnx-fp32-cpu` | [`jaredpalmer/kev-4b`](https://huggingface.co/jaredpalmer/kev-4b) at `139fdd94f1b6a6ad80cc15e08fcb99cac885a101` (weights identical to `main` at `6cfce5c2`) | `Qwen/Qwen3.5-4B-Base` at `1001bb4d…` | 16.9 GB |

- The request is rendered and escaped as Kev's `kev/api.py` and `kev/model.py` do at `6b719c3c` (tag `kev-1.0`): System One `noul`, `choice` and `score` questions, JSON states rendered with Python's `str()` of numbers and booleans, `<|…|>` in user text rewritten so it can never become a delimiter. Token ids come from the owned tokenizer `Starlings.Tokenizers` (profile `qwen2-bpe-kev`; both checkpoints ship the same `tokenizer.json`): no Microsoft.ML.Tokenizers, no Python.
- The Qwen3.5 backbone (Gated DeltaNet and full attention) runs as one fp32 ONNX graph. The state runs once; every question then runs as its own causal row continuing from the state's cache, so questions cannot influence one another. Kev's pointer head scores each option's `</opt>` against the question's `<decide>`, divided by the calibrated temperature.
- The package ships the asset locks (sizes and SHA-256), not the model files.

## Install

```sh
dotnet add package Starlings.Models.Kev --prerelease
dotnet add package Microsoft.ML.OnnxRuntime --version 1.30.0     # CPU natives (contrib ops LinearAttention, CausalConvWithState, ...)
```

## Use

```csharp
using Starlings.Models.Kev;

KevModelFiles files = KevModelAssets.OpenDirectory(KevModel.Kev4B, "/models/kev-4b");   // verifies every file, never downloads
using var kev = KevDecisionModel.Create(files);

var decision = kev.Decide(new KevRequest(
    "I was charged twice for order 1182. Please refund one of the charges.",
    [
        new("team", KevQuestion.Choice("Which team should handle this?", new Dictionary<string, KevContent?>
        {
            ["billing"] = "Charges and refunds", ["shipping"] = "Deliveries", ["returns"] = "Exchanges",
        })),
        new("urgent", KevQuestion.Noul("Does this need a reply today?")),
        new("severity", KevQuestion.Score("How severe is the problem?", "minor", "moderate", "serious")),
    ]));

Console.WriteLine($"{decision["team"].Choice} ({decision["team"].Confidence:F2})");   // billing (0.8…)
Console.WriteLine($"p(urgent) = {decision["urgent"].Yes:F3}");
Console.WriteLine($"severity = {decision["severity"].ExpectedScore:F2}");

// A System One request body as upstream's server takes it (experimental):
var fromJson = kev.Decide(KevRequest.ParseSystemOne(File.ReadAllText("request.json")));
```

- `KevOptions.Temperature = 1.0` returns the raw probabilities (upstream `KEV_TEMPERATURE=1.0`); null, the default, applies the calibrated temperature from `head.pt` (0.8B 2.351, 4B 2.406).
- Answers are unrounded. `Choice` is the first most likely key, `Confidence` is System One's `(p_max - 1/K) / (1 - 1/K)` (the score formula for score questions), `ExpectedScore` is `sum(i * p_i)`.
- States over 8,192 tokens (the validated context, `<state>` included) and question rows over 8,192 tokens (state plus branch) throw `KevInputTooLongException`. Nothing is truncated.
- Staged API for evidence: `Encode(record)` gives Kev's row form (`KevEncoding`), `Run(encoding)` the cached path, `RunUncached(encoding)` upstream's `forward()` layout.
- Memory: about 3.5 GB resident for 0.8B and 18 GB for 4B, plus the state cache.

## The bundle

No Hub repository ships an fp32 ONNX graph of Kev 1.0 (`midudev/kev-4b-ONNX` is int4 for the browser, with a mean drift of 0.04 in probability and flipped answers; `onnx-community/kev-4b-ONNX` is the previous Qwen3 generation, 4-bit). Starling builds the bundle from the pinned sources:

```sh
# Kev environment (model-pipeline/reference/kev), TORCH_FORCE_WEIGHTS_ONLY_LOAD=1, HF_HUB_OFFLINE=1
python model-pipeline/reference/kev/kev_merge.py --checkpoint <kev-4b dir> --base <Qwen3.5-4B-Base dir> --out merged
python -m onnxruntime_genai.models.builder -i merged -o onnx -p fp32 -e cpu \
  --extra_options exclude_lm_head=true exclude_mtp=true          # onnxruntime-genai 0.17.1
# bundle: onnx/model.onnx, onnx/model.onnx.data, merged/head.safetensors, merged/head.json, the Kev repo's tokenizer.json
```

`kev_merge.py` merges the LoRA in fp32 with Kev's own loader and writes the pointer head as `head.safetensors` plus `head.json`, so nothing at runtime unpickles `head.pt`. The builder's weights file is not byte-reproducible across runs (same size, different bytes; recorded for 0.8B in `model-pipeline/admissions/kev.md`), so the lock pins the bundle that was qualified; a rebuilt bundle needs `KevDecisionModel.Create(profile, paths…)` with its own digests and its own evidence. The graphs are not hosted yet: hosting under the `Starlinghq` Hugging Face organization is the maintainer's step, after which `KevModelAssets` gains download sources.

## Status

Every capability has an explicit status in `KevCapabilities.All`. Experimental: noul, choice and score decisions, calibrated and raw temperature, delimiter escaping, question isolation, the state cache within a request, JSON state rendering, cancellation, offline verification. Untested: concurrent calls, CUDA, Windows, Native AOT. Unsupported: states over 8,192 tokens, truncation, downloads, state reuse across requests, batched rows, upstream's 4-decimal rounding and `usage.output_tokens`, `KEV_DATE_FACTS`, option isolation, the HTTP server, bf16, fp16 and quantized graphs, training and calibration fitting, Kev-9B and Kev-27B.

## Evidence

<!-- kev:evidence -->
Development evidence, 2026-10-16, Linux x64 CPU, ONNX Runtime 1.30.0, references proposed (not adopted into `model-pipeline/protected/`). Bounds: `model-pipeline/protected/kev-0.8b/onnx-fp32/tolerances.yaml` for 0.8B; the same values proposed for 4B.

Onboarding route `kev-onnx-cpu-v1` on the Starling-written corpus `model-pipeline/reference/corpus/kev-decisions-v1.jsonl` (30 records, 58 questions, 44 labelled; states up to 1,531 tokens):

| Check (bound) | Kev-0.8B | Kev-4B |
| --- | --- | --- |
| Python ONNX Runtime vs PyTorch fp32 (2e-4) | 2.95e-5 | 6.49e-5 |
| .NET from reference ids, cached and uncached, and from text, vs PyTorch (2e-4) | 2.96e-5 | 6.49e-5 |
| Cached vs uncached rows (1e-5) | .NET 2.7e-7 | .NET 2.5e-7 |
| .NET vs Python ONNX Runtime (1e-6) | 1.24e-7 | 1.81e-7 |
| Argmax flips, accuracy, token ids | none, identical, identical | none, identical (0.909 on 44), identical |

Kev-4B, upstream's development partitions through Python ONNX Runtime against Kev's own `kev.benchmark` on CPU fp32:

| Partition | Questions | Max abs (mean, p99) | Flips | Accuracy (both) |
| --- | ---: | --- | ---: | --- |
| decision-v7, 24 fixture records (forward and cached layouts) | 28 | 8.3e-5 (7.3e-6, 7.3e-5) | 0 | n/a |
| transfer-v4 development | 764 | **2.05e-4** (1.2e-5, 9.7e-5) | 0 | 0.8171 (n = 656) |

One transfer-v4 question exceeds the 0.8B bound of 2e-4 by 2.5% (no flip, identical accuracy). Kev-4B has 24 Gated DeltaNet layers against 18, and the spike attributed most of the 0.8B error to the CPU `LinearAttention` kernel; the binding bound for 4B is the maintainer's decision. decision-v7 development (1,468 questions) was not run for 4B.

C# tests (`tests/Starlings.Models.Kev.Tests`): 109 tests, all passing with both bundles; rendering of 31 JSON values and the record and token ids of 9 System One requests equal upstream's output exactly.
<!-- /kev:evidence -->

## License

Kev's adapter and head: Apache-2.0 (model card metadata). Qwen3.5 base weights and tokenizer: Apache-2.0. Kev code: Apache-2.0 (no source is copied; the request rendering, encoder and head are reimplemented in C#). Kev's training and evaluation datasets have their own licenses; no suite record is in this repository.
