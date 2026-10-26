# Starlings.Models.Kev.Onnx

Kev 1.0 decision models in-process on ONNX Runtime: one document (the *state*) and typed questions about it in, a calibrated probability per option out, in one prefill pass without generating text. Two checkpoints, both **Experimental**:

| Profile | Checkpoint | Base | Bundle (hosted) |
| --- | --- | --- | --- |
| `kev-0.8b/onnx-fp32-cpu` | [`jaredpalmer/kev-0.8b`](https://huggingface.co/jaredpalmer/kev-0.8b) at `9a45d25eb2ab761841196625383fa1dff0e56c1e` | `Qwen/Qwen3.5-0.8B-Base` at `dc7cdfe2…` | 3.1 GB, [`Starlinghq/kev-0.8b-onnx`](https://huggingface.co/Starlinghq/kev-0.8b-onnx) at `8a9fe017…` |
| `kev-4b/onnx-fp32-cpu` | [`jaredpalmer/kev-4b`](https://huggingface.co/jaredpalmer/kev-4b) at `139fdd94f1b6a6ad80cc15e08fcb99cac885a101` (weights identical to `main` at `6cfce5c2`) | `Qwen/Qwen3.5-4B-Base` at `1001bb4d…` | 16.9 GB, [`Starlinghq/kev-4b-onnx`](https://huggingface.co/Starlinghq/kev-4b-onnx) at `8179da50…` |

- The request is rendered and escaped as Kev's `kev/api.py` and `kev/model.py` do at `6b719c3c` (tag `kev-1.0`): System One `noul`, `choice` and `score` questions, JSON states rendered with Python's `str()` of numbers and booleans, `<|…|>` in user text rewritten so it can never become a delimiter. Token ids come from the owned tokenizer `Starlings.Tokenizers` (profile `qwen2-bpe-kev`; both checkpoints ship the same `tokenizer.json`): no Microsoft.ML.Tokenizers, no Python.
- The Qwen3.5 backbone (Gated DeltaNet and full attention) runs as one fp32 ONNX graph. The state runs once; every question then runs as its own causal row continuing from the state's cache, so questions cannot influence one another. Kev's pointer head scores each option's `</opt>` against the question's `<decide>`, divided by the calibrated temperature.
- The package ships the asset locks (sizes and SHA-256), not the model files. The qualified bundles are hosted publicly on the `Starlinghq` Hugging Face organization (no token needed); `KevModelAssets.DownloadAsync` fetches and verifies them.
- On the CUDA provider the same bundles run as the profiles `kev-0.8b/onnx-fp32-cuda` and `kev-4b/onnx-fp32-cuda` (Experimental, `use_tf32=0`; see [CUDA](#cuda)).

## Maturity

<!-- catalog:maturity profiles=kev-0.8b/*,kev-4b/* -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `kev-0.8b/onnx-fp32-cpu` | experimental | linux-x64 cpu: experimental; linux-x64 cuda: experimental; windows-x64 cpu: untested |
| `kev-0.8b/onnx-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
| `kev-4b/onnx-fp32-cpu` | experimental | linux-x64 cpu: experimental; linux-x64 cuda: experimental; windows-x64 cpu: untested |
| `kev-4b/onnx-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
<!-- /catalog:maturity -->

## Install

Until 0.5 this was the package `Starlings.Models.Kev`, on nuget.org from release 0.3.0 (`docs/releases/ai-0.3.0.md`). The request types, rendering, encoder and checkpoint identities now form the model definition package `Starlings.Models.Kev` (no ONNX Runtime), which this package references; every type keeps the namespace `Starlings.Models.Kev`. Keep every `Starlings.*` package at the same version.

```sh
dotnet add package Starlings.Models.Kev.Onnx
dotnet add package Microsoft.ML.OnnxRuntime --version 1.30.0     # CPU natives (contrib ops LinearAttention, CausalConvWithState, ...)
# or, for CUDA (Linux x64; the CUDA 13 runtime and cuDNN 9 are not in the package):
dotnet add package Microsoft.ML.OnnxRuntime.Gpu.Linux --version 1.30.0
```

## Use

```csharp
using Starlings.Models.Kev;

// Setup, once: download the bundle into the acquisition cache and verify every file (3.1 GB for 0.8B, 16.9 GB for 4B).
await KevModelAssets.DownloadAsync(KevModel.Kev4B);

// Startup, offline: verify the cached files and load. OpenDirectory(model, dir) opens a bundle in any directory.
KevModelFiles files = KevModelAssets.OpenLocal(KevModel.Kev4B);
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

- `KevOptions.Temperature = 1.0` returns the raw probabilities (upstream `KEV_TEMPERATURE=1.0`); null, the default, applies the calibrated temperature from `head.pt` (0.8B 2.351, 4B 2.406). `KevDecisionModel.IsCalibrated` and every `KevAnswer.IsCalibrated` say whether the temperature in effect is the checkpoint's (null, or set to that value); `KevAnswer.Temperature` is the value used.
- Answers are unrounded. `Choice` is the first most likely key, `Confidence` is System One's `(p_max - 1/K) / (1 - 1/K)` (the score formula for score questions), `ExpectedScore` is `sum(i * p_i)`.
- States over 8,192 tokens (the validated context, `<state>` included) and question rows over 8,192 tokens (state plus branch) throw `KevInputTooLongException`. Nothing is truncated. `KevOptions.MaxBranchTokens` (null by default) adds a budget on a question's branch alone, the state not counted. Every over-long question is found in one pass: the exception lists them all (`OverLongQuestions`, each with `Index`, `Id`, `Limit`, `TokenCount`, `MaxTokens`), or, with `KevOptions.DeclineOverLongQuestions = true`, `Decide` answers the other questions and lists the over-long ones in `KevDecision.Declined`. The answered questions' probabilities are the same as without the declined ones.
- `KevDecisionModel.Placement` reports where the model runs: provider, CUDA device, profile id, whether native calls are serialized, the node placement when `OnnxSessionOptions.CapturePlacement` is set, and `StateCacheInHostMemory` (true for the qualified bundles, see [CUDA](#cuda)).
- State reuse: `CreateState(state)` runs a document once and returns a `KevState` that holds its cache; `Decide(state, questions)` answers any number of question sets against it without the state pass, with the same probabilities as `Decide(new KevRequest(state, questions))` (bit for bit on CPU over the nine request goldens, also from four concurrent callers). Dispose each state before the model. From the graph's cache shapes (fp32), the cache takes about 24 KB per state token plus 18 MB of DeltaNet state for Kev-0.8B (6 attention layers of 2 key/value heads, 18 DeltaNet layers of 16 x 128 x 128), and about 64 KB per token plus 48 MB for Kev-4B. CUDA: untested.
- `KevModelAssets.OpenLocal` and `OpenDirectory` with `CacheVerification.Sha256` (the default) hash the bundle; `KevModelFiles.ModelSha256` then carries the graph's digest and `KevDecisionModel.Create(files)` passes it to the session (`OnnxSessionOptions.KnownModelSha256`), so `model.onnx` is not read a second time at load.
- Staged API for evidence: `Encode(record)` gives Kev's row form (`KevEncoding`), `Run(encoding)` the cached path, `RunUncached(encoding)` upstream's `forward()` layout.
- Memory: about 3.5 GB resident for 0.8B and 18 GB for 4B, plus the state cache.
- **On Linux, start a long-running process with `MALLOC_MMAP_THRESHOLD_=65536`** (as for GLiNER2). Without it, a one-hour Kev-0.8B CUDA soak's loaded memory grew 64 MB/h (glibc arena retention, outputs correct throughout); with it, every Kev soak plateaued ([evidence](../../../docs/evidence/kev-soak-2026-10-17/README.md)).
- Concurrent `Decide` calls on one model are independent (`concurrent_calls`, Experimental). On CUDA each request's graph calls run one request at a time by default (`KevOptions.SerializeNativeCalls`).

## CUDA

```csharp
using Starlings.Runtime.Onnx;

using var kev = KevDecisionModel.Create(files, new KevOptions
{
    Session = new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cuda, CudaDeviceId = 0 },
});
// kev.ProfileId == "kev-4b/onnx-fp32-cuda"; kev.SerializesNativeCalls == true
```

- The CUDA profiles use the qualified bundle and lock unchanged: the builder's CPU export on the CUDA execution provider, with TF32 off (`use_tf32=0`, forced by `Starlings.Runtime.Onnx`; there is no TF32 option, and TF32 fails every fp32 bound). ONNX Runtime keeps the graph's `GroupQueryAttention` nodes on the CPU provider inside the CUDA session, and the state cache passes through host memory between rows; the builder's own CUDA graph (`-e cuda -p fp32`) is faster but did not load unpatched with ONNX Runtime GenAI 0.17.1 ([admission](../../../model-pipeline/admissions/kev.md), "GPU results"), so it is not a profile.
- Device memory: the 4B weights take about 17 GB, so Kev-4B needs a 24 GB card; 0.8B about 3.5 GB.
- Requests are serialized on CUDA by default: concurrent runs on one CUDA session returned wrong outputs for GLiNER2 and Qwen3.5 on ONNX Runtime 1.30, and unserialized Kev runs were not tried (`concurrent_calls.cuda.unserialized`: Untested). A cancelled CUDA call completes its current graph run before it throws, as for every Runtime.Onnx CUDA session.
- Evidence so far is local (an RTX 3090 workstation); the per-release GPU regression and the gates are not run yet. Other GPUs: untested.

## The bundle

No Hub repository ships an fp32 ONNX graph of Kev 1.0 (`midudev/kev-4b-ONNX` is int4 for the browser, with a mean drift of 0.04 in probability and flipped answers; `onnx-community/kev-4b-ONNX` is the previous Qwen3 generation, 4-bit). Starling builds the bundle from the pinned sources:

```sh
# Kev environment (model-pipeline/reference/kev), TORCH_FORCE_WEIGHTS_ONLY_LOAD=1, HF_HUB_OFFLINE=1
python model-pipeline/reference/kev/kev_merge.py --checkpoint <kev-4b dir> --base <Qwen3.5-4B-Base dir> --out merged
python -m onnxruntime_genai.models.builder -i merged -o onnx -p fp32 -e cpu \
  --extra_options exclude_lm_head=true exclude_mtp=true          # onnxruntime-genai 0.17.1
# bundle: onnx/model.onnx, onnx/model.onnx.data, merged/head.safetensors, merged/head.json, the Kev repo's tokenizer.json
```

`kev_merge.py` merges the LoRA in fp32 with Kev's own loader and writes the pointer head as `head.safetensors` plus `head.json`, so nothing at runtime unpickles `head.pt`. `model.onnx.data` is not byte-reproducible across builds: two Kev-4B builds differ in 33 of 452 tensors, 1 to 4 floats each (at most 1.9e-9 absolute), all of them LoRA-merged projections (`attn.k/v/o_proj`, `linear_attn.out_proj`); `model.onnx`, embeddings, norms and every unadapted weight are identical. The export is deterministic; the fp32 LoRA merge is not (reduction order, consistent with the different thread counts of the two builds). The lock pins the bundle that was qualified, and that bundle is the one hosted: [`Starlinghq/kev-0.8b-onnx`](https://huggingface.co/Starlinghq/kev-0.8b-onnx) at `8a9fe0175d294a89c6fa963a2245eb32e3fa1505` and [`Starlinghq/kev-4b-onnx`](https://huggingface.co/Starlinghq/kev-4b-onnx) at `8179da507de7de4bddfdbfa44894ff24b1dcd959` (tag `r1`; the five files plus README, NOTICE, LICENSE and SHA256SUMS). A rebuilt bundle needs `KevDecisionModel.Create(profile, paths…)` with its own digests and its own evidence.

## Status

Every capability has an explicit status in `KevCapabilities.All`; `KevCapabilities.Profiles` repeats each profile's statuses keyed as its registry manifest (`model-pipeline/registry/kev-*`), and a test keeps the two equal. Experimental: noul, choice and score decisions, calibrated and raw temperature, delimiter escaping, question isolation, the state cache within a request, JSON state rendering, cancellation, offline verification, downloads from the hosting repositories, concurrent calls (on CUDA serialized), CUDA, the branch budget and declining over-long questions, the placement report, the calibration flag, state reuse across calls on CPU. Untested: unserialized concurrent calls on CUDA, state reuse across calls on CUDA, Windows, Native AOT. Unsupported: states over 8,192 tokens, truncation, reuse of a common prefix of different states, batched rows, upstream's 4-decimal rounding and `usage.output_tokens`, `KEV_DATE_FACTS`, option isolation, the HTTP server, TF32, bf16, fp16 and quantized graphs, training and calibration fitting, Kev-9B and Kev-27B.

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

One transfer-v4 question exceeds the 0.8B bound of 2e-4 by 2.5% (no flip, identical accuracy). Kev-4B has 24 Gated DeltaNet layers against 18, and the spike attributed most of the 0.8B error to the CPU `LinearAttention` kernel; the binding bound for 4B is the maintainer's decision. The full decision-v7 development run for 4B (1,468 questions) was started and stopped for memory before it finished; it is not reported.

C# tests (`tests/Starlings.Models.Kev.Tests`): 109 tests, all passing with both bundles; rendering of 31 JSON values and the record and token ids of 9 System One requests equal upstream's output exactly.

CUDA, 2026-10-17, local RTX 3090 (24 GB, driver 610.57, system CUDA 13.3 runtime, cuDNN 9 from the `nvidia-cudnn` wheel), ONNX Runtime 1.30.0 (`Microsoft.ML.OnnxRuntime.Gpu.Linux`), `use_tf32=0`; `tests/Starlings.Models.Kev.Tests.Cuda`, 12 tests passing, against the same Kev PyTorch fp32 probabilities and bounds as on CPU:

| Check (bound) | Kev-0.8B | Kev-4B |
| --- | --- | --- |
| 9 request goldens vs PyTorch fp32 (2e-4), argmax | 1.33e-5, no flip | 4.17e-5, no flip |
| Cached vs uncached rows (1e-5) | 7.8e-7 | 7.2e-7 |
| 4 callers x 2 rounds of the goldens (serialized) | 72 of 72 bit-identical to sequential | 72 of 72 |
| Cancelled calls: device work left queued | 0 of 10 (drain at most 0.014 ms) | 0 of 10 (at most 0.019 ms) |
| One 8-question request (`many-questions`) | 105 ms | 299 ms |

The onboarding corpus with the `onnx` and `dotnet` stages on CUDA, through the route's decision evaluator: passed for both sizes, every path within 6.6e-5 of PyTorch (0.8B 2.85e-5, 4B 6.57e-5), .NET against Python ONNX Runtime on CUDA within 1.6e-7, no flip, identical accuracy and token ids ([evidence](../../../docs/evidence/kev-cuda-2026-10-17/README.md)).

Soaks, 2026-10-17 ([evidence](../../../docs/evidence/kev-soak-2026-10-17/README.md)): one-hour soaks of all four profiles pass with `MALLOC_MMAP_THRESHOLD_=65536` (Kev-4B CUDA also at the default allocator; Kev-0.8B CUDA at the default allocator grew 64 MB/h, with every output correct). On a RunPod RTX 3090 (driver 580.159.03) with the bundles downloaded anonymously from the hosting repositories, the 12 CUDA tests pass (goldens within 4.2e-5) and 10-minute soak smokes of both sizes pass.

Not yet run on CUDA: upstream's development partitions, the matched benchmark, and the per-release regression in the qualification workflow.
<!-- /kev:evidence -->

## Tests

`tests/Starlings.Models.Kev.Tests` (in `AI.slnx`; rendering, contracts and answer mapping run in CI, encoding tests need the pinned `tokenizer.json`, model tests need `STARLING_KEV_08B_BUNDLE` / `STARLING_KEV_4B_BUNDLE`) and the local-only `tests/Starlings.Models.Kev.Tests.Cuda` (not in `AI.slnx`; references the GPU natives):

```sh
export STARLING_CUDA_TESTS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0   # the GPU to use
export STARLING_KEV_08B_BUNDLE=/models/kev-0.8b STARLING_KEV_4B_BUNDLE=/models/kev-4b
# If the CUDA 13 runtime is not in the ldconfig cache, add it (and cuDNN 9) to LD_LIBRARY_PATH.
dotnet test --project ai-dotnet/tests/Starlings.Models.Kev.Tests.Cuda
```

The CUDA tests load one size at a time (Kev-4B needs about 17 GB of device memory). The fidelity tool runs the onboarding route's `dotnet` stage on CUDA with `--device cuda[:N]` from its `-p:UseCuda=true` build.

## License

Kev's adapter and head: Apache-2.0 (model card metadata). Qwen3.5 base weights and tokenizer: Apache-2.0. Kev code: Apache-2.0 (no source is copied; the request rendering, encoder and head are reimplemented in C#). Kev's training and evaluation datasets have their own licenses; no suite record is in this repository.
