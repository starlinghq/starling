# Starlings.Models.Gliner2

**Preview.** English entity extraction with [`fastino/gliner2-base-v1`](https://huggingface.co/fastino/gliner2-base-v1) at revision `f9634218e53580c56edf0de97ca1a7d3f1c2354e` (GLiNER2 2.0.0, source `55656fbf`), on ONNX Runtime 1.30.0, in-process and offline after an explicit acquisition. You name the entity labels; the model returns labelled spans with offsets and confidences. The API and the supported platforms may still change. The registry profile is Experimental; see [Path to Stable](#path-to-stable).

- GLiNER2's text handling is ported with Python semantics (normalization, the regular-expression word splitter, `str.lower()`), and the schema prefix is encoded with the owned tokenizer `Starlings.Tokenizers` (profile `gliner2-base-v1-unigram`). No Microsoft.ML.Tokenizers, no native tokenizer, no Python.
- One ONNX graph (token ids and routing positions to count and span logits); GLiNER2's count gate, threshold, overlap suppression and result order in C#.
- The package ships the asset lock (sizes and SHA-256), not the 838 MB of model files. Acquisition is an explicit call; inference never downloads.

## Install

```sh
dotnet add package Starlings.Models.Gliner2 --prerelease
dotnet add package Microsoft.ML.OnnxRuntime --version 1.30.0          # CPU natives
# or, for CUDA on Linux x64: Microsoft.ML.OnnxRuntime.Gpu.Linux 1.30.0 (reference exactly one)
```

## Use

```csharp
using Starlings.Models.Gliner2;

Gliner2ModelFiles files = Gliner2ModelAssets.OpenLocal();   // offline, verified, never downloads
using var extractor = Gliner2EntityExtractor.Create(files);

var result = extractor.Extract(
    "Apple CEO Tim Cook announced the iPhone 15 in Cupertino.",
    ["company", "person", "product", "location"],
    threshold: 0.5f);
foreach (var e in result.Entities)
{
    Console.WriteLine($"{e.Label}: {e.Text} [{e.Start}, {e.End}) {e.Confidence:F3}");
}
```

The ONNX graph is not on the Hub: Starling converts the pinned checkpoint (reproducible export, digest `0ba2db95...`) and will host it as a release asset. Until that release exists, `Gliner2ModelAssets.DownloadAsync` fetches `tokenizer.json` and reports the graph as not found; place the exported graph at `onnx/gliner2-entities-fp32.onnx` in the snapshot directory or call `Gliner2ModelAssets.OpenDirectory(dir)`.

## Contract

This is the contract a Stable `gliner2-base-v1/onnx-fp32-cpu` profile would commit to. Today the profile is **Experimental**. The statuses below are the current ones, from the registry manifest and `Gliner2Capabilities.All`. Support levels are defined in the [support policy](https://github.com/starlinghq/starling/blob/main/docs/support-policy.md).

| Item | Contract |
| --- | --- |
| Model | `fastino/gliner2-base-v1` at `f9634218e53580c56edf0de97ca1a7d3f1c2354e`; GLiNER2 source `55656fbfa01d3d4a77485e1a1eeeaf682990ccdf`. The ONNX graph is Starling's reproducible export (`onnx/gliner2-entities-fp32.onnx`, digest in `Gliner2ModelAssets`). |
| Tokenizer | `Starlings.Tokenizers` `0.3.0-preview.1`, profile `gliner2-base-v1-unigram` (fast-tokenizer semantics; byte-fallback pieces map to `[UNK]` as GLiNER2 does). |
| Runtime | ONNX Runtime 1.30.0, fp32. CPU; CUDA with `use_tf32=0` (draft manifest `onnx-fp32-cuda` in draft PR #6). |
| Input | English text and an ordered set of entity labels, threshold (default 0.5). One label set per call (also per batch). |
| Limits | At most 512 tokens, schema prefix included; longer texts throw `Gliner2InputTooLongException`, and nothing is truncated. Spans up to 8 words. |
| Output | Entities with label, text, confidence and offsets into the original text: `Start`/`End` in UTF-16 code units, and `ScalarStart`/`ScalarEnd` in Unicode scalar values. `Text == input[Start..End]` always. Order: labels in request order, then descending confidence. Per label, confidence-first suppression of overlapping spans; spans of different labels may overlap. |
| Numerical bound | The protected tolerances of the profile (span logits relative 1e-4; order and threshold ties within 2e-5 of the reference, per the maintainer decision of 2026-10-10). |
| Supported | None yet. |
| Experimental | entity extraction (CPU and CUDA), the staged API, threshold, original-text offsets, batch, cancellation, concurrent calls (serialized on CUDA by default), label descriptions, the closed count gate, offline asset resolution, Native AOT (CPU and CUDA natives). Linux x64. |
| Untested | Windows. |
| Unsupported | inputs over 512 tokens, unserialized concurrent calls on CUDA (wrong entities in 6 of 8 test runs), truncation, the `max_len` word cap, `extract_long`, GLiNER2's plain de-duplicated output, mixed label sets in one batch, classification, relation extraction, structured (JSON) extraction, per-label thresholds, overlap policies, the character splitter, non-English text, TF32, fp16, quantized graphs, streaming, persistence, training, Microsoft.Extensions.AI, ML.NET (`unsupported` from draft PR #6; no adapter exists). |

## Memory on Linux

With default glibc settings, a process that loads and unloads the extractor repeatedly keeps freed native memory resident: about 1.2 to 1.5 GiB between loads, rising by about 65 to 90 MiB over the first half hour of cycling and then holding. This is allocator retention, not a leak. To bound it, start the process with `MALLOC_MMAP_THRESHOLD_=65536`; in a soak this held the footprint after unloading at about 76 MiB and lowered the loaded footprint by about 740 MiB. Evidence: [GLiNER2 soak growth](https://github.com/starlinghq/starling/blob/main/docs/evidence/gliner2-soak-growth/README.md).

## Status

Profile `gliner2-base-v1/onnx-fp32-cpu`: Linux x64 CPU and CUDA (`use_tf32=0`) experimental against the protected development fixtures and binding tolerances (one workstation). Evidence and reproduction: [ai-dotnet/README.md](https://github.com/starlinghq/starling/blob/main/ai-dotnet/README.md#gliner2-entity-extraction-p-015). License of the weights and the converted graph: Apache-2.0 (model card); GLiNER2 by Fastino AI, backbone initialised from `microsoft/deberta-v3-base` (MIT).

## Path to Stable

The order of promotion (`docs/plan/production-readiness.md`) takes this profile to Preview first. Gate evidence is in the [catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md) once draft PR #4 lands. Agents produce evidence; only the maintainer records a gate or changes maturity.

- [ ] G0 provenance: today a checklist ([P-014 spike](https://github.com/starlinghq/starling/blob/main/docs/m0/gliner2-p014.md)). It needs the hosted graph: a public release asset that replaces `source_status: placeholder` (M3), and sign-off on the license stated only in the model card (M4).
- [ ] G1 and G2: today checklists. They passed on development fixtures ([.NET evaluator run](https://github.com/starlinghq/starling/blob/main/docs/evidence/p015-gliner2-dotnet/README.md)). The rerun at tokenizer `0.3.0-preview.1` passed, span logits relative 2.4e-5 against a 1e-4 bound (draft PR #6). Still needed: a run on the protected P-030 NER release set (M1).
- [ ] G3 task quality: the NER span-F1 evaluator is in draft PR #5. It needs a licensed NER dataset and a budget in `protected/` (M1, M2).
- [ ] G4 resource correctness: one-hour soak and the plan 10.2 operational tests (R1).
- [ ] G5 performance: the benchmark harness is in draft PR #7 (its run is uncontrolled and is not G5 evidence). Needs a controlled run against a budget (M2, M6).
- [ ] G6 distribution: the release drill on a seeded graph is a checklist (draft PR #6). Needs acquisition from public hosting (R9 after M3), and Windows x64 (R6, R9).
- [ ] G7 integration: no adapter exists. Needs a representative downstream use, or the scope stays without integration badges (M7).
- [x] `label_descriptions`, `count_gate_closed`, `aot`: experimental ([capability closure](https://github.com/starlinghq/starling/blob/main/docs/evidence/capability-closure-2026-10-14/README.md); the capability cases are proposed, not yet accepted fixtures). The validator refuses Stable with any `untested` capability.
- [ ] CUDA profile: the GPU run of the protected evaluator for the draft `onnx-fp32-cuda` manifest (M6).
- [ ] Qualification CI (P-018, R6), the compatibility matrix rows, the support policy, and a second release owner (M7).
