# ai-dotnet: unreleased changes

Changes merged to `main` after the last release record, collected here for the next one. The release-candidate change moves them into `ai-<version>.md` and empties this file. Nothing here is published until then.

## Starlings.Models.E5: `concurrent_calls` recorded for the CUDA profile; README qualification runner (#258, #252)

- `E5Capabilities.ProfileCapabilities["multilingual-e5-small/onnx-fp32-cuda"].Capabilities` (and the obsolete `E5Capabilities.Profiles`) now contains `concurrent_calls`, `Supported`, generated from the registry manifest, which now states it explicitly. The status is not new: the README's "Stable contract: CUDA" and `E5Capabilities.Statuses["concurrent_calls"]` already said supported, and code that read a missing key as anything else now sees it. Evidence for Supported: the binding RTX 3090 G4 one-hour soak of this profile, which ran with 2 concurrent callers (`docs/evidence/qualification-2026-10-16-cuda/`). Extra local coverage: `docs/evidence/cuda-concurrency-2026-11-01/` (RTX 4060 Ti, a local check, not qualification; #257 tracks the RTX 3090 rerun of that check).
- The "Supported platform" rows of the E5, EmbeddingGemma 2, GLiNER2 and Qwen3.5 (ONNX Runtime GenAI) READMEs keep the GitHub-hosted `ubuntu-latest` record of the weekly qualification up to 2026-10-22 and add the self-hosted `qualification` runner it runs on since #134 (Intel Core i7-5820K, AVX2, no AVX-512, transparent huge pages `madvise`). No contract changes.
