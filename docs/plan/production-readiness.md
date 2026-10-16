# Production readiness plan

Goal: take every registered model to **Stable** (plan section 14): documented contract, all qualification gates G0 to G7 passed in full (plan section 10), an owner, a compatibility matrix, and regression support. The manifest validator enforces part of this already: a `stable` profile needs every gate `passed` and no `untested` capability (`model-pipeline/src/model_pipeline/manifest.py`).

Audit date: 2026-10-13, at `fb27658`. Baseline on the workstation: 711 .NET tests, 473 pass, 0 fail, 238 skipped (weight-dependent tests skip without a local model cache).

## Starting point

| Profile | Maturity | G0 | G1 | G2 | G3 | G4 | G5 | G6 | G7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multilingual-e5-small `onnx-fp32-cpu` | preview | passed | passed* | passed* | checklist | checklist | checklist | passed | checklist |
| multilingual-e5-small `onnx-fp32-cuda` | experimental | checklist | not-run | not-run | none | none | none | not-run | none |
| gliner2-base-v1 `onnx-fp32-cpu` | experimental | checklist | checklist* | checklist* | none | none | none | none | none |
| qwen3.5-0.8b `ortgenai-fp32x-cpu`, `-fp32x-cuda`, `-fp16x-cuda` | experimental | checklist | checklist | checklist | none | none | none | none | none |
| embeddinggemma-2 text/image/video × cpu/cuda (6) | experimental | checklist | not-run | not-run | none | none | none | none | none |

\* Passed on agent-written development fixtures, and at tokenizer `0.1.0-preview.4`. The pin is now `0.3.0-preview.1`, and the model gates were not rerun.

Browser profiles for e5 (`onnx-{fp32,int8embed}-{wasm,webgpu}`) have protected tolerances but no registry manifest or package yet.

### With the wave 1 draft PRs (2026-10-13)

The audit table above is the baseline at `fb27658`. The table below shows the gate entries once draft PRs #4 to #7 merge. No gate became `passed` and no maturity changed: the manifests record `not-run` or `checklist` with evidence, and recording results stays with the maintainer (M10).

| Profile | Maturity | G0 | G1 | G2 | G3 | G4 | G5 | G6 | G7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multilingual-e5-small `onnx-fp32-cpu` | preview | passed | passed† | passed† | checklist | checklist | checklist | passed | checklist |
| multilingual-e5-small `onnx-fp32-cuda` | experimental | checklist | not-run | not-run | not-run | not-run | not-run | not-run | not-run |
| gliner2-base-v1 `onnx-fp32-cpu` | experimental | checklist | checklist† | checklist† | not-run | not-run | not-run | checklist | not-run |
| gliner2-base-v1 `onnx-fp32-cuda` (new manifest, draft PR #6) | experimental | checklist | not-run | not-run | not-run | not-run | not-run | not-run | not-run |
| qwen3.5-0.8b `ortgenai-fp32x-cpu` | experimental | checklist | checklist† | checklist† | not-run | checklist | not-run | checklist | not-run |
| qwen3.5-0.8b `ortgenai-fp32x-cuda`, `-fp16x-cuda` | experimental | checklist | checklist | checklist | not-run | checklist | not-run | not-run | not-run |
| embeddinggemma-2 text, image, video `-cpu` (3) | experimental | checklist | not-run† | not-run† | not-run | not-run | not-run | checklist | not-run |
| embeddinggemma-2 text, image, video `-cuda` (3) | experimental | checklist | not-run | not-run | not-run | not-run | not-run | not-run | not-run |

† G1/G2 rerun at tokenizer `0.3.0-preview.1` on the development fixtures (draft PR #6): all CPU profiles passed. e5 2.32e-7, GLiNER2 span logits relative 2.4e-5, EmbeddingGemma 2 text 3.7e-7 and image 2.24e-6; Qwen fp32x greedy 22/22 and step-0 1.38e-3. Video was not rerun. The gate entries themselves are unchanged, because these are not P-030 sets. The new `checklist` entries (G6 for the CPU profiles, from the `ai-0.1.0-preview.5` release drill on seeded graphs; G4 for Qwen, from P-016 sections 4 and 5) and the G3 to G7 `not-run` entries come from draft PR #6, which also sets GLiNER2 `mlnet` to `unsupported`.

Other wave 1 results:

- **Catalog and compatibility matrix (draft PR #4):** `model-pipeline catalog` generates `docs/catalog/` from the manifests, with a Stable readiness column, and CI checks it for staleness. It becomes the source of truth for status once merged. Whichever of #4 and #6 merges second regenerates it.
- **G3 evaluators (draft PR #5):** retrieval (nDCG@10, recall@k, MRR, int8 top-10 overlap), NER span F1, and the machine-checked Qwen generation task (G1 and G2 passed on the development fixtures). All G3 runs report "no budget: not a gate result" until the maintainer commits `task-quality.yaml` and datasets (M1, M2).
- **Benchmark harnesses (draft PR #7):** GLiNER2, EmbeddingGemma 2 (text and image, no video) and Qwen harnesses, a shared core, and `docs/benchmarks/PROTOCOL.md`. The development runs are uncontrolled (load average 10 to 22) and are not G5 evidence.
- **Stable contract docs (R10, draft PR):** `docs/support-policy.md`, the Stable contract and Path to Stable sections in the four model package READMEs, `docs/plan/maintained-update-drill.md`, and a read-only upstream change detector (`model-pipeline upstream`, P-020).

### Blockers common to all models

1. **No protected release sets (P-030).** Every G1/G2 pass so far is on development fixtures. No model has a held-out task metric (G3). Only the maintainer may author these sets.
2. **Hosting.** The GLiNER2, EmbeddingGemma 2 and Qwen graphs sit in private Hugging Face repositories (`source_status: placeholder`), so no clean install can acquire them (G0, G6).
3. **No qualification CI (P-018).** CI is `ubuntu-latest` only, model tests skip there, there is no Windows job, and nothing runs on a schedule. That means no compatibility matrix evidence beyond Linux x64, and no regression support.
4. **No soak harness (plan 10.2).** No one-hour repeated-request and load/unload run exists for any model (G4).
5. **Benchmarks.** Only e5 has a benchmark (one uncontrolled workstation). GLiNER2, EmbeddingGemma 2 and Qwen have none (G5). Update: harnesses for all three are in draft PR #7. Controlled runs and budgets are still missing (M2, M6).
6. **Nothing is published.** The `release` environment, NuGet prefix reservation and branch protection are still manual steps (`docs/kickoff.md`).

## Order of promotion

Follow plan section 17: one Stable model first, the others to Preview, then Stable as evidence lands.

1. **e5 `onnx-fp32-cpu` to Stable.** Closest; it needs G3, G4, G5 and G7 formalized, `aot` resolved, a G1/G2 rerun, Windows evidence and qualification CI.
2. **GLiNER2 CPU, Qwen fp32x CPU, EmbeddingGemma 2 text CPU to Preview.** They need hosting, the G1/G2 rerun on P-030 sets, and G3 to G7 checklists.
3. **CUDA profiles to Preview.** These need GPU runs of the protected evaluator; EmbeddingGemma 2 also needs W2 (the CUDA evaluator mode).
4. **The rest to Stable,** as each meets the full gate set. Qwen fp16x CUDA likely stays Preview, because of near-tie divergences (18/22 greedy).

## Agent workstreams

Each workstream is one draft PR. Agents never edit `model-pipeline/protected/` or `tolerances.*`, and never flip a gate to `passed` or a maturity upward; they produce the evidence and tooling, and the maintainer records the result.

### Wave 1 (independent, run in parallel)

| ID | Workstream | Gates | Deliverables | Status |
| --- | --- | --- | --- | --- |
| R1 | Operational qualification harness | G4 | One soak tool for all four models (repeated requests, load/unload cycles, working-set/native/VRAM sampling, plateau report; one-hour default, short mode for CI). Missing plan 10.2 tests per model: queue saturation, timed-out requests, failed init, malformed lock or manifest, unsupported provider, cancellation then reuse, disposal during shutdown, cancellation latency. | open |
| R2 | Benchmark harnesses | G5 | GLiNER2, EmbeddingGemma 2 (text/image/video) and Qwen generation harnesses modeled on `Starlings.Benchmarks.E5Harness`: raw results, environment manifest, warm/cold split, stage timing. Protocol doc. | draft PR #7 |
| R3 | Catalog and compatibility matrix | P-021, section 14 | Generator in `model-pipeline` producing the public catalog and a compatibility matrix from the manifests (no hand-kept list), checked in CI for staleness. | draft PR #4 |
| R4 | Manifest hygiene and gate records | G0 to G7 | G3 to G7 entries (`not-run` or `checklist`, with evidence links) in every manifest. Stale text fixed (EmbeddingGemma 2 FFmpeg.AutoGen references, Qwen hosting text). GLiNER2 `mlnet` corrected to `unsupported`. GLiNER2 `onnx-fp32-cuda` manifest drafted. G1/G2 rerun at tokenizer `0.3.0-preview.1` on the development fixtures, with evidence. | draft PR #6 |
| R5 | Task-quality evaluators | G3 | Evaluator tasks in `model-pipeline evaluate`: retrieval (nDCG@10 and recall@k; e5 and EmbeddingGemma 2; the int8 top-10 overlap metric for browser profiles), NER span F1 (GLiNER2), and a generation task with the template stage machine-checked against the protected tolerances (Qwen). Dataset paths come from the maintainer's P-030 sets; the agent supplies only the format, a public smoke example, and tests. | draft PR #5 |

### Wave 2 (after wave 1 merges)

| ID | Workstream | Gates | Deliverables | Status |
| --- | --- | --- | --- | --- |
| R6 | Qualification CI (P-018) | all | A Windows x64 CPU job. A scheduled qualification workflow that restores pinned models from a cache, runs the model tests, the protected evaluator and a short soak, and uploads evidence. Release credentials stay out of PR jobs. Maintainer approves `.github/` changes. | open |
| R7 | EmbeddingGemma 2 follow-up | G1, G2, G4 | W2 (CUDA mode for the fidelity tool), W1 (frame-call measurement, GPU), W3 (long-input fixture proposal, 2,049 to 8,192 tokens), W4 (FFmpeg 9 adapter). | open |
| R8 | Qwen template and LLM gaps | G1, G3, G4 | P-029 template corpus beyond the 21 prompts. Long-context tests above 1,666 tokens, or a narrowed claim. A drafted upstream report for the concurrent CUDA generator defect (the maintainer files it). | open |
| R9 | Distribution | G6 | AOT check for every core package (to `supported` or `unsupported`). A Windows clean-install drill. The ML.NET package run in the release drill. A download-from-public-hosting drill step, enabled once hosting is public. | open |
| R10 | Stable contract docs | section 14 | A support policy, a per-model Stable contract (the README "Stable waits for P-018" text replaced), and the M3 maintained-update and rollback drill (P-020, P-022). | draft PR (branch `r10-stable-contracts`) |

## Maintainer decisions and actions

These cannot be done by an agent. Each row names what it unblocks.

| ID | Action | Unblocks |
| --- | --- | --- |
| M1 | Curate P-030 release sets: tokenizer and retrieval sets for e5 and EmbeddingGemma 2 (text, image, video), a NER set for GLiNER2, and a generation set for Qwen. Commit them under `model-pipeline/protected/`. | G1 to G3 for every model |
| M2 | Choose G3 datasets (licensed) and quality budgets, and G5 performance budgets per profile. | G3, G5 |
| M3 | Make graph hosting public: GitHub release assets (`embeddinggemma-2-onnx-r1`, `-vision-onnx-r1`, GLiNER2) and the Qwen `ortgenai` build. Replace `source_status: placeholder`. | G0, G6 |
| M4 | Legal: D2 (FFmpeg adapter); sign-off on licenses stated only in model cards (e5 MIT, GLiNER2 Apache-2.0). | G0, adapter publish |
| M5 | Repository settings: branch protection, the `release` environment, the NuGet `Starlings` prefix reservation (`docs/kickoff.md`). | G6, any publish |
| M6 | Hardware: GPU time for CUDA evaluator runs and soaks, a quiet machine for controlled benchmarks, and Windows evidence (CI runner approval). | G2/G4/G5 on CUDA, compatibility matrix |
| M7 | Name a second release owner and a first named consumer (P-028). | Stable, G7 |
| M8 | Accept the e5 CUDA `PROPOSED.md` decisions. Decide whether a tokenizer bump requires a model G1/G2 rerun or whether the tokenizer conformance suite suffices (options in `docs/support-policy.md`, R10). | e5 CUDA, every tokenizer bump |
| M9 | File the upstream onnxruntime-genai concurrent-generator report (drafted in R8). Confirm CUDA serialization as the documented behavior. | Qwen G4 |
| M10 | Record gate results and promote maturity once evidence exists. | Every promotion |

## Tracking

Each workstream PR updates this document's tables when it lands. The catalog (R3) becomes the source of truth for status once it exists.

### Status 2026-10-16 (third promotion round)

- **Stable: EmbeddingGemma 2 `image-fp32-cpu` and Qwen `ortgenai-fp32x-cpu`.** Every gate passed and the validator agrees. The weekly qualification workflow now reruns their release-set G1 to G3 and the one-hour soak on hosted runners, green ([run 38848321624](https://github.com/starlinghq/starling/actions/runs/38848321624)). G2 reads the accepted reference tensors from `Starlinghq/starling-release-references`, pinned by the protected `hosted-reference.json` of each release set (#71). Qwen CPU G5 is recorded `passed` from the planned quiet-host rerun. Contracts: the `Starlings.Models.EmbeddingGemma2` README (images) and the `Starlings.Models.Qwen35` README. Same single-owner exception (M7 open).
- **Not promoted: EmbeddingGemma 2 `video-fp32-cpu`.** Its release-set G1 and G2 now pass weekly, but its hosted one-hour soak fails the growth rule in every run (about 280 MB/h). **Regression report:** the Stable text profile `embeddinggemma-2/onnx-fp32-cpu` failed its hosted soak in runs 38837950614 and 38841680114 (about 130 MB/h) and passed in 38831355609 and 38848321624 (intermittent).

### Status 2026-10-15 (second promotion round)

- **Stable: `multilingual-e5-small/onnx-fp32-cpu` and `embeddinggemma-2/onnx-fp32-cpu` (text).** Every gate passed, the validator agrees, and the weekly qualification workflow reruns their release-set G1 to G3 and a one-hour soak on hosted runners, green ([run 38822625245](https://github.com/starlinghq/starling/actions/runs/38822625245)). The blockers in the first-round entry below were resolved by #60 (EmbeddingGemma 2's text reference embeddings committed under `protected/`) and #66 (`malloc_trim` on session release; soak with transparent huge pages `madvise`). Contracts: the `Starlings.Models.E5` and `Starlings.Models.EmbeddingGemma2` READMEs. Single-owner exception as for GLiNER2 (M7 open).
- **Not promoted, though every gate is recorded passed: EmbeddingGemma 2 `image-fp32-cpu` and `video-fp32-cpu`.** The weekly workflow runs their soaks (the video soak was wired in this round) but not their release-set gates: G2 needs reference tensors kept outside Git that regenerate byte-identical only on the reference CPU. A maintainer decision is needed on how G2 can rerun off that CPU.
- **Not promoted: Qwen `ortgenai-fp32x-cpu`.** The manifest records G5 `failed`; the quiet-host rerun passed and qualifies (`docs/evidence/qualification-2026-10-15/`), but recording it is a maintainer change (M10). Its release-set gates are not in the weekly workflow either (full step-0 reference vectors outside Git).

### Status 2026-10-15 (first promotion)

- **First Stable profile: `gliner2-base-v1/onnx-fp32-cpu`** (#56 preconditions, then the maturity flip). Every gate passed; the weekly qualification workflow reruns its release-set G1 to G3 and a one-hour soak (with `MALLOC_MMAP_THRESHOLD_=65536`) on hosted runners, green. Single-owner exception recorded (M7 open).
- **e5 `onnx-fp32-cpu` and EmbeddingGemma 2 text `onnx-fp32-cpu`: every gate passed, Stable blocked.** e5's hosted one-hour soak fails the growth rule (about 50 MB/h, undiagnosed; the workstation hour plateaued). EmbeddingGemma 2's G2 cannot rerun off the reference CPU (outside-Git reference embeddings; maintainer action: commit them under `protected/`, SOURCE.md decision 2); its hosted soak also fails (about 160 MB/h).
- **CUDA profiles:** EmbeddingGemma 2 text, Qwen fp32x and Qwen fp16x pass every gate where recorded; e5 CUDA has G4 failed. Promotion of any CUDA profile awaits scheduled GPU regression runs (M6).
- **Versions:** maturity is per profile (generated README blocks); the next ai-dotnet version is `0.2.0` (`docs/kickoff.md`).

### Status 2026-10-14

- **Workstreams R1 to R10** are integrated in the `0.1.0-preview.6` release candidate (`docs/releases/ai-0.1.0-preview.6.md`), which merges draft PRs #3 to #12 and #14.
- **M3 done.** The four Starlinghq Hugging Face repositories are public at the pinned revisions; the code and manifests record `source_status: public`. They stay Hugging Face repositories rather than GitHub release assets, because acquisition already reads them.
- **M5 partly done.** The `release` environment exists (Mark as required reviewer, tags `ai-v*`, `NUGET_USER=MarkWalls`). Still open: the nuget.org trusted-publishing policy for repository `starling`, branch protection, and the `Starlings` prefix reservation if it was not requested with the tokenizer.
- **Interoperability.** Libraries integrate through `Microsoft.Extensions.AI.Abstractions` only; CI rejects cloud SDKs and AI provider packages in every library. Next agent workstream: tool calling and structured output in `Qwen35ChatClient`, the main gap for code ported from other `IChatClient` providers. Also next: chunked prefill as the CPU default for Qwen (R1 and R8 findings).
