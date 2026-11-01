# Support policy

Status: **draft for maintainer review** (workstream R10, `docs/plan/production-readiness.md`). This document states what each maturity label commits the project to. It does not change any label; maturity changes only when the maintainer records gate results (M10). Thirteen profiles are Stable. Since 2026-10-15: `gliner2-base-v1/onnx-fp32-cpu` (`docs/kickoff.md`, "First Stable profile"), `multilingual-e5-small/onnx-fp32-cpu` and `embeddinggemma-2/onnx-fp32-cpu` ("Stable CPU profiles, second round"). Since 2026-10-16: `embeddinggemma-2/image-fp32-cpu` and `qwen3.5-0.8b/ortgenai-fp32x-cpu` ("Stable CPU profiles, third round"), and on Linux x64 CUDA with an NVIDIA GeForce RTX 3090 `gliner2-base-v1/onnx-fp32-cuda`, `multilingual-e5-small/onnx-fp32-cuda`, `embeddinggemma-2/onnx-fp32-cuda`, `embeddinggemma-2/image-fp32-cuda`, `qwen3.5-0.8b/ortgenai-fp32x-cuda` and `qwen3.5-0.8b/ortgenai-fp16x-cuda` ("Stable CUDA profiles, first round"). Since 2026-10-17: `embeddinggemma-2/video-fp32-cpu` and, on the RTX 3090, `embeddinggemma-2/video-fp32-cuda` ("Stable video profiles"). Where this document proposes a number or a rule that the plan does not fix, it says so and lists it under [open questions](#open-questions).

Sources: plan sections 9, 10, 14 and 16 (`docs/plan/fast-faithful-ai-dotnet.md`), `.github/SUPPORT.md`, `.github/SECURITY.md`, `docs/kickoff.md`.

## What is supported

Support attaches to a **combination**, not to a model name. A combination is the four identities of plan section 9:

1. the package version (`Starlings.Models.E5` `0.2.0`);
2. the model bundle: the asset lock the package ships (repository, full commit SHA, file sizes and SHA-256), and the owned tokenizer package version in `processing.tokenizer.version`;
3. the runtime profile: backend and native package versions, provider, precision, OS and architecture;
4. the evidence: fixtures, tolerances, task-quality sets and benchmark protocol that qualified it.

The registry manifest (`model-pipeline/registry/<family>/<profile>/manifest.yaml`) records 2 to 4, and the release record (`docs/releases/ai-<version>.md`) ties them to 1. A combination that no manifest and release record describe is unsupported, whatever the label of its parts.

Each capability and each platform has its own status: `supported`, `unsupported`, `experimental` or `untested`. A missing value never means supported. A profile's maturity is the ceiling: an `experimental` capability inside a Stable profile carries no Stable commitment.

## Maturity levels in practice

Response and window figures are internal operating targets, not commercial SLAs (plan section 16). Support covers staffed work only. The windows marked *proposed* are R10 proposals awaiting the maintainer.

| Maturity | What a user can rely on | API and behavior changes | Fixes go to | Support window |
| --- | --- | --- | --- | --- |
| **Experimental** | Nothing beyond the recorded evidence. The profile may change or be removed in any release. | Any release, noted in the release record. | The next release only, best effort. | None. |
| **Preview** | The declared scope passes meaningful tests (plan 14): G0, G1, G2 and G6 in full, G3 to G5 and G7 as checklists. Published bytes behind a pinned identity never change. | Allowed between preview versions, with release notes and a migration note. | The next preview version. | Until the next preview version (*proposed*). Earlier previews stay installable and their locks stay resolvable, but they get no fixes. |
| **Stable** | The documented contract (inputs, outputs, limits, unsupported features) in the package README, every gate G0 to G7 passed in full, a named owner, rows in the compatibility matrix, and regression support (below). | Breaking API changes only in a new major version. An output-changing fix (tokenizer, template, preprocessing) ships as a new qualified profile and a release note, even with no public C# change (plan 14). | The latest minor version of the current major. Security and correctness fixes also go to the previous minor version. | The previous minor version gets security and correctness fixes for 6 months after the next minor ships (*proposed*). |
| **Maintenance** | Existing pinned combinations stay supported: security and correctness fixes, and the scheduled qualification keeps running. New upstream features are not pursued (plan 14). | None except fixes. | Patch versions of the maintained line. | Until `maintenance.support_ends` in the manifest, at least 6 months after entering Maintenance (*proposed*). |
| **Deprecated** | A replacement (`maintenance.replaced_by`), migration information, and a support end date (`maintenance.support_ends`) are published (plan 14). | None. | Security fixes only, until the end date. | Until `support_ends`. For a profile that was Stable, at least 6 months after the deprecation notice (*proposed*). |

Maturity belongs to profiles, not packages (`docs/kickoff.md`, "ai-dotnet package versions and maturity", 2026-10-15). One package version serves profiles of different maturities, so no package carries a blanket label: each package README has a maturity table generated from the registry manifests (`model-pipeline catalog`, checked in CI), and the package description points to it. The packages stay at `0.x`, so the public API may still change between minor versions; a Stable profile's contract does not.

### Public API deprecation

The packages are at `0.x`, so a public member may be removed or change shape between minor versions. It never does so without warning (*proposed*; see [open questions](#open-questions)):

1. **One minor version of notice.** A public type, member, parameter or enum value that minor version N+1 removes or changes incompatibly is marked `[Obsolete]` in minor version N, with the replacement added in the same release. A patch release never removes a public member.
2. **The message names the replacement and the removal.** The form is `[Obsolete("Use X; removed in the next minor version.")]`, where `X` is the replacing member, named in full. Where nothing replaces it, the message says what to do instead. The attribute is a warning, not an error (`error: false`), so existing code still compiles; a consumer that treats warnings as errors sees the build break at the upgrade, with the message.
3. **Behavior is unchanged while obsolete.** A member marked `[Obsolete]` behaves as before until it is removed, and keeps its XML documentation, which names the replacement too.
4. **The release record lists them.** Each release record (`docs/releases/ai-<version>.md`) lists every member it marks `[Obsolete]`, with its replacement, and every member it removes, with the release that marked it. A removal that no earlier release record announced is a release blocker. [`releases/BREAKING.md`](releases/BREAKING.md) collects these lists across versions, with the members pending removal; the release-candidate change adds its version's section there.
5. **Exceptions** are breaking changes that cannot keep the old member working: a security or correctness fix where the old behavior is unsafe or wrong. The release record states the exception, the reason and the migration.
6. **From 1.0 on**, a member marked `[Obsolete]` in a minor version is removed only in the next major version.

The window covers the C# API. Model outputs follow the profile rules above: an output-changing fix is a new qualified profile and a release note, whatever happens to the API.

### Promotion and demotion

- **Promotion** happens only when the maintainer records gate results in the manifest (M10). Agents produce evidence and tooling; they never set a gate to `passed` or raise maturity. The manifest validator refuses `stable` while any gate is not `passed` or any capability is `untested` (`model-pipeline/src/model_pipeline/manifest.py`). The catalog's Stable readiness column lists what still fails, and notes that a CUDA profile is not promoted until the per-release GPU run (the workflow's `cuda (runpod)` job, below) is green for it. The first promotion (GLiNER2 CPU, 2026-10-15) was made on the maintainer's instruction; the agent change set `maturity: stable` only where every gate was recorded passed and the scheduled regression runs were green.
- **Demotion** from Stable to Preview or Experimental needs a release note and a visible catalog change. It never rewrites what an earlier release promised: the earlier release record stays as published.

## Security and correctness issues

A security or correctness problem marks a release **affected**. It never silently rewrites the user's dependency lock (plan 14). In practice:

1. **Published bytes are immutable.** A package version, an asset lock and a hosted graph are never replaced behind an existing identity. A fix is a new package version and, when bundle files change, a new lock.
2. **Advisory record.** Each problem gets one record, published as a GitHub security advisory for vulnerabilities (`.github/SECURITY.md`) or a regression issue for correctness. It names the affected identities: package version range, profile IDs, lock commit and file digests, tokenizer version, provider and platform. It also states severity, workaround, the fixed version, and the evidence of the fix. A record under `docs/advisories/` is proposed so the catalog can link it (*proposed*; see open questions).
3. **Manifest and catalog.** The affected profile gets a `maintenance.known_issues` entry that links the record. The generated catalog shows it. The schema has no field for affected version ranges yet (*open question*).
4. **NuGet.** Affected package versions are marked deprecated on nuget.org (reason "critical bugs" or "other", with the fixed version as the alternative). They are not deleted. Deprecation and unlisting leave existing `packages.lock.json` restores working, so a user's build does not change until they choose to update.
5. **Runtime behavior does not change.** The runtime is local and offline after explicit acquisition, and an inference call never downloads (plan 14). Nothing phones home to check advisories. Users learn about advisories through NuGet's deprecation warnings at restore time, the catalog, and the advisory itself.
6. **Withdrawn upstream assets.** If an upstream repository deletes a pinned revision, or hosting has to be withdrawn (for example for legal reasons), the old lock can no longer download. `ModelAssetNotFoundException` says so explicitly, and the record names the replacement. Already-acquired caches keep working offline. The project cannot guarantee that third-party revisions stay available.

## Tokenizer bumps

Fixed by the plan (section 14):

- The owned tokenizer `Starlings.Tokenizers` is released independently. Each production model lock records the qualified tokenizer version (`processing.tokenizer.version`), and the pin lives in `ai-dotnet/Directory.Packages.props`.
- A newer tokenizer release is not automatically certified for existing bundles.
- A tokenizer change that alters outputs for a profile requires a new qualified model profile and a release note, even when no public C# method changes.

**Open for the maintainer (M8):** when a tokenizer bump leaves a profile's outputs unchanged, what must happen before the manifest records the new version? Two options:

| Option | What runs | Cost | Risk |
| --- | --- | --- | --- |
| A. Rerun model G1/G2 | `model-pipeline evaluate --require-gates G1,G2` for every profile on the new tokenizer (what R4, PR #6, did at `0.3.0-preview.1`: every CPU profile passed on the development fixtures). | One evaluator run per profile. CUDA profiles need GPU time (M6). | Low. Catches interactions the conformance suite does not exercise (offsets, batching, prompt prefixes). |
| B. Accept the tokenizer conformance suite | The tokenizer repository's conformance fixtures for the profile's `profile_id` pass unchanged, and the release notes say which profiles they cover. | None in this repository. | A component the profile uses could change in a way the fixtures do not cover. This is acceptable only if the suite covers every component the profile's `tokenizer.json` uses. |

A middle path is possible: B for patch releases whose diff does not touch a component the profile uses, A otherwise. Until the maintainer decides, this document makes no rule. The repository's practice so far is a tokenizer-suite check at each bump, plus one model rerun (R4) at `0.3.0-preview.1`. `0.3.0`, pinned by `0.2.0`, is the same code without the preview suffix; its conformance fixtures pass unchanged (`docs/releases/ai-0.2.0.md`). `0.3.1` (PR #128, for bge-m3) implements the `lstrip` and `rstrip` added-token options, which `0.3.0` refused at load time; no pinned asset of an existing profile declares either, so their code path is unreachable for those profiles, and the conformance fixtures pass unchanged (starlinghq/tokenizers#13). `0.3.2` (starlinghq/tokenizers#16, for Whisper) adds the GPT-2 split (`ByteLevel` `use_regex: true`) and the `whisper-bpe` profile; no pinned asset of an existing profile declares `use_regex: true`, so that path is unreachable for them, and the conformance fixtures pass unchanged. `0.3.3` (starlinghq/tokenizers#17, #18, #19, for LFM3) adds the `Sequence` post-processor and the Llama 3 split regex, both of which `0.3.2` refused at load, and restructures post-processing to upstream's merge-once shape; no pinned asset of an existing profile declares either component, and the conformance fixtures pass unchanged. The post-processing restructure is reachable by every profile, so for it this bump rests on those fixtures alone (option B): no model was rerun on 0.3.3 or 0.3.4 except LFM3. `0.3.4` (starlinghq/tokenizers#20) adds only the `clover-bpe-chat` profile, with no library change. Each is the middle path above; the maintainer's M8 decision still stands open.

## Compatibility matrix

The compatibility matrix is generated, never hand-kept: `model-pipeline catalog` writes `docs/catalog/compatibility-matrix.md`, `docs/catalog/README.md` and `docs/catalog/catalog.json` from the manifests' `compatibility`, `capabilities`, `execution` and `qualification` fields (P-021, #4). CI fails when the committed output is stale. A Stable support claim covers exactly the matrix rows with status `supported` for that profile. `unrecorded` and `untested` cells are not claims. Linux x64 CPU is `supported` for every Stable CPU profile, and Linux x64 CUDA for every Stable CUDA profile (on the RTX 3090; other GPUs are untested). The Windows x64 CPU rows of `gliner2-base-v1/onnx-fp32-cpu`, `multilingual-e5-small/onnx-fp32-cpu` and `embeddinggemma-2/onnx-fp32-cpu` are `experimental` (CI weight-free evidence only). Every other row is `untested` or `unrecorded`; macOS and ARM64 are untested.

Matrix rows beyond Linux x64 need qualification runs on that platform (P-018, workstream R6). CI has a Windows x64 build, weight-free test and clean-install job (`ai-dotnet (windows x64)`), but the scheduled qualification runs on Linux x64 only.

## Regression support

Regression support applies to **Stable** profiles only (`.github/SUPPORT.md`). Earlier maturities get best-effort fixes. The qualification job `experimental` runs the onboarding routes of the Experimental CPU profiles of Kev-0.8B, QwenChat and Whisper on the same schedule; it reports (a warning and the step summary) and never fails the workflow, and it is not regression support.

A regression is any of the following, on a supported matrix row:

- an output outside the profile's protected tolerances or task-quality budget;
- a G5 benchmark beyond the profile's performance budget;
- a G4 operational failure: a leak, a hang, unbounded growth, or state leakage between requests;
- an installation or offline-execution failure (G6).

Commitments:

1. **Intake.** Reports use the regression template (`.github/ISSUE_TEMPLATE/regression.yml`), with the four identities from the diagnostics and a reproducer. Triage, including affected profiles, severity and workaround, has an internal target of 5 working days (*proposed*; `.github/SECURITY.md` uses the same figure for acknowledgement).
2. **Detection without reports.** The scheduled qualification workflow (`.github/workflows/qualification.yml`, R6) runs **CPU regression weekly** (Mondays, on hosted runners; also on every release tag). It acquires the pinned public models anonymously and verifies them, runs the model tests, reruns G1, G2 and G3 of every Stable CPU profile on its accepted release sets (job `release-gates`, `tools/release-gates.sh`; bounds and budgets only from `protected/`), runs a one-hour soak per CPU profile (GLiNER2 with its documented `MALLOC_MMAP_THRESHOLD_=65536`), and uploads evidence. A failure there is handled like a report. **Without it there is no regression support, and so no Stable profile.**

   **CUDA profiles** (e5, GLiNER2, EmbeddingGemma 2 text, image and video, Qwen3.5 fp32x and fp16x) get the same detection from the workflow's `cuda (runpod)` job. **GPU regression runs for every release** (a push of an `ai-v*` release tag) **and on demand** (manual dispatch), not weekly; never on pull requests or in forks. It creates one on-demand RunPod GPU pod (an RTX 3090 only, secure cloud EU-CZ-1; the CUDA contracts cover the RTX 3090, so the job fails rather than run on another GPU) with the network volume that holds the .NET SDK, uv, the pinned CUDA 13 and cuDNN 9 wheels and the verified models. The pod checks out the exact commit and runs `tools/ci/runpod-cuda-pod.sh`:

   - the CUDA test projects of e5, GLiNER2, EmbeddingGemma 2 and Qwen3.5 (`ai-dotnet/tests/Starlings.Models.*.Tests.Cuda`; the pod does not run Kev's);
   - G1, G2 and G3 of every Stable CUDA profile on its accepted release sets, with bounds and budgets only from `protected/`, the commands of `docs/evidence/release-set-gates-2026-10-14/`, and TF32 off;
   - a short soak smoke test per Stable CUDA profile (10 minutes by default), under the profile's contract environment: `LD_LIBRARY_PATH` from the pinned nvidia wheels, and `MALLOC_MMAP_THRESHOLD_=65536` where the profile states it as a Linux deployment requirement (GLiNER2, EmbeddingGemma 2 video).

   The results are uploaded as the artifact `qualification-cuda`. Each run logs the pod id, GPU and cost, and it ends by terminating the pod and confirming through the RunPod API that no `starling-q-ci-*` pod remains. A run refuses to start below a $5 account balance; a typical run costs about $1 to $2.

   Release inputs outside Git (media from Wikimedia Commons, the pinned FFmpeg builds, and reference tensors regenerated at the recorded thread count) are cached on the volume after the first run. The producers and the evaluator check each one against the digests in Git, so a cached input can make a check fail but never pass. A missing input is reported as `not-run`, which fails the job. Every check must pass for the job to be green, and a failure is handled like a report.

   The short soak is a smoke test, not G4: wrong outputs, errors and crashes fail it. The soak tool's growth verdict (its rule is unchanged) is recorded but not binding there, because a short run sits on the allocator's warm-up ramp (`docs/evidence/cuda-soak-growth/`; the binding one-hour Qwen fp32x CUDA soak ramps 18.4 then 10.5 MiB per half-hour, 27.8 MiB in all). Growth is judged by the profile's one-hour G4 soaks.

   A CUDA profile has regression support once this job has run green for it on the release. Promotion still needs every gate recorded `passed` by the maintainer (M10).
3. **Fix.** The fix ships as a new patch version. It reruns every gate the change affects (the scoping table in `docs/plan/maintained-update-drill.md`), and the release record links the evidence. If the fix changes outputs, it is a new qualified profile (plan 14).
4. **Regression test.** The reproducer becomes a test. A new case in protected fixtures is proposed in a separate change, with its reference source, and committed by the maintainer (plan 10.1). Agents never edit `model-pipeline/protected/` or tolerances.
5. **Affected releases** are marked as in [Security and correctness issues](#security-and-correctness-issues).

## Owner

| Role | Holder | Status |
| --- | --- | --- |
| Accountable maintainer: technical lead, API owner and final release reviewer | Mark | Settled (`docs/kickoff.md`) |
| Owner of every registry profile (`maintenance.owner`) | Mark | Recorded in every manifest |
| Second release owner | **open** | M7. Plan 16 requires a second trusted human release owner before broad support is promised. Protected branches and the `release` environment must not let an implementation agent approve its own change. |
| First named consumer | **open** | M7, P-028. Needed for G7 (representative downstream use). |

The single-owner gap is a Stable blocker in `docs/plan/production-readiness.md` (M7). Until a second release owner is named, any Stable promotion carries a bus factor of one. The maintainer chose not to block the Stable labels on it: every promotion records a single-owner exception (`docs/kickoff.md`).

## Open questions

These are for the maintainer. None is settled by this document.

1. **M8, tokenizer bumps:** option A, B or the middle path (see [Tokenizer bumps](#tokenizer-bumps)).
2. **Support windows:** confirm or change the proposed figures: 6 months for the previous Stable minor, at least 6 months for Maintenance and Deprecated, the 5-working-day regression triage target, and a weekly scheduled qualification.
3. **Stable and 1.0:** settled 2026-10-15: a `0.x` package can carry a Stable profile; maturity is per profile and the next version is `0.2.0` (`docs/kickoff.md`).
4. **Second release owner (M7):** the first Stable promotion (GLiNER2 `onnx-fp32-cpu`, 2026-10-15) ships with a recorded single-owner exception (`docs/kickoff.md`); every later promotion and the `0.2.0` release (`docs/releases/ai-0.2.0.md`) carry the same exception. M7 stays open.
5. **Advisory records:** use GitHub advisories only, or also `docs/advisories/` records that the catalog links? Should the manifest schema gain a field for affected version ranges?
6. **Old and new profile identities** after an upstream update: see `docs/plan/maintained-update-drill.md`, open questions.
7. **API deprecation window:** confirm or change the proposed [public API deprecation](#public-api-deprecation) rule: one minor version with `[Obsolete]` (a warning naming the replacement) before a removal at `0.x`, removal only in a new major version from `1.0` on, and the release record listing both.
