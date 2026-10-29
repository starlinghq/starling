# Workflows

The four GitHub Actions workflows of this repository. This page is derived from the YAML files; the YAML and its comments are the authority. Scripts under `tools/` are described in [`tools/README.md`](../../tools/README.md).

| Workflow | Runs on | Secrets and environments |
| --- | --- | --- |
| [`ci`](#ci) | every pull request, pushes to `main`, manual dispatch | none |
| [`protected-evidence`](#protected-evidence) | pull requests, including label changes | none |
| [`qualification`](#qualification) | weekly schedule, `ai-v*` tags, manual dispatch | secret `RUNPOD_API_KEY` (job `cuda` only) |
| [`release`](#release) | `ai-v*` tags, manual dispatch (candidate only) | environment `release`, its variable `NUGET_USER`, OIDC for NuGet trusted publishing |

Every workflow sets `permissions: contents: read` at the top level. Only the `release` job `publish` asks for more.

## ci

`ci.yml`: fast pull-request checks (plan section 13.1). No secrets are used.

- **Triggers:** `pull_request`, `push` to `main`, `workflow_dispatch`. Concurrency group `ci-<ref>`, with `cancel-in-progress`.

| Job | Runner | What it checks |
| --- | --- | --- |
| `boundary` | ubuntu | The tests of the boundary tool (`python3 -I -m unittest discover -s tools/tests`), then `tools/check-boundary.py`: no name of an unrelated older project in files, file names or embedded data (the whole tree), or in the authors and messages of the commits since the baseline in `tools/check-boundary.baseline` (pushes) or of the pull request's commits (`--range <base sha>..<head sha>`). Full-history checkout. |
| `ai-release-drill` ("ai release-candidate drill") | ubuntu | `tools/release-drill-ai.sh --no-model`: packs from a clean clone, installs from a local folder feed into consumers outside the checkout, checks the restore graph, downloads e5's `tokenizer.json` and matches the fixtures. Nothing is published. |
| `ai-dotnet` | ubuntu | Fetches the pinned tokenizer assets (`tools/fetch-tokenizer-assets.sh`, cached by the script's SHA-256); builds `ai-dotnet/AI.slnx` (including tools and the CPU benchmark projects, which are never run); builds the samples `E5VectorStore`, `LocalRag` and `TicketTriage` from published packages in locked mode; runs `dotnet test --solution ai-dotnet/AI.slnx`. Model tests skip without the models. Then the library dependency boundaries for every `ai-dotnet/src/*/*.csproj` (below). |
| `ai-dotnet-windows` ("ai-dotnet (windows x64)") | windows | The same build and weight-free tests on Windows x64, then the Windows clean-install drill `tools/release-drill-ai.ps1 -NoModel`. |
| `model-pipeline` | ubuntu, Python 3.11 | In `model-pipeline/`: `ruff check .`, `ruff format --check .`, `pytest`, `model-pipeline validate registry`, `model-pipeline catalog --check` (the generated `docs/catalog/` is not stale) and `model-pipeline evidence-index --check` (the generated `docs/evidence/README.md` is not stale). |
| `browser-conformance-build` ("browser conformance app (build)") | as `ai-dotnet` | Publishes `browser/conformance/E5Conformance` (interpreter build, not in locked mode, as the qualification job `browser` does), so the app that qualifies the e5 browser profiles keeps building against the packages' source. No model and no browser run. |
| `docs` ("docs site (docfx)") | as `ai-dotnet` | `dotnet tool restore` (docfx, pinned in `.config/dotnet-tools.json`), then `dotnet docfx docs/docfx.json`: the API reference from the projects in `ai-dotnet/src/` and their XML docs, with the generated catalog, compatibility matrix, evidence index and the support policy. Uploaded as the artifact `docs-site`; nothing is deployed. |
| `repo` ("repository checks") | ubuntu | The backlog in `docs/plan/fast-faithful-ai-dotnet.md` parses (`tools/create_backlog_issues.py`); every CPU test project under `ai-dotnet/tests/` is in `AI.slnx`, so the `ai-dotnet` jobs run it (the `*.Tests.Cuda` projects stay local-only); no model weights committed (`.onnx`, `.safetensors`, `.gguf`, `.pt`, `.pth`, `.ckpt`, `.h5`); no `SixLabors.ImageSharp` in any project, props, targets or lock file; no `Sdcb.FFmpeg` or FFmpeg-binary package, and no committed FFmpeg library file. |

Library dependency boundaries (`ai-dotnet` job), checked on each library's transitive package list:

- No `Microsoft.ML.Tokenizers`, and no native ONNX Runtime or ONNX Runtime GenAI package (CPU or GPU): the application chooses the natives; libraries reference the managed bindings only.
- ML.NET only in `Starlings.Models.E5.MLNet` and `Starlings.Models.EmbeddingGemma2.MLNet`.
- SkiaSharp only in `Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp`.
- No cloud SDK or AI provider package (Azure, OpenAI, the `Microsoft.Extensions.AI` provider packages): libraries use `Microsoft.Extensions.AI.Abstractions` only.
- No third-party FFmpeg bindings package (`FFmpeg.AutoGen`, `FFmpegCore`, `Sdcb.FFmpeg`).
- No library references an FFmpeg video adapter project; adapters are referenced by applications only.

## protected-evidence

`protected-evidence.yml` (plan 10.1 and 12, backlog P-007). A pull request that changes protected fixtures, conformance expectations or tolerances must carry the `protected-change` label and must not also change implementation code, so an implementation change cannot move its own thresholds.

- **Triggers:** `pull_request` of types `opened`, `synchronize`, `reopened`, `labeled`, `unlabeled`, so it runs again when labels change.
- **Job `protected-evidence`** ("protected evidence changes"): full-history checkout, `pip install -e model-pipeline`, then `model-pipeline check-protected --base <base sha> --head <head sha> --labels-json <labels>`. It reads the event payload only; no secrets, read-only token.

## qualification

`qualification.yml`: scheduled qualification (plan section 13.1; backlog P-018; workstream R6 of `docs/plan/production-readiness.md`). It is the regression support of the Stable profiles (`docs/support-policy.md`). Nothing in it changes a manifest, a gate or a maturity; the maintainer records results (M10). Evidence from every job is uploaded as an artifact, also when a step fails.

- **Triggers:**
  - `schedule`: `23 4 * * 1` (Mondays 04:23 UTC): the CPU jobs. The `cuda` job does not run on the schedule.
  - `push` of tags `ai-v*`: every job, `cuda` included (GPU regression per release).
  - `workflow_dispatch`, with inputs `soak_duration` (default `1h`), `cuda_soak_duration` (default `10m`), `skip_cuda` (skip the paid RunPod job), and the diagnostics inputs `soak_only`, `soak_workloads` and `soak_variants`.
- Never runs on `pull_request`, so no fork code runs here. Concurrency group `qualification-<ref>`, not cancelled in progress.
- `CPU_PROFILES` lists six CPU registry profiles, the Stable ones; the Experimental Kev CPU profiles are not in it. The job `experimental` runs the onboarding routes of Experimental CPU profiles on their candidate contracts instead, and never gates. A profile whose manifest records a `source_status` of `placeholder` or `private` is skipped with a "hosting private" notice (`tools/qualification.py hosting`).
- Models are acquired anonymously in each job that needs them (`tools/acquire-models.sh`, size and SHA-256 verified), because all CPU models together exceed the actions cache limit.

| Job | Name | Condition | What it does | Artifact |
| --- | --- | --- | --- | --- |
| `assets` | model locks | repository is `starlinghq/starling` | One lock per public CPU profile (`tools/qualification.py lock`). | `qualification-locks` |
| `model-tests` | model tests (linux x64 cpu) | not `soak_only` | Acquires every CPU model and the tokenizer assets, builds `AI.slnx`, runs the .NET tests with the models present, one test module at a time. | `qualification-model-tests` |
| `evaluator` | evaluator `<profile>` | not `soak_only` | Per CPU profile with an onboarding recipe: the protected staged evaluator on the development fixtures, G1 and G2 required, plus the G3 task-quality evaluator on the public smoke example (not gate evidence). | `qualification-evaluator-<n>` |
| `browser` | browser e5 (wasm, headless chromium) | not `soak_only` | Acquires e5, builds the int8-embed graph (`quantize_e5_browser.py`, pinned SHA-256), publishes `browser/conformance/E5Conformance` and runs fp32 and int8-embed on the WASM backend in Playwright's Chromium; `evaluate_e5_browser.py --require-pass` checks every bound of the protected `onnx-*-wasm` tolerances, then the gate path runs: native ONNX Runtime on the same graph (`stage_e5_browser_native.py`), `model-pipeline browser-results` and `model-pipeline evaluate --require-gates G1,G2`. WebGPU is not run (no GPU). | `qualification-browser` |
| `experimental` | experimental `<id>` (reports, does not gate) | repository is `starlinghq/starling`, not `soak_only` | `continue-on-error`. For Kev-0.8B, Qwen3, Qwen2.5 and Whisper: acquires the candidate's snapshot and base (`tools/qualification.py lock-candidate`, `Starlings.Acquisition.Tool`, anonymous, size and SHA-256 verified), installs the route's hash-locked reference environment (and Kev's code at its pinned commit), and runs `model-pipeline run <recipe>`. A failure is a warning annotation and a step-summary line; it does not fail the workflow. Development comparisons only: no reference is adopted and no gate or maturity changes. | `qualification-experimental-<id>` |
| `release-gates` | release gates `<profile>` | not `soak_only` | G1, G2 and G3 on the accepted P-030 release sets for each Stable CPU profile (`tools/release-gates.sh`); every gate must pass. | `qualification-release-gates-<n>` |
| `soak` | soak `<profile>` | always (filtered by `soak_workloads`) | A G4 soak per CPU profile with `Starlings.Soak.Tool` (one hour by default). | `qualification-soak-<workload>` |
| `catalog` | catalog up to date | repository is `starlinghq/starling`, not `soak_only` | `model-pipeline catalog --check`: the catalog, compatibility matrix and package README maturity blocks are not stale. | none |
| `upstream` | upstream change detection | repository is `starlinghq/starling`, not `soak_only` | `model-pipeline upstream registry` (read-only Hub calls, no token). Exit 1 when a pinned Hub repository has a revision newer than the manifest's `latest_known_upstream_revision`: the failed run is the notification. | `qualification-upstream` |
| `cuda` | cuda (runpod) | repository is `starlinghq/starling`, and a push of an `ai-v*` tag or a dispatch without `soak_only` or `skip_cuda` | CUDA regression support on a RunPod GPU pod (below). | `qualification-cuda` |

**Job `cuda`.** The only job with a secret: `RUNPOD_API_KEY`, used by `tools/ci/runpod.py` on the runner only and never passed to the pod. Steps: balance check and sweep of stale pods (`preflight --min-balance 5`; no pod starts below a $5 balance), an SSH key generated per run, pod creation (`starling-q-ci-<run id>`, RTX 3090 only, with the network volume, waiting up to 30 minutes for stock), wait for SSH, a check that the GPU is an NVIDIA GeForce RTX 3090 with a driver of at least 580, then `tools/ci/runpod-cuda-pod.sh <sha>` on the pod: the CUDA test projects of e5, GLiNER2, EmbeddingGemma 2 and Qwen3.5, release-set G1, G2 and G3 of every Stable CUDA profile, and a short soak per Stable CUDA profile (the Kev CUDA profiles are not in the job) (a smoke test; its growth verdict is recorded but not binding). The runner polls the pod for up to `POD_RUN_MINUTES` (195), copies the results back, and always terminates the pod and confirms it is gone. A pod-side backstop removes the pod `BACKSTOP_MINUTES` (240) after creation if the runner is lost. Job timeout 290 minutes; concurrency group `qualification-cuda-runpod`. The job passes only if the pod run's exit code is 0.

## release

`release.yml`: releases the ai-dotnet packages (`Starlings.*`) to nuget.org.

```sh
git tag ai-v0.4.0 && git push origin ai-v0.4.0
```

- **Triggers:** `push` of tags `ai-v*`; `workflow_dispatch`, which builds the candidate only and never publishes. Concurrency group `release-<ref>`, not cancelled in progress.
- The tokenizer is not released here: it is the published `Starlings.Tokenizers` (`starlinghq/tokenizers`), pinned exactly.
- `HELD` lists package ids that are built and drilled but never published. It is empty: the FFmpeg video adapters' legal review (D2) was cleared on 2026-10-15.

| Job | Condition | What it does |
| --- | --- | --- |
| `candidate` ("release candidate") | always | Boundary check; reads the version from `ai-dotnet/src/Starlings.Models.E5` (`dotnet msbuild -getProperty:Version`). On a tag: the tag must equal `ai-v<version>` and point at a commit on `main`. The release record `docs/releases/ai-<version>.md` must exist. Then `tools/release-drill-ai.sh --no-model`, and the publishable set (every drilled `.nupkg` and `.snupkg` not in `HELD`) is uploaded as the artifact `packages` (90 days). |
| `publish` ("publish to nuget.org") | tag pushes only; needs `candidate` | Runs in the `release` environment, which has a required reviewer, so it waits for approval. Permissions `contents: write` (GitHub release) and `id-token: write` (NuGet trusted publishing). Downloads the `packages` artifact and checks no held package is in it; logs in with `NuGet/login` (OIDC, user `vars.NUGET_USER`; a short-lived key, no stored API key); pushes dependencies first (`Starlings.Acquisition`, `Starlings.Runtime.Onnx`, `Starlings.Models.Abstractions`, `Starlings.Generation.Abstractions`, `Starlings.Generation.OnnxGenAI`, `Starlings.Models.E5`, `Starlings.Models.Gliner2`, `Starlings.Models.EmbeddingGemma2`, `Starlings.Models.Qwen35`, `Starlings.Models.Qwen35.OnnxGenAI`, `Starlings.Models.Kev`, `Starlings.Models.QwenChat`, `Starlings.Models.Bert`), then the rest, without `--skip-duplicate` (a published version is immutable); creates the GitHub release with `gh release create`, the release record as notes and the packages attached (`--prerelease` for a version with a suffix). |
