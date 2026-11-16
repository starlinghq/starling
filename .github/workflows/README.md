# Workflows

The five GitHub Actions workflows of this repository. This page is derived from the YAML files; the YAML and its comments are the authority. Scripts under `tools/` are described in [`tools/README.md`](../../tools/README.md).

| Workflow | Runs on | Secrets and environments |
| --- | --- | --- |
| [`ci`](#ci) | every pull request, pushes to `main`, manual dispatch | none |
| [`protected-evidence`](#protected-evidence) | pull requests, including label changes | none |
| [`qualification`](#qualification) | weekly schedule, `ai-v*` tags, manual dispatch | none |
| [`release`](#release) | `ai-v*` tags, manual dispatch (candidate only) | environment `release`, its variable `NUGET_USER`, OIDC for NuGet trusted publishing |
| [`docs-pages`](#docs-pages) | `ai-v*` tags, manual dispatch | environment `github-pages`, OIDC for the Pages deployment |

Every workflow sets `permissions: contents: read` at the top level. Only the `release` job `publish` and the `docs-pages` job `deploy` ask for more.

## Runners and fork pull requests

Every job of `ci`, `protected-evidence` and `release` runs on the maintainer's self-hosted runner, labelled `[self-hosted, omarchy]`, except `ai-dotnet-windows`, which runs on GitHub-hosted `windows-latest`, and the `ci` and `protected-evidence` jobs of a fork pull request, which run on GitHub-hosted `ubuntu-latest`. Every job of `qualification` runs on the self-hosted runner labelled `qualification`; none uses a GPU. CUDA qualification runs outside GitHub Actions, on the maintainer's machine ([below](#cuda-qualification-local)).

`ci` and `protected-evidence` run on `pull_request`, so a pull request from a fork executes its own code. By default that code runs on GitHub-hosted runners: each job's `runs-on` compares `github.event.pull_request.head.repo.full_name` with `github.repository`, so a pull request whose fork was deleted, where the head repository is null, also stays on GitHub-hosted runners. This is not a security boundary: `pull_request` runs the workflow files from the pull request, so a fork can edit `runs-on` and reach the self-hosted runner. No workflow uses `pull_request_target`, and fork runs get no secrets and a read-only token. The repository's fork pull request approval policy still requires approval for every outside contributor's workflow run (Settings > Actions > General, "Require approval for all external contributors"; the API reports `all_external_contributors`). That approval is the control. Before approving, the maintainer reads the pull request's diff, including any change under `.github/`, `tools/` and build files, and never approves a fork run that changes `runs-on`. `qualification` and `release` never run on `pull_request`.

### Required status checks

Applied 2026-11-15 (#253). The `main` branch ruleset (id 25349484) requires, on every pull request into `main`:

- the eight jobs of the `ci` workflow: `boundary`, `ai release-candidate drill`, `ai-dotnet`, `ai-dotnet (windows x64)`, `model-pipeline`, `browser conformance app (build)`, `docs site (docfx)` and `repository checks`; and
- `protected evidence changes`, the single job of `protected-evidence`.

`ci` and `protected-evidence` run on every pull request (including from forks) and on pushes to `main`, and every job above runs on both events, so a pull request can always produce every required check. Nothing that runs only on tags, schedules or dispatch is required. Organization admins may still bypass the ruleset through a pull request (`docs/kickoff.md`, "Repository settings to apply by hand").

## ci

`ci.yml`: fast pull-request checks (plan section 13.1). No secrets are used.

- **Triggers:** `pull_request`, `push` to `main`, `workflow_dispatch`. Concurrency group `ci-<ref>`, with `cancel-in-progress`.

| Job | Runner | What it checks |
| --- | --- | --- |
| `boundary` | self-hosted `omarchy` | The tests of the boundary tool (`python3 -I -m unittest discover -s tools/tests`), then `tools/check-boundary.py`: no name of an unrelated older project in files, file names or embedded data (the whole tree), or in the authors and messages of the commits since the baseline in `tools/check-boundary.baseline` (pushes) or of the pull request's commits (`--range <base sha>..<head sha>`). Full-history checkout. |
| `ai-release-drill` ("ai release-candidate drill") | self-hosted `omarchy` | `tools/release-drill-ai.sh --no-model`: packs from a clean clone, installs from a local folder feed into consumers outside the checkout, checks the restore graph, downloads e5's `tokenizer.json` and matches the fixtures. Nothing is published. |
| `ai-dotnet` | self-hosted `omarchy` | Fetches the pinned tokenizer assets (`tools/fetch-tokenizer-assets.sh`, cached by the script's SHA-256); builds `ai-dotnet/AI.slnx` (including tools and the CPU benchmark projects, which are never run); builds the samples `E5VectorStore`, `LocalRag` and `TicketTriage` from published packages in locked mode; runs `dotnet test --solution ai-dotnet/AI.slnx`. Model tests skip without the models. Then the library dependency boundaries for every `ai-dotnet/src/*/*.csproj` (below). |
| `ai-dotnet-windows` ("ai-dotnet (windows x64)") | GitHub-hosted `windows-latest` | The same build and weight-free tests on Windows x64, then the Windows clean-install drill `tools/release-drill-ai.ps1 -NoModel`. |
| `model-pipeline` | self-hosted `omarchy`, Python 3.11 | In `model-pipeline/`: `ruff check .`, `ruff format --check .`, `pytest`, `model-pipeline validate registry`, `model-pipeline catalog --check` (the generated `docs/catalog/` is not stale) and `model-pipeline evidence-index --check` (the generated `docs/evidence/README.md` is not stale). |
| `browser-conformance-build` ("browser conformance app (build)") | as `ai-dotnet` | Publishes `browser/conformance/E5Conformance` (interpreter build, not in locked mode, as the qualification job `browser` does), so the app that qualifies the e5 browser profiles keeps building against the packages' source. No model and no browser run. |
| `docs` ("docs site (docfx)") | as `ai-dotnet` | `dotnet tool restore` (docfx, pinned in `.config/dotnet-tools.json`), then `dotnet docfx docs/docfx.json`: the API reference from the projects in `ai-dotnet/src/` and their XML docs, with the generated catalog, compatibility matrix, evidence index and the support policy. Uploaded as the artifact `docs-site`; nothing is deployed. |
| `repo` ("repository checks") | self-hosted `omarchy` | The backlog in `docs/plan/fast-faithful-ai-dotnet.md` parses (`tools/create_backlog_issues.py`); every CPU test project under `ai-dotnet/tests/` is in `AI.slnx`, so the `ai-dotnet` jobs run it (the `*.Tests.Cuda` projects are not in `AI.slnx`: no workflow runs them; `tools/cuda-qualification-local.sh` runs them on the maintainer's RTX 3090, [below](#cuda-qualification-local)); no model weights committed (`.onnx`, `.safetensors`, `.gguf`, `.pt`, `.pth`, `.ckpt`, `.h5`); no `SixLabors.ImageSharp` in any project, props, targets or lock file; no `Sdcb.FFmpeg` or FFmpeg-binary package, and no committed FFmpeg library file. |

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
- **Job `protected-evidence`** ("protected evidence changes"): full-history checkout of the pull request, a second checkout of the base commit (`github.event.pull_request.base.sha`) into `base/`, `pip install -e base/model-pipeline`, then, from the pull request checkout, `model-pipeline check-protected --base <base sha> --head <head sha> --labels-json <labels>`. The checker and its rules (`model-pipeline/src/model_pipeline/protection.py`) come from the base commit, so a pull request cannot empty the protected patterns and change `protected/` in one step. It reads the event payload only; no secrets, read-only token.
- **Rules:** a protected path is `model-pipeline/protected/**` or any `tolerances.*`. Implementation code is `ai-dotnet/`, `model-pipeline/src/`, `model-pipeline/tests/`, `model-pipeline/reference/`, `tools/`, `browser/` and `.github/workflows/`. The changed files come from `git diff --no-renames base...head`, so a file moved out of `protected/` counts under its old path.
- **Limit:** the workflow file itself still comes from the pull request (`pull_request` event), so a pull request that edits `protected-evidence.yml` changes how its own check runs, and this check then offers no protection: the edited workflow decides whether the base rules run at all. Required code-owner review on `.github/` (the `main` ruleset) is the control for such a pull request.

## qualification

`qualification.yml`: scheduled qualification (plan section 13.1; backlog P-018; workstream R6 of `docs/plan/production-readiness.md`). It is the regression support of the Stable profiles (`docs/support-policy.md`). Nothing in it changes a manifest, a gate or a maturity; the maintainer records results (M10). Evidence from every job is uploaded as an artifact, also when a step fails.

- **Triggers:**
  - `schedule`: `23 4 * * 1` (Mondays 04:23 UTC).
  - `push` of tags `ai-v*`: every job (CPU regression per release).
  - `workflow_dispatch`, with input `soak_duration` (default `1h`) and the diagnostics inputs `soak_only`, `soak_workloads` and `soak_variants`.
- Never runs on `pull_request`, so no fork code runs here. Concurrency group `qualification-<ref>`, not cancelled in progress.
- `CPU_PROFILES` lists six CPU registry profiles, the Stable ones; the Experimental Kev CPU profiles are not in it. The job `experimental` runs the onboarding routes of Experimental CPU profiles on their candidate contracts instead, and never gates. A profile whose manifest records a `source_status` of `placeholder` or `private` is skipped with a "hosting private" notice (`tools/qualification.py hosting`).
- Models are acquired anonymously in each job that needs them (`tools/acquire-models.sh`, size and SHA-256 verified), because all CPU models together exceed the actions cache limit.

| Job | Name | Condition | What it does | Artifact |
| --- | --- | --- | --- | --- |
| `assets` | model locks | repository is `starlinghq/starling` | One lock per public CPU profile (`tools/qualification.py lock`). | `qualification-locks` |
| `model-tests` | model tests (linux x64 cpu) | not `soak_only` | Acquires every CPU model and the tokenizer assets, prepares the EmbeddingGemma 2 test inputs that have a reproducible source (`tools/ci/eg2-test-inputs.sh`: the fixture image set checked against the protected `images.json`, the pinned FFmpeg 8.1 and 9.0 LGPL builds, the four release videos), builds `AI.slnx`, runs the .NET tests with the models present, one test module at a time, then `tools/ci/check-trx.py` on the TRX reports: the job fails when a test module with tests passed none of them (every test skipped), unless `tools/ci/trx-allow-zero.txt` lists it with a reason. The 22 synthetic EmbeddingGemma 2 fixture videos have no reproducible source, so the tests that need them still skip. | `qualification-model-tests` |
| `evaluator` | evaluator `<profile>` | not `soak_only` | Per CPU profile with an onboarding recipe: the protected staged evaluator on the development fixtures, G1 and G2 required, plus the G3 task-quality evaluator on the public smoke example (not gate evidence). | `qualification-evaluator-<n>` |
| `browser` | browser e5 (wasm, headless chromium) | not `soak_only` | Acquires e5, builds the int8-embed graph (`quantize_e5_browser.py`, pinned SHA-256), publishes `browser/conformance/E5Conformance` and runs fp32 and int8-embed on the WASM backend in Playwright's Chromium; `evaluate_e5_browser.py --require-pass` checks every bound of the protected `onnx-*-wasm` tolerances, then the gate path runs: native ONNX Runtime on the same graph (`stage_e5_browser_native.py`), `model-pipeline browser-results` and `model-pipeline evaluate --require-gates G1,G2`. WebGPU is not run (no GPU). | `qualification-browser` |
| `experimental` | experimental `<id>` (reports, does not gate) | repository is `starlinghq/starling`, not `soak_only` | `continue-on-error`. For Kev-0.8B, Qwen3, Qwen2.5 and Whisper: acquires the candidate's snapshot and base (`tools/qualification.py lock-candidate`, `Starlings.Acquisition.Tool`, anonymous, size and SHA-256 verified), installs the route's hash-locked reference environment (and Kev's code at its pinned commit), and runs `model-pipeline run <recipe>`. A failure is a warning annotation and a step-summary line; it does not fail the workflow. Development comparisons only: no reference is adopted and no gate or maturity changes. | `qualification-experimental-<id>` |
| `release-gates` | release gates `<profile>` | not `soak_only` | G1, G2 and G3 on the accepted P-030 release sets for each Stable CPU profile (`tools/release-gates.sh`); every gate must pass. | `qualification-release-gates-<n>` |
| `soak` | soak `<profile>` | always (filtered by `soak_workloads`) | A G4 soak per CPU profile with `Starlings.Soak.Tool` (one hour by default). | `qualification-soak-<workload>` |
| `catalog` | catalog up to date | repository is `starlinghq/starling`, not `soak_only` | `model-pipeline catalog --check`: the catalog, compatibility matrix and package README maturity blocks are not stale. | none |
| `upstream` | upstream change detection | repository is `starlinghq/starling`, not `soak_only` | `model-pipeline upstream registry` (read-only Hub calls, no token). Exit 1 when a pinned Hub repository has a revision newer than the manifest's `latest_known_upstream_revision`: the failed run is the notification. | `qualification-upstream` |

### CUDA qualification (local)

No workflow runs CUDA. The regression support of the CUDA profiles (`docs/support-policy.md`, "Regression support") is `tools/cuda-qualification-local.sh --gpu <index> <out dir> [soak duration, default 10m]`, which the maintainer runs on their own NVIDIA GeForce RTX 3090 (the only GPU the CUDA contracts cover; driver 580 or later) from a clean checkout of the commit to tag, before every `ai-v*` tag. It runs the CUDA test projects of e5, GLiNER2, EmbeddingGemma 2, Qwen3.5 and Kev (each judged by its exit status and by `tools/ci/check-trx.py`), G1, G2 and G3 of every Stable CUDA profile on its P-030 release sets, and a short soak per Stable CUDA profile (a smoke test; its growth verdict is recorded but not binding). It writes `summary.md`, `summary.tsv`, `status.txt`, `environment.txt`, `exit-code` and the per-module and per-profile results into the output directory, the layout the former RunPod job `cuda` (removed 2026-11-15) uploaded as the artifact `qualification-cuda`; the maintainer records the run's `summary.md` and `environment.txt` in `docs/evidence/gpu-regression-<date>/`, like the RunPod runs before it. It runs one step at a time and asks `agent-budget level` before each heavy step when that command is on the path; it never sets `RUNNER_ENVIRONMENT`, so tests that inject fatal signals (GitHub-hosted runners only) never run there. No workflow uses a secret for it.

## release

`release.yml`: releases the ai-dotnet packages (`Starlings.*`) to nuget.org.

```sh
git tag ai-v0.5.0 && git push origin ai-v0.5.0
```

- **Triggers:** `push` of tags `ai-v*`; `workflow_dispatch`, which builds the candidate only and never publishes. Concurrency group `release-<ref>`, not cancelled in progress.
- The tokenizer is not released here: it is the published `Starlings.Tokenizers` (`starlinghq/tokenizers`), pinned exactly.
- `HELD` lists package ids that are built and drilled but never published. It is empty: the FFmpeg video adapters' legal review (D2) was cleared on 2026-10-15.

| Job | Condition | What it does |
| --- | --- | --- |
| `candidate` ("release candidate") | always | Boundary check; reads the version from `ai-dotnet/src/Starlings.Models.E5` (`dotnet msbuild -getProperty:Version`). On a tag: the tag must equal `ai-v<version>` and point at a commit on `main`. The release record `docs/releases/ai-<version>.md` must exist. Then `tools/release-drill-ai.sh --no-model`, and the publishable set (every drilled `.nupkg` and `.snupkg` not in `HELD`) is uploaded as the artifact `packages` (90 days). |
| `publish` ("publish to nuget.org") | tag pushes only; needs `candidate` | Runs in the `release` environment, which has a required reviewer, so it waits for approval. Permissions `contents: write` (GitHub release) and `id-token: write` (NuGet trusted publishing). Downloads the `packages` artifact and checks no held package is in it; logs in with `NuGet/login` (OIDC, user `vars.NUGET_USER`; a short-lived key, no stored API key); pushes dependencies first (`Starlings.Acquisition`, `Starlings.Runtime.Onnx`, `Starlings.Models.Abstractions`, `Starlings.Generation.Abstractions`, `Starlings.Generation.OnnxGenAI`, `Starlings.Models.E5`, `Starlings.Models.Gliner2`, `Starlings.Models.EmbeddingGemma2`, `Starlings.Models.Qwen35`, `Starlings.Models.Qwen35.OnnxGenAI`, `Starlings.Models.Kev`, `Starlings.Models.QwenChat`, `Starlings.Models.Bert`), then the rest, without `--skip-duplicate` (a published version is immutable); creates the GitHub release through the REST API with `actions/github-script` (no `gh` CLI on the runner), the release record as notes and the packages attached (a pre-release for a version with a suffix); a rerun keeps an existing release for the tag and uploads only the assets it lacks. |

## docs-pages

`docs-pages.yml`: publishes the docfx documentation site to GitHub Pages (backlog #256). Pages is enabled with "GitHub Actions" as its source; that also created the `github-pages` environment.

- **Triggers:** `push` of tags `ai-v*` (the released version goes live) and `workflow_dispatch` (a manual run of `main`). Concurrency group `docs-pages-<ref>`, not cancelled in progress.

| Job | Runner | What it does |
| --- | --- | --- |
| `build` ("docs site (docfx)") | self-hosted `omarchy` | Exactly the `ci` job `docs` (same steps, same pinned docfx), but the site is uploaded with `actions/upload-pages-artifact` instead of `upload-artifact`. |
| `deploy` ("deploy to GitHub Pages") | GitHub-hosted `ubuntu-latest` | Needs `build`; runs in the `github-pages` environment with permissions `pages: write` and `id-token: write`, and deploys the artifact with `actions/deploy-pages`. The site answers at `https://starlinghq.github.io/starling/`. |
