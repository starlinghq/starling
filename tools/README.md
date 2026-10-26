# tools

Repository scripts: checks, release drills, qualification helpers and code generators. Each entry below is taken from the script's own header comment or argument parser; the script is the authority when they differ. Run the Python scripts with `python3 -I` from the repository root, as CI does.

"CI" names the workflow and job that run the script (`.github/workflows/`, described in [`.github/workflows/README.md`](../.github/workflows/README.md)). "Not run by a workflow" means it is run by hand or by another script.

## Checks

### `check-boundary.py`

Fails if any tracked file, file name or commit names a term that must never appear in this repository (a boundary with an unrelated older project). The terms are stored only as SHA-256 hashes, so the script does not name them either. It scans text lowercased and split into words, decompresses `.gz` files, and decodes embedded base64 (16 or more characters) and hex (32 or more) runs, so a term inside a fixture or test model is found too.

```sh
python3 -I tools/check-boundary.py                  # tracked files and names; commits since the recorded baseline
python3 -I tools/check-boundary.py --range A..B     # tracked files and names; the commits of a range (a pull request)
python3 -I tools/check-boundary.py --all-history    # tracked files and names; every commit reachable from HEAD
```

Files and file names are always scanned in full. Commits (author, committer and message) are scanned from a baseline on, because a published commit cannot be corrected without rewriting history: the baseline is the full commit SHA in `tools/check-boundary.baseline`, and the commits reachable from HEAD but not from it are scanned. The baseline and its ancestors passed the full-history scan when it was recorded; only the maintainer moves it. Without the file every commit reachable from HEAD is scanned. A baseline the clone does not hold (a shallow checkout) or a range that does not resolve is an error (exit 2), never a pass. Exit 1 when a term is found.

To add a term, append the SHA-256 of its lowercase alphanumeric form under its length in `FORBIDDEN`. starlinghq/roost keeps a copy of the script; it stays byte-identical to this one once roost adopts this version and records its own baseline file.

Tests: `python3 -I -m unittest discover -s tools/tests -v` (`tools/tests/test_check_boundary.py`: matching, the full-tree file scan, the baseline and range selection, in temporary repositories with a made-up term).

CI: `ci` job `boundary` (full-history checkout; the tests, then the check with `--range <base sha>..<head sha>` on a pull request and the baseline on a push); `release` job `candidate` (the baseline).

### `check-provenance.py`

G0 provenance check of registry manifests, with anonymous network access only (plan section 10, G0). For each manifest it checks and reports one line per finding:

1. Immutable source lock: `origin.code` and `origin.weights` pinned by a full 40-hex commit; every `origin.files` entry has a SHA-256 and a size; every hosted Starling file is addressed at a full commit and recorded `source_status: public`.
2. Accessible reference: every `origin.files` entry is fetched anonymously (no token, no cookies) from where the packages download it. Large (LFS) files are checked through the Hub's `X-Linked-Etag` and `X-Linked-Size` headers without downloading them; small files are downloaded and hashed. The pinned commits must resolve anonymously.
3. Rights recorded: `rights.code_license`, `weights_license`, `tokenizer_terms` and `redistribution` are present.

```sh
python3 -I tools/check-provenance.py [<manifest.yaml> ...]    # default: every model-pipeline/registry/*/*/manifest.yaml
```

Exits 0 only if every check passes. Never writes files and never sends a token. Needs PyYAML (installed with `model-pipeline`). Contacts huggingface.co.

Not run by a workflow.

### `create_backlog_issues.py`

Parses the backlog table in section 18 of [`docs/plan/fast-faithful-ai-dotnet.md`](../docs/plan/fast-faithful-ai-dotnet.md) (the single source for the backlog) and prints the GitHub issues it would create, with lane labels by backlog ID.

```sh
python3 -I tools/create_backlog_issues.py                                   # dry run
python3 -I tools/create_backlog_issues.py --create --repo starlinghq/starling # open the issues with gh
```

`--create` needs the GitHub CLI (`gh`), installed and authenticated.

CI: `ci` job `repo` ("Backlog parses from the plan", dry run with output discarded). A change to the section 18 heading or row format breaks this job.

## Model assets

### `fetch-tokenizer-assets.sh`

Downloads the pinned `tokenizer.json` assets the ai-dotnet model tests use and verifies their SHA-256. These are model-owner files and are never committed (plan section 9). A file already present with the right digest is not downloaded again.

```sh
tools/fetch-tokenizer-assets.sh                                   # into <repo>/.models (ignored by Git)
STARLING_MODELS=/data/models tools/fetch-tokenizer-assets.sh
```

Layout: `$STARLING_MODELS/<hub repo>/<full commit SHA>/<file>`. Revisions and digests come from `docs/m0/hub-pins.md` and `model-pipeline/candidates/*/profile.json`, matching `TestAssets.cs`. Derived assets that no Hub file holds (the `qwen35-bpe-chat` `tokenizer.json`) are built from a pinned source file and accepted only if their SHA-256 matches the pinned digest.

CI: `ci` jobs `ai-dotnet` and `ai-dotnet (windows x64)` (the cache key is the SHA-256 of this script); `qualification` job `model-tests`.

### `acquire-models.sh`

Acquires the pinned model files of registry profiles for the scheduled qualification.

```sh
STARLING_MODELS=<cache> tools/acquire-models.sh <locks dir> <profile>...
```

For each profile with public hosting it writes a lock with `qualification.py lock` (from the manifest's `origin.files`, plus the files in `model-pipeline/protected/<profile>/environment.json` when present), then downloads what is missing into `$STARLING_MODELS` and verifies every file's size and SHA-256 with `Starlings.Acquisition.Tool`. Anonymous: `HF_TOKEN` is removed from the download's environment. A profile with private hosting is skipped with a notice. Exits non-zero if any download or verification fails. `STARLING_MODELS` is required.

CI: `qualification` jobs `model-tests`, `release-gates` and `soak`.

### `fetch-release-references.py`

Downloads a release set's reference files kept outside Git and verifies them against the set's protected record, `model-pipeline/protected/<model>/release/<set>/hosted-reference.json` (a Hugging Face dataset repository, a full commit SHA, and a size and SHA-256 per file).

```sh
python3 -I tools/fetch-release-references.py [--prune] <protected hosted-reference.json> <out dir>
```

Each file is downloaded anonymously at the recorded commit and refused unless its size and SHA-256 match. A file already in `<out dir>` is kept only if it matches; otherwise it is downloaded again. `--prune` also deletes every other file under `<out dir>`. `<out dir>` must be outside the repository. Exit 0 when every file is verified, 1 on a mismatch or download failure, 2 on bad arguments. The evaluator later checks every array it reads against the release set's own digests, so this is a first gate, not the only one.

CI: not called directly by a workflow; `release-gates.sh` and `ci/runpod-cuda-pod.sh` use it.

## Qualification

### `qualification.py`

Helpers for the scheduled qualification workflow (P-018, workstream R6). Reads manifests and fixture environment records only (PyYAML); never writes outside the given output path.

```sh
python3 -I tools/qualification.py hosting <manifest.yaml>
python3 -I tools/qualification.py lock <manifest.yaml> <out.json> [--reference-environment <environment.json>]
python3 -I tools/qualification.py lock-candidate <profile.json> <out.json> [--base]
python3 -I tools/qualification.py snapshot <lock.json>
```

- `hosting` prints `public` or `private: <where>`. A profile is private when any `source_status` in its manifest is `placeholder` or `private`. With `$GITHUB_OUTPUT` set it also writes `hosting=public|private` there.
- `lock` writes a `Starlings.Acquisition` JSON lock of every file in `origin.files`, at the `origin.weights` repository and pinned revision. `--reference-environment` adds the files the protected reference run recorded when its model is the same repository and revision. Files Starling hosts outside the upstream repository carry their download `url` (lock `schema_version` 2). Only public Hub origins are supported.
- `lock-candidate` writes the same kind of lock (schema version 1) for an onboarding candidate contract (`model-pipeline/candidates/<name>/profile.json`): every file of `origin.files` at `origin.repo` and `origin.revision`, or with `--base` every file of `origin.base` (the pinned base model of the Kev and Whisper routes). Only Hugging Face repositories pinned by a full commit SHA are supported. Tests: `tools/tests/test_qualification.py` (skipped without PyYAML).
- `snapshot` prints the lock's snapshot directory relative to the cache root, `<repository>/<revision>`.

CI: `qualification` jobs `assets`, `evaluator`, `experimental` (`lock-candidate`) and `soak` (and through `acquire-models.sh`).

### `release-gates.sh`

Runs G1, G2 and G3 of one CPU profile on its accepted P-030 release sets: the regression support of Stable profiles (`docs/support-policy.md`).

```sh
tools/release-gates.sh <profile> --out <dir> --work <dir> [--reference-python <python>] [--threads <n>]
```

- `<profile>`: `multilingual-e5-small/onnx-fp32-cpu`, `embeddinggemma-2/onnx-fp32-cpu`, `gliner2-base-v1/onnx-fp32-cpu`, `embeddinggemma-2/image-fp32-cpu`, `embeddinggemma-2/video-fp32-cpu` or `qwen3.5-0.8b/ortgenai-fp32x-cpu`.
- `--out`: reports (each `model-pipeline evaluate` `report.json` and `report.md`) and a `summary.md`.
- `--work`: candidate outputs, fetched media and references; must be outside the repository.
- `--reference-python`: the Python of the profile's hash-locked reference environment (with `model-pipeline`); required for every profile except `qwen3.5-0.8b/ortgenai-fp32x-cpu`.
- `--threads`: ONNX Runtime intra-op threads of the producers (default 4).

The commands are the ones each evidence README under `docs/evidence/release-set-gates-2026-10-14/<profile>/` records. Reference tensors kept outside Git are fetched with `fetch-release-references.py`. Bounds and budgets come only from the profile's protected files, which the evaluator verifies against Git. The models must already be in `$STARLING_MODELS`. Exit 0 only if every gate of every evaluation passed; 1 otherwise, after all evaluations ran. Writes nothing to `model-pipeline/protected/` and changes no manifest.

CI: `qualification` job `release-gates`.

### `ci/runpod.py`

RunPod control for the `cuda (runpod)` job of the qualification workflow. Runs on the GitHub runner only. Reads the API key from `RUNPOD_API_KEY` and never prints it, writes it to a file or sends it anywhere but the RunPod API. It only creates, inspects or terminates pods whose name starts with `starling-q-ci-`.

```sh
python3 -I tools/ci/runpod.py preflight [--min-balance <usd>]      # balance check; terminates stale starling-q-ci- pods
python3 -I tools/ci/runpod.py create --name <n> --public-key <file> --backstop-minutes <m> [--stock-wait-minutes <m>]
python3 -I tools/ci/runpod.py ssh --name <n> [--wait-minutes <m>]  # prints "<ip> <port>" once public SSH is mapped
python3 -I tools/ci/runpod.py terminate --name <n> [--created-at <t>]  # terminates, confirms it is gone, logs wall time and cost
python3 -I tools/ci/runpod.py check                                 # exit 1 if any starling-q-ci- pod exists
```

`create` prints the pod id, GPU and hourly cost and writes `pod_id` to `$GITHUB_OUTPUT`. The pod is created on demand (secure cloud) with the network volume `starling-q-vol` (EU-CZ-1), on an NVIDIA GeForce RTX 3090 only. No other GPU is a fallback: with no RTX 3090 in stock the job fails, and a pod that reports another GPU is terminated.

CI: `qualification` job `cuda` (`preflight --min-balance 5`, `create`, `ssh`, `terminate`).

### `ci/runpod-cuda-pod.sh`

The pod side of the CUDA qualification. Runs on the RunPod pod with the network volume at `/workspace`; the `cuda` job copies it there and starts it detached, polls `<out>/status.txt` and `<out>/DONE`, and copies `<out>` back as the artifact `qualification-cuda`.

```sh
runpod-cuda-pod.sh <commit sha> <out dir> [soak duration, default 10m]
```

Order: inventory, checkout of `<sha>`, builds, reference environments, model acquisition, the CUDA test projects of e5, GLiNER2, EmbeddingGemma 2 and Qwen3.5, G1, G2 and G3 of each CUDA profile in the script's `PROFILES` list (the seven Stable CUDA profiles; not Kev's) on its accepted P-030 release sets, then one short soak per listed profile. Every check is a line `<profile> <check> <result>` in `<out>/summary.tsv`; a check whose input is missing is `not-run`, never `passed`. The exit status (also in `<out>/exit-code`) is 0 only if every check passed. Release inputs that are not in Git are cached under `/workspace/ci-cache` and verified against the digests in Git, so a cached file can make a check fail, never pass. G2 reference tensors are downloaded with `fetch-release-references.py`, never regenerated. Writes nothing to `model-pipeline/protected/`.

CI: `qualification` job `cuda` (on the pod).

### `measure-eg2-cuda-frame-calls.py`

W1 of [`docs/plan/embeddinggemma2-follow-up.md`](../docs/plan/embeddinggemma2-follow-up.md): measures EmbeddingGemma 2 video on CUDA with 2, 3 and 4 frames per vision call. For each size it runs the CUDA video parity tests (`ai-dotnet/tests/Starlings.Models.EmbeddingGemma2.Tests.Cuda`) once with `STARLING_EG2_CUDA_FRAMES_PER_CALL` set, sampling the GPU with `nvidia-smi`. It records embedding max abs and min cosine against the CPU PyTorch reference, peak GPU memory of the test process and of the GPU, the memory other processes use and the headroom left, and the wall clock. It changes nothing in the package.

```sh
python3 -I tools/measure-eg2-cuda-frame-calls.py --out <dir> [--gpu 1] [--frames 2 3 4] [--dotnet <dotnet>] \
    [--interval-ms 200] [--settle 5] [--no-build]
```

`--gpu` is the index in PCI bus order (default 1). Writes `<out>/frame-calls.json`, `<out>/frame-calls.md` and each run's test log; exits non-zero if any run failed. Needs the CUDA 13 runtime and cuDNN 9, the EmbeddingGemma 2 image graphs (`STARLING_MODELS` or `STARLING_EMBEDDINGGEMMA2_SNAPSHOT`), the videos (`STARLING_EMBEDDINGGEMMA2_VIDEOS`), optionally the reference tensors (`STARLING_EMBEDDINGGEMMA2_VIDEO_TENSORS`), and FFmpeg 8 (`STARLING_FFMPEG_DIR`). It runs on the shared GPU, so the maintainer runs it. Protocol: `docs/evidence/embeddinggemma2-video/frame-calls-protocol.md`.

Not run by a workflow.

## Release drills

### `release-drill-ai.sh`

Release-candidate drill for the Starlings packages (plan sections 10 G6 and 14). It proves that a new user can install and run the release candidate without the source checkout, a hidden cache or an upstream tokenizer. From a clean clone of the committed `HEAD` (uncommitted changes are not part of the drill) it packs the ai-dotnet packages with a locked restore and an empty NuGet cache, inspects every nuspec (version, license, readme, commit, exact dependency list with no native ONNX Runtime, no Microsoft.ML.Tokenizers and no FFmpeg bindings package), the packed files and the symbol packages, pushes them to a local folder feed only (never nuget.org), and builds the samples and the checkers in `release-drill/` outside the repository against that feed and nuget.org. It then checks the restore graph, acquires e5 through the sample's download step and runs the samples and checkers with no network against the protected fixtures and bounds.

```sh
tools/release-drill-ai.sh                                   # downloads the e5 files (487 MB) from the Hub
tools/release-drill-ai.sh --seed-model-from <snapshot dir>  # copies onnx/model.onnx from a local snapshot first
tools/release-drill-ai.sh --no-model                        # CI: downloads tokenizer.json only, skips the embedding checks
DRILL_KEEP=1 tools/release-drill-ai.sh ...                  # keep the temporary directory
```

Further options, each described in the script header:

| Option | Adds |
| --- | --- |
| `--seed-gliner2-from <dir>` | the GLiNER2 sample, seeded from `<dir>` |
| `--seed-embeddinggemma2-from <dir>` | the EmbeddingGemma 2 text search, and the image and video searches when `<dir>` holds their graphs and media |
| `--ffmpeg-dir <dir>`, `--ffmpeg9-dir <dir>` | the video search through the FFmpeg 8 or FFmpeg 9 adapter on the user's own LGPL FFmpeg shared build |
| `--ffmpeg-gpl-dir <dir>`, `--ffmpeg9-gpl-dir <dir>` | a GPL FFmpeg build the adapter must refuse |
| `--seed-qwen35-from <dir>` | the QwenChat sample on CPU, from a `build_ortgenai_bundle.py --variant fp32x` bundle |
| `--public-hosting`, `--public-hosting-only <labels>` | download GLiNER2, EmbeddingGemma 2 and Qwen 3.5 from public hosting with no token, then run offline |
| `--aot` | the Native AOT check (`aot-check-ai.sh`) and the AOT smoke check (`aot-smoke-ai.sh`) on the drill's feed |
| `--aot-smoke` | only the weight-free AOT smoke check (`aot-smoke-ai.sh`) on the drill's feed |
| `--cuda <device> [--cuda-lib-path <dirs>] [--cuda-only]` | the CUDA profiles on one GPU, through CUDA consumers whose only natives are the GPU packages |
| `--packages-from <dir>` | reuse the `.nupkg` and `.snupkg` files of an earlier drill instead of packing |

Every mode also checks the build-time native package checks: `Starlings.Runtime.Onnx` and `Starlings.Generation.OnnxGenAI` must pack their `buildTransitive` targets, no sample or checker may trigger `STARLINGS001` to `STARLINGS004`, and four throwaway consumers (`release-drill/native-check`) must trigger them (no native: warning; CPU and GPU native together: error; the latter restores the GPU packages, about 420 MB).

Packages and logs are copied to `artifacts/release-drill-ai/` (ignored by Git).

CI: `ci` job `ai-release-drill` and `release` job `candidate`, both with `--no-model`. The release workflow publishes the packages this drill packed.

### `release-drill-ai.ps1`

Windows clean-install drill (PowerShell 7.2 or later), the counterpart of `release-drill-ai.sh` for Windows x64. It clones the committed `HEAD`, checks the package and tokenizer versions, packs with a locked restore, inspects the nuspecs, packed files and symbols, pushes to a local folder feed, builds the samples and the `ai-consumer` and `mlnet-consumer` checkers outside the repository, checks the restore graph (no foreign tokenizer, GPU native or FFmpeg library), acquires e5, and runs the sample and checkers offline.

```powershell
pwsh tools/release-drill-ai.ps1 -NoModel
pwsh tools/release-drill-ai.ps1 -SeedModelFrom C:\models\e5-snapshot -SeedGliner2From C:\models\gliner2
```

Parameters: `-SeedModelFrom`, `-NoModel`, `-SeedGliner2From`, `-SeedEmbeddingGemma2From`, `-SeedQwen35From`, `-PublicHosting`, `-Keep`. Differences from the Linux drill, recorded as untested on Windows: "offline" means `HTTP_PROXY`, `HTTPS_PROXY` and `ALL_PROXY` point at a closed local port (no network namespace); the EmbeddingGemma 2 image and video searches are not run; the Native AOT check is Linux only.

CI: `ci` job `ai-dotnet (windows x64)` (`-NoModel`).

### `aot-check-ai.sh`

Native AOT check for the Starlings packages (plan 10 G6; production-readiness R9). For each target (one package and its dependencies) it builds `release-drill/aot-consumer` outside the repository against the packed packages twice: a JIT build that produces the reference results, and `dotnet publish -p:PublishAot=true`, whose trim and AOT warnings are recorded and whose native binary runs the same inference with no network. It compares the two result files: strings exactly, float vectors within `--float-bound` (default 1e-5; the check's own bound, not a protected tolerance).

```sh
tools/aot-check-ai.sh --models <cache> [--feed <dir>] [--targets a,b] [--download] [--out <dir>] [--run-unsupported] [--cuda]
```

- `--models`: model cache in the Starlings layout, read only; targets whose files are missing are skipped with a note.
- `--feed`: local folder feed of packed packages; without it the script packs the committed `HEAD` from a clean clone first.
- `--targets`: a subset of `acquisition runtime e5 e5-meai e5-mlnet gliner2 eg2 eg2-meai eg2-mlnet qwen35 qwen35-meai`.
- `--download`: the acquisition target also downloads e5's `tokenizer.json` from the public Hub (no token).
- `--out`: logs, warnings and results (default `artifacts/aot-check-ai`).
- `--cuda`: build against the CUDA natives and run every session on CUDA device 0; needs the CUDA 13 runtime and cuDNN 9 on `LD_LIBRARY_PATH` and an explicit `CUDA_VISIBLE_DEVICES`.
- `--run-unsupported`: also run the native binaries of targets recorded as AOT unsupported (the ML.NET adapters, which abort by design).

Exit 0 when every target that ran published, ran under AOT and matched the JIT results. Warnings do not fail the check; `summary.md` lists them.

Not run by a workflow; `release-drill-ai.sh --aot` calls it.

### `aot-smoke-ai.sh`

Weight-free Native AOT smoke check of every published package (backlog E1). For each package it builds `release-drill/aot-smoke` outside the repository against the packed package with the whole package assembly rooted (`TrimmerRootAssembly`, so ILC analyzes every member), publishes it with `-p:PublishAot=true`, records every trim and AOT warning, and runs the binary with no network (it only loads the assembly). No model file is needed.

```sh
tools/aot-smoke-ai.sh [--feed <dir>] [--packages a,b] [--out <dir>]
```

- `--feed`: local folder feed of packed packages; without it the script packs the committed `HEAD` from a clean clone first.
- `--packages`: package ids (default: every published package except the browser ones, which target `browser-wasm`).
- `--out`: logs, warnings and `summary.md` (default `artifacts/aot-smoke-ai`).

`summary.md` gives each package a verdict: `AOT-clean` (no warning anywhere), `clean in Starlings.*` (warnings only in third-party code) or `not clean`. Exit 0 when every package not recorded as AOT unsupported (the ML.NET adapters) published and ran; warnings do not fail the check. Record: [`docs/evidence/aot-smoke/`](../docs/evidence/aot-smoke/README.md).

Not run by a workflow; `release-drill-ai.sh --aot` and `--aot-smoke` call it.

### `release-drill/`

Clean-consumer programs that the drills copy into a temporary directory outside the repository and build against the packed packages from a local folder feed (`*.csproj.in` is the project template). They compile and run with nothing but those packages: no project references, no repository paths, no test framework. Each exits 0 only if every check passes.

| Directory | Used by | Usage and checks |
| --- | --- | --- |
| `ai-consumer` | `release-drill-ai.sh`, `release-drill-ai.ps1` | `AiConsumer tokenizer <cache> <fixture dir>`: downloads e5's `tokenizer.json`, then checks token ids against the protected fixtures and that the ONNX Runtime natives load. `AiConsumer full <cache> <fixture dir> <embedding_max_abs> <padded_batch_vs_single_max_abs>`: offline e5 embeddings through the typed API, Microsoft.Extensions.AI and ML.NET, within the protected bounds. |
| `mlnet-consumer` | `release-drill-ai.sh`, `release-drill-ai.ps1` | `MlNetConsumer e5 <cache>` or `MlNetConsumer eg2 <cache>`: the ML.NET transform's column type, row-wise and look-ahead batched results equal to the typed API, token counts, ranking, `Save` rejected; for EmbeddingGemma 2 also a Matryoshka 256 column and a rejected video column. |
| `cuda-consumer` | `release-drill-ai.sh --cuda` | `CudaConsumer <profile> <cache> <fixture dir> <tolerances.yaml> [<image dir>]` and `CudaConsumer eg2-video ... <video dir> <FFmpeg lib dir> 8` (or `9`): offline CUDA sessions on device 0, version diagnostics, outputs against the protected fixtures within the CUDA profile's bounds. |
| `aot-consumer` | `aot-check-ai.sh` | `AotConsumer <cache> <results file> [<empty download dir>]`: one target fixed at build time (`AotTarget`); runs the inference and writes results for the script to compare. |
| `aot-smoke` | `aot-smoke-ai.sh` | `AotSmoke`: one package fixed at build time (`SmokePackage`) and rooted; loads the package assembly and reports `Native AOT` mode. No inference. |
| `native-check` | `release-drill-ai.sh` | Project template only: one model package and zero, one or two native packages; the drill checks the `STARLINGS00x` diagnostic of each case. |

## Code generation

### `generate-ffmpeg-bindings.py`

Generates the FFmpeg bindings of the EmbeddingGemma 2 video adapters from FFmpeg's own headers. One adapter package per FFmpeg major (decision D3): `--major 8` (default) writes `Starlings.Models.EmbeddingGemma2.Video.FFmpeg`, `--major 9` writes `Starlings.Models.EmbeddingGemma2.Video.FFmpeg9`. The script binds only the declarations in its `SPEC`. It asks the C compiler: a type check (every function against its declared prototype, every field type by `_Static_assert`) and a layout probe (struct sizes, field offsets, constant values, library majors). From that it writes `FFmpegNative.g.cs`: explicit-layout structs, `[LibraryImport]` functions and constants. Run it against the headers of the targeted FFmpeg build on x86-64 Linux; the generated file records the target.

```sh
python3 -I tools/generate-ffmpeg-bindings.py --include <ffmpeg>/include [--major 8|9] [--check]
```

`--check` regenerates in memory and fails if the committed file differs. FFmpeg 9 headers: `model-pipeline/reference/embeddinggemma2/fetch_eg2_ffmpeg9_headers.sh <dir>` installs the pinned n9.0.1 headers into `<dir>/include`.

Not run by a workflow (`ci.yml` refers to it in a comment of the dependency-boundary check).
