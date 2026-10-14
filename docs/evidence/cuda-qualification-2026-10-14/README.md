# CUDA G1/G2 evidence on an RTX 3090, 2026-10-14

The protected staged evaluator (`model-pipeline evaluate`) ran once for every CUDA profile, and every `ai-dotnet/tests/*.Cuda` project ran once, on one RTX 3090. This was workstream R7 W2 with maintainer item M6 (GPU time) in `docs/plan/production-readiness.md`.

**These are development fixtures, not P-030 release sets.** Every case comes from the agent-written development corpora that the maintainer accepted as development fixtures (e5: M0 corpus; GLiNER2: PR #19; EmbeddingGemma 2: PR #29, PR #42 and PR #47; Qwen3.5: `prompts-v1.jsonl`). A pass here is G1/G2 evidence on development fixtures only, not release qualification. The manifests therefore record G1 and G2 as `checklist`, with a link to this directory, and never as `passed`. Maturity and capability statuses are unchanged. Nothing under `model-pipeline/protected/` or in any `tolerances.*` file was changed.

## Results

Each `report.md` / `report.json` is the unedited output of `model-pipeline evaluate <manifest> --results <dir> --out <dir> --require-gates G1,G2`. The protected evidence was verified against Git (merge base with `origin/main`, `8feb506`). The bounds are those of each profile's binding `tolerances.yaml`, with `use_tf32=0` in every stage.

| Profile | Exit | G1 | G2 | Key numbers (bound) | Failures |
| --- | ---: | --- | --- | --- | --- |
| [`multilingual-e5-small/onnx-fp32-cuda`](e5-onnx-fp32-cuda/README.md) | 0 | passed | passed | ids and masks exact 46/46. Embedding max abs: Python ORT CUDA 1.70e-7, C# from ids and from text 1.65e-7 (1e-5). Hidden state 2.27e-6 (1e-4). C# vs Python 9.0e-8 (1e-6); hidden states bit-identical. | none |
| [`gliner2-base-v1/onnx-fp32-cuda`](gliner2-onnx-fp32-cuda/README.md) | 0 | passed | passed | Processor exact 58/58. Span logits relative 1.42e-5 (1e-4), count logits 4.9e-5 (5e-4), scores 2.26e-6 (2e-5), confidence 1.11e-6 (2e-5), hidden 2.48e-5 (1e-4). Entities 58/58. C# vs Python span logits 0 (1e-6). | none |
| [`embeddinggemma-2/onnx-fp32-cuda`](embeddinggemma2-onnx-fp32-cuda/README.md) | 0 | passed | passed | ids exact 138/138. Embedding max abs, worst over every path and batch layout: 5.32e-7 (5e-6). Min cosine 0.99999999999 (0.9999999). Token embeddings 7.2e-6 relative (1e-4). C# vs Python 4.6e-7 (1e-6). | none |
| [`embeddinggemma-2/image-fp32-cuda`](embeddinggemma2-image-fp32-cuda/README.md) | 0 | passed | passed | ids 42/42 and RGB/resize/pixel/position digests on 35/35 images exact. Embedding 3.64e-6 (1e-5), min cosine 0.99999999965 (0.9999995). Soft tokens 1.44e-4 and token embeddings 7.7e-5 row-relative (1e-3). Batches 7.0e-7 (1e-5). | none |
| [`embeddinggemma-2/video-fp32-cuda`](embeddinggemma2-video-fp32-cuda/README.md) | 0 | passed | passed | ids 36/36, frame indices and every digest on 32 videos exact. Per-frame RGB exact on FFmpeg 8.1.3. Embedding 1.13e-6 (1e-5), min cosine 0.99999999996. Soft tokens 6.1e-5 and token embeddings 1.09e-4 row-relative (1e-3). Batches 1.95e-7 (1e-5). | none |
| [`qwen3.5-0.8b/ortgenai-fp32x-cuda`](qwen35-ortgenai-fp32x-cuda/README.md) | 4 | passed | **not-run** | Template and ids 21/21 exact. Greedy 22/22 exact. Step-0 log-prob 1.35e-3 (5e-3), top-5 5.9e-4 (2e-3). Cached vs uncached 4.7e-5 (2e-4). Concurrent equal to sequential. Chunked vs one-call prefill measured 5.9e-5, **no protected bound**. | none; G2 not-run (see below) |
| [`qwen3.5-0.8b/ortgenai-fp16x-cuda`](qwen35-ortgenai-fp16x-cuda/README.md) | 0 | passed | passed | Template and ids 21/21 exact. Greedy 19/22 exact; 3 divergences at reference margins 0.0004 to 0.0027 (bound < 0.05). Step-0 0.061 (0.15), top-5 0.041 (0.06). Cached 0.061 (0.1). Chunked 0.049 (0.1). | none |

**No check failed a protected tolerance.** One result is not a pass:

- **Qwen fp32x CUDA, G2 `not-run`, exit 4.** The binding `protected/qwen3.5-0.8b/ortgenai-fp32x-cuda/tolerances.yaml` has no `float.chunked_vs_one_call_prefill_logprob_max_abs` key, though the CPU and fp16x CUDA files have one. The evaluator measures the check (5.9e-5 on `long-license` with chunks of 100 and 128, and on `multi-turn` with chunks of 7). With no bound it reports the check, and therefore G2, as not-run. All 14 other checks passed. Nothing was adjusted. Setting the bound is a maintainer change to protected evidence (label `protected-change`). The CPU profile binds 1e-4, and the observed value is below that.

Every C# and Python stage ran on `CUDAExecutionProvider` (`session_providers` and `onnxruntime_providers` in the stage results), with provider options `use_tf32: 0`, checked by the evaluator's `provider_options` check in each stage.

## CUDA test projects

`dotnet test --project ai-dotnet/tests/<project> -c Release --output Detailed` with `STARLING_CUDA_TESTS=1`, on the same GPU. Logs are in [`cuda-tests/`](cuda-tests/README.md).

| Project | Total | Passed | Failed | Skipped | Duration |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Starlings.Models.E5.Tests.Cuda` | 7 | 6 | 0 | 1 | 7 s |
| `Starlings.Models.Gliner2.Tests.Cuda` | 4 | 4 | 0 | 0 | 11 s |
| `Starlings.Models.EmbeddingGemma2.Tests.Cuda` | 18 | 18 | 0 | 0 | 1 min 48 s |
| `Starlings.Models.Qwen35.Tests.Cuda` | 75 | 75 | 0 | 0 | 3 min 57 s |
| **Total** | **104** | **103** | **0** | **1** | |

The one skip is `CudaUnavailableTests.MissingCudaProviderLibraryFailsClearly`. It runs only with `STARLING_EXPECT_CUDA_UNAVAILABLE=1` in an environment where the CUDA provider cannot load, which this run was not.

## Hardware and software

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090, 24,576 MiB, compute capability 8.6, PCI `03:00.0`, VBIOS 94.02.4B.00.0B; selected with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`. The machine's other GPU (RTX 4060 Ti, fully used by another workload) was not used. |
| GPU load | Idle before the runs (298 MiB used by two small resident processes, 0% utilization). It was not otherwise shared during the runs. |
| Driver | 610.57.04 (CUDA UMD 13.3) |
| CUDA runtime on the loader path | The `nvidia-*` wheels of `model-pipeline/reference/encoder/requirements-cuda.txt` on `LD_LIBRARY_PATH`: `nvidia-cuda-runtime` 13.0.96, `nvidia-cublas` 13.1.1.3, `nvidia-cudnn-cu13` 9.24.0.43. The system toolkit at `/opt/cuda` is 13.3. |
| ONNX Runtime | 1.30.0: C# `Microsoft.ML.OnnxRuntime.Gpu.Linux` (`ai-dotnet/tools/Starlings.Fidelity.Tool.Cuda`, CUDA test projects); Python `onnxruntime-gpu` 1.30.0 |
| ONNX Runtime GenAI | 0.17.1 (`Microsoft.ML.OnnxRuntimeGenAI.Cuda`, `Starlings.Qwen35.Fidelity -p:UseCuda=true`) |
| Python reference environments | Python 3.11.16, torch 2.14.1; built with `uv` from `encoder/`, `gliner2/` and `embeddinggemma2/requirements-cuda.txt` (`--require-hashes`, except GLiNER2, whose Git requirement cannot be hashed). PyTorch reference hidden states stay fp32 CPU. |
| .NET | SDK 10.0.100, runtime 10.0.12 |
| CPU, OS | Intel i7-5820K (12 threads), 94 GB RAM, Linux 7.2.5 x86-64. Shared with another agent's CPU test runs (load average 3 to 10). |
| Code | Stage producers at `2db3dce` (e5, GLiNER2, EmbeddingGemma 2) and `c33d63c` (Qwen): pre-squash commits of branch `cuda-qualification` (PR #18), kept so they stay resolvable. The squashed code equals them except for line-wrapping the `candidate` strings in `stage_e5_export.py` and `stage_gliner2.py` (no behavior change). |

Wall clock of the recorded runs (UTC, 2026-10-14): evaluator runs 07:26 to 07:37 (e5 26 s, GLiNER2 47 s, EmbeddingGemma 2 text 1 min 35 s, image 48 s, video 1 min 16 s, Qwen fp32x 58 s, fp16x 24 s), CUDA tests 07:37 to 07:45. Including setup runs and the first attempts listed below, the GPU was in use for about 25 minutes, under 0.5 GPU-hours.

## Changes needed to run the CUDA profiles

- `ai-dotnet/tools/Starlings.Fidelity.Tool`: the `e5` and `gliner2` dumps take `--device cpu|cuda[:N]` (`ToolDevice`), as the EmbeddingGemma 2 dumps already did. On CUDA, the profile id defaults to the `-cuda` profile and the stage environment records the device. The CPU build and its default output are unchanged.
- `model-pipeline/reference/encoder/stage_e5_export.py` and `reference/gliner2/stage_gliner2.py export` take `--provider cuda` (`use_tf32=0`), as `stage_embeddinggemma2_export.py` already did. They refuse to run when the session has no CUDA provider.
- `ai-dotnet/tools/Starlings.Qwen35.Fidelity/packages.cuda.lock.json` was stale: it still pinned `Starlings.Tokenizers` as a project at 0.1.0-preview.4. It was regenerated by the `-p:UseCuda=true` restore.

## Run notes

- The first e5 attempt failed before running, because the script called `/usr/bin/time`, which is not installed. The first Qwen runs were at a commit with the uncommitted lock-file refresh (`-dirty`). Both Qwen runs were repeated at `c33d63c` with the same outcome; the reports here are from those repeats.
- The Qwen fp16x CUDA bundle was downloaded from `Starlinghq/qwen3.5-0.8b-ortgenai` at `95a690c1774f144957cb7b76986a9126e8b1bb5f` without a token, and its three files were checked against the repository's `SHA256SUMS` and the digests in `docs/evidence/p016-qwen35/README.md`.
- Reference tensors outside Git (`--reference-tensors`): the EmbeddingGemma 2 image and video spike `ref-clean-tensors` and the Qwen P-016 CPU reference `step0/` vectors. The evaluator checked every file against the protected digests before use.
- Video decoding used the pinned FFmpeg 8.1.3 LGPL shared build from `fetch_eg2_ffmpeg.sh` (`--ffmpeg`). Stage 1 records `ffmpeg_major` 8, so the per-frame RGB check was gated, and it passed.

## Reproduce

```sh
# environments (Python 3.11): uv venv -p 3.11 .models/venv-<n>-cuda && VIRTUAL_ENV=... uv pip install [--require-hashes] -r model-pipeline/reference/<n>/requirements-cuda.txt
#                             && VIRTUAL_ENV=... uv pip install --no-deps -e model-pipeline jsonschema pyyaml      (n: encoder, gliner2, embeddinggemma2)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> STARLING_MODELS=<cache>
export LD_LIBRARY_PATH=<venv>/lib/python3.11/site-packages/nvidia/cu13/lib:<venv>/lib/python3.11/site-packages/nvidia/cudnn/lib
dotnet build ai-dotnet/tools/Starlings.Fidelity.Tool.Cuda -c Release
T="dotnet ai-dotnet/tools/Starlings.Fidelity.Tool.Cuda/bin/Release/net10.0/Starlings.Fidelity.Tool.dll"

# e5
.models/venv-encoder-cuda/bin/python model-pipeline/reference/encoder/stage_e5_export.py --model-dir <e5 snapshot with model.safetensors> \
  --fixtures model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu --profile multilingual-e5-small/onnx-fp32-cuda --out <r> --provider cuda
$T e5 --out <r> --hidden --device cuda
# GLiNER2 (from model-pipeline/reference/gliner2)
.models/venv-gliner2-cuda/bin/python stage_gliner2.py export --model-dir <snapshot> --graph $STARLING_MODELS/fastino/gliner2-base-v1/<rev>/onnx/gliner2-entities-fp32.onnx \
  --graph-asset-prefix onnx/gliner2-entities-fp32 --fixtures <repo>/model-pipeline/protected/gliner2-base-v1/onnx-fp32-cpu \
  --profile gliner2-base-v1/onnx-fp32-cuda --out <r> --threads 1 --provider cuda
$T gliner2 --out <r> --threads 1 --device cuda
# EmbeddingGemma 2 (stage 2 for text only; image and video bind the C# path)
.models/venv-embeddinggemma2-cuda/bin/python model-pipeline/reference/embeddinggemma2/stage_embeddinggemma2_export.py --model-dir <snapshot> \
  --graph <cache>/onnx/embeddinggemma2-text-fp32.onnx --fixtures model-pipeline/protected/embeddinggemma-2/onnx-fp32-cpu \
  --profile embeddinggemma-2/onnx-fp32-cuda --out <r> --provider cuda
$T embeddinggemma2 --out <r> --device cuda
$T embeddinggemma2-images --out <r> --images <make_eg2_images.py output> --device cuda
$T embeddinggemma2-videos --out <r> --videos <make_eg2_videos.py output> --ffmpeg <FFmpeg 8 LGPL lib dir> --device cuda
# Qwen3.5
dotnet run --project ai-dotnet/tools/Starlings.Qwen35.Fidelity -c Release -p:UseCuda=true -- --bundle <snapshot> --variant fp32x|fp16x \
  --device cuda --prompts model-pipeline/reference/generation/prompts-v1.jsonl --reference model-pipeline/protected/qwen3.5-0.8b/reference \
  --out <r> --modes parity,semantics
# then, per profile
model-pipeline evaluate model-pipeline/registry/<family>/<profile>/manifest.yaml --results <r> --out <report> --require-gates G1,G2 \
  [--reference-tensors <image/video .npz dir | Qwen reference run dir with step0/>]
```
