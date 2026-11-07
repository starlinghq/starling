#!/usr/bin/env bash
# The CUDA video parity tests at ai-v0.5.0 that the pod script leaves skipped: same project and environment as the run,
# plus the fixture and FFmpeg variables those tests read.
set -uo pipefail
H=<local>
W=$H/ws
OUT=$H/out-video-tests-0.5.0
mkdir -p "$OUT"
export DOTNET_ROOT=$W/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 NUGET_PACKAGES=$W/nuget PATH=$W/dotnet:$PATH
export STARLING_MODELS=$W/models CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0
export LD_LIBRARY_PATH="$(find "$W"/venvs/eg2-cuda/lib/python3.11/site-packages/nvidia -maxdepth 2 -type d -name lib | sort | paste -sd: -)"
export STARLING_CUDA_TESTS=1
export STARLING_EMBEDDINGGEMMA2_IMAGES=$W/fixtures/images-r1
export STARLING_EMBEDDINGGEMMA2_VIDEO_TENSORS=$W/fixtures/video-spike/ref-clean-tensors
export STARLING_EMBEDDINGGEMMA2_VIDEOS=$W/fixtures/video-spike/videos
export STARLING_FFMPEG_DIR=$W/ci-cache/ffmpeg8/ffmpeg-n8.1.3-17-gc389840894-linux64-lgpl-shared-8.1/lib
export MALLOC_MMAP_THRESHOLD_=65536
cd "$H/ci/starling"
dotnet test --project ai-dotnet/tests/Starlings.Models.EmbeddingGemma2.Tests.Cuda -c Release --results-directory "$OUT/results" -- --report-xunit-trx >"$OUT/tests.log" 2>&1
echo "exit $?" >"$OUT/exit.txt"
