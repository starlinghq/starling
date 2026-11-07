#!/usr/bin/env bash
# At the main commit of the pod run: the EmbeddingGemma 2 CUDA tests with the video set and the rebuilt FFmpeg 8, then a
# one-hour e5 CUDA soak.
set -uo pipefail
H=<local>
W=$H/ws
OUT=$H/out-main-extra
mkdir -p "$OUT"
export DOTNET_ROOT=$W/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 NUGET_PACKAGES=$W/nuget PATH=$W/dotnet:$PATH
export STARLING_MODELS=$W/models CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0
export LD_LIBRARY_PATH="$(find "$W"/venvs/eg2-cuda/lib/python3.11/site-packages/nvidia -maxdepth 2 -type d -name lib | sort | paste -sd: -)"
cd "$H/ci-main/starling"
git rev-parse HEAD >"$OUT/commit.txt"
ff8="$(ls -d "$W"/ci-cache/ffmpeg8-lgpl/*/lib)"
echo "$ff8" >"$OUT/ffmpeg-dir.txt"
env STARLING_CUDA_TESTS=1 STARLING_EMBEDDINGGEMMA2_IMAGES="$W/fixtures/images-r1" \
    STARLING_EMBEDDINGGEMMA2_VIDEO_TENSORS="$W/fixtures/video-spike/ref-clean-tensors" \
    STARLING_EMBEDDINGGEMMA2_VIDEOS="$W/fixtures/video-spike/videos" STARLING_FFMPEG_DIR="$ff8" MALLOC_MMAP_THRESHOLD_=65536 \
    dotnet test --project ai-dotnet/tests/Starlings.Models.EmbeddingGemma2.Tests.Cuda -c Release --results-directory "$OUT/results" -- --report-xunit-trx >"$OUT/tests.log" 2>&1
echo "tests exit $?" >"$OUT/exit.txt"
{ nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader | head -1; cat /proc/loadavg; } >"$OUT/env.txt"
dotnet "$H/ci-main/build/soak/Starlings.Soak.Tool.dll" e5 --cuda --duration 1h --out "$OUT/e5" \
    --note "local RTX 3090 regression, main at $(git rev-parse --short=12 HEAD), CUDA libraries from the pinned eg2 requirements" >"$OUT/e5.log" 2>&1
echo "soak exit $?" >>"$OUT/exit.txt"
{ nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader | head -1; cat /proc/loadavg; } >>"$OUT/env.txt"
