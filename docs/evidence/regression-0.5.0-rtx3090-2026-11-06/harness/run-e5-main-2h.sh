#!/usr/bin/env bash
# The e5 CUDA soak at main for two hours, unchanged otherwise: does the growth of the one-hour run saturate or continue?
set -uo pipefail
H=<local>
W=$H/ws
OUT=$H/out-e5-main-2h
mkdir -p "$OUT"
export DOTNET_ROOT=$W/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 NUGET_PACKAGES=$W/nuget PATH=$W/dotnet:$PATH
export STARLING_MODELS=$W/models CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0
export LD_LIBRARY_PATH="$(find "$W"/venvs/eg2-cuda/lib/python3.11/site-packages/nvidia -maxdepth 2 -type d -name lib | sort | paste -sd: -)"
cd "$H/ci-main/starling"
git rev-parse HEAD >"$OUT/commit.txt"
{ nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader | head -1; cat /proc/loadavg; } >"$OUT/env.txt"
dotnet "$H/ci-main/build/soak/Starlings.Soak.Tool.dll" e5 --cuda --duration 2h --out "$OUT/e5" \
    --note "local RTX 3090, main at $(git rev-parse --short=12 HEAD), two hours, CUDA libraries from the pinned eg2 requirements" >"$OUT/e5.log" 2>&1
echo "soak exit $?" >>"$OUT/env.txt"
{ nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader | head -1; cat /proc/loadavg; } >>"$OUT/env.txt"
