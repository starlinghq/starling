#!/usr/bin/env bash
# After run-e5-diag.sh: e5 CUDA at main for two hours with MALLOC_MMAP_THRESHOLD_=65536. If the growth is heap
# allocator retention, the series go flat; if it is memory in use, they keep growing.
set -uo pipefail
H=<local>
W=$H/ws
OUT=$H/out-e5-main-2h-mmap
while systemctl --user is-active -q starling-e5-diag; do sleep 20; done
mkdir -p "$OUT"
export DOTNET_ROOT=$W/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 NUGET_PACKAGES=$W/nuget PATH=$W/dotnet:$PATH
export STARLING_MODELS=$W/models CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0
export LD_LIBRARY_PATH="$(find "$W"/venvs/eg2-cuda/lib/python3.11/site-packages/nvidia -maxdepth 2 -type d -name lib | sort | paste -sd: -)"
export MALLOC_MMAP_THRESHOLD_=65536
cd "$H/ci-main/starling"
git rev-parse HEAD >"$OUT/commit.txt"
{ nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader | head -1; cat /proc/loadavg; } >"$OUT/env.txt"
dotnet "$H/ci-main/build/soak/Starlings.Soak.Tool.dll" e5 --cuda --duration 2h --out "$OUT/e5" \
    --note "local RTX 3090, main, two hours, MALLOC_MMAP_THRESHOLD_=65536 diagnostic" >"$OUT/e5.log" 2>&1
echo "soak exit $?" >>"$OUT/env.txt"
{ nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader | head -1; cat /proc/loadavg; } >>"$OUT/env.txt"
