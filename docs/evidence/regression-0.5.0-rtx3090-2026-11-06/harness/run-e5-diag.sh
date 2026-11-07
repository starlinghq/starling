#!/usr/bin/env bash
# Two e5 CUDA soaks, one after the other: main for two hours with --malloc-trim (is the growth free memory that glibc
# keeps, or memory in use?), then the tag for two hours with the same options as out-e5-main-2h (like for like).
set -uo pipefail
H=<local>
W=$H/ws
export DOTNET_ROOT=$W/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 NUGET_PACKAGES=$W/nuget PATH=$W/dotnet:$PATH
export STARLING_MODELS=$W/models CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0
export LD_LIBRARY_PATH="$(find "$W"/venvs/eg2-cuda/lib/python3.11/site-packages/nvidia -maxdepth 2 -type d -name lib | sort | paste -sd: -)"
soak() {  # soak <out> <repo> <tool dll> <note> [options...]
    local out=$1 repo=$2 dll=$3 note=$4
    shift 4
    mkdir -p "$out"
    git -C "$repo" rev-parse HEAD >"$out/commit.txt"
    { nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader | head -1; cat /proc/loadavg; } >"$out/env.txt"
    (cd "$repo" && dotnet "$dll" e5 --cuda --duration 2h --out "$out/e5" --note "$note" "$@" >"$out/e5.log" 2>&1)
    echo "soak exit $?" >>"$out/env.txt"
    { nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader | head -1; cat /proc/loadavg; } >>"$out/env.txt"
}
soak "$H/out-e5-main-2h-trim" "$H/ci-main/starling" "$H/ci-main/build/soak/Starlings.Soak.Tool.dll" "local RTX 3090, main, two hours, malloc-trim diagnostic" --malloc-trim
soak "$H/out-e5-tag-2h" "$H/ci/starling" "$H/ci/build/soak/Starlings.Soak.Tool.dll" "local RTX 3090, tag ai-v0.5.0, two hours"
