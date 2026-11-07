#!/usr/bin/env bash
# One-hour CUDA soak of qwen3.5-0.8b/ortgenai-fp32x-cuda at ai-v0.5.0: same tool build, CUDA libraries and contract
# environment as the run's 10-minute soak, with the duration of the one-hour G4 rule.
set -uo pipefail
H=<local>
W=$H/ws
OUT=$H/out-soak-1h-0.5.0
mkdir -p "$OUT"
export DOTNET_ROOT=$W/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 PATH=$W/dotnet:$PATH
export STARLING_MODELS=$W/models CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0
export LD_LIBRARY_PATH="$(find "$W"/venvs/eg2-cuda/lib/python3.11/site-packages/nvidia -maxdepth 2 -type d -name lib | sort | paste -sd: -)"
cd "$H/ci/starling"
{ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1; cat /proc/loadavg; cat /sys/kernel/mm/transparent_hugepage/enabled; } >"$OUT/env.txt"
dotnet "$H/ci/build/soak/Starlings.Soak.Tool.dll" qwen35 --cuda --duration 1h --out "$OUT/qwen35" \
    --note "local RTX 3090 regression for 0.5.0, commit ce5225854fa7, CUDA libraries from the pinned eg2 requirements" >"$OUT/qwen35.log" 2>&1
echo "exit $?" >>"$OUT/env.txt"
cat /proc/loadavg >>"$OUT/env.txt"
