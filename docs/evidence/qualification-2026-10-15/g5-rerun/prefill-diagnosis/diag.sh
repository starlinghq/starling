#!/usr/bin/env bash
# diag.sh <rounds> <samples-scale>: interleaved warm short-decode64 runs on the pod, pinned to cores 16-21 (NUMA node 0).
set -uo pipefail
W=/workspace
export DOTNET_ROOT=$W/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 NUGET_PACKAGES=$W/nuget
export PATH=$W/bin:$DOTNET_ROOT:$PATH
export STARLING_MODELS=$W/models
rounds="$1"; scale="$2"
cd $W/starling
S=$W/models/Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17
PY=$W/venvs/genai-cpu/bin/python
H=ai-dotnet/benchmarks/Starlings.Benchmarks.Qwen35Harness/bin/Release
cs=$H/net10.0/Starlings.Benchmarks.Qwen35Harness
# (b) pip build swapped into a copy of the harness
rm -rf $H/pipswap && cp -a $H/net10.0 $H/pipswap
cp $W/venvs/genai-cpu/lib/python3.11/site-packages/onnxruntime/capi/libonnxruntime.so.1.30.0 $H/pipswap/runtimes/linux-x64/native/libonnxruntime.so
sha256sum $H/pipswap/runtimes/linux-x64/native/libonnxruntime.so $H/net10.0/runtimes/linux-x64/native/libonnxruntime.so
out=$W/out/qwen-prefill-diag; rm -rf $out; mkdir -p $out
common=(warm --provider cpu --threads 6 --inter-threads 1 --only short-decode64 --samples-scale "$scale")
pin=(taskset -c 16-21)
run() {  # run <label> <side-args> <cmd...>
  local label=$1 side=$2; shift 2
  read -r -a sa <<<"$side"
  echo "$(date -Is) $label (load $(cut -d' ' -f1-3 /proc/loadavg))"
  "${pin[@]}" "$@" "${common[@]}" "${sa[@]}" --out "$out/$label.jsonl" --note "diag $label" >"$out/$label.log" 2>&1 || echo "  exit $? (see $out/$label.log)"
}
for r in $(seq 1 "$rounds"); do
  run a-csharp-nuget "--variant fp32x" "$cs"
  run a-python "--snapshot $S --variant fp32x" "$PY" model-pipeline/reference/bench/bench_qwen35_matched.py
  run b-csharp-pipswap "--variant fp32x" $H/pipswap/Starlings.Benchmarks.Qwen35Harness
  run c-csharp-pgo0 "--variant fp32x" env DOTNET_TieredPGO=0 "$cs"
  run c-csharp-tc0 "--variant fp32x" env DOTNET_TieredCompilation=0 "$cs"
  run c-csharp-gcc0 "--variant fp32x" env DOTNET_gcConcurrent=0 "$cs"
done
echo DIAG-DONE
