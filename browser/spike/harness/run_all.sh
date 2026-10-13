#!/usr/bin/env bash
# The timing and drift runs of docs/m0/browser-wasm-spike.md, in sequence (load average recorded per step).
# Usage: run_all.sh <repo> <publish-interp> <publish-aot> <chrome> <tok models> <graphs dir> <out dir>
# PYTHON (the requirements-browser.txt environment) and DOTNET (dotnet executable) can be set.
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$1" interp="$2" aot="$3" chrome="$4" tok="$5" graphs="$6" out="$7"
python="${PYTHON:-python3}"
dotnet="${DOTNET:-dotnet}"
fixtures="$repo/model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu"
mkdir -p "$out"
: > "$out/load.txt"
step() { echo "$1 $(cut -d' ' -f1-3 /proc/loadavg)" >> "$out/load.txt"; }

step native-tokenizer
"$dotnet" "$repo/browser/spike/BrowserSpike.Native/bin/Release/net10.0/BrowserSpike.Native.dll" "$repo" "$tok" 3 3 \
  > "$out/tok-native.json"
for mode in interp aot; do
  publish="$interp"; [[ "$mode" == aot ]] && publish="$aot"
  step "browser-tokenizer-$mode"
  "$python" "$here/run_spike.py" tokenizer --publish "$publish" --label "$mode" --chrome "$chrome" --models-root "$tok" \
    --out "$out/tok-$mode.json"
done

step native-ort
"$python" "$repo/model-pipeline/reference/encoder/drift_e5_browser.py" native --fixtures "$fixtures" --graphs "$graphs" \
  --variants fp32 fp16 int8-dynamic int8-dynamic-pc int8-embed --threads 6 --out "$out/native-ort-py.json"

results=()
for variant in fp32 fp16 int8-dynamic int8-dynamic-pc int8-embed; do
  for backend in wasm webgpu; do
    # fp16 on WebGPU needs shader-f16, which the adapter does not expose here (docs/m0/browser-wasm-spike.md).
    [[ "$variant" == fp16 && "$backend" == webgpu ]] && continue
    step "e5-$variant-$backend"
    iterations=20; [[ "$backend" == webgpu ]] && iterations=10
    timeout 900 "$python" "$here/run_spike.py" e5 --publish "$aot" --label aot --chrome "$chrome" --models-root "$tok" \
      --graphs "$graphs" --variant "$variant" --backend "$backend" --threads 6 --iterations "$iterations" \
      --timeout-s 800 --out "$out/e5-$variant-$backend.json" && results+=("$out/e5-$variant-$backend.json")
  done
done
step e5-fp32-wasm-interp
timeout 900 "$python" "$here/run_spike.py" e5 --publish "$interp" --label interp --chrome "$chrome" --models-root "$tok" \
  --graphs "$graphs" --variant fp32 --backend wasm --threads 6 --iterations 20 --timeout-s 800 \
  --out "$out/e5-fp32-wasm-interp.json" && results+=("$out/e5-fp32-wasm-interp.json")
step e5-int8-embed-wasm-1thread-no-isolation
timeout 900 "$python" "$here/run_spike.py" e5 --publish "$aot" --label aot --chrome "$chrome" --models-root "$tok" \
  --graphs "$graphs" --variant int8-embed --backend wasm --threads 1 --no-isolation --iterations 20 --timeout-s 800 \
  --out "$out/e5-int8-embed-wasm-t1.json" && results+=("$out/e5-int8-embed-wasm-t1.json")
step done

"$python" "$repo/model-pipeline/reference/encoder/drift_e5_browser.py" browser --fixtures "$fixtures" \
  --native "$out/native-ort-py.json" --out "$out/browser-drift.json" "${results[@]}"
