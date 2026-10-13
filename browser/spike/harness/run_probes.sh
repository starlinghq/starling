#!/usr/bin/env bash
# Load-only probes of the other catalog graphs in onnxruntime-web (WASM backend), docs/m0/browser-wasm-spike.md.
# Usage: run_probes.sh <publish dir> <chrome> <models root> <out dir>
# <models root> holds the graphs at the paths below (the M0 workstation layout under ~/Projects/AI-models).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
publish="$1" chrome="$2" models="$3" out="$4"
python="${PYTHON:-python3}"

probe() {
  local name="$1" graph="$2" data="$3"
  timeout 900 "$python" "$here/run_spike.py" probe --publish "$publish" --label aot --chrome "$chrome" \
    --graphs "$models" --probe-graph "$graph" --probe-data "$data" --threads 6 --timeout-s 800 \
    --out "$out/probe-$name.json" || echo "probe $name: exit $?"
}

probe eg2-text-fp32 embeddinggemma2/onnx/eg2-text-dynamo-sdpa.onnx embeddinggemma2/onnx/eg2-text-dynamo-sdpa.onnx.data
probe gliner2-fp32 gliner2/onnx/gliner2-A-dynamo.onnx gliner2/onnx/gliner2-A-dynamo.onnx.data
probe qwen35-int4 gen/onnx/int4-cpu/model.onnx gen/onnx/int4-cpu/model.onnx.data
probe qwen35-fp32 gen/onnx/fp32-cpu/model.onnx gen/onnx/fp32-cpu/model.onnx.data
