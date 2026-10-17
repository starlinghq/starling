#!/usr/bin/env bash
# Sensitivity of the token embeddings to soft-token perturbations: Gaussian noise of sigma = noise x max|soft| added to
# the reference soft tokens, then the released multimodal text graph.
#   noise.sh <python with onnxruntime 1.30.0> <embeddinggemma2-text-mm-fp32.onnx> <hosted refs dir> <out dir>
set -euo pipefail
py="$1"; mm="$2"; refs="$3"; out="$4"
here="$(cd "$(dirname "$0")" && pwd)"
jsonl="$here/../../../../model-pipeline/protected/embeddinggemma-2/release/video-fp32-release/video-embeddings.jsonl"
for n in 1e-5 1e-4; do
  for sd in 1 2; do
    "$py" -I "$here/textgraph_diag.py" "$mm" "$jsonl" "$refs" "$out" \
      --rows commons-video-04,commons-video-05,commons-video-00 --noise "$n" --seed "$sd" | grep -v session | sed "s/^/noise $n seed $sd /"
  done
done
