#!/usr/bin/env bash
# Builds and runs the GLiNER2 CUDA stress reproducer (docs/evidence/gliner2-cuda-wrong-output/).
#   CUDA_VISIBLE_DEVICES=<gpu> ai-dotnet/tools/Starlings.Gliner2.CudaStress/run.sh [options]
# Needs the CUDA 13 runtime and cuDNN 9 on LD_LIBRARY_PATH (or in the ldconfig cache), an explicit
# CUDA_VISIBLE_DEVICES, and the GLiNER2 snapshot in STARLING_GLINER2_SNAPSHOT (tokenizer.json and
# onnx/gliner2-entities-fp32.onnx).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the GPU to use}"
: "${STARLING_GLINER2_SNAPSHOT:?set STARLING_GLINER2_SNAPSHOT to the GLiNER2 snapshot directory}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
dotnet="${DOTNET:-dotnet}"
"$dotnet" build "$here" -c Release --nologo -v quiet >/dev/null
exec "$dotnet" "$here/bin/Release/net10.0/Starlings.Gliner2.CudaStress.dll" "$STARLING_GLINER2_SNAPSHOT" "$@"
