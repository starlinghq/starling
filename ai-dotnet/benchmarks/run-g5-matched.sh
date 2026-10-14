#!/usr/bin/env bash
# Matched-backend G5 benchmark for the model harnesses: the C# harness and its Python same-backend runner
# (model-pipeline/reference/bench/) on the same graphs, workload file, thread settings and warm/cold protocol,
# interleaved over several rounds of fresh processes. Then check the budgets with `model-pipeline bench g5 <out.jsonl>`.
#
#   ai-dotnet/benchmarks/run-g5-matched.sh eg2-text|eg2-image|gliner2|qwen35 cpu|cuda <out.jsonl> \
#       [rounds=3] [threads=6] [cold-processes-per-round=10]
#
# Environment:
#   STARLING_MODELS        model cache (the acquisition layout); the snapshots below default to it
#   BENCH_PYTHON           Python with the runner's pinned environment (docs/benchmarks/PROTOCOL.md, "Running")
#   BENCH_NOTE             machine note recorded in every run's environment block
#   BENCH_SIDES            both (default) or python; the C# model harnesses are CPU builds only, so a cuda run is
#                          Python-only until they have CUDA builds
#   STARLING_EMBEDDINGGEMMA2_IMAGES   image set (make_eg2_images.py output), for eg2-image
#   CUDA_DEVICE_ORDER, CUDA_VISIBLE_DEVICES   required for cuda (the GPU to use; never implicit)
set -euo pipefail

model="${1:?eg2-text|eg2-image|gliner2|qwen35}"
provider="${2:?cpu|cuda}"
out="${3:?out.jsonl}"
rounds="${4:-3}"
threads="${5:-6}"
cold="${6:-10}"
sides="${BENCH_SIDES:-both}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
bench="$repo/model-pipeline/reference/bench"
python="${BENCH_PYTHON:?set BENCH_PYTHON to the benchmark Python environment of the runner}"
models="${STARLING_MODELS:?set STARLING_MODELS}"
note="${BENCH_NOTE:-}"

if [[ "$provider" == "cuda" ]]; then
  : "${CUDA_VISIBLE_DEVICES:?cuda runs need an explicit CUDA_VISIBLE_DEVICES (and CUDA_DEVICE_ORDER=PCI_BUS_ID)}"
  if [[ "$sides" != "python" ]]; then
    echo "The C# model harnesses are CPU builds only; set BENCH_SIDES=python for a Python-only CUDA run." >&2
    exit 2
  fi
fi

snapshot() { find "$models/$1" -mindepth 1 -maxdepth 1 -type d | sort | head -n 1; }
py_args=()
cs_warm=()
cs_cold=()
case "$model" in
  eg2-text|eg2-image)
    project="Starlings.Benchmarks.EmbeddingGemma2Harness"
    runner="$bench/bench_eg2_matched.py"
    path="${model#eg2-}"
    py_args=(--snapshot "${STARLING_EMBEDDINGGEMMA2_SNAPSHOT:-$(snapshot google/embeddinggemma-2)}" --paths "$path")
    cs_warm=(--paths "$path")
    cs_cold=(--paths "$path" --cold-path "$path")
    py_cold=(--cold-path "$path")
    ;;
  gliner2)
    project="Starlings.Benchmarks.Gliner2Harness"
    runner="$bench/bench_gliner2_matched.py"
    inputs="${out%.jsonl}.gliner2-inputs.jsonl"
    py_args=(--snapshot "${STARLING_GLINER2_SNAPSHOT:-$(snapshot fastino/gliner2-base-v1)}" --inputs "$inputs")
    py_cold=()
    ;;
  qwen35)
    project="Starlings.Benchmarks.Qwen35Harness"
    runner="$bench/bench_qwen35_matched.py"
    py_args=(--snapshot "${STARLING_QWEN35_SNAPSHOT:-$(snapshot Qwen/Qwen3.5-0.8B)}")
    py_cold=()
    ;;
  *)
    echo "unknown model '$model' (eg2-text, eg2-image, gliner2, qwen35; e5 has run-e5-matched.sh)" >&2
    exit 2
    ;;
esac

harness="$here/$project/bin/Release/net10.0/$project"
dotnet build "$here/$project" -c Release -v quiet -nologo >/dev/null
common=(--threads "$threads" --inter-threads 1 --out "$out" --note "$note")

if [[ "$model" == "gliner2" ]]; then
  # The exact tensors the C# Prepare builds, for the Python inference-stage runner (one dump for every round).
  "$harness" dump-inputs --out "$inputs" --threads "$threads"
fi

# Background load log (<out>.load.tsv), every 5 s: load average and each GPU's utilization and memory.
load_log="${out%.jsonl}.load.tsv"
(
  set +e +o pipefail
  while true; do
    gpus="$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | paste -sd';' -)"
    printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(cut -d' ' -f1-3 /proc/loadavg)" "$gpus"
    sleep 5
  done
) >>"$load_log" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true' EXIT

for round in $(seq 1 "$rounds"); do
  echo "== round $round/$rounds ($model, $provider)" >&2
  if [[ "$sides" == "both" ]]; then
    "$harness" warm "${common[@]}" "${cs_warm[@]}"
  fi
  "$python" "$runner" warm --provider "$provider" "${common[@]}" "${py_args[@]}"
  if [[ "$sides" == "both" ]]; then
    "$harness" cold "${common[@]}" "${cs_cold[@]}" --processes "$cold"
  fi
  "$python" "$runner" cold --provider "$provider" "${common[@]}" "${py_args[@]}" "${py_cold[@]}" --processes "$cold"
done
echo "raw results in $out; check: model-pipeline bench g5 $out" >&2
