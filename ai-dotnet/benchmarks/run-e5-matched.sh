#!/usr/bin/env bash
# Matched-backend e5 benchmark (P-008): C# (library and direct) and Python on the same ONNX Runtime version, model
# file, thread settings and workload file, interleaved over several rounds of fresh processes, then cold starts.
#
#   ai-dotnet/benchmarks/run-e5-matched.sh cpu|cuda <out.jsonl> [rounds=3] [threads=6] [cold-processes-per-round=10]
#
# Environment:
#   STARLING_E5_SNAPSHOT  e5 snapshot directory (onnx/model.onnx, tokenizer.json)
#   BENCH_PYTHON                    Python with model-pipeline/reference/encoder/requirements-bench.txt
#                                   (cpu) or requirements-bench-cuda.txt (cuda)
#   BENCH_NOTE                      machine note recorded in every run's environment block
#   CUDA_DEVICE_ORDER, CUDA_VISIBLE_DEVICES   required for cuda (the GPU to use; never implicit)
#
# Raw results are appended to <out.jsonl>; summarize with `model-pipeline bench summarize <out.jsonl>`.
set -euo pipefail

provider="${1:?cpu|cuda}"
out="${2:?out.jsonl}"
rounds="${3:-3}"
threads="${4:-6}"
cold="${5:-10}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
python="${BENCH_PYTHON:?set BENCH_PYTHON to the benchmark Python environment}"
note="${BENCH_NOTE:-}"
: "${STARLING_E5_SNAPSHOT:?set STARLING_E5_SNAPSHOT}"

project="Starlings.Benchmarks.E5Harness"
if [[ "$provider" == "cuda" ]]; then
  : "${CUDA_VISIBLE_DEVICES:?cuda runs need an explicit CUDA_VISIBLE_DEVICES (and CUDA_DEVICE_ORDER=PCI_BUS_ID)}"
  project="Starlings.Benchmarks.E5Harness.Cuda"
  # The C# harness loads the same CUDA 13 and cuDNN 9 libraries as the Python runner: those of its nvidia-* wheels.
  site="$("$python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  libs="$(find "$site/nvidia" -maxdepth 2 -type d -name lib | sort | paste -sd: -)"
  export LD_LIBRARY_PATH="$libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

dotnet build "$here/$project" -c Release -v quiet -nologo >/dev/null
harness="$here/$project/bin/Release/net10.0/$project"
py="$repo/model-pipeline/reference/encoder/bench_e5_matched.py"
common=(--provider "$provider" --threads "$threads" --inter-threads 1 --out "$out" --note "$note")

# Background load log (<out>.load.tsv), every 5 s for the whole run: load average, each GPU's utilization and memory,
# GPU compute processes, and the busiest processes that are not this benchmark (instantaneous %CPU from top). The
# per-run load_before/load_after blocks cannot tell this benchmark's own load from other jobs; this log can.
load_log="${out%.jsonl}.load.tsv"
(
  set +e +o pipefail  # a sampling failure (for example no foreign process to list) must not stop the sampler
  while true; do
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    gpus="$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | paste -sd';' -)"
    apps="$(nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
      | awk -F', ' '{n=$3; sub(/ .*/, "", n); sub(/.*\//, "", n); printf "%s,%s,%s,%s;", substr($1, 10), $2, n, $4}')"
    foreign="$(top -b -n 1 -c -w 300 -o %CPU 2>/dev/null | awk 'NR>7 && $9+0>=20' \
      | grep -v -e Starlings.Benchmarks -e bench_e5_matched -e ' top -b' \
      | awk '{cmd=$12; for(i=13;i<=NF&&i<=14;i++) cmd=cmd" "$i; printf "%s:%s ", $9, cmd}')"
    printf '%s\t%s\t%s\t%s\t%s\n' "$ts" "$(cut -d' ' -f1-3 /proc/loadavg)" "$gpus" "$apps" "$foreign"
    sleep 5
  done
) >>"$load_log" &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true' EXIT

for round in $(seq 1 "$rounds"); do
  echo "== round $round/$rounds ($provider)" >&2
  "$harness" warm "${common[@]}" --impl library
  "$harness" warm "${common[@]}" --impl direct --paths ids
  "$python" "$py" warm "${common[@]}"
  "$harness" cold "${common[@]}" --impl library --processes "$cold"
  "$python" "$py" cold "${common[@]}" --processes "$cold"
done
