#!/usr/bin/env bash
# CUDA soak growth diagnostics: diag.sh <pod-name> <model> <duration> <run>...
#   run = a | b | c | d | e | f   (see case below). One run at a time; each writes /workspace/out/<pod>/<run-name>/.
# A sidecar samples /proc/<pid>/smaps (by mapping class), /proc/<pid>/status and GPU memory.used every 20 s.
set -u
W=/workspace
pod="$1"; model="$2"; dur="$3"; shift 3
out=$W/out/$pod
mkdir -p "$out"
export DOTNET_ROOT=$W/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 NUGET_PACKAGES=$W/nuget
export STARLING_MODELS=$W/models
export STARLING_E5_SNAPSHOT=$W/models/intfloat/multilingual-e5-small/614241f622f53c4eeff9890bdc4f31cfecc418b3
export STARLING_EMBEDDINGGEMMA2_IMAGES=$W/fixtures/images-r1
export STARLING_EMBEDDINGGEMMA2_VIDEO_TENSORS=$W/fixtures/video-spike/ref-clean-tensors
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0
SOAK_DLL="${SOAK_DLL:-$W/build/soak-cuda/Starlings.Soak.Tool.dll}"
REPO="${REPO:-$W/starling}"
cd "$REPO"
gpu="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
cpu="$(lscpu | sed -n 's/^Model name: *//p' | head -1)"
log() { echo "$(date -Is) $*" | tee -a "$out/status.txt"; }
site="$($W/venvs/eg2-cuda/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
libs="$(find "$site/nvidia" -maxdepth 2 -type d -name lib | sort | paste -sd: -)"
{
  echo "pod: $pod (RunPod secure cloud EU-CZ-1, network volume starling-q-vol)"
  echo "commit: $(git rev-parse HEAD) dirty: $(git status --porcelain | wc -l) paths"
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
  lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core|Socket)'
  echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"
  echo "nproc: $(nproc)"
  ldd --version | head -1
  echo "LD_LIBRARY_PATH=$libs"
  echo "SOAK_DLL=$SOAK_DLL"
} >"$out/env.txt" 2>&1

for run in "$@"; do
  envs=(); args=()
  case "$run" in
    a) name=a-baseline ;;
    b) name=b-mmap-threshold-64k; envs=(MALLOC_MMAP_THRESHOLD_=65536) ;;
    c) name=c-malloc-arena-max-2; envs=(MALLOC_ARENA_MAX=2) ;;
    d) name=d-requests-only; args=(--keep-loaded) ;;
    e) name=e-load-unload-only; args=(--requests-per-cycle 1 --deadline-every 0) ;;
    f) name=f-malloc-trim; args=(--malloc-trim) ;;
    *) log "unknown run $run"; continue ;;
  esac
  dir="$out/$name"; mkdir -p "$dir"
  log "start $name ($model, $dur) ${envs[*]} ${args[*]}"
  t0=$SECONDS
  env LD_LIBRARY_PATH="$libs" "${envs[@]}" $W/dotnet/dotnet "$SOAK_DLL" "$model" --cuda \
      --duration "$dur" "${args[@]}" --out "$dir" \
      --note "CUDA soak growth diagnostic $name, RunPod $pod, $gpu, CPU $cpu, CUDA libraries from the pinned wheels${envs[*]:+, ${envs[*]}}" \
      >"$dir/run.log" 2>&1 &
  pid=$!
  : >"$dir/smaps.jsonl"
  while kill -0 $pid 2>/dev/null; do
    python3 -I $W/q/cuda-growth/smaps-sum.py $pid $((SECONDS - t0)) >>"$dir/smaps.jsonl" 2>/dev/null
    echo "$((SECONDS - t0)) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" >>"$dir/gpu.txt"
    sleep 20
  done
  wait $pid; rc=$?
  log "end $name exit $rc wall $((SECONDS - t0))s: $(grep -E '\] (PASS|FAIL|skipped)|^error:' "$dir/run.log" | tail -1 | cut -c1-300)"
done
log "DIAG DONE"
touch "$out/DONE"
