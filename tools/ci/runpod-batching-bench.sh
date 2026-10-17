#!/usr/bin/env bash
# Pod side of the batching sweep (Phase 0 of the batching investigation; ai-dotnet/benchmarks/README.md, "Batching
# sweep"). Runs on a RunPod RTX 3090 pod with the network volume starling-q-vol at /workspace, like
# runpod-cuda-pod.sh, whose environment it reuses. Start it detached and poll <out>/status.txt and <out>/DONE:
#
#   tools/ci/runpod-batching-bench.sh <commit sha> <out dir> [--quick]
#
# Order: inventory (GPU, driver, CPU quota, memory), checkout of <sha> from origin on the container disk, builds (the
# CPU and .Cuda harnesses of e5, GLiNER2 and EmbeddingGemma 2, Release, locked restore), model acquisition and
# re-verification into /workspace/models, then ai-dotnet/benchmarks/run-batching-sweep.sh on CUDA and on CPU, then the
# summaries. --quick runs the reduced grid (a smoke test of the pod path, minutes instead of hours).
#
# Each provider runs the baseline grid and then the batching configurations of the option contract (run-batching-sweep.sh,
# SWEEP_CONFIGS): sorted (--sort-by-length true), tokens (--max-batch-tokens, SWEEP_BATCH_TOKENS, default 4096),
# sorted-tokens (full grid only) and micro: single-item callers at concurrency 1/4/16 (--quick: 4/16) through the
# library's micro-batcher (MaxWait 0 ms / MaxRows 8; the full grid adds 2 ms / 16) next to a baseline run of
# single-item callers on the same grid. SWEEP_CONFIGS and SWEEP_BATCH_TOKENS pass through from this script's environment.
#
# Writes into <out>: status.txt (progress), environment.txt (inventory), checkout.log, build-*.log, acquire.log,
# cuda/ and cpu/ (raw sweep-*.jsonl, per-run logs, sweep-status.tsv, summary-<provider>.md and .json), summary.md (both
# providers and the status table), exit-code and DONE. Exit status 0 only if every step and sweep run passed.
#
# Uses from the volume: the .NET SDK (/workspace/dotnet), uv and its managed Python, the NuGet cache, the verified model
# cache (/workspace/models, re-verified here) and the eg2-cuda venv's pinned nvidia wheels (CUDA 13 runtime, cuDNN 9)
# for LD_LIBRARY_PATH. Never uses /workspace/shim (it turns `dotnet build` into a no-op), never changes
# /workspace/starling, and writes nothing under model-pipeline/protected/.
#
# CPU thread counts follow the container's CPU quota (the pods show 256 CPUs but run under a CFS quota of about 31;
# see runpod-cuda-pod.sh): the main CPU sweep uses min(quota, 16) intra-op threads and the thread sweep the powers of
# two up to the quota plus the quota itself. ORT's default (one thread per visible core) is never used on the pod.
set -uo pipefail

SHA="${1:?commit sha}"
OUT="${2:?out dir}"
QUICK="${3:-}"
[[ -z $QUICK || $QUICK == --quick ]] || { echo "third argument is --quick or nothing" >&2; exit 2; }
W=/workspace
CI=/root/ci-batching
REPO=$CI/starling
V=$CI/venvs
mkdir -p "$OUT" "$CI" "$V"
OUT="$(cd "$OUT" && pwd)"
: >"$OUT/summary.tsv"
START=$SECONDS

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/status.txt"; }
record() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >>"$OUT/summary.tsv"; log "result $1 $2: $3"; }
finish() {
    local rc=0
    if [[ ! -s "$OUT/summary.tsv" ]] || cut -f3 "$OUT/summary.tsv" | grep -qv '^passed$'; then rc=1; fi
    {
        echo "# Batching sweep on RunPod"
        echo
        echo "Commit \`$SHA\`${QUICK:+ ($QUICK)}, GPU $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)," \
            "CPU quota ${QUOTA_CPUS:-none} (${QUOTA_SOURCE:-}), main CPU threads ${CPU_THREADS:-n/a}, wall $(((SECONDS - START) / 60)) min."
        echo
        echo "| Scope | Step | Result |"
        echo "| --- | --- | --- |"
        awk -F'\t' '{ printf "| `%s` | %s | %s |\n", $1, $2, $3 }' "$OUT/summary.tsv"
        for p in cuda cpu; do
            if [[ -f "$OUT/$p/summary-$p.md" ]]; then
                echo
                sed 's/^# Batching sweep summary/# Batching sweep summary ('"$p"')/; s/^#/##/' "$OUT/$p/summary-$p.md"
            fi
        done
    } >"$OUT/summary.md"
    echo "$rc" >"$OUT/exit-code"
    log "POD RUN END exit $rc"
    touch "$OUT/DONE"
    exit "$rc"
}

export DOTNET_ROOT=$W/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 NUGET_PACKAGES=$W/nuget
export UV_PYTHON_INSTALL_DIR=$W/uv-python UV_CACHE_DIR=$CI/uv-cache UV_NO_PROGRESS=1
export PATH=$W/dotnet:$W/bin:$PATH
export STARLING_MODELS=$W/models
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0
export HF_HUB_DISABLE_TELEMETRY=1
unset HF_TOKEN

# The container's CPU quota, as runpod-cuda-pod.sh reads it (cgroup v2 cpu.max, else v1 cfs_quota_us).
cpu_quota() {
    local q p d
    if read -r q p 2>/dev/null </sys/fs/cgroup/cpu.max && [[ $q =~ ^[0-9]+$ && $p =~ ^[0-9]+$ && $p -gt 0 ]]; then
        echo "$((q / p > 0 ? q / p : 1)) cgroup-v2-cpu.max($q/$p)"
        return
    fi
    for d in /sys/fs/cgroup/cpu,cpuacct /sys/fs/cgroup/cpu /sys/fs/cgroup/cpuacct,cpu; do
        q="$(cat "$d/cpu.cfs_quota_us" 2>/dev/null)" p="$(cat "$d/cpu.cfs_period_us" 2>/dev/null)"
        if [[ $q =~ ^[0-9]+$ && $p =~ ^[0-9]+$ && $p -gt 0 ]]; then
            echo "$((q / p > 0 ? q / p : 1)) cgroup-v1-$d/cpu.cfs_quota_us($q/$p)"
            return
        fi
    done
    echo " none"
}
cpu_throttled() {
    local f
    if [[ -r /sys/fs/cgroup/cpu.stat ]] && grep -q '^throttled_usec' /sys/fs/cgroup/cpu.stat; then
        awk '/^nr_throttled/ { n = $2 } /^throttled_usec/ { t = $2 / 1e6 } END { printf "%d %.0f", n, t }' /sys/fs/cgroup/cpu.stat
        return
    fi
    for f in /sys/fs/cgroup/cpu,cpuacct/cpu.stat /sys/fs/cgroup/cpu/cpu.stat /sys/fs/cgroup/cpuacct,cpu/cpu.stat; do
        if [[ -r $f ]]; then
            awk '/^nr_throttled/ { n = $2 } /^throttled_time/ { t = $2 / 1e9 } END { printf "%d %.0f", n, t }' "$f"
            return
        fi
    done
    echo "n/a"
}
read -r QUOTA_CPUS QUOTA_SOURCE <<<"$(cpu_quota)"
AFFINITY_CPUS="$(nproc)"
CPU_CAP=16 # no quota readable: a fixed cap, never the host's core count
if [[ -n $QUOTA_CPUS ]]; then
    AVAILABLE=$((QUOTA_CPUS < AFFINITY_CPUS ? QUOTA_CPUS : AFFINITY_CPUS))
else
    AVAILABLE=$((CPU_CAP < AFFINITY_CPUS ? CPU_CAP : AFFINITY_CPUS))
fi
CPU_THREADS=$((AVAILABLE < 16 ? AVAILABLE : 16))
THREAD_LIST=""
for t in 1 2 4 8 16 32; do ((t <= AVAILABLE)) && THREAD_LIST="$THREAD_LIST $t"; done
[[ " $THREAD_LIST " == *" $AVAILABLE "* ]] || THREAD_LIST="$THREAD_LIST $AVAILABLE"
[[ $QUICK == --quick ]] && THREAD_LIST="1 $CPU_THREADS"

# ---------------------------------------------------------------------------------------------------------------------
log "pod run start: commit $SHA${QUICK:+ $QUICK}"
log "CPU: quota ${QUOTA_CPUS:-none} ($QUOTA_SOURCE), affinity $AFFINITY_CPUS; main CPU threads $CPU_THREADS, thread sweep:$THREAD_LIST"
{
    echo "== GPU"; nvidia-smi; nvidia-smi --query-gpu=index,name,driver_version,memory.total,pci.bus_id,clocks.max.sm,power.limit --format=csv
    echo "== CPU"; lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core|Socket|NUMA)'
    echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)  memory.max: $(cat /sys/fs/cgroup/memory.max 2>/dev/null)"
    for d in /sys/fs/cgroup/cpu,cpuacct /sys/fs/cgroup/cpu /sys/fs/cgroup/cpuacct,cpu; do
        [[ -r $d/cpu.cfs_quota_us ]] && echo "cgroup v1 $d: cfs_quota_us $(cat "$d/cpu.cfs_quota_us") cfs_period_us $(cat "$d/cpu.cfs_period_us" 2>/dev/null)"
    done
    echo "nproc: $AFFINITY_CPUS; CPU quota: ${QUOTA_CPUS:-none} ($QUOTA_SOURCE); main CPU threads $CPU_THREADS; thread sweep:$THREAD_LIST"
    echo "loadavg: $(cat /proc/loadavg); CFS throttled (periods, s): $(cpu_throttled)"
    free -g; uname -a
    echo "== disk"; df -h / "$W"
    echo "== tools"; "$W/dotnet/dotnet" --list-sdks; "$W/bin/uv" --version; git --version
} >"$OUT/environment.txt" 2>&1
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
log "GPU $gpu_name, driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
# CUDA results are recorded for the RTX 3090 only (the CUDA contracts' GPU); another GPU is not a fallback.
if [[ "$gpu_name" != "NVIDIA GeForce RTX 3090" ]]; then
    record all gpu-is-rtx-3090 "failed ($gpu_name)"
    finish
fi

# ---------------------------------------------------------------------------------------------------------------------
log "checkout"
if ! {
    rm -rf "$REPO" &&
        git clone -q --reference-if-able "$W/starling" --dissociate https://github.com/starlinghq/starling.git "$REPO" &&
        { git -C "$REPO" cat-file -e "$SHA^{commit}" 2>/dev/null || git -C "$REPO" fetch -q origin "$SHA"; } &&
        git -C "$REPO" checkout -q --detach "$SHA" &&
        [[ "$(git -C "$REPO" rev-parse HEAD)" == "$SHA" ]]
} >"$OUT/checkout.log" 2>&1; then
    record all checkout failed
    finish
fi
cd "$REPO" || exit 1
record all checkout passed

# model-pipeline (summaries, model acquisition locks) in its own environment on the container disk.
{
    uv venv -q --python 3.11 --python-preference only-managed "$V/mp" && uv pip install -q --python "$V/mp/bin/python" -e model-pipeline
} >"$OUT/venv-mp.log" 2>&1 || { record all model-pipeline-environment failed; finish; }
export PATH=$V/mp/bin:$PATH
export MODEL_PIPELINE=$V/mp/bin/model-pipeline

# The CUDA 13 runtime and cuDNN 9 from the volume's pinned wheels (eg2 requirements-cuda), as the CUDA contracts state.
site="$("$W/venvs/eg2-cuda/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null)"
CUDA_LIBS="$(find "$site/nvidia" -maxdepth 2 -type d -name lib 2>/dev/null | sort | paste -sd: -)"
if [[ -z $CUDA_LIBS ]]; then
    record all cuda-libraries "failed (no nvidia wheels under $W/venvs/eg2-cuda)"
    finish
fi
export LD_LIBRARY_PATH="$CUDA_LIBS"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >>"$OUT/environment.txt"

# ---------------------------------------------------------------------------------------------------------------------
log "builds"
BUILD_OK=1
for p in E5Harness Gliner2Harness EmbeddingGemma2Harness; do
    for s in "" .Cuda; do
        proj=ai-dotnet/benchmarks/Starlings.Benchmarks.$p$s
        if ! dotnet build "$proj" -c Release -v q -nologo -p:RestoreLockedMode=true >"$OUT/build-$p$s.log" 2>&1; then
            BUILD_OK=0
            record "Starlings.Benchmarks.$p$s" build "failed (build-$p$s.log)"
        fi
    done
done
[[ $BUILD_OK == 1 ]] && record all builds passed || finish

# ---------------------------------------------------------------------------------------------------------------------
log "acquire and verify models"
if tools/acquire-models.sh "$OUT/locks" multilingual-e5-small/onnx-fp32-cuda gliner2-base-v1/onnx-fp32-cuda embeddinggemma-2/onnx-fp32-cuda \
    >"$OUT/acquire.log" 2>&1; then
    record all models-acquired-and-verified passed
else
    record all models-acquired-and-verified failed
    finish
fi

# ---------------------------------------------------------------------------------------------------------------------
note="RunPod $gpu_name, CPU quota ${QUOTA_CPUS:-none} of $AFFINITY_CPUS visible, commit ${SHA:0:12}, CUDA libraries from the pinned eg2 requirements-cuda wheels"
bench() {  # bench <provider> [env assignments...]
    local provider=$1 rc=0 t0 t1 s=$SECONDS
    shift
    t0="$(cpu_throttled)"
    env "$@" SWEEP_BUILD=0 SWEEP_NOTE="$note" ai-dotnet/benchmarks/run-batching-sweep.sh "$provider" "$OUT/$provider" $QUICK \
        >"$OUT/sweep-$provider.log" 2>&1 || rc=$?
    t1="$(cpu_throttled)"
    log "sweep $provider: exit $rc, $((SECONDS - s)) s; CFS throttled before/after: $t0 / $t1; loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
    if [[ -f "$OUT/$provider/sweep-status.tsv" ]]; then
        while IFS=$'\t' read -r scope step result; do record "$scope" "$step" "$result"; done <"$OUT/$provider/sweep-status.tsv"
    else
        record "all/$provider" sweep "failed (exit $rc; sweep-$provider.log)"
    fi
}
log "sweep: CUDA"
bench cuda SWEEP_THREADS=4
log "sweep: CPU ($CPU_THREADS threads; thread sweep:$THREAD_LIST)"
bench cpu SWEEP_THREADS="$CPU_THREADS" SWEEP_THREAD_LIST="$THREAD_LIST" CUDA_VISIBLE_DEVICES=

log "repository dirty paths after the run: $(git status --porcelain | wc -l)"
finish
