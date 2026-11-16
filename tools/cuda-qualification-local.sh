#!/usr/bin/env bash
# CUDA qualification on the maintainer's machine: the regression support of the CUDA profiles (docs/support-policy.md,
# "Regression support"), run locally on an NVIDIA GeForce RTX 3090 before every release tag. It replaces the former
# RunPod job `cuda` of .github/workflows/qualification.yml and does the same work with the same output layout, so its
# evidence stays comparable with the runs recorded before (docs/evidence/gpu-regression-2026-10-16/ and later).
#
#   tools/cuda-qualification-local.sh --gpu <index> <out dir> [soak duration, default 10m]
#
# Run it from a clean checkout of the commit to qualify (the commit and any modified tracked file are recorded; a
# dirty checkout fails the check `checkout-clean`). Order: inventory, reference environments, builds, model
# acquisition, the CUDA test projects (each judged by its exit status and by tools/ci/check-trx.py), G1/G2/G3 of each
# CUDA profile on its accepted P-030 release sets, then one short soak per CUDA profile. Every check is recorded in
# <out>/summary.tsv as `<profile> <check> <result>`; a check whose input is missing is `not-run`, never `passed`.
# Exit status (also in <out>/exit-code): 0 only if every check passed. <out>/summary.md is the table to attach to the
# release evidence; <out>/status.txt is the progress log.
#
# Environment:
#   --gpu <index>        the GPU to use, by PCI bus order (CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=<index>);
#                        required, never a default device: on the shared workstation GPU 0 belongs to another
#                        workload. The run refuses to start unless that GPU is an NVIDIA GeForce RTX 3090 (the only GPU
#                        the CUDA contracts cover) with driver 580 or later.
#   STARLING_MODELS      required: the verified model cache (tools/acquire-models.sh downloads what is missing and
#                        verifies every file's size and SHA-256).
#   STARLING_EMBEDDINGGEMMA2_VIDEO_TENSORS
#                        the EmbeddingGemma 2 video reference tensors (run_embeddinggemma2_video.py --tensors) for the
#                        CUDA test projects; they have no reproducible public source, so the run records `not-run`
#                        without them.
#   STARLING_KEV_08B_BUNDLE, STARLING_KEV_4B_BUNDLE
#                        optional: Kev bundles for Starlings.Models.Kev.Tests.Cuda; default the acquired
#                        kev-0.8b/onnx-fp32-cuda and kev-4b/onnx-fp32-cuda snapshots.
#   CUDA_QUALIFICATION_WORK
#                        optional: where reference environments, builds and prepared release inputs live (default
#                        ${XDG_CACHE_HOME:-~/.cache}/starling-cuda-qualification); kept between runs. Release inputs
#                        are verified against the digests in Git on every use, so a cached file can make a check fail,
#                        never pass.
#   dotnet, uv, git, curl, nvidia-smi on PATH.
#
# Fatal-signal tests: tests that inject a fatal signal into the test process may run only on GitHub-hosted runners,
# where a crashed process cannot take a terminal session or another job with it; such a test gates on
# RUNNER_ENVIRONMENT=github-hosted. This script never sets RUNNER_ENVIRONMENT and unsets it for its own run, so no such
# test runs here, also when it is started from a shell that has it.
#
# Memory: the workstation is shared (other sessions, CI runners, a model server). Everything runs one step at a time
# (no background preparation, one test module at a time), and before each heavy step (a build, a test module, a gate
# producer, a soak) the script asks `agent-budget level` when that command is on PATH: on `critical` it waits with
# `agent-budget alert --from critical --timeout 300` (repeatedly) before going on, on `tight` builds use -m:2.
#
# Release inputs that are not in Git (media from Wikimedia Commons, the pinned FFmpeg builds, the fixture image set)
# are prepared under the work directory, each verified against the SHA-256 digests in Git by its producer or by the
# evaluator. The G2 reference tensors that release sets keep outside Git are never regenerated (they reproduce
# byte-identically only on the reference CPU): they are downloaded from the commit each protected
# hosted-reference.json pins and verified file by file (tools/fetch-release-references.py). Nothing here writes to
# model-pipeline/protected/.
set -uo pipefail
unset RUNNER_ENVIRONMENT HF_TOKEN

GPU=""
if [[ ${1:-} == --gpu ]]; then
    GPU="${2:-}"
    shift 2
fi
OUT="${1:-}"
SOAK_DURATION="${2:-10m}"
if [[ ! $GPU =~ ^[0-9]+$ || -z $OUT ]]; then
    sed -n '7p' "$0" >&2
    exit 2
fi
: "${STARLING_MODELS:?STARLING_MODELS must name the verified model cache}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${CUDA_QUALIFICATION_WORK:-${XDG_CACHE_HOME:-$HOME/.cache}/starling-cuda-qualification}"
B=$WORK/build
V=$WORK/venvs
CACHE=$WORK/cache
mkdir -p "$OUT" "$B" "$V" "$CACHE"
OUT="$(cd "$OUT" && pwd)"
rm -rf "$OUT/DONE" "$OUT/exit-code" "$OUT/tests" "$OUT/gates" "$OUT/soak" "$OUT/trx-check.md" # a reused out dir
: >"$OUT/summary.tsv"
: >"$OUT/status.txt"
START=$SECONDS
cd "$REPO" || exit 1
SHA="$(git rev-parse HEAD)"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/status.txt"; }
record() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >>"$OUT/summary.tsv"; log "result $1 $2: $3"; }
finish() {
    local rc=0
    if [[ ! -s "$OUT/summary.tsv" ]] || cut -f3 "$OUT/summary.tsv" | grep -qv '^passed$'; then rc=1; fi
    {
        echo "## CUDA qualification (local)"
        echo
        echo "Commit \`$SHA\`, GPU $GPU $(nvidia-smi -i "$GPU" --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1), host $(hostname 2>/dev/null || echo unknown), wall $(((SECONDS - START) / 60)) min."
        echo
        echo "| Profile | Check | Result |"
        echo "| --- | --- | --- |"
        awk -F'\t' '{ printf "| `%s` | %s | %s |\n", $1, $2, $3 }' "$OUT/summary.tsv"
        if [[ -s "$OUT/trx-check.md" ]]; then
            echo
            echo "### CUDA test modules (tools/ci/check-trx.py)"
            echo
            cat "$OUT/trx-check.md"
        fi
    } >"$OUT/summary.md"
    echo "$rc" >"$OUT/exit-code"
    log "LOCAL RUN END exit $rc"
    touch "$OUT/DONE"
    exit "$rc"
}

# Memory budget of the shared workstation (see the header). DOTNET_M is -m:2 for builds while the level is tight.
DOTNET_M=()
budget() {  # budget <what>: wait while the level is critical
    DOTNET_M=()
    command -v agent-budget >/dev/null 2>&1 || return 0
    local level
    level="$(agent-budget level 2>/dev/null || echo unknown)"
    while [[ $level == critical ]]; do
        log "memory budget critical before $1: waiting (agent-budget alert --from critical --timeout 300)"
        agent-budget alert --from critical --timeout 300 >/dev/null 2>&1 || true
        level="$(agent-budget level 2>/dev/null || echo unknown)"
    done
    [[ $level == tight ]] && DOTNET_M=(-m:2)
    return 0
}

export DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
export UV_NO_PROGRESS=1 HF_HUB_DISABLE_TELEMETRY=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU NVIDIA_TF32_OVERRIDE=0

# The CPU threads of the Python reference stages: the cgroup CPU quota when one is set (cgroup v2, else v1), else a
# fixed cap, never more than the CPUs this process may use. PyTorch and OpenMP otherwise size their pools from the host.
cpu_quota() {  # prints "<cpus> <source>"; <cpus> is "-" when no quota is set or readable
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
    echo "- none"
}
read -r QUOTA_CPUS QUOTA_SOURCE <<<"$(cpu_quota)"
[[ $QUOTA_CPUS == - ]] && QUOTA_CPUS=""
AFFINITY_CPUS="$(nproc)"
REF_THREADS_CAP=16
if [[ -n $QUOTA_CPUS ]]; then
    REF_THREADS=$((QUOTA_CPUS < AFFINITY_CPUS ? QUOTA_CPUS : AFFINITY_CPUS))
    REF_THREADS_SOURCE="quota from $QUOTA_SOURCE, affinity $AFFINITY_CPUS"
else
    REF_THREADS=$((REF_THREADS_CAP < AFFINITY_CPUS ? REF_THREADS_CAP : AFFINITY_CPUS))
    REF_THREADS_SOURCE="no CPU quota: fixed cap $REF_THREADS_CAP, affinity $AFFINITY_CPUS"
fi
REF_ENV=(OMP_NUM_THREADS="$REF_THREADS" MKL_NUM_THREADS="$REF_THREADS" OPENBLAS_NUM_THREADS="$REF_THREADS")

# ---------------------------------------------------------------------------------------------------------------------
log "local run start: commit $SHA, GPU $GPU, soak $SOAK_DURATION, work $WORK"
log "CPU: quota ${QUOTA_CPUS:-none} ($QUOTA_SOURCE), affinity $AFFINITY_CPUS; Python reference stages: $REF_THREADS threads ($REF_THREADS_SOURCE)"
{
    echo "== GPU"; nvidia-smi; nvidia-smi --query-gpu=index,name,driver_version,memory.total,pci.bus_id --format=csv
    echo "selected: CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "== CPU"; lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core|Socket)'
    echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)  memory.max: $(cat /sys/fs/cgroup/memory.max 2>/dev/null)"
    echo "nproc: $AFFINITY_CPUS; CPU quota: ${QUOTA_CPUS:-none} ($QUOTA_SOURCE); Python reference threads: $REF_THREADS ($REF_THREADS_SOURCE)"
    echo "loadavg: $(cat /proc/loadavg)"
    free -g; uname -a
    echo "transparent_hugepage: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo n/a)"
    echo "== disk"; df -h "$REPO" "$STARLING_MODELS" "$WORK"
    echo "== tools"; dotnet --list-sdks; uv --version; git --version
    echo "agent-budget: $(command -v agent-budget || echo 'not on PATH'); level: $(agent-budget level -v 2>/dev/null || echo n/a)"
    echo "RUNNER_ENVIRONMENT: ${RUNNER_ENVIRONMENT:-unset}"
    echo "== checkout"; git log -1 --format='%H %cI %s'; git status --porcelain --untracked-files=no
} >"$OUT/environment.txt" 2>&1
driver="$(nvidia-smi -i "$GPU" --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
gpu_name="$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
log "GPU $GPU: $gpu_name, driver $driver"
# The CUDA contracts cover the RTX 3090 (Ampere) only; results from another GPU are never recorded as this run's.
if [[ "$gpu_name" != "NVIDIA GeForce RTX 3090" ]]; then
    record all gpu-is-rtx-3090 "failed (GPU $GPU is '$gpu_name'; the CUDA contracts cover only the NVIDIA GeForce RTX 3090)"
    finish
fi
if [[ ! ${driver%%.*} =~ ^[0-9]+$ ]] || ((${driver%%.*} < 580)); then
    record all driver-580-or-later "failed (driver $driver; CUDA 13 needs 580 or later)"
    finish
fi
dirty="$(git status --porcelain --untracked-files=no | wc -l)"
if [[ $dirty == 0 ]]; then
    record all checkout-clean passed
else
    record all checkout-clean "failed ($dirty modified tracked paths; see environment.txt)"
fi
P="model-pipeline/protected"
REG="model-pipeline/registry"

# ---------------------------------------------------------------------------------------------------------------------
# Reference environments, from the hash-locked requirements at this commit, one at a time.
mkvenv() {  # mkvenv <name> <requirements> <hash mode> <index: pypi|torch-cpu>
    local name=$1 req=$2 mode=$3 index=$4 v=$V/$1 extra=()
    [[ $index == torch-cpu ]] && extra=(--extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match)
    {
        rm -rf "$v" &&
            uv venv -q --python 3.11 --python-preference only-managed "$v" &&
            uv pip install -q --python "$v/bin/python" "$mode" -r "$req" --index-url https://pypi.org/simple "${extra[@]}" &&
            uv pip install -q --python "$v/bin/python" --no-deps -e model-pipeline &&
            uv pip install -q --python "$v/bin/python" jsonschema pyyaml &&
            uv pip freeze --python "$v/bin/python" >"$OUT/venvs/$name.freeze.txt"
    } >"$OUT/venvs/$name.log" 2>&1
}
mkdir -p "$OUT/venvs"
log "reference environments"
{
    rm -rf "$V/mp" && uv venv -q --python 3.11 --python-preference only-managed "$V/mp" && uv pip install -q --python "$V/mp/bin/python" -e model-pipeline
} >"$OUT/venvs/mp.log" 2>&1 || { record all model-pipeline-environment failed; finish; }
export PATH=$V/mp/bin:$PATH # python3 (tools/qualification.py) and the model-pipeline CLI
budget "reference environments"
mkvenv encoder-cuda model-pipeline/reference/encoder/requirements-cuda.txt --require-hashes pypi
mkvenv gliner2-cuda model-pipeline/reference/gliner2/requirements-cuda.txt --verify-hashes pypi
mkvenv eg2-cuda model-pipeline/reference/embeddinggemma2/requirements-cuda.txt --require-hashes pypi
mkvenv eg2-video model-pipeline/reference/embeddinggemma2/requirements-video.txt --require-hashes torch-cpu
for v in encoder-cuda gliner2-cuda eg2-cuda eg2-video; do
    [[ -s "$OUT/venvs/$v.freeze.txt" ]] || record all "reference environment $v" "failed (venvs/$v.log)"
done

# The CUDA 13 runtime and cuDNN 9 from the pinned nvidia wheels of the eg2 requirements-cuda environment, as the CUDA
# contracts state.
cuda_libs() {
    local site
    site="$("$1" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" || return 1
    find "$site/nvidia" -maxdepth 2 -type d -name lib | sort | paste -sd: -
}
CUDA_LIBS="$(cuda_libs "$V/eg2-cuda/bin/python")"
export LD_LIBRARY_PATH="$CUDA_LIBS"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >>"$OUT/environment.txt"

# ---------------------------------------------------------------------------------------------------------------------
log "builds"
BUILD_OK=1
build() {  # build <log name> <dotnet build args...>
    local name=$1
    shift
    budget "build $name"
    if dotnet build "$@" -c Release -v q -nologo -p:RestoreLockedMode=true "${DOTNET_M[@]}" >"$OUT/build-$name.log" 2>&1; then :; else
        BUILD_OK=0
        log "build $name failed (build-$name.log)"
        return 1
    fi
}
build fidelity ai-dotnet/tools/Starlings.Fidelity.Tool.Cuda
build qwen35-fidelity ai-dotnet/tools/Starlings.Qwen35.Fidelity -p:UseCuda=true -o "$B/qwen35-fidelity"
build soak ai-dotnet/tools/Starlings.Soak.Tool -p:UseCuda=true -o "$B/soak"
build acquisition ai-dotnet/tools/Starlings.Acquisition.Tool
TOOL=(dotnet ai-dotnet/tools/Starlings.Fidelity.Tool.Cuda/bin/Release/net10.0/Starlings.Fidelity.Tool.dll)
QWEN=(dotnet "$B/qwen35-fidelity/Starlings.Qwen35.Fidelity.dll")
SOAK=(dotnet "$B/soak/Starlings.Soak.Tool.dll")
if [[ $BUILD_OK != 1 ]]; then
    record all builds failed
fi

# ---------------------------------------------------------------------------------------------------------------------
PROFILES=(
    multilingual-e5-small/onnx-fp32-cuda
    gliner2-base-v1/onnx-fp32-cuda
    embeddinggemma-2/onnx-fp32-cuda
    embeddinggemma-2/image-fp32-cuda
    embeddinggemma-2/video-fp32-cuda
    qwen3.5-0.8b/ortgenai-fp32x-cuda
    qwen3.5-0.8b/ortgenai-fp16x-cuda
)
# Kev's CUDA profiles are acquired for Starlings.Models.Kev.Tests.Cuda only (no release gates or soaks here).
KEV_PROFILES=(kev-0.8b/onnx-fp32-cuda kev-4b/onnx-fp32-cuda)
log "acquire and verify models"
if tools/acquire-models.sh "$OUT/locks" "${PROFILES[@]}" "${KEV_PROFILES[@]}" >"$OUT/acquire.log" 2>&1; then
    record all models-acquired-and-verified passed
else
    record all models-acquired-and-verified failed
fi
snap() {  # snapshot directory of a profile's lock
    echo "$STARLING_MODELS/$(python3 -I tools/qualification.py snapshot "$OUT/locks/${1//\//--}.json")"
}
E5_SNAP="$(snap multilingual-e5-small/onnx-fp32-cuda)"
G2_SNAP="$(snap gliner2-base-v1/onnx-fp32-cuda)"
EG2_SNAP="$(snap embeddinggemma-2/video-fp32-cuda)"
QW_SNAP="$(snap qwen3.5-0.8b/ortgenai-fp16x-cuda)"
export STARLING_E5_SNAPSHOT=$E5_SNAP
export STARLING_KEV_08B_BUNDLE="${STARLING_KEV_08B_BUNDLE:-$(snap kev-0.8b/onnx-fp32-cuda)}"
export STARLING_KEV_4B_BUNDLE="${STARLING_KEV_4B_BUNDLE:-$(snap kev-4b/onnx-fp32-cuda)}"

# ---------------------------------------------------------------------------------------------------------------------
# Release and test inputs outside Git, prepared before the GPU work (one step at a time on this shared machine).
mkdir -p "$OUT/prepare"
ready() { [[ -f "$1/.complete" ]]; }
log "test inputs: EmbeddingGemma 2 image set and the pinned FFmpeg builds (tools/ci/eg2-test-inputs.sh)"
if tools/ci/eg2-test-inputs.sh "$CACHE/eg2-test-inputs" >"$OUT/prepare/eg2-test-inputs.log" 2>&1; then
    # shellcheck disable=SC1091 # written by eg2-test-inputs.sh
    source "$CACHE/eg2-test-inputs/env"
else
    record all "test inputs (eg2-test-inputs.sh)" "failed (prepare/eg2-test-inputs.log)"
fi
FF8="${STARLING_FFMPEG_DIR:-$CACHE/eg2-test-inputs/missing-ffmpeg8}"
FF9="${STARLING_FFMPEG9_DIR:-$CACHE/eg2-test-inputs/missing-ffmpeg9}"
if [[ -z ${STARLING_EMBEDDINGGEMMA2_VIDEO_TENSORS:-} || ! -d ${STARLING_EMBEDDINGGEMMA2_VIDEO_TENSORS:-} ]]; then
    record all "test inputs (STARLING_EMBEDDINGGEMMA2_VIDEO_TENSORS)" \
        "not-run (set it to the run_embeddinggemma2_video.py --tensors output; the EmbeddingGemma 2 video CUDA tests need it)"
fi
MEDIA_REL=$CACHE/media/release
MEDIA_IMG=$CACHE/media/image-retrieval-v1
MEDIA_VID=$CACHE/media/video-retrieval-v1
# Hosted G2 references (protected hosted-reference.json): about 190 MB, downloaded and verified every run.
REF_IMG=$WORK/refs/eg2-image-release
REF_VID=$WORK/refs/eg2-video-release
REF_QW=$WORK/refs/qwen35-release-prompts
fetch_media() {  # fetch_media <media list> <dir>
    ready "$2" || { rm -rf "$2" && "$V/eg2-video/bin/python" -I model-pipeline/reference/embeddinggemma2/fetch_eg2_release_media.py \
        --media "$1" --out "$2" && touch "$2/.complete"; }
}
fetch_ref() {  # fetch_ref <protected hosted-reference.json> <dir> <log name>: download, verify, prune
    rm -f "$2/.complete"
    python3 -I tools/fetch-release-references.py --prune "$1" "$2" >"$OUT/prepare/$3.log" 2>&1 && touch "$2/.complete"
}
log "release inputs (cache $CACHE)"
fetch_ref "$P/embeddinggemma-2/release/image-fp32-release/hosted-reference.json" "$REF_IMG" ref-image
fetch_ref "$P/embeddinggemma-2/release/video-fp32-release/hosted-reference.json" "$REF_VID" ref-video
fetch_ref "$P/qwen3.5-0.8b/release/release-prompts/reference/hosted-reference.json" "$REF_QW" ref-qwen
fetch_media "$P/embeddinggemma-2/release/media/media.json" "$MEDIA_REL" >"$OUT/prepare/media-release.log" 2>&1
fetch_media "$P/embeddinggemma-2/release/image-retrieval-v1/image-media.json" "$MEDIA_IMG" >"$OUT/prepare/media-image-retrieval.log" 2>&1
fetch_media "$P/embeddinggemma-2/release/video-retrieval-v1/video-media.json" "$MEDIA_VID" >"$OUT/prepare/media-video-retrieval.log" 2>&1

# ---------------------------------------------------------------------------------------------------------------------
# The CUDA test projects, one at a time. Each is judged by its exit status, and all of them together by
# tools/ci/check-trx.py: a module whose every test skipped (a model or input missing) fails, so a run cannot pass on
# skipped tests.
log "CUDA test projects"
export STARLING_CUDA_TESTS=1
for t in E5 Gliner2 EmbeddingGemma2 Qwen35 Kev; do
    proj=ai-dotnet/tests/Starlings.Models.$t.Tests.Cuda
    budget "tests $t"
    if dotnet build "$proj" -c Release -v q -nologo -p:RestoreLockedMode=true "${DOTNET_M[@]}" >"$OUT/tests-$t.log" 2>&1 &&
        dotnet test --project "$proj" -c Release --no-build --results-directory "$OUT/tests/$t" -- --report-xunit-trx \
            >>"$OUT/tests-$t.log" 2>&1; then r=passed; else r=failed; fi
    counts="$(awk '/^Test run summary/ { s = 1 } s && /^ *(total|failed|succeeded|skipped):/ { gsub(/^ +/, ""); printf "%s%s", sep, $0; sep = ", " }' "$OUT/tests-$t.log")"
    record "Starlings.Models.$t.Tests.Cuda" "tests (${counts:-see tests-$t.log})" "$r"
done
unset STARLING_CUDA_TESTS
python3 -I tools/ci/check-trx.py "$OUT/tests" >"$OUT/trx-check.txt" 2>&1
trx_rc=$?
grep -E '^\|' "$OUT/trx-check.txt" >"$OUT/trx-check.md"
if [[ $trx_rc == 0 ]]; then
    record all "CUDA test modules each passed a test (check-trx.py)" passed
else
    record all "CUDA test modules each passed a test (check-trx.py)" "failed ($(tail -n 1 "$OUT/trx-check.txt"))"
fi

# ---------------------------------------------------------------------------------------------------------------------
# evaluate <profile> <name> <gates> <model-pipeline evaluate args...>: every named gate must be `passed` in report.json.
evaluate() {
    local profile=$1 name=$2 gates=$3
    shift 3
    local dir="$OUT/gates/${profile//\//--}/$name" status=0 verdict
    model-pipeline evaluate "$REG/$profile/manifest.yaml" "$@" --out "$dir" >"$dir.log" 2>&1 || status=$?
    verdict="$(python3 -I - "$dir/report.json" "$gates" <<'PY'
import json, sys
try:
    report = json.load(open(sys.argv[1], encoding="utf-8"))
except OSError:
    print("no report")
    sys.exit(1)
bad = []
for gate in sys.argv[2].split(","):
    status = report.get("gates", {}).get(gate, {}).get("status", "none")
    if status != "passed":
        bad.append(f"{gate} {status}")
print("passed" if not bad else ", ".join(bad))
PY
)"
    [[ $verdict == passed ]] || verdict="$verdict (evaluator exit $status)"
    record "$profile" "$name $gates" "$verdict"
}
# step <profile> <log name> <command...>: a producer; on failure the evaluation that needs it fails on missing results.
step() {
    local profile=$1 name=$2 rc=0 s=$SECONDS threads=""
    shift 2
    local logf="$OUT/gates/${profile//\//--}/$name.log"
    [[ " $* " =~ \ OMP_NUM_THREADS=([0-9]+)\  ]] && threads=", OMP_NUM_THREADS ${BASH_REMATCH[1]}"
    budget "$profile $name"
    "$@" >"$logf" 2>&1 || rc=$?
    log "$profile $name: exit $rc, $((SECONDS - s)) s (CPU quota ${QUOTA_CPUS:-none}$threads; loadavg $(cut -d' ' -f1-3 /proc/loadavg))"
    [[ $rc == 0 ]] || record "$profile" "$name" "failed (exit $rc)"
}
need() {  # need <profile> <check> <ready dir>: record not-run when a prepared input is missing
    ready "$3" && return 0
    record "$1" "$2" "not-run (input not prepared: ${3#"$WORK"/}; see prepare/)"
    return 1
}
work() { local d=$WORK/work/${1//\//--}; rm -rf "$d"; mkdir -p "$d" "$OUT/gates/${1//\//--}"; echo "$d"; }

gates_e5() {
    local pr=multilingual-e5-small/onnx-fp32-cuda w REL=$P/multilingual-e5-small/release/onnx-fp32-release
    local DS=$P/release-shared/retrieval/miracl-dev-hn-8lang-v1
    w="$(work $pr)"
    step $pr g1-holdout-producer "${TOOL[@]}" tokenizer --model e5 --tokenizer "$E5_SNAP/tokenizer.json" --fixtures "$REL/tokenizer.jsonl" --out "$w/g1-holdout"
    evaluate $pr g1-tokenizer-holdout G1 --tokenizer-fixtures "$REL/tokenizer.jsonl" --results "$w/g1-holdout"
    step $pr dotnet-stages "${TOOL[@]}" e5 --out "$w/stages" --fixtures "$REL" --snapshot "$E5_SNAP" --hidden --threads 4 --device cuda:0
    step $pr python-stage2 env "${REF_ENV[@]}" "$V/encoder-cuda/bin/python" model-pipeline/reference/encoder/stage_e5_export.py --model-dir "$E5_SNAP" \
        --fixtures "$REL" --profile $pr --out "$w/stages" --intra-op-threads 4 --provider cuda
    evaluate $pr g1-g2-staged G1,G2 --fixtures "$REL" --results "$w/stages" --require-gates G1,G2
    step $pr g3-producer "${TOOL[@]}" retrieval --model e5 --dataset "$DS" --out "$w/g3/embeddings.jsonl" --snapshot "$E5_SNAP" --threads 4 --device cuda:0
    evaluate $pr g3-retrieval G3 --g3 retrieval --dataset "$DS" --results "$w/g3/embeddings.jsonl"
}
gates_gliner2() {
    local pr=gliner2-base-v1/onnx-fp32-cuda w TOK=$P/gliner2-base-v1/release/tokenizer-release/gliner2-tokenizer.jsonl
    local REL=$P/gliner2-base-v1/release/extraction-release DS=$P/gliner2-base-v1/release/ner/wnut17-test-v1/gold.jsonl
    w="$(work $pr)"
    step $pr g1-tokenizer-producer "${TOOL[@]}" tokenizer --model gliner2 --tokenizer "$G2_SNAP/tokenizer.json" --fixtures "$TOK" --out "$w/g1-tokenizer"
    evaluate $pr g1-tokenizer-release G1 --tokenizer-fixtures "$TOK" --results "$w/g1-tokenizer"
    step $pr dotnet-stages "${TOOL[@]}" gliner2 --out "$w/stages" --fixtures "$REL" --snapshot "$G2_SNAP" --threads 4 --device cuda:0
    step $pr python-stage2 env OMP_NUM_THREADS=4 "$V/gliner2-cuda/bin/python" model-pipeline/reference/gliner2/stage_gliner2.py export \
        --model-dir "$G2_SNAP" --graph "$G2_SNAP/onnx/gliner2-entities-fp32.onnx" --graph-asset-prefix onnx/gliner2-entities-fp32 \
        --fixtures "$REL" --profile $pr --out "$w/stages" --threads 4 --provider cuda
    evaluate $pr g1-g2-extraction-release G1,G2 --fixtures "$REL" --results "$w/stages" --require-gates G1,G2
    step $pr g3-producer "${TOOL[@]}" ner --dataset "$DS" --out "$w/g3/predictions.jsonl" --snapshot "$G2_SNAP" --threads 4 --device cuda:0
    evaluate $pr g3-ner G3 --g3 ner --dataset "$DS" --results "$w/g3/predictions.jsonl"
}
gates_eg2_text() {
    # The release set's reference embeddings.jsonl is in Git since #60; the producers read the case ids from it.
    local pr=embeddinggemma-2/onnx-fp32-cuda w REL=$P/embeddinggemma-2/release/onnx-fp32-release
    local DS=$P/release-shared/retrieval/miracl-dev-hn-8lang-v1
    w="$(work $pr)"
    step $pr g1-holdout-producer "${TOOL[@]}" tokenizer --model embeddinggemma2 --tokenizer "$EG2_SNAP/tokenizer.json" --fixtures "$REL/tokenizer.jsonl" --out "$w/g1-holdout"
    evaluate $pr g1-tokenizer-holdout G1 --tokenizer-fixtures "$REL/tokenizer.jsonl" --results "$w/g1-holdout"
    step $pr dotnet-stages "${TOOL[@]}" embeddinggemma2 --out "$w/stages" --fixtures "$REL" --snapshot "$EG2_SNAP" --threads 6 --device cuda:0
    step $pr python-stage2 env "${REF_ENV[@]}" PYTHONPATH=model-pipeline/reference/embeddinggemma2 "$V/eg2-cuda/bin/python" \
        model-pipeline/reference/embeddinggemma2/stage_embeddinggemma2_export.py --model-dir "$EG2_SNAP" \
        --graph "$EG2_SNAP/onnx/embeddinggemma2-text-fp32.onnx" --fixtures "$REL" --profile $pr --out "$w/stages" \
        --intra-op-threads 6 --provider cuda
    evaluate $pr g2-release G1,G2 --fixtures "$REL" --results "$w/stages" --require-gates G1,G2
    step $pr g3-producer "${TOOL[@]}" retrieval --model embeddinggemma2 --dataset "$DS" --out "$w/g3/embeddings.jsonl" --snapshot "$EG2_SNAP" --threads 4 --device cuda:0
    evaluate $pr g3-retrieval G3 --g3 retrieval --dataset "$DS" --results "$w/g3/embeddings.jsonl"
}
gates_eg2_image() {
    local pr=embeddinggemma-2/image-fp32-cuda w REL=$P/embeddinggemma-2/release/image-fp32-release
    local SET=$P/embeddinggemma-2/release/image-retrieval-v1
    w="$(work $pr)"
    if need $pr "g1-g2-image-release G1,G2" "$REF_IMG" && need $pr "g1-g2-image-release G1,G2" "$MEDIA_REL"; then
        # One accepted image (79 MP) exceeds the 50 MP default: the release evaluation uses the option (experimental capability).
        step $pr dotnet-stages "${TOOL[@]}" embeddinggemma2-images --out "$w/stages" --images "$MEDIA_REL" --fixtures "$REL" \
            --threads 4 --max-image-pixels 80000000 --device cuda:0
        evaluate $pr g1-g2-image-release G1,G2 --fixtures "$REL" --results "$w/stages" --reference-tensors "$REF_IMG" --require-gates G1,G2
    fi
    if need $pr "g3-retrieval G3" "$MEDIA_IMG"; then
        step $pr g3-producer "${TOOL[@]}" retrieval --model embeddinggemma2-image --dataset "$SET" --media "$MEDIA_IMG" \
            --out "$w/g3/embeddings.jsonl" --device cuda --threads 16
        evaluate $pr g3-retrieval G3 --g3 retrieval --dataset "$SET" --results "$w/g3/embeddings.jsonl"
    fi
}
gates_eg2_video() {
    local pr=embeddinggemma-2/video-fp32-cuda w REL=$P/embeddinggemma-2/release/video-fp32-release
    local SET=$P/embeddinggemma-2/release/video-retrieval-v1
    w="$(work $pr)"
    if need $pr "g1-g2-video-release G1,G2" "$CACHE/eg2-test-inputs/ffmpeg9" && need $pr "g1-g2-video-release G1,G2" "$REF_VID" &&
        need $pr "g1-g2-video-release G1,G2" "$MEDIA_REL"; then
        step $pr dotnet-stages "${TOOL[@]}" embeddinggemma2-videos --out "$w/stages" --videos "$MEDIA_REL" --ffmpeg "$FF8" --ffmpeg9 "$FF9" \
            --fixtures "$REL" --threads 4 --device cuda:0
        evaluate $pr g1-g2-video-release G1,G2 --fixtures "$REL" --results "$w/stages" --reference-tensors "$REF_VID" --require-gates G1,G2
    fi
    if need $pr "g3-retrieval G3" "$MEDIA_VID" && need $pr "g3-retrieval G3" "$CACHE/eg2-test-inputs/ffmpeg8"; then
        step $pr g3-producer "${TOOL[@]}" retrieval --model embeddinggemma2-video --dataset "$SET" --media "$MEDIA_VID" \
            --out "$w/g3/embeddings.jsonl" --ffmpeg "$FF8" --device cuda --threads 16
        evaluate "$pr" g3-retrieval G3 --g3 retrieval --dataset "$SET" --results "$w/g3/embeddings.jsonl"
    fi
}
gates_qwen() {  # gates_qwen fp32x|fp16x
    local var=$1 pr=qwen3.5-0.8b/ortgenai-$1-cuda w TOK=$P/qwen3.5-0.8b/release/tokenizer-release/tokenizer.jsonl
    local RP=$P/qwen3.5-0.8b/release/release-prompts DS=$P/qwen3.5-0.8b/release/task-quality/gsm8k-test-v1/prompts.jsonl
    w="$(work "$pr")"
    step "$pr" g1-tokenizer-producer "${TOOL[@]}" tokenizer --model qwen35 --tokenizer "$QW_SNAP/transformers-5.18.0/tokenizer.json" --fixtures "$TOK" --out "$w/g1-tokenizer"
    evaluate "$pr" g1-tokenizer-release G1 --tokenizer-fixtures "$TOK" --results "$w/g1-tokenizer"
    if need "$pr" "g1-g2-release-prompts G1,G2" "$REF_QW"; then
        step "$pr" release-prompts "${QWEN[@]}" --bundle "$QW_SNAP" --variant "$var" --device cuda --prompts "$RP/prompts-release-v1.jsonl" \
            --reference "$RP/reference" --out "$w/run" --modes parity,fixture-semantics --threads 4
        evaluate "$pr" g1-g2-release-prompts G1,G2 --fixtures "$RP/reference" --results "$w/run" --reference-tensors "$REF_QW" --require-gates G1,G2
    fi
    step "$pr" g3-producer "${QWEN[@]}" generation --bundle "$QW_SNAP" --prompts "$DS" --out "$w/g3" --variant "$var" --device cuda --threads 4
    evaluate "$pr" g3-gsm8k G3 --g3 generation --dataset "$DS" --results "$w/g3"
}

log "release gates: e5, GLiNER2, EmbeddingGemma 2 text"
gates_e5
gates_gliner2
gates_eg2_text
log "release gates: Qwen3.5"
gates_qwen fp32x
gates_qwen fp16x
log "release gates: EmbeddingGemma 2 image and video"
gates_eg2_image
gates_eg2_video

# ---------------------------------------------------------------------------------------------------------------------
# Short soaks (screening, not the one-hour G4 run), one at a time, each under its profile's contract environment:
# MALLOC_MMAP_THRESHOLD_=65536 where the profile's README states it as a Linux deployment requirement (GLiNER2;
# EmbeddingGemma 2 video on CUDA). Transparent huge pages are the host's setting (the soak precondition is madvise; this
# script does not change kernel settings); every soak report records transparent_hugepage.
log "soaks ($SOAK_DURATION each), transparent_hugepage: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo n/a)"
mkdir -p "$OUT/soak"
soak() {  # soak <profile> <workload> [VAR=value...]
    local profile=$1 model=$2 rc=0
    shift 2
    local note
    note="local CUDA qualification, $gpu_name (GPU $GPU), commit ${SHA:0:12}, CUDA libraries from the pinned eg2 requirements-cuda wheels${*:+, $*}"
    budget "soak $model"
    env "$@" "${SOAK[@]}" "$model" --cuda --duration "$SOAK_DURATION" --out "$OUT/soak/$model" --note "$note" >"$OUT/soak/$model.log" 2>&1 || rc=$?
    local line smoke
    line="$(grep -E '\] (PASS|FAIL)|^error:' "$OUT/soak/$model.log" | tail -1 | cut -c1-200)"
    # Smoke test: wrong outputs, errors and crashes fail the check. The tool's growth verdict (its rule is unchanged)
    # is recorded but not binding here: a short run sits in the allocator's warm-up ramp (the binding one-hour Qwen fp32x
    # CUDA soak ramps 18.4 then 10.5 MiB per half-hour, 27.8 MiB in all), so growth is judged by the one-hour G4 runs.
    smoke="$(python3 -I - "$OUT/soak/$model" <<'PY'
import glob, json, sys
reports = sorted(glob.glob(sys.argv[1] + "/*.json"))
if not reports:
    print("failed: no soak report (crash or startup error)")
    sys.exit(0)
r = json.load(open(reports[-1], encoding="utf-8"))
cycles = len(r.get("cycles") or [])
if r.get("wrong_output_count", 1) or r.get("error_count", 1) or cycles == 0:
    print(f"failed: {r.get('wrong_output_count')} wrong outputs, {r.get('error_count')} errors, {cycles} cycles")
else:
    print(f"passed: smoke ({cycles} cycles, no wrong output or error); growth verdict not binding: {r.get('verdict', '')[:120]}")
PY
)"
    if [[ $smoke == passed* ]]; then
        log "$profile soak: ${smoke#passed: }"
        record "$profile" "soak $SOAK_DURATION smoke${*:+ $*}" passed
    else
        record "$profile" "soak $SOAK_DURATION smoke${*:+ $*}" "failed (exit $rc: ${smoke#failed: }; ${line:-see soak/$model.log})"
    fi
}
soak multilingual-e5-small/onnx-fp32-cuda e5
soak gliner2-base-v1/onnx-fp32-cuda gliner2 MALLOC_MMAP_THRESHOLD_=65536
soak embeddinggemma-2/onnx-fp32-cuda embeddinggemma2
soak embeddinggemma-2/image-fp32-cuda embeddinggemma2-image
soak embeddinggemma-2/video-fp32-cuda embeddinggemma2-video MALLOC_MMAP_THRESHOLD_=65536
soak qwen3.5-0.8b/ortgenai-fp32x-cuda qwen35
soak qwen3.5-0.8b/ortgenai-fp16x-cuda qwen35-fp16x

log "repository dirty paths after the run: $(git status --porcelain | wc -l)"
finish
