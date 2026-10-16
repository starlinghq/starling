#!/usr/bin/env bash
# Pod side of the scheduled CUDA qualification (.github/workflows/qualification.yml, job `cuda`). Runs on a RunPod GPU
# pod with the network volume starling-q-vol at /workspace; started detached by the job, which polls <out>/status.txt
# and <out>/DONE and copies <out> back as the workflow artifact.
#
#   runpod-cuda-pod.sh <commit sha> <out dir> [soak duration, default 10m]
#
# Order: inventory, checkout of <sha> on the container disk, builds, reference environments, model acquisition, the
# CUDA test projects, G1/G2/G3 of each CUDA profile on its accepted P-030 release sets, then one short soak per CUDA
# profile. Every check is recorded in <out>/summary.tsv as `<profile> <check> <result>`; a check whose input is missing
# is `not-run`, never `passed`. Exit status (also in <out>/exit-code): 0 only if every check passed.
#
# Uses from the volume: the .NET SDK (/workspace/dotnet), uv and its managed Python, the NuGet cache, the verified
# model cache (/workspace/models, re-verified here), the soak's development fixtures (/workspace/fixtures) and the
# eg2-cuda venv's pinned nvidia wheels (CUDA 13 runtime, cuDNN 9) for LD_LIBRARY_PATH. Never uses /workspace/shim (it
# turns `dotnet build` into a no-op) and never changes /workspace/starling. Release inputs that are not in Git (media
# from Wikimedia Commons, the pinned FFmpeg builds) are cached under /workspace/ci-cache after their first successful
# preparation; the producers and the evaluator verify every one of them against the SHA-256 digests in Git, so a cached
# file can make a check fail, never pass. Delete a cache directory to have the next run prepare it again. The G2
# reference tensors that release sets keep outside Git are never regenerated here (they reproduce byte-identically only
# on the reference CPU): they are downloaded to the container disk from the commit each protected hosted-reference.json
# pins and verified file by file (tools/fetch-release-references.py). Stale copies under /workspace/ci-cache/refs (the
# earlier regenerations) are deleted. Nothing here writes to model-pipeline/protected/.
set -uo pipefail

SHA="${1:?commit sha}"
OUT="${2:?out dir}"
SOAK_DURATION="${3:-10m}"
W=/workspace
CI=/root/ci
REPO=$CI/starling
B=$CI/build
V=$CI/venvs
mkdir -p "$OUT" "$CI" "$B" "$V"
OUT="$(cd "$OUT" && pwd)"
: >"$OUT/summary.tsv"
START=$SECONDS

log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/status.txt"; }
record() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >>"$OUT/summary.tsv"; log "result $1 $2: $3"; }
finish() {
    local rc=0
    if [[ ! -s "$OUT/summary.tsv" ]] || cut -f3 "$OUT/summary.tsv" | grep -qv '^passed$'; then rc=1; fi
    {
        echo "## CUDA qualification on RunPod"
        echo
        echo "Commit \`$SHA\`, GPU $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1), wall $(((SECONDS - START) / 60)) min."
        echo
        echo "| Profile | Check | Result |"
        echo "| --- | --- | --- |"
        awk -F'\t' '{ printf "| `%s` | %s | %s |\n", $1, $2, $3 }' "$OUT/summary.tsv"
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
# The network volume refuses chown: extract archives (the FFmpeg builds) as the current user.
export TAR_OPTIONS=--no-same-owner
unset HF_TOKEN

# ---------------------------------------------------------------------------------------------------------------------
log "pod run start: commit $SHA, soak $SOAK_DURATION"
{
    echo "== GPU"; nvidia-smi; nvidia-smi --query-gpu=index,name,driver_version,memory.total,pci.bus_id --format=csv
    echo "== CPU"; lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core|Socket)'
    echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max 2>/dev/null)  memory.max: $(cat /sys/fs/cgroup/memory.max 2>/dev/null)"
    free -g; uname -a
    echo "== disk"; df -h / "$W"
    echo "== volume"; ls -la "$W"; ls "$W/venvs" "$W/models" "$W/fixtures" "$W/ci-cache" 2>&1
    du -sh "$W"/* 2>/dev/null
    echo "== tools"; "$W/dotnet/dotnet" --list-sdks; "$W/bin/uv" --version; git --version
    echo "runpodctl: $(command -v runpodctl || echo missing)"
    echo "pod environment names: $(env | cut -d= -f1 | grep '^RUNPOD_' | sort | tr '\n' ' ')"
    echo "backstop: $(pgrep -fa 'sleep [0-9]+' | head -3)"
} >"$OUT/environment.txt" 2>&1
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
log "GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), driver $driver"

# Volume space: the earlier runs regenerated reference tensors into $W/ci-cache/refs on this CPU (wrong digests, and
# about 1.5 GB). They are replaced by the hosted, digest-verified references below, so the directory goes, as do
# partial downloads and FFmpeg archives already extracted. The sizes before and after are recorded; nothing outside
# ci-cache is deleted.
{
    echo "== volume before cleanup"; df -h "$W"; du -sh "$W"/* "$W"/ci-cache/* 2>/dev/null
    rm -rf "$W/ci-cache/refs"
    find "$W/ci-cache" -maxdepth 3 \( -name '*.part' -o -name '*.tmp' \) -delete 2>/dev/null
    rm -f "$W"/ci-cache/ffmpeg8/*.tar.xz "$W"/ci-cache/ffmpeg9/*.tar.xz
    echo "== volume after cleanup"; df -h "$W"; du -sh "$W"/ci-cache/* 2>/dev/null
} >"$OUT/volume-cleanup.txt" 2>&1
# Release-input cache: the volume when it has room, else the container disk (prepared again next run, in minutes), so
# a full volume can no longer make a check not-run.
CACHE=$W/ci-cache
free_gb=$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9)
log "volume: ${free_gb} GB free after removing stale ci-cache entries (volume-cleanup.txt)"
if [[ ${free_gb:-0} -lt 8 ]]; then
    CACHE=$CI/cache
    log "volume has ${free_gb} GB free: release inputs cached on the container disk ($CACHE) for this run only"
fi
mkdir -p "$CACHE"

# ---------------------------------------------------------------------------------------------------------------------
log "checkout"
if ! {
    git clone -q --reference-if-able "$W/starling" --dissociate https://github.com/starlinghq/starling.git "$REPO" &&
        git -C "$REPO" checkout -q --detach "$SHA" &&
        [[ "$(git -C "$REPO" rev-parse HEAD)" == "$SHA" ]]
} >"$OUT/checkout.log" 2>&1; then
    record all checkout failed
    finish
fi
cd "$REPO" || exit 1
P="model-pipeline/protected"
REG="model-pipeline/registry"

# ---------------------------------------------------------------------------------------------------------------------
# Reference environments, from the hash-locked requirements at this commit, on the container disk (built in parallel).
mkvenv() {  # mkvenv <name> <requirements> <hash mode> <index: pypi|torch-cpu>
    local name=$1 req=$2 mode=$3 index=$4 v=$V/$1 extra=()
    [[ $index == torch-cpu ]] && extra=(--extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match)
    {
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
    uv venv -q --python 3.11 --python-preference only-managed "$V/mp" && uv pip install -q --python "$V/mp/bin/python" -e model-pipeline
} >"$OUT/venvs/mp.log" 2>&1 || { record all model-pipeline-environment failed; finish; }
export PATH=$V/mp/bin:$PATH # python3 (tools/qualification.py) and the model-pipeline CLI
mkvenv encoder-cuda model-pipeline/reference/encoder/requirements-cuda.txt --require-hashes pypi &
mkvenv gliner2-cuda model-pipeline/reference/gliner2/requirements-cuda.txt --verify-hashes pypi &
mkvenv eg2-cuda model-pipeline/reference/embeddinggemma2/requirements-cuda.txt --require-hashes pypi &
mkvenv eg2-video model-pipeline/reference/embeddinggemma2/requirements-video.txt --require-hashes torch-cpu &
VENV_JOBS=$(jobs -p)

# The CUDA 13 runtime and cuDNN 9 from the volume's pinned wheels (eg2 requirements-cuda), as the CUDA contracts state.
cuda_libs() {
    local site
    site="$("$1" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" || return 1
    find "$site/nvidia" -maxdepth 2 -type d -name lib | sort | paste -sd: -
}
CUDA_LIBS="$(cuda_libs $W/venvs/eg2-cuda/bin/python)"
export LD_LIBRARY_PATH="$CUDA_LIBS"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >>"$OUT/environment.txt"

# ---------------------------------------------------------------------------------------------------------------------
log "builds"
BUILD_OK=1
build() {  # build <log name> <dotnet build args...>
    local name=$1
    shift
    if dotnet build "$@" -c Release -v q -nologo -p:RestoreLockedMode=true >"$OUT/build-$name.log" 2>&1; then :; else
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
log "acquire and verify models"
if tools/acquire-models.sh "$OUT/locks" "${PROFILES[@]}" >"$OUT/acquire.log" 2>&1; then
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
export STARLING_EMBEDDINGGEMMA2_IMAGES=$W/fixtures/images-r1
export STARLING_EMBEDDINGGEMMA2_VIDEO_TENSORS=$W/fixtures/video-spike/ref-clean-tensors

# ---------------------------------------------------------------------------------------------------------------------
# Release inputs outside Git, prepared in the background while the GPU runs tests and the first profiles.
FF8=$CACHE/ffmpeg8/ffmpeg-n8.1.3-17-gc389840894-linux64-lgpl-shared-8.1/lib
FF9=$CACHE/ffmpeg9/ffmpeg-n9.0.2-28-g669f5ab083-linux64-lgpl-shared-9.0/lib
MEDIA_REL=$CACHE/media/release
MEDIA_IMG=$CACHE/media/image-retrieval-v1
MEDIA_VID=$CACHE/media/video-retrieval-v1
# Hosted G2 references (protected hosted-reference.json), on the container disk: about 190 MB, downloaded every run.
REF_IMG=$CI/refs/eg2-image-release
REF_VID=$CI/refs/eg2-video-release
REF_QW=$CI/refs/qwen35-release-prompts
mkdir -p "$OUT/prepare"
ready() { [[ -f "$1/.complete" ]]; }
prepare_ffmpeg() {
    ready "$CACHE/ffmpeg8" || {
        rm -rf "$CACHE/ffmpeg8" && sh model-pipeline/reference/embeddinggemma2/fetch_eg2_ffmpeg.sh "$CACHE/ffmpeg8" &&
            rm -rf "$CACHE"/ffmpeg8/*gpl-shared-8.1.tar.xz "$CACHE"/ffmpeg8/*-gpl-shared-8.1 && touch "$CACHE/ffmpeg8/.complete"
    }
    ready "$CACHE/ffmpeg9" || {
        local a=ffmpeg-n9.0.2-28-g669f5ab083-linux64-lgpl-shared-9.0.tar.xz
        rm -rf "$CACHE/ffmpeg9" && mkdir -p "$CACHE/ffmpeg9" && cd "$CACHE/ffmpeg9" &&
            curl -fsSL -o "$a" "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-10-15-13-03/$a" &&
            echo "5838264a2cff06ddc6b7b2fc43edeb51358ead38fd56d82f83f49c270fe0f5ff  $a" | sha256sum -c - &&
            tar xf "$a" && rm -f "$a" && touch .complete
        cd "$REPO" || return 1
    }
}
fetch_media() {  # fetch_media <media list> <dir>
    ready "$2" || { rm -rf "$2" && "$V/eg2-video/bin/python" -I model-pipeline/reference/embeddinggemma2/fetch_eg2_release_media.py \
        --media "$1" --out "$2" && touch "$2/.complete"; }
}
fetch_ref() {  # fetch_ref <protected hosted-reference.json> <dir> <log name>: download, verify, prune
    rm -f "$2/.complete"
    python3 -I tools/fetch-release-references.py --prune "$1" "$2" >"$OUT/prepare/$3.log" 2>&1 && touch "$2/.complete"
}
prepare_refs() {  # the accepted image, video and Qwen G2 references, verified against the recorded SHA-256
    fetch_ref "$P/embeddinggemma-2/release/image-fp32-release/hosted-reference.json" "$REF_IMG" ref-image
    fetch_ref "$P/embeddinggemma-2/release/video-fp32-release/hosted-reference.json" "$REF_VID" ref-video
    fetch_ref "$P/qwen3.5-0.8b/release/release-prompts/reference/hosted-reference.json" "$REF_QW" ref-qwen
}
prepare_all() {
    prepare_ffmpeg >"$OUT/prepare/ffmpeg.log" 2>&1
    fetch_media "$P/embeddinggemma-2/release/media/media.json" "$MEDIA_REL" >"$OUT/prepare/media-release.log" 2>&1
    fetch_media "$P/embeddinggemma-2/release/image-retrieval-v1/image-media.json" "$MEDIA_IMG" >"$OUT/prepare/media-image-retrieval.log" 2>&1
    fetch_media "$P/embeddinggemma-2/release/video-retrieval-v1/video-media.json" "$MEDIA_VID" >"$OUT/prepare/media-video-retrieval.log" 2>&1
    wait
}
for j in $VENV_JOBS; do wait "$j" || true; done
for v in encoder-cuda gliner2-cuda eg2-cuda eg2-video; do
    [[ -s "$OUT/venvs/$v.freeze.txt" ]] || record all "reference environment $v" "failed (venvs/$v.log)"
done

# ---------------------------------------------------------------------------------------------------------------------
# The CUDA tests run before any background preparation: some time device work (cancellation drain) and must not share
# the CPU with the reference regenerations.
log "CUDA test projects"
export STARLING_CUDA_TESTS=1
for t in E5 Gliner2 EmbeddingGemma2 Qwen35; do
    proj=ai-dotnet/tests/Starlings.Models.$t.Tests.Cuda
    if dotnet test --project "$proj" -c Release --results-directory "$OUT/tests/$t" -- --report-xunit-trx \
        >"$OUT/tests-$t.log" 2>&1; then r=passed; else r=failed; fi
    counts="$(awk '/^Test run summary/ { s = 1 } s && /^ *(total|failed|succeeded|skipped):/ { gsub(/^ +/, ""); printf "%s%s", sep, $0; sep = ", " }' "$OUT/tests-$t.log")"
    record "Starlings.Models.$t.Tests.Cuda" "tests (${counts:-see tests-$t.log})" "$r"
done
unset STARLING_CUDA_TESTS

log "release inputs: preparing in the background (cache $CACHE)"
prepare_refs
prepare_all &
PREPARE_JOB=$!

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
    local profile=$1 name=$2 rc=0 s=$SECONDS
    shift 2
    local logf="$OUT/gates/${profile//\//--}/$name.log"
    "$@" >"$logf" 2>&1 || rc=$?
    log "$profile $name: exit $rc, $((SECONDS - s)) s"
    [[ $rc == 0 ]] || record "$profile" "$name" "failed (exit $rc)"
}
need() {  # need <profile> <check> <ready dir>: record not-run when a prepared input is missing
    ready "$3" && return 0
    record "$1" "$2" "not-run (input not prepared: ${3#"$CACHE"/}; see prepare/)"
    return 1
}
work() { local d=$CI/work/${1//\//--}; mkdir -p "$d" "$OUT/gates/${1//\//--}"; echo "$d"; }

if [[ $BUILD_OK != 1 ]]; then
    record all builds failed
fi

gates_e5() {
    local pr=multilingual-e5-small/onnx-fp32-cuda w REL=$P/multilingual-e5-small/release/onnx-fp32-release
    local DS=$P/release-shared/retrieval/miracl-dev-hn-8lang-v1
    w="$(work $pr)"
    step $pr g1-holdout-producer "${TOOL[@]}" tokenizer --model e5 --tokenizer "$E5_SNAP/tokenizer.json" --fixtures "$REL/tokenizer.jsonl" --out "$w/g1-holdout"
    evaluate $pr g1-tokenizer-holdout G1 --tokenizer-fixtures "$REL/tokenizer.jsonl" --results "$w/g1-holdout"
    step $pr dotnet-stages "${TOOL[@]}" e5 --out "$w/stages" --fixtures "$REL" --snapshot "$E5_SNAP" --hidden --threads 4 --device cuda:0
    step $pr python-stage2 "$V/encoder-cuda/bin/python" model-pipeline/reference/encoder/stage_e5_export.py --model-dir "$E5_SNAP" \
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
    step $pr python-stage2 env PYTHONPATH=model-pipeline/reference/embeddinggemma2 "$V/eg2-cuda/bin/python" \
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
    if need $pr "g1-g2-image-release G1,G2" "$REF_IMG"; then
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
    if need $pr "g1-g2-video-release G1,G2" "$CACHE/ffmpeg9" && need $pr "g1-g2-video-release G1,G2" "$REF_VID"; then
        step $pr dotnet-stages "${TOOL[@]}" embeddinggemma2-videos --out "$w/stages" --videos "$MEDIA_REL" --ffmpeg "$FF8" --ffmpeg9 "$FF9" \
            --fixtures "$REL" --threads 4 --device cuda:0
        evaluate $pr g1-g2-video-release G1,G2 --fixtures "$REL" --results "$w/stages" --reference-tensors "$REF_VID" --require-gates G1,G2
    fi
    if need $pr "g3-retrieval G3" "$MEDIA_VID" && need $pr "g3-retrieval G3" "$CACHE/ffmpeg8"; then
        step $pr g3-producer "${TOOL[@]}" retrieval --model embeddinggemma2-video --dataset "$SET" --media "$MEDIA_VID" \
            --out "$w/g3/embeddings.jsonl" --ffmpeg "$FF8" --device cuda --threads 16
        evaluate $pr g3-retrieval G3 --g3 retrieval --dataset "$SET" --results "$w/g3/embeddings.jsonl"
    fi
}
gates_qwen() {  # gates_qwen fp32x|fp16x
    local var=$1 pr=qwen3.5-0.8b/ortgenai-$1-cuda w TOK=$P/qwen3.5-0.8b/release/tokenizer-release/tokenizer.jsonl
    local RP=$P/qwen3.5-0.8b/release/release-prompts DS=$P/qwen3.5-0.8b/release/task-quality/gsm8k-test-v1/prompts.jsonl
    w="$(work $pr)"
    step $pr g1-tokenizer-producer "${TOOL[@]}" tokenizer --model qwen35 --tokenizer "$QW_SNAP/transformers-5.18.0/tokenizer.json" --fixtures "$TOK" --out "$w/g1-tokenizer"
    evaluate $pr g1-tokenizer-release G1 --tokenizer-fixtures "$TOK" --results "$w/g1-tokenizer"
    if need $pr "g1-g2-release-prompts G1,G2" "$REF_QW"; then
        step $pr release-prompts "${QWEN[@]}" --bundle "$QW_SNAP" --variant "$var" --device cuda --prompts "$RP/prompts-release-v1.jsonl" \
            --reference "$RP/reference" --out "$w/run" --modes parity,fixture-semantics --threads 4
        evaluate $pr g1-g2-release-prompts G1,G2 --fixtures "$RP/reference" --results "$w/run" --reference-tensors "$REF_QW" --require-gates G1,G2
    fi
    step $pr g3-producer "${QWEN[@]}" generation --bundle "$QW_SNAP" --prompts "$DS" --out "$w/g3" --variant "$var" --device cuda --threads 4
    evaluate $pr g3-gsm8k G3 --g3 generation --dataset "$DS" --results "$w/g3"
}

log "release gates: e5, GLiNER2, EmbeddingGemma 2 text"
gates_e5
gates_gliner2
gates_eg2_text
log "release gates: Qwen3.5"
gates_qwen fp32x
gates_qwen fp16x
log "release gates: EmbeddingGemma 2 image and video (waiting for media)"
wait "$PREPARE_JOB"
gates_eg2_image
gates_eg2_video

# ---------------------------------------------------------------------------------------------------------------------
# Short soaks (screening, not the one-hour G4 run), one at a time with nothing else running, each under its profile's
# contract environment: MALLOC_MMAP_THRESHOLD_=65536 where the profile's README states it as a Linux deployment
# requirement (GLiNER2; EmbeddingGemma 2 video on CUDA).
wait
# The soak precondition of the hosted job (THP madvise, not always) where the container may set it; the host's setting
# otherwise. Every soak report records transparent_hugepage.
echo madvise >/sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
log "soaks ($SOAK_DURATION each), transparent_hugepage: $(cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo n/a)"
mkdir -p "$OUT/soak"
soak() {  # soak <profile> <workload> [VAR=value...]
    local profile=$1 model=$2 rc=0
    shift 2
    local note
    note="scheduled CUDA qualification on RunPod, $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), commit ${SHA:0:12}, CUDA libraries from the pinned eg2 requirements-cuda wheels${*:+, $*}"
    env "$@" "${SOAK[@]}" "$model" --cuda --duration "$SOAK_DURATION" --out "$OUT/soak/$model" --note "$note" >"$OUT/soak/$model.log" 2>&1 || rc=$?
    local line
    line="$(grep -E '\] (PASS|FAIL)|^error:' "$OUT/soak/$model.log" | tail -1 | cut -c1-200)"
    if [[ $rc == 0 ]]; then record "$profile" "soak $SOAK_DURATION${*:+ $*}" passed; else record "$profile" "soak $SOAK_DURATION${*:+ $*}" "failed (exit $rc: ${line:-see soak/$model.log})"; fi
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
