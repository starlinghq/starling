#!/usr/bin/env bash
# Batching sweep (Phase 0 of the batching investigation; README "Batching sweep"): the `sweep` command of the e5,
# GLiNER2 and EmbeddingGemma 2 (text) harnesses on one provider, then on CPU an intra-op thread sweep and the library's
# default rows per run, then the batching configurations of the option contract (length-sorted chunking, a token
# budget, both, and cross-request micro-batching with single-item callers at concurrency 4 and 16 against plain
# concurrency), then one markdown summary (model-pipeline bench sweep).
#
#   ai-dotnet/benchmarks/run-batching-sweep.sh cpu|cuda <out dir> [--quick]
#
# cpu runs the CPU builds, cuda the .Cuda builds (CUDA natives, device 0 of CUDA_VISIBLE_DEVICES, TF32 off; the CUDA 13
# and cuDNN 9 libraries must be on LD_LIBRARY_PATH, as in run-g5-matched.sh). Every run is a fresh process; a run that
# fails (quality, assets, crash) is recorded in <out>/sweep-status.tsv and the others still run. Exit status 0 only if
# every run passed.
#
# Environment:
#   STARLING_MODELS      model cache (acquisition layout); nothing is downloaded
#   SWEEP_MODELS         models (default "e5 gliner2 eg2")
#   SWEEP_THREADS        intra_op_num_threads of the main sweep (default: cpu 6, cuda 4)
#   SWEEP_THREAD_LIST    CPU thread sweep values (default "1 2 4 6"; set empty to skip)
#   SWEEP_MAX_INFLIGHT   --max-inflight-items on CPU (default 256: batch x concurrency above it is skipped and listed)
#   SWEEP_MAX_INFLIGHT_GLINER2  the same for GLiNER2 on CPU (default 32: long b32 c1 peaked at 10.2 GB RSS and long
#                        b32 c4 was killed for memory on a 16 GB budget)
#   SWEEP_CONFIGS        batching configurations after the baseline (default "sorted tokens sorted-tokens micro";
#                        --quick: "sorted tokens micro"; set empty to skip): sorted = --sort-by-length true with
#                        8 rows per ORT run (calls of 16 to 64 items split into runs, as the library default does)
#                        next to the same grid unsorted, tokens = --max-batch-tokens SWEEP_BATCH_TOKENS with one
#                        run per call otherwise (compared with the baseline grid), sorted-tokens = both, micro =
#                        single-item callers through the library's
#                        micro-batcher (MaxWait 0 ms, MaxRows 8; the full grid adds MaxWait 2 ms, MaxRows 16) next to a
#                        baseline run of single-item callers at the same concurrency
#   SWEEP_BATCH_TOKENS   padded-cell budget of the tokens configurations (default 4096)
#   SWEEP_ARGS           extra sweep arguments for every run (development smokes, for example "--pool 16")
#   SWEEP_NOTE           machine note recorded in every run
#   SWEEP_BUILD          1 (default) builds the harnesses first (Release); 0 uses the existing binaries
#   MODEL_PIPELINE       the model-pipeline command (default: model-pipeline on PATH)
set -uo pipefail

provider="${1:?cpu|cuda}"
out="${2:?out dir}"
quick="${3:-}"
case "$provider" in cpu | cuda) ;; *) echo "provider is cpu or cuda" >&2; exit 2 ;; esac
[[ -z $quick || $quick == --quick ]] || { echo "third argument is --quick or nothing" >&2; exit 2; }
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${STARLING_MODELS:?set STARLING_MODELS to the model cache}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"
models="${SWEEP_MODELS:-e5 gliner2 eg2}"
note="${SWEEP_NOTE:-}"
mp="${MODEL_PIPELINE:-model-pipeline}"
read -r -a extra <<<"${SWEEP_ARGS:-}"
suffix=""
threads="${SWEEP_THREADS:-6}"
if [[ $provider == cuda ]]; then
    : "${CUDA_VISIBLE_DEVICES:?cuda runs need an explicit CUDA_VISIBLE_DEVICES (and CUDA_DEVICE_ORDER=PCI_BUS_ID)}"
    suffix=".Cuda"
    threads="${SWEEP_THREADS:-4}"
fi
thread_list="${SWEEP_THREAD_LIST-1 2 4 6}"
inflight="${SWEEP_MAX_INFLIGHT:-256}"
grid=()
thread_grid=(--sets mixed --batches "1,8,32" --concurrency "1,4,16" --preliminary false)
default_grid=(--sets "mixed,long" --batches "8,16,32,64" --concurrency 1 --preliminary false --max-batch-size default)
configs="${SWEEP_CONFIGS-sorted tokens sorted-tokens micro}"
batch_tokens="${SWEEP_BATCH_TOKENS:-4096}"
# Explicit-batch configurations: mixed and long lengths, where sorting and a budget change the padding. Sorting only
# matters when a call is split into several ORT runs, so the sorted grid caps runs at 8 rows (its unsorted twin runs
# first); the token budget itself decides the split.
config_grid=(--sets "mixed,long" --batches "8,16,32,64" --concurrency "1,4,16" --preliminary false)
sorted_grid=(--sets "mixed,long" --batches "16,32,64" --concurrency "1,4,16" --preliminary false --max-batch-size 8)
# Micro-batching: single-item callers (batch 1); the baseline run of the same grid is plain concurrency.
micro_grid=(--sets "short,mixed,long" --batches 1 --concurrency "1,4,16" --preliminary false)
micro_variants=("w0-r8:--microbatch-wait-ms 0 --microbatch-max-rows 8" "w2-r16:--microbatch-wait-ms 2 --microbatch-max-rows 16")
if [[ $quick == --quick ]]; then
    grid=(--quick true)
    thread_grid=(--sets mixed --batches "1,8" --concurrency "1,4" --quick true --preliminary false)
    default_grid=(--sets mixed --batches 32 --concurrency 1 --quick true --preliminary false --max-batch-size default)
    configs="${SWEEP_CONFIGS-sorted tokens micro}"
    config_grid=(--sets mixed --batches "8,32" --concurrency "1,4" --quick true --preliminary false)
    sorted_grid=(--sets mixed --batches 32 --concurrency "1,4" --quick true --preliminary false --max-batch-size 8)
    micro_grid=(--sets "short,mixed" --batches 1 --concurrency "4,16" --quick true --preliminary false)
    micro_variants=("w0-r8:--microbatch-wait-ms 0 --microbatch-max-rows 8")
fi

project() {
    case "$1" in
        e5) echo Starlings.Benchmarks.E5Harness ;;
        gliner2) echo Starlings.Benchmarks.Gliner2Harness ;;
        eg2) echo Starlings.Benchmarks.EmbeddingGemma2Harness ;;
        *) return 1 ;;
    esac
}

status=0
: >"$out/sweep-status.tsv"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$out/sweep.log" >&2; }
record() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >>"$out/sweep-status.tsv"; [[ $3 == passed ]] || status=1; }

for m in $models; do
    p="$(project "$m")" || { log "unknown model $m (e5, gliner2, eg2)"; exit 2; }
    if [[ ${SWEEP_BUILD:-1} == 1 ]]; then
        if ! dotnet build "$here/$p$suffix" -c Release -v q -nologo >"$out/build-$m-$provider.log" 2>&1; then
            record "$m/$provider" build "failed (build-$m-$provider.log)"
            continue
        fi
    fi
done

# sweep <model> <label> <threads> <args...>: one process; its records go to sweep-<model>-<provider>-<label>.jsonl.
sweep() {
    local m=$1 label=$2 t=$3 p dll rc=0 s=$SECONDS
    shift 3
    p="$(project "$m")"
    dll="$here/$p$suffix/bin/Release/net10.0/$p$suffix.dll"
    local file="$out/sweep-$m-$provider-$label.jsonl"
    log "$m $provider $label: threads $t $*"
    dotnet "$dll" sweep --provider "$provider" --threads "$t" --out "$file" --note "$note" "$@" "${extra[@]}" \
        >"$out/sweep-$m-$provider-$label.log" 2>&1 || rc=$?
    case $rc in
        0) record "$m/$provider" "$label" passed ;;
        1) record "$m/$provider" "$label" "failed (quality; see sweep-$m-$provider-$label.log)" ;;
        3) record "$m/$provider" "$label" "not-run (assets missing; see sweep-$m-$provider-$label.log)" ;;
        *) record "$m/$provider" "$label" "failed (exit $rc; see sweep-$m-$provider-$label.log)" ;;
    esac
    log "$m $provider $label: exit $rc, $((SECONDS - s)) s"
}

# configs <model> <threads> [guard args...]: the batching configurations of SWEEP_CONFIGS.
run_configs() {
    local m=$1 t=$2 c v name
    shift 2
    for c in $configs; do
        case $c in
            sorted)
                sweep "$m" "mbs8-t$t" "$t" "${sorted_grid[@]}" "$@"
                sweep "$m" "sorted-mbs8-t$t" "$t" "${sorted_grid[@]}" --sort-by-length true "$@"
                ;;
            tokens) sweep "$m" "tokens$batch_tokens-t$t" "$t" "${config_grid[@]}" --max-batch-tokens "$batch_tokens" "$@" ;;
            sorted-tokens)
                sweep "$m" "sorted-tokens$batch_tokens-t$t" "$t" "${config_grid[@]}" --sort-by-length true \
                    --max-batch-tokens "$batch_tokens" "$@"
                ;;
            micro)
                sweep "$m" "b1-baseline-t$t" "$t" "${micro_grid[@]}" "$@"
                for v in "${micro_variants[@]}"; do
                    name=${v%%:*}
                    read -r -a margs <<<"${v#*:}"
                    sweep "$m" "micro-$name-t$t" "$t" "${micro_grid[@]}" "${margs[@]}" "$@"
                done
                ;;
            *) log "unknown batching configuration $c (sorted, tokens, sorted-tokens, micro)"; status=1 ;;
        esac
    done
}

for m in $models; do
    grep -q "^$m/$provider	build	" "$out/sweep-status.tsv" && continue
    inflight="${SWEEP_MAX_INFLIGHT:-256}"
    [[ $m == gliner2 ]] && inflight="${SWEEP_MAX_INFLIGHT_GLINER2:-32}"
    if [[ $provider == cpu ]]; then
        sweep "$m" "t$threads" "$threads" "${grid[@]}" --max-inflight-items "$inflight"
        for t in $thread_list; do
            [[ $t == "$threads" ]] && continue
            sweep "$m" "threads-t$t" "$t" "${thread_grid[@]}" --max-inflight-items "$inflight"
        done
        sweep "$m" "default-mbs-t$threads" "$threads" "${default_grid[@]}"
        run_configs "$m" "$threads" --max-inflight-items "$inflight"
    else
        sweep "$m" "t$threads" "$threads" "${grid[@]}"
        sweep "$m" "default-mbs-t$threads" "$threads" "${default_grid[@]}"
        run_configs "$m" "$threads"
    fi
done

shopt -s nullglob
files=("$out"/sweep-*-"$provider"-*.jsonl)
if ((${#files[@]})); then
    if "$mp" bench sweep "${files[@]}" --out "$out/summary-$provider.md" --json "$out/summary-$provider.json" \
        >"$out/summarize-$provider.log" 2>&1; then
        record "all/$provider" summary passed
    else
        record "all/$provider" summary "failed (summarize-$provider.log)"
    fi
fi
log "done: exit $status ($out/sweep-status.tsv)"
exit "$status"
