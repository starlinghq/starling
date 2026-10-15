#!/usr/bin/env bash
# G1, G2 and G3 of one CPU profile on its accepted P-030 release sets (regression support for Stable profiles,
# docs/support-policy.md; run weekly by .github/workflows/qualification.yml, job `release-gates`).
#
#   tools/release-gates.sh <profile> --out <dir> --work <dir> --reference-python <python> [--threads <n>]
#
# <profile>            multilingual-e5-small/onnx-fp32-cpu, embeddinggemma-2/onnx-fp32-cpu or gliner2-base-v1/onnx-fp32-cpu
# --out                reports (each `model-pipeline evaluate` report.json and report.md) and a summary.md
# --work               candidate outputs and regenerated references; must be outside the repository (the evaluator
#                      refuses an outside-Git reference inside it)
# --reference-python   python of the profile's reference environment (hash-locked requirements, with model-pipeline)
# --threads            ONNX Runtime intra-op threads of the producers (default 4)
#
# The commands are the ones each evidence README records (docs/evidence/release-set-gates-2026-10-14/<profile>/):
# the C# producers (Starlings.Fidelity.Tool), the pinned Python stage 2, and `model-pipeline evaluate` with the release
# set addressed explicitly. Bounds and budgets come only from the profile's protected files; the evaluator verifies every
# protected file against Git (--protected-base auto). The models must already be in $STARLING_MODELS (verified by the
# acquisition tool). Exit 0 only if every gate of every evaluation passed; 1 otherwise, after all evaluations ran.
# Nothing here writes to model-pipeline/protected/ or changes a manifest.
set -euo pipefail

usage() {
    sed -n '2,20p' "$0" >&2
    exit 2
}

[[ $# -ge 1 ]] || usage
profile="$1"
shift
out=""
work=""
refpy=""
threads=4
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) out="$2"; shift 2 ;;
        --work) work="$2"; shift 2 ;;
        --reference-python) refpy="$2"; shift 2 ;;
        --threads) threads="$2"; shift 2 ;;
        *) usage ;;
    esac
done
[[ -n "$out" && -n "$work" && -n "$refpy" ]] || usage
: "${STARLING_MODELS:?STARLING_MODELS must name the model cache}"

repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"
mkdir -p "$out" "$work"
out="$(cd "$out" && pwd)"
work="$(cd "$work" && pwd)"
case "$work/" in
    "$repo"/*) echo "error: --work $work is inside the repository" >&2; exit 2 ;;
esac

P=model-pipeline/protected
MAN="model-pipeline/registry/$profile/manifest.yaml"
[[ -f "$MAN" ]] || { echo "error: no manifest $MAN" >&2; exit 2; }

# The snapshot directory of the profile's lock in the acquisition cache.
python3 -I tools/qualification.py lock "$MAN" "$work/snapshot.lock.json" >/dev/null
SNAP="$STARLING_MODELS/$(python3 -I tools/qualification.py snapshot "$work/snapshot.lock.json")"
[[ -d "$SNAP" ]] || { echo "error: snapshot $SNAP not acquired" >&2; exit 2; }

dotnet build ai-dotnet/tools/Starlings.Fidelity.Tool --configuration Release --nologo -v quiet
TOOL=(dotnet ai-dotnet/tools/Starlings.Fidelity.Tool/bin/Release/net10.0/Starlings.Fidelity.Tool.dll)

results=()
failed=0

# evaluate <name> <gates> <model-pipeline evaluate args...>: runs the evaluator into $out/<name>, then requires every
# named gate to be `passed` in its report.json (the exit code alone does not say which gate a report covers).
evaluate() {
    local name="$1" gates="$2"
    shift 2
    local status=0
    model-pipeline evaluate "$MAN" "$@" --out "$out/$name" || status=$?
    local verdict
    verdict="$(python3 -I - "$out/$name/report.json" "$gates" <<'PY'
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
sys.exit(1 if bad else 0)
PY
)" || true
    echo "$name: evaluator exit $status; $gates: $verdict"
    results+=("| $name | $gates | $verdict | $status |")
    if [[ "$verdict" != passed ]]; then
        failed=1
    fi
}

# step <log name> <command...>: runs a producer with its output in $work/<log>.log; a failure is recorded, and the
# evaluation that needs it then fails on missing results.
step() {
    local name="$1"
    shift
    local s rc=0
    s=$(date +%s)
    "$@" >"$work/$name.log" 2>&1 || rc=$?
    echo "$name: exit $rc, $(( $(date +%s) - s )) s"
    if [[ $rc -ne 0 ]]; then
        tail -n 40 "$work/$name.log" >&2
        results+=("| $name | (producer) | exit $rc | - |")
        failed=1
    fi
    cp "$work/$name.log" "$out/" 2>/dev/null || true
}

case "$profile" in
    multilingual-e5-small/onnx-fp32-cpu)
        REL=$P/multilingual-e5-small/release/onnx-fp32-release
        DS=$P/release-shared/retrieval/miracl-dev-hn-8lang-v1
        step g1-holdout-producer "${TOOL[@]}" tokenizer --model e5 --tokenizer "$SNAP/tokenizer.json" \
            --fixtures "$REL/tokenizer.jsonl" --out "$work/g1-holdout"
        evaluate g1-tokenizer-holdout G1 --tokenizer-fixtures "$REL/tokenizer.jsonl" --results "$work/g1-holdout"
        step dotnet-stages "${TOOL[@]}" e5 --out "$work/stages" --fixtures "$REL" --snapshot "$SNAP" --hidden --threads "$threads"
        step python-stage2 "$refpy" model-pipeline/reference/encoder/stage_e5_export.py --model-dir "$SNAP" \
            --fixtures "$REL" --profile "$profile" --out "$work/stages" --intra-op-threads "$threads"
        evaluate g1-g2-staged G1,G2 --fixtures "$REL" --results "$work/stages" --require-gates G1,G2
        step g3-producer "${TOOL[@]}" retrieval --model e5 --dataset "$DS" --out "$work/g3/embeddings.jsonl" \
            --snapshot "$SNAP" --threads "$threads"
        evaluate g3-retrieval G3 --g3 retrieval --dataset "$DS" --results "$work/g3/embeddings.jsonl"
        ;;
    embeddinggemma-2/onnx-fp32-cpu)
        REL=$P/embeddinggemma-2/release/onnx-fp32-release
        DS=$P/release-shared/retrieval/miracl-dev-hn-8lang-v1
        # The release set's embeddings.jsonl is kept outside Git, pinned by the digest in its protected reference.json
        # (#43). Regenerate it as SOURCE.md states, at the recorded torch_threads (6), and let the evaluator bind it
        # by digest (--reference-embeddings, #45). The reference was produced on an AVX2 CPU; the ISA is held there.
        step reference-regeneration env OMP_NUM_THREADS=6 ATEN_CPU_CAPABILITY=avx2 ONEDNN_MAX_CPU_ISA=AVX2 \
            MKL_ENABLE_INSTRUCTIONS=AVX2 MKL_CBWR=AVX2 PYTHONPATH=model-pipeline/reference/embeddinggemma2 \
            "$refpy" model-pipeline/reference/embeddinggemma2/run_embeddinggemma2.py --model-dir "$SNAP" \
            --revision 914f7f89142e33e77833254d9c9b90c3cef7303b \
            --corpus "$P/embeddinggemma-2/release/tokenizer-holdout/corpus.jsonl" --out "$work/reference" \
            --batch-ids massive-km-KH-9760,massive-ja-JP-2798,massive-ar-SA-15109,massive-th-TH-13186,humaneval-120,oasst1-06f21312-d9c1-4970-9d7a-b129e8069aba
        python3 -I - "$work/reference" "$REL" <<'PY' | tee "$out/reference-digests.txt"
import hashlib, json, pathlib, sys
ref, rel = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "missing"
want = json.loads((rel / "reference.json").read_text(encoding="utf-8")).get("embeddings", {}).get("sha256")
got = sha(ref / "embeddings.jsonl")
print(f"embeddings.jsonl {got} ({'equals' if got == want else 'differs from'} reference.json embeddings.sha256 {want})")
for name in ("tokenizer.jsonl", "tokenizer-batch.json", "tokenizer-implementations.json"):
    a, b = sha(ref / name), sha(rel / name)
    print(f"{name} {a} ({'byte-identical to' if a == b else 'differs from'} the protected file {b})")
PY
        # The producers read the case ids from embeddings.jsonl: a copy of the release set plus the regenerated file,
        # outside the repository. The evaluation itself reads the real protected set.
        mkdir -p "$work/fixtures"
        cp "$REL"/* "$work/fixtures/"
        cp "$work/reference/embeddings.jsonl" "$work/fixtures/" 2>/dev/null || true
        step g1-holdout-producer "${TOOL[@]}" tokenizer --model embeddinggemma2 --tokenizer "$SNAP/tokenizer.json" \
            --fixtures "$REL/tokenizer.jsonl" --out "$work/g1-holdout"
        evaluate g1-tokenizer-holdout G1 --tokenizer-fixtures "$REL/tokenizer.jsonl" --results "$work/g1-holdout"
        step dotnet-stages "${TOOL[@]}" embeddinggemma2 --out "$work/stages" --fixtures "$work/fixtures" \
            --snapshot "$SNAP" --threads "$threads"
        step python-stage2 env PYTHONPATH=model-pipeline/reference/embeddinggemma2 "$refpy" \
            model-pipeline/reference/embeddinggemma2/stage_embeddinggemma2_export.py --model-dir "$SNAP" \
            --graph "$SNAP/onnx/embeddinggemma2-text-fp32.onnx" --fixtures "$work/fixtures" --profile "$profile" \
            --out "$work/stages" --intra-op-threads "$threads"
        evaluate g2-release G1,G2 --fixtures "$REL" --reference-embeddings "$work/reference/embeddings.jsonl" \
            --results "$work/stages" --require-gates G1,G2
        step g3-producer "${TOOL[@]}" retrieval --model embeddinggemma2 --dataset "$DS" \
            --out "$work/g3/embeddings.jsonl" --snapshot "$SNAP" --threads "$threads"
        evaluate g3-retrieval G3 --g3 retrieval --dataset "$DS" --results "$work/g3/embeddings.jsonl"
        ;;
    gliner2-base-v1/onnx-fp32-cpu)
        TOK=$P/gliner2-base-v1/release/tokenizer-release/gliner2-tokenizer.jsonl
        REL=$P/gliner2-base-v1/release/extraction-release
        DS=$P/gliner2-base-v1/release/ner/wnut17-test-v1/gold.jsonl
        step g1-tokenizer-producer "${TOOL[@]}" tokenizer --model gliner2 --tokenizer "$SNAP/tokenizer.json" \
            --fixtures "$TOK" --out "$work/g1-tokenizer"
        evaluate g1-tokenizer-release G1 --tokenizer-fixtures "$TOK" --results "$work/g1-tokenizer"
        step dotnet-stages "${TOOL[@]}" gliner2 --out "$work/stages" --fixtures "$REL" --snapshot "$SNAP" --threads "$threads"
        step python-stage2 env OMP_NUM_THREADS="$threads" "$refpy" model-pipeline/reference/gliner2/stage_gliner2.py export \
            --model-dir "$SNAP" --graph "$SNAP/onnx/gliner2-entities-fp32.onnx" \
            --graph-asset-prefix onnx/gliner2-entities-fp32 --fixtures "$REL" --profile "$profile" \
            --out "$work/stages" --threads "$threads"
        evaluate g1-g2-extraction-release G1,G2 --fixtures "$REL" --results "$work/stages" --require-gates G1,G2
        step g3-producer "${TOOL[@]}" ner --dataset "$DS" --out "$work/g3/predictions.jsonl" --snapshot "$SNAP" \
            --threads "$threads"
        evaluate g3-ner G3 --g3 ner --dataset "$DS" --results "$work/g3/predictions.jsonl"
        ;;
    *)
        echo "error: no release-set recipe for $profile" >&2
        exit 2
        ;;
esac

{
    echo "## Release-set gates: \`$profile\`"
    echo
    echo "| Evaluation | Gates | Result | Evaluator exit |"
    echo "| --- | --- | --- | --- |"
    printf '%s\n' "${results[@]}"
    if [[ -f "$out/reference-digests.txt" ]]; then
        echo
        echo '```'
        cat "$out/reference-digests.txt"
        echo '```'
    fi
} >"$out/summary.md"
cat "$out/summary.md"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    cat "$out/summary.md" >>"$GITHUB_STEP_SUMMARY"
fi
exit "$failed"
