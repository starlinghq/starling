#!/usr/bin/env bash
# Native AOT check for the Starlings packages (plan 10 G6; production-readiness R9).
#
# For each target (one package and its dependencies) it builds tools/release-drill/aot-consumer outside the repository
# against the packed packages, twice from the same project:
#   1. a JIT build (dotnet build), run on the cached model files: the reference results,
#   2. dotnet publish -p:PublishAot=true -r <rid>: every trim/AOT warning is recorded (one per call site), then the native
#      binary runs the same inference with no network,
# and compares the two result files: strings (token ids, entities, generated text, shapes) exactly, float vectors within
# --float-bound (default 1e-5; this check's own bound, not a protected tolerance: AOT and JIT may pick different SIMD
# widths for the managed pooling and normalization).
#
#   tools/aot-check-ai.sh --models <cache> [--feed <dir>] [--targets a,b] [--download] [--out <dir>]
#                         [--run-unsupported] [--cuda]
#
#   --models <cache>  model cache in the Starlings layout (<repo>/<commit>/<files>); read only. Targets whose files
#                     are missing are skipped with a note.
#   --feed <dir>      local folder feed holding the packed Starlings.* packages (the release drill passes its feed).
#                     Without it the script packs the committed HEAD from a clean clone first.
#   --targets a,b     subset of: acquisition runtime e5 e5-meai e5-mlnet gliner2 eg2 eg2-meai eg2-mlnet qwen35 qwen35-meai
#   --download        the acquisition target also downloads e5's tokenizer.json from the public Hub (no token) into an
#                     empty directory, under JIT and under AOT.
#   --out <dir>       where the per-target logs, warnings and results go (default <repo>/artifacts/aot-check-ai).
#   --cuda            build the consumers against the CUDA natives (Microsoft.ML.OnnxRuntime.Gpu.Linux 1.30.0;
#                     Microsoft.ML.OnnxRuntimeGenAI.Cuda 0.17.1) and run every session on CUDA device 0 (TF32 off).
#                     Both the JIT and the AOT run use the GPU, so JIT vs AOT compares like with like. Needs the CUDA 13
#                     runtime and cuDNN 9 on LD_LIBRARY_PATH and an explicit CUDA_VISIBLE_DEVICES; the qwen35 targets
#                     then also run the fp16x variant. Default targets: those with a CUDA profile (no acquisition,
#                     runtime or ML.NET targets).
#   --run-unsupported also run the native binaries of the targets recorded as AOT unsupported, to revisit their
#                     status. Off by default: those binaries abort (SIGABRT) by design, which leaves a core dump and
#                     a crash notification on every run. When on, the run has core dumps disabled (ulimit -c 0).
#
# Exit code 0 when every target that ran published, ran under AOT and matched the JIT results, except the targets whose
# AOT status is unsupported (the ML.NET adapters: Microsoft.ML builds its row getters with Reflection.Emit), which are
# published (their warnings recorded) but not run unless --run-unsupported is given. Warnings do not fail the check: they are the evidence (summary.md lists them).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
all_targets=(acquisition runtime e5 e5-meai e5-mlnet gliner2 eg2 eg2-meai eg2-mlnet qwen35 qwen35-meai)
models=""
feed=""
targets=("${all_targets[@]}")
download=0
run_unsupported=0
provider=cpu
targets_given=0
out="$root/artifacts/aot-check-ai"
float_bound="1e-5"
# Targets whose AOT status is unsupported, with the reason recorded in docs/evidence/aot/README.md: a failure there is
# the expected result and does not fail the check; a pass is reported so the status can be revisited.
expected_unsupported=" e5-mlnet eg2-mlnet "
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models) models="$(cd "$2" && pwd)"; shift 2 ;;
        --feed) feed="$(cd "$2" && pwd)"; shift 2 ;;
        --targets) IFS=, read -r -a targets <<<"$2"; targets_given=1; shift 2 ;;
        --cuda) provider=cuda; shift ;;
        --download) download=1; shift ;;
        --run-unsupported) run_unsupported=1; shift ;;
        --out) mkdir -p "$2"; out="$(cd "$2" && pwd)"; shift 2 ;;
        --float-bound) float_bound="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$models" ]] || { echo "--models <cache> is required" >&2; exit 2; }
if [[ "$provider" == cuda ]]; then
    [[ "$targets_given" == 1 ]] || targets=(e5 e5-meai gliner2 eg2 eg2-meai qwen35 qwen35-meai)
    [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { echo "--cuda needs an explicit CUDA_VISIBLE_DEVICES" >&2; exit 2; }
    [[ "$out" != "$root/artifacts/aot-check-ai" ]] || out="$root/artifacts/aot-check-ai-cuda"
fi

export DOTNET_NOLOGO=1 DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1
unset STARLING_MODELS HF_TOKEN

work="$(mktemp -d "${TMPDIR:-/tmp}/starlings-aot.XXXXXX")"
cleanup() {
    if [[ "${AOT_KEEP:-0}" == "1" ]]; then echo "kept: $work"; else rm -rf -- "$work"; fi
}
trap cleanup EXIT
step() { printf '\n== %s\n' "$*"; }
fail() { echo "AOT CHECK FAIL: $*" >&2; exit 1; }

if unshare -rn true 2>/dev/null; then
    offline() { unshare -rn "$@"; }
    offline_how="network namespace without interfaces (unshare -rn)"
else
    offline() { HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9 NO_PROXY= "$@"; }
    offline_how="proxy variables pointing at a closed port (weaker)"
fi

rid="$(dotnet --info | sed -n 's/^ *RID: *//p' | head -1)"
step "0. Inputs"
echo "commit:  $(git -C "$root" rev-parse HEAD)"
echo "dotnet:  $(cd "$work" && dotnet --version), RID $rid"
echo "models:  $models"
echo "offline: $offline_how"
echo "linker:  $(command -v clang || command -v gcc || echo none)"
echo "provider: $provider${CUDA_VISIBLE_DEVICES:+ (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)}"

if [[ -z "$feed" ]]; then
    step "1. Pack the committed HEAD from a clean clone"
    src="$work/src"
    git clone --quiet --no-hardlinks "$root" "$src"
    git -C "$src" -c advice.detachedHead=false checkout --quiet --detach "$(git -C "$root" rev-parse HEAD)"
    export NUGET_PACKAGES="$work/nuget-build"
    feed="$work/feed"
    mkdir -p "$feed"
    for proj in Acquisition Runtime.Onnx Models.E5 Models.E5.Extensions.AI Models.E5.MLNet Models.Gliner2 \
        Models.EmbeddingGemma2 Models.EmbeddingGemma2.Extensions.AI Models.EmbeddingGemma2.MLNet Generation.Abstractions Generation.OnnxGenAI \
        Models.Qwen35 Models.Qwen35.OnnxGenAI Models.Qwen35.Extensions.AI; do
        (cd "$src" && CI=true dotnet pack "ai-dotnet/src/Starlings.$proj" -c Release -o "$feed" >"$work/pack-$proj.log" 2>&1) \
            || { cat "$work/pack-$proj.log"; fail "pack failed: $proj"; }
    done
    consumer_src="$src/tools/release-drill/aot-consumer"
else
    step "1. Packages from $feed"
    consumer_src="$root/tools/release-drill/aot-consumer"
fi
version="$(find "$feed" -name 'Starlings.Models.E5.*.nupkg' ! -name '*.Extensions.*' ! -name '*.MLNet.*' | sed 's:.*/Starlings\.Models\.E5\.\(.*\)\.nupkg:\1:' | head -1)"
[[ -n "$version" ]] || fail "no Starlings.Models.E5 package in $feed"
echo "version: $version"

e5_dir="intfloat/multilingual-e5-small/614241f622f53c4eeff9890bdc4f31cfecc418b3"
needs() {
    case "$1" in
        acquisition|runtime|e5|e5-meai|e5-mlnet) echo "$e5_dir/onnx/model.onnx $e5_dir/tokenizer.json" ;;
        gliner2) echo "fastino/gliner2-base-v1/f9634218e53580c56edf0de97ca1a7d3f1c2354e/onnx/gliner2-entities-fp32.onnx" ;;
        eg2) echo "google/embeddinggemma-2/914f7f89142e33e77833254d9c9b90c3cef7303b/onnx/embeddinggemma2-text-fp32.onnx google/embeddinggemma-2/914f7f89142e33e77833254d9c9b90c3cef7303b/onnx/embeddinggemma2-vision-fp32.onnx google/embeddinggemma-2/914f7f89142e33e77833254d9c9b90c3cef7303b/onnx/embeddinggemma2-text-mm-fp32.onnx" ;;
        eg2-meai|eg2-mlnet) echo "google/embeddinggemma-2/914f7f89142e33e77833254d9c9b90c3cef7303b/onnx/embeddinggemma2-text-fp32.onnx" ;;
        qwen35|qwen35-meai)
            echo "Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17/ortgenai-fp32x/model.onnx.data"
            if [[ "$provider" == cuda ]]; then echo "Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17/ortgenai-fp16x/model.onnx.data"; fi ;;
        *) return 1 ;;
    esac
}

apps="$work/apps"
mkdir -p "$apps" "$out"
cat >"$apps/NuGet.config" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="aot-local" value="$feed" />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
  </packageSources>
  <fallbackPackageFolders>
    <clear />
  </fallbackPackageFolders>
  <packageSourceMapping>
    <packageSource key="aot-local">
      <package pattern="Starlings.*" />
    </packageSource>
    <packageSource key="nuget.org">
      <package pattern="Starlings.Tokenizers" />
      <package pattern="*" />
    </packageSource>
  </packageSourceMapping>
</configuration>
EOF
export NUGET_PACKAGES="$work/nuget-consumer" NUGET_HTTP_CACHE_PATH="$work/nuget-http-consumer"

expected() {
    if [[ "$expected_unsupported" == *" $1 "* ]]; then echo "expected: AOT unsupported"; fi
}

summary="$out/summary.md"
{
    echo "| Target | Packages | Starlings.* warnings | Other warnings | AOT publish | AOT run | JIT vs AOT |"
    echo "| --- | --- | --- | --- | --- | --- | --- |"
} >"$summary"
failed=0
for t in "${targets[@]}"; do
    files="$(needs "$t")" || fail "unknown target $t"
    missing=""
    for f in $files; do [[ -f "$models/$f" ]] || missing="$missing $f"; done
    step "AOT target $t"
    if [[ -n "$missing" ]]; then
        echo "skipped: model files missing from $models:$missing"
        echo "| $t | | | | | skipped (model files missing) | |" >>"$summary"
        continue
    fi
    app="$apps/$t"
    mkdir -p "$app"
    cp "$consumer_src/Program.cs" "$app/"
    sed -e "s/@VERSION@/$version/g" -e "s/@TARGET@/$t/g" -e "s/@PROVIDER@/$provider/g" "$consumer_src/AotConsumer.csproj.in" >"$app/AotConsumer.csproj"
    log="$out/$t"
    mkdir -p "$log"
    (cd "$app" && dotnet restore >"$log/restore.log" 2>&1) || { cat "$log/restore.log"; fail "$t: restore failed"; }
    starlings="$(cd "$app" && dotnet list package --include-transitive | sed -n 's/^ *[>]* *\(Starlings\.[^ ]*\) .*/\1/p' | sort -u | tr '\n' ' ')"
    echo "packages: $starlings"
    (cd "$app" && dotnet build -c Release --no-restore >"$log/build-jit.log" 2>&1) || { cat "$log/build-jit.log"; fail "$t: JIT build failed"; }
    dl_jit=() dl_aot=()
    if [[ "$t" == acquisition && "$download" == 1 ]]; then
        dl_jit=("$work/download-jit-$t") dl_aot=("$work/download-aot-$t")
        mkdir -p "${dl_jit[0]}" "${dl_aot[0]}"
    fi
    run_jit=(dotnet "$app/bin/Release/net10.0/AotConsumer.dll" "$models" "$log/results-jit.txt" "${dl_jit[@]}")
    if [[ ${#dl_jit[@]} -eq 0 ]]; then run_jit=(offline "${run_jit[@]}"); fi
    "${run_jit[@]}" >"$log/run-jit.log" 2>&1 || { cat "$log/run-jit.log"; fail "$t: JIT run failed"; }
    publish_ok=yes
    (cd "$app" && dotnet publish -c Release -r "$rid" -p:PublishAot=true -o "$app/publish" >"$log/publish-aot.log" 2>&1) || publish_ok=no
    # One line per distinct warning (ILC prints each once; MSBuild repeats them in the summary).
    grep -Eo 'warning (IL|AOT)[0-9]+:.*' "$log/publish-aot.log" | sed 's/ \[[^]]*\.csproj[^]]*\]$//' | sort -u >"$log/warnings.txt" || true
    own="$(grep -c 'Starlings\.' "$log/warnings.txt" || true)"
    other="$(( $(wc -l <"$log/warnings.txt") - own ))"
    echo "publish: $publish_ok; warnings: $own in Starlings.* code, $other elsewhere (details: $log/warnings.txt)"
    if [[ "$publish_ok" == no ]]; then
        tail -30 "$log/publish-aot.log"
        echo "| $t | $starlings | $own | $other | **failed** | not run | $(expected "$t") |" >>"$summary"
        [[ "$expected_unsupported" == *" $t "* ]] || failed=1
        continue
    fi
    ls -la "$app/publish" >"$log/publish-files.txt"
    size="$(du -h "$app/publish/AotConsumer" | cut -f1)"
    if [[ "$expected_unsupported" == *" $t "* && "$run_unsupported" != 1 ]]; then
        # Known to abort under AOT (SIGABRT): do not run it, so no core dump or crash notification is produced.
        echo "not run: $t is recorded as AOT unsupported (pass --run-unsupported to revisit)"
        echo "| $t | $starlings | $own | $other | ok ($size) | not run | $(expected "$t") |" >>"$summary"
        continue
    fi
    run_aot=("$app/publish/AotConsumer" "$models" "$log/results-aot.txt" "${dl_aot[@]}")
    if [[ ${#dl_aot[@]} -eq 0 ]]; then run_aot=(offline "${run_aot[@]}"); fi
    if [[ "$expected_unsupported" == *" $t "* ]]; then
        run_aot=(bash -c 'ulimit -c 0; exec "$@"' _ "${run_aot[@]}")
    fi
    if ! "${run_aot[@]}" >"$log/run-aot.log" 2>&1; then
        tail -30 "$log/run-aot.log"
        echo "| $t | $starlings | $own | $other | ok ($size) | **failed**: $(grep -m1 -E 'Exception|Unhandled' "$log/run-aot.log" | cut -c1-200 | tr '|' '/') | $(expected "$t") |" >>"$summary"
        [[ "$expected_unsupported" == *" $t "* ]] || failed=1
        continue
    fi
    grep -q '^mode:    Native AOT' "$log/run-aot.log" || fail "$t: the published binary did not report Native AOT"
    if verdict="$(python3 -I - "$log/results-jit.txt" "$log/results-aot.txt" "$float_bound" <<'PY'
import sys
def load(path):
    rows = {}
    for line in open(path, encoding="utf-8").read().split("\n"):
        if line:
            kind, key, value = line.split("\t", 2)
            rows[key] = (kind, value)
    return rows
jit, aot, bound = load(sys.argv[1]), load(sys.argv[2]), float(sys.argv[3])
problems, strings, vectors, worst, notes = [], 0, 0, 0.0, []
if jit.keys() != aot.keys():
    problems.append(f"keys differ: {sorted(jit.keys() ^ aot.keys())}")
for key in sorted(jit.keys() & aot.keys()):
    (kind, a), (_, b) = jit[key], aot[key]
    if kind == "inf":
        if a != b:
            notes.append(f"{key} differs (reported only): JIT {a!r}, AOT {b!r}")
    elif kind == "str":
        strings += 1
        if a != b:
            problems.append(f"{key}: {a[:80]!r} != {b[:80]!r}")
    else:
        vectors += 1
        x = [float(v) for v in a.split(",")] if a else []
        y = [float(v) for v in b.split(",")] if b else []
        if len(x) != len(y):
            problems.append(f"{key}: length {len(x)} != {len(y)}")
            continue
        d = max((abs(p - q) for p, q in zip(x, y)), default=0.0)
        worst = max(worst, d)
        if not d <= bound:
            problems.append(f"{key}: max abs {d:.3g} > {bound:g}")
print((f"{strings} strings exact, {vectors} vectors max abs {worst:.3g}" if not problems else "; ".join(problems))
      + "".join(f"; {n}" for n in notes))
sys.exit(1 if problems else 0)
PY
)"; then
        echo "ok:   AOT results equal JIT: $verdict"
        if [[ "$expected_unsupported" == *" $t "* ]]; then
            echo "note: $t is recorded as AOT unsupported but now passes: revisit its status"
            verdict="$verdict (recorded unsupported: revisit)"
        fi
        echo "| $t | $starlings | $own | $other | ok ($size) | ok | $verdict |" >>"$summary"
    else
        echo "FAIL: AOT results differ from JIT: $verdict"
        echo "| $t | $starlings | $own | $other | ok ($size) | ok | **differs**: $verdict |" >>"$summary"
        failed=1
    fi
done

step "Summary ($out/summary.md)"
cat "$summary"
[[ "$failed" == 0 ]] || fail "at least one target failed to publish, run or match (see above)"
echo "AOT CHECK PASS"
