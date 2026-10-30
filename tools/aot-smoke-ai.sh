#!/usr/bin/env bash
# Weight-free Native AOT smoke check of every published Starlings package (backlog E1; complements tools/aot-check-ai.sh).
#
# For each package it builds tools/release-drill/aot-smoke outside the repository against the packed package from a
# local folder feed, with the whole package assembly rooted, publishes it with -p:PublishAot=true -r <rid>, records
# every trim/AOT warning (one per call site), and runs the binary (no network, no model file: it only loads the
# package's metadata). It needs no model cache, so it can run wherever the packages can be packed.
# tools/aot-check-ai.sh is the inference check (JIT and AOT results compared on real model files); this one covers
# every package, including those that check does not.
#
#   tools/aot-smoke-ai.sh [--feed <dir>] [--packages a,b] [--out <dir>]
#
#   --feed <dir>      local folder feed holding the packed Starlings.* packages (the release drill passes its feed).
#                     Without it the script packs the committed HEAD from a clean clone first.
#   --packages a,b    package ids (default: every package below; ids missing from the feed are reported as such)
#   --out <dir>       where the per-package logs, warnings and summary.md go (default <repo>/artifacts/aot-smoke-ai).
#
# summary.md has one row per package: warnings in Starlings.* code, other warnings, publish and run results, and the
# verdict "AOT-clean" (published, ran, no warning anywhere), "clean in Starlings.*" (warnings only in third-party
# code) or "not clean". Exit code 0 when every package that is not recorded as AOT unsupported published and ran; a
# warning is evidence, not a failure. The browser packages (Starlings.Runtime.OnnxWeb, .WebGpu, Starlings.Models.E5.Browser)
# are not published here: they target browser-wasm, whose trimming and AOT run in browser/conformance/.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
all_packages=(
    Starlings.Acquisition Starlings.Runtime.Onnx Starlings.Models.Abstractions Starlings.Models.E5
    Starlings.Models.E5.Extensions.AI Starlings.Models.E5.MLNet Starlings.Models.Gliner2 Starlings.Models.EmbeddingGemma2
    Starlings.Models.EmbeddingGemma2.Extensions.AI Starlings.Models.EmbeddingGemma2.MLNet
    Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp Starlings.Models.EmbeddingGemma2.Video.FFmpeg
    Starlings.Models.EmbeddingGemma2.Video.FFmpeg9 Starlings.Generation.Abstractions Starlings.Generation.OnnxGenAI
    Starlings.Models.Qwen35 Starlings.Models.Qwen35.OnnxGenAI Starlings.Models.Qwen35.Extensions.AI Starlings.Models.Kev
    Starlings.Models.Kev.Onnx Starlings.Models.QwenChat Starlings.Models.QwenChat.OnnxGenAI Starlings.Models.Bert
    Starlings.Models.Whisper Starlings.Models.Lfm3
)
# AOT unsupported with the reason in docs/evidence/aot/README.md (Microsoft.ML builds its row getters with
# Reflection.Emit): a failure there is the recorded status and does not fail the check.
expected_unsupported=" Starlings.Models.E5.MLNet Starlings.Models.EmbeddingGemma2.MLNet "
feed=""
packages=("${all_packages[@]}")
out="$root/artifacts/aot-smoke-ai"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --feed) feed="$(cd "$2" && pwd)"; shift 2 ;;
        --packages) IFS=, read -r -a packages <<<"$2"; shift 2 ;;
        --out) mkdir -p "$2"; out="$(cd "$2" && pwd)"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

export DOTNET_NOLOGO=1 DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1
unset STARLING_MODELS HF_TOKEN

work="$(mktemp -d "${TMPDIR:-/tmp}/starlings-aot-smoke.XXXXXX")"
cleanup() {
    if [[ "${AOT_KEEP:-0}" == "1" ]]; then echo "kept: $work"; else rm -rf -- "$work"; fi
}
trap cleanup EXIT
step() { printf '\n== %s\n' "$*"; }
fail() { echo "AOT SMOKE FAIL: $*" >&2; exit 1; }

if unshare -rn true 2>/dev/null; then
    offline() { unshare -rn "$@"; }
    offline_how="network namespace without interfaces (unshare -rn)"
else
    offline() { HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9 NO_PROXY= "$@"; }
    offline_how="proxy variables pointing at a closed port (weaker)"
fi

rid="$(dotnet --info | sed -n 's/^ *RID: *//p' | head -1)"
step "0. Inputs"
echo "commit:   $(git -C "$root" rev-parse HEAD)"
echo "dotnet:   $(cd "$work" && dotnet --version), RID $rid"
echo "offline:  $offline_how"
echo "linker:   $(command -v clang || command -v gcc || echo none)"

if [[ -z "$feed" ]]; then
    step "1. Pack the committed HEAD from a clean clone"
    src="$work/src"
    git clone --quiet --no-hardlinks "$root" "$src"
    git -C "$src" -c advice.detachedHead=false checkout --quiet --detach "$(git -C "$root" rev-parse HEAD)"
    export NUGET_PACKAGES="$work/nuget-build"
    feed="$work/feed"
    mkdir -p "$feed"
    for id in "${packages[@]}"; do
        [[ -d "$src/ai-dotnet/src/$id" ]] || continue
        (cd "$src" && CI=true dotnet pack "ai-dotnet/src/$id" -c Release -o "$feed" >"$work/pack-$id.log" 2>&1) \
            || { cat "$work/pack-$id.log"; fail "pack failed: $id"; }
    done
    consumer_src="$src/tools/release-drill/aot-smoke"
else
    step "1. Packages from $feed"
    consumer_src="$root/tools/release-drill/aot-smoke"
fi
version="$(find "$feed" -name 'Starlings.Acquisition.*.nupkg' | sed 's:.*/Starlings\.Acquisition\.\(.*\)\.nupkg:\1:' | head -1)"
[[ -n "$version" ]] || fail "no Starlings.Acquisition package in $feed"
echo "version: $version"

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

summary="$out/summary.md"
{
    echo "Native AOT smoke check ($rid, Starlings $version, commit $(git -C "$root" rev-parse --short HEAD)); tools/aot-smoke-ai.sh."
    echo
    echo "| Package | Starlings.* warnings | Other warnings | AOT publish | AOT run | Verdict |"
    echo "| --- | --- | --- | --- | --- | --- |"
} >"$summary"
failed=0
for id in "${packages[@]}"; do
    step "AOT smoke $id"
    if [[ ! -f "$feed/$id.$version.nupkg" ]]; then
        echo "skipped: $id $version is not in the feed"
        echo "| $id | | | | | not packed |" >>"$summary"
        continue
    fi
    app="$apps/$id"
    log="$out/$id"
    mkdir -p "$app" "$log"
    cp "$consumer_src/Program.cs" "$app/"
    sed -e "s/@VERSION@/$version/g" -e "s/@PACKAGE@/$id/g" "$consumer_src/AotSmoke.csproj.in" >"$app/AotSmoke.csproj"
    (cd "$app" && dotnet restore -r "$rid" >"$log/restore.log" 2>&1) || { cat "$log/restore.log"; fail "$id: restore failed"; }
    publish_ok=yes
    (cd "$app" && dotnet publish -c Release -r "$rid" --no-restore -o "$app/publish" >"$log/publish-aot.log" 2>&1) || publish_ok=no
    # One line per distinct warning (ILC prints each once; MSBuild repeats them in the summary).
    grep -Eo 'warning (IL|AOT)[0-9]+:.*' "$log/publish-aot.log" | sed 's/ \[[^]]*\.csproj[^]]*\]$//' | sort -u >"$log/warnings.txt" || true
    own="$(grep -c 'Starlings\.' "$log/warnings.txt" || true)"
    other="$(( $(wc -l <"$log/warnings.txt") - own ))"
    echo "publish: $publish_ok; warnings: $own in Starlings.* code, $other elsewhere (details: $log/warnings.txt)"
    note=""
    [[ "$expected_unsupported" != *" $id "* ]] || note=" (recorded AOT unsupported)"
    if [[ "$publish_ok" == no ]]; then
        tail -30 "$log/publish-aot.log"
        echo "| $id | $own | $other | **failed** | not run | not clean$note |" >>"$summary"
        [[ -n "$note" ]] || failed=1
        continue
    fi
    run_ok=yes
    offline bash -c 'ulimit -c 0; exec "$@"' _ "$app/publish/AotSmoke" >"$log/run-aot.log" 2>&1 || run_ok=no
    if [[ "$run_ok" == yes ]] && ! grep -q '^mode:    Native AOT' "$log/run-aot.log"; then run_ok=no; fi
    if [[ "$run_ok" == no ]]; then
        tail -20 "$log/run-aot.log"
        echo "| $id | $own | $other | ok | **failed** | not clean$note |" >>"$summary"
        [[ -n "$note" ]] || failed=1
        continue
    fi
    if [[ "$own" == 0 && "$other" == 0 ]]; then verdict="AOT-clean"
    elif [[ "$own" == 0 ]]; then verdict="clean in Starlings.*"
    else verdict="not clean"; fi
    echo "run: $(sed -n 's/^package: //p' "$log/run-aot.log"); verdict: $verdict"
    echo "| $id | $own | $other | ok | ok | $verdict$note |" >>"$summary"
done

step "Summary ($summary)"
cat "$summary"
[[ "$failed" == 0 ]] || fail "at least one package failed to publish or run (see above)"
echo "AOT SMOKE PASS"
