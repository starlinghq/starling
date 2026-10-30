#!/usr/bin/env bash
# Release-candidate drill for the Starlings packages: e5 (P-010), GLiNER2 (P-015), the pilot LLM (P-016) and
# EmbeddingGemma 2 (text, image and video embeddings);
# plan sections 10 G6 and 14.
#
# Proves a new user can install and run the release candidate without the source checkout, a hidden cache or an
# upstream tokenizer. From a clean clone of the committed HEAD it
#   1. packs the ai-dotnet packages (Runtime.Onnx, Acquisition,
#      Models.E5, Models.E5.Extensions.AI, Models.E5.MLNet, Models.Gliner2, Generation.Abstractions, Generation.OnnxGenAI,
#      Models.Qwen35, Models.Qwen35.OnnxGenAI, Models.EmbeddingGemma2, Models.EmbeddingGemma2.Extensions.AI,
#      Models.EmbeddingGemma2.MLNet, Models.EmbeddingGemma2.Imaging.SkiaSharp, Models.EmbeddingGemma2.Video.FFmpeg,
#      Models.EmbeddingGemma2.Video.FFmpeg9, Models.Qwen35.Extensions.AI, Models.Kev, Models.Kev.Onnx, Models.QwenChat,
#      Models.QwenChat.OnnxGenAI, Models.Bert, Models.Lfm3), locked restore, empty private NuGet cache,
#   2. inspects every nuspec (version, MIT OR Apache-2.0 or, for Models.Lfm3, the packed LFM Open License file with
#      license acceptance required, readme, repository commit, exact dependency list: no native ONNX Runtime or
#      ONNX Runtime GenAI, no Microsoft.ML.Tokenizers, no FFmpeg bindings package), the packed files (no
#      model assets, no FFmpeg library) and the symbol packages
#      (SourceLink to the commit),
#   3. pushes them to a local folder feed only (never nuget.org),
#   4. copies the samples (ai-dotnet/samples/E5Search, Gliner2Ner, QwenChat, EmbeddingGemma2Search) and the checkers (tools/release-drill/ai-consumer,
#      bert-consumer: the built-in BERT profiles from the package alone, offline) to a temporary
#      directory outside the repository, whose NuGet.config knows only that feed (Starlings.*) and nuget.org
#      (Starlings.Tokenizers, ONNX Runtime, Microsoft.Extensions.*), with no fallback folders and an isolated empty package cache,
#   5. proves the restore graph holds only packages (no project references), that the Starlings packages are
#      the packed bytes (Starlings.Tokenizers: the nuget.org bytes the lock files record), and that Microsoft.ML.Tokenizers and any Hugging Face (Rust) tokenizers binding or native
#      library are absent from the graph, the package cache and the build output,
#   6. acquires the e5 model into an isolated empty cache through the sample's explicit download step (a real Hub
#      download; tokenizer.json is always downloaded, onnx/model.onnx too unless seeded),
#   7. runs the sample and the checker with no network: search results, the Extensions.AI and ML.NET paths, and embeddings against
#      the protected fixtures within the bounds read from the protected tolerances file,
#   8c. runs the ML.NET checker (tools/release-drill/mlnet-consumer) on e5 with no network: Starlings.Models.E5.MLNet's
#      column type, row-wise and look-ahead batched transforms equal to the typed API, token counts, ranking, Save
#      rejected; with --seed-embeddinggemma2-from the same for Starlings.Models.EmbeddingGemma2.MLNet (9e), plus a
#      Matryoshka 256 column,
#   8b. with --seed-embeddinggemma2-from: seeds the EmbeddingGemma 2 text-path graph (release not published yet) into
#      the isolated cache, acquires tokenizer.json and config_sentence_transformers.json from the Hub through the
#      EmbeddingGemma2Search sample's download step, and runs its search with no network (expected top documents, the
#      Extensions.AI path equal to the typed API, Matryoshka 256, the 8,192-token rejection); when <dir> also holds the
#      vision and multimodal text graphs and an images/ directory with the fixture photos, seeds the graphs, verifies the
#      image lock through the sample's download-images step and runs its text-to-image search offline (expected top
#      images, the Extensions.AI image path equal to the typed API); when <dir> also holds videos/ (fixture clips of
#      make_eg2_videos.py) and --ffmpeg-dir names the user's own FFmpeg 8.x shared libraries, runs its text-to-video
#      search offline through the FFmpeg adapter (expected top clips, the Extensions.AI video path equal to the typed
#      API, frames without timing rejected). With --ffmpeg9-dir (the user's own FFmpeg 9.x shared libraries, LGPL) the
#      same search runs through a copy of the sample that references Starlings.Models.EmbeddingGemma2.Video.FFmpeg9
#      instead (EmbeddingGemma2SearchFFmpeg9: the package id and namespace swapped, nothing else). With --ffmpeg-gpl-dir
#      or --ffmpeg9-gpl-dir (a GPL FFmpeg 8 or 9 build) the matching sample must refuse that build without the opt-in
#      (exit 3, "is not LGPL"). No FFmpeg binary is packed, restored or built into any output,
#   8. with --seed-gliner2-from: seeds the GLiNER2 ONNX graph into the isolated cache, acquires
#      tokenizer.json from the Hub through the Gliner2Ner sample's explicit download step, and runs its extraction with
#      no network (expected entities, offsets inside the text, the 512-token rejection),
#   9. with --seed-qwen35-from: seeds the pilot LLM's files (the fp32x graph, the runtime tokenizer.json and
#      chat_template.jinja; the drill runs without a token) into the
#      isolated cache, verifies them through the QwenChat sample's explicit download step, and asks one question with no
#      network (streamed answer, stop at <|im_end|>, usage), through the typed API and through the IChatClient
#      adapter (the same answer),
#  11. with --public-hosting: downloads the GLiNER2, EmbeddingGemma 2 and Qwen 3.5 files from their public hosting into
#      a new empty cache with no token and no seed, then runs each sample offline from that cache (the hosting is
#      public since 2026-10-14). Off by default because it downloads about 4 GB; without the flag it prints
#      "skipped: not requested". A model whose packaged library still records its hosting as private or a placeholder
#      is skipped with "skipped: hosting private",
#      With --public-hosting the EmbeddingGemma 2 image graphs (vision and multimodal text) are also downloaded with
#      no token and, when the seed holds images/, the image search runs offline from that cache; with the label
#      "EmbeddingGemma 2 videos" the same graphs (the video path's vision graphs) are downloaded or verified into that
#      cache and, when the seed holds videos/ and --ffmpeg-dir is given, the video search runs offline from it; the
#      Qwen 3.5 fp16x bundle is downloaded too and, with --cuda, answers offline on the GPU. --public-hosting-only
#      <labels> limits step 11 to a comma-separated subset of: GLiNER2, EmbeddingGemma 2, EmbeddingGemma 2 images,
#      EmbeddingGemma 2 videos, Qwen 3.5, Qwen 3.5 fp16x,
#  12. with --aot: tools/aot-check-ai.sh on the drill's feed and model cache (Native AOT publish of a consumer per
#      package, trim/AOT warnings recorded, AOT results compared with JIT results), then (12b, also with --aot-smoke
#      alone) tools/aot-smoke-ai.sh: every package published under Native AOT with its whole assembly rooted, no model
#      files, its warnings recorded and a summary of which packages are AOT-clean,
#  13. with --cuda <device>: the CUDA profiles from the same packed packages. Step 6 also builds two CUDA consumers
#      outside the repository whose only native packages are the GPU ones, in place of the CPU natives:
#      tools/release-drill/cuda-consumer (Microsoft.ML.OnnxRuntime.Gpu.Linux 1.30.0) and a copy of QwenChat with
#      Microsoft.ML.OnnxRuntimeGenAI.Cuda 0.17.1, and checks their restore graphs (the GPU package exactly once, no CPU
#      natives). Step 13 seeds and verifies e5, GLiNER2, EmbeddingGemma 2 text and image and Qwen 3.5 fp32x and fp16x
#      through the packages' own download steps, then runs each CUDA profile with no network on the one GPU named by
#      <device> (CUDA_DEVICE_ORDER=PCI_BUS_ID, CUDA_VISIBLE_DEVICES=<device>; TF32 off is forced by the libraries):
#      version diagnostics (packages, ONNX Runtime native and managed, providers, provider options, mapped native
#      libraries) and outputs against the protected fixtures within each CUDA profile's protected bounds; Qwen 3.5
#      fp32x and fp16x give the protected reference answer of the en-short prompt through the typed API and the
#      IChatClient adapter. Needs --seed-gliner2-from, --seed-embeddinggemma2-from (with the image graphs and images/)
#      and --seed-qwen35-from (with ortgenai-fp16x/). The CUDA 13 runtime, cuBLAS and cuDNN 9 are the machine's
#      (never in a package): --cuda-lib-path <dir[:dir]> puts them on LD_LIBRARY_PATH for the CUDA runs only.
#      Video on CUDA (embeddinggemma-2/video-fp32-cuda), when the seed holds video-fixtures/ (the 22 make_eg2_videos.py
#      fixture videos) and videos/, and --ffmpeg-dir is given: the CUDA consumer decodes the fixture videos with the
#      FFmpeg 8 adapter (and with the FFmpeg 9 adapter when --ffmpeg9-dir is given) and checks sampled indices and
#      frames (exact) and embeddings (the CUDA profile's protected bounds), also through Extensions.AI; then
#      EmbeddingGemma2SearchCuda (the sample with Microsoft.ML.OnnxRuntime.Gpu.Linux in place of the CPU natives) runs
#      its video search with --cuda.
#
# Resuming (each call stays short): --packages-from <dir> reuses the .nupkg and .snupkg files of an earlier drill
# (artifacts/release-drill-ai/) instead of packing; their nuspecs' commit must differ from HEAD in nothing that is
# packed (ai-dotnet/src, the ai-dotnet and root build files, global.json, the licenses), and every package check of
# step 4 still runs.
# --cuda-only skips the CPU runs (steps 7 to 10 and 12).
#
#   tools/release-drill-ai.sh                                   # downloads both files (487 MB) from the Hub
#   tools/release-drill-ai.sh --seed-model-from <snapshot dir>  # copies onnx/model.onnx from a local snapshot into the
#                                                               # isolated cache first; tokenizer.json is still downloaded
#   tools/release-drill-ai.sh --no-model                        # CI: no model download; downloads tokenizer.json only
#                                                               # and skips the embedding checks
#   ... --seed-gliner2-from <dir>                               # also run the GLiNER2 sample; <dir> holds
#                                                               # onnx/gliner2-entities-fp32.onnx (the graph's source is a
#                                                               # placeholder until the release exists). Combines with
#                                                               # any mode above, including --no-model
#   ... --seed-embeddinggemma2-from <dir>                       # also run the EmbeddingGemma 2 sample; <dir> holds
#                                                               # onnx/embeddinggemma2-text-fp32.onnx, and for the image
#                                                               # search also onnx/embeddinggemma2-vision-fp32.onnx,
#                                                               # onnx/embeddinggemma2-text-mm-fp32.onnx and images/
#                                                               # (make_eg2_images.py photos: chelsea.png, coffee.png,
#                                                               # rocket.jpg, astronaut.png, hubble_deep_field.jpg, ...),
#                                                               # and for the video search videos/ (make_eg2_videos.py
#                                                               # clips: h264-1fps-64x48-8s.mp4,
#                                                               # h264-1.9s-320x240-30fps.mp4,
#                                                               # h264-portrait-360x640-30fps-4s.mp4)
#   ... --ffmpeg-dir <dir>                                      # with --seed-embeddinggemma2-from: the lib directory of
#                                                               # the user's own FFmpeg 8.x shared build (LGPL) for the
#                                                               # video search; FFmpeg is never shipped or downloaded;
#                                                               # the seed may also hold video-fixtures/ (the 22
#                                                               # make_eg2_videos.py videos) for the CUDA video check
#   ... --ffmpeg9-dir <dir>                                     # also the video search through the FFmpeg 9 adapter on
#                                                               # the user's own FFmpeg 9.x shared build (LGPL)
#   ... --ffmpeg-gpl-dir <dir> | --ffmpeg9-gpl-dir <dir>        # a GPL FFmpeg 8 or 9 build the adapter must refuse
#   ... --seed-qwen35-from <dir>                                # also run the QwenChat sample on CPU; <dir> is a bundle built
#                                                               # by build_ortgenai_bundle.py --variant fp32x (3.1 GB)
#   ... --public-hosting                                        # also download GLiNER2, EmbeddingGemma 2 and Qwen 3.5
#                                                               # from public hosting (no token, no seed; ~7 GB); only
#                                                               # once maintainer action M3 has made the hosting public
#   ... --aot                                                   # also run the Native AOT check (tools/aot-check-ai.sh)
#                                                               # and the AOT smoke check (tools/aot-smoke-ai.sh)
#   ... --aot-smoke                                             # only the weight-free AOT smoke check of every package
#   ... --cuda <device> [--cuda-lib-path <dirs>] [--cuda-only]  # also run the CUDA profiles on one GPU (UUID or
#                                                               # PCI-ordered index)
#   ... --packages-from artifacts/release-drill-ai              # reuse the packages of an earlier drill
#   DRILL_KEEP=1 tools/release-drill-ai.sh ...                  # keep the temporary directory for inspection
#
# Uncommitted changes are not part of the drill: it builds exactly the committed HEAD.
# Packages and logs are copied to <repo>/artifacts/release-drill-ai/ (ignored by Git).
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
upstream_url="https://github.com/starlinghq/starling.git"
e5_repo="intfloat/multilingual-e5-small"
e5_commit="614241f622f53c4eeff9890bdc4f31cfecc418b3"
e5_model_sha256="ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665"
fixtures_rel="model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu"
gliner2_repo="fastino/gliner2-base-v1"
gliner2_commit="f9634218e53580c56edf0de97ca1a7d3f1c2354e"
gliner2_graph="onnx/gliner2-entities-fp32.onnx"
eg2_repo="google/embeddinggemma-2"
eg2_commit="914f7f89142e33e77833254d9c9b90c3cef7303b"
eg2_graph="onnx/embeddinggemma2-text-fp32.onnx"
eg2_image_graphs=(onnx/embeddinggemma2-vision-fp32.onnx onnx/embeddinggemma2-text-mm-fp32.onnx)
# The six photos of make_eg2_images.py (scikit-image data); the derived and synthetic images would compete in the search.
eg2_photos=(chelsea.png coffee.png camera.png astronaut.png rocket.jpg hubble_deep_field.jpg)
skia_version="4.153.1"
manifest_rel="model-pipeline/registry/multilingual-e5-small/onnx-fp32-cpu/manifest.yaml"
ort_version="1.30.0"
genai_version="0.17.1"
qwen_repo="Qwen/Qwen3.5-0.8B"
qwen_commit="2fc06364715b967f1860aea9cf38778875588b17"

mode="download"
seed=""
gliner2_seed=""
eg2_seed=""
qwen_seed=""
ffmpeg_dir=""
ffmpeg9_dir=""
ffmpeg_gpl_dir=""
ffmpeg9_gpl_dir=""
public_hosting=0
public_only=""
aot=0
aot_smoke=0
cuda_device=""
cuda_lib_path=""
cuda_only=0
packages_from=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda) cuda_device="$2"; shift 2 ;;
        --cuda-lib-path) cuda_lib_path="$2"; shift 2 ;;
        --cuda-only) cuda_only=1; shift ;;
        --packages-from) packages_from="$(cd "$2" && pwd)"; shift 2 ;;
        --public-hosting-only) public_hosting=1; public_only="$2"; shift 2 ;;
        --seed-model-from) mode="seed"; seed="$(cd "$2" && pwd)"; shift 2 ;;
        --seed-gliner2-from) gliner2_seed="$(cd "$2" && pwd)"; shift 2 ;;
        --seed-embeddinggemma2-from) eg2_seed="$(cd "$2" && pwd)"; shift 2 ;;
        --seed-qwen35-from) qwen_seed="$(cd "$2" && pwd)"; shift 2 ;;
        --ffmpeg-dir) ffmpeg_dir="$(cd "$2" && pwd)"; shift 2 ;;
        --ffmpeg9-dir) ffmpeg9_dir="$(cd "$2" && pwd)"; shift 2 ;;
        --ffmpeg-gpl-dir) ffmpeg_gpl_dir="$(cd "$2" && pwd)"; shift 2 ;;
        --ffmpeg9-gpl-dir) ffmpeg9_gpl_dir="$(cd "$2" && pwd)"; shift 2 ;;
        --no-model) mode="no-model"; shift ;;
        --public-hosting) public_hosting=1; shift ;;
        --aot) aot=1; shift ;;
        --aot-smoke) aot_smoke=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
if [[ "$cuda_only" == 1 && -z "$cuda_device" ]]; then
    echo "--cuda-only needs --cuda <device>" >&2
    exit 2
fi
if [[ -n "$cuda_device" ]]; then
    [[ -n "$gliner2_seed" && -n "$eg2_seed" && -n "$qwen_seed" && "$mode" != "no-model" ]] \
        || { echo "--cuda needs --seed-gliner2-from, --seed-embeddinggemma2-from, --seed-qwen35-from and an e5 model (not --no-model)" >&2; exit 2; }
    [[ -d "$eg2_seed/images" && -f "$qwen_seed/ortgenai-fp16x/model.onnx.data" ]] \
        || { echo "--cuda needs images/ in the EmbeddingGemma 2 seed and ortgenai-fp16x/ in the Qwen 3.5 seed" >&2; exit 2; }
fi

export DOTNET_NOLOGO=1 DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1
# Nothing from the developer's environment may point the apps at an existing model cache or token.
unset STARLING_MODELS STARLING_EMBEDDINGGEMMA2_SNAPSHOT STARLING_EMBEDDINGGEMMA2_IMAGES STARLING_EMBEDDINGGEMMA2_VIDEOS STARLING_FFMPEG_DIR STARLING_E5_SNAPSHOT STARLING_GLINER2_SNAPSHOT STARLING_GLINER2_FIXTURES STARLING_QWEN35_SNAPSHOT HF_TOKEN CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER

work="$(mktemp -d "${TMPDIR:-/tmp}/starlings-drill.XXXXXX")"
cleanup() {
    if [[ "${DRILL_KEEP:-0}" == "1" ]]; then
        echo "kept: $work"
    else
        rm -rf -- "$work"
    fi
}
trap cleanup EXIT

step() { printf '\n== %s\n' "$*"; }
fail() {
    echo "DRILL FAIL: $*" >&2
    exit 1
}

# Runs a command with no network: a new network namespace when unprivileged user namespaces are available,
# otherwise (some CI kernels forbid them) with every proxy variable pointing at a closed port, which .NET honors.
if unshare -rn true 2>/dev/null; then
    offline_how="network namespace without interfaces (unshare -rn)"
    offline() { unshare -rn "$@"; }
else
    offline_how="proxy variables pointing at a closed port (unshare unavailable; weaker)"
    offline() { HTTP_PROXY=http://127.0.0.1:9 HTTPS_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9 NO_PROXY= "$@"; }
fi

step "0. Inputs"
sha="$(git -C "$root" rev-parse HEAD)"
echo "commit:    $sha"
if [[ -n "$(git -C "$root" status --porcelain)" ]]; then
    echo "note:      working tree has uncommitted changes; the drill ignores them and builds $sha"
fi
echo "dotnet:    $(command -v dotnet) (SDK in the clone: pinned by global.json; outside: $(cd "$work" && dotnet --version))"
echo "mode:      $mode${seed:+ (seed: $seed)}${gliner2_seed:+; GLiNER2 graph seed: $gliner2_seed}${eg2_seed:+; EmbeddingGemma 2 graph seed: $eg2_seed}${ffmpeg_dir:+; user FFmpeg: $ffmpeg_dir}${ffmpeg9_dir:+; user FFmpeg 9: $ffmpeg9_dir}${ffmpeg_gpl_dir:+; GPL FFmpeg 8 to refuse: $ffmpeg_gpl_dir}${ffmpeg9_gpl_dir:+; GPL FFmpeg 9 to refuse: $ffmpeg9_gpl_dir}${qwen_seed:+; Qwen3.5 bundle seed: $qwen_seed}"
echo "offline:   $offline_how"
echo "work dir:  $work"
if [[ -n "$cuda_device" ]]; then
    cuda_note=""
    [[ "$cuda_only" != 1 ]] || cuda_note="; CPU runs skipped (--cuda-only)"
    echo "cuda:      device $cuda_device (CUDA_DEVICE_ORDER=PCI_BUS_ID)${cuda_lib_path:+; CUDA runtime and cuDNN from $cuda_lib_path}$cuda_note"
    nvidia-smi --query-gpu=index,pci.bus_id,uuid,name,driver_version --format=csv,noheader 2>/dev/null | sed 's/^/gpu:       /' || true
fi
[[ -z "$packages_from" ]] || echo "packages:  reused from $packages_from"

step "1. Clean checkout of $sha"
src="$work/src"
git clone --quiet --no-hardlinks "$root" "$src"
git -C "$src" -c advice.detachedHead=false checkout --quiet --detach "$sha"
# SourceLink and the nuspec repository element point at the public repository, as in a GitHub checkout.
git -C "$src" remote set-url origin "$upstream_url"
[[ -z "$(git -C "$src" status --porcelain --ignored)" ]] || fail "clone is not clean"
echo "ok:   no build outputs, caches or untracked files in the checkout"

step "2. Versions"
export NUGET_PACKAGES="$work/nuget-build" NUGET_HTTP_CACHE_PATH="$work/nuget-http-build"
# The tokenizer is the published Starlings.Tokenizers (starlinghq/tokenizers): its exact version is the central pin and
# its bytes are the SHA-512 the committed lock files record. It is restored from nuget.org, never packed here.
tok_version="$(sed -n 's:.*<PackageVersion Include="Starlings.Tokenizers" Version="\[\([^],]*\)\]".*:\1:p' "$src/ai-dotnet/Directory.Packages.props")"
[[ -n "$tok_version" ]] || fail "ai-dotnet/Directory.Packages.props does not pin Starlings.Tokenizers exactly"
tok_sha512="$(python3 - "$src/ai-dotnet/src" <<'PY'
import json, pathlib, sys
hashes = {entry["contentHash"]
          for lock in pathlib.Path(sys.argv[1]).glob("*/packages.lock.json")
          for deps in json.loads(lock.read_text())["dependencies"].values()
          for name, entry in deps.items() if name == "Starlings.Tokenizers"}
assert len(hashes) == 1, f"lock files record {len(hashes)} Starlings.Tokenizers hashes"
print(hashes.pop())
PY
)" || fail "cannot read the locked Starlings.Tokenizers hash"
ai_version="$(cd "$src" && dotnet msbuild ai-dotnet/src/Starlings.Models.E5 -getProperty:Version)"
echo "Starlings.Tokenizers: $tok_version"
echo "ai-dotnet packages:            $ai_version"
# Version scheme (docs/kickoff.md, "ai-dotnet package versions and maturity"): 0.MINOR.PATCH, or 0.MINOR.PATCH-preview.N
# for a candidate before it; maturity is per profile (README maturity block), not a package label. The tokenizer keeps
# its own scheme, the same shape since 0.3.0 (starlinghq/tokenizers docs/releasing.md).
[[ "$ai_version" =~ ^0\.[0-9]+\.[0-9]+(-preview\.[0-9]+)?$ ]] \
    || fail "ai-dotnet version $ai_version is not 0.MINOR.PATCH or 0.MINOR.PATCH-preview.N"
[[ "$tok_version" =~ ^0\.[0-9]+\.[0-9]+(-preview\.[0-9]+)?$ ]] \
    || fail "tokenizer version $tok_version is not 0.MINOR.PATCH or 0.MINOR.PATCH-preview.N"
# NuGet refuses a stable package with a prerelease dependency (NU5104): a release without a suffix needs a tokenizer without one.
if [[ "$ai_version" != *-preview.* && "$tok_version" == *-preview.* ]]; then
    fail "ai-dotnet $ai_version is a release but pins the prerelease tokenizer $tok_version"
fi
sample_version="$(sed -n 's:.*<StarlingsVersion>\(.*\)</StarlingsVersion>.*:\1:p' "$src/ai-dotnet/samples/E5Search/E5Search.csproj")"
[[ "$sample_version" == "$ai_version" ]] || fail "sample references $sample_version, the packages are $ai_version"
gliner2_sample_version="$(sed -n 's:.*<StarlingsVersion>\(.*\)</StarlingsVersion>.*:\1:p' "$src/ai-dotnet/samples/Gliner2Ner/Gliner2Ner.csproj")"
[[ "$gliner2_sample_version" == "$ai_version" ]] || fail "Gliner2Ner references $gliner2_sample_version, the packages are $ai_version"
eg2_sample_version="$(sed -n 's:.*<StarlingsVersion>\(.*\)</StarlingsVersion>.*:\1:p' "$src/ai-dotnet/samples/EmbeddingGemma2Search/EmbeddingGemma2Search.csproj")"
[[ "$eg2_sample_version" == "$ai_version" ]] || fail "EmbeddingGemma2Search references $eg2_sample_version, the packages are $ai_version"
for m in "$src"/model-pipeline/registry/embeddinggemma-2/*/manifest.yaml; do
    eg2_manifest_tok="$(sed -n 's/^    version: \(.*\)$/\1/p' "$m" | head -1)"
    [[ "$eg2_manifest_tok" == "$tok_version" ]] || fail "$m pins tokenizer $eg2_manifest_tok, the tokenizer is $tok_version"
done
qwen_sample_version="$(sed -n 's:.*<StarlingsVersion>\(.*\)</StarlingsVersion>.*:\1:p' "$src/ai-dotnet/samples/QwenChat/QwenChat.csproj")"
[[ "$qwen_sample_version" == "$ai_version" ]] || fail "QwenChat references $qwen_sample_version, the packages are $ai_version"
for m in "$src"/model-pipeline/registry/qwen3.5-0.8b/*/manifest.yaml; do
    qwen_manifest_tok="$(sed -n 's/^    version: \(.*\)$/\1/p' "$m" | head -1)"
    [[ "$qwen_manifest_tok" == "$tok_version" ]] || fail "$m pins tokenizer $qwen_manifest_tok, the tokenizer is $tok_version"
done
gliner2_manifest_tok="$(sed -n 's/^    version: \(.*\)$/\1/p' "$src/model-pipeline/registry/gliner2-base-v1/onnx-fp32-cpu/manifest.yaml" | head -1)"
[[ "$gliner2_manifest_tok" == "$tok_version" ]] || fail "GLiNER2 manifest pins tokenizer $gliner2_manifest_tok, the tokenizer is $tok_version"
manifest_tok="$(sed -n 's/^    version: \(.*\)$/\1/p' "$src/$manifest_rel" | head -1)"
[[ "$manifest_tok" == "$tok_version" ]] || fail "manifest processing.tokenizer.version is $manifest_tok, the tokenizer is $tok_version"
echo "ok:   sample references $ai_version; manifest pins tokenizer $manifest_tok"

step "3. Pack (locked restore, empty package cache)"
pack="$work/pack"
projects=(
    ai-dotnet/src/Starlings.Acquisition
    ai-dotnet/src/Starlings.Runtime.Onnx
    ai-dotnet/src/Starlings.Models.Abstractions
    ai-dotnet/src/Starlings.Models.E5
    ai-dotnet/src/Starlings.Models.E5.Extensions.AI
    ai-dotnet/src/Starlings.Runtime.OnnxWeb
    ai-dotnet/src/Starlings.Runtime.OnnxWeb.WebGpu
    ai-dotnet/src/Starlings.Models.E5.Browser
    ai-dotnet/src/Starlings.Models.E5.MLNet
    ai-dotnet/src/Starlings.Models.Gliner2
    ai-dotnet/src/Starlings.Models.EmbeddingGemma2
    ai-dotnet/src/Starlings.Models.EmbeddingGemma2.Extensions.AI
    ai-dotnet/src/Starlings.Models.EmbeddingGemma2.MLNet
    ai-dotnet/src/Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp
    ai-dotnet/src/Starlings.Models.EmbeddingGemma2.Video.FFmpeg
    ai-dotnet/src/Starlings.Models.EmbeddingGemma2.Video.FFmpeg9
    ai-dotnet/src/Starlings.Generation.Abstractions
    ai-dotnet/src/Starlings.Generation.OnnxGenAI
    ai-dotnet/src/Starlings.Models.Qwen35
    ai-dotnet/src/Starlings.Models.Qwen35.OnnxGenAI
    ai-dotnet/src/Starlings.Models.Qwen35.Extensions.AI
    ai-dotnet/src/Starlings.Models.Kev
    ai-dotnet/src/Starlings.Models.Kev.Onnx
    ai-dotnet/src/Starlings.Models.QwenChat
    ai-dotnet/src/Starlings.Models.QwenChat.OnnxGenAI
    ai-dotnet/src/Starlings.Models.Bert
    ai-dotnet/src/Starlings.Models.Lfm3
)
pack_sha="$sha"
if [[ -z "$packages_from" ]]; then
    for proj in "${projects[@]}"; do
        (cd "$src" && CI=true dotnet pack "$proj" --configuration Release --output "$pack" >"$work/pack-$(basename "$proj").log" 2>&1) \
            || { cat "$work/pack-$(basename "$proj").log"; fail "pack failed: $proj"; }
    done
else
    mkdir -p "$pack"
    for proj in "${projects[@]}"; do
        id="$(basename "$proj")"
        cp "$packages_from/$id.$ai_version.nupkg" "$packages_from/$id.$ai_version.snupkg" "$pack/" \
            || fail "--packages-from: $packages_from lacks $id $ai_version"
    done
    pack_sha="$(unzip -p "$pack/Starlings.Models.E5.$ai_version.nupkg" Starlings.Models.E5.nuspec | sed -n 's:.*<repository [^>]*commit="\([0-9a-f]*\)".*:\1:p')"
    git -C "$src" cat-file -e "$pack_sha^{commit}" 2>/dev/null || fail "--packages-from: packed commit $pack_sha is not in the history of $sha"
    packed_paths=(ai-dotnet/src model-pipeline/routes model-pipeline/candidates ai-dotnet/Directory.Build.props ai-dotnet/Directory.Build.targets ai-dotnet/Directory.Packages.props Directory.Build.props .editorconfig global.json LICENSE-APACHE LICENSE-MIT)
    if ! git -C "$src" diff --quiet "$pack_sha" "$sha" -- "${packed_paths[@]}"; then
        git -C "$src" diff --stat "$pack_sha" "$sha" -- "${packed_paths[@]}"
        fail "--packages-from: the packages were built from $pack_sha, which differs from $sha in packed sources"
    fi
    echo "ok:   packages reused from $packages_from, built from $pack_sha; no change to ${packed_paths[*]} between $pack_sha and $sha"
fi
ls "$pack"

step "4. Inspect packages"
# id|version|expected dependencies (sorted "id version" lines)
expect_deps() {
    case "$1" in
        Starlings.Tokenizers) ;;
        Starlings.Acquisition) ;;
        Starlings.Runtime.Onnx) echo "Microsoft.ML.OnnxRuntime.Managed [$ort_version]" ;;
        Starlings.Models.Abstractions) echo "Starlings.Acquisition [$ai_version]" ;;
        Starlings.Models.E5)
            echo "Starlings.Acquisition [$ai_version]"
            echo "Starlings.Models.Abstractions [$ai_version]"
            echo "Starlings.Runtime.Onnx [$ai_version]"
            echo "Starlings.Tokenizers [$tok_version]" ;;
        Starlings.Models.E5.Extensions.AI)
            echo "Starlings.Models.E5 [$ai_version]"
            echo "Microsoft.Extensions.AI.Abstractions 10.10.1"
            echo "Microsoft.Extensions.DependencyInjection.Abstractions 10.0.12" ;;
        Starlings.Runtime.OnnxWeb) ;;
        Starlings.Runtime.OnnxWeb.WebGpu)
            echo "Starlings.Runtime.OnnxWeb [$ai_version]" ;;
        Starlings.Models.E5.Browser)
            echo "Starlings.Models.E5 [$ai_version]"
            echo "Starlings.Models.E5.Extensions.AI [$ai_version]"
            echo "Starlings.Runtime.OnnxWeb [$ai_version]" ;;
        Starlings.Models.E5.MLNet)
            echo "Starlings.Models.E5 [$ai_version]"
            echo "Microsoft.ML 5.0.0" ;;
        Starlings.Models.Gliner2)
            echo "Starlings.Acquisition [$ai_version]"
            echo "Starlings.Models.Abstractions [$ai_version]"
            echo "Starlings.Runtime.Onnx [$ai_version]"
            echo "Starlings.Tokenizers [$tok_version]" ;;
        Starlings.Models.EmbeddingGemma2)
            echo "Starlings.Acquisition [$ai_version]"
            echo "Starlings.Models.Abstractions [$ai_version]"
            echo "Starlings.Runtime.Onnx [$ai_version]"
            echo "Starlings.Tokenizers [$tok_version]" ;;
        Starlings.Models.EmbeddingGemma2.Extensions.AI)
            echo "Starlings.Models.EmbeddingGemma2 [$ai_version]"
            echo "Microsoft.Extensions.AI.Abstractions 10.10.1"
            echo "Microsoft.Extensions.DependencyInjection.Abstractions 10.0.12" ;;
        Starlings.Models.EmbeddingGemma2.MLNet)
            echo "Starlings.Models.EmbeddingGemma2 [$ai_version]"
            echo "Microsoft.ML 5.0.0" ;;
        Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp)
            echo "Starlings.Models.EmbeddingGemma2 [$ai_version]"
            echo "SkiaSharp [$skia_version]"
            echo "SkiaSharp.NativeAssets.Linux.NoDependencies [$skia_version]" ;;
        Starlings.Models.EmbeddingGemma2.Video.FFmpeg | Starlings.Models.EmbeddingGemma2.Video.FFmpeg9)
            echo "Starlings.Models.EmbeddingGemma2 [$ai_version]" ;;
        Starlings.Generation.Abstractions)
            echo "Starlings.Tokenizers [$tok_version]"
            echo "Microsoft.Extensions.AI.Abstractions 10.10.1" ;;
        Starlings.Generation.OnnxGenAI)
            echo "Starlings.Generation.Abstractions [$ai_version]"
            echo "Starlings.Tokenizers [$tok_version]"
            echo "Microsoft.ML.OnnxRuntimeGenAI.Managed [$genai_version]" ;;
        Starlings.Models.Qwen35)
            echo "Starlings.Acquisition [$ai_version]"
            echo "Starlings.Generation.Abstractions [$ai_version]"
            echo "Starlings.Tokenizers [$tok_version]" ;;
        Starlings.Models.Qwen35.OnnxGenAI)
            echo "Starlings.Acquisition [$ai_version]"
            echo "Starlings.Generation.OnnxGenAI [$ai_version]"
            echo "Starlings.Models.Qwen35 [$ai_version]"
            echo "Starlings.Tokenizers [$tok_version]" ;;
        Starlings.Models.Qwen35.Extensions.AI)
            echo "Starlings.Models.Qwen35.OnnxGenAI [$ai_version]"
            echo "Microsoft.Extensions.AI.Abstractions 10.10.1"
            echo "Microsoft.Extensions.DependencyInjection.Abstractions 10.0.12" ;;
        Starlings.Models.Kev)
            echo "Starlings.Tokenizers [$tok_version]" ;;
        Starlings.Models.Kev.Onnx)
            echo "Starlings.Acquisition [$ai_version]"
            echo "Starlings.Models.Kev [$ai_version]"
            echo "Starlings.Runtime.Onnx [$ai_version]"
            echo "Starlings.Tokenizers [$tok_version]" ;;
        Starlings.Models.QwenChat)
            echo "Starlings.Generation.Abstractions [$ai_version]"
            echo "Starlings.Tokenizers [$tok_version]" ;;
        Starlings.Models.QwenChat.OnnxGenAI)
            echo "Starlings.Acquisition [$ai_version]"
            echo "Starlings.Generation.OnnxGenAI [$ai_version]"
            echo "Starlings.Models.QwenChat [$ai_version]"
            echo "Starlings.Tokenizers [$tok_version]" ;;
        Starlings.Models.Bert)
            echo "Starlings.Acquisition [$ai_version]"
            echo "Starlings.Models.Abstractions [$ai_version]"
            echo "Starlings.Runtime.Onnx [$ai_version]"
            echo "Starlings.Tokenizers [$tok_version]" ;;
        Starlings.Models.Lfm3)
            echo "Starlings.Acquisition [$ai_version]"
            echo "Starlings.Generation.Abstractions [$ai_version]"
            echo "Starlings.Runtime.Onnx [$ai_version]"
            echo "Starlings.Tokenizers [$tok_version]" ;;
    esac
}
for proj in "${projects[@]}"; do
    id="$(basename "$proj")"
    version="$ai_version"
    [[ "$id" == Starlings.Tokenizers ]] && version="$tok_version"
    nupkg="$pack/$id.$version.nupkg"
    [[ -f "$nupkg" ]] || fail "expected $nupkg"
    [[ -f "${nupkg%.nupkg}.snupkg" ]] || fail "symbol package missing for $id"
    nuspec="$(unzip -p "$nupkg" "$id.nuspec")"
    contents="$(unzip -Z1 "$nupkg")"
    grep -q "<version>$version</version>" <<<"$nuspec" || fail "$id: nuspec version"
    if [[ "$id" == Starlings.Models.Lfm3 ]]; then
        # The model's license governs this package: the LFM Open License v1.0 is its license file, acceptance is
        # required, and the revenue condition opens the file and the description (model-pipeline/admissions/lfm3.md).
        grep -q '<license type="file">LICENSE.txt</license>' <<<"$nuspec" || fail "$id: nuspec license file"
        grep -q '<requireLicenseAcceptance>true</requireLicenseAcceptance>' <<<"$nuspec" || fail "$id: license acceptance not required"
        grep -q '<description>MODEL LICENSE: LFM Open License v1.0' <<<"$nuspec" || fail "$id: description does not open with the model license"
        unzip -p "$nupkg" LICENSE.txt | grep -q 'USD 10,000,000' || fail "$id: packed LICENSE.txt lacks the revenue condition"
        unzip -p "$nupkg" LICENSE.txt | grep -q '^END OF TERMS AND CONDITIONS' || fail "$id: packed LICENSE.txt lacks the full license text"
        unzip -Z1 "$nupkg" | grep -qx 'THIRD-PARTY-NOTICES.md' || fail "$id: package lacks THIRD-PARTY-NOTICES.md"
    else
        grep -q '<license type="expression">MIT OR Apache-2.0</license>' <<<"$nuspec" || fail "$id: nuspec license"
    fi
    grep -q '<readme>README.md</readme>' <<<"$nuspec" || fail "$id: nuspec readme"
    grep -q "url=\"https://github.com/starlinghq/starling\".*commit=\"$pack_sha\"" <<<"$nuspec" || fail "$id: nuspec repository commit"
    if [[ "$id" != Starlings.Tokenizers ]]; then
        # Maturity is per profile: the description points to the README, whose maturity block the catalog generates
        # from the registry manifests (model-pipeline catalog --check keeps it current). No blanket Preview label.
        if grep -q '<description>Preview\. ' <<<"$nuspec"; then fail "$id: description starts with a blanket Preview label"; fi
        grep -q 'Maturity is per model profile' <<<"$nuspec" || fail "$id: description does not state per-profile maturity"
        readme_text="$(unzip -p "$nupkg" README.md)"
        grep -q '^<!-- catalog:maturity profiles=' <<<"$readme_text" || fail "$id: packed README lacks the generated maturity block"
        grep -q '^| Profile | Maturity | Platforms |$' <<<"$readme_text" || fail "$id: packed README maturity block is empty"
        if grep -q '\*\*Preview\.\*\*' <<<"$readme_text"; then fail "$id: packed README carries a blanket Preview label"; fi
    fi
    actual="$(sed -n 's:.*<dependency id="\([^"]*\)" version="\([^"]*\)".*:\1 \2:p' <<<"$nuspec" | sort)"
    expected="$(expect_deps "$id" | sort)"
    [[ "$actual" == "$expected" ]] || fail "$id: dependencies
$actual
expected
$expected"
    for f in "lib/net10.0/$id.dll" "lib/net10.0/$id.xml" README.md; do
        grep -qx "$f" <<<"$contents" || fail "$id: package lacks $f"
    done
    # The build-time native package checks travel to every consumer (README, "Native package check").
    case "$id" in
        Starlings.Runtime.Onnx | Starlings.Generation.OnnxGenAI)
            grep -qx "buildTransitive/$id.targets" <<<"$contents" || fail "$id: package lacks buildTransitive/$id.targets" ;;
    esac
    # FFmpeg is user-supplied: no package may carry an FFmpeg library (libavcodec.so.62, avcodec-62.dll, ...).
    if grep -Eqi '(^|/)(lib)?(avcodec|avformat|avutil|avfilter|avdevice|swscale|swresample|postproc)([-.][0-9]+)*\.(so|dll|dylib)(\.[0-9]+)*$' <<<"$contents"; then
        fail "$id: package contains an FFmpeg library"
    fi
    if grep -Eqi '\.(onnx|safetensors|bin|gguf)$|tokenizer\.json$' <<<"$contents"; then
        fail "$id: package contains model assets"
    fi
    pdb="$(unzip -p "${nupkg%.nupkg}.snupkg" "lib/net10.0/$id.pdb" | grep -a -o "raw.githubusercontent.com/starlinghq/starling/$pack_sha/\*" | head -1 || true)"
    [[ -n "$pdb" ]] || fail "$id: symbols lack SourceLink to $pack_sha"
    label="per-profile maturity in README and description"
    # The tokenizer is released on its own (P-025), with its own README and version.
    [[ "$id" == Starlings.Tokenizers ]] && label="README, own version"
    license_label="MIT OR Apache-2.0"
    [[ "$id" == Starlings.Models.Lfm3 ]] && license_label="LFM Open License v1.0 (license file, acceptance required)"
    echo "ok:   $id $version: $license_label, $label, repository commit, SourceLink to the commit; dependencies: ${expected:-none}" | tr '\n' ' '
    echo
    echo "      sha256 $(sha256sum "$nupkg" | cut -d' ' -f1)"
done

step "5. Push to a local folder feed (never nuget.org)"
feed="$work/feed"
mkdir -p "$feed"
for nupkg in "$pack"/*.nupkg; do
    dotnet nuget push "$nupkg" --source "$feed" >/dev/null
done
ls "$feed"

step "6. Clean consumers outside the repository"
apps="$work/apps"
mkdir -p "$apps/E5Search" "$apps/Gliner2Ner" "$apps/QwenChat" "$apps/EmbeddingGemma2Search" "$apps/EmbeddingGemma2SearchFFmpeg9" "$apps/AiConsumer" "$apps/MlNetConsumer" "$apps/BertConsumer" "$work/fixtures"
# Only the sample's sources and project file; nothing above it (its Directory.Build.* files exist only to stop
# repository imports inside the checkout).
cp "$src/ai-dotnet/samples/E5Search/Program.cs" "$src/ai-dotnet/samples/E5Search/E5Search.csproj" "$apps/E5Search/"
cp "$src/ai-dotnet/samples/Gliner2Ner/Program.cs" "$src/ai-dotnet/samples/Gliner2Ner/Gliner2Ner.csproj" "$apps/Gliner2Ner/"
cp "$src/ai-dotnet/samples/EmbeddingGemma2Search/Program.cs" "$src/ai-dotnet/samples/EmbeddingGemma2Search/EmbeddingGemma2Search.csproj" "$apps/EmbeddingGemma2Search/"
cp "$src/ai-dotnet/samples/QwenChat/Program.cs" "$src/ai-dotnet/samples/QwenChat/QwenChat.csproj" "$apps/QwenChat/"
cp "$src/tools/release-drill/ai-consumer/Program.cs" "$apps/AiConsumer/"
sed "s/@VERSION@/$ai_version/" "$src/tools/release-drill/ai-consumer/AiConsumer.csproj.in" >"$apps/AiConsumer/AiConsumer.csproj"
cp "$src/tools/release-drill/mlnet-consumer/Program.cs" "$apps/MlNetConsumer/"
sed "s/@VERSION@/$ai_version/" "$src/tools/release-drill/mlnet-consumer/MlNetConsumer.csproj.in" >"$apps/MlNetConsumer/MlNetConsumer.csproj"
cp "$src/tools/release-drill/bert-consumer/Program.cs" "$apps/BertConsumer/"
sed "s/@VERSION@/$ai_version/" "$src/tools/release-drill/bert-consumer/BertConsumer.csproj.in" >"$apps/BertConsumer/BertConsumer.csproj"
cp "$src/$fixtures_rel/embeddings.jsonl" "$src/$fixtures_rel/tokenizer.jsonl" "$work/fixtures/"
# The sample on the FFmpeg 9 adapter: the same code with the package id and namespace of Starlings.Models.EmbeddingGemma2.Video.FFmpeg9.
sed 's|using Starlings.Models.EmbeddingGemma2.Video.FFmpeg;|using Starlings.Models.EmbeddingGemma2.Video.FFmpeg9;|' \
    "$src/ai-dotnet/samples/EmbeddingGemma2Search/Program.cs" >"$apps/EmbeddingGemma2SearchFFmpeg9/Program.cs"
sed 's|<PackageReference Include="Starlings.Models.EmbeddingGemma2.Video.FFmpeg" |<PackageReference Include="Starlings.Models.EmbeddingGemma2.Video.FFmpeg9" |' \
    "$src/ai-dotnet/samples/EmbeddingGemma2Search/EmbeddingGemma2Search.csproj" >"$apps/EmbeddingGemma2SearchFFmpeg9/EmbeddingGemma2SearchFFmpeg9.csproj"
grep -q '^using Starlings.Models.EmbeddingGemma2.Video.FFmpeg9;$' "$apps/EmbeddingGemma2SearchFFmpeg9/Program.cs" \
    && grep -q '<PackageReference Include="Starlings.Models.EmbeddingGemma2.Video.FFmpeg9" ' "$apps/EmbeddingGemma2SearchFFmpeg9/EmbeddingGemma2SearchFFmpeg9.csproj" \
    || fail "EmbeddingGemma2SearchFFmpeg9: could not swap the FFmpeg 8 adapter for Starlings.Models.EmbeddingGemma2.Video.FFmpeg9"
consumer_apps=(E5Search Gliner2Ner QwenChat EmbeddingGemma2Search EmbeddingGemma2SearchFFmpeg9 AiConsumer MlNetConsumer BertConsumer)
if [[ -n "$cuda_device" ]]; then
    # The CUDA consumers: the GPU native packages in place of the CPU ones, never both (two libonnxruntime.so copies).
    mkdir -p "$apps/CudaConsumer" "$apps/QwenChatCuda" "$apps/EmbeddingGemma2SearchCuda"
    cp "$src/ai-dotnet/samples/EmbeddingGemma2Search/Program.cs" "$apps/EmbeddingGemma2SearchCuda/"
    sed 's|<PackageReference Include="Microsoft.ML.OnnxRuntime" Version="|<PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu.Linux" Version="|' \
        "$src/ai-dotnet/samples/EmbeddingGemma2Search/EmbeddingGemma2Search.csproj" >"$apps/EmbeddingGemma2SearchCuda/EmbeddingGemma2SearchCuda.csproj"
    grep -q "Microsoft.ML.OnnxRuntime.Gpu.Linux\" Version=\"$ort_version\"" "$apps/EmbeddingGemma2SearchCuda/EmbeddingGemma2SearchCuda.csproj" \
        || fail "EmbeddingGemma2SearchCuda: could not swap the ONNX Runtime natives for Microsoft.ML.OnnxRuntime.Gpu.Linux $ort_version"
    consumer_apps+=(EmbeddingGemma2SearchCuda)
    cp "$src/tools/release-drill/cuda-consumer/Program.cs" "$apps/CudaConsumer/"
    sed "s/@VERSION@/$ai_version/" "$src/tools/release-drill/cuda-consumer/CudaConsumer.csproj.in" >"$apps/CudaConsumer/CudaConsumer.csproj"
    cp "$src/ai-dotnet/samples/QwenChat/Program.cs" "$apps/QwenChatCuda/"
    sed 's|<PackageReference Include="Microsoft.ML.OnnxRuntimeGenAI" Version="|<PackageReference Include="Microsoft.ML.OnnxRuntimeGenAI.Cuda" Version="|' \
        "$src/ai-dotnet/samples/QwenChat/QwenChat.csproj" >"$apps/QwenChatCuda/QwenChatCuda.csproj"
    grep -q "Microsoft.ML.OnnxRuntimeGenAI.Cuda\" Version=\"$genai_version\"" "$apps/QwenChatCuda/QwenChatCuda.csproj" \
        || fail "QwenChatCuda: could not swap the GenAI natives for Microsoft.ML.OnnxRuntimeGenAI.Cuda $genai_version"
    consumer_apps+=(CudaConsumer QwenChatCuda)
fi
cat >"$apps/NuGet.config" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <packageSources>
    <clear />
    <add key="drill-local" value="$feed" />
    <add key="nuget.org" value="https://api.nuget.org/v3/index.json" />
  </packageSources>
  <fallbackPackageFolders>
    <clear />
  </fallbackPackageFolders>
  <packageSourceMapping>
    <packageSource key="drill-local">
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
for app in "${consumer_apps[@]}"; do
    (cd "$apps/$app" && dotnet restore >"$work/restore-$app.log" 2>&1) || { cat "$work/restore-$app.log"; fail "$app restore failed"; }
    (cd "$apps/$app" && dotnet list package --include-transitive) | tee "$work/$app-packages.txt"
done
python3 - "$pack" "$tok_sha512" "$ort_version" "$genai_version" $(for app in "${consumer_apps[@]}"; do echo "$apps/$app/obj/project.assets.json"; done) <<'EOF' || fail "restore graph check failed"
import base64, hashlib, json, os, re, sys
pack, tok_sha512, ort_version, genai_version = sys.argv[1:5]
for path in sys.argv[5:]:
    assets = json.load(open(path))
    libraries = assets["libraries"]
    names = sorted(libraries)
    print(f"{path.split('/')[-3]}: {len(names)} packages: {', '.join(names)}")
    assert all(lib["type"] == "package" for lib in libraries.values()), "non-package library (project reference)"
    # Microsoft.ML.Tokenizers, Hugging Face tokenizers bindings (any package with "tokenizer" in its id other than
    # ours), and GPU natives in a CPU application.
    foreign = [n for n in names if re.search(r"tokeni[sz]er", n, re.I) and not n.startswith("Starlings.Tokenizers/")]
    assert not foreign, f"foreign tokenizer packages: {foreign}"
    app = path.split("/")[-3]
    gpu = [n for n in names if re.match(r"Microsoft\.ML\.OnnxRuntime(GenAI\.Cuda|\.Gpu)", n, re.I)]
    cpu_natives = [n for n in names if n.split("/")[0] in ("Microsoft.ML.OnnxRuntime", "Microsoft.ML.OnnxRuntimeGenAI")]
    if app in ("CudaConsumer", "EmbeddingGemma2SearchCuda"):
        # The CUDA natives of ONNX Runtime, exactly once and at the pinned version; no CPU natives beside them.
        assert gpu == [f"Microsoft.ML.OnnxRuntime.Gpu.Linux/{ort_version}"], f"{app} GPU natives: {gpu}"
        assert not cpu_natives, f"CPU natives restored beside the GPU ones: {cpu_natives}"
    elif app == "QwenChatCuda":
        # Microsoft.ML.OnnxRuntimeGenAI.Cuda brings Microsoft.ML.OnnxRuntime.Gpu (Linux and Windows natives) at the pinned ORT.
        want = sorted([f"Microsoft.ML.OnnxRuntimeGenAI.Cuda/{genai_version}", f"Microsoft.ML.OnnxRuntime.Gpu/{ort_version}",
                       f"Microsoft.ML.OnnxRuntime.Gpu.Linux/{ort_version}", f"Microsoft.ML.OnnxRuntime.Gpu.Windows/{ort_version}"])
        assert gpu == want, f"QwenChatCuda GPU natives: {gpu}, expected {want}"
        assert not cpu_natives, f"CPU natives restored beside the GPU ones: {cpu_natives}"
    else:
        assert not gpu, "GPU natives restored"
    assert not [n for n in names if re.match(r"SixLabors\.", n, re.I)], "ImageSharp restored"
    skia = [n for n in names if n.lower().startswith("skiasharp")]
    eg2_search = ("EmbeddingGemma2Search", "EmbeddingGemma2SearchFFmpeg9", "EmbeddingGemma2SearchCuda")
    assert not skia or app in eg2_search + ("CudaConsumer",), f"SkiaSharp restored outside the image consumers: {skia}"
    # FFmpeg: the video adapter carries its own generated bindings; no FFmpeg bindings package and never FFmpeg binaries.
    bundled = [n for n in names if re.search(r"sdcb\.ffmpeg|ffmpeg\.(native|runtime|redist)|ffmpeg\.autogen\.(redist|runtime)|ffmpegcore\.binaries|xabe\.ffmpeg\.downloader", n, re.I)]
    assert not bundled, f"FFmpeg binaries or Sdcb.FFmpeg restored: {bundled}"
    bindings = [n for n in names if re.match(r"(ffmpeg\.autogen|ffmpegcore|sdcb\.ffmpeg)/", n, re.I)]
    assert not bindings, f"FFmpeg bindings package restored in {app}: {bindings}"
    if app in ("QwenChat", "QwenChatCuda"):
        # The pilot LLM: GenAI CPU natives (which bring ONNX Runtime CPU), or the CUDA natives; no Runtime.Onnx, e5 or
        # GLiNER2 packages.
        common = (["Starlings.Tokenizers", "Starlings.Acquisition", "Microsoft.ML.OnnxRuntime",
                   "Microsoft.ML.OnnxRuntimeGenAI", "Microsoft.ML.OnnxRuntimeGenAI.Managed"] if app == "QwenChat"
                  else ["Starlings.Tokenizers", "Starlings.Acquisition", "Microsoft.ML.OnnxRuntime.Gpu",
                        "Microsoft.ML.OnnxRuntime.Gpu.Linux", "Microsoft.ML.OnnxRuntimeGenAI.Cuda",
                        "Microsoft.ML.OnnxRuntimeGenAI.Managed", "Microsoft.ML.OnnxRuntime.Managed"])
        models = ["Starlings.Generation.Abstractions", "Starlings.Generation.OnnxGenAI", "Starlings.Models.Qwen35",
                  "Starlings.Models.Qwen35.OnnxGenAI", "Starlings.Models.Qwen35.Extensions.AI", "Microsoft.Extensions.AI.Abstractions"]
        # One Microsoft.Extensions.AI.Abstractions: Generation.Abstractions' 10.10.1 lifts the 9.8.0 that GenAI's managed binding asks for.
        mea = [n for n in names if n.split("/")[0] == "Microsoft.Extensions.AI.Abstractions"]
        assert mea == ["Microsoft.Extensions.AI.Abstractions/10.10.1"], f"Microsoft.Extensions.AI.Abstractions resolved as {mea}"
        unrelated = ("Starlings.Runtime.Onnx", "Starlings.Models.E5", "Starlings.Models.Gliner2",
                     "Starlings.Models.EmbeddingGemma2", "Starlings.Models.Abstractions")
        assert not [n for n in names if n.split("/")[0] in unrelated], "unrelated packages restored"
    else:
        common = ["Starlings.Tokenizers", "Starlings.Acquisition", "Starlings.Runtime.Onnx", "Starlings.Models.Abstractions",
                  "Microsoft.ML.OnnxRuntime.Gpu.Linux" if app in ("CudaConsumer", "EmbeddingGemma2SearchCuda") else "Microsoft.ML.OnnxRuntime",
                  "Microsoft.ML.OnnxRuntime.Managed"]
        # Each EmbeddingGemma2Search copy restores exactly one video adapter: FFmpeg9 for its FFmpeg 9 copy, FFmpeg otherwise.
        video_adapters = [n.split("/")[0] for n in names if n.startswith("Starlings.Models.EmbeddingGemma2.Video.")]
        if app in eg2_search:
            want_adapter = "Starlings.Models.EmbeddingGemma2.Video.FFmpeg9" if app == "EmbeddingGemma2SearchFFmpeg9" else "Starlings.Models.EmbeddingGemma2.Video.FFmpeg"
            assert video_adapters == [want_adapter], f"{app} video adapters: {video_adapters}, expected [{want_adapter}]"
        elif app != "CudaConsumer":
            assert not video_adapters, f"{app} restored a video adapter: {video_adapters}"
        models = (["Starlings.Models.Gliner2"] if app == "Gliner2Ner"
                  else ["Starlings.Models.E5", "Starlings.Models.E5.Extensions.AI", "Starlings.Models.Gliner2",
                        "Starlings.Models.EmbeddingGemma2", "Starlings.Models.EmbeddingGemma2.Extensions.AI",
                        "Microsoft.Extensions.AI.Abstractions", "Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp",
                        "SkiaSharp", "SkiaSharp.NativeAssets.Linux.NoDependencies",
                        "Starlings.Models.EmbeddingGemma2.Video.FFmpeg", "Starlings.Models.EmbeddingGemma2.Video.FFmpeg9"] if app == "CudaConsumer"
                  else ["Starlings.Models.EmbeddingGemma2", "Starlings.Models.EmbeddingGemma2.Extensions.AI",
                        "Microsoft.Extensions.AI.Abstractions", "Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp",
                        "SkiaSharp", "SkiaSharp.NativeAssets.Linux.NoDependencies", want_adapter] if app in eg2_search
                  else ["Starlings.Models.E5", "Starlings.Models.E5.MLNet", "Starlings.Models.EmbeddingGemma2",
                        "Starlings.Models.EmbeddingGemma2.MLNet", "Microsoft.ML"] if app == "MlNetConsumer"
                  else ["Starlings.Models.Bert"] if app == "BertConsumer"
                  else ["Starlings.Models.E5", "Starlings.Models.E5.Extensions.AI"])
    if app == "BertConsumer":
        # Starlings.Models.Bert alone: its four sibling dependencies and the CPU natives, no other model package.
        others = [n for n in names if n.startswith("Starlings.") and n.split("/")[0] not in common + models]
        assert not others, f"BertConsumer restored other Starlings packages: {others}"
    if app == "AiConsumer":
        # The checker also consumes the ML.NET adapter: Microsoft.ML must come without any tokenizer package.
        models += ["Starlings.Models.E5.MLNet", "Microsoft.ML"]
    for want in common + models:
        hits = [n for n in names if n.split("/")[0] == want]
        assert len(hits) == 1, f"{want} not restored exactly once"
        if want == "Starlings.Tokenizers":
            assert libraries[hits[0]]["sha512"] == tok_sha512, f"{hits[0]} differs from the locked nuget.org package"
        elif want.startswith("Starlings."):
            name, version = hits[0].split("/")
            nupkg = os.path.join(pack, f"{name}.{version}.nupkg")
            packed = base64.b64encode(hashlib.sha512(open(nupkg, "rb").read()).digest()).decode()
            assert libraries[hits[0]]["sha512"] == packed, f"{hits[0]} differs from the packed package"
    natives = (f"GPU natives only ({', '.join(gpu)})" if app in ("CudaConsumer", "QwenChatCuda", "EmbeddingGemma2SearchCuda") else "no GPU natives")
    print(f"ok:   {app}: only packages; Starlings.* sha512 equal the packed .nupkg files and the locked tokenizer; no foreign tokenizer; {natives}")
EOF
if find "$NUGET_PACKAGES" -iname '*tokeni[sz]er*' | grep -vi '/starlings\.tokenizers' | grep -q .; then
    find "$NUGET_PACKAGES" -iname '*tokeni[sz]er*' | grep -vi '/starlings\.tokenizers'
    fail "foreign tokenizer files in the consumer package cache"
fi
echo "ok:   consumer package cache holds only: $(ls "$NUGET_PACKAGES" | tr '\n' ' ')"
for app in "${consumer_apps[@]}"; do
    (cd "$apps/$app" && dotnet build --configuration Release --no-restore >"$work/build-$app.log" 2>&1) \
        || { cat "$work/build-$app.log"; fail "$app build failed"; }
    # Each consumer references exactly one native package per runtime: the native package checks stay quiet.
    if grep -Eq '(warning|error) STARLINGS00[1-4]' "$work/build-$app.log"; then
        grep -E '(warning|error) STARLINGS00[1-4]' "$work/build-$app.log" | sort -u
        fail "$app: a native package check fired on a consumer with one native package per runtime"
    fi
done
echo "ok:   no native package check fired in ${consumer_apps[*]}"

# The build-time native package checks (buildTransitive targets of Starlings.Runtime.Onnx and
# Starlings.Generation.OnnxGenAI) on throwaway consumers: no native warns, a CPU and a GPU native together fail.
# The two-native cases restore the GPU packages (about 420 MB) into the isolated consumer cache.
native_cases=(
    "ort-none|Starlings.Models.E5||warning STARLINGS001"
    "ort-both|Starlings.Models.E5|Microsoft.ML.OnnxRuntime/$ort_version Microsoft.ML.OnnxRuntime.Gpu.Linux/$ort_version|error STARLINGS002"
    "genai-none|Starlings.Models.Qwen35.OnnxGenAI||warning STARLINGS003"
    "genai-both|Starlings.Models.Qwen35.OnnxGenAI|Microsoft.ML.OnnxRuntimeGenAI/$genai_version Microsoft.ML.OnnxRuntimeGenAI.Cuda/$genai_version|error STARLINGS004"
)
for entry in "${native_cases[@]}"; do
    IFS='|' read -r name package natives want <<<"$entry"
    case_dir="$apps/NativeCheck-$name"
    mkdir -p "$case_dir"
    refs=""
    for native in $natives; do refs="$refs<PackageReference Include=\"${native%/*}\" Version=\"${native#*/}\" />"; done
    sed -e "s|@VERSION@|$ai_version|" -e "s|@PACKAGE@|$package|" -e "s|@NATIVES@|$refs|" \
        "$src/tools/release-drill/native-check/NativeCheck.csproj.in" >"$case_dir/NativeCheck.csproj"
    echo 'Console.WriteLine("native check");' >"$case_dir/Program.cs"
    status=0
    (cd "$case_dir" && dotnet build --configuration Release >"$work/native-check-$name.log" 2>&1) || status=$?
    grep -q "$want" "$work/native-check-$name.log" || { tail -30 "$work/native-check-$name.log"; fail "native check $name: expected $want"; }
    if [[ "$want" == error* && "$status" == 0 ]]; then fail "native check $name: the build passed despite $want"; fi
    if [[ "$want" == warning* && "$status" != 0 ]]; then tail -30 "$work/native-check-$name.log"; fail "native check $name: the build failed"; fi
    echo "ok:   native check $name: $package${natives:+ + $natives}: $want"
done
if find "$apps" -path '*/bin/*' \( -iname '*tokeni[sz]er*' -a ! -iname 'Starlings.Tokenizers.*' \) | grep -q .; then
    fail "foreign tokenizer files in a build output"
fi
if grep -rIl --exclude='*.log' -F "$root" "$apps"; then
    fail "consumer files mention the repository path"
fi
echo "ok:   no file in the consumers (sources, NuGet.config, obj, bin) names the repository path"
echo "ok:   build outputs hold no tokenizer library other than Starlings.Tokenizers.dll"
if find "$NUGET_PACKAGES" "$apps" -regextype posix-extended -iregex '.*/(lib)?(avcodec|avformat|avutil|avfilter|avdevice|swscale|swresample|postproc)([-.][0-9]+)*\.(so|dll|dylib)(\.[0-9]+)*' | grep -q .; then
    find "$NUGET_PACKAGES" "$apps" -regextype posix-extended -iregex '.*/(lib)?(avcodec|avformat|avutil|avfilter|avdevice|swscale|swresample|postproc)([-.][0-9]+)*\.(so|dll|dylib)(\.[0-9]+)*'
    fail "an FFmpeg library is in the consumer package cache or a build output (FFmpeg must be user-supplied)"
fi
echo "ok:   no FFmpeg library in the consumer package cache or any build output (the video adapter loads the user's FFmpeg through its own bindings)"

# The browser sample (Blazor WebAssembly) from the packed browser packages: restored from the local feed only, every
# Starlings.* package equal to the packed bytes, published without the wasm-tools workload (interpreter), and the output
# carries onnxruntime-web, the module and onnxruntime's notices. Inference needs a browser: browser/conformance/ and the
# qualification job "browser" run it.
browser_sample_version="$(sed -n 's:.*<StarlingsVersion>\(.*\)</StarlingsVersion>.*:\1:p' "$src/ai-dotnet/samples/E5BrowserSearch/E5BrowserSearch.csproj")"
[[ "$browser_sample_version" == "$ai_version" ]] || fail "E5BrowserSearch references $browser_sample_version, the packages are $ai_version"
mkdir -p "$apps/E5BrowserSearch"
cp -r "$src/ai-dotnet/samples/E5BrowserSearch/." "$apps/E5BrowserSearch/"
rm -rf "$apps/E5BrowserSearch/bin" "$apps/E5BrowserSearch/obj"
(cd "$apps/E5BrowserSearch" && dotnet publish --configuration Release -o "$work/browser-publish" >"$work/publish-E5BrowserSearch.log" 2>&1) \
    || { cat "$work/publish-E5BrowserSearch.log"; fail "E5BrowserSearch publish failed"; }
python3 - "$pack" "$apps/E5BrowserSearch/obj/project.assets.json" <<'PYEOF' || fail "browser sample restore check failed"
import base64, hashlib, json, os, sys
pack, path = sys.argv[1:3]
libraries = json.load(open(path))["libraries"]
names = sorted(libraries)
assert all(lib["type"] == "package" for lib in libraries.values()), "non-package library (project reference)"
for want in ("Starlings.Models.E5.Browser", "Starlings.Runtime.OnnxWeb", "Starlings.Runtime.OnnxWeb.WebGpu",
             "Starlings.Models.E5", "Starlings.Models.E5.Extensions.AI"):
    assert len([n for n in names if n.split("/")[0] == want]) == 1, f"{want} not restored exactly once"
for name_version in names:
    name, version = name_version.split("/")
    if name.startswith("Starlings.") and name != "Starlings.Tokenizers":
        packed = base64.b64encode(hashlib.sha512(open(os.path.join(pack, f"{name}.{version}.nupkg"), "rb").read()).digest()).decode()
        assert libraries[name_version]["sha512"] == packed, f"{name_version} differs from the packed package"
# A browser app needs no native ONNX Runtime: only the managed binding comes in, through Starlings.Runtime.Onnx.
natives = [n for n in names if n.split("/")[0] in ("Microsoft.ML.OnnxRuntime", "Microsoft.ML.OnnxRuntime.Gpu.Linux")]
assert not natives, f"native ONNX Runtime restored into the browser sample: {natives}"
print(f"ok:   E5BrowserSearch: {len(names)} packages, Starlings.* equal the packed .nupkg files, no ONNX Runtime natives")
PYEOF
for f in starlings-onnxweb.js ort/ort.wasm.min.mjs ort/ort-wasm-simd-threaded.wasm ort/LICENSE-onnxruntime.txt ort/ThirdPartyNotices-onnxruntime.txt; do
    [[ -f "$work/browser-publish/wwwroot/_content/Starlings.Runtime.OnnxWeb/$f" ]] || fail "E5BrowserSearch publish lacks _content/Starlings.Runtime.OnnxWeb/$f"
done
for f in ort/ort.webgpu.min.mjs ort/ort-wasm-simd-threaded.asyncify.wasm ort/ThirdPartyNotices-onnxruntime.txt; do
    [[ -f "$work/browser-publish/wwwroot/_content/Starlings.Runtime.OnnxWeb.WebGpu/$f" ]] || fail "E5BrowserSearch publish lacks _content/Starlings.Runtime.OnnxWeb.WebGpu/$f"
done
echo "ok:   E5BrowserSearch publishes from the packed packages with onnxruntime-web, its module and onnxruntime's notices"

sample=(dotnet "$apps/E5Search/bin/Release/net10.0/E5Search.dll")
gliner2_sample=(dotnet "$apps/Gliner2Ner/bin/Release/net10.0/Gliner2Ner.dll")
eg2_sample=(dotnet "$apps/EmbeddingGemma2Search/bin/Release/net10.0/EmbeddingGemma2Search.dll")
qwen_sample=(dotnet "$apps/QwenChat/bin/Release/net10.0/QwenChat.dll")
checker=(dotnet "$apps/AiConsumer/bin/Release/net10.0/AiConsumer.dll")
mlnet_checker=(dotnet "$apps/MlNetConsumer/bin/Release/net10.0/MlNetConsumer.dll")
bert_checker=(dotnet "$apps/BertConsumer/bin/Release/net10.0/BertConsumer.dll")
models="$work/models"
mkdir -p "$models"

# Every mode: the built-in BERT profiles from the packed package, offline (no model; the snapshot directory is empty).
mkdir -p "$work/bert-empty"
offline "${bert_checker[@]}" "$work/bert-empty" | tee "$work/consumer-bert.log" || fail "BERT checker failed"
grep -qx 'RESULT: PASS' "$work/consumer-bert.log" || fail "BERT checker did not pass"
echo "ok:   Starlings.Models.Bert: the seven built-in profiles parse from the package alone, offline; Create verifies before loading"

if [[ "$cuda_only" == 1 ]]; then
    echo "note: --cuda-only: the CPU runs (steps 7 to 10 and 12) were skipped; the CPU consumers were restored and built"
elif [[ "$mode" == "no-model" ]]; then
    step "7. Acquisition of tokenizer.json only (real Hub download; no model)"
    "${checker[@]}" tokenizer "$models" "$work/fixtures" | tee "$work/consumer-tokenizer.log" || fail "checker failed"
    grep -qx 'RESULT: PASS' "$work/consumer-tokenizer.log" || fail "checker did not pass"
    echo "note: --no-model: the model download, offline search and embedding checks were skipped"
    echo "note: --no-model: the ML.NET packages were packed, inspected and their checker built, but not run"
else
    step "7. Explicit acquisition into an isolated empty cache (real Hub download)"
    if [[ "$mode" == "seed" ]]; then
        target="$models/$e5_repo/$e5_commit/onnx"
        mkdir -p "$target"
        cp --reflink=auto "$seed/onnx/model.onnx" "$target/model.onnx"
        echo "seeded: onnx/model.onnx from $seed (the store verifies size and SHA-256 before use)"
    fi
    "${sample[@]}" download --cache "$models" | tee "$work/sample-download.log" || fail "sample download failed"
    grep -q '^tokenizer.json: downloaded,' "$work/sample-download.log" || fail "tokenizer.json was not downloaded"
    if [[ "$mode" == "download" ]]; then
        grep -q '^onnx/model.onnx: downloaded,' "$work/sample-download.log" || fail "onnx/model.onnx was not downloaded"
    fi
    echo "ok:   cache content: $(cd "$models" && find . -type f | sort | tr '\n' ' ')"
    [[ "$(sha256sum "$models/$e5_repo/$e5_commit/onnx/model.onnx" | cut -d' ' -f1)" == "$e5_model_sha256" ]] || fail "model digest"

    step "8. Offline: sample search and fixture checks with no network"
    if offline curl -sS --max-time 5 -o /dev/null https://huggingface.co 2>/dev/null; then
        fail "the offline wrapper still reaches huggingface.co"
    fi
    echo "ok:   huggingface.co unreachable inside the wrapper ($offline_how)"
    offline "${sample[@]}" search --cache "$models" | tee "$work/sample-search.log" || fail "sample search failed"
    for expect in 'how much protein should a female eat:protein' 'Warum rostet Eisen?:rust' \
        '日本の首都はどこですか:tokyo' 'highest mountain in the world:summit'; do
        query="${expect%:*}"
        top="${expect##*:}"
        grep -A1 -xF "query: $query" "$work/sample-search.log" | grep -q "^  top: $top " || fail "sample: '$query' did not rank '$top' first"
    done
    grep -q 'max abs difference from the typed API 0$' "$work/sample-search.log" || fail "Extensions.AI path differs from the typed API"
    echo "ok:   sample ranks the expected passage first for all four queries; Extensions.AI vectors equal the typed API"
    tol="$src/$fixtures_rel/tolerances.yaml"
    read_float() { awk -v key="$1" '/^[^ ]/{section=$1} section=="float:" && $1==key":"{print $2}' "$tol"; }
    embedding_bound="$(read_float embedding_max_abs)"
    batch_bound="$(read_float padded_batch_vs_single_embedding_max_abs)"
    [[ -n "$embedding_bound" && -n "$batch_bound" ]] || fail "could not read the bounds from $fixtures_rel/tolerances.yaml"
    echo "bounds (protected $fixtures_rel/tolerances.yaml): float.embedding_max_abs $embedding_bound, float.padded_batch_vs_single_embedding_max_abs $batch_bound"
    offline "${checker[@]}" full "$models" "$work/fixtures" "$embedding_bound" "$batch_bound" | tee "$work/consumer-full.log" \
        || fail "checker failed"
    grep -qx 'RESULT: PASS' "$work/consumer-full.log" || fail "checker did not pass"

    step "8c. Offline: ML.NET checker on e5 (Starlings.Models.E5.MLNet)"
    offline "${mlnet_checker[@]}" e5 "$models" | tee "$work/consumer-mlnet-e5.log" || fail "ML.NET checker (e5) failed"
    grep -qx 'RESULT: PASS' "$work/consumer-mlnet-e5.log" || fail "ML.NET checker (e5) did not pass"
fi

if [[ -n "$gliner2_seed" && "$cuda_only" != 1 ]]; then
    step "9. GLiNER2: seeded graph, tokenizer.json from the Hub, offline extraction"
    target="$models/$gliner2_repo/$gliner2_commit/$(dirname "$gliner2_graph")"
    mkdir -p "$target"
    cp --reflink=auto "$gliner2_seed/$gliner2_graph" "$target/"
    echo "seeded: $gliner2_graph from $gliner2_seed (the store verifies size and SHA-256 before use)"
    "${gliner2_sample[@]}" download --cache "$models" | tee "$work/sample-gliner2-download.log" || fail "Gliner2Ner download failed"
    grep -q '^tokenizer.json: downloaded,' "$work/sample-gliner2-download.log" || fail "GLiNER2 tokenizer.json was not downloaded"
    grep -q "^$gliner2_graph: already cached," "$work/sample-gliner2-download.log" || fail "the seeded GLiNER2 graph was not verified from the cache"
    if offline curl -sS --max-time 5 -o /dev/null https://huggingface.co 2>/dev/null; then
        fail "the offline wrapper still reaches huggingface.co"
    fi
    offline "${gliner2_sample[@]}" extract --cache "$models" | tee "$work/sample-gliner2-extract.log" || fail "Gliner2Ner extract failed"
    for expect in 'organization: Apple \[0, 5)' 'person: Tim Cook \[10, 18)' 'location: Cupertino \[46, 55)' \
        'person: Marie Curie \[0, 11)' 'person: Pierre Curie \[' 'long input rejected: [0-9]+ tokens > 512'; do
        grep -Eq "^ *$expect" "$work/sample-gliner2-extract.log" || fail "Gliner2Ner: expected '$expect'"
    done
    echo "ok:   GLiNER2 sample extracts the expected entities offline and rejects the 600-word input"
elif [[ -z "$gliner2_seed" ]]; then
    echo "note: no --seed-gliner2-from: the GLiNER2 package was packed, inspected and its sample built, but not run"
fi

eg2_search_ffmpeg9=(dotnet "$apps/EmbeddingGemma2SearchFFmpeg9/bin/Release/net10.0/EmbeddingGemma2SearchFFmpeg9.dll")
# Checks one log of the sample's text-to-video search: <log> <FFmpeg major> <label> [<bound>]. The user's FFmpeg is the
# given major and LGPL, the expected clip ranks first for each query, frames without timing are rejected, and the
# Extensions.AI video path equals the typed API: exactly (CPU: one video per call and in a batch give the same bits) or,
# with <bound>, within it (CUDA: the generator's single video against the typed batch).
check_video_search() {
    local log="$1" major="$2" label="$3" bound="${4:-}"
    grep -Eq "^ffmpeg: n$major\\..* \\(LGPL\\)" "$log" || fail "$label: FFmpeg $major LGPL not reported"
    for expect in 'colour bars test pattern with a moving white square:h264-1fps-64x48-8s.mp4' \
        'colourful circles and rectangles moving over a dark background:h264-1.9s-320x240-30fps.mp4' \
        'white words scrolling across a colour gradient:h264-portrait-360x640-30fps-4s.mp4'; do
        query="${expect%:*}"
        top="${expect##*:}"
        grep -A1 -xF "video query: $query" "$log" | grep -q "^  top: $top " || fail "$label: '$query' did not rank '$top' first"
    done
    if [[ -z "$bound" ]]; then
        grep -q '^extensions.ai videos: .* max abs difference from the typed API 0$' "$log" || fail "$label: Extensions.AI video path differs from the typed API"
    else
        diff="$(sed -n 's/^extensions.ai videos: .* max abs difference from the typed API \(.*\)$/\1/p' "$log")"
        [[ -n "$diff" ]] && python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)" "$diff" "$bound" \
            || fail "$label: Extensions.AI video path differs from the typed API by '$diff' (bound $bound)"
    fi
    grep -q '^frames without timing rejected' "$log" || fail "$label: frames without timing were not rejected"
}
# A GPL (or nonfree) FFmpeg build is refused by the adapter without the explicit opt-in: <label> <lib dir> <sample...>.
check_gpl_refused() {
    local label="$1" dir="$2"
    shift 2
    set +e
    offline "$@" videos --cache "$models" --videos "$work/eg2-videos" --ffmpeg "$dir" >"$work/sample-$label-gpl.log" 2>&1
    local rc=$?
    set -e
    cat "$work/sample-$label-gpl.log"
    [[ "$rc" == 3 ]] && grep -q '^videos: The FFmpeg build at .* is not LGPL (' "$work/sample-$label-gpl.log" \
        || fail "$label: the GPL FFmpeg build at $dir was not refused (exit $rc)"
    echo "ok:   $label: the GPL FFmpeg build at $dir is refused without AllowNonLgplBuild (exit 3)"
}

if [[ -n "$eg2_seed" && "$cuda_only" != 1 ]]; then
    step "9b. EmbeddingGemma 2: seeded graph, tokenizer.json and prompt config from the Hub, offline search"
    target="$models/$eg2_repo/$eg2_commit/$(dirname "$eg2_graph")"
    mkdir -p "$target"
    cp --reflink=auto "$eg2_seed/$eg2_graph" "$target/"
    echo "seeded: $eg2_graph from $eg2_seed (the store verifies size and SHA-256 before use)"
    "${eg2_sample[@]}" download --cache "$models" | tee "$work/sample-eg2-download.log" || fail "EmbeddingGemma2Search download failed"
    grep -q '^tokenizer.json: downloaded,' "$work/sample-eg2-download.log" || fail "EmbeddingGemma 2 tokenizer.json was not downloaded"
    grep -q '^config_sentence_transformers.json: downloaded,' "$work/sample-eg2-download.log" || fail "config_sentence_transformers.json was not downloaded"
    grep -q "^$eg2_graph: already cached," "$work/sample-eg2-download.log" || fail "the seeded EmbeddingGemma 2 graph was not verified from the cache"
    if offline curl -sS --max-time 5 -o /dev/null https://huggingface.co 2>/dev/null; then
        fail "the offline wrapper still reaches huggingface.co"
    fi
    offline "${eg2_sample[@]}" search --cache "$models" | tee "$work/sample-eg2-search.log" || fail "EmbeddingGemma2Search search failed"
    for expect in 'how much protein should a female eat:protein' 'Warum rostet Eisen?:rust' \
        '日本の首都はどこですか:tokyo' 'highest mountain in the world:summit'; do
        query="${expect%:*}"
        top="${expect##*:}"
        grep -A1 -xF "query: $query" "$work/sample-eg2-search.log" | grep -q "^  top: $top " || fail "EmbeddingGemma2Search: '$query' did not rank '$top' first"
    done
    grep -q 'max abs difference from the typed API 0$' "$work/sample-eg2-search.log" || fail "EmbeddingGemma 2 Extensions.AI path differs from the typed API"
    grep -q '^matryoshka 256: 256 dimensions' "$work/sample-eg2-search.log" || fail "EmbeddingGemma 2 Matryoshka 256 missing"
    grep -Eq '^long input rejected: [0-9]+ tokens > 8192' "$work/sample-eg2-search.log" || fail "EmbeddingGemma 2 did not reject the long input"
    echo "ok:   EmbeddingGemma 2 sample ranks the expected document first offline, Extensions.AI equals the typed API, rejects > 8,192 tokens"
    step "9e. Offline: ML.NET checker on EmbeddingGemma 2 (Starlings.Models.EmbeddingGemma2.MLNet)"
    offline "${mlnet_checker[@]}" eg2 "$models" | tee "$work/consumer-mlnet-eg2.log" || fail "ML.NET checker (EmbeddingGemma 2) failed"
    grep -qx 'RESULT: PASS' "$work/consumer-mlnet-eg2.log" || fail "ML.NET checker (EmbeddingGemma 2) did not pass"
    if [[ -f "$eg2_seed/${eg2_image_graphs[0]}" && -f "$eg2_seed/${eg2_image_graphs[1]}" && -d "$eg2_seed/images" ]]; then
        step "9c. EmbeddingGemma 2 images: seeded vision and multimodal graphs, image lock verified, offline text-to-image search"
        for g in "${eg2_image_graphs[@]}"; do
            cp --reflink=auto "$eg2_seed/$g" "$target/"
        done
        mkdir -p "$work/eg2-images"
        for f in "${eg2_photos[@]}"; do cp "$eg2_seed/images/$f" "$work/eg2-images/"; done
        "${eg2_sample[@]}" download-images --cache "$models" | tee "$work/sample-eg2-download-images.log" || fail "EmbeddingGemma2Search download-images failed"
        for g in "${eg2_image_graphs[@]}" "$eg2_graph" tokenizer.json config_sentence_transformers.json; do
            grep -q "^$g: already cached," "$work/sample-eg2-download-images.log" || fail "$g was not verified from the cache"
        done
        offline "${eg2_sample[@]}" images --cache "$models" --images "$work/eg2-images" | tee "$work/sample-eg2-images.log" || fail "EmbeddingGemma2Search images failed"
        for expect in 'a photo of a cat:chelsea.png' 'a cup of coffee:coffee.png' 'a rocket launch:rocket.jpg' \
            'portrait of an astronaut:astronaut.png' 'deep space full of galaxies:hubble_deep_field.jpg'; do
            query="${expect%:*}"
            top="${expect##*:}"
            grep -A1 -xF "image query: $query" "$work/sample-eg2-images.log" | grep -q "^  top: $top " || fail "EmbeddingGemma2Search images: '$query' did not rank '$top' first"
        done
        grep -q '^extensions.ai images: .* max abs difference from the typed API 0$' "$work/sample-eg2-images.log" || fail "EmbeddingGemma 2 Extensions.AI image path differs from the typed API"
        grep -q '^interleaved: 768 dimensions' "$work/sample-eg2-images.log" || fail "EmbeddingGemma 2 interleaved input missing"
        echo "ok:   EmbeddingGemma 2 image search ranks the expected photo first offline; Extensions.AI images equal the typed API"
    else
        echo "note: $eg2_seed has no image graphs or images/: the image search was built but not run"
    fi
    if [[ -f "$eg2_seed/${eg2_image_graphs[0]}" && -d "$eg2_seed/videos" && -n "$ffmpeg_dir" ]]; then
        step "9d. EmbeddingGemma 2 videos: user-supplied FFmpeg 8 ($ffmpeg_dir), offline text-to-video search"
        for g in "${eg2_image_graphs[@]}"; do
            [[ -f "$target/$(basename "$g")" ]] || cp --reflink=auto "$eg2_seed/$g" "$target/"
        done
        mkdir -p "$work/eg2-videos"
        cp "$eg2_seed"/videos/*.mp4 "$eg2_seed"/videos/*.webm "$work/eg2-videos/" 2>/dev/null || true
        offline "${eg2_sample[@]}" videos --cache "$models" --videos "$work/eg2-videos" --ffmpeg "$ffmpeg_dir" | tee "$work/sample-eg2-videos.log" || fail "EmbeddingGemma2Search videos failed"
        check_video_search "$work/sample-eg2-videos.log" 8 "EmbeddingGemma2Search videos"
        echo "ok:   EmbeddingGemma 2 video search ranks the expected clip first offline on the user-supplied FFmpeg 8 (Starlings.Models.EmbeddingGemma2.Video.FFmpeg); Extensions.AI videos equal the typed API"
        if [[ -n "$ffmpeg9_dir" ]]; then
            step "9f. EmbeddingGemma 2 videos: user-supplied FFmpeg 9 ($ffmpeg9_dir), offline text-to-video search (EmbeddingGemma2SearchFFmpeg9)"
            offline "${eg2_search_ffmpeg9[@]}" videos --cache "$models" --videos "$work/eg2-videos" --ffmpeg "$ffmpeg9_dir" | tee "$work/sample-eg2-videos-ffmpeg9.log" || fail "EmbeddingGemma2SearchFFmpeg9 videos failed"
            check_video_search "$work/sample-eg2-videos-ffmpeg9.log" 9 "EmbeddingGemma2SearchFFmpeg9 videos"
            grep -q 'Starlings.Models.EmbeddingGemma2.Video.FFmpeg9 ' "$work/sample-eg2-videos-ffmpeg9.log" || fail "EmbeddingGemma2SearchFFmpeg9: the FFmpeg9 adapter was not loaded"
            echo "ok:   EmbeddingGemma 2 video search ranks the expected clip first offline on the user-supplied FFmpeg 9 (Starlings.Models.EmbeddingGemma2.Video.FFmpeg9); Extensions.AI videos equal the typed API"
        else
            echo "note: no --ffmpeg9-dir: the FFmpeg 9 sample was built but its video search not run"
        fi
        [[ -z "$ffmpeg_gpl_dir" ]] || check_gpl_refused eg2-videos-ffmpeg8 "$ffmpeg_gpl_dir" "${eg2_sample[@]}"
        [[ -z "$ffmpeg9_gpl_dir" ]] || check_gpl_refused eg2-videos-ffmpeg9 "$ffmpeg9_gpl_dir" "${eg2_search_ffmpeg9[@]}"
    else
        echo "note: no videos/ in $eg2_seed or no --ffmpeg-dir: the video search was built but not run"
    fi
elif [[ -z "$eg2_seed" ]]; then
    echo "note: no --seed-embeddinggemma2-from: the EmbeddingGemma 2 packages (including the ML.NET adapter) were packed, inspected and the sample and checker built, but not run"
fi

if [[ -n "$qwen_seed" && "$cuda_only" != 1 ]]; then
    step "10. Qwen3.5: seeded files verified by the download step, offline streamed answer"
    target="$models/$qwen_repo/$qwen_commit"
    mkdir -p "$target/ortgenai-fp32x" "$target/transformers-5.18.0"
    for f in genai_config.json model.onnx model.onnx.data; do
        cp --reflink=auto "$qwen_seed/ortgenai-fp32x/$f" "$target/ortgenai-fp32x/"
    done
    cp --reflink=auto "$qwen_seed/transformers-5.18.0/tokenizer.json" "$target/transformers-5.18.0/"
    cp --reflink=auto "$qwen_seed/chat_template.jinja" "$target/"
    echo "seeded: ortgenai-fp32x/, transformers-5.18.0/tokenizer.json and chat_template.jinja from $qwen_seed (the store verifies sizes and SHA-256)"
    "${qwen_sample[@]}" download --cache "$models" | tee "$work/sample-qwen-download.log" || fail "QwenChat download failed"
    grep -q '^chat_template.jinja: already cached,' "$work/sample-qwen-download.log" || fail "the seeded chat_template.jinja was not verified from the cache"
    grep -q '^ortgenai-fp32x/model.onnx.data: already cached,' "$work/sample-qwen-download.log" || fail "the seeded graph was not verified from the cache"
    if offline curl -sS --max-time 5 -o /dev/null https://huggingface.co 2>/dev/null; then
        fail "the offline wrapper still reaches huggingface.co"
    fi
    offline "${qwen_sample[@]}" ask --cache "$models" --max-new 64 "What is the capital of France? Answer in one sentence." \
        | tee "$work/sample-qwen-ask.log" || fail "QwenChat ask failed"
    grep -q '^assistant: .*Paris' "$work/sample-qwen-ask.log" || fail "QwenChat: expected Paris in the answer"
    grep -Eq '^\[Stop; 24 prompt \+ [0-9]+ completion tokens' "$work/sample-qwen-ask.log" || fail "QwenChat: expected a Stop finish with 24 prompt tokens"
    if grep -q 'im_end' "$work/sample-qwen-ask.log"; then
        fail "QwenChat: the stop token leaked into the text"
    fi
    echo "ok:   QwenChat answers offline from the packages, stops at <|im_end|> without printing it, reports usage"
    offline "${qwen_sample[@]}" ask --cache "$models" --max-new 64 --chat-client "What is the capital of France? Answer in one sentence." \
        | tee "$work/sample-qwen-ask-chat-client.log" || fail "QwenChat ask --chat-client failed"
    grep -Eq '^\[Stop; 24 prompt \+ [0-9]+ completion tokens; IChatClient\]' "$work/sample-qwen-ask-chat-client.log" \
        || fail "QwenChat --chat-client: expected a Stop finish with 24 prompt tokens"
    [[ "$(grep '^assistant: ' "$work/sample-qwen-ask.log")" == "$(grep '^assistant: ' "$work/sample-qwen-ask-chat-client.log")" ]] \
        || fail "QwenChat --chat-client: the IChatClient answer differs from the typed API's"
    [[ "$(grep -Eo '24 prompt \+ [0-9]+ completion' "$work/sample-qwen-ask.log")" == "$(grep -Eo '24 prompt \+ [0-9]+ completion' "$work/sample-qwen-ask-chat-client.log")" ]] \
        || fail "QwenChat --chat-client: usage differs from the typed API's"
    echo "ok:   the IChatClient adapter streams the same answer, finish reason and usage"
elif [[ -z "$qwen_seed" ]]; then
    echo "note: no --seed-qwen35-from: the pilot LLM packages were packed, inspected and the QwenChat sample built, but not run"
fi

step "11. Download from public hosting (no token, no seed)"
# Each entry: label|source file in the clone|C# constant that records the hosting as private or a placeholder.
hosting=(
    "GLiNER2|ai-dotnet/src/Starlings.Models.Gliner2/Gliner2ModelAssets.cs|ModelSourceIsPlaceholder = true"
    "EmbeddingGemma 2|ai-dotnet/src/Starlings.Models.EmbeddingGemma2/EmbeddingGemma2ModelAssets.cs|ModelSourceIsPlaceholder = true"
    "EmbeddingGemma 2 images|ai-dotnet/src/Starlings.Models.EmbeddingGemma2/EmbeddingGemma2ModelAssets.cs|ImageModelSourceIsPlaceholder = true"
    "EmbeddingGemma 2 videos|ai-dotnet/src/Starlings.Models.EmbeddingGemma2/EmbeddingGemma2ModelAssets.cs|ImageModelSourceIsPlaceholder = true"
    "Qwen 3.5|ai-dotnet/src/Starlings.Models.Qwen35.OnnxGenAI/Qwen35ModelAssets.cs|HostingIsPrivate = true"
    "Qwen 3.5 fp16x|ai-dotnet/src/Starlings.Models.Qwen35.OnnxGenAI/Qwen35ModelAssets.cs|HostingIsPrivate = true"
)
# The protected greedy reference of the drill's question (Qwen 3.5 reference fixtures, row en-short): the answer text
# without the stop token, and the completion token count without it.
read -r qwen_ref_completion qwen_ref_text < <(python3 - "$src/model-pipeline/protected/qwen3.5-0.8b/reference/generation.jsonl" <<'PY2'
import json, sys
row = next(r for r in map(json.loads, open(sys.argv[1], encoding="utf-8")) if r["id"] == "en-short")
assert row["messages"] == [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}], row["messages"]
text = row["output_text"].removesuffix("<|im_end|>")
print(len(row["output_ids"]) - (1 if row["finish_reason"] == "eos" else 0), text)
PY2
) || fail "cannot read the Qwen 3.5 reference row en-short"
[[ -n "$qwen_ref_text" ]] || fail "cannot read the Qwen 3.5 reference row en-short"
# Checks one QwenChat log against that reference: the same text, a Stop finish, 24 prompt tokens and the same count.
check_qwen_reference() {
    grep -qxF "assistant: $qwen_ref_text" "$1" || fail "$2: the answer is not the protected reference '$qwen_ref_text'"
    grep -Eq "^\[Stop; 24 prompt \+ $qwen_ref_completion completion tokens" "$1" || fail "$2: expected Stop with 24 prompt + $qwen_ref_completion completion tokens"
    if grep -q 'im_end' "$1"; then
        fail "$2: the stop token leaked into the text"
    fi
}
qwen_question="What is the capital of France? Answer in one sentence."
cuda_env=()
if [[ -n "$cuda_device" ]]; then
    # One GPU, chosen by the caller; the CUDA runtime, cuBLAS and cuDNN of the machine on the loader path for CUDA runs only.
    cuda_env=(env CUDA_DEVICE_ORDER=PCI_BUS_ID "CUDA_VISIBLE_DEVICES=$cuda_device" "LD_LIBRARY_PATH=${cuda_lib_path}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}")
    cuda_checker=(dotnet "$apps/CudaConsumer/bin/Release/net10.0/CudaConsumer.dll")
    qwen_cuda=(dotnet "$apps/QwenChatCuda/bin/Release/net10.0/QwenChatCuda.dll")
fi
eg2_images_all="$work/eg2-images-all"
if [[ -n "$eg2_seed" && -d "$eg2_seed/images" ]]; then
    # The whole make_eg2_images.py set with its images.json (the CUDA checker reads it); the search uses PNG and JPEG.
    cp -r "$eg2_seed/images" "$eg2_images_all"
fi
public_ran=""
if [[ "$public_hosting" != 1 ]]; then
    echo "skipped: not requested (pass --public-hosting to download the GLiNER2, EmbeddingGemma 2 and Qwen 3.5 files, about 4 GB, from their public hosting)"
else
    public_models="$work/models-public"
    mkdir -p "$public_models"
    [[ -z "${HF_TOKEN:-}" ]] || fail "HF_TOKEN is set inside the drill"
    for entry in "${hosting[@]}"; do
        IFS='|' read -r label file marker <<<"$entry"
        if [[ -n "$public_only" && ",$public_only," != *",$label,"* ]]; then
            echo "skipped: $label not in --public-hosting-only $public_only"
            continue
        fi
        if grep -qF "$marker" "$src/$file"; then
            echo "skipped: hosting private ($label: $file still records $marker)"
            continue
        fi
        case "$label" in
            GLiNER2)
                "${gliner2_sample[@]}" download --cache "$public_models" | tee "$work/public-gliner2-download.log" || fail "GLiNER2 public download failed"
                grep -q "^$gliner2_graph: downloaded," "$work/public-gliner2-download.log" || fail "GLiNER2 graph was not downloaded"
                offline "${gliner2_sample[@]}" extract --cache "$public_models" | tee "$work/public-gliner2-extract.log" || fail "GLiNER2 extract from public files failed"
                grep -Eq '^ *person: Tim Cook \[10, 18\)' "$work/public-gliner2-extract.log" || fail "GLiNER2 from public files: expected 'person: Tim Cook [10, 18)'"
                ;;
            "EmbeddingGemma 2")
                "${eg2_sample[@]}" download --cache "$public_models" | tee "$work/public-eg2-download.log" || fail "EmbeddingGemma 2 public download failed"
                grep -q "^$eg2_graph: downloaded," "$work/public-eg2-download.log" || fail "EmbeddingGemma 2 graph was not downloaded"
                offline "${eg2_sample[@]}" search --cache "$public_models" | tee "$work/public-eg2-search.log" || fail "EmbeddingGemma 2 search from public files failed"
                grep -A1 -xF "query: how much protein should a female eat" "$work/public-eg2-search.log" | grep -q "^  top: protein " || fail "EmbeddingGemma 2 from public files: expected 'protein' first"
                ;;
            "EmbeddingGemma 2 images")
                "${eg2_sample[@]}" download-images --cache "$public_models" | tee "$work/public-eg2-download-images.log" || fail "EmbeddingGemma 2 image graphs public download failed"
                for g in "${eg2_image_graphs[@]}"; do
                    grep -q "^$g: downloaded," "$work/public-eg2-download-images.log" || fail "$g was not downloaded"
                done
                for g in "$eg2_graph" tokenizer.json config_sentence_transformers.json; do
                    grep -Eq "^$g: (downloaded|already cached)," "$work/public-eg2-download-images.log" || fail "$g was not verified"
                done
                if [[ -d "$eg2_images_all" ]]; then
                    mkdir -p "$work/public-eg2-images"
                    for f in "${eg2_photos[@]}"; do cp "$eg2_images_all/$f" "$work/public-eg2-images/"; done
                    offline "${eg2_sample[@]}" images --cache "$public_models" --images "$work/public-eg2-images" | tee "$work/public-eg2-images.log" || fail "EmbeddingGemma 2 image search from public files failed"
                    for expect in 'a photo of a cat:chelsea.png' 'a cup of coffee:coffee.png' 'a rocket launch:rocket.jpg' \
                        'portrait of an astronaut:astronaut.png' 'deep space full of galaxies:hubble_deep_field.jpg'; do
                        query="${expect%:*}"
                        top="${expect##*:}"
                        grep -A1 -xF "image query: $query" "$work/public-eg2-images.log" | grep -q "^  top: $top " || fail "EmbeddingGemma 2 images from public files: '$query' did not rank '$top' first"
                    done
                    grep -q '^extensions.ai images: .* max abs difference from the typed API 0$' "$work/public-eg2-images.log" || fail "EmbeddingGemma 2 Extensions.AI image path from public files differs from the typed API"
                else
                    echo "note: no images/ in the EmbeddingGemma 2 seed: the image graphs were downloaded and verified but the image search was not run"
                fi
                ;;
            "EmbeddingGemma 2 videos")
                # The video path runs on the image lock's vision and multimodal graphs: downloaded here with no token, or
                # already downloaded (and verified again) by the "EmbeddingGemma 2 images" entry into the same cache.
                "${eg2_sample[@]}" download-images --cache "$public_models" | tee "$work/public-eg2-download-videos.log" || fail "EmbeddingGemma 2 video graphs public download failed"
                for g in "${eg2_image_graphs[@]}" "$eg2_graph" tokenizer.json config_sentence_transformers.json; do
                    grep -Eq "^$g: (downloaded|already cached)," "$work/public-eg2-download-videos.log" || fail "$g was not verified"
                done
                if [[ -n "$eg2_seed" && -d "$eg2_seed/videos" && -n "$ffmpeg_dir" ]]; then
                    mkdir -p "$work/public-eg2-videos"
                    cp "$eg2_seed"/videos/*.mp4 "$eg2_seed"/videos/*.webm "$work/public-eg2-videos/" 2>/dev/null || true
                    offline "${eg2_sample[@]}" videos --cache "$public_models" --videos "$work/public-eg2-videos" --ffmpeg "$ffmpeg_dir" | tee "$work/public-eg2-videos.log" || fail "EmbeddingGemma 2 video search from public files failed"
                    check_video_search "$work/public-eg2-videos.log" 8 "EmbeddingGemma 2 videos from public files"
                else
                    echo "note: no videos/ in the EmbeddingGemma 2 seed or no --ffmpeg-dir: the video path's graphs were downloaded and verified but the video search was not run"
                fi
                ;;
            "Qwen 3.5 fp16x")
                "${qwen_sample[@]}" download --variant fp16x --cache "$public_models" | tee "$work/public-qwen-fp16x-download.log" || fail "Qwen 3.5 fp16x public download failed"
                grep -q '^ortgenai-fp16x/model.onnx.data: downloaded,' "$work/public-qwen-fp16x-download.log" || fail "Qwen 3.5 fp16x graph was not downloaded"
                if [[ -n "$cuda_device" ]]; then
                    offline "${cuda_env[@]}" "${qwen_cuda[@]}" ask --cache "$public_models" --variant fp16x --cuda --max-new 64 "$qwen_question" \
                        | tee "$work/public-qwen-fp16x-ask-cuda.log" || fail "QwenChatCuda fp16x from public files failed"
                    check_qwen_reference "$work/public-qwen-fp16x-ask-cuda.log" "QwenChatCuda fp16x from public files"
                else
                    echo "note: fp16x runs on CUDA only (pass --cuda): downloaded and verified, not run"
                fi
                ;;
            "Qwen 3.5")
                "${qwen_sample[@]}" download --cache "$public_models" | tee "$work/public-qwen-download.log" || fail "Qwen 3.5 public download failed"
                grep -q '^ortgenai-fp32x/model.onnx.data: downloaded,' "$work/public-qwen-download.log" || fail "Qwen 3.5 graph was not downloaded"
                offline "${qwen_sample[@]}" ask --cache "$public_models" --max-new 64 "What is the capital of France? Answer in one sentence." \
                    | tee "$work/public-qwen-ask.log" || fail "QwenChat from public files failed"
                grep -q '^assistant: .*Paris' "$work/public-qwen-ask.log" || fail "QwenChat from public files: expected Paris"
                ;;
        esac
        echo "ok:   $label downloaded from public hosting with no token and no seed, and ran offline from that cache"
        public_ran="$public_ran${public_ran:+, }$label"
    done
fi

if [[ "$aot" == 1 && "$cuda_only" != 1 ]]; then
    step "12. Native AOT check (tools/aot-check-ai.sh on the drill feed and model cache)"
    aot_download=()
    [[ "$mode" == "no-model" ]] || aot_download=(--download)
    "$src/tools/aot-check-ai.sh" --feed "$feed" --models "$models" --out "$work/aot" "${aot_download[@]}" | tee "$work/aot-check.log" \
        || fail "the Native AOT check failed"
else
    echo "note: no --aot: the Native AOT check was not run"
fi
if [[ ( "$aot" == 1 || "$aot_smoke" == 1 ) && "$cuda_only" != 1 ]]; then
    step "12b. Native AOT smoke check of every package (tools/aot-smoke-ai.sh on the drill feed; no model files)"
    "$src/tools/aot-smoke-ai.sh" --feed "$feed" --out "$work/aot-smoke" | tee "$work/aot-smoke.log" \
        || fail "the Native AOT smoke check failed"
else
    echo "note: no --aot or --aot-smoke: the Native AOT smoke check was not run"
fi

cuda_ran=""
if [[ -n "$cuda_device" ]]; then
    step "13. CUDA profiles on device $cuda_device: seeded files verified by the packages' download steps, offline runs on the GPU"
    cuda_models="$work/models-cuda"
    # Seeds: the same local files as the CPU steps, into a new isolated cache; each package's download step verifies
    # size and SHA-256 of every file and downloads the small files (tokenizers, prompt config) from their pinned hosting.
    if [[ "$mode" == "seed" ]]; then
        mkdir -p "$cuda_models/$e5_repo/$e5_commit/onnx"
        cp --reflink=auto "$seed/onnx/model.onnx" "$cuda_models/$e5_repo/$e5_commit/onnx/"
    fi
    "${sample[@]}" download --cache "$cuda_models" | tee "$work/cuda-e5-download.log" || fail "E5Search download (CUDA cache) failed"
    mkdir -p "$cuda_models/$gliner2_repo/$gliner2_commit/onnx"
    cp --reflink=auto "$gliner2_seed/$gliner2_graph" "$cuda_models/$gliner2_repo/$gliner2_commit/onnx/"
    "${gliner2_sample[@]}" download --cache "$cuda_models" | tee "$work/cuda-gliner2-download.log" || fail "Gliner2Ner download (CUDA cache) failed"
    grep -q "^$gliner2_graph: already cached," "$work/cuda-gliner2-download.log" || fail "the seeded GLiNER2 graph was not verified from the CUDA cache"
    mkdir -p "$cuda_models/$eg2_repo/$eg2_commit/onnx"
    for g in "$eg2_graph" "${eg2_image_graphs[@]}"; do
        cp --reflink=auto "$eg2_seed/$g" "$cuda_models/$eg2_repo/$eg2_commit/onnx/"
    done
    "${eg2_sample[@]}" download-images --cache "$cuda_models" | tee "$work/cuda-eg2-download-images.log" || fail "EmbeddingGemma2Search download-images (CUDA cache) failed"
    for g in "$eg2_graph" "${eg2_image_graphs[@]}"; do
        grep -q "^$g: already cached," "$work/cuda-eg2-download-images.log" || fail "the seeded $g was not verified from the CUDA cache"
    done
    qwen_target="$cuda_models/$qwen_repo/$qwen_commit"
    mkdir -p "$qwen_target/transformers-5.18.0"
    for v in fp32x fp16x; do
        mkdir -p "$qwen_target/ortgenai-$v"
        for f in genai_config.json model.onnx model.onnx.data; do
            cp --reflink=auto "$qwen_seed/ortgenai-$v/$f" "$qwen_target/ortgenai-$v/"
        done
    done
    cp --reflink=auto "$qwen_seed/transformers-5.18.0/tokenizer.json" "$qwen_target/transformers-5.18.0/"
    cp --reflink=auto "$qwen_seed/chat_template.jinja" "$qwen_target/"
    for v in fp32x fp16x; do
        "${qwen_sample[@]}" download --variant "$v" --cache "$cuda_models" | tee "$work/cuda-qwen-$v-download.log" || fail "QwenChat download --variant $v (CUDA cache) failed"
        grep -q "^ortgenai-$v/model.onnx.data: already cached," "$work/cuda-qwen-$v-download.log" || fail "the seeded Qwen 3.5 $v graph was not verified from the CUDA cache"
    done
    if offline curl -sS --max-time 5 -o /dev/null https://huggingface.co 2>/dev/null; then
        fail "the offline wrapper still reaches huggingface.co"
    fi
    protected="$src/model-pipeline/protected"
    # profile|checker mode|fixture directory|protected tolerances of the CUDA profile
    cuda_profiles=(
        "multilingual-e5-small/onnx-fp32-cuda|e5|multilingual-e5-small/onnx-fp32-cpu|multilingual-e5-small/onnx-fp32-cuda/tolerances.yaml"
        "gliner2-base-v1/onnx-fp32-cuda|gliner2|gliner2-base-v1/onnx-fp32-cpu|gliner2-base-v1/onnx-fp32-cuda/tolerances.yaml"
        "embeddinggemma-2/onnx-fp32-cuda|eg2-text|embeddinggemma-2/onnx-fp32-cpu|embeddinggemma-2/onnx-fp32-cuda/tolerances.yaml"
        "embeddinggemma-2/image-fp32-cuda|eg2-image|embeddinggemma-2/image-fp32-cpu|embeddinggemma-2/image-fp32-cuda/tolerances.yaml"
    )
    for entry in "${cuda_profiles[@]}"; do
        IFS='|' read -r profile checker_mode fixture_rel tolerances_rel <<<"$entry"
        extra=()
        [[ "$checker_mode" != eg2-image ]] || extra=("$eg2_images_all")
        offline "${cuda_env[@]}" "${cuda_checker[@]}" "$checker_mode" "$cuda_models" "$protected/$fixture_rel" "$protected/$tolerances_rel" "${extra[@]}" \
            | tee "$work/cuda-$checker_mode.log" || fail "CudaConsumer $checker_mode ($profile) failed"
        grep -qx 'RESULT: PASS' "$work/cuda-$checker_mode.log" || fail "CudaConsumer $checker_mode ($profile) did not pass"
        echo "ok:   $profile: offline on CUDA from the packed packages; outputs within the protected bounds of $tolerances_rel"
        cuda_ran="$cuda_ran${cuda_ran:+, }$profile"
    done
    if [[ -d "$eg2_seed/video-fixtures" && -d "$eg2_seed/videos" && -n "$ffmpeg_dir" ]]; then
        # Video on CUDA: the fixture videos through each adapter on the user's FFmpeg, then the sample's video search.
        profile="embeddinggemma-2/video-fp32-cuda"
        video_tolerances="$protected/embeddinggemma-2/video-fp32-cuda/tolerances.yaml"
        for major in 8 9; do
            dir="$ffmpeg_dir"
            if [[ "$major" == 9 ]]; then
                [[ -n "$ffmpeg9_dir" ]] || { echo "note: no --ffmpeg9-dir: the FFmpeg 9 adapter was not run on CUDA"; continue; }
                dir="$ffmpeg9_dir"
            fi
            offline "${cuda_env[@]}" "${cuda_checker[@]}" eg2-video "$cuda_models" "$protected/embeddinggemma-2/video-fp32-cpu" "$video_tolerances" \
                "$eg2_seed/video-fixtures" "$dir" "$major" | tee "$work/cuda-eg2-video-ffmpeg$major.log" || fail "CudaConsumer eg2-video FFmpeg $major ($profile) failed"
            grep -qx 'RESULT: PASS' "$work/cuda-eg2-video-ffmpeg$major.log" || fail "CudaConsumer eg2-video FFmpeg $major ($profile) did not pass"
            echo "ok:   $profile, FFmpeg $major adapter: offline on CUDA from the packed packages; frames exact and embeddings within the protected bounds"
        done
        eg2_search_cuda=(dotnet "$apps/EmbeddingGemma2SearchCuda/bin/Release/net10.0/EmbeddingGemma2SearchCuda.dll")
        mkdir -p "$work/cuda-eg2-videos"
        cp "$eg2_seed"/videos/*.mp4 "$eg2_seed"/videos/*.webm "$work/cuda-eg2-videos/" 2>/dev/null || true
        offline "${cuda_env[@]}" "${eg2_search_cuda[@]}" videos --cuda --cache "$cuda_models" --videos "$work/cuda-eg2-videos" --ffmpeg "$ffmpeg_dir" \
            | tee "$work/cuda-eg2-search-videos.log" || fail "EmbeddingGemma2SearchCuda videos --cuda failed"
        for s in vision "multimodal text"; do
            grep -Eq "^session $s: providers CUDAExecutionProvider.*; options .*device_id=0.*use_tf32=0" "$work/cuda-eg2-search-videos.log" \
                || fail "EmbeddingGemma2SearchCuda: the $s session does not run on CUDAExecutionProvider with device_id=0 and use_tf32=0"
        done
        grep -q "^onnxruntime: native $ort_version, managed $ort_version" "$work/cuda-eg2-search-videos.log" || fail "EmbeddingGemma2SearchCuda: ONNX Runtime $ort_version not reported"
        batch_bound="$(awk '/^[^ ]/{section=$1} section=="float:" && $1=="padded_batch_vs_single_embedding_max_abs:"{print $2}' "$video_tolerances")"
        [[ -n "$batch_bound" ]] || fail "cannot read float.padded_batch_vs_single_embedding_max_abs from $video_tolerances"
        check_video_search "$work/cuda-eg2-search-videos.log" 8 "EmbeddingGemma2SearchCuda videos" "$batch_bound"
        echo "ok:   $profile: the sample's video search on CUDA from the packed packages; Extensions.AI within $batch_bound of the typed batch"
        cuda_ran="$cuda_ran${cuda_ran:+, }$profile"
    else
        echo "note: no video-fixtures/ or videos/ in $eg2_seed, or no --ffmpeg-dir: embeddinggemma-2/video-fp32-cuda was built but not run"
    fi
    for v in fp32x fp16x; do
        profile="qwen3.5-0.8b/ortgenai-$v-cuda"
        offline "${cuda_env[@]}" "${qwen_cuda[@]}" ask --cache "$cuda_models" --variant "$v" --cuda --max-new 64 "$qwen_question" \
            | tee "$work/cuda-qwen-$v-ask.log" || fail "QwenChatCuda $v failed"
        grep -Eq "^model: .* on Cuda, ONNX Runtime GenAI $genai_version$" "$work/cuda-qwen-$v-ask.log" \
            || fail "QwenChatCuda $v: expected the model on Cuda with ONNX Runtime GenAI $genai_version"
        grep -q "^model: .*$v" "$work/cuda-qwen-$v-ask.log" || fail "QwenChatCuda $v: the loaded profile is not $v"
        grep -Eq "^packages: Starlings.Models.Qwen35 $ai_version, Starlings.Generation.OnnxGenAI $ai_version, Starlings.Tokenizers $tok_version, .*; GenAI managed $genai_version, native Microsoft.ML.OnnxRuntimeGenAI.Cuda/$genai_version$" "$work/cuda-qwen-$v-ask.log" \
            || fail "QwenChatCuda $v: expected the packed packages and Microsoft.ML.OnnxRuntimeGenAI.Cuda $genai_version in the diagnostics"
        check_qwen_reference "$work/cuda-qwen-$v-ask.log" "QwenChatCuda $v"
        offline "${cuda_env[@]}" "${qwen_cuda[@]}" ask --cache "$cuda_models" --variant "$v" --cuda --max-new 64 --chat-client "$qwen_question" \
            | tee "$work/cuda-qwen-$v-ask-chat-client.log" || fail "QwenChatCuda $v --chat-client failed"
        grep -Eq "^\[Stop; 24 prompt \+ $qwen_ref_completion completion tokens; IChatClient\]" "$work/cuda-qwen-$v-ask-chat-client.log" \
            || fail "QwenChatCuda $v --chat-client: expected Stop with 24 prompt + $qwen_ref_completion completion tokens"
        [[ "$(grep '^assistant: ' "$work/cuda-qwen-$v-ask.log")" == "$(grep '^assistant: ' "$work/cuda-qwen-$v-ask-chat-client.log")" ]] \
            || fail "QwenChatCuda $v --chat-client: the IChatClient answer differs from the typed API's"
        echo "ok:   $profile: offline on CUDA from the packed packages; the protected reference answer through the typed API and IChatClient"
        cuda_ran="$cuda_ran${cuda_ran:+, }$profile"
    done
fi

out="$root/artifacts/release-drill-ai"
rm -rf "$out"
mkdir -p "$out"
cp "$pack"/*.nupkg "$pack"/*.snupkg "$work"/*-packages.txt "$out/"
cp "$work"/sample-*.log "$work"/consumer-*.log "$work"/public-*.log "$work"/cuda-*.log "$work"/aot-check.log "$work"/aot-smoke.log "$work"/native-check-*.log "$out/" 2>/dev/null || true
if [[ -d "$work/aot" ]]; then cp -r "$work/aot" "$out/aot"; fi
if [[ -d "$work/aot-smoke" ]]; then cp -r "$work/aot-smoke" "$out/aot-smoke"; fi
aot_label=""
[[ "$aot" != 1 ]] || aot_label=", AOT check"
cpu_label="$mode${gliner2_seed:+, GLiNER2 run}${eg2_seed:+, EmbeddingGemma 2 run}${ffmpeg_dir:+ with FFmpeg $ffmpeg_dir}${ffmpeg9_dir:+ and FFmpeg 9 $ffmpeg9_dir}${qwen_seed:+, Qwen3.5 run}"
[[ "$cuda_only" != 1 ]] || cpu_label="CPU runs skipped (--cuda-only)"
step "DRILL PASS ($cpu_label${public_ran:+, public hosting: $public_ran}$aot_label${cuda_ran:+, CUDA: $cuda_ran}): Starlings $ai_version + Tokenizers $tok_version from $sha (packages and logs in artifacts/release-drill-ai/)"
