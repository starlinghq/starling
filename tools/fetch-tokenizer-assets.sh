#!/usr/bin/env bash
# Downloads the pinned tokenizer.json assets used by the ai-dotnet model tests
# and verifies their SHA-256. Assets are model-owner files and never committed (plan section 9).
#
#   tools/fetch-tokenizer-assets.sh            # into <repo>/.models (ignored by Git)
#   STARLING_MODELS=/data/models tools/fetch-tokenizer-assets.sh
#
# Layout: $STARLING_MODELS/<hub repo>/<full commit SHA>/<file>. Revisions and digests come from
# docs/m0/hub-pins.md and model-pipeline/candidates/*/profile.json, matching the tests' *TestAssets.cs files.
# A file already present with the right digest is not downloaded again.
#
# Derived assets are runtime pipelines that no Hub file holds. The qwen35-bpe-chat profile's
# tokenizer.json is the pipeline transformers 5.18.0 builds for Qwen/Qwen3.5-0.8B at 2fc06364
# (Qwen3_5Tokenizer, serialized with backend_tokenizer.to_str(pretty=True); reproduced by
# model-pipeline/reference/qwen-bpe/run_qwen_bpe_tokenizer.py --write-live). It differs from the
# Kev-0.8B tokenizer.json only in the Split regex (\p{M} added), so it is derived from that file by
# replacing the regex line and accepted only if its SHA-256 matches the pinned digest.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
dest="${STARLING_MODELS:-$root/.models}"

# repo|commit|path|sha256
assets=(
    "intfloat/multilingual-e5-small|614241f622f53c4eeff9890bdc4f31cfecc418b3|tokenizer.json|0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"
    "fastino/gliner2-base-v1|f9634218e53580c56edf0de97ca1a7d3f1c2354e|tokenizer.json|1b7fbabfb4c690bed84c6793bfecae9b8dfe205751b04f9ffd1e76a1e7df9c16"
    "jaredpalmer/kev-0.8b|9a45d25eb2ab761841196625383fa1dff0e56c1e|tokenizer.json|06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
    "jaredpalmer/kev-4b|139fdd94f1b6a6ad80cc15e08fcb99cac885a101|tokenizer.json|06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
    "google/embeddinggemma-2|914f7f89142e33e77833254d9c9b90c3cef7303b|tokenizer.json|4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"
    "sentence-transformers/all-MiniLM-L6-v2|1110a243fdf4706b3f48f1d95db1a4f5529b4d41|tokenizer.json|be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"
    "BAAI/bge-small-en-v1.5|5c38ec7c405ec4b44b94cc5a9bb96e735b38267a|tokenizer.json|d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
    "cross-encoder/ms-marco-MiniLM-L6-v2|233902d25c440f23af6f7d6e94d2946bac0bee0a|tokenizer.json|d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
    "nomic-ai/nomic-embed-text-v1.5|e9b6763023c676ca8431644204f50c2b100d9aab|tokenizer.json|d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
    "BAAI/bge-base-en-v1.5|a5beb1e3e68b9ab74eb54cfd186867f64f240e1a|tokenizer.json|d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
    "BAAI/bge-m3|5617a9f61b028005a4858fdac845db406aefb181|tokenizer.json|21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08"
    "Starlinghq/bge-reranker-v2-m3-onnx|047d4622e93d347828457e8a13bccd997fb6a969|tokenizer.json|69564b696052886ed0ac63fa393e928384e0f8caada38c1f4864a9bfbf379c15"
    "Qwen/Qwen3-0.6B|c1899de289a04d12100db370d81485cdf75e47ca|tokenizer.json|aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
    "Qwen/Qwen2.5-0.5B-Instruct|7ae557604adf67be50417f59c2c2f167def9a775|tokenizer.json|c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
    "Diffuspace/Clover-35B-A3B-Preview|4b0402b65e523d0183a14c431955438714cb16ff|tokenizer.json|7c62ba7d8bb13291c94459896cd27537858beb58a88a037b09a150e213420988"
)

# target repo|commit|path|sha256 <- source repo|commit|path (the source must be in assets above)
derived=(
    "Qwen/Qwen3.5-0.8B|2fc06364715b967f1860aea9cf38778875588b17|transformers-5.18.0/tokenizer.json|a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f|jaredpalmer/kev-0.8b|9a45d25eb2ab761841196625383fa1dff0e56c1e|tokenizer.json"
)
# The part of the Split regex (JSON text, exactly once in the source) that differs between the two.
qwen2_regex_part='[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*'
qwen35_regex_part='[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*'


sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}

status=0
for entry in "${assets[@]}"; do
    IFS='|' read -r repo commit path digest <<<"$entry"
    target="$dest/$repo/$commit/$path"
    if [[ -f "$target" && "$(sha256 "$target")" == "$digest" ]]; then
        echo "ok:   $repo@$commit/$path (present)"
        continue
    fi
    mkdir -p "$(dirname "$target")"
    tmp="$target.partial"
    curl --fail --silent --show-error --location --retry 3 \
        -o "$tmp" "https://huggingface.co/$repo/resolve/$commit/$path"
    actual="$(sha256 "$tmp")"
    if [[ "$actual" != "$digest" ]]; then
        echo "FAIL: $repo@$commit/$path sha256 $actual, expected $digest" >&2
        rm -f "$tmp"
        status=1
        continue
    fi
    mv "$tmp" "$target"
    echo "ok:   $repo@$commit/$path (downloaded)"
done

for entry in "${derived[@]}"; do
    IFS='|' read -r repo commit path digest src_repo src_commit src_path <<<"$entry"
    target="$dest/$repo/$commit/$path"
    if [[ -f "$target" && "$(sha256 "$target")" == "$digest" ]]; then
        echo "ok:   $repo@$commit/$path (present)"
        continue
    fi
    source="$dest/$src_repo/$src_commit/$src_path"
    if [[ ! -f "$source" ]]; then
        echo "FAIL: $repo@$commit/$path needs $src_repo@$src_commit/$src_path" >&2
        status=1
        continue
    fi
    mkdir -p "$(dirname "$target")"
    tmp="$target.partial"
    FROM="$qwen2_regex_part" TO="$qwen35_regex_part" \
        perl -0777 -pe 'my $n = s/\Q$ENV{FROM}\E/$ENV{TO}/g; die "expected one regex, found $n\n" if $n != 1' \
        "$source" >"$tmp"
    actual="$(sha256 "$tmp")"
    if [[ "$actual" != "$digest" ]]; then
        echo "FAIL: $repo@$commit/$path sha256 $actual, expected $digest" >&2
        rm -f "$tmp"
        status=1
        continue
    fi
    mv "$tmp" "$target"
    echo "ok:   $repo@$commit/$path (derived from $src_repo@$src_commit/$src_path)"
done

exit "$status"
