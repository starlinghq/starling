#!/usr/bin/env bash
# Downloads and verifies the e5 runtime files (onnx/model.onnx and tokenizer.json of
# intfloat/multilingual-e5-small at the pinned commit) with the acquisition library (P-009).
# Thin wrapper over:
#
#   dotnet run --project ai-dotnet/tools/Starlings.Acquisition.Tool -- download multilingual-e5-small --cache <dir>
#
#   ai-dotnet/tools/fetch-e5-model.sh [cache-dir]
#
# The cache directory defaults to $STARLING_MODELS, else <repo>/.models (ignored by Git), which is where the tests look.
# Files land at <cache-dir>/intfloat/multilingual-e5-small/<commit>/. Files already present are verified, not
# downloaded again; an interrupted download resumes. Exits 1 on a digest mismatch, a refused or missing file, or a
# network failure. The lock (sizes and SHA-256) is E5ModelAssets.Lock, checked against the registry manifests by tests.
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cache="${1:-${STARLING_MODELS:-$root/.models}}"
# The SDK from $DOTNET_ROOT when set, else the one on PATH.
if [ -n "${DOTNET_ROOT:-}" ] && [ -x "$DOTNET_ROOT/dotnet" ]; then
    dotnet="$DOTNET_ROOT/dotnet"
else
    dotnet="dotnet"
fi

exec "$dotnet" run --project "$root/ai-dotnet/tools/Starlings.Acquisition.Tool" -- download multilingual-e5-small --cache "$cache"
