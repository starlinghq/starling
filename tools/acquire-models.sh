#!/usr/bin/env bash
# Acquire the pinned model files of registry profiles for the scheduled qualification (.github/workflows/qualification.yml).
#
#   tools/acquire-models.sh <locks dir> <profile>...
#
# For each profile with public hosting: writes its lock from the manifest's origin.files (plus the files the protected
# reference run recorded, model-pipeline/protected/<profile>/environment.json, when present) with
# tools/qualification.py, then downloads what is missing into $STARLING_MODELS and verifies every file's size and
# SHA-256 with Starlings.Acquisition.Tool. Anonymous: no token is read or sent. A profile with private hosting is
# skipped with a notice. Exit non-zero if any download or verification fails.
set -euo pipefail
[[ $# -ge 2 ]] || { sed -n '2,10p' "$0" >&2; exit 2; }
: "${STARLING_MODELS:?STARLING_MODELS must name the model cache}"
locks="$1"
shift
repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"
mkdir -p "$locks" "$STARLING_MODELS"
dotnet build ai-dotnet/tools/Starlings.Acquisition.Tool --configuration Release --nologo -v quiet
for profile in "$@"; do
    manifest="model-pipeline/registry/$profile/manifest.yaml"
    hosting="$(python3 -I tools/qualification.py hosting "$manifest")"
    if [[ "$hosting" != public ]]; then
        echo "::notice title=$profile skipped::hosting private (M3): $hosting"
        continue
    fi
    extra=()
    if [[ -f "model-pipeline/protected/$profile/environment.json" ]]; then
        extra=(--reference-environment "model-pipeline/protected/$profile/environment.json")
    fi
    lock="$locks/${profile//\//--}.json"
    python3 -I tools/qualification.py lock "$manifest" "$lock" "${extra[@]}"
    env -u HF_TOKEN dotnet ai-dotnet/tools/Starlings.Acquisition.Tool/bin/Release/net10.0/Starlings.Acquisition.Tool.dll \
        download "$lock" --cache "$STARLING_MODELS"
done
df -h "$STARLING_MODELS"
