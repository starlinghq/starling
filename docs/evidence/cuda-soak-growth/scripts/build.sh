#!/usr/bin/env bash
# build.sh <sha>: build the CUDA soak tool at <sha> (branch cuda-soak-growth) into /workspace/build/soak-cuda-<sha8>.
set -euo pipefail
W=/workspace
sha="$1"
export DOTNET_ROOT=$W/dotnet DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 NUGET_PACKAGES=$W/nuget
src=$W/starling-cuda-growth
if [[ ! -d $src/.git ]]; then git clone -q https://github.com/starlinghq/starling.git $src; fi
cd $src
git fetch -q origin cuda-soak-growth
git checkout -q --detach "$sha"
test "$(git rev-parse HEAD)" = "$sha"
out=$W/build/soak-cuda-growth-${sha:0:8}
$W/dotnet/dotnet build ai-dotnet/tools/Starlings.Soak.Tool -c Release -v q -nologo -p:RestoreLockedMode=true -p:UseCuda=true -o "$out"
ls "$out/Starlings.Soak.Tool.dll"
echo BUILD OK
