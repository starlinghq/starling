#!/bin/bash
# Installs the toolchain for Claude Code cloud sessions: .NET 10 SDK and the model-pipeline package.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(pwd)}"

# builds.dotnet.microsoft.com is not reachable from cloud sessions; Ubuntu packages are.
if ! dotnet --list-sdks 2>/dev/null | grep -q '^10\.'; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get install -y -qq dotnet-sdk-10.0 >/dev/null 2>&1 \
    || { apt-get update -qq && apt-get install -y -qq dotnet-sdk-10.0 >/dev/null; }
fi

export DOTNET_NOLOGO=true DOTNET_CLI_TELEMETRY_OPTOUT=true
if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
  echo 'export DOTNET_NOLOGO=true DOTNET_CLI_TELEMETRY_OPTOUT=true' >> "$CLAUDE_ENV_FILE"
fi

dotnet restore ai-dotnet/AI.slnx --verbosity quiet

python3 -m pip install --quiet --disable-pip-version-check -e 'model-pipeline[dev]'
