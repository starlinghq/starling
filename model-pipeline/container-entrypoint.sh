#!/usr/bin/env bash
# Entrypoint of model-pipeline/Dockerfile. The checkout is mounted at /work, so on start the pipeline package is
# installed from it, editable and without dependencies (the image already holds them; no network), into the pipeline
# venv and into each reference venv (as ONBOARDING.md does), so the CLI and the reference runners use the mounted code.
set -euo pipefail

if [[ ! -f /work/model-pipeline/pyproject.toml ]]; then
    echo "mount a starlinghq/starling checkout at /work (docker run -v \"\$PWD:/work\" ...)" >&2
    exit 2
fi

for venv in /opt/venvs/*/; do
    uv pip install --quiet --offline --python "${venv}bin/python" --no-deps -e /work/model-pipeline
done

export PATH="/opt/venvs/pipeline/bin:$PATH"
exec "$@"
