# Contributor image for recipe-driven onboarding (ONBOARDING.md, "Container image"): the .NET SDK that global.json
# pins, uv, and the hash-locked reference environments of the CPU onboarding routes, so a contributor does not
# rebuild PyTorch environments. No CI job builds or publishes it, and it holds no model files: snapshots and the
# Starlings model cache are mounted at run time.
#
#   docker build -f model-pipeline/Dockerfile -t starlings-onboarding model-pipeline
#   docker run --rm -it -v "$PWD:/work" -v "$HOME/.cache/starlings-models:/models" starlings-onboarding \
#     model-pipeline run model-pipeline/recipes/<name>-cpu.yaml --snapshot /models/<snapshot> \
#     --reference-python /opt/venvs/encoder/bin/python --out artifacts/onboarding/<name>
#
# Build context: model-pipeline/ (only the requirement locks are copied). ENVIRONMENTS selects the environments:
#   encoder     reference/encoder/requirements.txt, Python 3.11, CPU torch (bert-onnx-cpu-v1, e5-onnx-cpu-v1)
#   generation  reference/generation/requirements.txt, Python 3.12, the CUDA 13.0 torch wheel the lock pins
#               and its CUDA dependency wheels (qwen-genai-cpu-v1 runs it on CPU)
#   whisper     reference/whisper/requirements.txt, Python 3.11, CPU torch (whisper-onnx-cpu-v1)
# The Kev environment also needs Kev installed from a pinned Git checkout and is not built here.

# .NET SDK 10.0.100 (global.json, rollForward disable), Ubuntu 24.04, by digest.
FROM mcr.microsoft.com/dotnet/sdk:10.0.100@sha256:c7445f141c04f1a6b454181bd098dcfa606c61ba0bd213d0a702489e5bd4cd71

ARG ENVIRONMENTS="encoder generation whisper"
# uv from PyPI, by version and SHA-256 of its manylinux x86_64 wheel; it also provides the pinned Python builds.
ARG UV_VERSION=0.11.32
ARG UV_SHA256=3da76cd4e2697de30928b8a8524bd39183ac1e08cb7e72833807c022b7cba6c4

ENV DOTNET_NOLOGO=1 \
    DOTNET_CLI_TELEMETRY_OPTOUT=1 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_LINK_MODE=copy \
    STARLING_MODELS=/models \
    HF_HOME=/models/.hf

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/uv \
    && printf 'uv==%s --hash=sha256:%s\n' "$UV_VERSION" "$UV_SHA256" > /tmp/uv.txt \
    && /opt/uv/bin/pip install --no-cache-dir --require-hashes -r /tmp/uv.txt \
    && ln -s /opt/uv/bin/uv /usr/local/bin/uv \
    && rm /tmp/uv.txt

COPY reference/encoder/requirements.txt /opt/locks/encoder.txt
COPY reference/generation/requirements.txt /opt/locks/generation.txt
COPY reference/whisper/requirements.txt /opt/locks/whisper.txt

# One venv per lock, installed with --require-hashes exactly as ONBOARDING.md and FIVE_MODELS.md describe.
RUN set -eu; \
    for env in $ENVIRONMENTS; do \
        case "$env" in \
            encoder) python=3.11; extra=https://download.pytorch.org/whl/cpu ;; \
            generation) python=3.12; extra=https://download.pytorch.org/whl/cu130 ;; \
            whisper) python=3.11; extra=https://download.pytorch.org/whl/cpu ;; \
            *) echo "unknown environment: $env" >&2; exit 2 ;; \
        esac; \
        uv venv --python "$python" "/opt/venvs/$env"; \
        uv pip install --python "/opt/venvs/$env/bin/python" --require-hashes -r "/opt/locks/$env.txt" \
            --index-url https://pypi.org/simple --extra-index-url "$extra" --index-strategy unsafe-best-match; \
        # The one pipeline dependency the reference locks do not pin (the runners import model_pipeline).
        uv pip install --python "/opt/venvs/$env/bin/python" 'jsonschema>=4.21'; \
    done; \
    uv venv --python 3.11 /opt/venvs/pipeline; \
    uv pip install --python /opt/venvs/pipeline/bin/python 'jsonschema>=4.21' 'pyyaml>=6.0' 'pytest>=8' 'ruff>=0.6'; \
    rm -rf "$UV_CACHE_DIR"

COPY container-entrypoint.sh /usr/local/bin/starlings-onboarding
RUN chmod 0755 /usr/local/bin/starlings-onboarding

WORKDIR /work
VOLUME ["/models"]
ENTRYPOINT ["/usr/local/bin/starlings-onboarding"]
CMD ["bash"]
