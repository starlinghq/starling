# Reference runners

Pinned upstream execution that produces reference fixtures (backlog P-003, P-004). Each runner has its own `requirements.txt`, compiled with `uv pip compile --generate-hashes` from `requirements.in`, because upstream projects need incompatible versions (GLiNER2 caps transformers below 5). Reference runners are CPU and fp32. `encoder/run_e5_cuda.py` compares GPU and other execution settings against those CPU references and does not produce goldens.

Shared helpers live in `src/model_pipeline/fixtures.py` (pure Python, tested in `tests/test_fixtures.py`). Install the package into each runner environment with `pip install --no-deps -e .` plus `jsonschema pyyaml`.

| Path | Upstream | Environment | Output |
| --- | --- | --- | --- |
| `corpus/tokenizer-corpus-v1.jsonl` | none: proposed conformance cases (plan 6.2) | | input to every runner |
| `encoder/run_e5.py` | `intfloat/multilingual-e5-small` | `encoder/requirements.txt` (torch 2.14.1 CPU, transformers 5.18.0, sentence-transformers 6.1.0, tokenizers 0.23.2, onnxruntime 1.30.0) | `protected/multilingual-e5-small/onnx-fp32-cpu/` |
| `encoder/run_e5_cuda.py` | `intfloat/multilingual-e5-small` | `encoder/requirements-cuda.txt` (torch 2.14.1 CUDA 13.0, onnxruntime-gpu 1.30.0 with CUDA 13 and cuDNN 9 wheels; Linux x64, NVIDIA) | `protected/multilingual-e5-small/onnx-fp32-cuda/` (parity, placement, environment; no goldens) |
| `encoder/stage_e5_export.py` | same | `encoder/requirements.txt` | stage 2 (`export`) results for `model-pipeline evaluate`: Python ONNX Runtime CPU embeddings and hidden states, PyTorch hidden states on the same ids (P-007) |
| `encoder/compare_e5_csharp.py`, `encoder/bench_e5_ort.py` | same | `encoder/requirements-cuda.txt` | C#-against-Python ORT output comparison; exploratory matched-backend timing (`docs/m0/e5-gpu.md`) |
| `gliner2/run_gliner2_tokenizer.py` | `fastino/gliner2-base-v1`, GLiNER2 source at `55656fbf` | `gliner2/requirements.txt` (transformers 4.57.6, tokenizers 0.22.2, gliner2 from Git at the pinned commit) | `protected/gliner2-base-v1/tokenizer/` |
| `corpus/gliner2-extraction-corpus-v1.jsonl` | none: proposed extraction cases (plan 10.1), agent-written development data, not the P-030 release set | | input to the GLiNER2 extraction runners |
| `gliner2/run_gliner2_extraction.py` | same, P-014 spike | `gliner2/requirements-export.txt` (requirements.txt plus onnx 1.23.1, onnxscript 0.7.2, onnxruntime 1.30.0) | entity-extraction reference (public output, model inputs, stage tensors); proposed `protected/gliner2-base-v1/onnx-fp32-cpu/` |
| `gliner2/gliner2_entity.py` | same | either GLiNER2 environment | exportable restatement of the entity path (route A full graph, route B backbone, heads) and a pure-Python decoder |
| `gliner2/export_gliner2_onnx.py`, `gliner2/parity_gliner2_ort.py`, `gliner2/dump_gliner2_ort.py` | same | `requirements-export.txt`; GPU: `gliner2/requirements-cuda.txt` (torch 2.14.1 CUDA 13.0, onnxruntime-gpu 1.30.0) | export attempts, ORT CPU/CUDA parity and placement, Python ORT outputs for the C# comparison (`docs/m0/gliner2-p014.md`) |
| `embeddinggemma2/run_embeddinggemma2.py` | `google/embeddinggemma-2` at `914f7f89`, admission spike, text path | `embeddinggemma2/requirements.txt` (torch 2.14.1 CPU, transformers 5.19.0, sentence-transformers 6.1.0, tokenizers 0.23.3, onnxruntime 1.30.0, onnx 1.23.1, onnxscript 0.7.2; Pillow and torchvision because sentence-transformers loads the multimodal processor) | tokenizer fixtures, sentence-transformers embeddings with and without prompts, tokenizer implementation check, reference stability; proposed `protected/embeddinggemma-2/` |
| `embeddinggemma2/eg2_text.py`, `export_embeddinggemma2_onnx.py`, `parity_embeddinggemma2_ort.py`, `compare_embeddinggemma2_csharp.py` | same | `requirements.txt`; GPU: `embeddinggemma2/requirements-cuda.txt` (torch 2.14.1 CUDA 13.0, onnxruntime-gpu 1.30.0) | exportable text path, ONNX export (dynamo, TorchScript), ORT CPU/CUDA parity, placement, batch and sliding-window checks, C# comparison (`docs/m0/embeddinggemma2-spike.md`) |
| `bench/bench_eg2_matched.py`, `bench/bench_gliner2_matched.py`, `bench/bench_qwen35_matched.py` (core `bench/matched_core.py`, also used by `encoder/bench_e5_matched.py`) | the pinned graphs and bundle of each profile | EmbeddingGemma 2 text and GLiNER2: `encoder/requirements-bench.txt` (CUDA: `requirements-bench-cuda.txt`); EmbeddingGemma 2 image: `embeddinggemma2/requirements.txt` (CUDA: `requirements-cuda.txt`); Qwen: `bench/requirements-genai.txt` (onnxruntime-genai 0.17.1, onnxruntime 1.30.0; CUDA: `requirements-genai-cuda.txt`) | Python same-backend G5 timing, `starlings-bench-raw/1` (`docs/benchmarks/PROTOCOL.md`); not fixtures |
| `encoder/quantize_e5_browser.py`, `encoder/drift_e5_browser.py` | `intfloat/multilingual-e5-small` | `encoder/requirements-browser.txt` (onnxruntime 1.30.0, onnx 1.23.1, Playwright 1.63.0; no torch) | browser spike (`docs/m0/browser-wasm-spike.md`): fp16 and int8 variants of the ONNX graph with hashes, their drift against the protected fp32 embeddings natively and from browser runs (`browser/spike/harness/`); exploratory, not fixtures |
| `compare_tokenizers.py` | any snapshot with `tokenizer.json` | either environment | tokenizer implementation differences against `tokenizer.json` |
| `corpus/tokenizer-corpus-qwen-bpe-v1.jsonl` | none: proposed byte-level BPE cases (contractions in mixed case, digit, whitespace and newline runs, combining marks, NFC, Indic, Thai, Arabic, CJK, emoji sequences, astral letters and digits, every Qwen added token and near-misses, long words), agent-written | | input to the Qwen BPE runner, after `tokenizer-corpus-v1.jsonl` |
| `qwen-bpe/run_qwen_bpe_tokenizer.py` | profile `qwen2-bpe-kev`: `Qwen/Qwen3.5-0.8B-Base` at `dc7cdfe2` loaded as Kev does, asset `jaredpalmer/kev-0.8b@9a45d25e` `tokenizer.json`; profile `qwen35-bpe-chat`: `Qwen/Qwen3.5-0.8B` at `2fc06364`, asset = the serialized transformers 5.18.0 pipeline | `qwen-bpe/requirements-kev.txt` (transformers 5.17.0, tokenizers 0.23.2) and `qwen-bpe/requirements-chat.txt` (transformers 5.18.0, tokenizers 0.23.2); no torch | proposed `protected/qwen-bpe/{kev,chat}/tokenizer/` (the tokenizer's copies are in starlinghq/tokenizers); optional randomized differential dumps and the regex class probe |
| `gemma-bpe/run_gemma_bpe_tokenizer.py` | profile `gemma-bpe-embeddinggemma2`: asset `google/embeddinggemma-2@914f7f89` `tokenizer.json` (fixtures are the accepted `protected/embeddinggemma-2/onnx-fp32-cpu/tokenizer*`) | `gemma-bpe/requirements.txt` (tokenizers 0.23.3, the fixtures' version); no torch, no transformers | randomized differential dumps (committed one in `tests/Starlings.Tokenizers.Tests/TestData/gemma-bpe/differential.jsonl.gz` in starlinghq/tokenizers, large ones outside Git) and the synthetic component reference `TestData/gemma-bpe/components.json`; not fixtures |
| `kev/` | `jaredpalmer/kev-0.8b` P-027 spike | upstream Kev `uv.lock` with torch 2.8.0 CPU build; `kev/requirements-cuda.txt` for the CUDA runs | spike evidence, not fixtures (`model-pipeline/admissions/kev.md`, "Spike results" and "GPU results") |
| `kev/run_kev.py` (with `kev_merge.py`, `kev_ort.py`) | `jaredpalmer/kev-0.8b` and `kev-4b` on their pinned Qwen3.5 bases, Kev code at `6b719c3c` | `kev/requirements.txt` (Python 3.12; Kev's `uv.lock` with torch 2.8.0 CPU build, transformers 5.17.0, onnxruntime 1.30.0, plus the ONNX Runtime GenAI builder), Kev installed with `--no-deps` from a checkout at the pinned commit | stages of the `kev-onnx-cpu-v1` onboarding route (`src/model_pipeline/onboarding_kev.py`): PyTorch reference, bundle build, Python ONNX Runtime (`onnx --provider cpu|cuda`; `cuda` runs the same bundle with TF32 off, for the `onnx-fp32-cuda` profiles); proposals, not fixtures |
| `corpus/kev-decisions-v1.jsonl` | none: Starling-written decision records | | input to the Kev route |
| `bert/run_bert.py` | the BERT-route candidates (`model-pipeline/candidates/{minilm,bge-small,bge-base,ms-marco,nomic-embed}/`) | `encoder/requirements.txt` | original-framework reference and Python ONNX Runtime stages of the `bert-onnx-cpu-v1` route (`src/model_pipeline/onboarding_bert.py`); proposals, not fixtures |
| `corpus/asr-corpus-v1.jsonl`, `corpus/asr-v1/*.wav` | four LibriSpeech dev-clean utterances (CC BY 4.0) from `hf-internal-testing/librispeech_asr_dummy` at `5be91486`, as 16 kHz mono int16 WAV with their SHA-256 and LibriSpeech transcripts; written by `whisper/make_asr_corpus.py` | `make_asr_corpus.py` only (pyarrow, soundfile, outside the lock) | input to the Whisper route |
| `whisper/run_whisper.py` | `openai/whisper-large-v3-turbo` at `41f01f3f` (PyTorch fp32 reference), `onnx-community/whisper-large-v3-turbo-ONNX` at `76ba4fb5` (fp16 graphs) (`model-pipeline/candidates/whisper-large-v3-turbo/`) | `whisper/requirements.txt` (torch 2.14.1 CPU, transformers 5.18.0, tokenizers 0.23.2, onnxruntime 1.30.0) | transformers reference (mel on the numpy and default torch feature paths, encoder hidden state, `generate` transcript, teacher-forced logits) and Python ONNX Runtime stages of the `whisper-onnx-cpu-v1` route (`src/model_pipeline/onboarding_whisper.py`); proposals, not fixtures |
| `qwen-chat/run_reference.py`, `build_bundle.py`, `run_genai.py`, `common.py` | `Qwen/Qwen3-0.6B`, `Qwen/Qwen2.5-0.5B-Instruct` (`model-pipeline/candidates/{qwen3,qwen25}/`) | `generation/requirements.txt` | transformers fp32 CPU reference, ONNX Runtime GenAI bundle build and Python GenAI path of the `qwen-genai-cpu-v1` route (`src/model_pipeline/onboarding_generation.py`); proposals, not fixtures |
| `generation/` | `Qwen/Qwen3.5-0.8B` at `2fc06364`, M0 generation-backend check | `generation/requirements.txt` (transformers 5.18.0 reference); C# runners under `generation/dotnet/` (ONNX Runtime GenAI 0.17.1, LLamaSharp 0.27.0) | backend comparison evidence, not fixtures (`docs/m0/generation-backend-check.md`) |
| `generation/make_template_corpus.py`, `generation/make_long_prompts.py` | the same checkpoint, backlog P-029 and R8 | `generation/requirements.txt` | proposed fixtures in `model-pipeline/proposed/qwen3.5-0.8b/` (`template/`: chat-template corpus; `long-context/`: `prompts-long-v1.jsonl` through `run_reference.py` and `make_generation_fixture.py`) |

Model snapshots are downloaded by commit SHA into `.models/` (ignored by Git) and hashed; see `docs/m0/hub-pins.md`. For runtime assets, the acquisition command (P-009) is `ai-dotnet/tools/Starlings.Acquisition.Tool` (library `Starlings.Acquisition`).

## Running

```sh
uv venv -p 3.11 .models/venv-encoder
VIRTUAL_ENV=.models/venv-encoder uv pip install --require-hashes -r model-pipeline/reference/encoder/requirements.txt \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
VIRTUAL_ENV=.models/venv-encoder uv pip install --no-deps -e model-pipeline jsonschema pyyaml
.models/venv-encoder/bin/python model-pipeline/reference/encoder/run_e5.py \
  --model-dir .models/intfloat_multilingual-e5-small@614241f6 --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 \
  --corpus model-pipeline/reference/corpus/tokenizer-corpus-v1.jsonl \
  --out model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu
```

GPU parity (select the GPU by PCI order; the model snapshot is the same):

```sh
uv venv -p 3.11 .models/venv-encoder-cuda
VIRTUAL_ENV=.models/venv-encoder-cuda uv pip install --require-hashes -r model-pipeline/reference/encoder/requirements-cuda.txt
VIRTUAL_ENV=.models/venv-encoder-cuda uv pip install --no-deps -e model-pipeline jsonschema pyyaml
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n> .models/venv-encoder-cuda/bin/python \
  model-pipeline/reference/encoder/run_e5_cuda.py \
  --model-dir .models/intfloat_multilingual-e5-small@614241f6 --revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 \
  --fixtures model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu \
  --out model-pipeline/protected/multilingual-e5-small/onnx-fp32-cuda --exploratory-out .models/e5-exploratory
```

Staged fidelity evaluation of the e5 CPU profile (P-007; the evaluator runs in the `model-pipeline` dev environment, the C# tool needs the .NET SDK):

```sh
.models/venv-encoder/bin/python model-pipeline/reference/encoder/stage_e5_export.py \
  --model-dir .models/intfloat_multilingual-e5-small@614241f6 \
  --fixtures model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu \
  --profile multilingual-e5-small/onnx-fp32-cpu --out .models/e5-results
dotnet run --project ai-dotnet/tools/Starlings.Fidelity.Tool -c Release -- \
  e5 --snapshot .models/intfloat_multilingual-e5-small@614241f6 --out .models/e5-results --hidden
model-pipeline evaluate model-pipeline/registry/multilingual-e5-small/onnx-fp32-cpu/manifest.yaml \
  --results .models/e5-results --out .models/e5-report --require-gates G1,G2
```

The GLiNER2 environment is built the same way from `gliner2/requirements.txt` without `--require-hashes` (the `gliner2` line is a Git URL pinned by commit, which pip cannot hash).

GLiNER2 entity extraction spike (P-014; `model.safetensors` is needed, 833,938,108 bytes, SHA-256 `845fc4bd...`):

```sh
uv venv -p 3.11 .models/venv-gliner2-export
VIRTUAL_ENV=.models/venv-gliner2-export uv pip install -r model-pipeline/reference/gliner2/requirements-export.txt \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
VIRTUAL_ENV=.models/venv-gliner2-export uv pip install --no-deps -e model-pipeline jsonschema pyyaml
cd model-pipeline/reference/gliner2
P=../../../.models/venv-gliner2-export/bin/python; M=../../../.models/fastino_gliner2-base-v1@f9634218
$P run_gliner2_extraction.py --model-dir $M --revision f9634218e53580c56edf0de97ca1a7d3f1c2354e \
  --corpus ../corpus/gliner2-extraction-corpus-v1.jsonl --out ../../../.models/g2-ref --tensors-out ../../../.models/g2-ref/tensors
$P export_gliner2_onnx.py --model-dir $M --out ../../../.models/g2-onnx                      # A, B, H with both exporters
$P export_gliner2_onnx.py --model-dir $M --out ../../../.models/g2-onnx --routes A,H --exporters legacy --explicit-attention
$P parity_gliner2_ort.py --onnx-dir ../../../.models/g2-onnx --reference ../../../.models/g2-ref --out ../../../.models/g2-parity-cpu.json
# Stage results for `model-pipeline evaluate` (task entity-extraction): stage 1 re-runs GLiNER2's Python processor,
# stage 2 runs Python ONNX Runtime on the route A graph with the fixture inputs and recomputes PyTorch hidden states.
# --graph-asset-prefix (the path without ".onnx") must match the path the manifest's origin.files pins the graph under;
# the manifest pins onnx/gliner2-entities-fp32.onnx, and the default prefix onnx/gliner2-entity-route-a does not match it.
F=../../protected/gliner2-base-v1/onnx-fp32-cpu   # extraction.jsonl (accepted 2026-10-10, PR #19)
$P stage_gliner2.py tokenizer --model-dir $M --fixtures $F --profile gliner2-base-v1/onnx-fp32-cpu --out ../../../.models/g2-stage
$P stage_gliner2.py export --model-dir $M --graph ../../../.models/g2-onnx/gliner2-A-dynamo.onnx --fixtures $F \
  --profile gliner2-base-v1/onnx-fp32-cpu --out ../../../.models/g2-stage --threads 1
# GPU (requirements-cuda.txt environment; select the GPU by PCI order):
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n> <cuda venv>/bin/python parity_gliner2_ort.py --provider cuda --cuda-tf32 0,1 \
  --graphs A-dynamo --batch-max-tokens 600 --onnx-dir ../../../.models/g2-onnx --reference ../../../.models/g2-ref --out <json>
```

Qwen byte-level BPE profiles (tokenizer only; the runner refuses to write fixtures unless the live transformers pipeline serializes to exactly the profile asset). Snapshots hold only the tokenizer files (`fetch_snapshot.py --include 'tokenizer*' --include vocab.json --include merges.txt --include config.json`); assets come from `tools/fetch-tokenizer-assets.sh`:

```sh
for p in kev chat; do
  uv venv -p 3.11 .models/venv-qwen-$p
  VIRTUAL_ENV=.models/venv-qwen-$p uv pip install --require-hashes -r model-pipeline/reference/qwen-bpe/requirements-$p.txt
  VIRTUAL_ENV=.models/venv-qwen-$p uv pip install --no-deps -e model-pipeline jsonschema pyyaml
done
cd model-pipeline/reference/qwen-bpe
../../../.models/venv-qwen-kev/bin/python run_qwen_bpe_tokenizer.py --profile kev \
  --snapshot <Qwen_Qwen3.5-0.8B-Base@dc7cdfe2> --asset <models>/jaredpalmer/kev-0.8b/9a45d25eb2ab761841196625383fa1dff0e56c1e/tokenizer.json \
  --corpus ../corpus/tokenizer-corpus-v1.jsonl --corpus ../corpus/tokenizer-corpus-qwen-bpe-v1.jsonl --out <out>/kev \
  --differential-out <out>/kev/differential.jsonl.gz --unicode-classes <out>/unicode-classes.json
../../../.models/venv-qwen-chat/bin/python run_qwen_bpe_tokenizer.py --profile chat \
  --snapshot <Qwen_Qwen3.5-0.8B@2fc06364> --asset <models>/Qwen/Qwen3.5-0.8B/2fc06364715b967f1860aea9cf38778875588b17/transformers-5.18.0/tokenizer.json \
  --corpus ../corpus/tokenizer-corpus-v1.jsonl --corpus ../corpus/tokenizer-corpus-qwen-bpe-v1.jsonl --out <out>/chat \
  --differential-out <out>/chat/differential.jsonl.gz
# Large differential runs, kept outside Git; STARLING_QWEN_DIFFERENTIAL_DIR=<dir> makes QwenBpeDifferentialTests read them:
#   --differential-out <dir>/kev-seed8-1m.jsonl.gz --differential-cases 1000000 --decode-cases 100000 --variant-cases 50000 --seed 8
```

The differential dump is deterministic for a seed (gzip without timestamp). `--write-live <file>` writes the serialized live pipeline, which is how the `qwen35-bpe-chat` asset was produced.

Gemma BPE tokenizer profile (tokenizer only; the asset comes from `tools/fetch-tokenizer-assets.sh`):

```sh
uv venv -p 3.11 .models/venv-gemma-bpe
VIRTUAL_ENV=.models/venv-gemma-bpe uv pip install --require-hashes -r model-pipeline/reference/gemma-bpe/requirements.txt
cd model-pipeline/reference/gemma-bpe
../../../.models/venv-gemma-bpe/bin/python run_gemma_bpe_tokenizer.py \
  --asset <models>/google/embeddinggemma-2/914f7f89142e33e77833254d9c9b90c3cef7303b/tokenizer.json \
  --differential-out <tokenizers checkout>/tests/Starlings.Tokenizers.Tests/TestData/gemma-bpe/differential.jsonl.gz \
  --components-out <tokenizers checkout>/tests/Starlings.Tokenizers.Tests/TestData/gemma-bpe/components.json
# Large differential runs, kept outside Git; STARLING_GEMMA_DIFFERENTIAL_DIR=<dir> makes GemmaBpeDifferentialTests read them:
#   --differential-out <dir>/gemma-seed8-1m.jsonl.gz --differential-cases 1000000 --decode-cases 100000 --variant-cases 50000 --seed 8
```

The dump is deterministic for a seed; its header lists the option variants as text edits of the asset, which the .NET test applies the same way.

EmbeddingGemma 2 spike (text path; snapshot downloaded by commit SHA, `docs/m0/hub-pins.md`):

```sh
uv venv -p 3.11 .models/venv-eg2
VIRTUAL_ENV=.models/venv-eg2 uv pip install --require-hashes -r model-pipeline/reference/embeddinggemma2/requirements.txt \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
VIRTUAL_ENV=.models/venv-eg2 uv pip install --no-deps -e model-pipeline jsonschema pyyaml
cd model-pipeline/reference/embeddinggemma2
P=../../../.models/venv-eg2/bin/python; M=<snapshot dir>
$P run_embeddinggemma2.py --model-dir $M --revision 914f7f89142e33e77833254d9c9b90c3cef7303b \
  --corpus ../corpus/tokenizer-corpus-v1.jsonl --out <ref dir>
$P export_embeddinggemma2_onnx.py --model-dir $M --out <onnx dir> --exporters dynamo,legacy
$P parity_embeddinggemma2_ort.py --model-dir $M --graph <onnx dir>/eg2-text-dynamo-sdpa.onnx --reference <ref dir> \
  --out parity-cpu.json --threads 12 --dump ort-cpu.npz
# GPU (requirements-cuda.txt environment; select the GPU by PCI order):
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n> <cuda venv>/bin/python parity_embeddinggemma2_ort.py --provider cuda --cuda-tf32 0 \
  --model-dir $M --graph <onnx dir>/eg2-text-dynamo-sdpa.onnx --reference <ref dir> --out parity-cuda.json --dump ort-cuda.npz
```

EmbeddingGemma 2 video spike (`docs/m0/embeddinggemma2-video-spike.md`; the text environment plus torchcodec, PyAV and OpenCV, and a pinned FFmpeg build):

```sh
uv venv -p 3.11 .models/venv-eg2-video
VIRTUAL_ENV=.models/venv-eg2-video uv pip install --require-hashes -r model-pipeline/reference/embeddinggemma2/requirements-video.txt \
  --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
VIRTUAL_ENV=.models/venv-eg2-video uv pip install --no-deps -e model-pipeline jsonschema pyyaml
cd model-pipeline/reference/embeddinggemma2
P=../../../.models/venv-eg2-video/bin/python; M=<snapshot dir>; F=<ffmpeg dir>; V=<video dir>
sh fetch_eg2_ffmpeg.sh $F   # BtbN builds of FFmpeg n8.1.3-17-gc389840894, checked by SHA-256
GPL=$F/ffmpeg-n8.1.3-17-gc389840894-linux64-gpl-shared-8.1; LGPL=$F/ffmpeg-n8.1.3-17-gc389840894-linux64-lgpl-shared-8.1
$P make_eg2_videos.py --ffmpeg-bin $GPL/bin --decoder-bin $LGPL/bin --out $V   # byte-identical on regeneration
LD_LIBRARY_PATH=$LGPL/lib $P run_embeddinggemma2_video.py --model-dir $M --revision 914f7f89142e33e77833254d9c9b90c3cef7303b \
  --videos $V --out <ref dir> --tensors <tensor dir>
$P parity_embeddinggemma2_video_ort.py --vision <graphs>/embeddinggemma2-vision-fp32.onnx \
  --text <graphs>/embeddinggemma2-text-mm-fp32.onnx --reference <ref dir> --tensors <tensor dir> --out parity-cpu.json --threads 12
LD_LIBRARY_PATH=$LGPL/lib $P decode_eg2_videos.py --decoder pyav --videos $V --reference <ref dir> \
  --reference-tensors <tensor dir> --out <decode dir>   # also opencv, ffmpeg-cli, numpy-nearest, numpy-bilinear, external:<dir>
$P make_eg2_sampling_cases.py --out sampling-cases.json
```

## Fixture format (version 1)

- JSON Lines, UTF-8, LF, one compact object per line, non-ASCII kept literal. Read with a split on `\n` only: corpus text contains U+0085 and U+2028, which `str.splitlines()` treats as line breaks.
- Every case carries its input `text` (and `pair`), so a fixture is usable without the corpus file.
- Offsets are exclusive-end and recorded in three units, `unicode-scalars`, `utf16-code-units` and `utf8-bytes`. The upstream unit is Unicode scalar values (indices into a Python `str`); the other two are conversions of it. A C# consumer slices `string` with `utf16-code-units`. Special tokens have offsets `(0, 0)`.
- Floats are float32 values written with the shortest decimal that round-trips to the same float32 bits.
- `environment.json` names the runner, the repository commit it ran from (`-dirty` if the tree had uncommitted changes), the model revision, the SHA-256 of every asset and of the corpus, and interpreter, package and CPU details.
- Fixtures under `protected/` change only through maintainer review (CODEOWNERS, plan 10.1). Tolerances are not set here.
