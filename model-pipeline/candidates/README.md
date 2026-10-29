# Candidate contracts

Each subdirectory holds one `profile.json`: a proposed contract for a model that is onboarded through a recipe ([`../recipes/README.md`](../recipes/README.md)). The Kev, Qwen chat and BERT-route candidates also have registry manifests under `../registry/` (Experimental; the BERT ones since the ai-dotnet 0.4.0 release candidate, so `Starlings.Models.Bert` can state per-profile maturity); Whisper has none. A candidate pins the exact source files, the runtime, the behavior the route implements and the development comparison policy. It is not a support claim and grants no gate or maturity. Registry profiles are under [`../registry/`](../registry/README.md).

Format: [`../schemas/onboarding-candidate.schema.json`](../schemas/onboarding-candidate.schema.json).

| Field | Meaning |
| --- | --- |
| `format_version` | Always `1`. |
| `id` | Profile ID, `<model>/<profile>`. |
| `task` | `embeddings`, `reranking`, `generation` or `decision`. |
| `origin.repo`, `origin.revision`, `origin.files` | Hugging Face repository, full commit SHA, and the path, size and SHA-256 of every file the route reads. The `assets` stage verifies each one. |
| `origin.base` (optional) | A second pinned repository the model is built on, with its own files (Kev's Qwen3.5 base). |
| `origin.code` (optional) | Upstream inference code the reference runs, pinned by full commit SHA (Kev). |
| `runtime` | `backend` (`onnx` or `ortgenai`), `version` (`1.30.0` or `0.17.1`), `provider` (`CPUExecutionProvider` only), `precision` (`fp32` only). |
| `behavior` | The route-specific contract (for BERT: pooling, normalization, width, token limit, prompts). |
| `evaluation.status`, `evaluation.policy` | Always `proposed`; the policy holds the predeclared development bounds. |

The schema is only the first check. Each plugin compares the whole contract field by field with its own allow-list (Python in `src/model_pipeline/onboarding_*.py`, and the matching .NET library) and refuses anything else. Changing a value here without changing the route code makes the run fail. The BERT route's allow-list is a data file both sides read, [`../routes/bert-onnx-cpu-v1/contracts.json`](../routes/README.md); the other routes keep theirs in code.

| Directory | Profile | Upstream (revision) | Task | Plugin |
| --- | --- | --- | --- | --- |
| `minilm/` | `all-minilm-l6-v2/onnx-fp32-cpu` | `sentence-transformers/all-MiniLM-L6-v2` (`1110a243`) | embeddings: 384 dimensions, mean pooling, L2, 256 tokens | `bert-onnx-cpu-v1` |
| `bge-small/` | `bge-small-en-v1.5/onnx-fp32-cpu` | `BAAI/bge-small-en-v1.5` (`5c38ec7c`) | embeddings: 384 dimensions, CLS pooling, L2, 512 tokens, `query` prompt | `bert-onnx-cpu-v1` |
| `bge-base/` | `bge-base-en-v1.5/onnx-fp32-cpu` | `BAAI/bge-base-en-v1.5` (`a5beb1e3`) | embeddings: 768 dimensions, CLS pooling, L2, 512 tokens, `query` prompt | `bert-onnx-cpu-v1` |
| `ms-marco/` | `ms-marco-minilm-l6-v2/onnx-fp32-cpu` | `cross-encoder/ms-marco-MiniLM-L6-v2` (`233902d2`) | reranking: raw scalar logits, 512 tokens | `bert-onnx-cpu-v1` |
| `nomic-embed/` | `nomic-embed-text-v1.5/onnx-fp32-cpu` | `nomic-ai/nomic-embed-text-v1.5` (`e9b67630`) | embeddings: 768 dimensions, mean pooling, no normalization, 8,192 tokens | `bert-onnx-cpu-v1` |
| `qwen3/` | `qwen3-0.6b/ortgenai-fp32-cpu` | `Qwen/Qwen3-0.6B` (`c1899de2`) | generation: greedy text chat, thinking disabled | `qwen-genai-cpu-v1` |
| `qwen25/` | `qwen2.5-0.5b-instruct/ortgenai-fp32-cpu` | `Qwen/Qwen2.5-0.5B-Instruct` (`7ae55760`) | generation: greedy text chat | `qwen-genai-cpu-v1` |
| `kev-0.8b/` | `kev-0.8b/onnx-fp32-cpu` | `jaredpalmer/kev-0.8b` (`9a45d25e`) on `Qwen/Qwen3.5-0.8B-Base` (`dc7cdfe2`) | decision | `kev-onnx-cpu-v1` |
| `kev-4b/` | `kev-4b/onnx-fp32-cpu` | `jaredpalmer/kev-4b` (`139fdd94`) on `Qwen/Qwen3.5-4B-Base` (`1001bb4d`) | decision | `kev-onnx-cpu-v1` |
| `whisper-large-v3-turbo/` | `whisper-large-v3-turbo/onnx-fp16-cpu` | `onnx-community/whisper-large-v3-turbo-ONNX` (`76ba4fb5`, fp16 graphs) on `openai/whisper-large-v3-turbo` (`41f01f3f`, PyTorch reference) | transcription: 30 s of 16 kHz audio, English, greedy, no timestamps | `whisper-onnx-cpu-v1` |
| `lfm3-1.7b/` | `lfm3-1.7b/onnx-fp32-cpu` | `LiquidAI/LFM3-1.7B-ONNX` (`c680759a`, fp32 graph) on `LiquidAI/LFM3-1.7B` (`40ebe0aa`, PyTorch reference) | generation: greedy text chat, thinking on or off | none yet; development evidence from `reference/lfm3/run_lfm3.py` and `Starlings.Models.Lfm3.Tests` (`docs/evidence/lfm3-onboarding-2026-10-28/`) |

The full revisions and file digests are in each `profile.json`. Licenses of the five-model candidates are listed in [FIVE_MODELS.md](../FIVE_MODELS.md); candidates have no license fields.

Adding a model to an existing route means a new candidate, a recipe, and matching entries in the route's Python and .NET allow-lists, with tests (`skills/starling-add-model/SKILL.md`). For the BERT route, `model-pipeline scaffold <name>` generates the recipe, the contract table entry and the other inputs from the candidate ([ONBOARDING.md](../ONBOARDING.md#scaffold-a-bert-route-candidate)). Fresh references and policies stay proposals: adoption into `protected/` and any registry manifest are maintainer changes.
