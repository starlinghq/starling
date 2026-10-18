# Onboarding recipes

Each file is one versioned onboarding recipe for `model-pipeline plan` and `model-pipeline run` ([ONBOARDING.md](../ONBOARDING.md)). A recipe names a plugin and repository-relative inputs. It contains no shell commands and no machine paths: snapshots, interpreters and output directories are command-line options or workflow bindings.

Format: [`../schemas/onboarding-recipe.schema.json`](../schemas/onboarding-recipe.schema.json). Every field is required and no other field is allowed.

| Field | Meaning |
| --- | --- |
| `recipe_version` | Always `1`. |
| `id` | Lowercase letters, digits, `.` and `-`. |
| `plugin` | A plugin ID registered in `PLUGINS` in `src/model_pipeline/onboarding_plugins.py`. |
| `manifest` | The registry manifest (E5) or candidate contract (`../candidates/`) the run checks against. |
| `reference_corpus` | Input cases for the fresh reference. |
| `reference_environment` | The hash-locked `requirements.txt` the reference interpreter must match. |

Paths are repository-relative. Absolute paths, backslashes, `.` or `..` segments, paths that escape the repository, duplicate keys and unknown plugins are refused.

| Recipe | Plugin | Contract | Corpus | Reference environment |
| --- | --- | --- | --- | --- |
| `multilingual-e5-small-cpu.yaml` | `e5-onnx-cpu-v1` | `registry/multilingual-e5-small/onnx-fp32-cpu/manifest.yaml` | `reference/corpus/tokenizer-corpus-v1.jsonl` | `reference/encoder/requirements.txt` |
| `minilm-cpu.yaml` | `bert-onnx-cpu-v1` | `candidates/minilm/profile.json` | `corpora/five-models/embeddings.jsonl` | `reference/encoder/requirements.txt` |
| `bge-small-cpu.yaml` | `bert-onnx-cpu-v1` | `candidates/bge-small/profile.json` | `corpora/five-models/bge.jsonl` | `reference/encoder/requirements.txt` |
| `bge-base-cpu.yaml` | `bert-onnx-cpu-v1` | `candidates/bge-base/profile.json` | `corpora/five-models/embeddings.jsonl` | `reference/encoder/requirements.txt` |
| `ms-marco-cpu.yaml` | `bert-onnx-cpu-v1` | `candidates/ms-marco/profile.json` | `corpora/five-models/reranking.jsonl` | `reference/encoder/requirements.txt` |
| `nomic-embed-cpu.yaml` | `bert-onnx-cpu-v1` | `candidates/nomic-embed/profile.json` | `corpora/five-models/embeddings.jsonl` | `reference/encoder/requirements.txt` |
| `qwen3-cpu.yaml` | `qwen-genai-cpu-v1` | `candidates/qwen3/profile.json` | `corpora/five-models/chat.jsonl` | `reference/generation/requirements.txt` |
| `qwen25-cpu.yaml` | `qwen-genai-cpu-v1` | `candidates/qwen25/profile.json` | `corpora/five-models/chat.jsonl` | `reference/generation/requirements.txt` |
| `kev-0.8b-cpu.yaml` | `kev-onnx-cpu-v1` | `candidates/kev-0.8b/profile.json` | `reference/corpus/kev-decisions-v1.jsonl` | `reference/kev/requirements.txt` |
| `kev-4b-cpu.yaml` | `kev-onnx-cpu-v1` | `candidates/kev-4b/profile.json` | `reference/corpus/kev-decisions-v1.jsonl` | `reference/kev/requirements.txt` |

Contract, corpus and environment paths in the table are relative to `model-pipeline/`; in the recipe files they start with `model-pipeline/`.

The five-model workflow (`../workflows/five-popular-models-cpu.json`, [FIVE_MODELS.md](../FIVE_MODELS.md)) runs the MiniLM, BGE, MS MARCO, Qwen3 and Qwen2.5 recipes. The E5, bge-base, nomic-embed and Kev recipes run on their own with `model-pipeline run`. The Kev recipes also need `--base-snapshot` ("Kev route" in [ONBOARDING.md](../ONBOARDING.md)).

```sh
.venv/bin/model-pipeline plan model-pipeline/recipes/<recipe>.yaml
```

`plan` validates the recipe and describes its stages without weights, the reference environment or a .NET SDK.
