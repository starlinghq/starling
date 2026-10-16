---
name: starling-add-model
description: Use when contributing a new model to starlinghq/starling — checking whether it fits an existing onboarding route, adding it to that route (or proposing a new one), running the model-pipeline onboarding workflow to compare the .NET implementation against a fresh Python reference, and opening a draft pull request that adds the model as Experimental. Covers what a contributor may change and what only the maintainer may change.
---

# Adding a model to Starling

Starling admits a model only with evidence: its files are pinned by commit and SHA-256, a Python reference run produces the expected outputs, and the .NET implementation is compared against that reference by an independent evaluator. A contributor's PR brings a model in as **Experimental** with *proposed* evidence. Promotion (Preview, Stable) is the maintainer's decision after the release gates G0–G7.

Work from a clone of https://github.com/starlinghq/starling on Linux x64. Read `CLAUDE.md` first; its rules are enforced by CI and review.

## 0. Rules that will get a PR rejected

- Never commit weights or large assets (`.onnx`, `.safetensors`, `.gguf`, `.pt`, `.bin` checkpoints). CI fails on them.
- Never edit `model-pipeline/protected/` or any `tolerances.*` file, and never edit a fresh reference to make a comparison pass. Fresh references are proposals; the maintainer adopts them into `protected/` in a separate `protected-change` PR.
- Never set a gate to `passed` or raise `maturity`. Only the maintainer does.
- Pin upstream code and weights by **full commit SHA**, never a branch or tag.
- Tokenization comes only from `Starlings.Tokenizers` (separate repo, starlinghq/tokenizers). Never add `Microsoft.ML.Tokenizers` or another tokenizer under `ai-dotnet/src/`. If the model needs a tokenizer type that `Starlings.Tokenizers` doesn't have, that work happens in the tokenizers repo first.
- Every capability gets an explicit status (`supported`, `unsupported`, `experimental`, `untested`). Missing never means supported.
- Record any adopted third-party source in `ai-dotnet/THIRD-PARTY-NOTICES.md` (and `ai-dotnet/PROVENANCE.md` for ported code) in the same change.
- After two unsuccessful repair attempts on the same failure, stop and report the evidence instead of continuing to patch.
- Open the PR as a **draft**.

## 1. Start with an issue

Open a [model request](https://github.com/starlinghq/starling/issues/new/choose) (`model-request` template): model, intended use, license, suggested runtime route, hardware. A new architecture family needs a **design proposal** issue before code. Admission is scored on demand, backend fit, reuse, licensing, testability and maintenance (plan §7, `docs/plan/fast-faithful-ai-dotnet.md`); missing redistribution rights block admission outright, so check the code license, weights license and tokenizer terms first.

## 2. Find the route

Set up the pipeline environment once (from the repo root):

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e './model-pipeline[dev]'
```

Download the model snapshot **at a pinned revision** outside the repository (or below `.models/`, which is ignored), then inspect it:

```sh
curl -s https://huggingface.co/api/models/<org>/<model> | python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])"  # current full SHA
# Download only the files the route pins (the BERT set shown, same as candidates/minilm/profile.json);
# a full repository can be five times larger.
uvx --from huggingface_hub hf download <org>/<model> \
  1_Pooling/config.json README.md config.json config_sentence_transformers.json data_config.json \
  model.safetensors modules.json onnx/model.onnx sentence_bert_config.json special_tokens_map.json \
  tokenizer.json tokenizer_config.json vocab.txt \
  --revision <full-commit-sha> --local-dir .models/snapshots/<name>
.venv/bin/model-pipeline inspect .models/snapshots/<name>
```

`inspect` is read-only and only ever suggests `e5-onnx-cpu-v1` or `unknown`, so decide the route yourself:

| Route (plugin) | Fits when the model is... | Code to extend |
| --- | --- | --- |
| `bert-onnx-cpu-v1` | A BERT encoder, uncased WordPiece (vocab 30,522), hidden size 384, right padding, whose Hub repo ships `onnx/model.onnx`; embeddings (mean or CLS pooling, L2-normalized) or a cross-encoder reranker | `model-pipeline/src/model_pipeline/onboarding_bert.py`, `ai-dotnet/src/Starlings.Models.Bert/` |
| `qwen-genai-cpu-v1` | A Qwen-family decoder for ONNX Runtime GenAI 0.17.1, CPU fp32 | `model-pipeline/src/model_pipeline/onboarding_generation.py`, `ai-dotnet/src/Starlings.Models.QwenChat/` |
| `e5-onnx-cpu-v1` | Exactly the pinned multilingual-e5-small (the registry route) | none; it refuses other models by design |
| none | Anything else: a different architecture, hidden size, tokenizer, runtime or precision | a new plugin (step 6) |

Every route is an explicit allow-list: each contract is checked field by field in Python **and** in C#, and anything not listed is refused. That is deliberate. Adding a model is a code change, not a config change.

## 3. Add a model to an existing route (BERT example)

Use the existing MiniLM files as templates. To find every place they are registered, search for both the repository name and the candidate directory: `grep -rln "all-MiniLM-L6-v2\|candidates/minilm\|minilm-cpu" --exclude-dir=docs --exclude-dir=.git .`. Leave `FIVE_MODELS.md` and the five-model workflow alone unless your model belongs to that group. A new BERT-route model touches:

1. **`model-pipeline/candidates/<name>/profile.json`**: the candidate contract (schema `model-pipeline/schemas/onboarding-candidate.schema.json`). Copy `candidates/minilm/profile.json`, then:
   - Set `id`, `origin.repo` and `origin.revision` (full SHA).
   - Set `origin.files`: path, size and SHA-256 for the same file set the template pins, taken from your pinned snapshot (`sha256sum`, `stat -c %s`). The `assets` stage verifies each one.
   - Set `task` and `behavior` (`max_tokens`, `pooling`, `prompts`, …). For `max_tokens`, use `max_seq_length` from `sentence_bert_config.json`. It is what the reference runner enforces, even when the model card says something else.
   - Keep `evaluation.status: "proposed"` and the existing `evaluation.policy` bounds. Do not loosen them.

   The candidate has no license fields. State the code license, weights license and tokenizer terms in the issue and in the PR description.
2. **`model-pipeline/recipes/<name>-cpu.yaml`**: copy `recipes/minilm-cpu.yaml` and point `manifest` at your candidate. Reuse the shared corpus unless the task needs another one.
3. **`model-pipeline/src/model_pipeline/onboarding_bert.py`**: add the repo to `CONTRACTS` as `(task, max_tokens, pooling)`. If it needs a prompt, extend the `prompts` rule.
4. **`ai-dotnet/src/Starlings.Models.Bert/BertModelProfile.cs`**: add the same repo to the `repository switch`, with identical values. Python and C# must agree exactly.
5. **Tests:**
   - `model-pipeline/tests/test_onboarding_bert.py`: accept the new contract and refuse a near-miss, for example a wrong pooling.
   - `ai-dotnet/tests/Starlings.Models.Bert.Tests/ContractTests.cs`: add an `InlineData` row and a near-miss case. Also update the `ProfileData` helper, which builds each test profile's `max_tokens`, `pooling` and `prompts` per repository.
6. **`tools/fetch-tokenizer-assets.sh`**: add `"<repo>|<sha>|tokenizer.json|<sha256>"`, so the pinned asset list stays complete, the same as the existing BERT entries.
7. **`ai-dotnet/src/Starlings.Models.Bert/README.md`**: list the model, its status and anything it doesn't support.
8. **Optionally, `model-pipeline/workflows/*.json`**: add the model to a workflow only if it belongs to that group.

For the Qwen route, the pattern is the same with `onboarding_generation.py` and the generation reference environment.

## 4. Run the onboarding pipeline

Create the reference environment (hash-locked; the encoder route shown; use `reference/generation/requirements.txt` with Python 3.12 for generation):

```sh
uv venv --python 3.11 .models/venv-reference
uv pip install --python .models/venv-reference/bin/python --require-hashes \
  -r model-pipeline/reference/encoder/requirements.txt \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match
uv pip install --python .models/venv-reference/bin/python -e model-pipeline
```

Check the recipe, then run it. Use absolute paths; outputs inside the repo must be below `artifacts/` or `.models/`:

```sh
.venv/bin/model-pipeline plan model-pipeline/recipes/<name>-cpu.yaml
.venv/bin/model-pipeline run model-pipeline/recipes/<name>-cpu.yaml \
  --snapshot "$PWD/.models/snapshots/<name>" \
  --reference-python "$PWD/.models/venv-reference/bin/python" \
  --out "$PWD/artifacts/onboarding/<name>" --threads 2 --timeout 1800
```

The BERT stages are `assets → environment → reference → onnx → dotnet → evaluate → review`:
- verify the snapshot against your candidate's digests;
- build the reference environment;
- run the PyTorch reference;
- run the upstream ONNX graph from Python;
- run the .NET implementation (from token ids and from text);
- compare all of them under the candidate's policy.

The exit status means: `0` passed, `1` a check failed, `2` bad input or a refused operation. Rerun with `--resume` after a fix; valid stages are reused. **Don't edit artifacts to turn a failure into a pass.** A failure is the finding: either the contract is wrong (fix the candidate or the code) or the .NET implementation differs from upstream (fix the implementation). The run keeps logs per stage under `stages/<name>/`, and failures under `failure-task-*.json`.

When it passes, export it for review and summarize the comparison:

```sh
.venv/bin/model-pipeline propose artifacts/onboarding/<name> --out artifacts/onboarding/reviews/<name>
.venv/bin/python skills/starling-add-model/summarize_report.py artifacts/onboarding/<name>
```

The summary is a Markdown table with the largest difference against each bound, for every comparison: reference, Python ONNX, and .NET from token ids and from text. The full per-case evidence is in `stages/evaluate/<hash>/output/report.json`, and the review verdict is in `stages/review/<hash>/output/review.md`. A clean BERT run passes every check, with differences around 1e-7 against bounds of 1e-5.

Package-only consumer evidence (`--distribution`, described in `model-pipeline/FIVE_MODELS.md`) is optional and needs everything committed first. Skip it in a first PR unless the maintainer asks for it.

## 5. Check locally, then open a draft PR

Run what CI runs. Use the .NET SDK version pinned in `global.json`, and the `.venv` from step 2 for the Python tools:

```sh
python3 -I tools/check-boundary.py
tools/fetch-tokenizer-assets.sh
dotnet test --solution ai-dotnet/AI.slnx          # a single project: dotnet test --project <dir>
cd model-pipeline
../.venv/bin/python -m pytest
../.venv/bin/ruff check . && ../.venv/bin/ruff format --check .
../.venv/bin/model-pipeline validate registry && ../.venv/bin/model-pipeline catalog --check
cd ..
```

Tests that need model weights are skipped when the weights are absent; that's expected. If you changed a package reference, run `dotnet restore` and commit the updated `packages.lock.json`. A model added to an existing route only reuses that route's code, so it needs no `THIRD-PARTY-NOTICES.md` entry. Ported or adapted source does need one.

The PR (template `.github/pull_request_template.md`) contains:
- the candidate, the recipe, and the route code with its tests in Python and C#;
- the tokenizer asset line and README updates;
- under **Evidence**: the run's exit status, the `summarize_report.py` table, the total number of checks, the pinned revision and the reference environment;
- the licenses: code, weights and tokenizer terms;
- every unsupported feature, listed explicitly.

It must **not** contain weights, artifacts, protected files, tolerance changes, gate statuses or maturity changes. Mark it as a draft. The model lands as Experimental.

## 6. A model that fits no route

Open a design proposal first. Then follow "Add another execution plugin" in `model-pipeline/ONBOARDING.md`. In short:
- specify the upstream behavior contract and the exact source assets;
- implement `describe_plan` and `build_stages` with `Stage(name, dependencies, inputs, parameters, execute)`;
- register the plugin in recipe validation and CLI dispatch;
- declare every input that affects a stage's result;
- emit evidence in the evaluator's format;
- add cheap orchestration and failure tests, then qualify a real model.

A new task (generation, extraction, vision) needs its own evaluator checks; an embedding evaluator can't qualify it. A new .NET library follows `ai-dotnet/src/Starlings.Models.E5/` as its template: the `*ModelAssets` lock, `*Capabilities` with profiles, tests in `ai-dotnet/tests/`, a sample. Add the projects to `ai-dotnet/AI.slnx`; package versions are central in `ai-dotnet/Directory.Packages.props`. Expect several PRs: plugin, library, then evidence.

## 7. What happens after your PR (maintainer only)

You don't need to do these, and your PR must not attempt them:
- **References and tolerances:** the maintainer reviews the proposed references and moves them byte-identical into `model-pipeline/protected/` in a `protected-change` PR, and sets the tolerances.
- **Hosted graphs:** Starling-built graphs are hosted under the `Starlinghq` Hugging Face organization, and the maintainer creates those repositories.
- **Registry and maturity:** writing the registry manifest (`model-pipeline/registry/<family>/<profile>/manifest.yaml`), G3 task datasets and budgets, gate statuses, and maturity changes.
- **Qualification CI:** wiring the profile into `.github/workflows/qualification.yml`.

Preview needs G0, G1, G2 and G6 passed; Stable needs all of G0–G7, plus an owner and regression support (`docs/support-policy.md`).
