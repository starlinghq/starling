# Five-model CPU onboarding

The [workflow definition](workflows/five-popular-models-cpu.json) runs five independent recipes
sequentially. Each verifies the original snapshot, produces a fresh framework reference, executes
the backend and owned .NET implementation, and checks its predeclared development policy.
EmbeddingGemma 2 is excluded from this workflow and is handled separately. The BERT-route
recipe [`nomic-embed-cpu.yaml`](recipes/nomic-embed-cpu.yaml) (`nomic-ai/nomic-embed-text-v1.5`,
768 dimensions, mean pooling, no normalization, 8,192 tokens) and the Kev recipes are not in this
workflow; run them with `model-pipeline run` ([ONBOARDING.md](ONBOARDING.md)).

All five profiles are experimental. A completed workflow means its requested checks passed;
it does not adopt references, award protected G1/G2 qualification, change support status, approve
a release, or establish controlled performance. The existing E5 protected route is described
in [ONBOARDING.md](ONBOARDING.md).

## Profiles and licenses

Original repository code is licensed under the MIT License or the Apache License, Version 2.0, at your option. Upstream model assets retain the following licenses;
the linked model cards refer to the exact admitted revisions. Candidate contracts under
[`candidates/`](candidates/) lock the complete source file sizes and SHA-256 values.

| Binding ID | Pinned upstream model | Recipe | Behavior | Asset license |
| --- | --- | --- | --- | --- |
| `qwen3` | [Qwen/Qwen3-0.6B @ c1899de289a04d12100db370d81485cdf75e47ca](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/README.md) | [`qwen3-cpu.yaml`](recipes/qwen3-cpu.yaml) | Greedy text chat, thinking disabled | Apache-2.0 |
| `qwen25` | [Qwen/Qwen2.5-0.5B-Instruct @ 7ae557604adf67be50417f59c2c2f167def9a775](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/README.md) | [`qwen25-cpu.yaml`](recipes/qwen25-cpu.yaml) | Greedy text chat, upstream default system instruction | Apache-2.0 |
| `minilm` | [sentence-transformers/all-MiniLM-L6-v2 @ 1110a243fdf4706b3f48f1d95db1a4f5529b4d41](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md) | [`minilm-cpu.yaml`](recipes/minilm-cpu.yaml) | 384 dimensions, masked mean/L2, 256 tokens | Apache-2.0 |
| `bge-small` | [BAAI/bge-small-en-v1.5 @ 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a](https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/README.md) | [`bge-small-cpu.yaml`](recipes/bge-small-cpu.yaml) | 384 dimensions, CLS/L2, 512 tokens, explicit query instruction | MIT |
| `ms-marco` | [cross-encoder/ms-marco-MiniLM-L6-v2 @ 233902d25c440f23af6f7d6e94d2946bac0bee0a](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/blob/233902d25c440f23af6f7d6e94d2946bac0bee0a/README.md) | [`ms-marco-cpu.yaml`](recipes/ms-marco-cpu.yaml) | Paired inputs, raw scalar logits, 512 tokens | Apache-2.0 |

The BERT profiles use ONNX Runtime 1.30.0 CPU fp32 and right padding. Token limits include special
tokens. MiniLM has no prompt. BGE's `query` prompt prepends
`Represent this sentence for searching relevant passages: `; plain inputs and passages receive
no automatic prefix. MS MARCO uses longest-first pair truncation and identity activation, so its
scores are not calibrated probabilities. See the [BERT API](../ai-dotnet/src/Starlings.Models.Bert/README.md).

The Qwen profiles use ONNX Runtime GenAI 0.17.1 CPU fp32. The proposed corpus uses a 16-token
generation limit. Qwen3 explicitly uses `enable_thinking=false`; tools, multimodal input and
sampling are unsupported. Each request owns a fresh generator/cache. See the
[Qwen API](../ai-dotnet/src/Starlings.Models.QwenChat/README.md) for template, padding
overlay and cancellation limits.

## Prepare local bindings

Run from the repository root on Linux x64 with the SDK pinned in `global.json` on `PATH`.
Use the pipeline development environment from [ONBOARDING.md](ONBOARDING.md#prepare).
Create separate reference environments for the encoder and generation locks:

```sh
uv venv --python 3.11 .models/venv-reference
uv pip install --python .models/venv-reference/bin/python --require-hashes \
  -r model-pipeline/reference/encoder/requirements.txt \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match
uv pip install --python .models/venv-reference/bin/python -e model-pipeline

uv venv --python 3.12 .models/venv-generation
uv pip install --python .models/venv-generation/bin/python --require-hashes \
  -r model-pipeline/reference/generation/requirements.txt \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match
uv pip install --python .models/venv-generation/bin/python -e model-pipeline
```

The generation lock includes a CUDA-flavored PyTorch wheel and its dependencies; reference and
backend execution still select CPU. This lock is not a CPU-only installation. Do not substitute
package versions: the environment stage compares installed distributions with the recipe's lock
and records the actual toolchain identity. Hash-checked installation establishes wheel provenance;
the runtime version probe alone does not.

Provide a separate, immutable local snapshot for each model with every file declared in its
candidate contract, including original reference weights and configuration/tokenizer assets.
The BERT snapshots also require their pinned `onnx/model.onnx`. Qwen bundles are built from the
original snapshots by the workflow; a prebuilt runtime bundle does not replace them. The workflow
does not download or modify snapshots. Keep weights, generated graphs and large tensors outside Git.

Create an ignored local JSON file such as `.models/five-model-bindings.json`. Every path must be
absolute; replace `/path/to/...` with your paths. Binding IDs must exactly match the workflow,
with no extra entries, and each snapshot must be distinct.

```json
{
  "qwen3": {"snapshot": "/path/to/snapshots/qwen3", "reference_python": "/path/to/repo/.models/venv-generation/bin/python"},
  "qwen25": {"snapshot": "/path/to/snapshots/qwen25", "reference_python": "/path/to/repo/.models/venv-generation/bin/python"},
  "minilm": {"snapshot": "/path/to/snapshots/minilm", "reference_python": "/path/to/repo/.models/venv-reference/bin/python"},
  "bge-small": {"snapshot": "/path/to/snapshots/bge-small", "reference_python": "/path/to/repo/.models/venv-reference/bin/python"},
  "ms-marco": {"snapshot": "/path/to/snapshots/ms-marco", "reference_python": "/path/to/repo/.models/venv-reference/bin/python"}
}
```

Machine paths belong in bindings, not portable recipes. Output roots must not overlap snapshots,
bindings, interpreters, recipes or their inputs; repository outputs must be below `artifacts/`
or `.models/` and contain no symbolic links.

## Execute, resume and review

```sh
.venv/bin/model-pipeline plan model-pipeline/recipes/minilm-cpu.yaml
.venv/bin/model-pipeline workflow model-pipeline/workflows/five-popular-models-cpu.json \
  --bindings .models/five-model-bindings.json \
  --out artifacts/onboarding/five-models \
  --threads 2 --timeout 1800
```

`--threads` bounds CPU workers; `--timeout` is a per-subprocess limit in seconds, not a deadline
for the whole workflow. Models run sequentially in definition order. A failed model retains its
logs and attempts; later independent models still run. `workflow.json` records aggregate status
and links to each `<binding-id>/run.json`. Exit `0` means all requested model runs completed,
`1` means at least one failed, and `2` means invalid input or a refused operation.

BERT stages are `assets`, `environment`, `reference`, `onnx`, `dotnet`, `evaluate`, `review`.
Qwen stages replace `onnx` with `bundle` and `backend`. The evaluators require complete case and
path evidence, valid identities/shapes, exact discrete comparisons, finite values and the
predeclared continuous bounds. Fresh references and comparison policies remain proposals,
even when development parity passes; protected G1/G2 remain unapproved.

Repeat the workflow command with `--resume`. Cache reuse verifies current declared inputs,
code, environment, dependencies and the entire recorded output tree. Changed, missing, corrupt
or extra output files invalidate the affected stage and dependents. Independent valid models
can stay cached. Use one writer per output root; summaries from prior attempts are retained in
`history/`. Do not edit artifacts to turn a failure into a pass.

Export each completed model independently to a new review directory:

```sh
.venv/bin/model-pipeline propose artifacts/onboarding/five-models/minilm \
  --out artifacts/onboarding/reviews/minilm
```

The exporter verifies the run before copying it. Read `stages/review/review.json`,
`stages/review/review.md` and the evaluation report, and retain `workflow.json` for the aggregate
result. Local digests detect corruption; they are not signed release attestations.

## Optional package-only distribution evidence

Commit all relevant candidate, runtime, tokenizer, tool, recipe, corpus and build sources first.
Then request distribution for every model, using the same run root to reuse valid inference:

```sh
.venv/bin/model-pipeline workflow model-pipeline/workflows/five-popular-models-cpu.json \
  --bindings .models/five-model-bindings.json \
  --out artifacts/onboarding/five-models \
  --threads 2 --timeout 1800 --resume --distribution
```

This stage refuses differences between the relevant working sources and committed HEAD. It packs
committed source, creates an external consumer with package references and isolated NuGet caches,
checks package/native identities, and seeds verified real model assets. Packing/restoring may
need network access. Before inference it removes the checkout and package caches, then runs the
published consumer under `unshare -rn` without network interfaces. Linux user/network namespaces
must be enabled; inability to create the namespace fails the stage, with no environment-only
fallback. Python is used for development evaluation, not by the consumer's inference process.

The output retains package evidence, locks, command logs, `distribution.json` and
`consumer-parity.json`; transient workspaces and model copies are removed. Passing distribution
adds development package-consumption evidence. References and policies still require independent
maintainer adoption before protected release qualification.

## Tokenizer assets in CI

```sh
tools/fetch-tokenizer-assets.sh
STARLING_REQUIRE_ASSETS=1 dotnet test --solution ai-dotnet/AI.slnx
```

The fetch script downloads only pinned tokenizer assets, including the `tokenizer.json` of all
four BERT-route models (the three here and nomic-embed-text-v1.5), and verifies SHA-256 against
the pins in `docs/m0/hub-pins.md` and `candidates/*/profile.json`. Its default layout is
`.models/<Hub repo>/<full commit>/tokenizer.json`; set `STARLING_MODELS` consistently for fetching
and tests to use another asset root. Existing correct files are reused. Missing assets fail
tests when `STARLING_REQUIRE_ASSETS=1`; a wrong digest always fails. CI already calls this script
and keys its asset cache on the script content. These tokenizer assets do not supply the full
snapshots required for model inference, and CI fetching does not grant model qualification.
