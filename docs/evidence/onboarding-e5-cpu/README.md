# Real E5 onboarding validation

Validated implementation commit `b8d2395d47be6e1811988ac711b9aa69cd4c4cd3` on Linux x64 CPU, 2026-10-10 local time (2026-10-11 UTC). Later evidence-only commits do not change the tested implementation.

The pinned `multilingual-e5-small/onnx-fp32-cpu` recipe ran the original Python reference, Python ONNX Runtime, the owned .NET tokenizer/encoder, the protected evaluator, and a real-model clean-consumer package drill. G1 and G2 passed with no first divergence. The evaluator checked 46 embedding cases, exact token IDs/masks, single and padded batches, final vectors, and intermediate hidden states. See [fidelity-report.md](fidelity-report.md), [the raw report](fidelity-report.json), and [summary.json](summary.json).

The [distribution log](distribution.log) records the clean clone, locked package builds, isolated NuGet/model caches, package-byte and dependency checks, explicit acquisition, native ORT 1.30.0 loading, and execution with a network namespace containing no interfaces. The typed API, Extensions.AI generator, and ML.NET transformer reproduced the protected embeddings (maximum absolute error about 2.33e-7 against the 1e-5 bound). No foreign managed/native tokenizer was loaded. GLiNER2 was packed and its sample built, but its model was not run.

## Workflow checks

- A full uncached run, including distribution, completed in 285.69 seconds. A verified resume took 9.26 seconds and reused all eight stages. These are observed shared-workstation timings, not a controlled performance benchmark or an estimate for onboarding a novel model.
- Appending bytes to the generated review Markdown caused `propose` to exit 2 and refuse the bundle. Resuming rebuilt only `review`; the seven model/distribution stages remained cached. The subsequent proposal export verified and copied 57 files.
- A separate snapshot view with an altered config failed size/SHA-256 verification at `assets` (exit 1), blocked all downstream inference, and produced a structured failure task. The original source snapshot was untouched.
- EmbeddingGemma metadata inspection identified its unsupported Split configuration, BPE unknown token/byte fallback, and decoder operations. It did not claim an execution route or qualify that model.

## Reproduce

Prepare the environments and complete source snapshot as described in [the onboarding guide](../../../model-pipeline/ONBOARDING.md), and put the pinned .NET SDK on PATH. Run from the repository root:

```sh
.venv/bin/model-pipeline run model-pipeline/recipes/multilingual-e5-small-cpu.yaml \
  --snapshot /path/to/snapshot-614241f6 \
  --reference-python .models/venv-reference/bin/python \
  --out artifacts/onboarding/e5-cpu --threads 2 --timeout 1800 --distribution

# Repeat with --resume; every unchanged, intact stage should be cached.
.venv/bin/model-pipeline run model-pipeline/recipes/multilingual-e5-small-cpu.yaml \
  --snapshot /path/to/snapshot-614241f6 \
  --reference-python .models/venv-reference/bin/python \
  --out artifacts/onboarding/e5-cpu --threads 2 --timeout 1800 --distribution --resume

.venv/bin/model-pipeline propose artifacts/onboarding/e5-cpu \
  --out artifacts/onboarding/e5-review
```

## Regression checks

271 Python tests passed; Ruff lint/format checks passed; all three registry manifests validated. The tokenizer suite passed 1,117 tests with pinned assets required and no skips. The .NET suite passed 233 tests with the real E5 snapshot, with 10 skips for optional network/acquisition checks and the unrelated GLiNER2 graph. The tokenizer independence check passed.

Protected fixtures and tolerances were not changed. Fresh references remain proposals. This evidence validates the declared E5 CPU workflow and distribution checks; G0/G3/G4/G5/G7 and full release promotion remain outside this pipeline run. No weights or large tensors are committed here; complete local run attempts and the portable bundle are under ignored `artifacts/onboarding/`.
