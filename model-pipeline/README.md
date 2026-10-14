# model-pipeline

The delivery system: model registry, manifest schema, admission records, reference runners, evaluation, and protected evidence. It produces reviewed changes and evidence, run as versioned command-line stages from CI (plan section 8). It is not a workflow platform.

## Layout

| Path | Contents | Backlog |
| --- | --- | --- |
| `schemas/model-manifest.schema.json` | Manifest schema (draft). | P-002 |
| `schemas/stage-results.schema.json` | Stage results format (versions 1 and 2): candidate outputs the evaluator compares. | P-007, P-015 |
| `schemas/benchmark-raw.schema.json` | Raw benchmark result records (`starlings-bench-raw/1`), written by the C# and Python harnesses. | P-008 |
| `registry/` | One `manifest.yaml` per model profile. | P-002, P-021 |
| `admissions/` | Admission records; start from `TEMPLATE.md`. | P-027, stage 1 |
| `protected/` | Reference fixtures and tolerances. Maintainer review only. | P-003, P-007 |
| `reference/` | Reference runners with pinned requirements, and the tokenizer conformance corpus. See `reference/README.md`. | P-003, P-004, P-027 |
| `src/model_pipeline/` | CLI and stage implementations: `manifest.py` (validation), `fixtures.py` (fixture format), `results.py` (stage results format), `evaluate.py` (staged fidelity evaluator; per task `evaluate_*.py`), `evaluate_task_quality.py` (G3 task metrics), `protection.py` (protected evidence rules). | P-002 onward |
| `tests/` | Pipeline tests with synthetic fixtures. | |
| `src/model_pipeline/catalog.py` | Generates `docs/catalog/` (README, compatibility matrix, `catalog.json`) from the registry; `model-pipeline catalog --check` runs in CI. | P-021 |

## Commands

For recipe-driven execution, verified resume, capability inspection, and review bundles, see
[the onboarding guide](ONBOARDING.md). The E5 recipe uses its existing protected evaluator;
the [five-model CPU workflow](FIVE_MODELS.md) adds experimental BERT embeddings/reranking and
Qwen text generation with proposed development comparisons.

```sh
pip install -e '.[dev]'
model-pipeline validate registry
model-pipeline catalog [--check]            # P-021: regenerate docs/catalog (or fail if stale)
model-pipeline bench summarize <raw.jsonl>   # P-008: medians, p5-p95, bootstrap CIs, C# vs Python
pytest
ruff check . && ruff format --check .
```

The onboarding runner orchestrates model-specific reference/backend/.NET tools with resumable stages,
and workflows aggregate independent per-model records. The reference implementations in `reference/`
remain model-specific. Additional model routes and upstream change detection (P-020) remain follow-on work.
Acquisition and verified loading (P-009) is the C# `Starlings.Acquisition` library and tool in `ai-dotnet/`.

## Staged fidelity evaluator (P-007)

```sh
model-pipeline evaluate registry/<family>/<profile>/manifest.yaml --results <dir> --out <dir> [--require-gates G1,G2]
```

Runners write candidate outputs in the stage results format (`src/model_pipeline/results.py`, `schemas/stage-results.schema.json`), one `<stage>.results.json` per stage, with float32 sidecar files for large tensors such as hidden states. The evaluator compares them with the profile's protected fixtures, using only the profile's protected tolerances, in this order:

| Stage | Producer (e5) | Compared with | Gate |
| --- | --- | --- | --- |
| 1 `tokenizer` | `ai-dotnet/tools/Starlings.Fidelity.Tool`: owned tokenizer ids and masks per fixture text | `tokenizer.jsonl`, exact | G1 |
| 2 `export` | `reference/encoder/stage_e5_export.py`: Python ONNX Runtime on `onnx/model.onnx`, and PyTorch hidden states recomputed on the same ids | PyTorch hidden states, `embeddings.jsonl` | G2 |
| 3a `dotnet` from ids | the same C# tool: `E5Encoder` on fixture ids, padded batches | `embeddings.jsonl`; Python ONNX Runtime from stage 2 on the same machine and provider | G2 |
| 3b `dotnet` from text | the same C# tool: `E5TextEmbedder` (owned tokenizer + encoder), single and batch | `embeddings.jsonl` | G2 |

Stage attribution: the report names the first stage that diverges and one finding per failing stage. Stage 2 runs the same artifact as stage 3 in Python, so Python ONNX Runtime failing against PyTorch is an export or backend fault, upstream of .NET; Python passing while C# differs from Python on identical ids beyond the same-machine bound is a .NET fault (binding, preprocessing or pooling); C# matching Python while both miss the reference is attributed to the export; an exact tokenizer and a passing ids path with a failing text path is the orchestration between them (processor). Without a stage 2 run on the same machine, a C# failure against the reference is reported as unattributed. Each finding carries a failure-taxonomy label from plan section 8.

What the evaluator enforces:

- Every protected case is required in every stage; a missing case fails rather than shrinking the set. NaN or Inf fails. A stage that ran other ids, another profile, or an artifact whose SHA-256 differs from the manifest's `origin.files` is invalid.
- Every output path a gate covers (export embeddings, .NET embeddings from ids, .NET embeddings from text) needs its own padded-batch evidence: a batch with more than one row and differing input lengths. Missing or degenerate batch evidence, and cases missing from any stage (candidate or the stage it is compared with), are reported as `missing evidence` against the stage that lacks them, and the gate fails.
- Every key in the tolerances file must be evaluated. A key the evaluator does not implement leaves its gate `not-run`; a key made irrelevant by the manifest (offsets when `behavior.offsets_unit` is `none`) is reported `not-applicable` with the reason. `report_only` metrics (mean, p99, minimum cosine) are reported, never bound. `float.min_cosine` binds the minimum cosine (float64) of every comparison against the reference embeddings; `float.token_embeddings_relative` binds export-stage token embeddings (tensors `token_embeddings` and `reference_token_embeddings`) per case: for each case, `max_i |d_i| / max(1, max_i |ref_i|) <= r`, the largest absolute error relative to the case's largest reference magnitude (the reading the EmbeddingGemma 2 spike measured; maintainer decision 2026-10-11). The report gives the worst case's value (`max_relative`, `worst_case`) and, report only, the per-value figure `max_i |d_i| / max(1, |ref_i|)` (`per_value_max_relative`, `per_value_worst_case`), which never binds. When a profile's protected tolerances bind no `float.last_hidden_state_max_abs`, the hidden-state comparison is reported as not-applicable (measured, never gating). Embedding fixture rows computed with a named prompt (field `prompt` other than `none`) are cases `<id>@<prompt>`; a .NET stage with no `text_embedding` output leaves the text checks not-run.
- The report (`report.json`, `report.md`) has max, mean and p99 absolute error and pass/fail per check and tolerance key, the digests of every input, and gate statuses in the manifest vocabulary (`passed`, `failed`, `not-run`). The evaluator never edits a manifest; a maintainer copies a gate status into the registry by pull request.

Exit codes: 0 no failure, 1 a check failed, 2 bad input, 3 protected evidence refused or changed, 4 a `--require-gates` gate did not pass or the profile has no protected tolerances file yet.

### Protected thresholds

- Fixtures and tolerances are read only from the manifest's `qualification` paths, which must lie inside `model-pipeline/protected/` (no absolute paths, `..`, or symbolic links). The tolerances file must name the same profile.
- There is no option to supply thresholds: `--tolerances`, `--threshold`, `--bound` and similar are refused with exit code 3, and a results file containing a `tolerance`, `threshold` or `bound` key anywhere is rejected.
- The evaluator checks with Git that the tolerances and fixtures it read equal the merge base with `origin/main` (`--protected-base auto`, or a given commit), including uncommitted and untracked changes. If they differ, the run exits 3 and no gate can pass. With `--protected-base none` the run is unverified and no gate can pass either.
- `model-pipeline check-protected`, run by `.github/workflows/protected-evidence.yml` on every pull request, fails when a pull request changes protected evidence (`model-pipeline/protected/`, any `tolerances.*`) without the `protected-change` label, or changes protected evidence together with implementation code (`ai-dotnet/`, `model-pipeline/src|tests/`).

These make an unreviewed threshold change fail visibly. They do not replace code-owner review: CODEOWNERS lists the protected paths, but on the current plan the repository has no branch protection, so neither review nor these checks block a merge by themselves.

Evidence for the e5 CPU profile on real model output: `docs/evidence/p007-e5-onnx-fp32-cpu/`.

Implemented for the `embedding` task, the `entity-extraction` task (manifest task `extraction`, below), the `image-embedding` task (EmbeddingGemma 2 images, below), the `video-embedding` task (EmbeddingGemma 2 videos, below) and the `generation` task (Qwen3.5, below). Task-quality metrics (G3) are `--g3`, below. The Kev tolerances (`protected/kev-0.8b/onnx-fp32/`) use other keys (probabilities, argmax flips, cached vs uncached) and have no registry manifest yet; the evaluator rejects tasks it has no checks for.

A profile whose manifest names a tolerances path that does not exist yet (the path must still lie inside `model-pipeline/protected/`, in an existing directory) is measured, never qualified: every check that needs a bound is `not-run`, both gates are `not-run` (or `failed` if a structural or identity check fails), the report lists `expected_tolerance_keys`, and the exit code is 4. Deleting an existing tolerances file is a protected change (exit 3). A tolerances file placed anywhere else, for example in the results directory, is never read.

### Image embeddings

`src/model_pipeline/evaluate_image.py`, manifest task `image-embedding` (profiles `embeddinggemma-2/image-fp32-*`). Fixtures in the profile's protected directory: `image-embeddings.jsonl` (rendered ids, per image the decoded-RGB and preprocessing digests, the embedding), `images.json` and `tensors.json`. Results use stage results format version 2 with `"task": "image-embedding"`; the C# producer is `ai-dotnet/tools/Starlings.Fidelity.Tool -- embeddinggemma2-images`.

| Stage | Results file | Per-case outputs | Compared with | Gate |
| --- | --- | --- | --- | --- |
| 1 rendering and preprocessing | `tokenizer.results.json` | `input_ids`; per image k `image{k}_rgb_sha256`, `image{k}_resized_sha256`, `image{k}_pixel_values_sha256`, `image{k}_positions_sha256` (strings) | fixture ids and digests, exact (`discrete.*`, `decoder_skiasharp.decoded_rgb_vs_pil`) | G1 |
| 2 export | | | not applicable: the image tolerances bind the C# path; Python ONNX Runtime parity is the protected `parity-cpu.json` | |
| 3 .NET from decoded RGB | `dotnet.results.json` | `embedding`; tensors `soft_tokens` (cases `<row>/image<k>`) and `token_embeddings` (image rows); batches of `embedding` | embedding max abs and min cosine; soft tokens and token embeddings row-relative (max \|d\| / max \|ref\| per image or row) against the reference `.npz` tensors; batches against the stage's single rows; NaN/Inf; provider options; assets against `origin.files` | G2 |

Rows whose image is 16-bit (PIL mode `I;16` in `images.json`) are excluded by contract and listed in the report. The reference tensors are not in Git: pass `--reference-tensors <dir>` (the `.npz` files of `run_embeddinggemma2_image.py --tensors`); every array is checked against `tensors.json` before use. Without them the soft-token and token-embedding checks are not-run and G2 cannot pass.

### Video embeddings

`src/model_pipeline/evaluate_video.py`, manifest task `video-embedding` (profiles `embeddinggemma-2/video-fp32-*`). Fixtures in the profile's protected directory: `video-embeddings.jsonl` (rendered ids, per video the sampled indices, per-frame and preprocessing digests, the embedding), `videos.json` and `tensors.json`. Results use stage results format version 2 with `"task": "video-embedding"`; the C# producer is `ai-dotnet/tools/Starlings.Fidelity.Tool -- embeddinggemma2-videos`.

| Stage | Results file | Per-case outputs | Compared with | Gate |
| --- | --- | --- | --- | --- |
| 1 decoding, sampling, rendering, preprocessing | `tokenizer.results.json` (environment `ffmpeg_version`, `ffmpeg_major`) | `input_ids`; per video k `video{k}_frame_indices` (ints), `video{k}_frame_sha256` (list), `video{k}_frames_sha256`, `video{k}_resized_sha256`, `video{k}_pixel_values_sha256`, `video{k}_positions_sha256`; `image0_rgb_sha256` for rows with a still | fixture ids, indices and digests, exact (`discrete.*`); per-frame RGB under `decoder_ffmpeg.decoded_rgb_vs_torchcodec`, gated only when `ffmpeg_major` equals `decoder_ffmpeg.supported_ffmpeg_major` (another major: reported, check not-run, G1 cannot pass) | G1 |
| 2 export | | | not applicable, as for images | |
| 3 .NET from decoded frames | `dotnet.results.json` | `embedding`; tensors `soft_tokens` (cases `<row>/video<k>`, frames x soft tokens by 512) and `token_embeddings` (media rows); batches of `embedding` | as for images, soft tokens per video | G2 |

The row `vid-array-no-metadata` (frames without timing metadata) is excluded by contract: not required, but evaluated in every stage when any stage's results contain it (a run with `AllowVideosWithoutTiming`); the report lists it under `excluded_by_contract` and, when evaluated, `evaluated_although_excluded`. Reference tensors: `--reference-tensors <dir>` with the `.npz` files of `run_embeddinggemma2_video.py --tensors`, as for images.

### Entity extraction (P-015)

`src/model_pipeline/evaluate_extraction.py`, for span extractors with a count gate (GLiNER2). Fixtures: `extraction.jsonl` in the profile's protected directory (proposed in PR #19 for `gliner2-base-v1/onnx-fp32-cpu`). Results use stage results format version 2 with `"task": "entity-extraction"`.

| Stage | Results file | Per-case outputs | Compared with | Gate |
| --- | --- | --- | --- | --- |
| 1 processor | `tokenizer.results.json` | `normalized_text` (string), `words` (strings), `start_map`, `end_map`, `input_ids`, `attention_mask`, `word_index`, `schema_index` | fixture fields of the same names, exact (`attention_mask`: all ones); character maps also checked well formed against the stage's own text | G1 |
| 2 export | `export.results.json` | `input_ids`, `word_index`, `schema_index` (the fixture inputs it ran), `count_logits`, `span_logits` (valid spans only, row-major over label, start word, width, as `span_logits_valid`), `entities`; tensors `last_hidden_state` and `reference_last_hidden_state` ([tokens, hidden]) | count logits (abs), gate decision (exact), span logits (relative: \|d\| / max(1, \|ref\|)), scores (float32 sigmoid of the span logits, abs), entities (rule below), confidence of matched entities (abs), hidden state (abs) | G2 |
| 3 .NET from inputs | `dotnet.results.json` | same as stage 2 except tensors | same, plus C# vs Python ORT span logits when both stages ran on the same machine and provider | G2 |
| 4 .NET from text | `dotnet.results.json` | `text_entities` | entities rule, confidence | G2 |

An entity result is `{"gate_open": bool, "entities": [{"label", "text", "start", "end", "confidence"}]}`, items in output order (grouped by label in schema label order, then the implementation's order within a label), offsets in Unicode scalars of the normalized text (the manifest's `offsets_unit` must be `unicode-scalars`). Results files must not carry the extraction threshold: the case threshold comes from the fixtures, and a `threshold` key anywhere in a results file is refused like any other threshold.

Padded-batch evidence is required per output path: `span_logits` in stage 2, `span_logits` and `text_entities` in stage 3, each with a batch of more than one row and differing input lengths (`len(input_ids)` of the fixture). Batch rows are compared with the stage's own single rows and with the reference (`count_logits` batches, if present, too; `entities`/`text_entities` batches with the entity rule).

Entity rule (docs/m0/gliner2-p014.md, section 6): label, text, start, end and order must match the reference exactly, with two allowed exceptions, each listed in `entity_exceptions` of the report and in the Markdown summary:

- order swap: two items of one label in a different relative order, only when their reference confidences differ by at most `entities.order_swap_max_confidence_delta`;
- threshold flip: an item present on one side only, only when its reference score is within `entities.threshold_flip_max_score_margin` of the case threshold. Reference scores of spans the reference did not return are the float32 sigmoid of the fixture's span logit.

Anything else fails: a different gate decision, items out of label order, duplicates, an item added or dropped far from the threshold (including one that appears because a flipped item no longer suppresses it), an added item placed against the reference-score order beyond the swap bound, a span wider than the model's maximum width, offsets outside the normalized text, or a `text` that is not `normalized_text[start:end].strip()`. Setting either exception bound to 0 makes the rule exact.

Stage attribution follows the embedding task: invalid identity or inputs is `packaging/harness`; a missing case or missing batch evidence is `missing evidence` on the stage that lacks it; processor mismatches are `processor` (normalization, words, maps, positions) or `tokenizer` (ids, mask only); export logits, gate or hidden state failing is `export/operator or precision`, entities failing with logits in bounds is `threshold or ordering boundary`; C# beyond the same-machine bound is a .NET binding fault, C# entities failing with logits in bounds is `.NET decoding`, batch-only failures are batching, and a text-path failure with an exact processor and a passing inputs path is the `processor` orchestration of the C# text path.

Tolerance keys (flattened YAML paths) the maintainer sets in `protected/<family>/<profile>/tolerances.yaml`; the proposed values are from docs/m0/gliner2-p014.md section 6. Every key except the last is required for its checks to run; any other key leaves its gate `not-run`.

```yaml
profile: gliner2-base-v1/onnx-fp32-cpu
reference: <what the reference outputs are>
discrete:            # all "exact"
  normalized_text: exact
  words: exact
  character_maps: exact      # start_map and end_map
  token_ids: exact           # input_ids
  attention_mask: exact
  word_positions: exact      # word_index
  schema_positions: exact    # schema_index
  gate_decision: exact       # argmax(count_logits) > 0
float:
  last_hidden_state_max_abs: 1.0e-4
  count_logits_max_abs: 5.0e-4
  span_logits_relative: 1.0e-4                       # |d| <= bound * max(1, |ref|), valid spans
  scores_max_abs: 2.0e-5
  confidence_max_abs: 2.0e-5                         # matched entities
  padded_batch_vs_single_span_logits_relative: 1.0e-4
  nan_or_inf: fail
  report_only: [mean_abs, p99_abs]
entities:
  identity: exact
  order_swap_max_confidence_delta: 2.0e-5            # 0 for exact order
  threshold_flip_max_score_margin: 2.0e-5            # 0 for no flips
provider_options: {}                                 # CUDA profile: {use_tf32: 0}
csharp_vs_python_same_machine_and_provider:          # optional; without it the comparison is report-only
  span_logits_relative: <not proposed in P-014>
```

The GLiNER2 stage 1 and 2 producer is `reference/gliner2/stage_gliner2.py` (see `reference/README.md`).

### Generation (P-016, workstream R5)

`src/model_pipeline/evaluate_generation.py`, manifest task `generation` (profiles `qwen3.5-0.8b/ortgenai-*`). It makes the binding Qwen tolerances machine-checked; before it, only the C# tests and `reference/generation/compare.py` checked them. Fixture: `generation.jsonl` in the profile's protected fixtures directory (`protected/qwen3.5-0.8b/reference/`, format `starlings.generation-fixture` version 1). Readers (rows, ids, float32 sidecars) are shared with `evaluate_generation_candidate.py`.

`--results` is the output directory of `ai-dotnet/tools/Starlings.Qwen35.Fidelity --modes parity,semantics` as it is, not stage results files: `results.jsonl` (per case `rendered`, `render_error`, `backend_prompt_ids`, `prompt_ids` fed, `output_ids` with the stop token, `finish_reason`, the streaming and usage fields), `step0/<id>.f32` (log-softmax after the prefill, float32 little-endian), `semantics.json` and `environment.json`. A `threshold`-like key anywhere in them is refused (exit 3).

| Stage | Checks (tolerance key) | Gate |
| --- | --- | --- |
| template (`tokenizer`) | owned template rendering exact (`discrete.template_rendering`); owned tokenizer ids exact (`discrete.prompt_ids`); the tools prompt must be rejected (excluded by contract, listed) | G1 |
| `dotnet-ids`, generation from the fixture prompt ids | greedy ids (`discrete.greedy_ids`: `exact`, `exact_except_listed_margin_below_<m>` or `equal_until_first_divergence_at_margin_below_<m>`; a divergence is allowed only where the fixture `step_margin` at the first differing step is below `m`, and each one is listed); `discrete.finish_reason` (`equal` or `equal_when_ids_equal`); `discrete.step0_argmax`; `float.step0_top5_logprob_max_abs` (reference top 5 from the fixture); `float.step0_logprob_max_abs` (full vocabulary); `float.nan_or_inf`; `discrete.streaming_reassembly`; `discrete.usage`; `provider_options` against `environment.json` | G2 |
| `dotnet-text`, runtime semantics | `float.cached_vs_uncached_logprob_max_abs`, `float.chunked_vs_one_call_prefill_logprob_max_abs` (chunk runs whose one-call prefill failed are listed, not measured), `discrete.concurrent_requests` (`equal_to_sequential`) from `semantics.json` | G2 |

The full reference step-0 vectors (22 x 248,320 float32) are not in Git; the fixture keeps the top 64 per case and each vector's SHA-256. Pass `--reference-tensors <dir>` with `step0/<id>.f32` from `run_reference.py`; every file is checked against the fixture digest. Without them `float.step0_logprob_max_abs` is measured on the 64 reference ids only: above the bound it fails (the full maximum can only be larger), otherwise it is not-run and G2 cannot pass. A missing `semantics.json` or `environment.json` leaves its checks not-run.

```sh
dotnet run --project ai-dotnet/tools/Starlings.Qwen35.Fidelity -c Release -- --bundle <snapshot> --variant fp32x \
  --device cpu --prompts model-pipeline/reference/generation/prompts-v1.jsonl \
  --reference model-pipeline/protected/qwen3.5-0.8b/reference --out <run> --modes parity,semantics
model-pipeline evaluate registry/qwen3.5-0.8b/ortgenai-fp32x-cpu/manifest.yaml --results <run> --out <report> \
  [--reference-tensors <reference run dir>]
```

These are the accepted development fixtures (agent-written prompts), not the P-030 release set; a pass here is G1/G2 evidence on development fixtures only.

## Task quality (G3, workstream R5)

`src/model_pipeline/evaluate_task_quality.py`: held-out task metrics for plan section 10 G3, through `model-pipeline evaluate --g3 retrieval|ner|generation`. It computes metrics on a dataset and compares them with a quality budget read only from the maintainer-owned file `task-quality.yaml` in the profile's protected directory (the directory of the manifest's `qualification.tolerances`, for example `protected/multilingual-e5-small/onnx-fp32-cpu/task-quality.yaml`). **No such file exists yet.** Without it, without an entry for the task, or when the dataset digest is not the one the entry names, the report says "no budget: not a gate result", G3 is `not-run`, and the exit code is 4. With a budget, every listed metric must meet its minimum (exit 0) or G3 fails (exit 1); a changed or untracked budget file is exit 3, as for tolerances.

**Choosing the real G3 datasets (licensed, held out) and the budgets is a maintainer decision: M1/M2 in `docs/plan/production-readiness.md`.** This repository supplies only the formats, the metrics, and a SMOKE example under `model-pipeline/tests/fixtures/g3-smoke/` (hand-written, labeled `SMOKE` in each `dataset.json`, never gate evidence). Agents do not author the sets or the budget file.

Budget file (every value is a minimum; metric names as in the report):

```yaml
profile: multilingual-e5-small/onnx-fp32-cpu
retrieval:
  dataset_sha256: <report dataset.sha256>
  minimum: {ndcg@10: <m>, recall@100: <m>, mrr@10: <m>}
  # int8 browser profiles (kickoff decision 3: at least 0.95 against fp32): needs --baseline
  # minimum: {top10_overlap_vs_baseline.mean: 0.95}
ner:
  dataset_sha256: <digest>
  minimum: {exact.micro.f1: <m>, partial.micro.f1: <m>}
generation:
  dataset_sha256: <digest>
  minimum: {exact_match: <m>}
```

The dataset digest is the SHA-256 over `<file name>\0<file sha256>\n` of `queries.jsonl`, `corpus.jsonl`, `qrels.jsonl` (retrieval), of the gold file (NER) or of the prompts file (generation); every report prints it. Candidate files carry outputs only: a `threshold`-like key is refused (exit 3).

### Retrieval (e5, EmbeddingGemma 2)

Dataset: a directory with

| File | One JSON object per line |
| --- | --- |
| `queries.jsonl` | `{"id": str, "text": str}` |
| `corpus.jsonl` | `{"id": str, "text": str}` (only `text` is embedded) |
| `qrels.jsonl` | `{"query_id": str, "doc_id": str, "relevance": int >= 0}`; every query needs at least one judgment above 0 |
| `dataset.json` (optional) | name, status, provenance, license; copied into the report |

Candidate (`--results`, and `--baseline`): JSON lines `{"kind": "query" | "document", "id": str, "embedding": [float], "prompt": str (optional)}`, one per query and document of the dataset, same dimension, finite. The producer applies the model's prompts (e5 `query: ` / `passage: `; EmbeddingGemma 2 query and document prompts). Producer: `dotnet run --project ai-dotnet/tools/Starlings.Fidelity.Tool -c Release -- retrieval --model e5|embeddinggemma2 --dataset <dir> --out <file>`; the reference runners can write the same format.

Metrics: exhaustive ranking by cosine similarity (ties by document id); per query, then averaged: `ndcg@10` (gain = relevance, log2 discount, as trec_eval and BEIR), `recall@1|5|10|100` (relevant = relevance > 0), `mrr@10`. With `--baseline <fp32 embeddings>`: `top10_overlap_vs_baseline.mean` and `.min`, per query |top 10 of the candidate ∩ top 10 of the baseline| / 10, the browser int8 metric. Image and video retrieval use the same format once a producer embeds media.

### Named-entity recognition (GLiNER2)

Gold: one JSON lines file, per row `{"id": str, "text": str, "labels": [str] (the label set queried), "entities": [{"start": int, "end": int, "label": str}]}`, offsets in Unicode code points of `text` (Python `str` indices), end exclusive; `entities` may be empty; a gold entity may carry `text` (ignored). Predictions (`--results`): per row `{"id": str, "entities": [{"start", "end", "label", ...}]}` with the same offset unit; labels must be in the row's labels. Producer: `dotnet run --project ai-dotnet/tools/Starlings.Fidelity.Tool -c Release -- ner --dataset <gold.jsonl> --out <file>` (`Gliner2EntityExtractor.Extract(text, labels)` at the extractor's default operating point).

Metrics: span-level precision, recall and F1, `exact` (same start, end and label) and `partial` (same label, any overlap), with one-to-one matching (largest overlap first), so a duplicate prediction is a false positive. Per label (`exact.label.<label>.f1`), micro (pooled counts: `exact.micro.f1`) and macro (mean over labels with at least one gold entity; labels without gold entities are listed and not averaged).

### Generation (Qwen3.5, exact match)

Dataset: one prompts file in the format of `reference/generation/prompts-v1.jsonl` (`id`, `messages`, `max_new_tokens`, optional `chat_template_kwargs`) whose rows also carry the reference `answer` (a string); `dataset.json` beside it (optional) is copied into the report. Candidate (`--results`, and `--baseline`): JSON lines with `id` and `output_text` (the decoded completion), one per prompt, or a runner's output directory (its `generations.jsonl`, else `results.jsonl`). Producer: `dotnet run --project ai-dotnet/tools/Starlings.Qwen35.Fidelity -c Release -- generation --bundle <snapshot> --prompts <prompts.jsonl> --out <dir> [--variant fp32x|fp16x] [--device cpu|cuda] [--threads N]` (the public chat path: owned template and tokenizer, `enable_thinking` from `chat_template_kwargs`, greedy, the row's `max_new_tokens`). The `results.jsonl` of `--modes parity` and of the transformers reference (`reference/generation/run_reference.py`) have the same fields.

Metric `exact_match`: the fraction of rows whose answer matches. The answer is the content of the last `\boxed{...}` of the output (balanced braces), else the last number in it; `$`, `,`, spaces, `%` and a trailing `.` are dropped and a `\text{...}` wrapper is removed; it matches when both sides parse as numbers equal within 1e-6, or the normalized strings are equal. With `--baseline <reference generations>` the report adds `baseline_exact_match` and the rows whose correctness flips, which trace a budget miss to near-tie divergences. The report lists every row's extracted answer.

Python runs reference and export jobs only. Certified in-process C# inference never needs it.
