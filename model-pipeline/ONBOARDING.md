# Recipe-driven model onboarding

The onboarding CLI runs a versioned recipe, records exact inputs and outputs, resumes verified stages, and exports evidence for review. The first execution plugin wraps the existing **multilingual-e5-small / ONNX fp32 CPU** tools. Additional experimental BERT and Qwen plugins run the [five-model CPU workflow](FIVE_MODELS.md) with proposed development comparisons; the BERT plugin also runs the `nomic-embed-text-v1.5` recipe outside that workflow. The experimental Kev plugin runs the two Kev decision checkpoints ([Kev route](#kev-route)). The registered plugins are listed in `src/model_pipeline/onboarding_plugins.py`; recipes and candidates are described in [`recipes/README.md`](recipes/README.md) and [`candidates/README.md`](candidates/README.md). These routes support Linux CPU execution; other models still need their own execution plugin and independent qualification contract.

This is the implementation of the first operational increment in [the work plan](../docs/plan/model-onboarding-pipeline.md). Python coordinates development/reference work; delivered .NET inference still has no Python dependency.

## Prepare

Run from the repository root. Use the .NET SDK pinned by `global.json` and Python 3.11. Put the SDK's `dotnet` executable on `PATH` (a runtime-only installation is insufficient).

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e './model-pipeline[dev]'

uv venv --python 3.11 .models/venv-reference
uv pip install --python .models/venv-reference/bin/python --require-hashes \
  -r model-pipeline/reference/encoder/requirements.txt \
  --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match
uv pip install --python .models/venv-reference/bin/python -e model-pipeline
```

Supply a full local snapshot of `intfloat/multilingual-e5-small` at commit `614241f622f53c4eeff9890bdc4f31cfecc418b3`, including the reference weights, configuration/tokenizer files, and `onnx/model.onnx`. The runtime-only two-file acquisition cache is insufficient for executing the original Python reference. The `assets` stage verifies every file declared by the registry manifest against its size and SHA-256 before running anything expensive. It also records ancillary files that upstream libraries may read. This command does not download or mutate source snapshots.

The reference environment is separate from pipeline development dependencies. A run records installed versions and rejects a mismatch with the reference requirements lock. Environment installation should use the lock's hashes; the runtime check verifies installed version identities, not the provenance of every installed wheel. Model execution runs in offline mode; package restore and the optional distribution drill may need network access.

## Inspect and plan

```sh
.venv/bin/model-pipeline inspect /path/to/pinned-snapshot
.venv/bin/model-pipeline plan model-pipeline/recipes/multilingual-e5-small-cpu.yaml
```

Inspection reads bounded local JSON files without importing upstream model code. It reports observed architecture/configuration, tokenizer operations and parameters, sentence-transformers modules, and missing or unknown capabilities. A suggested route is a **candidate**, not a supported-model claim. Malformed or unsupported metadata remains visible. Inspection currently describes the tokenizer implementation in this checkout; it does not predict future component support.

`plugin_route` names one of these routes, or `unknown` with the reason:

| Route | Suggested when |
| --- | --- |
| `e5-onnx-cpu-v1` | The pinned e5 geometry: `BertModel`, 384 wide, 512 positions, Unigram tokenizer, mean pooling and a `Normalize` module. |
| `bert-onnx-cpu-v1` | `model_type` `bert`, `xlm-roberta` or `nomic_bert`, a WordPiece or Unigram tokenizer whose every component is supported, and either a sentence-transformers `Pooling` module with exactly one of mean or CLS pooling (task `embeddings`; `l2` normalization when a `Normalize` module follows) or, without one, a single-logit `BertForSequenceClassification` or `XLMRobertaForSequenceClassification` head (task `reranking`). `facts` carries the observed width, positions, pooling and normalization. |
| `qwen-genai-cpu-v1` | `model_type` `qwen2` or `qwen3` with the matching `Qwen2ForCausalLM` or `Qwen3ForCausalLM` architecture and a supported BPE tokenizer. |

Each plugin still accepts only the repositories of its own contract table and refuses everything else, so a suggestion is where to start, not a result.

## Scaffold a BERT-route candidate

```sh
.venv/bin/model-pipeline scaffold <name>            # print the generated pieces as JSON
.venv/bin/model-pipeline scaffold <name> --write    # write them into the checkout
```

For `candidates/<name>/profile.json` on the `bert-onnx-cpu-v1` route, `scaffold` derives the recipe (`recipes/<name>-cpu.yaml`), the repository's entry in the route contract table `routes/bert-onnx-cpu-v1/contracts.json`, the `tools/fetch-tokenizer-assets.sh` line, the profile row of the `Starlings.Models.Bert` README, the `InlineData` row of the .NET contract tests, and a C# `ModelAssetLock` of the runtime files (printed only). The contract table is the one allow-list of the route: `onboarding_bert.py` loads it and `Starlings.Models.Bert` embeds it as a resource, and both validate it with the same rules before use. Pieces already present are reported as `unchanged`; a candidate that contradicts an existing table entry, or that the route does not implement, is refused and nothing is written. The scaffold changes no gate, tolerance, protected file or maturity; the evidence still comes from `run`.

## Container image

`model-pipeline/Dockerfile` builds a contributor image with the .NET SDK `global.json` pins (by digest), uv (by version and wheel SHA-256), and the hash-locked reference environments of the CPU routes under `/opt/venvs/` (`encoder`, `generation`, `whisper`; build argument `ENVIRONMENTS` selects them). The checkout is mounted at `/work` and the model cache at `/models` (`STARLING_MODELS`); the entrypoint installs the mounted `model-pipeline` into every environment without dependencies, offline and without build isolation (setuptools is preinstalled), so no PyTorch environment is rebuilt per contributor. The image holds no model file. No CI job builds or publishes it, and it has not been built in CI or on the qualification machine: its status is untested.

```sh
docker build -f model-pipeline/Dockerfile -t starlings-onboarding model-pipeline
docker run --rm -it -v "$PWD:/work" -v "$HOME/.cache/starlings-models:/models" starlings-onboarding \
  model-pipeline run model-pipeline/recipes/<name>-cpu.yaml --snapshot /models/<snapshot> \
  --reference-python /opt/venvs/encoder/bin/python --out artifacts/onboarding/<name>
```

The generation environment carries the CUDA 13.0 PyTorch wheel and the CUDA dependency wheels its lock pins (the route runs it on CPU), so it is the largest of the three. Pipeline dependencies outside the reference locks (`jsonschema`, and `pytest` and `ruff` in the pipeline environment) are resolved at build time, not hash-locked. The Kev environment needs a pinned Kev checkout and is not part of the image.

Recipes contain a plugin ID and repository-relative paths to the existing manifest, corpus, and reference environment. They contain no shell commands. Unknown fields, unsupported plugins, duplicate keys, traversal, and escaping paths are rejected. `plan` does not need weights, the reference environment, or an installed .NET SDK.

## Execute and resume

```sh
.venv/bin/model-pipeline run model-pipeline/recipes/multilingual-e5-small-cpu.yaml \
  --snapshot /path/to/pinned-snapshot \
  --reference-python .models/venv-reference/bin/python \
  --out artifacts/onboarding/e5-cpu \
  --threads 2 --timeout 1800
```

`--repo-root` sets the repository root that recipe paths resolve against (on `plan`, `run` and `workflow`; default: the repository found above the recipe or workflow file). `--base-snapshot` is only for routes whose model is built on a pinned base (Kev); other plugins refuse it.

E5 stages:

| Stage | Evidence |
| --- | --- |
| `assets` | Verified source lock and observed ancillary-file digests |
| `environment` | Python packages, SDK/runtime, host identity, lock validation |
| `reference` | Fresh Python reference outputs, explicitly proposed fixtures |
| `export` | Python ONNX Runtime and original-framework comparison on protected fixture inputs |
| `dotnet` | Owned tokenizer and C# outputs from IDs and text, including padded batches |
| `evaluate` | Existing protected evaluator report; G1 and G2 are required to pass |
| `distribution` (optional) | Real-model clean-consumer drill from committed code |
| `review` | Source lock, environment, fidelity report, scope and qualification limitations |

Progress goes to stderr; stdout contains a compact JSON summary. `run.json` contains the full execution record. Each attempt has a separate directory under `stages/<name>/`, command logs, timings, fingerprints, and digests of output files. Failed attempts remain available. A failed run also produces `failure-task-<run-id>.json` for a bounded human or LLM investigation; no LLM service is called automatically.

Run the same command with `--resume` to reuse completed work. Reuse requires matching input, code, environment, and dependency identities **and** intact output files. Failed or interrupted stages rerun. Changed/corrupt/missing artifacts invalidate their stage and dependents; independent valid stages can remain cached. Extra output files also invalidate a cache entry. Only one process may write a run directory at a time. SIGINT is recorded and child process groups are stopped. After a hard process kill, the next resume retries the unfinished attempt.

Outputs inside the repository must live below `artifacts/` or `.models/`. They cannot overlap input snapshots or protected evidence. Stage callbacks are trusted repository code, not a sandbox for arbitrary upstream code; they must declare all behavior-affecting dependencies and write only to the supplied output directory. Execute untrusted upstream implementations in an appropriately isolated worker as described by the main plan.

## Check package consumption

Commit the candidate before requesting distribution validation:

```sh
.venv/bin/model-pipeline run model-pipeline/recipes/multilingual-e5-small-cpu.yaml \
  --snapshot /path/to/pinned-snapshot \
  --reference-python .models/venv-reference/bin/python \
  --out artifacts/onboarding/e5-cpu \
  --threads 2 --timeout 1800 --resume --distribution
```

This invokes the existing `tools/release-drill-ai.sh --seed-model-from` in a temporary clone. It packs committed source, installs into a clean external consumer with isolated NuGet/model caches, acquires the pinned tokenizer, and checks real embeddings plus adapters during offline execution. The full model is seeded from the verified snapshot; this is not `--no-model`. Relevant source changes must be committed so the drill and fidelity candidate cannot silently differ. The drill's logs and package evidence are retained in the stage output.

Passing this pipeline proves the checks recorded in its reports. It does **not** approve a release, change registry support statuses, establish controlled performance, or satisfy every release gate. Fresh reference fixtures remain proposals and never overwrite protected fixtures or tolerances. Fixture adoption is a separate maintainer-reviewed change.

## Export for review

```sh
.venv/bin/model-pipeline propose artifacts/onboarding/e5-cpu \
  --out artifacts/onboarding/e5-review
```

The destination must be new. Before copying artifacts, the exporter verifies the completed run's identity/metadata records, dependency links, and complete output trees. The bundle contains portable stage files plus the original run record (which retains original execution paths). Review `stages/review/review.md`, `review.json`, and the fidelity report. These local digests detect accidental corruption; they are not a signed release attestation. Publication and PR creation remain explicit external actions.

Exit codes: `0` means the requested operation completed; `1` means a pipeline stage failed; `2` means invalid arguments, inputs, or a refused operation. Existing `evaluate` and `check-protected` exit codes are unchanged.

## Kev route

The `kev-onnx-cpu-v1` plugin (`src/model_pipeline/onboarding_kev.py`) runs `recipes/kev-0.8b-cpu.yaml` and `recipes/kev-4b-cpu.yaml`. It accepts only the two pinned Kev checkpoints, their pinned Qwen3.5 base and the pinned Kev code commit (`candidates/kev-0.8b/`, `candidates/kev-4b/`). Stages: `assets`, `environment`, `reference`, `build`, `onnx`, `dotnet`, `evaluate`, `review`.

```sh
.venv/bin/model-pipeline run model-pipeline/recipes/kev-0.8b-cpu.yaml \
  --snapshot /path/to/kev-0.8b-snapshot \
  --base-snapshot /path/to/hf-cache/models--Qwen--Qwen3.5-0.8B-Base/snapshots/<revision> \
  --reference-python /path/to/kev-venv/bin/python \
  --out artifacts/onboarding/kev-0.8b --threads 2 --timeout 1800
```

- `--base-snapshot` is a snapshot directory in Hugging Face cache layout (`<cache>/models--<org>--<name>/snapshots/<revision>`), because Kev loads its base and tokenizer through transformers from that cache, offline.
- The reference environment is `reference/kev/requirements.txt` (Python 3.12, torch 2.8.0 CPU build), plus Kev installed with `--no-deps` from a clean Git checkout at the pinned commit. The `environment` stage refuses a checkout at another commit or with local changes.
- For Kev-0.8B the `evaluate` stage reads its bounds from `protected/kev-0.8b/onnx-fp32/tolerances.yaml` (`src/model_pipeline/evaluate_decision.py`). Kev-4B has no binding tolerances yet; the Kev-0.8B bounds are proposed unchanged.
- `--distribution` is refused: this plugin implements no distribution stage.
- The workflow command takes no base snapshot, so Kev runs only through `run`.

References and fixtures stay proposals. The Kev registry manifests (`registry/kev-0.8b/`, `registry/kev-4b/`, CPU and CUDA) are Experimental with every gate `not-run`; this route's reports do not change them.

## Whisper route

The `whisper-onnx-cpu-v1` plugin (`src/model_pipeline/onboarding_whisper.py`) runs
`recipes/whisper-large-v3-turbo-cpu.yaml`. It accepts only the pinned fp16 graphs of
`onnx-community/whisper-large-v3-turbo-ONNX` and, as `origin.base`, the pinned `openai/whisper-large-v3-turbo` PyTorch
checkpoint they were exported from (`candidates/whisper-large-v3-turbo/`). Stages: `assets`, `environment`,
`reference`, `onnx`, `dotnet`, `evaluate`, `review`.

```sh
.venv/bin/model-pipeline run model-pipeline/recipes/whisper-large-v3-turbo-cpu.yaml \
  --snapshot /path/to/whisper-large-v3-turbo-ONNX-snapshot \
  --base-snapshot /path/to/whisper-large-v3-turbo-snapshot \
  --reference-python /path/to/whisper-venv/bin/python \
  --out artifacts/onboarding/whisper-large-v3-turbo --threads 2 --timeout 3600
```

- `--base-snapshot` is a plain snapshot directory (no cache layout needed). The `assets` stage verifies both
  repositories' pins and checks their `config.json`, `generation_config.json` and `preprocessor_config.json` against
  the contract.
- The reference environment is `reference/whisper/requirements.txt` (Python 3.11, torch 2.14.1 CPU, transformers
  5.18.0, onnxruntime 1.30.0). The corpus is `reference/corpus/asr-corpus-v3.jsonl`: LibriSpeech WAV clips pinned by
  SHA-256 (the four single utterances of `asr-corpus-v1.jsonl`; from v2, a 27.9 s composite of six consecutive
  utterances and a composite cut at 30 s inside its last utterance, which the ground-truth sanity bound leaves out; and
  two long-form composites of 34.5 s and 59.1 s).
- KV cache: the onnx stage and .NET also decode every clip through `decoder_with_past_model_fp16.onnx` (the
  full-context graph once over the prompt, then one cached step per token). Ids, windows and segments must equal the
  full-context decode exactly, and .NET's KV loop must equal Python's (ids exact, step logits within
  `same_backend_max_abs`). The KV loop's step logits against the full-context logits teacher-forced on the same ids are
  not bit-identical (fp16 summation order); they are bounded by the candidate's `evaluation.proposed_policy`
  `kv_logits_max_abs`, adopted into `protected/` at 0.02 (it started as the candidate's proposal). Adopting moves the key
  from the candidate's `proposed_policy` into its `policy` in the same protected-change PR that adds it to the tolerance
  file; the route accepts the key in exactly one of the two places, and the binding check requires the candidate policy
  and the tolerance file to hold the same keys. The report names the source (`kv_logits_max_abs_source`). The .NET public audio paths
  run the library default, the KV cache.
- A clip of more than 3000 mel frames (480,160 samples or more) is long-form, exactly as transformers `generate` decides (`is_shortform = frames <= 3000`; the route, the evaluator and the library all decide by frames): one spectrogram over the
  whole clip (`WhisperFeatureExtractor(truncation=False, padding="longest")`, numpy and torch paths, against
  `WhisperMelSpectrogram.ExtractLongForm`), decoded with timestamps window after window (the contract's `long_form`
  block pins greedy decoding, temperature 0 with no fallback, no conditioning on earlier windows and no thresholds).
  Its checks are the whole-clip mel, the default feature path's ids, and the timestamped checks below, plus .NET's
  plain-text `Transcribe`. A long-form clip has no single-window decode, so no encoder hidden state or teacher-forced
  logits are compared for it: the fidelity bounds are measured on the clips up to 30 s only. A corpus-level check
  requires some long-form clip to take three windows or more.
- Evidence is checked at three levels. Intermediate tensors: the .NET mel against the reference mel; the fp16 encoder
  hidden state (separately on the positions that cover audio and on the zero-padded tail) and the decoder logits,
  teacher-forced on the reference ids, against the fp32 PyTorch reference. Same backend: .NET against Python ONNX
  Runtime on the same graphs (hidden state and logits). Output: greedy ids and transcripts identical between .NET
  and Python ONNX Runtime, word error rate against the reference transcript, and a corpus-level word error rate
  against the LibriSpeech text as a sanity bound.
- Every clip is also decoded with segment timestamps: transformers `generate(return_timestamps=True)` (greedy, no
  fallback, no conditioning on earlier windows, no thresholds) in the reference, the timestamp rules and seek loop
  restated in numpy in the onnx stage, and `WhisperTranscriber.TranscribeWithTimestamps` in .NET. The reference stage
  replays each window through the numpy rules on teacher-forced logits and refuses any difference from `generate()`'s
  ids or segments. Window ids and segment times are compared exactly against the reference, and window ids, segments
  and texts exactly between .NET and Python ONNX Runtime; these are correctness checks under the binding
  `token_ids: exact` and `transcript: exact`, with no new tolerance. Two corpus-level checks require that some clip
  emits a timestamp pair and that some clip's decode ends inside a segment and is decoded again from the last closing
  timestamp, so the window-level checks cannot pass without exercising those branches.
- The reference stage records the mel on both of `WhisperFeatureExtractor`'s paths (numpy, which the library
  restates, and the default torch path) and checks that the default path yields the same ids.
- `--distribution` is refused, and the workflow command takes no base snapshot, so Whisper runs only through `run`.

The binding tolerances are `protected/whisper-large-v3-turbo/onnx-fp16/tolerances.yaml` (#142); the plugin requires the
candidate policy to equal them. References stay proposals: no reference fixtures are adopted. The registry manifest
(`registry/whisper-large-v3-turbo/onnx-fp16-cpu/`) is Experimental with every gate `not-run`; this route's reports do not
change it.

## PP-OCRv5 route

The `ppocr-onnx-cpu-v1` plugin (`src/model_pipeline/onboarding_ppocr.py`) runs `recipes/ppocrv5-mobile-cpu.yaml`. It
accepts only the pinned official ONNX exports `PaddlePaddle/PP-OCRv5_mobile_det_onnx` and
`PaddlePaddle/PP-OCRv5_mobile_rec_onnx` and the pinned Paddle inference models they were exported from
(`PaddlePaddle/PP-OCRv5_mobile_det`, `PaddlePaddle/PP-OCRv5_mobile_rec`), listed as `origin.components` of
`candidates/ppocrv5-mobile/`. Stages: `assets`, `environment`, `reference`, `onnx`, `dotnet`, `evaluate`, `review`.

```sh
.venv/bin/model-pipeline run model-pipeline/recipes/ppocrv5-mobile-cpu.yaml \
  --snapshot "$PWD/.models/snapshots/ppocrv5-mobile" \
  --reference-python "$PWD/.models/venv-ppocr/bin/python" \
  --out "$PWD/artifacts/onboarding/ppocrv5-mobile" --threads 2 --timeout 1800
```

- `--snapshot` is one directory holding the four repositories as `det/`, `rec/` (ONNX) and `det-paddle/`,
  `rec-paddle/` (Paddle). The `assets` stage verifies every pin and checks all four `inference.yml` files field by field
  against the contract: the preprocessing operators and their constants, the DB post-processing constants, and the
  recognizer dictionary (18,383 entries, by SHA-256).
- The reference environment is `reference/ppocr/requirements.txt` (Python 3.12: paddlepaddle 3.3.1 CPU, paddleocr
  3.7.0, paddlex 3.7.2, onnxruntime 1.30.0); install it with `uv venv --python 3.12 .models/venv-ppocr`, then
  `uv pip install --require-hashes -r model-pipeline/reference/ppocr/requirements.txt`, then `-e model-pipeline`. The
  corpus is `reference/corpus/ocr-corpus-v1.jsonl`: ten rendered images with known text, pinned by SHA-256.
- The reference is PaddleOCR on the Paddle inference engine with oneDNN off (paddlepaddle 3.3.1 fails these graphs
  under oneDNN). The runner calls PaddleOCR's own preprocessing, DB post-processing, crop and CTC operators step by step
  to record every intermediate, and checks that the steps equal `PaddleOCR.predict` on each image. The `onnx` stage
  repeats this with `engine="onnxruntime"` and also runs the graphs on the reference's own detector and recognizer
  inputs.
- Evidence is checked at four levels. Preprocessing: the .NET detector input from the image, and the .NET recognizer
  input cropped from each reference box, exactly equal to the reference (both restate OpenCV's fixed-point uint8
  `INTER_LINEAR` resize and `INTER_CUBIC` perspective warp). Graph outputs: probability map and recognizer
  probabilities, .NET against Python ONNX Runtime on the same inputs (`same_backend_max_abs`), Python ONNX Runtime
  against Paddle (bounded). Geometry: boxes matched one-to-one by IoU, from the .NET DB post-processing on the reference
  probability map, and end to end from the image; a box may appear on one side only when its box score lies within
  `box_flip_margin` of the box threshold, and the matched lines must keep the reference's reading order. Text: CTC text
  exact on every matched box; line confidences bounded (Python ONNX Runtime against Paddle, .NET against Python); and,
  per image and over the corpus, a character error rate against the drawn text as a sanity bound (a text-free image
  must give no line). The reference stage must name the Paddle engine, CPU, oneDNN off and the pinned versions.
- `--distribution` is refused, and the workflow command takes no extra snapshots, so PP-OCRv5 runs only through `run`.

References and the comparison policy are proposals; no protected tolerances or fixtures exist for this profile, and
there is no registry manifest yet (the maintainer adds it with the adopted tolerances).

## Add another execution plugin

1. Specify the upstream behavior contract and exact source assets in a reviewed registry manifest or proposed candidate contract. Reuse existing tokenizer/runtime components only where their supported parameters match.
2. Implement `describe_plan` and `build_stages` using `Stage(name, dependencies, inputs, parameters, execute)`, and declare `SHARED_DISTRIBUTION`. Register the plugin in `PLUGINS` in `src/model_pipeline/onboarding_plugins.py`; recipe validation and CLI dispatch both read that table. The first plugin deliberately rejects non-E5 or different-precision profiles.
3. Declare file inputs, dependency outputs, algorithm/configuration parameters, relevant source/build files, and toolchain identity. Do not include unrelated Markdown or a blanket Git HEAD in expensive-stage keys. Include the committed candidate identity for distribution and review.
4. Produce the evidence format required by the route's independent evaluator. Protected routes use the existing stage-results format; proposed candidates use their task-specific development reports. New tasks such as generation need their own checks for templates, logits, caching, stopping, isolation, and streaming; an embedding evaluator cannot qualify them.
5. Add cheap orchestration/failure tests, then qualify an actual model and retain a small evidence summary. A fixture-generating implementation cannot approve its own goldens or loosen thresholds.

Measure hands-on time, elapsed time, cache reuse, and compute per model. Shared capabilities and reliable resume should reduce repeated work; no numerical speedup is claimed until representative onboarding runs establish it.
