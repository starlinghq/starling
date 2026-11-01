# Profile extraction

Status: **experimental**. Code: `model-pipeline/src/model_pipeline/interventions/capture.py` (capture graphs) and `extract.py` (extraction). Spec schema: `model-pipeline/schemas/intervention-extraction-spec.schema.json`. Tests: `model-pipeline/tests/test_intervention_capture.py`, `test_intervention_extract.py`, and the env-gated `test_intervention_extract_lfm3.py`. Design: [architecture.md](architecture.md). Output format: [profile-format.md](profile-format.md).

Extraction turns two contrasting, paired prompt sets into a `directional_projection` profile. For each candidate layer it measures how well the mean-difference direction separates the two sets on held-out pairs, picks layers on that evidence, and writes `profile.json`, its directions files and a report. The first profile made this way is LFM3-1.7B fp32 `lfm3-1.7b-concise`, under `model-pipeline/interventions/profiles/lfm3-1.7b/concise/`.

A profile records a direction in activation space that tells the two prompt sets apart. It is not evidence of a behavioural effect. See "What the concise profile shows" below.

## Positioning

Starling ships the mechanism together with benign profiles: style profiles first, and safety-increasing ones later. The host application decides which callers get which profile, at what strength; the library does no access control. The repository contains no refusal-direction, jailbreak or other safety-reducing profile, prompt set or sample.

LFM3-1.7B is under the LFM Open License v1.0: Apache-2.0 terms plus a USD 10M annual revenue threshold for commercial use. It has no clause on behaviour modification. Derived graphs are built locally from the user's own copy of the weights and are never distributed. The profile files ship only directions (2,048 floats per site) and the prompt sets, which were written for this repository and carry its license.

## Command

```sh
cd model-pipeline
pip install -e '.[interventions-extract]'      # numpy, onnx, onnxruntime, tokenizers
model-pipeline interventions extract interventions/profiles/lfm3-1.7b/concise/spec.yaml \
    --snapshot "$STARLING_MODELS/LiquidAI/LFM3-1.7B-ONNX/c680759afee1ce8310c90d6c621f4558200dee77" \
    --cache "$HOME/.cache/starlings/interventions"      # must be on the snapshot's file system
```

`--snapshot` and `--cache` default to `STARLING_LFM3_SNAPSHOT` and `STARLING_LFM3_INTERVENTION_CACHE`. The outputs go next to the spec unless `--out` is given. `--threads` sets ONNX Runtime's intra-op threads. The same entry point is `python -m model_pipeline.interventions.extract`.

## Capture graphs

`compile_capture` / `get_or_compile_capture` (in `capture.py`) build the base graph with the site mapping's tensors for the requested `(layer, target)` sites added to `graph.output`. Nothing else changes. `validate_capture_model` checks that inputs, nodes (in order), initializers, `value_info`, IR version and opset imports are byte-identical to the base. It also checks that the outputs are the base outputs followed by exactly the site tensors not already exposed. Every base output is computed as before, and each extra output is the value the base graph computes internally. Tests compare both with the same tensors exposed by hand on the base graph, on a synthetic graph and on the real LFM3 graph (bit-identical; logits identical too).

The derived-artifact rules are the same as for intervention graphs ([compiler.md](compiler.md), [cache.md](cache.md)):

- Mappings are keyed by graph hash, and unknown graphs are unsupported.
- No weight is loaded or copied; the weights are hard links next to `model.onnx`.
- Entries are content-addressed and published atomically under a lock.
- Every open is verified (Fast: graph SHA-256 plus link sizes and inode identity; Full: hashes everything).
- The bundle is never written.

Capture entries live apart from intervention entries:

```
<cache>/<model>/intervention-captures/<key>/
    model.onnx                  the capture graph
    model.onnx_data...          hard links to the base weights
    capture-manifest.json       schema_version 1, kind "capture" (Python-only; no .NET reader)
```

The key is the SHA-256 of `starling-intervention-capture-key 1` followed by the capture version, site-mapping version, opset, precision, the base artifact lines (as in the intervention key) and one `output <layer> <target> <tensor>` line per site, sorted by layer. A capture graph has no profile and no strength input, so it is not a valid `compiler-manifest.json` artifact. The C# reader lists only 64-hex directories under `<model>/interventions/`, so it never sees capture entries.

## Spec

YAML, validated against the schema, with strict keys. Paths are relative to the spec file. Machine-specific locations are command-line options and never go in the spec.

| Key | Meaning |
| --- | --- |
| `profile_id`, `description` | Written to `profile.json`. |
| `model` | Cache model name for the capture entry (e.g. `lfm3`). |
| `base_model_sha256`, `graph` | The base graph as the asset lock records it. The snapshot's file must match, and a site mapping must exist. |
| `chat_template` | `lfm3`: one user turn and the assistant response, thinking disabled (`Lfm3ChatTemplate`). |
| `operator`, `target` | `directional_projection`, `residual_post`. |
| `prompt_sets.positive` / `.negative` | `label` and `path` of a JSON lines file of `{"id", "prompt", "response"}`. Both files list the same ids in the same order with the same prompts and different responses. |
| `pooling` | `response_mean` or `last_token` (see below). |
| `candidate_layers` | The sites to capture and score. |
| `rank`, `direction_method` | Directions per selected site; `mean_difference_pca`. |
| `scale` | The scale `s` written for every direction. |
| `evaluation.folds` | Cross-validation folds over pairs. |
| `selection.max_layers`, `selection.min_auc` | Layer selection rule. |
| `seed` | Fold assignment. |

## Method

**Rendering and tokens.** Each example is rendered as `<|startoftext|><|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think></think>{response}<|im_end|>`, which is what the LFM3 template renders for that turn pair, and tokenized with the snapshot's `tokenizer.json`. The real-model test checks the renderer and the HF `tokenizers` encoder against every case of the LFM3 template corpus. The response tokens are the ids after the generation prompt, up to but not including `<|im_end|>`. The extractor checks that the generation prompt tokenizes to a prefix of the turn.

**Prefill only.** Each example is one forward pass of the capture graph with empty caches. Nothing is generated.

**Pooling (chosen: `response_mean`).** Each site's residual is averaged over the response tokens. The contrast is in the answers, not the prompts, so the response tokens carry it. Taking the mean over all of them, rather than one position, makes the vector describe the whole answer. `last_token` (the residual at the last response token) is available for sets where one position carries the contrast.

**Directions (chosen: `mean_difference_pca`).** Per layer, the paired differences are `d_i = positive_i - negative_i`. Pairs share the prompt, so topic cancels within each pair. Direction 1 is `mean(d) / |mean(d)|`, signed towards `positive`. Directions 2..`rank` are the top principal components of the centred `d_i` after direction 1 is projected out. A QR step re-orthonormalizes the rows, so the profile's orthonormality requirement holds by construction (error about 1e-8). The alternative, stacking several layers' mean differences, was not used because the profile format already holds one direction set per layer.

**Separability, held out.** With 2,048 dimensions and 64 pairs, any direction fitted on the same data separates them almost perfectly, so in-sample numbers cannot pick layers. The pairs are split into `folds` folds by `seed`. For each fold, the direction is fitted on the other folds and that fold's samples are projected onto it. The pooled out-of-fold projections give:

- the projection AUC (Mann-Whitney, ties count half);
- Cohen's d (pooled standard deviation);
- the paired accuracy (the share of pairs whose positive sample projects higher than its negative sample).

In-sample values are reported next to them for comparison only.

**Selection.** The layers with held-out AUC >= `min_auc` are ranked by held-out Cohen's d (ties: lower layer), and the first `max_layers` are selected. Each selected layer's directions are fitted on all pairs. If no layer qualifies, the report is written and no profile is.

**Output.** `profile.json` (canonical form, scales = `scale`), `directions/layer<L>-residual_post.f32`, `extraction-report.json` (all numbers, prompt-set and spec hashes, the capture key) and `extraction-report.md` (the same as tables). Given the same activations, the output is byte-identical.

## What the concise profile shows

[`extraction-report.md`](../../model-pipeline/interventions/profiles/lfm3-1.7b/concise/extraction-report.md) has the numbers. 64 hand-written pairs: everyday science, how-to and general-knowledge questions, each with a one-sentence answer (`concise`, 12 to 21 tokens) and a three-to-five-sentence answer (`verbose`, 76 to 106 tokens). Rank 1, scale 1, 4 folds.

- Every layer separates the sets on held-out pairs (held-out AUC 0.98 to 1.00). The task is easy at every depth, so AUC cannot choose layers. Cohen's d does: it is largest in layers 2 to 11 and falls off after layer 13. The selected layers are 2, 4 and 11.
- **Confounds.** The verbose answers are about five times longer, so a response-mean vector also encodes length and position statistics. They also use more connective and explanatory wording. The direction separates *these* answer styles. Whether it is a style direction or a length direction is not established.
- **Operator semantics.** `directional_projection` scales the component of the residual along `Q`; it does not add a vector. Strength 1 with scale 1 removes that component, strength 0 leaves the graph unchanged, and negative strengths amplify the existing component. The extraction itself does not show which strength, if any, makes LFM3 answer more concisely; a small greedy-decoding probe is recorded below.

### Behavioural probe

The profile was compiled into the fp32 graph and greedy-decoded on CPU with ONNX Runtime 1.30.0, for 4 prompts that are not in the prompt set (thinking disabled, at most 384 new tokens per answer). The results are in the profile's [README](../../model-pipeline/interventions/profiles/lfm3-1.7b/concise/README.md#behavioural-probe-not-an-evaluation).

- At strength 0 and at strength 1 (removal), every answer reached the cap.
- At -1, 2 of 4 answers ended before the cap.
- At -2, 4 of 4 ended, after 146 to 229 tokens, with coherent text.

With this operator, "more concise" therefore points to a negative strength, which amplifies the component. Four prompts and one decoding setting are an observation, not an evaluation.

## Not covered

- Profiles other than LFM3 fp32 `concise`, chat templates other than LFM3, multi-turn or system-prompt examples.
- A behavioural evaluation of any profile: quality, side effects, strengths, other decoding settings.
- Untangling style from length in the concise direction, for example with length-matched pairs.
- CUDA capture, the fp16 graph (no site mapping), Windows.
