# Profile evaluation

Status: **experimental**. Code: `model-pipeline/src/model_pipeline/interventions/evaluate.py`. Spec schema: `model-pipeline/schemas/intervention-evaluation-spec.schema.json`. Tests: `model-pipeline/tests/test_intervention_evaluate.py` and the env-gated `test_intervention_evaluate_lfm3.py`. First report: [`docs/evidence/lfm3-concise-evaluation-2026-10-31`](../evidence/lfm3-concise-evaluation-2026-10-31/README.md), for `lfm3-1.7b-concise`.

[Extraction](extraction.md) shows that a profile's directions separate two prompt sets. Evaluation asks whether applying the profile, at a given strength, changes what the model generates in the intended way, what else it changes, and whether a direction with no meaning would do the same. It writes one report per profile with a recommended strength range, or none.

## Command

```sh
cd model-pipeline
pip install -e '.[interventions-extract]'      # numpy, onnx, onnxruntime, tokenizers
model-pipeline interventions evaluate interventions/profiles/lfm3-1.7b/concise/evaluation/spec.yaml \
    --snapshot "$STARLING_MODELS/LiquidAI/LFM3-1.7B-ONNX/c680759afee1ce8310c90d6c621f4558200dee77" \
    --cache "$HOME/.cache/starlings/interventions" \
    --out ../docs/evidence/<name> [--provider cuda --device-id 0] [--threads N]
```

`--snapshot` and `--cache` default to `STARLING_LFM3_SNAPSHOT` and `STARLING_LFM3_INTERVENTION_CACHE`; the cache must be on the snapshot's file system. `--provider cuda` needs `onnxruntime-gpu` (with cuDNN) in place of `onnxruntime`. The same entry point is `python -m model_pipeline.interventions.evaluate`. Outputs: `evaluation-report.json`, `evaluation-report.md` (rendered from the JSON) and `generations.jsonl` (every answer, with its scores).

## Spec

YAML, validated against the schema, with strict keys; paths are relative to the spec file. The recommendation thresholds are part of the spec, so they are fixed (and committed) before a run.

| Key | Meaning |
| --- | --- |
| `profile`, `model`, `graph`, `chat_template` | The profile, the cache model name, the base graph in the snapshot (its SHA-256 must be the profile's base) and `lfm3`. |
| `strengths` | The sweep; must contain 0. With `directional_projection`, negative strengths amplify the component along the directions and positive strengths remove it (1 with scale 1 removes it fully), so the sweep should cover both signs. |
| `generation` | `greedy`; the token caps for target answers and capability answers. |
| `target` | `effect` (`shorter_responses`), the prompt set, the labels of the profile's two prompt sets and the pooling used for the internal measure. |
| `neutral`, `capability` | The drift texts and the capability items. |
| `control` | `seed`, `scaling` (`matched_rms` or `profile`) and the strengths to run the control at; must contain 0. |
| `cross_check` | Prompts, strengths and token count for the CPU comparison. |
| `recommendation` | The thresholds below. |

Prompt sets (JSON lines):

- target: `{id, prompt, keywords, reference_positive, reference_negative}`. Held out from the extraction sets. `keywords` is a list of groups; an answer passes when it contains a word of every group. The two references answer the prompt in the profile's two styles.
- neutral: `{id, prompt, response}`, ordinary texts that are not about the profile's style.
- capability: `{id, kind, prompt, check}`, with `check.type` one of `exact`, `last_integer`, `uppercase_contains`, `bullets`, `one_word`, `json_object`, `starts_with`, `max_words`. No model output is executed: code items ask what a snippet prints and compare the text.

The neutral and capability sets are profile-independent: `model-pipeline/interventions/evaluation/`. All three sets were written for this repository.

## Method

**Decode path.** Generation runs the real derived graph (the profile compiled by `get_or_compile`, the strength passed as the graph's strength input on every call) in a Python ONNX Runtime loop: one prefill call with empty caches, then one call per token with the `present*` outputs fed back, greedy, stopping at `<|im_end|>`. That is the loop `Lfm3Sequence` runs. A C# console tool over `Lfm3ChatModel` was the alternative. It was not used, because the Python loop needs no new project and also gives the prefill logits (for drift) and the capture graph (for the internal measure) in the same process, and because the two loops were already shown to agree: the LFM3 onboarding evidence (`lfm3-onboarding-2026-10-28`) recorded equal greedy ids from the .NET implementation (`Lfm3ParityTests`) and an independent Python ONNX Runtime greedy loop over the same graph, on 19 of 19 cases, and [lfm3.md](lfm3.md) records that the .NET derived-graph path at strength 0 equals the original graph. `test_intervention_evaluate_lfm3.py` reruns the evaluator's loop on the concise derived graph at strength 0 and requires the onboarding ids for every single-turn case.

**Provider.** The report names the provider. On CUDA every comparison stays within one session: drift compares strength s with strength 0 *of the same session*, and each arm's lengths are compared with its own strength 0. A CPU cross-check greedy-decodes the first prompts on the CPU provider and reports whether the ids match the main run.

**Target effect, behavioural.** For each strength, every target prompt is answered (thinking disabled). The report gives the median and mean generated tokens, the number of answers that ended before the cap (a capped answer's length is a lower bound, and a median at the cap is marked censored), the paired count of prompts whose answer got shorter or longer than at strength 0, and a quality check: the answer contains every keyword group and repeats no word 6-gram 3 or more times. The keyword groups were checked against both reference answers of each prompt, so the check accepts a good answer in either style.

**Target effect, internal.** Each generated answer is run through the *base* graph's capture graph (the same entry extraction uses), pooled as in extraction and projected on direction 1 of each profile layer. The target set's reference answers are projected the same way; their held-out AUC and Cohen's d are reported, and `position` maps the answers onto the reference scale per layer (0 = mean of the negative references, 1 = mean of the positive ones), averaged over layers. Measuring inside the derived graph instead would be circular: at a site, the projection after the operator is `(1 - strength * scale)` times the projection before it by construction.

**Drift.** Each neutral text is teacher-forced (one prefill, all positions' logits). At every position that predicts a response token or the closing `<|im_end|>`, the evaluator computes KL(strength 0 ‖ strength s) of the next-token distributions in float64. The per-text mean is then averaged over texts; the maximum over texts is reported too.

**Capability.** Every item is answered greedily with a short cap and scored programmatically. The report gives the correct count per kind.

**Control.** A profile with a seeded random unit direction at each of the profile's sites (`random_orthonormal(rank, hidden, seed + i)`), compiled and run like the profile. With `matched_rms`, each site's scale is the profile's scale times the ratio of the per-token RMS projections, on the reference answers' response tokens, of the profile direction over the random direction. At the same strength both then change the residual by the same RMS amount, in a direction with no meaning. Without matching, a random direction's component is so small that amplifying it barely changes anything, and the control would show nothing. The report also gives the control's drift, so the reader can see how far "the same perturbation size" really produced the same disturbance.

**Recommendation.** A strength other than 0 is recommended when every rule holds against strength 0 of the profile arm:

- the median length falls by at least `min_length_reduction`;
- at least as many answers end before the cap;
- the quality pass rate drops by at most `max_quality_drop`;
- capability drops by at most `max_capability_drop` items;
- mean neutral KL is at most `max_mean_kl`;
- specificity, the profile's length reduction minus the control's at the same strength, is at least `min_specificity`. This needs a control run at that strength.

The range is reported when the passing strengths are contiguous in the sweep, on one side of 0. Strengths between sweep points were not run. The rules are applied as written, with no tuning after the run, and the report shows each rule's pass or fail per strength.

## Limits

- One decoding setting (greedy), one chat template (LFM3, one user turn, thinking disabled), 16 target prompts, 16 drift texts and 24 capability items. These are small sets: a difference of one or two answers is within noise.
- The quality check is a keyword and repetition check, not a judgement of factual accuracy or helpfulness. Read `generations.jsonl` for that.
- The drift and capability thresholds were fixed before the run, without calibration.
- Only `shorter_responses` is implemented as a target effect.
- Real-model runs are local; CI runs only the model-free tests.
