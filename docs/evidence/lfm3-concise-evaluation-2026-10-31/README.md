# LFM3-1.7B `concise` profile: strength-sweep evaluation, 2026-10-31

Development evidence for the experimental intervention profile `lfm3-1.7b-concise` (`model-pipeline/interventions/profiles/lfm3-1.7b/concise/`), produced with `model-pipeline interventions evaluate` ([method](../../interventions/evaluation.md)). It is not release qualification: no gate ran and nothing here is binding.

**Result: no strength is recommended.** No strength in the sweep met every rule fixed in the spec before the run. Amplifying the direction (strength -1.5 to -2) shortens answers, and more than a magnitude-matched random direction does. At those strengths, though, the model loses 3 of 14 capability items, and at -2 drift goes over the limit. Removing the direction (positive strengths) makes answers longer, not shorter.

## Inputs

- Graph: `LiquidAI/LFM3-1.7B-ONNX` at `c680759afee1ce8310c90d6c621f4558200dee77`, fp32 `onnx/model.onnx` (SHA-256 `add6653f…5ef77`), site mapping `lfm3-1.7b/fp32/1`.
- Profile: `profile.json` SHA-256 `86be44f5…`. `residual_post` after layers 2, 4 and 11, rank 1, scale 1. The direction points towards the concise answers of the extraction set.
- Spec: `model-pipeline/interventions/profiles/lfm3-1.7b/concise/evaluation/spec.yaml`. It was committed with its thresholds before the run.
- Prompt sets, all written for this repository:
  - target: 16 everyday questions, none of them in the extraction set or the earlier 4-prompt probe;
  - neutral: 16 teacher-forced texts;
  - capability: 24 items, 8 each of arithmetic, instruction following and "what does this Python print".
- Sweep: profile at -3, -2, -1.5, -1, -0.5, 0, 0.5, 1 and 2. Control at -3, -2, -1.5, -1, 0 and 1.
- Control: one seeded random unit direction per site (seed 20261101). Its scales (4.15, 5.89, 4.37 at layers 2, 4, 11) make its per-token RMS projection on the reference answers equal to the profile direction's.
- Decoding: greedy, thinking disabled. At most 512 new tokens for target answers and 128 for capability answers.
- Environment:
  - Linux x64, Intel Core i7-5820K;
  - NVIDIA RTX 4060 Ti (driver 615.78.08);
  - ONNX Runtime 1.30.0 (`onnxruntime-gpu`, cuDNN 9.24), CUDA execution provider on device 1 with TF32 off;
  - Python 3.11, HF `tokenizers` 0.23.3.
  - Wall time: 55 minutes.

## Results

All numbers are from [`evaluation-report.md`](evaluation-report.md). Every answer is in `generations.jsonl`.

| Strength | Median tokens (control) | Ended / 16 | Quality / 16 | Capability / 24 (control) | Mean neutral KL (control) | Answer position |
| --- | --- | --- | --- | --- | --- | --- |
| -3 | ≥512 (254.5) | 6 | 3 | 4 (9) | 2.05 (0.28) | 0.04 |
| -2 | 207.5 (260.5) | 16 | 13 | 11 (10) | 0.59 (0.14) | 0.23 |
| -1.5 | 227 (304) | 16 | 15 | 11 (8) | 0.31 (0.10) | 0.04 |
| -1 | 310.5 (316.5) | 15 | 15 | 9 (9) | 0.16 (0.05) | -0.04 |
| -0.5 | 290 | 15 | 16 | 13 | 0.05 | -0.04 |
| **0** | **299** | **12** | **16** | **14** | 0 | -0.07 |
| 0.5 | 339 | 14 | 15 | 14 | 0.03 | -0.09 |
| 1 | 389 (317.5) | 12 | 16 | 14 (12) | 0.08 (0.04) | -0.12 |
| 2 | 392 | 12 | 14 | 12 | 0.27 | -0.17 |

"Answer position" places the generated answers on the reference-answer scale, measured in the base graph: 0 is the verbose references, 1 the concise ones.

What the numbers show:

- **Amplifying shortens, within a narrow band.**
  - At -2, the median answer is 31% shorter (207.5 vs 299 tokens), and 15 of 16 answers are shorter than at strength 0. All 16 ended before the cap, against 12 of 16 at strength 0. At -1.5 the median is 24% shorter.
  - The matched control shortens by 13% at -2 and lengthens by 2% at -1.5. Specificity is therefore +0.18 and +0.26, both above the 0.15 threshold.
  - At -1 and -0.5 the median does not move.
  - At -3 the model breaks down: 10 of 16 answers fall into repetition loops and run to the cap.
- **Removal makes answers longer.** At +1 and +2 the median rises by 30% (389 and 392 tokens). The control at +1 changes it by 6%. Answers end before the cap as often as at strength 0. So the component along the direction does carry length, in both signs. But removing it, which is what the operator does at positive strengths, pushes towards *longer* answers.
- **The internal measure agrees in direction but is small.** In the base graph, the answers at -2 sit at 0.23 on the verbose-to-concise reference scale, against -0.07 at strength 0. At +2 they sit at -0.17. The control stays between -0.08 and -0.05. Even the shortest answers are much closer to the verbose references than to the concise ones: a 200-token answer is still not a one-sentence answer.
- **Side effects at the strengths that shorten.**
  - Capability falls from 14 to 11 of 24 at -1.5 and -2, but the matched control falls about as far (8 to 10 of 24). That drop looks like a generic effect of perturbing the residual by this much, not something specific to this direction.
  - At -1 to -2, 1 to 4 capability answers start with a spontaneous `<think>` block, although thinking is disabled. This never happens at strength 0 or in the control. Several arithmetic misses at negative strengths are step-by-step answers cut off by the 128-token cap before the final number.
  - Drift on neutral texts is 3 to 4 times the control's at the same strength from -1 to -2 (0.59 vs 0.14 nats at -2), and 7 times at -3. Equal RMS perturbation therefore does not mean equal disturbance: this direction matters more to the model's next-token distribution than a random one.
  - Quality (keywords present and no repetition loop) holds at -1.5 (15/16) and drops to 13/16 at -2. The -2 rust answer, for example, is correct and compact. Factual accuracy was not judged.
- **Why no strength qualifies.** The rule results are listed per strength in the report. -2 fails quality (a drop of 3/16, above the 0.125 limit), capability (-3 items, the limit is -2) and drift (0.59, the limit is 0.5). -1.5 fails the length rule (24% against 25%) and capability (-3 items). Every other strength fails the length rule. The thresholds were not changed after the run.

The honest reading is this. With the `directional_projection` operator, this profile can make LFM3 answer more briefly only by amplifying the existing component. The useful window is narrow (about -1.5 to -2), and beyond it the model breaks down. Inside it, the shortening is specific to the direction, but it costs capability and drift, which the pre-set rules do not accept. Removal, the operator's natural use, does the opposite. An additive operator, adding a fixed vector along the direction instead of scaling the component the residual already has, might suit a style profile better. That needs a sibling change and was not evaluated here. A direction extracted from length-matched pairs might also behave differently, since this one mixes style with length ([extraction.md](../../interventions/extraction.md)).

## Faithfulness checks

| Check | Result |
| --- | --- |
| Evaluator decode loop on the concise derived graph at strength 0, CPU, vs the onboarding evidence's greedy ids for the original graph (`test_intervention_evaluate_lfm3.py`, every single-turn non-thinking case of the LFM3 template corpus) | equal ids on every case |
| CUDA (this run) vs CPU, greedy ids of the first 48 new tokens, 2 prompts at strengths 0 and -2 | 4 of 4 equal |
| Profile arm at 0 vs control arm at 0 (two different derived graphs, operator inactive) | identical text for all 40 answers |

## Not done

- **The .NET path.** Generation ran in the Python ONNX Runtime loop, not through `Lfm3ChatModel` (see the method's "Decode path" for why and for the existing parity evidence).
- **Other settings.** No sampling, no other decoding settings, no system prompts or multi-turn conversations.
- **Other providers and graphs.** CPU was used only for the cross-check. No Windows, no fp16 graph.
- **A second control seed and an unmatched control.** In a smoke run, an unmatched random direction at -2 gave a neutral KL of 0.0016 against the profile's 0.22: as a control it shows nothing.
- **Judgement of factual accuracy or helpfulness.** The quality check is keywords plus repetition only.
- **Sample size.** The sets are small. A difference of one or two answers is within noise.
- **The additive operator.** Not evaluated.

## Files

| File | Contents |
| --- | --- |
| `evaluation-report.json` | Every number: per arm and strength, the length, quality, capability and drift; the internal projections; the control scales; the rule results; the cross-check |
| `evaluation-report.md` | The same, rendered from the JSON (a test checks this) |
| `generations.jsonl` | Every answer, 600 rows: arm, set, id, strength, tokens, whether it ended, text and its scores |
| `run.log` | The command's log, with machine paths shortened |

Command (paths shortened):

```sh
cd model-pipeline
LD_LIBRARY_PATH=<venv>/site-packages/nvidia/cudnn/lib:/opt/cuda/lib64 \
model-pipeline interventions evaluate interventions/profiles/lfm3-1.7b/concise/evaluation/spec.yaml \
    --snapshot <LFM3-1.7B-ONNX snapshot> --cache <cache on the snapshot's file system> \
    --out ../docs/evidence/lfm3-concise-evaluation-2026-10-31 --provider cuda --device-id 1 --threads 4
```

Deciding which callers may apply this profile, and at what strength, is the host application's job. This evidence gives that decision no recommended strength.
