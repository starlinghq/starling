# Guardrails: first measurement (2026-11-16)

What the experimental guardrails (`Starlings.Guardrails`, `Starlings.Guardrails.Extensions.AI`, `Starlings.Models.Kev.Guardrails`) detect on input text, and what each level costs. Input side only: prompt injection, jailbreak and prompt-extraction attempts against an assistant. The output (harmful response) questions are **unmeasured**: no licensed evaluation set has been chosen.

## Setup

- Linux x64, Intel i7-5820K (6 cores, AVX2), CPU only, ONNX Runtime 1.30.0; Kev-0.8B bundle `Starlinghq/kev-0.8b-onnx` at `8a9fe017` (fp32), Kev's default `KevOptions` (calibrated temperature).
- `tools/Starlings.Guardrails.Eval.Tool`, one process per row, each model run under `systemd-run --user --scope -p MemoryMax=12G`.
- Data, hand-written for this measurement, no third-party text: `data/dev.jsonl` (24 attacks, 24 benign look-alikes) for choosing rules, questions and thresholds; `data/heldout.jsonl` (16 attacks, 16 benign look-alikes) written after the dev runs and before the pattern rules were ported, then run once per level. Benign items are deliberately close to attacks: "ignore" and "instructions" in ordinary requests, questions about jailbreaks and prompt injection, role-play requests, and the same carrier formats (reviews, search results, invites, PDF footers, code comments) without an embedded instruction.
- Caveats: one author wrote the rules, the questions and both sets; 32 held-out items give wide intervals (one item moves TPR or FPR by about 6 points); no multilingual or long-document cases beyond a few short French, German and Spanish items.

## Held-out results (reported)

| Level | Guard | AUC | Attacks caught (TPR) | Benign flagged (FPR) | Latency p50 / p95 | Peak working set |
| --- | --- | --- | --- | --- | --- | --- |
| Light | `patterns` | 0.56 | 2/16 (0.12) | 0/16 (0.00) | 0.02 / 3.5 ms | 40 MB |
| (model alone) | `kev-0.8b` | 0.84 | 12/16 (0.75) | 3/16 (0.19) | 463 / 546 ms | 2,267 MB |
| Balanced | `patterns>kev-0.8b` | 0.83 | 12/16 (0.75) | 3/16 (0.19) | 478 / 568 ms | 2,259 MB |
| Strict | `patterns>kev-0.8b` | 0.84 | 12/16 (0.75) | 3/16 (0.19) | 474 / 543 ms | 2,266 MB |

After the held-out runs, one pattern rule was narrowed (`DAN` made case-sensitive, so the name "Dan" does not flag); the light level was re-run on both sets and its counts are identical (dev 19/24 and 0/24, held out 2/16 and 0/16).

Kev's misses: instructions embedded in carried content (a product review, a search result, a calendar invite) and one context-window dump request. Kev's false positives: a question about jailbreak techniques for testing, IKEA assembly instructions, and a "hypothetically" mortgage question. The model load takes about 10 s and is not in the latency.

## Dev results (used for choices, not a quality claim)

| Guard | AUC | TPR | FPR | p50 |
| --- | --- | --- | --- | --- |
| `patterns` | 0.90 | 19/24 | 0/24 | 0.02 ms |
| `kev-0.8b`, choice question (default) | 0.87 | 21/24 | 5/24 | 438 ms |
| `kev-0.8b`, one noul question | 0.91 | 23/24 | 13/24 | 370 ms |

The noul question ranks slightly better but its scores sit high: at the 0.5 threshold it flags over half of the benign items, so the choice question is the default. The pattern rules were written while reading the dev set: their drop from 19/24 on dev to 2/16 on held-out is the overfitting this split exists to expose.

An earlier spike on the dev set also tried multilingual-e5-small with 8 exemplars per side (AUC 0.88, 6 ms, about 680 MB) and three noul questions per category on Kev (FPR 0.67 at 0.5); neither is in the packages.

## Conclusions

- The patterns are a free first tier with no false positives here but low recall on unseen phrasings. They are not a guard on their own.
- Zero-shot Kev-0.8B is a weak detector: 3 in 4 attacks at 1 in 5 false positives, half a second per text on CPU, 2.3 GB. It misses indirect injection, the case that matters most for tool results.
- Kev-4B was not run (16.9 GB fp32; same training, so no large change is expected, unmeasured).
- A "be very sure" level needs a purpose-trained guard model (a model admission); the cascade and the `IContentGuard` contract are built to take one.

## Reproduce

```sh
dotnet build ai-dotnet/tools/Starlings.Guardrails.Eval.Tool -c Release
cd ai-dotnet/tools/Starlings.Guardrails.Eval.Tool/bin/Release/net10.0
dotnet Starlings.Guardrails.Eval.Tool.dll --set heldout --level light
dotnet Starlings.Guardrails.Eval.Tool.dll --set heldout --level model      # also balanced, strict; needs the cached Kev-0.8B bundle
dotnet Starlings.Guardrails.Eval.Tool.dll --set dev --level model --question noul
```
