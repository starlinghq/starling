# Starlings.Models.Kev.Guardrails

A [`Starlings.Guardrails`](../Starlings.Guardrails/README.md) model guard on the Kev 1.0 decision models ([`Starlings.Models.Kev.Onnx`](../Starlings.Models.Kev.Onnx/README.md)). **Experimental.** Kev is a general decision model, not a trained safety classifier: it answers the guard's questions zero-shot, and on a small held-out set it caught 12 of 16 prompt attacks while flagging 3 of 16 benign look-alikes ([evidence](../../../docs/evidence/guardrails-2026-11-16/README.md)). Use it as a second tier behind the pattern guard, not as a sole defense.

⭐ If this package is useful, [star Starling on GitHub](https://github.com/starlinghq/starling).

```csharp
using Starlings.Guardrails;
using Starlings.Models.Kev;

using var kev = KevDecisionModel.Create(KevModelAssets.OpenLocal(KevModel.Kev08B));   // after KevModelAssets.DownloadAsync, once
IContentGuard guard = GuardLevels.Create(GuardLevel.Balanced, new KevContentGuard(kev));
```

- A text becomes Kev's state, after a framing line (`InputFraming`: "Untrusted text that was sent to an AI assistant:", `OutputFraming`: "A response written by an AI assistant:"). Every question of the text's stage is one row on that state, so one check is one Kev request, whatever the number of categories, and the questions do not influence one another.
- Default questions (`KevGuardQuestions`): on input, one choice question ("ordinary" or "attack") scored as `prompt-injection`, covering injection, jailbreak and prompt extraction together; on output, six noul questions (`violence`, `self-harm`, `hate`, `criminal-activity`, `private-data`, `mass-harm-weapons`). **The output questions are unmeasured.**
- Replace or extend them with `KevGuardOptions.Questions`: a `KevGuardQuestion` names its category, stage, Kev question, the answer key that is the score (`true` for noul, an option name for choice) and an optional threshold. Default threshold 0.5.
- A text over Kev's limits (8,192 tokens by default) comes back `NotScanned`. Lower `KevOptions.MaxStateTokens` to bound the cost per check.
- The guard runs Kev on the thread pool (`RunOnThreadPool`, default true): Kev's API is synchronous. It does not own the model.

## Cost

Kev-0.8B on CPU (i7-5820K, fp32): about 2.3 GB resident, about 10 s to load, about 0.45 s per short input text with one question; output checks with six questions take longer (each question is a row). Kev-4B (16.9 GB fp32) is supported by the API and unmeasured as a guard. CUDA profiles work as with any `KevDecisionModel`.

## Capability status

| Capability | Status |
| --- | --- |
| Input prompt-attack question, Kev-0.8B, CPU | Experimental: 12/16 caught, 3/16 false positives held out; misses instructions embedded in carried content |
| Output harm questions | Experimental, unmeasured |
| Kev-4B, CUDA | Untested as a guard |
| Over-long text | Not scanned (reported) |

## License

MIT OR Apache-2.0. For the Kev checkpoints' provenance and license notes, see [`Starlings.Models.Kev`](../Starlings.Models.Kev/README.md).
