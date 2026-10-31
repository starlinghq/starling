# LFM3 interventions (V1)

Status: **experimental**. V1 evidence uses synthetic random orthonormal directions; no behavioural profile ships or has been evaluated. Deciding which callers may apply which profile is the host application's job.

LFM3-1.7B (`Starlings.Models.Lfm3`, the fp32 step graph of `LiquidAI/LFM3-1.7B-ONNX` at `c680759a`) can run with one intervention profile compiled into a derived graph, with the strength chosen per request. The integration is the source-only companion `Starlings.Models.Lfm3.Interventions`; `Starlings.Models.Lfm3` keeps its public API, and an application that does not use interventions loads the original graph exactly as before.

## Pieces

| Piece | Where |
| --- | --- |
| Site mapping `lfm3-1.7b/fp32/1` | `model-pipeline/src/model_pipeline/interventions/injection_sites.py`, keyed to the fp32 lock's `onnx/model.onnx` SHA-256 `add6653f…5ef77`. Sites: `residual_post` after each of the 24 decoder layers (`/model/layers.L/Add_2/output_0`, float32, hidden 2048). Both readers of each tensor are rewired. The fp16 graph has no mapping. |
| Synthetic compile | `python -m model_pipeline.interventions.lfm3_synthetic`: writes profile `lfm3-synthetic-test` (rank 8 after layers 8 and 16, seeded random orthonormal directions, scales 1) and compiles it with `get_or_compile`, passing the lock's external data hashes. |
| Load and verify | `Lfm3Interventions.CreateModel` / `CreateChatModel`: open the cache entry (`InterventionArtifactCache.OpenEntry`: name = key), check the manifest before any session exists, load the derived graph through `OnnxModelSession`. |
| Per-request strength | `model.CreateSequence(selection)`, `chat.Generate/GenerateStreaming/Chat(..., selection, ...)` extension methods. |
| Graph check | `Lfm3Model` accepts the 39 inputs of the LFM3 step graph plus exactly the manifest's one strength input. |

## Compile, load, request

The cache must be on the snapshot's file system: the derived graph reaches the 7 GB of weights through hard links ([cache.md](cache.md)).

```sh
cd model-pipeline
python -m model_pipeline.interventions.lfm3_synthetic \
    --snapshot "$STARLING_MODELS/LiquidAI/LFM3-1.7B-ONNX/c680759afee1ce8310c90d6c621f4558200dee77" \
    --cache "$HOME/.cache/starlings/interventions"
# prints <cache>/lfm3/interventions/<key>
```

```csharp
using Starlings.Interventions;
using Starlings.Models.Lfm3;
using Starlings.Models.Lfm3.Interventions;

Lfm3ModelFiles files = Lfm3ModelAssets.OpenLocal(Lfm3Precision.Fp32);
using Lfm3Model model = Lfm3Interventions.CreateModel(files, entryDirectory);

// One request at strength 0.5: the strength is fixed for this sequence and passed on every native call.
using (Lfm3Sequence sequence = model.CreateSequence(new InterventionSelection("lfm3-synthetic-test", 0.5f)))
{
    sequence.AppendTokens(promptIds);
    int next = sequence.Step();
}

// Concurrently, another request without a selection: strength 0, the original model's logits.
using Lfm3Sequence plain = model.CreateSequence(selection: null);
```

With a tokenizer, `Lfm3Interventions.CreateChatModel(files, entryDirectory)` returns an `Lfm3ChatModel`, and `chat.Chat(turns, selection, options)` / `chat.GenerateStreaming(promptIds, selection, options)` take the selection per call.

### What `CreateModel` checks before loading

1. `InterventionArtifactCache.OpenEntry` (Fast by default): the directory name is the key recomputed from the manifest; manifest and profile parse and match, directions verify, the derived graph's SHA-256, the linked weights' sizes and their identity with the base artifacts. `Full` also hashes the linked weights and the snapshot's base files.
2. The manifest's base graph SHA-256 equals the lock's entry for the requested precision's graph. A wrong base graph (another export, the fp16 graph) throws `InterventionMismatchException` and no session is created.
3. The manifest precision matches (`fp32`).
4. Exactly one extra input: the strength, float32, shape `[1]`, not named like an LFM3 input.
5. Every site is inside the 24 decoder layers at hidden size 2048.

The session takes the derived graph's verified SHA-256 as `KnownModelSha256`, so `GetInterventionDiagnostics()` reports the original identity (the lock's), the loaded identity (the derived graph's, source `Known`), the profile id, the strength input and the 39 model inputs.

### Request semantics

- The selection is resolved when the sequence is created and fixed for its lifetime; `Lfm3Sequence` has no way to change it. Each sequence owns its one-element strength array; each native call wraps it in a tensor that lives until the call returns. No tensor is shared between sequences.
- No selection on a derived graph: strength 0. A selection on the original graph: `InvalidOperationException`. A selection naming another profile: `InterventionMismatchException`.

## Evidence (2026-10-31, linux-x64 CPU, ONNX Runtime 1.30.0)

The derived graph of the synthetic profile (cache key `5ff9b1d7…d025`, derived `model.onnx` 539,490 bytes, SHA-256 `8818cf82…a2af`; the weights hard-linked, not copied) against the original fp32 graph, prompt = template corpus case `arithmetic` (22 ids):

| Check | Result | Where |
| --- | --- | --- |
| Mapping tensors and consumers are every reader of each residual tensor in the real graph | 24/24 | `test_intervention_lfm3.py::test_mapping_matches_the_real_graph_exactly` |
| Strength 0 vs original, last-position logits (Python ORT) | max abs 0 | `test_strength_zero_matches_the_base_logits_and_strength_one_differs` |
| Strength 1 vs original (Python ORT) | max abs 1.97 | same |
| Layer-8 site tensor vs NumPy `y - a((yQᵀ)⊙s)Q`, strengths 1 and 0.5 | max abs 2.0e-7, 2.3e-7 (max abs y 6.85) | `test_site_tensor_matches_the_numpy_projection` |
| Strength 0 vs original, logits (.NET) | max abs 0; no selection equals strength 0 | `Lfm3InterventionModelTests.StrengthZeroMatchesTheOriginalGraph` |
| Strength 1 / 0.5 vs original (.NET) | max abs 1.97 / 0.966 | `StrengthOneChangesTheLogits` |
| Greedy ids, 6 steps, strength 0 vs original (.NET) | equal | `StrengthZeroMatchesTheOriginalGraph` |
| 10 concurrent sequences, strengths 0, 0.5, 1, -1, 2 on one session vs sequential | within 1e-5 each | `ConcurrentSequencesWithDifferentStrengthsAreIsolated` |
| Prefill vs token-by-token at strength 1 | max abs 2.8e-5, same argmax | `PrefillAndTokenByTokenAgree` |
| fp16 graph hash, wrong base hash, wrong precision, scalar or colliding strength input, site layer 24 | rejected before any session | `ManifestCheckTests`, `TheArtifactIsRejectedForAnotherBaseGraphBeforeAnySession` |

Strength 0 is bit-identical here; other graphs, providers or optimization levels may only match within rounding (the extra reader of the residual tensor can change operator fusion).

The synthetic directions at strength 1 left the 6 greedy ids of this prompt unchanged although the logits moved by up to 1.97: the directions are random, not behavioural.

## Not covered

- The real-model tests need `STARLING_LFM3_SNAPSHOT` and a compiled entry (`STARLING_LFM3_INTERVENTION` for .NET, `STARLING_LFM3_INTERVENTION_CACHE` for Python); CI runs only the model-free tests (manifest checks, mapping).
- The chat-level test (`ChatGenerateAndStreamingPassTheSelection`) skips: the pinned `Starlings.Tokenizers` 0.3.2 rejects the LFM3 tokenizer's pre-tokenizer regex (as for `Lfm3ParityTests`, see [the onboarding evidence](../evidence/lfm3-onboarding-2026-10-28/README.md)). The model-level tests exercise the same sequence path.
- The fp16 graph (no site mapping), CUDA, Windows, AOT publish.
- More than one profile per graph, hot swap, behavioural profiles.
