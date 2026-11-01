# LFM3 interventions (V1)

Status: **experimental**. The mechanism evidence below uses synthetic random orthonormal directions. One extracted style profile, `lfm3-1.7b-concise` (`model-pipeline/interventions/profiles/lfm3-1.7b/concise/`), has held-out separability evidence ([extraction.md](extraction.md)) and a strength-sweep evaluation ([evidence](../evidence/lfm3-concise-evaluation-2026-10-31/README.md)). The evaluation recommends no strength: amplifying the direction shortens answers within a narrow band, but at a cost in capability and drift; removing it lengthens them. Deciding which callers may apply which profile is the host application's job.

LFM3-1.7B (`Starlings.Models.Lfm3`, the fp32 step graph of `LiquidAI/LFM3-1.7B-ONNX` at `c680759a`) can run with one or more intervention profiles compiled into a derived graph, with the strengths chosen per request. The integration is the source-only companion `Starlings.Models.Lfm3.Interventions`; `Starlings.Models.Lfm3` keeps its public API, and an application that does not use interventions loads the original graph exactly as before.

## Pieces

| Piece | Where |
| --- | --- |
| Site mapping `lfm3-1.7b/fp32/1` | `model-pipeline/src/model_pipeline/interventions/injection_sites.py`, keyed to the fp32 lock's `onnx/model.onnx` SHA-256 `add6653f…5ef77`. Sites: `residual_post` after each of the 24 decoder layers (`/model/layers.L/Add_2/output_0`, float32, hidden 2048). Both readers of each tensor are rewired. The fp16 graph has no mapping. |
| Synthetic compile | `python -m model_pipeline.interventions.lfm3_synthetic`: writes profile `lfm3-synthetic-test` (rank 8 after layers 8 and 16, seeded random orthonormal directions, scales 1) and compiles it with `get_or_compile`, passing the lock's external data hashes. |
| Load and verify | `Lfm3Interventions.CreateModel` / `CreateChatModel`: open the cache entry (`InterventionArtifactCache.OpenEntry`: name = key), check the manifest before any session exists, load the derived graph through `OnnxModelSession`. |
| Synthetic two-profile compile | `python -m model_pipeline.interventions.lfm3_synthetic --multi`: profile 0 is `lfm3-synthetic-test` (byte for byte the profile above), profile 1 `lfm3-synthetic-add` (`directional_addition`, rank 2, scales 1 and 0.5, after layers 12 and 16); compiled with `get_or_compile_profiles` into one graph with strength input `[2]`. Layer 16 is shared: the projection runs first, then the addition. |
| Per-request strength | `model.CreateSequence(selection)`, `chat.Generate/GenerateStreaming/Chat(..., selection, ...)` extension methods. The selection is a set of `(profile id, strength)` pairs resolved to the strength vector (unnamed profiles 0). |
| Graph check | `Lfm3Model` accepts the 39 inputs of the LFM3 step graph plus exactly the manifest's one strength input (shape `[n]`, n = profiles). |

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

`Lfm3Interventions.CreateChatModel(files, entryDirectory)` also loads the snapshot's `tokenizer.json` (`Starlings.Tokenizers` 0.3.4, the pin; profile `lfm3-bpe-chat`) and returns an `Lfm3ChatModel` that owns the graph; `chat.Chat(turns, selection, options)` / `chat.GenerateStreaming(promptIds, selection, options)` take the selection per call. The chat example, with its `using` directives, is in the [package README](../../ai-dotnet/src/Starlings.Models.Lfm3.Interventions/README.md#compile-once-then-load); the application adds the ONNX Runtime native package (`Microsoft.ML.OnnxRuntime`).

### What `CreateModel` checks before loading

1. `InterventionArtifactCache.OpenEntry` (Fast by default): the directory name is the key recomputed from the manifest; manifest and profile parse and match, directions verify, the derived graph's SHA-256, the linked weights' sizes, device, inode and modification time against the manifest (and, with the snapshot directory, that they are still the snapshot's files), and their identity with the base artifacts. `Full` also hashes the linked weights and the snapshot's base files.
2. The manifest's base graph SHA-256 equals the lock's entry for the requested precision's graph. A wrong base graph (another export, the fp16 graph) throws `InterventionMismatchException` and no session is created.
3. The manifest precision matches (`fp32`), and every base artifact (the graph and its four external data files) is the lock's file with the lock's SHA-256 and size; no locked graph file is missing. The graph file holds only offsets into the weights, so its hash alone does not identify them, and Fast verification compares the linked weights with the manifest by size and file identity, not by hash.
4. The site mapping version is `lfm3-1.7b/fp32/1` (`Lfm3Interventions.Fp32SiteMappingVersion`); another mapping version is rejected until this integration is verified with it.
5. Exactly one extra input: the strength, float32, shape `[n]` with `n` the number of profiles (`[1]` for a schema version 1 manifest), not named like an LFM3 input.
6. Every site is inside the 24 decoder layers at hidden size 2048.

The session takes the derived graph's verified SHA-256 as `KnownModelSha256`, so `GetInterventionDiagnostics()` reports the original identity (the lock's), the loaded identity (the derived graph's, source `Known`), every profile id (strength-vector order), the strength input and the 39 model inputs.

### Request semantics

- The selection is resolved to the strength vector when the sequence is created (each named profile gets its strength, every other profile 0) and fixed for its lifetime; `Lfm3Sequence` has no way to change it. Each sequence owns its `[n]` strength array (a copy); each native call wraps it in a tensor that lives until the call returns. No tensor is shared between sequences.
- No selection on a derived graph: every strength 0. A selection on the original graph: `InvalidOperationException`. A selection naming a profile the graph does not carry: `InterventionMismatchException`.
- Single-profile entries (manifest schema version 1) load and run as before: a strength vector of length 1.

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
| Chat model (tokenizer 0.3.4): `Generate` and `GenerateStreaming` at strength 1 give the same ids, equal to the sequence's greedy ids; no selection gives the original graph's ids; another profile id throws; the tokenizer encodes the rendered conversation to the reference prompt ids and `Chat(turns, selection)` gives the same ids and text as `Generate` | pass (6 new tokens) | `ChatGenerateAndStreamingPassTheSelection` |
| fp16 graph hash, wrong base hash, external data not the locked weights, missing locked weight file, unknown site mapping version, wrong precision, scalar or colliding strength input, site layer 24 | rejected before any session | `ManifestCheckTests`, `TheArtifactIsRejectedForAnotherBaseGraphBeforeAnySession` |

Two profiles in one graph (cache key `7496ccf0…1e06b`, manifest schema version 2, strength input `[2]`), same prompt, linux-x64 CPU:

| Check | Result | Where |
| --- | --- | --- |
| Vector (0, 0) vs original, logits (Python ORT and .NET) | max abs 0; no selection = (0, 0) = `{add: 0}` | `test_strength_vector_zero_matches_the_base_and_one_profile_matches_its_single_graph`, `Lfm3MultiProfileModelTests.TheZeroVectorMatchesTheOriginalGraph` |
| (1, 0) vs the single-profile graph at strength 1 (Python ORT and .NET) | max abs 0 | same, `OneProfileAloneMatchesTheSingleProfileGraph` |
| (0, 1) vs original; (1, 0) vs (1, 1) | max abs 10.3; 9.77 | same, `ProfilesCombineAndSelectionOrderDoesNotMatter` |
| Shared layer-16 site vs NumPy `y1 = y - a((yQ0ᵀ)⊙s0)Q0`, `y2 = y1 + b(s1Q1)`, strengths (1, 1) and (0.5, -2) | max abs ≤ 6.7e-7 (projection), ≤ 1.1e-6 (addition) | `test_shared_site_matches_the_numpy_composition` |
| Selection order `{test: 1, add: 1}` vs `{add: 1, test: 1}` (.NET) | equal | `ProfilesCombineAndSelectionOrderDoesNotMatter` |
| 10 concurrent sequences, vectors (0,0), (1,0), (0,1), (0.5,-1), (-1,2) on one session vs sequential (.NET) | within 1e-5 each | `ConcurrentSequencesWithDifferentVectorsAreIsolated` |
| Unknown profile id, wrong vector length | rejected (`InterventionMismatchException`, `ArgumentException`) | `ASelectionOfAnUnknownProfileOrAWrongVectorIsRejected` |
| Diagnostics list both profile ids | yes | `DiagnosticsListEveryProfile` |
| The single-profile class (`Lfm3InterventionModelTests`) on a fresh schema 1 entry and on the entry compiled by the previous change | 7 passed, 1 skipped (chat, then blocked by the 0.3.2 tokenizer pin) each | |

The synthetic addition at strength 1 moves the logits far more than the projection (10.3 against 1.97): it adds a unit-norm direction to the residual stream whatever its content. The directions are random, not behavioural.

The real-model test classes each hold an fp32 session in a process-wide fixture; run each in its own test process (`-class ...`). They share an xunit collection, so they never run in parallel.

Strength 0 is bit-identical here; other graphs, providers or optimization levels may only match within rounding (the extra reader of the residual tensor can change operator fusion).

The synthetic directions at strength 1 left the 6 greedy ids of this prompt unchanged although the logits moved by up to 1.97: the directions are random, not behavioural.

The compile command above and both code examples (this page's sequence example and the package README's chat example, with `OpenLocal`'s default SHA-256 verification) ran end to end against the same snapshot; the compile reproduced the cache key `5ff9b1d7…d025`.

## Not covered

- The real-model tests need `STARLING_LFM3_SNAPSHOT` and a compiled entry (`STARLING_LFM3_INTERVENTION`, and `STARLING_LFM3_INTERVENTION_MULTI` for the two-profile graph, for .NET; `STARLING_LFM3_INTERVENTION_CACHE` for Python); CI runs only the model-free tests (manifest checks, mapping).
- The fp16 graph (no site mapping), CUDA, Windows, AOT publish.
- Hot swap. On LFM3 two profiles per graph were measured; three only on the synthetic graphs of the compiler tests. A behavioural evaluation of any profile other than `concise`.
