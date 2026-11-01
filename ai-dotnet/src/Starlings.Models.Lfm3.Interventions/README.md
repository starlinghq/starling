# Starlings.Models.Lfm3.Interventions

Graph interventions for LFM3-1.7B ([Starlings.Models.Lfm3](../Starlings.Models.Lfm3/README.md)): load a derived intervention graph (the original fp32 weights, hard-linked, plus one or more profiles' low-rank operators at chosen residual-stream sites) and run each request with its own strengths.

This library is **experimental** and source only (a packable library cannot depend on the source-only `Starlings.Interventions`; `Starlings.Models.Lfm3` itself is unchanged for applications that do not use interventions). V1 evidence uses synthetic random orthonormal directions only: no behavioural profile ships or has been evaluated. Deciding which callers may apply which profile is the host application's job.

The LFM3 weights are under the LFM Open License v1.0 (see the LFM3 package README).

## Compile once, then load

The derived graph is compiled by the Python compiler into the derived-artifact cache ([cache.md](https://github.com/starlinghq/starling/blob/main/docs/interventions/cache.md)). The cache must be on the snapshot's file system (hard links). The supported base graph is the fp32 `onnx/model.onnx` of the asset lock (site mapping `lfm3-1.7b/fp32/1`: the residual stream after any of the 24 decoder layers); the fp16 graph has a model-pipeline mapping (`lfm3-1.7b/fp16/1`), but this integration has not been verified with it and rejects fp16 entries.

```sh
cd model-pipeline
python -m model_pipeline.interventions.lfm3_synthetic \
    --snapshot "$STARLING_MODELS/LiquidAI/LFM3-1.7B-ONNX/c680759afee1ce8310c90d6c621f4558200dee77" \
    --cache "$HOME/.cache/starlings/interventions"
# prints the entry directory, <cache>/lfm3/interventions/<key>
```

That command compiles a synthetic test profile; a real profile goes through `model_pipeline.interventions.get_or_compile` the same way. With `--multi` it compiles two synthetic profiles (a projection and an addition) into one graph with a strength vector of length 2 (`get_or_compile_profiles`).

The application adds the ONNX Runtime native package (`Microsoft.ML.OnnxRuntime`), as for `Starlings.Models.Lfm3`.

```csharp
using Starlings.Generation;          // ChatTurn, GenerationOptions, GenerationResult
using Starlings.Interventions;
using Starlings.Models.Lfm3;
using Starlings.Models.Lfm3.Interventions;
using Starlings.Runtime.Onnx;        // OnnxInterventionDiagnostics

IReadOnlyList<ChatTurn> turns = [new ChatTurn(ChatTurnRole.User, "What is 2 + 2? Answer briefly.")];
Lfm3ModelFiles files = Lfm3ModelAssets.OpenLocal(Lfm3Precision.Fp32);
// Checks the manifest before any session exists: graph and external data = the fp32 lock's files, precision fp32,
// site mapping lfm3-1.7b/fp32/1, exactly one float32 [n] strength input (n profiles), sites inside the 24 layers. Then loads the derived graph through OnnxModelSession.
// The tokenizer is the snapshot's tokenizer.json through Starlings.Tokenizers (profile lfm3-bpe-chat, from 0.3.3).
using Lfm3ChatModel chat = Lfm3Interventions.CreateChatModel(files, entryDirectory);

var selection = new InterventionSelection("lfm3-synthetic-test", strength: 0.5f);
GenerationResult steered = chat.Chat(turns, selection, new GenerationOptions { MaxNewTokens = 64 });
GenerationResult plain = chat.Chat(turns, selection: null);   // strength 0: the original model's output

OnnxInterventionDiagnostics d = chat.Model.GetInterventionDiagnostics();  // identities, profile ids, strength input

// A graph compiled with several profiles: each request picks its own combination.
var both = new InterventionSelection([new("lfm3-synthetic-test", 1f), new("lfm3-synthetic-add", 0.5f)]);
using Lfm3Sequence sequence = chat.Model.CreateSequence(both);   // strength vector [1, 0.5], fixed for the sequence
```

## Behaviour

- **Per request.** The selection is resolved to the strength vector when a sequence is created (`model.CreateSequence(selection)`, or a `Generate`/`GenerateStreaming`/`Chat` call): each named profile gets its strength, every other compiled profile 0. The vector is fixed for the sequence's lifetime; there is no way to change it midway. Each native call gets a `[n]` strength tensor over the sequence's own array, alive for the whole call. Concurrent requests with different vectors share the weights and nothing else.
- **No selection** on an intervention graph means every strength 0. A selection on the original graph throws `InvalidOperationException`; one naming a profile the graph does not carry throws `InterventionMismatchException`.
- **Original graph unchanged.** Without interventions, load with `Lfm3Model.Create(files)`: same inputs, same cost, no code from this library runs.
- **Graph check.** `Lfm3Model` accepts the LFM3 step graph's inputs plus exactly the manifest's one strength input, and nothing else.
- **Earlier graphs.** A single-profile entry (manifest schema version 1, strength `[1]`) loads as before; it is a strength vector of length 1.

## Capabilities

`Lfm3InterventionCapabilities.All`: `graph.fp32`, `selection.per_request`, `profiles.multiple_per_graph` and `operator.directional_addition` are Experimental; `graph.fp16`, hot swap and behavioural profiles are Unsupported; `cuda` and `aot` are Untested.
