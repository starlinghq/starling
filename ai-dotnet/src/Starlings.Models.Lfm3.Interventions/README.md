# Starlings.Models.Lfm3.Interventions

Graph interventions for LFM3-1.7B ([Starlings.Models.Lfm3](../Starlings.Models.Lfm3/README.md)): load a derived intervention graph (the original fp32 weights, hard-linked, plus a low-rank directional projection at chosen residual-stream sites) and run each request with its own strength.

This library is **experimental** and source only (a packable library cannot depend on the source-only `Starlings.Interventions`; `Starlings.Models.Lfm3` itself is unchanged for applications that do not use interventions). V1 evidence uses synthetic random orthonormal directions only: no behavioural profile ships or has been evaluated. Deciding which callers may apply which profile is the host application's job.

The LFM3 weights are under the LFM Open License v1.0 (see the LFM3 package README).

## Compile once, then load

The derived graph is compiled by the Python compiler into the derived-artifact cache ([cache.md](https://github.com/starlinghq/starling/blob/main/docs/interventions/cache.md)). The cache must be on the snapshot's file system (hard links). The supported base graph is the fp32 `onnx/model.onnx` of the asset lock (site mapping `lfm3-1.7b/fp32/1`: the residual stream after any of the 24 decoder layers); the fp16 graph has no mapping and is rejected.

```sh
cd model-pipeline
python -m model_pipeline.interventions.lfm3_synthetic \
    --snapshot "$STARLING_MODELS/LiquidAI/LFM3-1.7B-ONNX/c680759afee1ce8310c90d6c621f4558200dee77" \
    --cache "$HOME/.cache/starlings/interventions"
# prints the entry directory, <cache>/lfm3/interventions/<key>
```

That command compiles a synthetic test profile; a real profile goes through `model_pipeline.interventions.get_or_compile` the same way.

```csharp
using Starlings.Interventions;
using Starlings.Models.Lfm3;
using Starlings.Models.Lfm3.Interventions;

Lfm3ModelFiles files = Lfm3ModelAssets.OpenLocal(Lfm3Precision.Fp32);
// Checks the manifest before any session exists: graph and external data = the fp32 lock's files, precision fp32,
// site mapping lfm3-1.7b/fp32/1, exactly one float32 [1] strength input, sites inside the 24 layers. Then loads the derived graph through OnnxModelSession.
using Lfm3ChatModel chat = Lfm3Interventions.CreateChatModel(files, entryDirectory);

var selection = new InterventionSelection("lfm3-synthetic-test", strength: 0.5f);
GenerationResult steered = chat.Chat(turns, selection, new GenerationOptions { MaxNewTokens = 64 });
GenerationResult plain = chat.Chat(turns, selection: null);   // strength 0: the original model's output

OnnxInterventionDiagnostics d = chat.Model.GetInterventionDiagnostics();  // identities, profile id, strength input
```

## Behaviour

- **Per request.** The selection is resolved when a sequence is created (`model.CreateSequence(selection)`, or a `Generate`/`GenerateStreaming`/`Chat` call) and fixed for its lifetime; there is no way to change it midway. Each native call gets a strength tensor over the sequence's own array, alive for the whole call. Concurrent requests with different strengths share the weights and nothing else.
- **No selection** on an intervention graph means strength 0. A selection on the original graph throws `InvalidOperationException`; one naming another profile throws `InterventionMismatchException`.
- **Original graph unchanged.** Without interventions, load with `Lfm3Model.Create(files)`: same inputs, same cost, no code from this library runs.
- **Graph check.** `Lfm3Model` accepts the LFM3 step graph's inputs plus exactly the manifest's one strength input, and nothing else.

## Capabilities

`Lfm3InterventionCapabilities.All`: `graph.fp32` and `selection.per_request` are Experimental; `graph.fp16`, multiple profiles, hot swap and behavioural profiles are Unsupported; `cuda` and `aot` are Untested.
