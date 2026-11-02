# Starlings.Models.Kev

The model definitions of the Kev 1.0 decision models ([`jaredpalmer/kev-0.8b`](https://huggingface.co/jaredpalmer/kev-0.8b) and [`jaredpalmer/kev-4b`](https://huggingface.co/jaredpalmer/kev-4b)), without a runtime: requests, Kev's request rendering and escaping, the encoder that turns a request into token-id rows, the checkpoint identities, and the result types. No ONNX Runtime dependency (only the owned tokenizer `Starlings.Tokenizers`, and `Starlings.Acquisition` for the checkpoint locks). To run the models on ONNX Runtime, reference [`Starlings.Models.Kev.Onnx`](https://github.com/starlinghq/starling/blob/main/ai-dotnet/src/Starlings.Models.Kev.Onnx/README.md), which brings this package. Experimental; the API may still change.

- `KevRequest` (a state and named `KevQuestion`s: `Noul`, `Choice`, `Score`; `ParseSystemOne` for a System One request body) and `KevContent` (text, or JSON rendered as Kev does).
- `KevText`: Kev's rendering of JSON values (Python `str()` of numbers and booleans), option text, and `EscapeSpecialTokens`, which rewrites `<|…|>` in user text so it can never become a delimiter.
- `KevRequest.ToRecord()`: the rendered record Kev's encoder reads (`KevRecord`, `KevRecordQuestion`).
- `KevEncoder`: `FromFile(tokenizerPath)` checks Kev's delimiter ids in the `qwen2-bpe-kev` tokenizer; `Encode(record)` returns the state ids and one row per question (`KevEncoding`, `KevRow`: ids, positions continuing after the state, the `<decide>` index and each option's `</opt>` index), or throws `KevInputTooLongException` (state or row over 8,192 tokens; with `maxBranchTokens`, a branch over that budget). Every over-long question is found in one pass: the exception lists them all (`OverLongQuestions`, each a `KevOverLongQuestion` with `Index`, `Id`, `Limit`, `TokenCount`, `MaxTokens`), or, with `decline: true`, the encoding leaves them out and lists them in `KevEncoding.Declined`. `EncodeState` and `EncodeRows` are the two halves, for runtimes that reuse one state across question sets.
- `KevModelProfile` (`Kev08B`, `Kev4B`): repositories, revisions, base checkpoints, head and adapter digests, dimensions and calibrated temperature. `CheckpointLock` and `BaseCheckpointLock` are the asset locks of the Hub checkpoints themselves (Kev's adapter, head and tokenizer files; the frozen base's configuration, safetensors weights and tokenizer files), the inputs of the merge that builds the ONNX bundle. The package also carries them under `locks/` (`kev-0.8b`, `kev-4b`, `qwen3.5-0.8b-base`, `qwen3.5-4b-base`, each `.checkpoint.lock.json`; JSON lock schema version 2, read by `ModelAssetLock.Load`). Nothing downloads unless you pass a lock to `ModelAssetStore.DownloadAsync`.
- `KevDecision` (with `Declined`) and `KevAnswer` (with `Temperature` and `IsCalibrated`): the result types the runtime returns.

```csharp
using Starlings.Models.Kev;

var request = new KevRequest("The order arrived two weeks late and the box was crushed.",
    [new("sentiment", KevQuestion.Choice("How does the customer feel?", "positive", "neutral", "negative"))]);
KevEncoding rows = KevEncoder.FromFile(tokenizerPath).Encode(request.ToRecord());
// rows.StateIds is the prefill; each rows.Rows[k] runs on the state's cache and ends at its Decide index.
```

## Maturity

<!-- catalog:maturity profiles=kev-0.8b/*,kev-4b/* -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `kev-0.8b/onnx-fp32-cpu` | experimental | linux-x64 cpu: experimental; linux-x64 cuda: experimental; windows-x64 cpu: untested |
| `kev-0.8b/onnx-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
| `kev-4b/onnx-fp32-cpu` | experimental | linux-x64 cpu: experimental; linux-x64 cuda: experimental; windows-x64 cpu: untested |
| `kev-4b/onnx-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
<!-- /catalog:maturity -->

## Attribution and license

The rendering, escaping and encoding reimplement Kev's `kev/api.py` and `kev/model.py` at `6b719c3c3f367295f6ef336f4f751cf5ff970abc` (tag `kev-1.0`) by Jared Palmer, Apache-2.0; no upstream file is copied. This package's code is MIT OR Apache-2.0.
