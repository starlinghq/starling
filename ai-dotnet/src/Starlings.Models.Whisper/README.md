# Starlings.Models.Whisper

Speech-to-text for `openai/whisper-large-v3-turbo` on ONNX Runtime: 16 kHz audio to the 128-bin log-mel
spectrogram, through the encoder and the autoregressive decoder (greedy), and back to text through the owned
tokenizer (`Starlings.Tokenizers`, profile `whisper-bpe`). This package ships the asset lock, not the weights; acquire
them explicitly. The application adds the ONNX Runtime native package.

This library is **experimental** and not yet qualified. The log-mel spectrogram, the encoder and the greedy decoder are
verified bit-exact against the reference (transformers `WhisperFeatureExtractor` numpy path and the same ONNX graphs);
end-to-end transcript parity against a fresh transformers reference is checked by the model-pipeline route
`whisper-onnx-cpu-v1` (`model-pipeline/candidates/whisper-large-v3-turbo/`). Its tolerances are adopted
(`model-pipeline/protected/whisper-large-v3-turbo/onnx-fp16/tolerances.yaml`); no reference fixtures are adopted and no
release gate has run.

## Maturity

<!-- catalog:maturity profiles=whisper-large-v3-turbo/* -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `whisper-large-v3-turbo/onnx-fp16-cpu` | experimental | linux-x64 cpu: experimental; linux-x64 cuda: untested; windows-x64 cpu: untested |
<!-- /catalog:maturity -->

## Install

```
dotnet add package Starlings.Models.Whisper
dotnet add package Microsoft.ML.OnnxRuntime     # CPU native runtime
```

## Quick start

```csharp
using Starlings.Models.Whisper;

// One-time setup: download the fp16 graphs and tokenizer into the asset store.
var files = await WhisperModelAssets.DownloadAsync();

// Load offline at startup.
using var transcriber = WhisperTranscriber.Create(
    files.EncoderPath, files.DecoderPath, files.TokenizerPath,
    new WhisperOptions { Language = "en" });

// 16 kHz float32 PCM, up to 30 seconds.
string text = transcriber.Transcribe(waveform);
```

## Model

| Setting | Value |
| --- | --- |
| Repository | `onnx-community/whisper-large-v3-turbo-ONNX` |
| Revision | `76ba4fb597cca51a329a1820e49282e26248b36b` |
| Mel | 128 bins, n_fft 400 / hop 160, 16 kHz, Slaney scale, 30 s → 3000 frames |
| Encoder | 32 layers, d_model 1280 → `last_hidden_state` [1500, 1280] |
| Decoder | 4 layers → `logits` [seq, 51866] |
| Vocabulary | 51866 |

## Supported

- Transcription of up to 30 s of 16 kHz audio, greedy decoding, no timestamps. Shorter clips are zero-padded to 30 s
  (`WhisperMelSpectrogram.ExtractChunk`, as the feature extractor pads); longer clips are refused, not truncated.
- Explicit language, a code of `WhisperModel.LanguageToId` (case-insensitive). `WhisperOptions.Language` defaults to
  `"en"` (`WhisperOptions.DefaultLanguage`), the language the onboarding evidence covers. An unknown code throws
  `WhisperUnsupportedLanguageException` (an `ArgumentException` that carries the code), from `WhisperTranscriber.Create`
  and from `WhisperModel.Prompt`.
- Auto-detection is opt-in with `Language = null` and **untested**: the language slot then holds `<|startoflm|>`, which
  is a task token, not a language token. Transformers instead picks the most likely language token after
  `<|startoftranscript|>`.
- `WhisperTranscriber.Create` checks `tokenizer.json` against `WhisperModelAssets.TokenizerSha256` before it loads the
  graphs, and throws `InvalidDataException` for any other file. The graphs are checked only when
  `OnnxSessionOptions.ExpectedModelSha256` is set.
- The asset lock and explicit acquisition.

## Unsupported (for now)

Translation, timestamped output, chunking/streaming of clips longer than 30 s, batching, KV-cache decoding,
CUDA qualification, persistence, training.

Without a KV cache, every decode step re-runs the decoder over all ids so far. `WhisperModel.Generate` writes the
`[seq, 51866]` logits of those steps into one buffer per call, which grows by doubling, instead of a new array per step.

## Capabilities

See `WhisperCapabilities`; transcription is `Experimental` until the evaluator qualifies it. The registered profile is
`WhisperCapabilities.CpuProfile` (the profile in the maturity table above, Experimental).
`WhisperCapabilities.StableProfile` is the same string under its old name; it is obsolete because the profile is not
Stable, and is removed in the next minor version.
