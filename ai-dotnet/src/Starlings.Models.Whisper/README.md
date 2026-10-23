# Starlings.Models.Whisper

Speech-to-text for `openai/whisper-large-v3-turbo` on ONNX Runtime: 16 kHz audio to the 128-bin log-mel
spectrogram, through the encoder and the autoregressive decoder (greedy), and back to text through the owned
tokenizer (`Starlings.Tokenizers`, profile `whisper-bpe`). This package ships the asset lock, not the weights; acquire
them explicitly. The application adds the ONNX Runtime native package.

This library is **experimental** and not yet qualified. The log-mel spectrogram, the encoder and the greedy decoder are
verified bit-exact against the reference (transformers `WhisperFeatureExtractor` numpy path and the same ONNX graphs);
end-to-end transcript parity against a fresh transformers reference is checked by the model-pipeline route
`whisper-onnx-cpu-v1` (`model-pipeline/candidates/whisper-large-v3-turbo/`). Its evidence is proposed, not yet adopted.

## Install

Not published to nuget.org yet: the package is not in the 0.4.0 publishable set (`docs/releases/ai-0.4.0.md`, "Not published"), because its profile has no registry manifest and so no per-profile maturity statement. Build it from this repository (a `ProjectReference` to `ai-dotnet/src/Starlings.Models.Whisper/Starlings.Models.Whisper.csproj`). Once published:

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
- Explicit language (two-letter code); the onboarding evidence covers English (`en`). Auto-detection
  (`Language = null`, prompted with `<|startoflm|>`) is **untested**: it differs from transformers, which picks the most
  likely language token after `<|startoftranscript|>`.
- The asset lock and explicit acquisition.

## Unsupported (for now)

Translation, timestamped output, chunking/streaming of clips longer than 30 s, batching, KV-cache decoding,
CUDA qualification, persistence, training.

## Capabilities

See `WhisperCapabilities`; transcription is `Experimental` until the evaluator qualifies it.
