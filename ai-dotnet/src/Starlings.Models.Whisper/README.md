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

Not published to nuget.org yet: the package is not in the 0.4.0 publishable set (`docs/releases/ai-0.4.0.md`, "Not published"), because it is not in the release drill's package set and its description lacks the per-profile maturity line. Build it from this repository (a `ProjectReference` to `ai-dotnet/src/Starlings.Models.Whisper/Starlings.Models.Whisper.csproj`). Once published:

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
using var transcriber = WhisperTranscriber.Create(files, new WhisperOptions { Language = "en" });

// 16 kHz float32 PCM, up to 30 seconds.
string text = transcriber.Transcribe(waveform);

// The same clip with segment timestamps (seconds on the clip's timeline).
WhisperTranscription result = transcriber.TranscribeWithTimestamps(waveform);
foreach (WhisperSegment segment in result.Segments)
{
    Console.WriteLine($"[{segment.Start:0.00} -> {segment.End:0.00}]{segment.Text}");
}
```

`WhisperTranscriber.Create(files)` and `WhisperModel.Create(files)` pass the graph digests a snapshot verified with `CacheVerification.Sha256` already computed (`WhisperModelFiles.EncoderSha256`, `DecoderSha256`) to the sessions as `OnnxSessionOptions.KnownModelSha256`, so the graphs are read once at startup. The path overloads hash each graph as before.

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

- Transcription of up to 30 s of 16 kHz audio, greedy decoding. Shorter clips are zero-padded to 30 s
  (`WhisperMelSpectrogram.ExtractChunk`, as the feature extractor pads); longer clips are refused, not truncated.
- Segment timestamps (experimental): `WhisperTranscriber.TranscribeWithTimestamps` (or `TranscribeMelWithTimestamps`)
  returns a `WhisperTranscription`: the text, the segments (`Start`, `End` in seconds, `Text`, token `Ids`) and the
  decoded windows. The segments are those transformers 5.18.0 `generate(return_timestamps=True, return_segments=True)`
  returns with greedy decoding (not the tokenizer's `decode(output_offsets=True)`, a different algorithm): the
  `WhisperTimeStampLogitsProcessor` rules with `generation_config.json`'s values (`WhisperModel.GenerateWithTimestamps`)
  and its sequential seek loop, which decodes again from the last closing timestamp when a window ends inside a
  segment. The route's corpus exercises a timestamp pair and that re-decode (a clip cut at 30 s, two windows); the
  448-position cut-off and a window that closes no segment are covered by unit tests only.
  `WhisperOptions.NoTimestamps = false` makes `Transcribe` return that text.
- The timestamp-or-text decision compares the timestamps' summed log-probability with the best text token's in double
  precision; transformers computes it in float32. On a near-tie the two can pick differently (a timestamp against a
  text token); the route's replay of every reference window found no such tie on its corpus, but it is not a
  bit-for-bit guarantee.
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

Translation, word-level timestamps (the pinned graphs do not export the alignment heads' cross-attention), chunking
and streaming of clips longer than 30 s, temperature fallback and the compression-ratio, log-probability and no-speech
thresholds, conditioning on earlier text or prompts, batching, KV-cache decoding, CUDA qualification, persistence,
training.

Without a KV cache, every decode step re-runs the decoder over all ids so far. `WhisperModel.Generate` writes the
`[seq, 51866]` logits of those steps into one buffer per call, which grows by doubling, instead of a new array per step.

## Capabilities

See `WhisperCapabilities`; transcription is `Experimental` until the evaluator qualifies it. The registered profile is
`WhisperCapabilities.CpuProfile` (the profile in the maturity table above, Experimental).
`WhisperCapabilities.StableProfile` is the same string under its old name; it is obsolete because the profile is not
Stable, and is removed in the next minor version.
