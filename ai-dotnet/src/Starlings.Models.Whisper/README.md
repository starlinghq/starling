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

// 16 kHz float32 PCM. Over 30 seconds, the clip is transcribed sequentially in 30 s windows.
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

- Transcription of 16 kHz audio, greedy decoding, in one window when the spectrogram fits one (at most 3000 frames).
  Up to 30 s the clip is zero-padded to 30 s (`WhisperMelSpectrogram.ExtractChunk`, as the feature extractor pads). A
  clip of 480,001 to 480,159 samples still has 3000 frames over the whole clip and is decoded in one window too, as
  transformers treats it (`is_shortform = frames <= 3000`). Longer clips: long-form, below.
- Long-form audio of more than 3000 frames, from 480,160 samples (experimental): `Transcribe` and `TranscribeWithTimestamps` follow transformers 5.18.0
  sequential long-form `generate`: one spectrogram over the whole clip (`WhisperMelSpectrogram.ExtractLongForm`, as
  `WhisperFeatureExtractor(truncation=False, padding="longest")` computes it, the max - 8 clamp taken over the whole
  clip), decoded in 3000-frame windows with timestamps; each next window starts at the last closing timestamp, or
  after the whole window when its last segment is closed. Segment times are on the clip's timeline. Greedy, with no
  temperature fallback, no compression-ratio, log-probability or no-speech threshold, and no conditioning on earlier
  windows. A long-form clip has no single-window, timestamp-free decode (transformers refuses
  `return_timestamps=False` for more than 3000 frames), so `Transcribe` returns the timestamped text.
  Cost of the whole-clip spectrogram: its naive DFT (kept for bit-exactness; an FFT sums in another order) takes about
  70 s of one core per hour of audio, and it allocates about 18 bytes per sample (about 1 GB per hour of audio) on top
  of the caller's samples. The `CancellationToken` of `Transcribe` and `TranscribeWithTimestamps` is observed inside the
  spectrogram (every 1024 frames). There is no length limit: bound the audio you accept.
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

Translation, word-level timestamps (the pinned graphs do not export the alignment heads' cross-attention), streaming,
the long-form temperature fallback and the compression-ratio, log-probability and no-speech thresholds, conditioning
on earlier text or prompts, batching, the merged decoder graph, CUDA qualification, persistence, training.

## Decoder loop

The KV cache is the default (experimental): `WhisperModelAssets.Lock` includes `onnx/decoder_with_past_model_fp16.onnx`
(the same pinned revision), and `WhisperModel` runs the full-context decoder once over the prompt, then
`decoder_with_past` one token per step with every layer's cached keys and values. `WhisperOptions.DecoderMode =
WhisperDecoderMode.FullContext` keeps the reference loop, which re-runs the full-context decoder over all ids at every
step (into one logits buffer per call that grows by doubling) and does not load the KV-cache graph. The path overloads
of `Create` take the KV-cache graph as an extra path; without it the loop is full-context. A cache acquired before the lock gained the
KV-cache graph must run `WhisperModelAssets.DownloadAsync` again (it fetches only that file) before `OpenLocal` or
`OpenDirectory` succeed, also when you use `FullContext`.

The two loops' logits are **not bit-identical**: the fp16 graphs sum the attention and projections over the whole prefix
in one matrix product on the full-context path and one row at a time on the cached path, so the float16 accumulations
round differently (about 1e-3 to 1e-2 in the logits, independent of ONNX Runtime's graph optimizations). The onboarding
route bounds that difference (the candidate's proposed `kv_logits_max_abs`) and checks that ids, segments and
transcripts are identical on every corpus clip, long-form included, and that .NET's KV loop equals Python ONNX
Runtime's bit for bit.

## Capabilities

See `WhisperCapabilities`; transcription is `Experimental` until the evaluator qualifies it. The registered profile is
`WhisperCapabilities.CpuProfile` (the profile in the maturity table above, Experimental).
`WhisperCapabilities.StableProfile` is the same string under its old name; it is obsolete because the profile is not
Stable, and is removed in the next minor version.
