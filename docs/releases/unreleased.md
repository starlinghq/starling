# ai-dotnet: unreleased changes

Changes merged to `main` after the last release record, collected here for the next one. The release-candidate change moves them into `ai-<version>.md` and empties this file. Nothing here is published until then.

## New package: Starlings.Models.Lfm3 (LFM3-1.7B, Experimental; #217)

**Model license: LFM Open License v1.0.** Commercial use of LFM3 is licensed only to a legal entity below USD 10 million annual revenue (section 5). The package is the first whose NuGet license is not `MIT OR Apache-2.0`. Its license file, `LICENSE.txt`, states the condition first, then gives the full LFM Open License v1.0 text and the split: the ported chat template is under that license, and the rest of the code is MIT OR Apache-2.0. NuGet asks for acceptance on install (`requireLicenseAcceptance`). The description and README open with the condition, and `THIRD-PARTY-NOTICES.md` is packed. The release drill checks all of this (`tools/release-drill-ai.sh`, `.ps1`).

**What it does.** `LiquidAI/LFM3-1.7B` text generation on ONNX Runtime 1.30.0 through Liquid's ONNX export (`LiquidAI/LFM3-1.7B-ONNX` at `c680759a`). It has an owned greedy decode loop over the step graph (`Lfm3Model`, `Lfm3Sequence`, `Lfm3ChatModel`), a C# port of the chat template for text turns (`Lfm3ChatTemplate`), and asset locks for the fp32 and fp16 graphs (`Lfm3ModelAssets`). It needs `Starlings.Tokenizers` 0.3.3 or later (profile `lfm3-bpe-chat`); the pin is now 0.3.4 (below), so `release.yml` no longer holds it (`HELD` is empty). Registry profile `lfm3-1.7b/onnx-fp32-cpu`, Experimental. Evidence: `docs/evidence/lfm3-onboarding-2026-10-28/`. Not run: the PyTorch reference, fp16 and CUDA.

## Tokenizer: Starlings.Tokenizers 0.3.4

Every package that tokenizes now pins `Starlings.Tokenizers [0.3.4]` (0.3.2 in 0.4.0), and every manifest's `processing.tokenizer.version` moved with the pin. 0.3.3 adds the `Sequence` post-processor, the Llama 3 split regex and the profiles `lfm3-bpe-chat`, `qwen3-bpe-embedding` and `qwen3-bpe-chat`; 0.3.4 adds only the profile `clover-bpe-chat`. 0.3.2 rejected a `Sequence` post-processor and the Llama 3 regex at load, so no pinned asset of an existing profile declares either, and the tokenizer's conformance fixtures pass unchanged (starlinghq/tokenizers CHANGELOG). The LFM3 tokenizer now loads, so `Lfm3ParityTests.PromptIdsMatchTheReferenceTokenizer` and the LFM3 intervention chat test run against the pin.

## Starlings.Models.Kev (#166)

- `KevDecisionModel.CanLoad(options, out reason)` checks the options and the requested provider (`OnnxRuntimeInfo.TryInitializeProvider`) before any graph is read, so a consumer need not load 3 to 17 GB to learn that CUDA is unavailable. Capabilities `load_check` (experimental) and `load_check.cuda` (untested, no GPU evidence).
- `KevDecision[id]` is a hash lookup (ordinal, first answer on a repeated id) instead of a linear scan; record equality and copies are unchanged. A null `id` now throws `ArgumentNullException` (it threw `KeyNotFoundException` before).
- README: noul answers are in no, yes order; `Yes` and `Probabilities["true"]` give p(yes) without reordering. `KevInputTooLongException` stays an `ArgumentException`, as the shared `InputTooLongException` base is.

## Capability key `text.truncation` (#193)

The capability key `truncation` keeps its encoder-side meaning (the encoder rejects id sequences over the profile's limit instead of truncating them), so no Stable profile's documented contract changes. A new key, `text.truncation`, says what happens to text: `Supported` in `E5Capabilities.Statuses` (`E5Tokenizer` truncates on the right to 512 tokens, as the reference), `Unsupported` in `EmbeddingGemma2Capabilities` and `Gliner2Capabilities` (longer text is rejected), and the new `BertCapabilities.TextTruncation` is `experimental`, as the whole BERT route is. The README tables name the old row "encoder-side truncation" and point to the typed `ModelProfileCapabilities.Truncation` (`TruncationPolicy`) as the authoritative text policy. Decision recorded in `docs/kickoff.md` ("Capability key `truncation`", 2026-11-01).

## Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp: stream byte limit (#194)

New `EmbeddingGemma2ImageDecoderOptions.MaxEncodedBytes` (default 64 MiB, 67,108,864; 1 to `Array.MaxLength`). `EmbeddingGemma2ImageDecoder.Decode(Stream)` now rejects a seekable stream whose remaining length is over the limit before reading anything, and a non-seekable stream as soon as more than the limit has been read, with `EmbeddingGemma2ImageTooLargeException`. Before, it buffered the whole stream before the `MaxPixels` header check. **Behavior change:** a stream above 64 MiB that decoded before now throws; raise `MaxEncodedBytes` to keep it. The byte-span and file overloads are unchanged, and decoding itself is unchanged, so the PIL-equality evidence of the image profiles still holds (no G2 rerun needed). Decision recorded in `docs/kickoff.md` ("SkiaSharp `Decode(Stream)` byte bound", 2026-11-01).

## Starlings.Models.Whisper

- **Segment timestamps (experimental, #221).** `WhisperTranscriber.TranscribeWithTimestamps` and
  `TranscribeMelWithTimestamps` return a `WhisperTranscription` (text, `WhisperSegment` list with start and end in
  seconds, text and token ids, and the decoded `WhisperWindow`s); with #222 below, for audio of any length. `WhisperModel.GenerateWithTimestamps`
  applies transformers 5.18.0 `WhisperTimeStampLogitsProcessor`; the transcriber follows `generate(return_timestamps=True)`'s
  sequential seek loop, greedy, without fallbacks. Capability `timestamps` moves from unsupported to experimental; the new
  `word_timestamps` is unsupported.
- **Obsolete:** `WhisperCapabilities.StableProfile`, replaced by `WhisperCapabilities.CpuProfile` (same value; the profile is
  Experimental, not Stable). Removed in the next minor version.
- `WhisperTokenizer.Decode` drops timestamp tokens (ids from 50365), as transformers `WhisperTokenizer.decode` does.
- `WhisperOptions.NoTimestamps = false` now makes `Transcribe` return the timestamped decode's text, instead of
  prompting without `<|notimestamps|>` and decoding without the timestamp rules.
- **KV-cache decoding (experimental, #192), now the default.** `WhisperModelAssets.Lock` adds
  `onnx/decoder_with_past_model_fp16.onnx` (317,742,004 bytes, same revision), so `DownloadAsync` fetches about 318 MB
  more. **Upgrading:** a cache acquired before this change lacks that file, so `WhisperModelAssets.OpenLocal` and
  `OpenDirectory` throw `ModelAssetMissingException` until `WhisperModelAssets.DownloadAsync` runs again (it fetches only
  the missing file). This applies with `DecoderMode = FullContext` too: the lock is one set of files, and that mode
  merely does not load the graph. `WhisperModel` runs the full-context decoder once over the prompt and then one cached step per token;
  `WhisperOptions.DecoderMode = WhisperDecoderMode.FullContext` keeps the previous loop. Logits differ from the
  full-context loop in the last bits (fp16 summation order; bounded on the route), ids are equal. New:
  `WhisperDecoderMode`, `WhisperOptions.DecoderMode` and `DecoderWithPastSession`, `WhisperModelFiles.DecoderWithPastPath`,
  path overloads of `WhisperModel.Create` and `WhisperTranscriber.Create` with the KV-cache graph, and
  `WhisperModel.GenerateRecordingLogits` (diagnostics). Capability `kv_cache`: experimental.
- **Long-form transcription (experimental, #222).** `Transcribe` and `TranscribeWithTimestamps` accept clips
  of more than 3000 mel frames (480,160 samples or more; up to 480,159 samples stay one window, as in transformers) and
  transcribe them as transformers 5.18.0 sequential long-form `generate` does: the new
  `WhisperMelSpectrogram.ExtractLongForm` computes one spectrogram over the whole clip, and the timestamped decode walks
  it in 30 s windows, seeking to the last closing timestamp. `TranscribeMelWithTimestamps` takes a spectrogram of any
  number of frames. Greedy only; the temperature fallback, compression-ratio, log-probability and no-speech thresholds
  and conditioning on earlier windows stay unsupported (new capability keys say so). Capability `chunking` moves from
  unsupported to experimental. The Whisper route reads `asr-corpus-v3.jsonl`, which adds a 34.5 s and a 59.1 s
  composite of consecutive LibriSpeech utterances. `WhisperMelSpectrogram.ExtractLongForm` takes a `CancellationToken`
  (observed every 1024 frames); the spectrogram costs about 70 s of one core and about 1 GB per hour of audio.
- The Whisper onboarding route reads `asr-corpus-v2.jsonl`: the four clips of v1, a 27.9 s composite of six
  consecutive LibriSpeech utterances, and a composite cut at 30 s inside its last utterance (its decode ends inside a
  segment and is decoded again from the last closing timestamp). It checks the timestamped decode of every clip and
  that the corpus exercises a timestamp pair and that re-decode.

## Starlings.Models.Abstractions: reranking helpers for vector-search pipelines (Experimental; #168)

New `Reranking` and `RerankedItem<T>(Item, Index, Score)` put an `IReranker` (today `BertReranker`, for example `bge-reranker-v2-m3`) after a first-stage retriever. Microsoft.Extensions.AI has no reranker abstraction, so the helpers take any sequence plus a text selector and add no package reference: `RerankAsync` reads an `IAsyncEnumerable<T>` such as a `Microsoft.Extensions.VectorData` `SearchAsync` top-k, scores it with one `Score` call and orders by descending score (ties in input order, optional `top`); `Rerank` is the synchronous form; `AsScoringFunction` wraps an `IReranker` as a `Func<string, IReadOnlyList<string>, Task<float[]>>`, and a `RerankAsync` overload takes any such scorer. Scoring runs on the calling thread and returns a completed task. API only and Experimental; every reranker profile stays Experimental (the Stable `bge-reranker-v2-m3` profile is still open in #168).
