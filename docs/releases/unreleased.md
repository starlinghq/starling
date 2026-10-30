# ai-dotnet: unreleased changes

Changes merged to `main` after the last release record, collected here for the next one. The release-candidate change moves them into `ai-<version>.md` and empties this file. Nothing here is published until then.

## New package: Starlings.Models.Lfm3 (LFM3-1.7B, Experimental; #217)

**Model license: LFM Open License v1.0.** Commercial use of LFM3 is licensed only to a legal entity below USD 10 million annual revenue (section 5). The package is the first whose NuGet license is not `MIT OR Apache-2.0`. Its license file, `LICENSE.txt`, states the condition first, then gives the full LFM Open License v1.0 text and the split: the ported chat template is under that license, and the rest of the code is MIT OR Apache-2.0. NuGet asks for acceptance on install (`requireLicenseAcceptance`). The description and README open with the condition, and `THIRD-PARTY-NOTICES.md` is packed. The release drill checks all of this (`tools/release-drill-ai.sh`, `.ps1`).

**What it does.** `LiquidAI/LFM3-1.7B` text generation on ONNX Runtime 1.30.0 through Liquid's ONNX export (`LiquidAI/LFM3-1.7B-ONNX` at `c680759a`). It has an owned greedy decode loop over the step graph (`Lfm3Model`, `Lfm3Sequence`, `Lfm3ChatModel`), a C# port of the chat template for text turns (`Lfm3ChatTemplate`), and asset locks for the fp32 and fp16 graphs (`Lfm3ModelAssets`). It needs `Starlings.Tokenizers` 0.3.3 (profile `lfm3-bpe-chat`); until the pin moves there, `release.yml` holds it from publishing (`HELD`). Registry profile `lfm3-1.7b/onnx-fp32-cpu`, Experimental. Evidence: `docs/evidence/lfm3-onboarding-2026-10-28/`. Not run: the PyTorch reference, fp16 and CUDA.

## Starlings.Models.Whisper

- **Segment timestamps (experimental, #221).** `WhisperTranscriber.TranscribeWithTimestamps` and
  `TranscribeMelWithTimestamps` return a `WhisperTranscription` (text, `WhisperSegment` list with start and end in
  seconds, text and token ids, and the decoded `WhisperWindow`s) for up to 30 s of audio. `WhisperModel.GenerateWithTimestamps`
  applies transformers 5.18.0 `WhisperTimeStampLogitsProcessor`; the transcriber follows `generate(return_timestamps=True)`'s
  sequential seek loop, greedy, without fallbacks. Capability `timestamps` moves from unsupported to experimental; the new
  `word_timestamps` is unsupported.
- `WhisperTokenizer.Decode` drops timestamp tokens (ids from 50365), as transformers `WhisperTokenizer.decode` does.
- `WhisperOptions.NoTimestamps = false` now makes `Transcribe` return the timestamped decode's text, instead of
  prompting without `<|notimestamps|>` and decoding without the timestamp rules.
- The Whisper onboarding route reads `asr-corpus-v2.jsonl`: the four clips of v1 plus a 27.9 s composite of six
  consecutive LibriSpeech utterances, and checks the timestamped decode of every clip.
