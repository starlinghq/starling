# A text-to-speech route for Kokoro-82M: design note

Status: **proposed**, 2026-11-01 (#184). Nothing in this note changes product code, a capability status, a manifest or a tolerance. It is input for a maintainer decision, not a support claim.

## Problem

Starling has speech input (`Starlings.Models.Whisper`) but no speech output. `onnx-community/Kokoro-82M-v1.0-ONNX` (base model `hexgrad/Kokoro-82M`, Apache-2.0) is the small, widely used TTS model with an existing ONNX export. No route fits it: the existing routes are the encoder (BERT), GenAI, Whisper, Kev and PP-OCR routes, and none of them has text-to-waveform evaluation.

The graph, as the ONNX repository's README describes it, takes `input_ids` (int64, shape `[1, n + 2]`: phoneme ids with a pad id 0 at each end, at most 510 phoneme ids), `style` (float32 `[1, 256]`) and `speed` (float32 `[1]`), and returns a float32 waveform at 24 kHz. `style` is a row of a per-voice file (`voices/<name>.bin`, float32, 510 rows of 256) selected by the number of phoneme ids. The repository ships fp32 (`onnx/model.onnx`, 326 MB) and several quantized graphs, and 54 voice files.

## Constraints

- **Phonemisation is GPL-adjacent.** The reference text front end is misaki, which falls back to espeak-ng (GPL) for out-of-vocabulary words. A Starling library may not take a GPL dependency, and a .NET grapheme-to-phoneme port is a project of its own.
- **The repository's `tokenizer.json` cannot be used as is.** It is a per-character phoneme vocabulary built from a `Replace` normalizer with a character-class regex and a `Split` with an empty regex; both are unsupported in starlinghq/tokenizers. The phoneme-to-id table itself is small (the `vocab` map in `hexgrad/Kokoro-82M` `config.json`, under 200 entries) and can be pinned as data.
- **No remote code.** The PyTorch reference is the `kokoro` pip package: model code outside the pinned transformers and outside this repository. Running it is excluded unless the maintainer adopts that code with notices. The first route's reference is therefore Python ONNX Runtime on the pinned graph, with the contract above, and the route's evidence says plainly that it checks the .NET pipeline against the same graph, not against PyTorch.
- **Parity is waveform-level.** The metric must be declared before runs: for example maximum absolute sample difference and an SNR figure per utterance, on fixed phoneme-id inputs, voices and speeds. The thresholds are maintainer-owned and are not proposed here.
- **Determinism.** Check whether the graph contains random ops (noise injection in the decoder is common in StyleTTS-derived models). If it does, parity needs a fixed seed path or the route must document that outputs are not bit-reproducible, and the metric changes accordingly.
- **Scope.** WAV encoding only; no playback, no streaming in the first profile. CPU only, so the GPU budget in #210 does not apply. Licence Apache-2.0, to be verified again at admission with the revision pinned by full commit SHA.

## Options

**A. Pre-phonemised input contract.** The library accepts phonemes, never raw text, in one of two forms:
- A1. phoneme ids (`ReadOnlySpan<long>`), checked against the pinned vocabulary and the 510 limit;
- A2. a phoneme string, mapped character by character with the pinned table; unknown characters are rejected (the upstream normalizer drops them silently, which the library should not imitate).
The caller brings any G2P they like (misaki in a separate process, a lexicon, their own tool). No GPL code in Starling.

**B. A .NET phonemiser.** A port of misaki's English lexicon path without the espeak-ng fallback. Large, language-specific, and its own parity problem (phonemes, then audio).

**C. Bundle espeak-ng.** Excluded: GPL.

## Recommendation

Option A for the first Experimental profile, offering both A1 and A2 (A2 is a thin layer over A1 and makes the API usable from a phoneme string a user already has). Option B can follow as a separate package if demand appears; it does not change the route, only adds a front end.

Proposed shape, for discussion only: a `Starlings.Models.Kokoro` package with a voice set loaded from pinned voice files, a synthesizer taking phonemes, voice and speed and returning `float[]` samples plus the sample rate, and a WAV writer (16-bit PCM and 32-bit float). Capabilities: `text_input` unsupported, `phoneme_input` experimental, `streaming` unsupported, `voice_blending` unsupported, `speed` experimental.

## Checklist (one PR each)

1. Maintainer decisions: accept the pre-phonemised contract; name the parity metric (thresholds are set separately by the maintainer); confirm the ONNX-reference-only scope for the first profile.
2. Inspect the pinned fp32 graph (inputs, outputs, opset, random ops) and record the findings in this note; pin the ONNX repository and `hexgrad/Kokoro-82M` `config.json` by full commit SHA.
3. `model-pipeline`: a `tts` reference runner on Python ONNX Runtime, a fixed phoneme-id corpus (short, long near 510, punctuation, several voices and speeds) with its source recorded, and a waveform comparison evaluator. Orchestration tests.
4. `Starlings.Models.Kokoro`: asset lock (graph, phoneme table, chosen voices), phoneme-id and phoneme-string inputs with validation, style row selection, synthesis, WAV writer, capability statuses. Unit tests without the model; model tests gated as the other model packages do.
5. The `tts-onnx-cpu-v1` plugin wiring the reference, the .NET run and the evaluator; candidate profile and recipe.
6. Onboarding run on CPU, evidence under `docs/evidence/`, profile Experimental. The maintainer adds tolerances and the registry manifest.
7. Later, separately: a .NET phonemiser (option B), quantized graphs, CUDA.
