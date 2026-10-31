"""Pinned Whisper reference (transformers, PyTorch fp32) or ONNX Runtime execution, into proposed artifacts only.

reference: for each corpus clip, the transformers WhisperFeatureExtractor log-mel spectrogram (both its numpy path,
           which Starlings.Models.Whisper restates, and its default torch path), the encoder last_hidden_state, the
           greedy transcript from WhisperForConditionalGeneration.generate, and the decoder logits teacher-forced on
           those ids. The numpy-path mel is the input every later stage uses. Timestamped: generate(return_timestamps=
           True, return_segments=True) on the same mel, replayed window by window through this file's port of the
           timestamp rules and segment split over teacher-forced logits, which must reproduce its ids and segments.
           A clip over 30 s is long-form: the spectrogram of the whole clip (both feature paths, as
           WhisperFeatureExtractor(truncation=False, padding="longest") computes it) and the timestamped decode only.
onnx:      the pinned fp16 graphs on ONNX Runtime from the reference mel: encoder hidden state, an independent greedy
           loop with the generation config's suppression lists, and logits teacher-forced on the reference ids.
           Timestamped: the same greedy loop with the timestamp rules, and the sequential seek loop that splits each
           window into segments.
"""

from __future__ import annotations

import argparse
import json
import platform
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from model_pipeline import fixtures
from model_pipeline.onboarding_whisper import FORMAT, load_corpus, prompt, validate_candidate_behavior


def read_audio(path: Path, sample_rate: int) -> np.ndarray:
    """16-bit mono PCM WAV to float32 in [-1, 1) (sample / 32768, exact in float32)."""
    with wave.open(str(path), "rb") as reader:
        if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) != (1, 2, sample_rate):
            raise ValueError(f"{path}: expected 16-bit mono {sample_rate} Hz PCM")
        frames = reader.readframes(reader.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / np.float32(32768)


def suppressed(logits: np.ndarray, behavior: dict, first_step: bool) -> np.ndarray:
    """transformers SuppressTokensLogitsProcessor plus SuppressTokensAtBeginLogitsProcessor on one logits row."""
    row = logits.astype(np.float32, copy=True)
    row[behavior["suppress_tokens"]] = -np.inf
    if first_step:
        row[behavior["begin_suppress_tokens"]] = -np.inf
    return row


def timestamp_rules(row: np.ndarray, generated: list[int], behavior: dict) -> np.ndarray:
    """transformers 5.18.0 WhisperTimeStampLogitsProcessor on one suppressed logits row, in float64.

    `generated` holds the tokens after the decoder prompt (begin_index). The masks follow the pinned source line by
    line; the log-softmax test is restated in float64 with left-to-right sums, as Starlings.Models.Whisper computes it
    (transformers runs it in float32: the decision can differ only on a float32 tie, which the reference replay would
    report).
    """
    special, timing = behavior["special_tokens"], behavior["timestamps"]
    begin, eos = timing["timestamp_begin"], special["end_of_text"]
    row = row.astype(np.float64, copy=True)
    row[special["no_timestamps"]] = -np.inf
    last = len(generated) >= 1 and generated[-1] >= begin
    penultimate = len(generated) < 2 or generated[-2] >= begin
    if last:
        if penultimate:
            row[begin:] = -np.inf
        else:
            row[:eos] = -np.inf
    timestamps = [token for token in generated if token >= begin]
    if timestamps:
        limit = timestamps[-1] if last and not penultimate else timestamps[-1] + 1
        row[begin:limit] = -np.inf
    if not generated:
        row[:begin] = -np.inf
        row[begin + timing["max_initial_timestamp_index"] + 1 :] = -np.inf
    peak = row.max()
    logprobs = row - (peak + np.log(np.cumsum(np.exp(row - peak))[-1]))
    tail = logprobs[begin:]
    tail_peak = tail.max()
    timestamp_logprob = -np.inf if tail_peak == -np.inf else tail_peak + np.log(np.cumsum(np.exp(tail - tail_peak))[-1])
    if timestamp_logprob > logprobs[:begin].max():
        row[:begin] = -np.inf
    return row


def next_token(logits: np.ndarray, generated: list[int], behavior: dict) -> int:
    """The greedy choice after suppression and the timestamp rules (transformers' processor order)."""
    return int(np.argmax(timestamp_rules(suppressed(logits, behavior, not generated), generated, behavior)))


def split_segments(tokens: list[int], seek: int, seek_frames: int, behavior: dict) -> tuple[list[dict], int]:
    """transformers 5.18.0 WhisperGenerationMixin._retrieve_segment for one window (no token timestamps).

    `tokens` are the window's generated tokens without end of text. Returns the segments with absolute start and end
    seconds (float64, computed in transformers' order) and the number of mel frames to advance.
    """
    timing = behavior["timestamps"]
    begin, precision, stride = timing["timestamp_begin"], timing["time_precision"], timing["input_stride"]
    offset = seek * precision / stride
    flags = [token >= begin for token in tokens]
    single_ending = flags[-2:] == [False, True]
    slices = [index + 1 for index in range(len(tokens) - 1) if flags[index] and flags[index + 1]]
    if slices:
        segments = []
        if single_ending:
            slices.append(len(tokens))
        else:
            slices[-1] += 1
        last_slice = 0
        for position, current in enumerate(slices):
            piece = tokens[last_slice:current]
            end_index = -1 if position != len(slices) - 1 or single_ending else -2
            segments.append(
                {
                    "start": offset + float(piece[0] - begin) * precision,
                    "end": offset + float(piece[end_index] - begin) * precision,
                    "ids": piece,
                }
            )
            last_slice = current
        advance = seek_frames if single_ending else (tokens[last_slice - 2] - begin) * stride
        return segments, advance
    # transformers multiplies the frame count (a long tensor) by Python floats, which torch evaluates in float32.
    last_position = int(np.float32(seek_frames) * np.float32(timing["feature_precision"]) / np.float32(precision))
    stamps = [token for token in tokens if token >= begin]
    end = offset + last_position * precision
    if stamps and stamps[-1] != begin:
        end = offset + float(stamps[-1] - begin) * precision
    return [{"start": offset, "end": end, "ids": list(tokens)}], seek_frames


def window(mel: np.ndarray, seek: int, frames: int, num_frames: int) -> np.ndarray:
    """transformers' _get_input_segment: the mel frames [seek, seek + frames), zero-padded to one 30 s window."""
    piece = np.zeros((mel.shape[0], num_frames), dtype=np.float32)
    piece[:, :frames] = mel[:, seek : seek + frames]
    return piece


class Tensors:
    """One little-endian float32 sidecar holding every named per-case tensor, with its layout."""

    def __init__(self, path: Path) -> None:
        self.path, self.layout = path, []
        self.handle = path.open("wb")

    def add(self, identifier: str, name: str, values: np.ndarray) -> None:
        values = np.ascontiguousarray(values, dtype="<f4")
        if not np.isfinite(values).all():
            raise ValueError(f"{identifier}/{name}: non-finite values")
        self.layout.append(
            {"id": identifier, "name": name, "offset": self.handle.tell() // 4, "shape": list(values.shape)}
        )
        self.handle.write(values.tobytes())

    def close(self) -> dict:
        self.handle.close()
        return {"file": self.path.name, "sha256": fixtures.sha256_file(self.path), "cases": self.layout}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("reference", "onnx"), required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.threads <= 64:
        parser.error("threads must be 1 through 64")
    candidate = json.loads(args.profile.read_text(encoding="utf-8"))
    validate_candidate_behavior(candidate)
    behavior = candidate["behavior"]
    eos = behavior["special_tokens"]["end_of_text"]
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True, exist_ok=True)
    profile_hash = fixtures.sha256_file(args.profile)
    corpus = load_corpus(args.corpus, candidate)
    tensors = Tensors(args.out / f"{args.stage}-tensors.f32")
    cases = []
    if args.stage == "reference":
        from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration, WhisperTokenizerFast
        from transformers.modeling_outputs import BaseModelOutput

        extractor = WhisperFeatureExtractor.from_pretrained(str(args.snapshot))
        tokenizer = WhisperTokenizerFast.from_pretrained(str(args.snapshot))
        model = WhisperForConditionalGeneration.from_pretrained(
            str(args.snapshot), dtype=torch.float32, attn_implementation="eager"
        ).eval()

        options = {"task": "transcribe", "return_timestamps": False, "do_sample": False, "num_beams": 1}
        # Sequential timestamped decoding, pinned by the contract's long_form block: greedy, no temperature fallback,
        # no conditioning on earlier windows, and no compression-ratio, log-probability or no-speech threshold.
        pinned = {key: value for key, value in behavior["long_form"].items() if key not in ("strategy", "features")}
        timestamp_options = {
            "task": "transcribe",
            "return_timestamps": True,
            "return_segments": True,
            "do_sample": False,
            "num_beams": 1,
            **pinned,
        }
        num_frames = behavior["num_frames"]

        def timestamped(features: np.ndarray, language: str) -> dict:
            """generate(return_timestamps=True) on the mel, replayed through this file's port of the rules and split."""
            output = model.generate(torch.from_numpy(features[None]), language=language, **timestamp_options)
            produced = output["segments"][0]
            windows = []
            for segment in produced:
                if not windows or segment["result"] is not windows[-1]:
                    windows.append(segment["result"])
            total = features.shape[1]
            seek, records, segments = 0, [], []
            for result in windows:
                ids = result.tolist() if isinstance(result, torch.Tensor) else result["sequences"].tolist()
                frames = min(total - seek, num_frames)
                hidden = model.model.encoder(torch.from_numpy(window(features, seek, frames, num_frames)[None]))
                logits = model(encoder_outputs=(hidden.last_hidden_state,), decoder_input_ids=torch.tensor([ids[:-1]]))
                logits = logits.logits[0].numpy()
                begin_index = len(timestamp_prompt)
                replay = []
                for position in range(begin_index - 1, len(ids) - 1):
                    replay.append(next_token(logits[position], ids[begin_index : position + 1], behavior))
                if ids[:begin_index] != timestamp_prompt or replay != ids[begin_index:]:
                    raise ValueError(f"window at frame {seek}: generate() ids are not the replayed greedy path")
                tokens = ids[begin_index:-1] if ids[-1] == eos else ids[begin_index:]
                pieces, advance = split_segments(tokens, seek, frames, behavior)
                records.append({"seek": seek, "ids": ids})
                segments += pieces
                seek += advance
            if seek < total or len(segments) != len(produced):
                raise ValueError("the replayed seek loop does not cover the windows generate() decoded")
            for mine, theirs in zip(segments, produced, strict=True):
                if (mine["start"], mine["end"], mine["ids"]) != (
                    float(theirs["start"]),
                    float(theirs["end"]),
                    theirs["tokens"].tolist(),
                ):
                    raise ValueError(f"segment {mine} differs from generate()'s {theirs['start']}..{theirs['end']}")
                mine["text"] = tokenizer.decode(mine["ids"], skip_special_tokens=True)
            sequence = output["sequences"][0].tolist()
            if sequence != [token for segment in segments for token in segment["ids"]]:
                raise ValueError("generate() sequences are not the concatenated segments")
            return {
                "windows": records,
                "segments": segments,
                "transcript": tokenizer.decode(sequence, skip_special_tokens=True),
            }

        def transcribe(features: np.ndarray, language: str) -> tuple[torch.Tensor, list[int]]:
            """The recorded encoder output, and generate() decoding from it instead of running the encoder again."""
            hidden = model.model.encoder(torch.from_numpy(features[None])).last_hidden_state
            # transformers 5 returns only the generated text tokens: no decoder prompt, no end of text.
            generated = model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=hidden), language=language, **options
            )[0].tolist()
            return hidden, generated

        def transcribe_default(features: np.ndarray, language: str) -> list[int]:
            """Plain generate(input_features), the call users make: its own encoder pass stays an independent check."""
            return model.generate(torch.from_numpy(features[None]), language=language, **options)[0].tolist()

        def long_form_case(row: dict, audio: np.ndarray) -> dict:
            """Long-form: one spectrogram over the whole clip, decoded with timestamps window after window."""
            numpy_mel = extractor._np_extract_fbank_features(audio[None], "cpu")[0].astype(np.float32)
            torch_mel = extractor._torch_extract_fbank_features(audio[None], "cpu")[0].astype(np.float32)
            # The call transformers documents for long-form input: no truncation, no padding beyond the clip.
            default_mel = extractor(
                audio, sampling_rate=behavior["sample_rate"], truncation=False, padding="longest", return_tensors="np"
            ).input_features[0]
            if not np.array_equal(default_mel, torch_mel):
                raise ValueError(f"{row['id']}: the default feature path is not the torch path this runner records")
            if numpy_mel.shape != (behavior["num_mel_bins"], len(audio) // behavior["hop_length"]):
                raise ValueError(f"{row['id']}: unexpected long-form spectrogram shape {numpy_mel.shape}")
            with torch.inference_mode():
                stamped = timestamped(numpy_mel, row["language"])
                default = model.generate(
                    torch.from_numpy(default_mel[None]), language=row["language"], **timestamp_options
                )
            tensors.add(row["id"], "mel", numpy_mel)
            tensors.add(row["id"], "mel_torch", torch_mel)
            return {
                "id": row["id"],
                "language": row["language"],
                "audio_sha256": row["sha256"],
                "samples": row["samples"],
                "text": row["text"],
                "long_form": True,
                "timestamped": stamped,
                "default_path_ids": default["sequences"][0].tolist(),
            }

        for row in corpus:
            audio = read_audio(row["path"], behavior["sample_rate"])
            timestamp_prompt = prompt(candidate, row["language"], timestamps=True)
            if len(audio) > behavior["max_samples"]:
                cases.append(long_form_case(row, audio))
                continue
            padded = np.zeros(behavior["max_samples"], dtype=np.float32)
            padded[: len(audio)] = audio
            default_mel = extractor(audio, sampling_rate=behavior["sample_rate"], return_tensors="np").input_features[0]
            numpy_mel = extractor._np_extract_fbank_features(padded[None], "cpu")[0].astype(np.float32)
            torch_mel = extractor._torch_extract_fbank_features(padded[None], "cpu")[0].astype(np.float32)
            if not np.array_equal(default_mel, torch_mel):
                raise ValueError(f"{row['id']}: the default feature path is not the torch path this runner records")
            expected_prompt = prompt(candidate, row["language"])
            with torch.inference_mode():
                stamped = timestamped(numpy_mel, row["language"])
                hidden, generated = transcribe(numpy_mel, row["language"])
                default_generated = transcribe_default(default_mel, row["language"])
                ids = expected_prompt + generated + [eos]
                if len(ids) > behavior["max_target_positions"]:
                    raise ValueError(f"{row['id']}: generation reached the maximum length without end of text")
                logits = model(encoder_outputs=(hidden,), decoder_input_ids=torch.tensor([ids[:-1]])).logits[0].numpy()
            # The reference ids must be this model's own greedy path, ending at end of text.
            start = len(expected_prompt) - 1
            greedy = [
                int(np.argmax(suppressed(logits[position], behavior, position == start)))
                for position in range(start, len(ids) - 1)
            ]
            if greedy != ids[len(expected_prompt) :]:
                raise ValueError(f"{row['id']}: generate() ids are not the greedy path of the teacher-forced logits")
            tensors.add(row["id"], "mel", numpy_mel)
            tensors.add(row["id"], "mel_torch", torch_mel)
            tensors.add(row["id"], "hidden", hidden[0].numpy())
            tensors.add(row["id"], "logits", logits)
            cases.append(
                {
                    "id": row["id"],
                    "language": row["language"],
                    "audio_sha256": row["sha256"],
                    "samples": row["samples"],
                    "text": row["text"],
                    "ids": ids,
                    "transcript": tokenizer.decode(ids, skip_special_tokens=True),
                    "default_path_ids": expected_prompt + default_generated + [eos],
                    "long_form": False,
                    "timestamped": stamped,
                }
            )
        assets = ("model.safetensors", "preprocessor_config.json", "tokenizer.json")
    else:
        from tokenizers import Tokenizer

        if args.reference is None:
            parser.error("onnx requires --reference")
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        if reference["candidate_sha256"] != profile_hash or reference["profile"] != candidate["id"]:
            raise ValueError("reference identity differs from the selected candidate")
        reference_cases = {row["id"]: row for row in reference["cases"]}
        blob = np.fromfile(args.reference.parent / reference["tensors"]["file"], dtype="<f4")
        reference_mel = {
            entry["id"]: blob[entry["offset"] : entry["offset"] + int(np.prod(entry["shape"]))].reshape(entry["shape"])
            for entry in reference["tensors"]["cases"]
            if entry["name"] == "mel"
        }
        tokenizer = Tokenizer.from_file(str(args.snapshot / behavior["tokenizer_file"]))
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        encoder = ort.InferenceSession(
            str(args.snapshot / behavior["encoder_file"]), options, providers=["CPUExecutionProvider"]
        )
        decoder = ort.InferenceSession(
            str(args.snapshot / behavior["decoder_file"]), options, providers=["CPUExecutionProvider"]
        )
        if [value.name for value in encoder.get_inputs()] != ["input_features"] or [
            value.name for value in decoder.get_inputs()
        ] != ["input_ids", "encoder_hidden_states"]:
            raise ValueError("unsupported ONNX graph input signature")

        def decode(ids: list[int], hidden: np.ndarray) -> np.ndarray:
            feeds = {"input_ids": np.array([ids], dtype=np.int64), "encoder_hidden_states": hidden}
            return decoder.run(["logits"], feeds)[0][0]

        begin = behavior["timestamps"]["timestamp_begin"]

        def text(ids: list[int]) -> str:
            """Timestamp tokens are not special in tokenizer.json: drop them, as WhisperTokenizer does, then decode."""
            return tokenizer.decode([token for token in ids if token < begin], skip_special_tokens=True)

        def timestamped(mel: np.ndarray, language: str) -> dict:
            """The sequential seek loop: decode a window with the timestamp rules, split it, advance, repeat."""
            start = prompt(candidate, language, timestamps=True)
            num_frames, total = behavior["num_frames"], mel.shape[1]
            seek, records, segments = 0, [], []
            while seek < total:
                frames = min(total - seek, num_frames)
                hidden = encoder.run(
                    ["last_hidden_state"], {"input_features": window(mel, seek, frames, num_frames)[None]}
                )[0]
                ids = list(start)
                while len(ids) < behavior["max_target_positions"]:
                    ids.append(next_token(decode(ids, hidden)[-1], ids[len(start) :], behavior))
                    if ids[-1] == eos:
                        break
                tokens = ids[len(start) : -1] if ids[-1] == eos else ids[len(start) :]
                pieces, advance = split_segments(tokens, seek, frames, behavior)
                records.append({"seek": seek, "ids": ids})
                segments += [{**piece, "text": text(piece["ids"])} for piece in pieces]
                seek += advance
            sequence = [token for segment in segments for token in segment["ids"]]
            return {"windows": records, "segments": segments, "transcript": text(sequence)}

        for row in corpus:
            source = reference_cases[row["id"]]
            if source["audio_sha256"] != row["sha256"]:
                raise ValueError(f"{row['id']}: reference audio differs from the corpus")
            if source["long_form"]:
                cases.append(
                    {
                        "id": row["id"],
                        "language": row["language"],
                        "audio_sha256": row["sha256"],
                        "long_form": True,
                        "timestamped": timestamped(reference_mel[row["id"]], row["language"]),
                    }
                )
                continue
            hidden = encoder.run(["last_hidden_state"], {"input_features": reference_mel[row["id"]][None]})[0]
            ids = prompt(candidate, row["language"])
            first = True
            while len(ids) < behavior["max_target_positions"]:
                ids.append(int(np.argmax(suppressed(decode(ids, hidden)[-1], behavior, first))))
                first = False
                if ids[-1] == eos:
                    break
            tensors.add(row["id"], "hidden", hidden[0])
            tensors.add(row["id"], "logits", decode(source["ids"][:-1], hidden))
            stamped = timestamped(reference_mel[row["id"]], row["language"])
            cases.append(
                {
                    "id": row["id"],
                    "language": row["language"],
                    "audio_sha256": row["sha256"],
                    "ids": ids,
                    "forced_ids": source["ids"],
                    "transcript": tokenizer.decode(ids, skip_special_tokens=True),
                    "long_form": False,
                    "timestamped": stamped,
                }
            )
        assets = (behavior["encoder_file"], behavior["decoder_file"], behavior["tokenizer_file"])
    document = {
        "format": FORMAT,
        "format_version": 1,
        "profile": candidate["id"],
        "task": candidate["task"],
        "stage": args.stage,
        "candidate_sha256": profile_hash,
        "reference_status": "proposed",
        "assets": [{"path": name, "sha256": fixtures.sha256_file(args.snapshot / name)} for name in assets],
        "environment": {
            **fixtures.environment(["torch", "transformers", "tokenizers", "onnxruntime", "numpy"]),
            "machine": platform.node(),
            "provider": "CPUExecutionProvider",
            "threads": args.threads,
            "onnxruntime": ort.__version__,
        },
        "cases": cases,
        "tensors": tensors.close(),
    }
    if args.reference is not None:
        document["reference_sha256"] = fixtures.sha256_file(args.reference)
    fixtures.write_json(args.out / f"{args.stage}.json", document)
    print(f"{candidate['id']}: {args.stage}, {len(cases)} clips, original references remain proposed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
