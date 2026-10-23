"""Pinned Whisper reference (transformers, PyTorch fp32) or ONNX Runtime execution, into proposed artifacts only.

reference: for each corpus clip, the transformers WhisperFeatureExtractor log-mel spectrogram (both its numpy path,
           which Starlings.Models.Whisper restates, and its default torch path), the encoder last_hidden_state, the
           greedy transcript from WhisperForConditionalGeneration.generate, and the decoder logits teacher-forced on
           those ids. The numpy-path mel is the input every later stage uses.
onnx:      the pinned fp16 graphs on ONNX Runtime from the reference mel: encoder hidden state, an independent greedy
           loop with the generation config's suppression lists, and logits teacher-forced on the reference ids.
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

        extractor = WhisperFeatureExtractor.from_pretrained(str(args.snapshot))
        tokenizer = WhisperTokenizerFast.from_pretrained(str(args.snapshot))
        model = WhisperForConditionalGeneration.from_pretrained(
            str(args.snapshot), dtype=torch.float32, attn_implementation="eager"
        ).eval()

        def transcribe(features: np.ndarray, language: str) -> tuple[torch.Tensor, list[int]]:
            features = torch.from_numpy(features[None])
            hidden = model.model.encoder(features).last_hidden_state
            # transformers 5 returns only the generated text tokens: no decoder prompt, no end of text.
            generated = model.generate(
                features, language=language, task="transcribe", return_timestamps=False, do_sample=False, num_beams=1
            )[0].tolist()
            return hidden, generated

        for row in corpus:
            audio = read_audio(row["path"], behavior["sample_rate"])
            padded = np.zeros(behavior["max_samples"], dtype=np.float32)
            padded[: len(audio)] = audio
            default_mel = extractor(audio, sampling_rate=behavior["sample_rate"], return_tensors="np").input_features[0]
            numpy_mel = extractor._np_extract_fbank_features(padded[None], "cpu")[0].astype(np.float32)
            torch_mel = extractor._torch_extract_fbank_features(padded[None], "cpu")[0].astype(np.float32)
            if not np.array_equal(default_mel, torch_mel):
                raise ValueError(f"{row['id']}: the default feature path is not the torch path this runner records")
            expected_prompt = prompt(candidate, row["language"])
            with torch.inference_mode():
                hidden, generated = transcribe(numpy_mel, row["language"])
                _, default_generated = transcribe(default_mel, row["language"])
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

        for row in corpus:
            source = reference_cases[row["id"]]
            if source["audio_sha256"] != row["sha256"]:
                raise ValueError(f"{row['id']}: reference audio differs from the corpus")
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
            cases.append(
                {
                    "id": row["id"],
                    "language": row["language"],
                    "audio_sha256": row["sha256"],
                    "ids": ids,
                    "forced_ids": source["ids"],
                    "transcript": tokenizer.decode(ids, skip_special_tokens=True),
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
