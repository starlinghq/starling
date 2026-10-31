"""Write the ASR reference corpora: pinned LibriSpeech clips as 16 kHz mono int16 WAV plus JSONL indexes.

The clips come from hf-internal-testing/librispeech_asr_dummy (LibriSpeech dev-clean, CC BY 4.0) at a pinned
revision. The parquet is downloaded by the caller and verified here by SHA-256; the FLAC payloads are decoded to their
int16 samples unchanged (no resampling), so the WAV holds the exact LibriSpeech samples. The ground-truth text is
LibriSpeech's own transcript, recorded only for a corpus-level sanity bound; parity is judged against the fresh
reference, never against this text.

asr-corpus-v1.jsonl holds the four single utterances (asr-v1/). asr-corpus-v2.jsonl holds those four plus composite
clips (asr-v2/): consecutive utterances of one chapter joined by GAP_SAMPLES of digital silence, their ground truth the
utterance transcripts joined by spaces. A composite clip gives the timestamped decode several segments to emit.
asr-corpus-v3.jsonl holds v2 plus composites longer than 30 s (asr-v3/) for sequential long-form transcription.

Run once, outside the reference lock (pyarrow and soundfile are only needed here):

    uv run --no-project --with pyarrow==21.0.0 --with soundfile==0.13.1 --with numpy==2.4.6 python -I \\
        model-pipeline/reference/whisper/make_asr_corpus.py --parquet <validation-00000-of-00001.parquet> \\
        --out model-pipeline/reference/corpus
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import wave
from pathlib import Path

import numpy
import pyarrow.parquet as pq
import soundfile

DATASET = "hf-internal-testing/librispeech_asr_dummy"
REVISION = "5be91486e11a2d616f4ec5db8d3fd248585ac07a"
FILE = "clean/validation-00000-of-00001.parquet"
FILE_SHA256 = "4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603"
# Differently sized clips: a typical sentence, a very short one, a long one, and a number the model writes as digits.
UTTERANCES = ("1272-128104-0000", "1272-135031-0009", "1272-128104-0002", "1272-141231-0013")
SAMPLE_RATE = 16000
# Composite clips: consecutive utterances of one chapter. v2's are at most 30 s; v3's are longer (long-form).
COMPOSITES = {
    "v2": {
        "librispeech-1272-135031-0003-0008": tuple(f"1272-135031-{index:04d}" for index in range(3, 9)),
        # Cut at exactly 30 s, inside the third utterance: speech runs into the window's end, so the decode ends inside
        # a segment and transformers decodes again from the last closing timestamp (a second, shorter window).
        "librispeech-1272-128104-0009-0011-cut30": tuple(f"1272-128104-{index:04d}" for index in range(9, 12)),
    },
    "v3": {
        "librispeech-1272-135031-0000-0002": tuple(f"1272-135031-{index:04d}" for index in range(0, 3)),
        "librispeech-1272-141231-0000-0008": tuple(f"1272-141231-{index:04d}" for index in range(0, 9)),
    },
}
# Composites cut to this many samples; their text is the full utterances' (more than is spoken), so the route leaves
# them out of the ground-truth sanity bound.
TRUNCATED = {"librispeech-1272-128104-0009-0011-cut30": 30 * SAMPLE_RATE}
GAP_SAMPLES = SAMPLE_RATE // 2
MAX_SAMPLES = 30 * SAMPLE_RATE


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode(rows: dict, utterance: str):
    samples, rate = soundfile.read(io.BytesIO(rows[utterance]["audio"]["bytes"]), dtype="int16")
    if rate != SAMPLE_RATE or samples.ndim != 1:
        raise SystemExit(f"{utterance}: expected 16 kHz mono, found {rate} Hz with shape {samples.shape}")
    return samples


def wav(samples) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(samples.astype("<i2").tobytes())
    return buffer.getvalue()


def write_index(path: Path, index: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in index:
            handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.parquet.read_bytes()) != FILE_SHA256:
        raise SystemExit(f"{args.parquet}: SHA-256 differs from the pinned {DATASET}@{REVISION} {FILE}")
    rows = {row["id"]: row for row in pq.read_table(args.parquet).to_pylist()}
    source = {"dataset": DATASET, "revision": REVISION, "file": FILE}
    audio_dir = args.out / "asr-v1"
    audio_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for utterance in UTTERANCES:
        samples = decode(rows, utterance)
        data = wav(samples)
        name = f"librispeech-{utterance}.wav"
        (audio_dir / name).write_bytes(data)
        index.append(
            {
                "id": f"librispeech-{utterance}",
                "audio": f"asr-v1/{name}",
                "sha256": sha256(data),
                "sample_rate": SAMPLE_RATE,
                "samples": len(samples),
                "language": "en",
                "text": rows[utterance]["text"],
                "source": {**source, "utterance": utterance},
                "license": "CC-BY-4.0",
            }
        )
    write_index(args.out / "asr-corpus-v1.jsonl", index)
    for version, composites in COMPOSITES.items():
        composite_dir = args.out / f"asr-{version}"
        composite_dir.mkdir(parents=True, exist_ok=True)
        for identifier, utterances in composites.items():
            gap = numpy.zeros(GAP_SAMPLES, dtype=numpy.int16)
            parts = [decode(rows, utterance) for utterance in utterances]
            samples = numpy.concatenate([piece for part in parts for piece in (part, gap)][:-1])
            if identifier in TRUNCATED:
                samples = samples[: TRUNCATED[identifier]]
            if (len(samples) > MAX_SAMPLES) != (version != "v2"):
                raise SystemExit(
                    f"{identifier}: {len(samples)} samples, but {version} composites are on the other side of 30 s"
                )
            data = wav(samples)
            (composite_dir / f"{identifier}.wav").write_bytes(data)
            index.append(
                {
                    "id": identifier,
                    "audio": f"asr-{version}/{identifier}.wav",
                    "sha256": sha256(data),
                    "sample_rate": SAMPLE_RATE,
                    "samples": len(samples),
                    "language": "en",
                    "text": " ".join(rows[utterance]["text"] for utterance in utterances),
                    "source": {
                        **source,
                        "utterances": list(utterances),
                        "gap_samples": GAP_SAMPLES,
                        **({"truncated_to_samples": TRUNCATED[identifier]} if identifier in TRUNCATED else {}),
                    },
                    "license": "CC-BY-4.0",
                }
            )
        write_index(args.out / f"asr-corpus-{version}.jsonl", index)
    print(f"wrote {len(index)} clips to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
