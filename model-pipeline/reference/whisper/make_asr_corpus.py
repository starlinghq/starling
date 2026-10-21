"""Write the ASR reference corpus: pinned LibriSpeech clips as 16 kHz mono int16 WAV plus a JSONL index.

The clips come from hf-internal-testing/librispeech_asr_dummy (LibriSpeech dev-clean, CC BY 4.0) at a pinned
revision. The parquet is downloaded by the caller and verified here by SHA-256; the FLAC payloads are decoded to their
int16 samples unchanged (no resampling), so the WAV holds the exact LibriSpeech samples. The ground-truth text is
LibriSpeech's own transcript, recorded only for a corpus-level sanity bound; parity is judged against the fresh
reference, never against this text.

Run once, outside the reference lock (pyarrow and soundfile are only needed here):

    uv run --no-project --with pyarrow==21.0.0 --with soundfile==0.13.1 python -I \\
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

import pyarrow.parquet as pq
import soundfile

DATASET = "hf-internal-testing/librispeech_asr_dummy"
REVISION = "5be91486e11a2d616f4ec5db8d3fd248585ac07a"
FILE = "clean/validation-00000-of-00001.parquet"
FILE_SHA256 = "4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603"
# Differently sized clips: a typical sentence, a very short one, a long one, and a number the model writes as digits.
UTTERANCES = ("1272-128104-0000", "1272-135031-0009", "1272-128104-0002", "1272-141231-0013")
SAMPLE_RATE = 16000


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.parquet.read_bytes()) != FILE_SHA256:
        raise SystemExit(f"{args.parquet}: SHA-256 differs from the pinned {DATASET}@{REVISION} {FILE}")
    rows = {row["id"]: row for row in pq.read_table(args.parquet).to_pylist()}
    audio_dir = args.out / "asr-v1"
    audio_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for utterance in UTTERANCES:
        row = rows[utterance]
        samples, rate = soundfile.read(io.BytesIO(row["audio"]["bytes"]), dtype="int16")
        if rate != SAMPLE_RATE or samples.ndim != 1:
            raise SystemExit(f"{utterance}: expected 16 kHz mono, found {rate} Hz with shape {samples.shape}")
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(SAMPLE_RATE)
            writer.writeframes(samples.astype("<i2").tobytes())
        data = buffer.getvalue()
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
                "text": row["text"],
                "source": {"dataset": DATASET, "revision": REVISION, "file": FILE, "utterance": utterance},
                "license": "CC-BY-4.0",
            }
        )
    with (args.out / "asr-corpus-v1.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for entry in index:
            handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
    print(f"wrote {len(index)} clips to {audio_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
