"""Evidence serialization for the proposed text-only Qwen CPU profiles."""

from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def corpus_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    ids = [row["id"] for row in rows]
    if not rows or len(ids) != len(set(ids)) or any(not isinstance(name, str) or not name for name in ids):
        raise ValueError("corpus requires nonempty unique string ids")
    for row in rows:
        if row.get("enable_thinking", False) is not False:
            raise ValueError("profile explicitly scopes Qwen3 enable_thinking=false")
        if not row.get("messages") or any(
            message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), str)
            or set(message) - {"role", "content", "reasoning_content"}
            for message in row["messages"]
        ):
            raise ValueError("only nonempty text conversations are supported; tools and multimodal input unsupported")
    return rows


def logits_record(out: Path, index: int, values, vocabulary: int) -> dict:
    raw = values.astype("<f4").tobytes()
    if len(raw) != vocabulary * 4:
        raise ValueError("wrong initial logits shape")
    name = f"logits-{index:04d}.f32"
    (out / name).write_bytes(raw)
    return {"path": name, "dtype": "float32-le", "shape": [1, vocabulary], "sha256": hashlib.sha256(raw).hexdigest()}


def row_record(
    candidate: dict,
    record: dict,
    backend: str,
    text: str,
    prompt: list[int],
    tokens: list[int],
    stop: int | None,
    output_text: str,
    logits: dict,
    checks: dict,
) -> dict:
    return {
        "id": record["id"],
        "profile": candidate["id"],
        "revision": candidate["origin"]["revision"],
        "backend": backend,
        "runtime": {
            "backend": backend,
            "version": metadata.version("transformers" if backend == "transformers" else "onnxruntime-genai"),
            "precision": "fp32",
            "provider": "CPUExecutionProvider",
        },
        "prompt_text": text,
        "prompt_ids": prompt,
        "output_ids": tokens,
        "output_text": output_text,
        "finish_reason": "stop" if stop is not None else "length",
        "stop_token_id": stop,
        "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(tokens)},
        "initial_logits": logits,
        "checks": checks,
    }
