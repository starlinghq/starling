"""Strict development parity for proposed Qwen generation evidence.

This evaluator cannot adopt fixtures or award protected G1/G2 qualification.
All logits, identities, corpus rows and runtime semantic checks are required.
"""

from __future__ import annotations

import array
import hashlib
import json
import math
import sys
from pathlib import Path, PurePosixPath

from model_pipeline.onboarding_runner import PipelineError

POLICY = {"logits_max_abs": 0.005, "logits_mean_abs": 0.0005, "greedy_ids": "exact", "prompt_ids": "exact"}
TRANSFORMERS_VERSION = "5.18.0"
CHECKS = {
    "transformers": {"cache"},
    "python-genai": {"cache", "isolation"},
    "dotnet": {"cache", "isolation", "stream", "usage", "cancellation", "token_stop", "string_stop"},
}


def read_rows(path: Path) -> dict[str, dict]:
    """JSON lines keyed by a unique, non-empty string `id`; an empty file is an error.

    Shared with the protected generation task (`evaluate_generation.py`), for fixtures and runner results alike.
    """
    result = {}
    try:
        with path.open(encoding="utf-8") as stream:
            lines = [line for line in stream if line.strip()]
        for line in lines:
            row = json.loads(line)
            name = row["id"]
            if not isinstance(name, str) or not name or name in result:
                raise PipelineError(f"missing or duplicate evidence id in {path}: {name!r}")
            result[name] = row
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise PipelineError(f"invalid generation evidence {path}: {exc}") from exc
    if not result:
        raise PipelineError(f"empty generation evidence: {path}")
    return result


_rows = read_rows


def valid_ids(value: object, vocabulary: int, *, empty: bool = True) -> bool:
    """A list of token ids in [0, vocabulary) (non-empty unless `empty`)."""
    return (
        isinstance(value, list)
        and (empty or bool(value))
        and all(type(token) is int and 0 <= token < vocabulary for token in value)
    )


_ids = valid_ids


def read_float32(directory: Path, relative: str, count: int, sha256: str | None = None) -> array.array:
    """Read `count` little-endian float32 values from `directory/relative`, confined to `directory`.

    With `sha256`, the file's digest must equal it. Non-finite values are returned, not rejected: callers decide.
    """
    path_text = PurePosixPath(relative)
    if path_text.is_absolute() or ".." in path_text.parts or "\\" in relative:
        raise PipelineError(f"unsafe tensor path {relative!r}")
    path = directory / path_text
    if not path.resolve().is_relative_to(directory.resolve()):
        raise PipelineError(f"tensor path {relative!r} escapes {directory}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PipelineError(f"cannot read tensor {path}: {exc}") from exc
    if len(raw) != count * 4:
        raise PipelineError(f"{path}: {len(raw)} bytes, expected {count} float32 values")
    if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256:
        raise PipelineError(f"{path}: SHA-256 differs from the recorded digest")
    values = array.array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def _logits(directory: Path, row: dict, vocabulary: int) -> array.array:
    try:
        tensor = row["initial_logits"]
        if (
            tensor["dtype"] != "float32-le"
            or tensor["shape"] != [1, vocabulary]
            or any(type(dimension) is not int for dimension in tensor["shape"])
        ):
            raise PipelineError("wrong logits dtype/shape")
        try:
            values = read_float32(directory, tensor["path"], vocabulary, tensor["sha256"])
        except PipelineError as exc:
            raise PipelineError(f"logits size/hash mismatch or unsafe path: {exc}") from exc
        if not all(math.isfinite(v) for v in values):
            raise PipelineError("non-finite initial logits")
        return values
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError(f"invalid logits evidence: {exc}") from exc


def evaluate(candidate: dict, corpus: Path, reference: Path, backend: Path, dotnet: Path) -> dict:
    """Return a report; structural errors raise, numerical/discrete divergence fails."""
    declared = candidate["evaluation"]["policy"]
    if any(declared.get(key) != value for key, value in POLICY.items()):
        raise PipelineError("generation candidate must use the predeclared development policy")
    expected = _rows(corpus)
    sources = {"transformers": reference, "python-genai": backend, "dotnet": dotnet}
    vocabulary = candidate["behavior"]["vocab_size"]
    eos = candidate["behavior"]["eos_token_ids"]
    failures = []
    evidence = {}
    for source, directory in sources.items():
        rows = _rows(directory / "results.jsonl")
        if set(rows) != set(expected):
            raise PipelineError(f"{source}: corpus/evidence row identity mismatch")
        evidence[source] = rows
        tensors = set()
        for name, row in rows.items():
            if (
                row.get("profile") != candidate["id"]
                or row.get("revision") != candidate["origin"]["revision"]
                or row.get("backend") != source
                or not isinstance(row.get("prompt_text"), str)
                or not _ids(row.get("prompt_ids"), vocabulary, empty=False)
                or not _ids(row.get("output_ids"), vocabulary)
                or not isinstance(row.get("output_text"), str)
            ):
                raise PipelineError(f"{source}/{name}: wrong identity or missing text/token evidence")
            runtime = row.get("runtime", {})
            if (
                not isinstance(runtime, dict)
                or runtime.get("backend") != source
                or runtime.get("provider") != "CPUExecutionProvider"
                or runtime.get("precision") != "fp32"
                or not isinstance(runtime.get("version"), str)
                or not runtime["version"]
                or (source != "transformers" and runtime["version"] != candidate["runtime"]["version"])
                or (source == "transformers" and runtime["version"] != TRANSFORMERS_VERSION)
            ):
                raise PipelineError(f"{source}/{name}: wrong runtime identity")
            maximum = expected[name].get("max_new_tokens", candidate["behavior"]["max_new_tokens"])
            if type(maximum) is not int or maximum < 1:
                raise PipelineError(f"{name}: max_new_tokens must be positive")
            missing = [key for key in ("finish_reason", "stop_token_id", "usage") if key not in row]
            if missing:
                raise PipelineError(f"{source}/{name}: malformed evidence, missing {missing}")
            finish = row["finish_reason"]
            stop = row["stop_token_id"]
            if not (
                (finish == "length" and stop is None and len(row["output_ids"]) == maximum)
                or (finish == "stop" and type(stop) is int and stop in eos and len(row["output_ids"]) < maximum)
            ):
                raise PipelineError(f"{source}/{name}: incomplete or inconsistent termination evidence")
            usage = row.get("usage")
            if usage != {"prompt_tokens": len(row["prompt_ids"]), "completion_tokens": len(row["output_ids"])}:
                raise PipelineError(f"{source}/{name}: incomplete or inconsistent usage")
            checks = row.get("checks", {})
            if not isinstance(checks, dict) or not CHECKS[source].issubset(checks):
                raise PipelineError(f"{source}/{name}: missing semantic checks {sorted(CHECKS[source])}")
            for check in CHECKS[source]:
                if checks[check] is not True:
                    failures.append(f"{source}/{name}: {check} failed")
            _logits(directory, row, vocabulary)
            tensor_path = row["initial_logits"]["path"]
            if tensor_path in tensors:
                raise PipelineError(f"{source}/{name}: duplicate tensor evidence path")
            tensors.add(tensor_path)
            if source == "dotnet" and "cancellation_between_steps" not in row:
                raise PipelineError(f"{source}/{name}: missing between-step cancellation evidence")
            if (
                source == "dotnet"
                and row["cancellation_between_steps"] is not None
                and row["cancellation_between_steps"] is not True
            ):
                failures.append(f"{source}/{name}: between-step cancellation failed")
    if not any(row["cancellation_between_steps"] is True for row in evidence["dotnet"].values()):
        raise PipelineError("no .NET corpus case exercised cancellation between decode steps")
    comparisons = []
    for name in expected:
        reference_row = evidence["transformers"][name]
        ref_logits = _logits(reference, reference_row, vocabulary)
        for source in ("python-genai", "dotnet"):
            row = evidence[source][name]
            for key in (
                "prompt_text",
                "prompt_ids",
                "output_ids",
                "output_text",
                "finish_reason",
                "stop_token_id",
                "usage",
            ):
                if row[key] != reference_row[key]:
                    failures.append(f"{source}/{name}: {key} differs")
            logits = _logits(sources[source], row, vocabulary)
            diffs = [abs(a - b) for a, b in zip(ref_logits, logits, strict=True)]
            maximum, mean = max(diffs), sum(diffs) / len(diffs)
            comparisons.append({"id": name, "backend": source, "logits_max_abs": maximum, "logits_mean_abs": mean})
            if maximum > declared["logits_max_abs"] or mean > declared["logits_mean_abs"]:
                failures.append(f"{source}/{name}: initial logits exceed predefined policy")
    return {
        "format_version": 1,
        "profile": candidate["id"],
        "status": "failed" if failures else "passed",
        "reference_status": "proposed",
        "protected_release_qualification": "unapproved",
        "policy": declared,
        "rows": len(expected),
        "comparisons": comparisons,
        "failures": failures,
    }
