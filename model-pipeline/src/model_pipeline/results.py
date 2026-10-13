"""Stage results format, versions 1 and 2 (backlog P-007, P-015).

Every runner that produces candidate outputs for the staged fidelity evaluator writes one JSON
file per stage in this format: the owned tokenizer (stage `tokenizer`), Python ONNX Runtime
against the PyTorch reference (stage `export`), and the C# implementation (stage `dotnet`).
The evaluator reads them next to the protected fixtures and tolerances of the profile.

A results file carries outputs only. It never carries thresholds: the evaluator rejects any
file with a tolerance-like key (see `FORBIDDEN_KEYS`), so a runner cannot loosen a bound.

Layout (schema: `schemas/stage-results.schema.json`):

    {
      "format": "starlings.stage-results",
      "format_version": 1,
      "profile": "<family>/<profile>",
      "stage": "tokenizer" | "export" | "dotnet",
      "candidate": "what produced the outputs",
      "producer": {"name": "...", "repository_commit": "..." | null},
      "environment": {"machine": "...", "provider": "cpu", "provider_options": {...}, ...},
      "assets": [{"path": "onnx/model.onnx", "sha256": "..."}],
      "cases": {"<case id>": {"<output>": [numbers], ...}},
      "batches": {"<name>": {"output": "<case output>", "case_ids": [...], "values": [[numbers], ...]}},
      "tensors": {"<output>": {"file": "<name>.f32", "dtype": "float32", "sha256": "...",
                               "cases": [{"id": "...", "shape": [n, d]}]}}
    }

Tensors are too large for JSON (hidden states). They live in a sidecar file next to the
results file: little-endian float32 values of each case, row-major, concatenated in `cases`
order. The evaluator checks the sidecar's SHA-256 and size before reading it.

Version 2 adds a required `task` and typed per-case outputs for extraction tasks. Version 1 files
(numeric outputs only, task implicitly `embedding`) stay valid, and the embedding producers still
write version 1. In version 2 a case output or batch row is one of:

    [numbers]                               logits, ids, positions, character maps
    "string"                                normalized text
    ["strings"]                             words
    {"gate_open": bool,                     an entity result: items in output order (label order,
     "entities": [{"label": ..., "text": ...,  then the implementation's order within a label);
                   "start": n, "end": n,    offsets in the profile's offsets unit, exclusive end
                   "confidence": x}]}

Output names per stage for task `entity-extraction` are in `ENTITY_EXTRACTION_OUTPUTS`;
`evaluate_extraction.py` documents what each is compared with.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

FORMAT = "starlings.stage-results"
FORMAT_VERSION = 1  # written when no task is given (embedding producers), so their files do not change
FORMAT_VERSIONS = (1, 2)
TASKS = ("embedding", "entity-extraction", "image-embedding", "video-embedding")
STAGES = ("tokenizer", "export", "dotnet")

# Per-case outputs of task `entity-extraction`, by stage. Stage 1 is the processor (GLiNER2: normalization,
# word splitting, schema prefix, subword encoding, routing positions); the others run the graph on the fixture
# inputs, and `text_entities` is the .NET end-to-end path from text.
ENTITY_EXTRACTION_OUTPUTS = {
    "tokenizer": (
        "normalized_text",
        "words",
        "start_map",
        "end_map",
        "input_ids",
        "attention_mask",
        "word_index",
        "schema_index",
    ),
    "export": ("input_ids", "word_index", "schema_index", "count_logits", "span_logits", "entities"),
    "dotnet": ("input_ids", "word_index", "schema_index", "count_logits", "span_logits", "entities", "text_entities"),
}
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "stage-results.schema.json"
RESULTS_SUFFIX = ".results.json"

# Keys that would let a results file carry its own thresholds. Rejected anywhere in the file.
FORBIDDEN_KEYS = frozenset({"tolerance", "tolerances", "threshold", "thresholds", "bound", "bounds", "max_abs_bound"})


class ResultsError(ValueError):
    """A results file is malformed, inconsistent, or tries to carry thresholds."""


class ToleranceOverrideError(ResultsError):
    """A results file or caller tried to supply thresholds. Only protected tolerances apply."""


@dataclass
class Tensor:
    """Per-case float32 arrays read from a sidecar, flattened, with their shapes."""

    shapes: dict[str, list[int]]
    values: dict[str, list[float]]


@dataclass
class StageResults:
    path: Path
    stage: str
    profile: str
    candidate: str
    producer: dict[str, Any]
    environment: dict[str, Any]
    assets: list[dict[str, str]]
    cases: dict[str, dict[str, list[float] | list[int]]]
    batches: dict[str, dict[str, Any]]
    tensor_specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    sha256: str = ""
    task: str = "embedding"
    format_version: int = 1

    def tensor(self, name: str) -> Tensor:
        spec = self.tensor_specs[name]
        return read_tensor(self.path.parent, spec)


@cache
def _validator():
    # Imported here: reference runner environments install this package with --no-deps and only write results.
    from jsonschema import Draft202012Validator

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _find_forbidden(value: Any, where: str = "") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            here = f"{where}/{key}"
            if str(key).lower() in FORBIDDEN_KEYS:
                found.append(here)
            found += _find_forbidden(item, here)
    elif isinstance(value, list):
        # Numeric arrays are large and cannot hold keys: descend only into objects and lists of containers.
        for i, item in enumerate(value):
            if isinstance(item, dict) or (isinstance(item, list) and item and isinstance(item[0], dict | list)):
                found += _find_forbidden(item, f"{where}/{i}")
    return found


def load(path: Path) -> StageResults:
    raw_bytes = path.read_bytes()
    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResultsError(f"{path}: not UTF-8 JSON: {exc}") from exc
    forbidden = _find_forbidden(raw)
    if forbidden:
        raise ToleranceOverrideError(
            f"{path}: results files carry outputs only; thresholds come from the profile's protected "
            f"tolerances file. Rejected keys: {', '.join(forbidden)}"
        )
    errors = sorted(_validator().iter_errors(raw), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        loc = "/".join(str(p) for p in first.absolute_path) or "<root>"
        raise ResultsError(f"{path}: {loc}: {first.message}")
    for name, batch in raw.get("batches", {}).items():
        if len(batch["case_ids"]) != len(batch["values"]):
            raise ResultsError(
                f"{path}: batch {name!r}: {len(batch['case_ids'])} case ids, {len(batch['values'])} rows"
            )
    return StageResults(
        path=path,
        stage=raw["stage"],
        profile=raw["profile"],
        candidate=raw["candidate"],
        producer=raw["producer"],
        environment=raw.get("environment", {}),
        assets=raw.get("assets", []),
        cases=raw.get("cases", {}),
        batches=raw.get("batches", {}),
        tensor_specs=raw.get("tensors", {}),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        task=raw.get("task", "embedding"),
        format_version=raw["format_version"],
    )


def load_directory(directory: Path) -> dict[str, StageResults]:
    """Every `*.results.json` in `directory`, keyed by stage. Two files for one stage is an error."""
    if not directory.is_dir():
        raise ResultsError(f"{directory}: not a directory")
    out: dict[str, StageResults] = {}
    for path in sorted(directory.glob(f"*{RESULTS_SUFFIX}")):
        result = load(path)
        if result.stage in out:
            raise ResultsError(f"two results files for stage {result.stage!r}: {out[result.stage].path}, {path}")
        out[result.stage] = result
    return out


def _sidecar_path(directory: Path, name: str) -> Path:
    if "/" in name or "\\" in name or name.startswith("."):
        raise ResultsError(f"tensor file {name!r} must be a plain file name next to the results file")
    return directory / name


def read_tensor(directory: Path, spec: dict[str, Any]) -> Tensor:
    path = _sidecar_path(directory, spec["file"])
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != spec["sha256"]:
        raise ResultsError(f"{path}: SHA-256 {digest} does not match the results file ({spec['sha256']})")
    shapes, values, pos = {}, {}, 0
    for case in spec["cases"]:
        count = math.prod(case["shape"])
        end = pos + 4 * count
        if end > len(data):
            raise ResultsError(f"{path}: truncated at case {case['id']!r}")
        values[case["id"]] = list(struct.unpack_from(f"<{count}f", data, pos))
        shapes[case["id"]] = list(case["shape"])
        pos = end
    if pos != len(data):
        raise ResultsError(f"{path}: {len(data) - pos} trailing bytes")
    return Tensor(shapes=shapes, values=values)


def write_tensor(directory: Path, file_name: str, arrays: Iterable[tuple[str, Sequence[int], Iterable[float]]]) -> dict:
    """Write per-case float32 arrays to a sidecar; returns the `tensors` entry for the results file."""
    path = _sidecar_path(directory, file_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    cases = []
    with path.open("wb") as handle:
        for case_id, shape, flat in arrays:
            values = list(flat)
            if len(values) != math.prod(shape):
                raise ValueError(f"{case_id}: {len(values)} values for shape {list(shape)}")
            chunk = struct.pack(f"<{len(values)}f", *values)
            handle.write(chunk)
            digest.update(chunk)
            cases.append({"id": case_id, "shape": [int(s) for s in shape]})
    return {"file": file_name, "dtype": "float32", "sha256": digest.hexdigest(), "cases": cases}


def new_results(
    *,
    profile: str,
    stage: str,
    candidate: str,
    producer: dict[str, Any],
    environment: dict[str, Any],
    assets: list[dict[str, str]],
    task: str | None = None,
) -> dict[str, Any]:
    """A results skeleton: version 1 without `task` (embedding), version 2 with one."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    if task is not None and task not in TASKS:
        raise ValueError(f"unknown task {task!r}")
    head: dict[str, Any] = {"format": FORMAT, "format_version": FORMAT_VERSION}
    if task is not None:
        head = {"format": FORMAT, "format_version": 2, "task": task}
    return {
        **head,
        "profile": profile,
        "stage": stage,
        "candidate": candidate,
        "producer": producer,
        "environment": environment,
        "assets": assets,
        "cases": {},
        "batches": {},
        "tensors": {},
    }
