"""JSON asset lock validation: the format of Starlings.Acquisition's ModelAssetLock, versions 1 to 3.

The JSON lock is the interchange format for locks between Starling and its consumers. The schema is
`schemas/asset-lock.schema.json`; the checks a schema cannot express (duplicate file paths, compared without case as
the .NET reader does) are added here.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from model_pipeline.manifest import Problem

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "asset-lock.schema.json"
SCHEMA_VERSIONS = (1, 2, 3)


@cache
def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def lowest_version(lock: dict[str, Any]) -> int:
    """The version a writer uses for this lock: 3 with display_name or hosting, else 2 with a file url, else 1."""
    if "display_name" in lock or "hosting" in lock:
        return 3
    if any("url" in f for f in lock.get("files", [])):
        return 2
    return 1


def validate(lock: object, path: Path) -> list[Problem]:
    problems = [
        Problem(path, "/".join(str(p) for p in err.absolute_path) or "<root>", err.message)
        for err in sorted(_validator().iter_errors(lock), key=lambda e: list(e.absolute_path))
    ]
    if problems or not isinstance(lock, dict):
        return problems
    seen: set[str] = set()
    for index, file in enumerate(lock["files"]):
        key = file["path"].casefold()
        if key in seen:
            problems.append(Problem(path, f"files/{index}/path", f"duplicate file path {file['path']!r}"))
        seen.add(key)
    return problems


def validate_file(path: Path) -> list[Problem]:
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [Problem(path, "<root>", f"not a readable JSON document: {exc}")]
    return validate(lock, path)
