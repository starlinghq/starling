"""Model manifest loading and validation (plan section 9, backlog P-002)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "model-manifest.schema.json"
MANIFEST_NAME = "manifest.yaml"
GATES = tuple(f"G{i}" for i in range(8))
# Execution backend of the browser profiles (onnxruntime-web in a .NET browser-wasm app; follow-up plan W6).
BROWSER_BACKEND = "onnxruntime-web"


def is_browser(manifest: dict[str, Any]) -> bool:
    return manifest.get("execution", {}).get("backend", {}).get("name") == BROWSER_BACKEND


def graph_asset(manifest: dict[str, Any]) -> dict[str, Any]:
    """The graph a profile runs: `execution.graph.asset` when the manifest has one, else its first `.onnx` file."""
    asset = (manifest.get("execution", {}).get("graph") or {}).get("asset")
    if isinstance(asset, dict) and "path" in asset:
        return asset
    for f in manifest["origin"]["files"]:
        if f["path"].endswith(".onnx"):
            return f
    raise ValueError(f"{manifest['id']}: the manifest names no ONNX graph")


class _Loader(yaml.SafeLoader):
    """SafeLoader that keeps dates as strings so they validate as JSON Schema `format: date`."""


_Loader.yaml_implicit_resolvers = {
    key: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:timestamp"]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def load(path: Path) -> object:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_Loader)  # noqa: S506 - safe loader subclass


@dataclass(frozen=True)
class Problem:
    path: Path
    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.location}: {self.message}"


@cache
def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


@dataclass(frozen=True)
class StableBlockers:
    """What keeps a profile from the Stable label under the validator's rules (plan section 10: every gate applies in
    full before any Stable label; section 9: no untested capability in a Stable profile)."""

    untested_capabilities: tuple[str, ...]
    gates_not_passed: tuple[str, ...]

    def __bool__(self) -> bool:
        return bool(self.untested_capabilities or self.gates_not_passed)


def stable_blockers(manifest: dict[str, Any]) -> StableBlockers:
    """The Stable rules, independent of the manifest's current maturity. The validator and the catalog share them."""
    capabilities = manifest.get("capabilities", {})
    gates = manifest.get("qualification", {}).get("gates", {})
    return StableBlockers(
        untested_capabilities=tuple(sorted(k for k, v in capabilities.items() if v == "untested")),
        gates_not_passed=tuple(sorted(g for g in GATES if gates.get(g, {}).get("status") != "passed")),
    )


def _semantic_problems(manifest: dict[str, Any]) -> list[tuple[str, str]]:
    """Rules the schema cannot express."""
    problems = []
    if manifest.get("maturity") == "stable":
        blockers = stable_blockers(manifest)
        if blockers.untested_capabilities:
            untested = ", ".join(blockers.untested_capabilities)
            problems.append(("capabilities", f"stable profile has untested capabilities: {untested}"))
        if blockers.gates_not_passed:
            unpassed = ", ".join(blockers.gates_not_passed)
            problems.append(("qualification/gates", f"stable profile has gates not passed: {unpassed}"))
    return problems


def validate_file(path: Path) -> list[Problem]:
    try:
        manifest = load(path)
    except yaml.YAMLError as exc:
        return [Problem(path, "<root>", f"invalid YAML: {exc}")]

    problems = [
        Problem(path, "/".join(str(p) for p in err.absolute_path) or "<root>", err.message)
        for err in sorted(_validator().iter_errors(manifest), key=lambda e: list(e.absolute_path))
    ]
    if not problems:
        problems += [Problem(path, loc, msg) for loc, msg in _semantic_problems(manifest)]
    return problems


def find_manifests(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob(MANIFEST_NAME))
