#!/usr/bin/env python3
"""Helpers for the scheduled qualification workflow (.github/workflows/qualification.yml, P-018, workstream R6).

    python3 -I tools/qualification.py hosting <manifest.yaml>
        Prints "public" or "private: <where>". A profile is private when any `source_status` in its manifest is
        `placeholder` or `private` (graphs in private repositories until maintainer action M3). With
        $GITHUB_OUTPUT set, also writes `hosting=public|private` there.

    python3 -I tools/qualification.py lock <manifest.yaml> <out.json> [--reference-environment <environment.json>]
        Writes a model asset lock (Starlings.Acquisition JSON lock, schema_version 1) of every file in the
        manifest's `origin.files`, at `origin.weights`' Hugging Face repository and pinned revision. The acquisition
        tool downloads and verifies it (`Starlings.Acquisition.Tool download <out.json>`). With
        --reference-environment (the protected fixtures' environment.json, read only), the files the reference run
        recorded in `assets` are added too, when its `model` is the same repository and revision (e5's reference
        runner also reads `onnx/tokenizer.json`, which origin.files does not list). For e5 this is the full source
        snapshot the onboarding recipe needs (reference weights, configuration, tokenizer, ONNX graph); its runtime
        files are the same cache entries the tests and the soak read. Only public, single-repository Hub origins are
        supported; anything else is an error, never a guess.

    python3 -I tools/qualification.py snapshot <lock.json>
        Prints the lock's snapshot directory relative to the cache root: <repository>/<revision>.

Reads manifests and fixture environment records only (PyYAML, installed with model-pipeline). Never writes outside the given output path.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import yaml

PRIVATE_STATUSES = {"placeholder", "private"}
HUB = re.compile(r"^https://huggingface\.co/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/?$")
SHA = re.compile(r"^[0-9a-f]{40}$")


def _statuses(node, path="") -> list[tuple[str, str]]:
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else str(key)
            if key == "source_status":
                found.append((here, str(value)))
            else:
                found.extend(_statuses(value, here))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found.extend(_statuses(value, f"{path}[{i}]"))
    return found


def hosting(manifest: Path) -> int:
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    private = [f"{where}={status}" for where, status in _statuses(data) if status in PRIVATE_STATUSES]
    verdict = "private" if private else "public"
    print(f"{verdict}: {', '.join(private)}" if private else verdict)
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            fh.write(f"hosting={verdict}\n")
    return 0


def lock(manifest: Path, out: Path, reference_environment: Path | None = None) -> int:
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    if any(status in PRIVATE_STATUSES for _, status in _statuses(data)):
        print(f"error: {manifest}: hosting private (M3); no lock is written", file=sys.stderr)
        return 2
    origin = data.get("origin") or {}
    weights = origin.get("weights") or {}
    match = HUB.match(str(weights.get("url", "")))
    revision = str(weights.get("revision", ""))
    files = origin.get("files") or []
    if not match or not SHA.match(revision) or not files:
        print(
            f"error: {manifest}: origin.weights must be a Hugging Face repository URL with a full commit SHA, and "
            "origin.files must list the files",
            file=sys.stderr,
        )
        return 2
    entries = []
    for f in files:
        if not isinstance(f, dict) or not {"path", "sha256", "size"} <= f.keys() or "url" in f or "repository" in f:
            print(f"error: {manifest}: unsupported origin.files entry {f!r}", file=sys.stderr)
            return 2
        entries.append({"path": str(f["path"]), "size": int(f["size"]), "sha256": str(f["sha256"])})
    if reference_environment is not None:
        env = json.loads(reference_environment.read_text(encoding="utf-8"))
        model = env.get("model") or {}
        if model.get("repo") != match.group(1) or model.get("revision") != revision:
            print(f"error: {reference_environment}: model {model} is not {match.group(1)}@{revision}", file=sys.stderr)
            return 2
        known = {e["path"]: e for e in entries}
        for asset in env.get("assets") or []:
            entry = {"path": str(asset["path"]), "size": int(asset["size"]), "sha256": str(asset["sha256"])}
            if entry["path"] not in known:
                entries.append(entry)
                known[entry["path"]] = entry
            elif known[entry["path"]] != entry:
                print(f"error: {entry['path']}: manifest and reference environment disagree", file=sys.stderr)
                return 2
    document = {
        "schema_version": 1,
        "name": str(data["id"]).replace("/", "-") + "-snapshot",
        "repository": match.group(1),
        "revision": revision,
        "files": entries,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"{out}: {match.group(1)}@{revision[:8]}, {len(entries)} files, {sum(e['size'] for e in entries)} bytes")
    return 0


def snapshot(lock_file: Path) -> int:
    document = json.loads(lock_file.read_text(encoding="utf-8"))
    print(f"{document['repository']}/{document['revision']}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "hosting":
        return hosting(Path(argv[1]))
    if len(argv) == 2 and argv[0] == "snapshot":
        return snapshot(Path(argv[1]))
    if len(argv) == 3 and argv[0] == "lock":
        return lock(Path(argv[1]), Path(argv[2]))
    if len(argv) == 5 and argv[0] == "lock" and argv[3] == "--reference-environment":
        return lock(Path(argv[1]), Path(argv[2]), Path(argv[4]))
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
