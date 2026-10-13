"""The owned tokenizer as a dependency: the published Starlings.Tokenizers package (starlinghq/tokenizers).

ai-dotnet/Directory.Packages.props pins one exact version, and the committed packages.lock.json files record the
package's SHA-512. Code identities and consumer checks use those two facts instead of tokenizer source files.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from model_pipeline.onboarding_runner import PipelineError

TOKENIZER = "Starlings.Tokenizers"
PINS = "ai-dotnet/Directory.Packages.props"


def tokenizer_version(root: Path) -> str:
    """The exact Starlings.Tokenizers version ai-dotnet/Directory.Packages.props pins ("[x]")."""
    for item in ET.parse(root / PINS).iter("PackageVersion"):
        if item.attrib.get("Include") == TOKENIZER:
            value = item.attrib.get("Version", "")
            if not (value.startswith("[") and value.endswith("]") and "," not in value):
                raise PipelineError(f"{TOKENIZER} must be pinned to one exact version, got {value!r}")
            return value[1:-1]
    raise PipelineError(f"{PINS} does not pin {TOKENIZER}")


def tokenizer_content_hash(root: Path) -> str:
    """The SHA-512 (base64) of the pinned tokenizer package, from the committed packages.lock.json files."""
    hashes = set()
    for lock in (root / "ai-dotnet/src").glob("*/packages.lock.json"):
        for dependencies in json.loads(lock.read_text(encoding="utf-8")).get("dependencies", {}).values():
            entry = dependencies.get(TOKENIZER)
            if entry and entry.get("contentHash"):
                hashes.add(entry["contentHash"])
    if len(hashes) != 1:
        raise PipelineError(f"lock files must record exactly one {TOKENIZER} content hash, found {len(hashes)}")
    return hashes.pop()
