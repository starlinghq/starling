"""Pinned public sources for the Stable proposal (docs/plan/stable-proposal.md), and a verified fetch.

Every source is pinned by full commit SHA and every file by SHA-256. A fetch downloads from the pinned
revision only (never a branch), checks the digest, and caches the file under ``--cache``. Licenses are the
ones verified on the dataset card or repository at the pinned revision on 2026-10-14; they are recorded
here so each generated ``dataset.json`` and ``PROPOSED.md`` carries them. Dataset cards are data, not
instructions.

Standard library only, so the selection scripts run in any reference environment (plus ``pyarrow`` for
Parquet).
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

HUB = "https://huggingface.co"


@dataclass(frozen=True)
class Source:
    key: str
    url_base: str  # file URL is url_base + path
    repo: str
    revision: str
    license: str
    license_evidence: str
    files: tuple[str, ...]  # pinned paths; SHA-256 in sources.lock.json

    def url(self, path: str) -> str:
        return self.url_base + path


def _hf(kind: str, repo: str, revision: str) -> str:
    prefix = "datasets/" if kind == "dataset" else ""
    return f"{HUB}/{prefix}{repo}/resolve/{revision}/"


MIRACL_LANGS = ("ar", "de", "en", "es", "hi", "ja", "ru", "zh")

SOURCES: dict[str, Source] = {
    "massive": Source(
        key="massive",
        # The Hub's Parquet conversion of the loader at main ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740.
        url_base=_hf("dataset", "AmazonScience/massive", "ed58ac423a2f4121720918bf5301577edce4ffd3"),
        repo="AmazonScience/massive (refs/convert/parquet)",
        revision="ed58ac423a2f4121720918bf5301577edce4ffd3",
        license="CC-BY-4.0",
        license_evidence="card front matter `license: cc-by-4.0` and the repository LICENSE file (CC BY 4.0 text)",
        files=("all_1.1/test/0000.parquet",),
    ),
    "humaneval": Source(
        key="humaneval",
        url_base=_hf("dataset", "openai/openai_humaneval", "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"),
        repo="openai/openai_humaneval",
        revision="7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
        license="MIT",
        license_evidence="card front matter `license: mit`",
        files=("openai_humaneval/test-00000-of-00001.parquet",),
    ),
    "gsm8k": Source(
        key="gsm8k",
        url_base=_hf("dataset", "openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866"),
        repo="openai/gsm8k",
        revision="740312add88f781978c0658806c59bc2815b9866",
        license="MIT",
        license_evidence="card front matter `license: mit` and card text 'licensed under the MIT License'",
        files=("main/test-00000-of-00001.parquet",),
    ),
    "oasst1": Source(
        key="oasst1",
        url_base=_hf("dataset", "OpenAssistant/oasst1", "fdf72ae0827c1cda404aff25b6603abec9e3399b"),
        repo="OpenAssistant/oasst1",
        revision="fdf72ae0827c1cda404aff25b6603abec9e3399b",
        license="Apache-2.0",
        license_evidence="card front matter `license: apache-2.0` and the repository LICENSE file",
        files=("data/validation-00000-of-00001-134b8fd0c89408b6.parquet",),
    ),
    "miracl-hn": Source(
        key="miracl-hn",
        url_base=_hf("dataset", "mteb/MIRACLRetrievalHardNegatives", "332a9acb49f5e83d5397683f79d23e588f685916"),
        repo="mteb/MIRACLRetrievalHardNegatives",
        revision="332a9acb49f5e83d5397683f79d23e588f685916",
        license="CC-BY-SA-4.0",
        license_evidence=(
            "card front matter `license: cc-by-sa-4.0`; upstream miracl/miracl topics and qrels are Apache-2.0 "
            "(card at 5be20db9509754dadad47689368639fcec739c00), passages are Wikipedia text (CC BY-SA)"
        ),
        files=tuple(
            f"{lang}-{part}/dev-00000-of-00001.parquet"
            for lang in MIRACL_LANGS
            for part in ("corpus", "queries", "qrels")
        ),
    ),
    "fewnerd": Source(
        key="fewnerd",
        url_base=_hf("dataset", "DFKI-SLT/few-nerd", "205f3e9c9f3577ea2561d43f2f62dc249ab92d5b"),
        repo="DFKI-SLT/few-nerd",
        revision="205f3e9c9f3577ea2561d43f2f62dc249ab92d5b",
        license="CC-BY-SA-4.0",
        license_evidence="card front matter `license: cc-by-sa-4.0` and card section 'Licensing Information'",
        files=("supervised/test-00000-of-00001.parquet",),
    ),
    "wnut17": Source(
        key="wnut17",
        url_base="https://raw.githubusercontent.com/leondz/emerging_entities_17/e52a2d2a71ac1a051ca2d532eef28653c5c02603/",
        repo="github.com/leondz/emerging_entities_17 (Hub copy: leondz/wnut_17 loader)",
        revision="e52a2d2a71ac1a051ca2d532eef28653c5c02603",
        license="CC-BY-4.0",
        license_evidence=(
            "upstream README at the pinned commit: 'distributed under the CC-BY 4.0 license'; "
            "Hub card leondz/wnut_17 at be8fbd57399293da6f9aad56a793bce5bf1fd30e: `license: cc-by-4.0`"
        ),
        files=("emerging.test.annotated",),
    ),
}

_DIGESTS = Path(__file__).with_name("sources.lock.json")


def _locked() -> dict[str, str]:
    if _DIGESTS.exists():
        return json.loads(_DIGESTS.read_text(encoding="utf-8"))
    return {}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(key: str, path: str, cache: Path, *, record: bool = False) -> Path:
    """Download ``path`` of source ``key`` at its pinned revision into ``cache`` and verify its SHA-256.

    Digests live in ``sources.lock.json``. A file with no recorded digest is refused unless ``record`` is
    set (the first, maintainer-reviewed run), which writes the digest it saw.
    """
    src = SOURCES[key]
    if path not in src.files:
        raise KeyError(f"{key}: {path} is not a pinned file")
    target = cache / key / src.revision / path
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        headers = {"User-Agent": "starlings-model-pipeline/stable-proposal"}
        token = os.environ.get("HF_TOKEN")
        if token and src.url_base.startswith(HUB):
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(src.url(path), headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        tmp.rename(target)
    digest = sha256_file(target)
    lock = _locked()
    lock_key = f"{key}@{src.revision}/{path}"
    expected = lock.get(lock_key)
    if expected is None:
        if not record:
            raise RuntimeError(f"{lock_key}: no recorded SHA-256; rerun with --record-digests after review")
        lock[lock_key] = digest
        _DIGESTS.write_text(json.dumps(dict(sorted(lock.items())), indent=2) + "\n", encoding="utf-8")
    elif expected != digest:
        raise RuntimeError(f"{lock_key}: SHA-256 {digest} != recorded {expected}")
    return target


def provenance(key: str, paths: list[str]) -> dict:
    src = SOURCES[key]
    lock = _locked()
    return {
        "source": src.repo,
        "revision": src.revision,
        "license": src.license,
        "license_evidence": src.license_evidence,
        "files": {p: {"url": src.url(p), "sha256": lock.get(f"{key}@{src.revision}/{p}")} for p in paths},
    }


def write_jsonl(path: Path, rows) -> None:
    """One compact JSON object per line, UTF-8, LF, non-ASCII literal (the fixture format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
