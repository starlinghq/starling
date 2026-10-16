#!/usr/bin/env python3
"""Download a release set's reference files kept outside Git, verified against its protected record.

    python3 -I tools/fetch-release-references.py [--prune] <protected hosted-reference.json> <out dir>

The record (`model-pipeline/protected/<model>/release/<set>/hosted-reference.json`) names a Hugging Face dataset
repository, a full commit SHA and, per file, its size and SHA-256. Every file is downloaded anonymously at that commit
into <out dir> under its recorded relative path, and refused unless its size and SHA-256 equal the record. The evaluator
then checks every array or vector it reads against the digests the release set itself records (`tensors.json`,
`environment.json`), so this download check is a first gate, not the only one. A file already in <out dir> is kept
only if it matches the record; otherwise it is deleted and downloaded again, so a stale copy cannot stay. --prune also
deletes every other file under <out dir> (a cache directory that earlier held other references).

Exit 0 when every file is present and verified, 1 on any mismatch or download failure, 2 on bad arguments.
Nothing is written inside the repository: <out dir> must be outside it.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTECTED = REPO_ROOT / "model-pipeline" / "protected"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, attempts: int = 4) -> None:
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "starling-fetch-release-references"})
            with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
            tmp.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == attempts:
                raise
            print(f"  retry {attempt} for {dest.name}: {e}", file=sys.stderr)
            time.sleep(5 * attempt)


def main(argv: list[str]) -> int:
    prune = bool(argv) and argv[0] == "--prune"
    if prune:
        argv = argv[1:]
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    record_path = Path(argv[0]).resolve()
    out = Path(argv[1]).resolve()
    if PROTECTED not in record_path.parents:
        print(f"error: {record_path} is not under {PROTECTED}", file=sys.stderr)
        return 2
    if out == REPO_ROOT or REPO_ROOT in out.parents:
        print(f"error: out dir {out} is inside the repository", file=sys.stderr)
        return 2
    record = json.loads(record_path.read_text(encoding="utf-8"))
    src = record["source"]
    if src.get("repo_type") != "dataset" or not re.fullmatch(r"[0-9a-f]{40}", src.get("revision", "")):
        print("error: the record must name a dataset repository and a full 40-hex commit", file=sys.stderr)
        return 2
    base = f"https://huggingface.co/datasets/{src['repository']}/resolve/{src['revision']}/{src['path']}"
    out.mkdir(parents=True, exist_ok=True)
    for rel in record["files"]:
        if rel.startswith("/") or ".." in Path(rel).parts:
            print(f"error: unsafe path {rel!r} in the record", file=sys.stderr)
            return 2
    if prune:
        keep = {(out / rel).resolve() for rel in record["files"]}
        for f in sorted(p for p in out.rglob("*") if p.is_file() or p.is_symlink()):
            if f.resolve() not in keep:
                print(f"pruned {f.relative_to(out)}")
                f.unlink()
    bad = 0
    for rel, want in record["files"].items():
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not (dest.is_file() and dest.stat().st_size == want["size"] and sha256_file(dest) == want["sha256"]):
            if dest.exists() or dest.is_symlink():
                print(f"stale {rel}: does not match the record; deleted")
                dest.unlink()
            try:
                download(f"{base}/{urllib.parse.quote(rel)}", dest)
            except Exception as e:  # noqa: BLE001 - reported per file, the job fails below
                print(f"FAIL {rel}: download: {e}")
                bad += 1
                continue
        size, digest = dest.stat().st_size, sha256_file(dest)
        if size == want["size"] and digest == want["sha256"]:
            print(f"ok   {rel} {digest}")
        else:
            print(f"FAIL {rel}: {size} bytes, SHA-256 {digest}; record {want['size']} bytes, {want['sha256']}")
            dest.unlink(missing_ok=True)
            bad += 1
    n = len(record["files"])
    source = f"{src['repository']}@{src['revision']}"
    print(f"{n - bad}/{n} files verified against {record_path.relative_to(REPO_ROOT)} ({source})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
