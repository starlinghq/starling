"""Download a Hugging Face repository snapshot by full commit SHA and verify it.

Resolves ``--revision`` to a commit SHA through the Hub API, downloads every listed file (or the ``--include``
subset) from ``resolve/<sha>/<path>``, computes SHA-256 locally, checks size and (for LFS files) the Hub's
SHA-256, and writes ``hashes.json`` next to the files. Standard library only.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HUB = "https://huggingface.co"


def _get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def _headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True, help="branch, tag or commit; resolved to a full commit SHA")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--include", action="append", default=[], help="glob; repeatable; default all files")
    args = ap.parse_args(argv)

    info = _get_json(f"{HUB}/api/models/{args.repo}/revision/{urllib.parse.quote(args.revision, safe='')}?blobs=true")
    sha = info["sha"]
    if args.revision != sha:
        print(f"resolved {args.repo}@{args.revision} -> {sha}", file=sys.stderr)
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for sib in sorted(info["siblings"], key=lambda s: s["rfilename"]):
        name = sib["rfilename"]
        if args.include and not any(fnmatch.fnmatch(name, pat) for pat in args.include):
            continue
        dest = args.out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        size = sib.get("size")
        if not (dest.exists() and dest.stat().st_size == size):
            url = f"{HUB}/{args.repo}/resolve/{sha}/{urllib.parse.quote(name)}"
            tmp = dest.with_suffix(dest.suffix + ".part")
            req = urllib.request.Request(url, headers=_headers())
            with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as f:
                while chunk := resp.read(1 << 22):
                    f.write(chunk)
            tmp.replace(dest)
        digest = sha256_file(dest)
        lfs = (sib.get("lfs") or {}).get("sha256")
        actual = dest.stat().st_size
        if actual != size or (lfs and lfs != digest):
            print(f"MISMATCH {name}: size {actual} vs {size}, sha256 {digest} vs {lfs}", file=sys.stderr)
            return 1
        records.append(
            {"path": name, "size": actual, "sha256": digest, "hub_lfs_sha256": lfs, "blob_id": sib.get("blobId")}
        )
        print(f"{digest}  {actual:>12}  {name}")
    manifest = {
        "repo": args.repo,
        "requested_revision": args.revision,
        "commit": sha,
        "last_modified": info.get("lastModified"),
        "files": records,
    }
    (args.out / "hashes.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
