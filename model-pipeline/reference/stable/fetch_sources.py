"""Fetch every pinned source of the Stable proposal and verify it against ``sources.lock.json``.

Usage:
    python -I fetch_sources.py --cache <dir> [--only massive,gsm8k] [--record-digests]

``--record-digests`` records the SHA-256 of files that have none yet (first run only; review the diff).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sources  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--record-digests", action="store_true")
    args = ap.parse_args()
    only = {k for k in args.only.split(",") if k}
    for key, src in sources.SOURCES.items():
        if only and key not in only:
            continue
        for path in src.files:
            target = sources.fetch(key, path, args.cache, record=args.record_digests)
            print(f"{key}@{src.revision[:8]}/{path}: {target.stat().st_size} bytes ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
