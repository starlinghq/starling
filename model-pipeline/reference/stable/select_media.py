"""Select (and fetch) the proposed EmbeddingGemma 2 image and video release media from Wikimedia Commons.

Commons is a live site, so the selection is recorded, not re-derived: ``select`` queries fixed categories, keeps
files under a fixed license allowlist, draws a seeded sample and writes ``media.json`` with, per file, the title,
the file revision timestamp, the Commons SHA-1, size, MIME type, author/credit and license as Commons reports them,
and the description page. ``fetch`` downloads exactly those revisions, verifies SHA-1 and size, records SHA-256
and the decoded size, and writes ``images.json`` / ``videos.json`` for the EmbeddingGemma 2 runners. Media bytes are
never committed.

License allowlist: public domain (including US government works), CC0, CC BY (any version). CC BY-SA, GFDL,
non-commercial and unknown licenses are excluded, so a fixture derived from the media carries attribution
obligations only. Every file's license is as Commons states it in the file's ``extmetadata`` (data, not
instructions); the maintainer confirms each before acceptance.

Selection (deterministic for a seed and the recorded category snapshot):
- images: ``Category:Featured pictures on Wikimedia Commons``, the first ``--max-members`` members in category
  sort-key order from each of the sort-key prefixes ``A E I M Q U`` (Commons re-stamps membership times, so a
  time window is not stable; prefixes spread the draw over the alphabet), JPEG or PNG, at most 20 MB; draw
  ``--images`` with
  ``random.Random(f"{seed}:images")`` from the eligible titles sorted by title.
- videos: ``Category:Videos by NASA``, WebM or Ogg video, at most 40 MB and 60 s; draw ``--videos`` with
  ``random.Random(f"{seed}:videos")``.

Usage:
    python -I select_media.py select --out <media.json> [--seed 20261014]
    python -I select_media.py fetch --media <media.json> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sources  # noqa: E402

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "starlings-model-pipeline/stable-proposal (https://github.com/starlinghq/starling)"}
IMAGE_CATEGORY = "Category:Featured pictures on Wikimedia Commons"
VIDEO_CATEGORY = "Category:Videos by NASA"
PREFIXES = ("A", "E", "I", "M", "Q", "U")
ALLOWED = re.compile(
    r"^(pd|pd-.*|public domain|cc0|cc-zero|cc0-1\.0|cc[- ]by[- ][0-9.]+|cc-by-[0-9.]+)$", re.IGNORECASE
)
EXCLUDED = re.compile(r"(sa|nc|nd|gfdl)", re.IGNORECASE)


def _get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except OSError:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Commons API failed: {url}")


def _download(url: str) -> bytes:
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                raise
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"rate limited: {url}")


def _members(category: str, extra: dict, limit: int | None = None) -> list[dict]:
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmtype": "file",
        "gcmlimit": "100",
        "prop": "imageinfo",
        "iiprop": "url|sha1|size|mime|timestamp|extmetadata|metadata",
        "iiextmetadatafilter": "LicenseShortName|License|Artist|Credit|AttributionRequired|UsageTerms",
        **extra,
    }
    pages = []
    while True:
        data = _get(params)
        pages += data.get("query", {}).get("pages", [])
        if "continue" not in data or (limit is not None and len(pages) >= limit):
            return pages[:limit] if limit is not None else pages
        params.update(data["continue"])


def _meta(info: dict, key: str) -> str:
    return re.sub(r"<[^>]+>", "", ((info.get("extmetadata") or {}).get(key) or {}).get("value", "")).strip()


def _duration(info: dict) -> float | None:
    for m in info.get("metadata") or []:
        if m.get("name") in ("playtime_seconds", "length", "duration"):
            try:
                return float(m["value"])
            except (TypeError, ValueError):
                return None
    return None


def _record(page: dict) -> dict | None:
    info = (page.get("imageinfo") or [None])[0]
    if not info:
        return None
    lic = _meta(info, "License") or _meta(info, "LicenseShortName")
    if not ALLOWED.match(lic) or (EXCLUDED.search(lic) and not lic.lower().startswith("pd")):
        return None
    return {
        "title": page["title"],
        "description_url": info["descriptionurl"],
        "revision_timestamp": info["timestamp"],
        "url": info["url"],
        "sha1": info["sha1"],
        "bytes": info["size"],
        "width": info.get("width"),
        "height": info.get("height"),
        "mime": info["mime"],
        "duration_s": _duration(info),
        "license": lic,
        "license_short_name": _meta(info, "LicenseShortName"),
        "artist": _meta(info, "Artist"),
        "credit": _meta(info, "Credit"),
    }


def select(args: argparse.Namespace) -> int:
    imgs = []
    for prefix in PREFIXES:
        imgs += _members(IMAGE_CATEGORY, {"gcmstartsortkeyprefix": prefix}, limit=args.max_members)
    imgs = list({p["title"]: p for p in imgs}.values())
    images = sorted(
        (r for r in map(_record, imgs) if r and r["mime"] in ("image/jpeg", "image/png") and r["bytes"] <= 20_000_000),
        key=lambda r: r["title"],
    )
    vids = _members(VIDEO_CATEGORY, {})
    videos = sorted(
        (
            r
            for r in map(_record, vids)
            if r
            and r["mime"] in ("video/webm", "application/ogg", "video/ogg")
            and r["bytes"] <= 40_000_000
            and r["duration_s"] is not None
            and r["duration_s"] <= 60
        ),
        key=lambda r: r["title"],
    )
    picked_images = sorted(random.Random(f"{args.seed}:images").sample(images, args.images), key=lambda r: r["title"])
    picked_videos = sorted(random.Random(f"{args.seed}:videos").sample(videos, args.videos), key=lambda r: r["title"])
    sources.write_json(
        args.out,
        {
            "status": "PROPOSED: candidate P-030 media for the EmbeddingGemma 2 image and video release sets",
            "script": "model-pipeline/reference/stable/select_media.py",
            "queried": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed": args.seed,
            "license_allowlist": "public domain, CC0, CC BY (any version); CC BY-SA, GFDL, NC, ND and unknown excluded",
            "images": {
                "category": IMAGE_CATEGORY,
                "first_members_per_sort_key_prefix": args.max_members,
                "sort_key_prefixes": list(PREFIXES),
                "members": len(imgs),
                "eligible": len(images),
                "selected": picked_images,
            },
            "videos": {
                "category": VIDEO_CATEGORY,
                "members": len(vids),
                "eligible": len(videos),
                "selected": picked_videos,
            },
        },
    )
    print(
        f"images {len(picked_images)}/{len(images)} eligible of {len(imgs)}; "
        f"videos {len(picked_videos)}/{len(videos)} eligible of {len(vids)}"
    )
    return 0


def fetch(args: argparse.Namespace) -> int:
    media = json.loads(args.media.read_text(encoding="utf-8"))
    out: dict[str, list[dict]] = {"images": [], "videos": []}
    for kind in ("images", "videos"):
        for i, rec in enumerate(media[kind]["selected"]):
            ext = Path(urllib.parse.urlparse(rec["url"]).path).suffix.lower()
            name = f"commons-{kind[:-1]}-{i:02d}{ext}"
            target = args.out / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(_download(rec["url"]))
                time.sleep(5)
            data = target.read_bytes()
            if hashlib.sha1(data).hexdigest() != rec["sha1"] or len(data) != rec["bytes"]:
                raise SystemExit(f"{rec['title']}: SHA-1 or size differs from the recorded revision")
            out[kind].append(
                {
                    "id": name.rsplit(".", 1)[0],
                    "file": name,
                    "made": "fetched",
                    "source": f"{rec['description_url']} (revision {rec['revision_timestamp']}, SHA-1 {rec['sha1']})",
                    "license": f"{rec['license_short_name'] or rec['license']}; {rec['artist'] or rec['credit']}",
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    sources.write_json(args.out / "images.json", {"images": out["images"]})
    sources.write_json(args.out / "videos.json", {"videos": out["videos"]})
    print(f"fetched {len(out['images'])} images and {len(out['videos'])} videos into {args.out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--seed", type=int, default=20261014)
    s.add_argument("--images", type=int, default=16)
    s.add_argument("--videos", type=int, default=6)
    s.add_argument("--max-members", type=int, default=100)
    f = sub.add_parser("fetch")
    f.add_argument("--media", type=Path, required=True)
    f.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    return select(args) if args.cmd == "select" else fetch(args)


if __name__ == "__main__":
    raise SystemExit(main())
