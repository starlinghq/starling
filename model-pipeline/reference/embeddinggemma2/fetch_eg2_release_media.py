"""Fetch the EmbeddingGemma 2 image and video release media from Wikimedia Commons, verifying each file's license.

Input: a media list in the format of `media.json` (the Stable proposal's `select_media.py select`): per file the
Commons title, description page, file revision timestamp, download URL, SHA-1 and size. Commons is a live site, so
this script trusts nothing it did not check today:

1. License: for every file it queries the Commons API `imageinfo` `extmetadata` (`LicenseShortName`, `License`,
   `Artist`, `Credit`, `UsageTerms`, `AttributionRequired`) and keeps the file only if the license is public
   domain, CC0 or CC BY (any version), and neither the license nor the usage terms name share-alike, non-commercial,
   no-derivatives or the GFDL. Everything else is rejected and listed with the reason. The PD-* and CC-* templates
   of the description page are recorded too (the public-domain rationale). Commons metadata is data, not a legal
   review; the maintainer confirms each license on the description page.
2. Revision: the current Commons SHA-1 and size must equal the recorded revision (a re-uploaded file is rejected).
3. Bytes: each kept file is downloaded once into `--out` (5 s between downloads; upload.wikimedia.org rate-limits
   bulk clients), checked against the recorded SHA-1 and size, and recorded by SHA-256.

Writes into `--out`: the media files (never committed), `images.json` and `videos.json` in the manifest shape the
EmbeddingGemma 2 image and video reference runners read (id, file, how it was made, source, license, attribution,
format and size, SHA-256; images also the SHA-256 of the decoded pixels, `np.asarray(Image.open(f))`, as
`make_eg2_images.py` records), and `licenses.json` with the extmetadata of every file, kept or rejected. The two
manifests and `licenses.json` are small and can be committed; the media are pinned by URL and SHA-256 instead.
Ids are `commons-image-NN` / `commons-video-NN`, NN the file's position in the media list.

Usage:
    python -I fetch_eg2_release_media.py --media <media.json> --out <dir> [--licenses-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "starlings-model-pipeline/release-media (https://github.com/starlinghq/starling)"}
FIELDS = ("LicenseShortName", "License", "Artist", "Credit", "UsageTerms", "AttributionRequired")
ALLOWED = re.compile(
    r"^(pd|pd-.*|public domain|cc0|cc-zero|cc0-1\.0|cc[- ]by[- ][0-9.]+|cc-by-[0-9.]+)$", re.IGNORECASE
)
FORBIDDEN = re.compile(
    r"(share[- ]?alike|\bsa\b|-sa\b|non[- ]?commercial|\bnc\b|-nc\b|no[- ]?deriv|\bnd\b|-nd\b|gfdl)", re.I
)


def _get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:  # noqa: S310
                return json.load(r)
        except OSError:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Commons API failed: {url}")


def _download(url: str) -> bytes:
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:  # noqa: S310
                return r.read()
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                raise
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"rate limited: {url}")


def _text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", value or "")).strip()


def extmetadata(titles: list[str]) -> dict[str, dict]:
    """Current imageinfo (SHA-1, size, timestamp) and license extmetadata per title, 20 titles per request."""
    out = {}
    for i in range(0, len(titles), 20):
        data = _get(
            {
                "action": "query",
                "titles": "|".join(titles[i : i + 20]),
                "prop": "imageinfo",
                "iiprop": "sha1|size|timestamp|extmetadata",
                "iiextmetadatafilter": "|".join(FIELDS),
            }
        )
        normalized = {n["to"]: n["from"] for n in data.get("query", {}).get("normalized", [])}
        for page in data.get("query", {}).get("pages", []):
            info = (page.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata") or {}
            out[normalized.get(page["title"], page["title"])] = {
                "sha1": info.get("sha1"),
                "bytes": info.get("size"),
                "revision_timestamp": info.get("timestamp"),
                **{f: _text((meta.get(f) or {}).get("value", "")) for f in FIELDS},
            }
    return out


def license_templates(title: str) -> list[str]:
    """License templates on the description page (PD-*, CC-*): the public-domain rationale `License: pd` hides."""
    data = _get({"action": "query", "titles": title, "prop": "templates", "tlnamespace": "10", "tllimit": "500"})
    names = [
        t["title"].split(":", 1)[1] for p in data.get("query", {}).get("pages", []) for t in p.get("templates", [])
    ]
    return sorted(n for n in names if re.match(r"^(PD-|Cc-|CC-|Cc0|CC0)", n) and "/" not in n)


def license_verdict(meta: dict) -> str | None:
    """None when the license is allowed; otherwise the reason for rejecting the file."""
    lic = meta.get("License") or meta.get("LicenseShortName") or ""
    short = meta.get("LicenseShortName") or ""
    if not ALLOWED.match(lic):
        return f"license {lic!r} ({short!r}) is not public domain, CC0 or CC BY"
    for field in ("License", "LicenseShortName", "UsageTerms"):
        if not lic.lower().startswith("pd") and FORBIDDEN.search(meta.get(field) or ""):
            return f"{field} {meta.get(field)!r} names a share-alike, non-commercial, no-derivatives or GFDL term"
    if lic.lower().startswith("pd") and FORBIDDEN.search((meta.get("UsageTerms") or "") + " " + short):
        return f"public domain file with restrictive terms {meta.get('UsageTerms')!r}"
    return None


def image_properties(path: Path) -> dict:
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        exif = im.getexif().get(0x0112, 1)
        props = {"format": im.format, "mode": im.mode, "width": im.width, "height": im.height}
        pixels = np.asarray(im)
    return props | {
        "exif_orientation": int(exif),
        "decoded_sha256": hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest(),
        "decoded_dtype": str(pixels.dtype),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--media", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--licenses-only", action="store_true", help="check the licenses and revisions; download nothing")
    args = ap.parse_args()
    media = json.loads(args.media.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, list[dict]] = {"images": [], "videos": []}
    licenses, rejected = [], []
    for kind in ("images", "videos"):
        selected = media[kind]["selected"]
        current = extmetadata([r["title"] for r in selected])
        for i, rec in enumerate(selected):
            name = f"commons-{kind[:-1]}-{i:02d}"
            meta = current.get(rec["title"], {}) | {"templates": license_templates(rec["title"])}
            reason = license_verdict(meta)
            if reason is None and (meta.get("sha1") != rec["sha1"] or meta.get("bytes") != rec["bytes"]):
                reason = f"current revision {meta.get('revision_timestamp')} differs from the recorded one"
            licenses.append(
                {"id": name, "title": rec["title"], "description_url": rec["description_url"], "kept": reason is None}
                | {"reason": reason}
                | meta
            )
            if reason is not None:
                rejected.append({"id": name, "title": rec["title"], "reason": reason})
                print(f"REJECTED {name} {rec['title']}: {reason}")
                continue
            if args.licenses_only:
                print(f"ok {name} {meta['LicenseShortName']} ({meta['License']}): {rec['title']}")
                continue
            ext = Path(urllib.parse.urlparse(rec["url"]).path).suffix.lower()
            target = args.out / f"{name}{ext}"
            if not target.exists():
                target.write_bytes(_download(rec["url"]))
                time.sleep(5)
            data = target.read_bytes()
            if hashlib.sha1(data).hexdigest() != rec["sha1"] or len(data) != rec["bytes"]:  # noqa: S324 (Commons id)
                raise SystemExit(f"{rec['title']}: SHA-1 or size differs from the recorded revision")
            entry = {
                "id": name,
                "file": target.name,
                "made": "fetched unchanged from Wikimedia Commons",
                "title": rec["title"],
                "source": rec["description_url"],
                "url": rec["url"].split("?")[0],
                "revision_timestamp": rec["revision_timestamp"],
                "sha1": rec["sha1"],
                "license": meta["LicenseShortName"],
                "license_template": meta["License"],
                "license_page_templates": meta["templates"],
                "artist": meta["Artist"],
                "credit": meta["Credit"],
                "usage_terms": meta["UsageTerms"],
                "mime": rec["mime"],
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            if kind == "images":
                entry |= image_properties(target)
            else:
                entry |= {"width": rec["width"], "height": rec["height"], "duration_s": rec["duration_s"]}
            manifests[kind].append(entry)
            print(f"ok {name} {entry['license']} {entry['bytes']} bytes: {rec['title']}")
    source = {"media_list": args.media.name, "media_list_sha256": hashlib.sha256(args.media.read_bytes()).hexdigest()}
    checked = {"checked": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script": Path(__file__).name}
    (args.out / "licenses.json").write_text(
        json.dumps(source | checked | {"rejected": rejected, "files": licenses}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not args.licenses_only:
        for kind in ("images", "videos"):
            (args.out / f"{kind}.json").write_text(
                json.dumps(
                    source | {"generator": checked["script"], kind: manifests[kind]}, indent=2, ensure_ascii=False
                )
                + "\n",
                encoding="utf-8",
            )
    print(f"kept {len(manifests['images'])} images, {len(manifests['videos'])} videos; rejected {len(rejected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
