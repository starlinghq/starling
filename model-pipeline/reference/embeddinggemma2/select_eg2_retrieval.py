"""Select the EmbeddingGemma 2 text-to-image and text-to-video retrieval sets (G3) from Wikimedia Commons.

Each set pairs a Commons file with its own description, written by the file's uploader: the description is the query
and the file is its only relevant document. No text or file is written or chosen by hand. The procedure is a fixed
filter plus one seeded draw, so a selection depends only on the seed and on Commons at query time; Commons is a live
site, so the selection is recorded (`select` writes the media list, with each description verbatim) and everything
downstream is rebuilt from that record, never by re-querying.

    python -I select_eg2_retrieval.py select --kind image|video --out <media.json> [--exclude <media.json>]...
    python -I fetch_eg2_release_media.py --media <media.json> --out <media dir>
    python -I select_eg2_retrieval.py build --media <media.json> --fetched <media dir> --out <dataset dir>

`select`:
- image: `Category:Featured pictures on Wikimedia Commons`, members in sort-key order from each sort-key prefix A to Z,
  read in API batches of 50 until at least `--max-members` (default 50) per prefix have file information (Commons
  re-stamps membership times, so a time window is not stable); JPEG or PNG, at
  most 15 MB and 50,000,000 pixels (the EmbeddingGemma 2 package's default decode limit, so no row needs a raised
  limit); draw `--count` (default 150) with `random.Random(f"{seed}:image-retrieval")`.
- video: `Category:Videos by NASA`; WebM only (two of the six Ogg release videos cannot be opened by upstream's
  torchcodec `seek_mode="exact"`, and an Ogg Theora seek quirk differs from sequential decoding), at most 40 MB and
  60 s; draw `--count` (default 50) with `random.Random(f"{seed}:video-retrieval")`.
- Both: license public domain, CC0 or CC BY by the same rule as `fetch_eg2_release_media.py`, which checks it again
  at fetch time; files whose title appears in an `--exclude` media list are dropped (the P-030 release media, so the
  sets stay disjoint). The query is the file's English `ImageDescription` (Commons `extmetadata`, language `en`),
  with HTML entities unescaped, then tags removed and whitespace collapsed; files whose query has fewer than 20
  characters are dropped, a longer query is cut to its first 300 characters at the last space, and files whose query
  equals another eligible file's query are dropped (one relevant document per query). The candidates are sorted by
  title before the draw.

`build` writes the retrieval dataset in the evaluator's format (`model-pipeline/README.md`, "Task quality (G3)") from
the media list and the fetch script's manifest (`images.json` or `videos.json`, which holds only the files whose
license and revision passed at fetch time):
- `queries.jsonl`: {"id": "q-<media id>", "text": <recorded description>}
- `corpus.jsonl`: {"id": <media id>, "file": <file name>, "sha256": <file SHA-256>}; no text: the document is the
  media file, pinned by digest, and the files stay outside Git.
- `qrels.jsonl`: {"query_id", "doc_id", "relevance": 1}, one per query.
- `dataset.json`: name, modality, the prompt convention (queries with the model's `query` prompt, media without a
  prompt, as the model card says), the media list's SHA-256, per-file source, license and attribution.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
import string
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_eg2_release_media import ALLOWED, FORBIDDEN, _get  # noqa: E402

CATEGORIES = {"image": "Category:Featured pictures on Wikimedia Commons", "video": "Category:Videos by NASA"}
MIMES = {"image": ("image/jpeg", "image/png"), "video": ("video/webm",)}
MAX_BYTES = {"image": 15_000_000, "video": 40_000_000}
MAX_PIXELS = 50_000_000
MAX_SECONDS = 60
MIN_QUERY, MAX_QUERY = 20, 300
FIELDS = "LicenseShortName|License|Artist|Credit|AttributionRequired|UsageTerms|ImageDescription"
PROMPTS = {"query": "query", "document": None}


def clean(value: str) -> str:
    """Description text: entities unescaped, then tags removed and whitespace collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(value or ""))).strip()


def query_text(description: str) -> str | None:
    text = clean(description)
    if len(text) < MIN_QUERY:
        return None
    if len(text) > MAX_QUERY:
        cut = text[:MAX_QUERY]
        text = cut[: cut.rfind(" ")] if " " in cut else cut
    return text


def _members(category: str, extra: dict, limit: int | None) -> list[dict]:
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": category,
        "gcmtype": "file",
        "gcmlimit": "50",
        "prop": "imageinfo",
        "iiprop": "url|sha1|size|mime|timestamp|extmetadata|metadata",
        "iiextmetadatafilter": FIELDS,
        "iiextmetadatalanguage": "en",
        **extra,
    }
    pages: dict[str, dict] = {}
    while True:
        data = _get(params)
        for page in data.get("query", {}).get("pages", []):
            if page.get("imageinfo"):
                pages.setdefault(page["title"], page)
        if "continue" not in data or (limit is not None and len(pages) >= limit):
            break
        params.update(data["continue"])
    return [pages[t] for t in sorted(pages)]


def _meta(info: dict, key: str) -> str:
    return clean(((info.get("extmetadata") or {}).get(key) or {}).get("value", ""))


def _duration(info: dict) -> float | None:
    for m in info.get("metadata") or []:
        if m.get("name") in ("playtime_seconds", "length", "duration"):
            try:
                return float(m["value"])
            except (TypeError, ValueError):
                return None
    return None


def record(page: dict, kind: str) -> dict | None:
    """The media-list entry of an eligible file, else None."""
    info = page["imageinfo"][0]
    lic = _meta(info, "License") or _meta(info, "LicenseShortName")
    terms = " ".join(_meta(info, k) for k in ("License", "LicenseShortName", "UsageTerms"))
    if not ALLOWED.match(lic) or FORBIDDEN.search(terms):
        return None
    if info.get("mime") not in MIMES[kind] or info.get("size", 0) > MAX_BYTES[kind]:
        return None
    width, height = info.get("width") or 0, info.get("height") or 0
    duration = _duration(info)
    if kind == "image" and width * height > MAX_PIXELS:
        return None
    if kind == "video" and (duration is None or duration > MAX_SECONDS):
        return None
    raw = ((info.get("extmetadata") or {}).get("ImageDescription") or {}).get("value", "")
    query = query_text(raw)
    if query is None:
        return None
    return {
        "title": page["title"],
        "description_url": info["descriptionurl"],
        "revision_timestamp": info["timestamp"],
        "url": info["url"],
        "sha1": info["sha1"],
        "bytes": info["size"],
        "width": width,
        "height": height,
        "mime": info["mime"],
        "duration_s": duration,
        "license": lic,
        "license_short_name": _meta(info, "LicenseShortName"),
        "artist": _meta(info, "Artist"),
        "credit": _meta(info, "Credit"),
        "description": query,
    }


def select(args: argparse.Namespace) -> int:
    kind = args.kind
    excluded: set[str] = set()
    for path in args.exclude:
        media = json.loads(path.read_text(encoding="utf-8"))
        for key in ("images", "videos"):
            excluded |= {r["title"] for r in media.get(key, {}).get("selected", [])}
    if kind == "image":
        prefixes = list(string.ascii_uppercase)
        pages: dict[str, dict] = {}
        for prefix in prefixes:
            for page in _members(CATEGORIES[kind], {"gcmstartsortkeyprefix": prefix}, args.max_members):
                pages.setdefault(page["title"], page)
        members = [pages[t] for t in sorted(pages)]
    else:
        prefixes = []
        members = _members(CATEGORIES[kind], {}, None)
    eligible = [r for r in (record(p, kind) for p in members) if r and r["title"] not in excluded]
    counts: dict[str, int] = {}
    for r in eligible:
        counts[r["description"]] = counts.get(r["description"], 0) + 1
    eligible = sorted((r for r in eligible if counts[r["description"]] == 1), key=lambda r: r["title"])
    if len(eligible) < args.count:
        raise SystemExit(f"only {len(eligible)} eligible {kind} files for {args.count}")
    picked = sorted(
        random.Random(f"{args.seed}:{kind}-retrieval").sample(eligible, args.count), key=lambda r: r["title"]
    )
    key = f"{kind}s"
    out = {
        "script": "model-pipeline/reference/embeddinggemma2/select_eg2_retrieval.py",
        "queried": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "license_allowlist": "public domain, CC0, CC BY (any version); CC BY-SA, GFDL, NC, ND and unknown excluded",
        "excluded_media_lists": [p.name for p in args.exclude],
        key: {
            "category": CATEGORIES[kind],
            "sort_key_prefixes": prefixes,
            "first_members_per_sort_key_prefix": args.max_members if prefixes else None,
            "members": len(members),
            "eligible": len(eligible),
            "selected": picked,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{kind}: {len(picked)} drawn of {len(eligible)} eligible, {len(members)} members")
    return 0


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def build(args: argparse.Namespace) -> int:
    media = json.loads(args.media.read_text(encoding="utf-8"))
    kinds = [k for k in ("images", "videos") if k in media]
    if len(kinds) != 1:
        raise SystemExit(f"{args.media}: expected exactly one of images or videos")
    key = kinds[0]
    kind = key[:-1]
    by_title = {r["title"]: r for r in media[key]["selected"]}
    fetched = json.loads((args.fetched / f"{key}.json").read_text(encoding="utf-8"))
    if fetched.get("media_list_sha256") != hashlib.sha256(args.media.read_bytes()).hexdigest():
        raise SystemExit(f"{args.fetched}/{key}.json was not fetched from {args.media}")
    queries, corpus, qrels, files = [], [], [], []
    for entry in fetched[key]:
        rec = by_title[entry["title"]]
        data = (args.fetched / entry["file"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise SystemExit(f"{entry['file']}: SHA-256 differs from {key}.json")
        qid = f"q-{entry['id']}"
        queries.append({"id": qid, "text": rec["description"]})
        corpus.append({"id": entry["id"], "file": entry["file"], "sha256": entry["sha256"]})
        qrels.append({"query_id": qid, "doc_id": entry["id"], "relevance": 1})
        files.append(
            {
                "id": entry["id"],
                "title": entry["title"],
                "source": entry["source"],
                "url": entry["url"],
                "revision_timestamp": entry["revision_timestamp"],
                "sha1": entry["sha1"],
                "bytes": entry["bytes"],
                "license": entry["license"],
                "license_page_templates": entry["license_page_templates"],
                "artist": entry["artist"],
                "credit": entry["credit"],
            }
        )
    args.out.mkdir(parents=True, exist_ok=True)
    _jsonl(args.out / "queries.jsonl", queries)
    _jsonl(args.out / "corpus.jsonl", corpus)
    _jsonl(args.out / "qrels.jsonl", qrels)
    meta = {
        "name": args.name,
        "modality": kind,
        "prompts": {
            "query": "the model's `query` prompt (task: search result | query: )",
            "document": f"none: the {kind} alone, without a prompt (model card: prefixes apply to text only)",
        },
        "queries": "the file's English Commons description (ImageDescription), cleaned and cut as the script says",
        "selection": {
            "script": "model-pipeline/reference/embeddinggemma2/select_eg2_retrieval.py",
            "seed": media["seed"],
            "queried": media["queried"],
            "media_list": args.media.name,
            "media_list_sha256": hashlib.sha256(args.media.read_bytes()).hexdigest(),
            "fetch": "model-pipeline/reference/embeddinggemma2/fetch_eg2_release_media.py",
            "selected": len(media[key]["selected"]),
            "kept_at_fetch": len(files),
        },
        "media_in_git": False,
        "files": files,
    }
    (args.out / "dataset.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(queries)} queries, {len(corpus)} {kind} documents")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--kind", choices=("image", "video"), required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--seed", type=int, default=20261015)
    s.add_argument("--count", type=int)
    s.add_argument("--max-members", type=int, default=50)
    s.add_argument("--exclude", type=Path, action="append", default=[])
    b = sub.add_parser("build")
    b.add_argument("--media", type=Path, required=True)
    b.add_argument("--fetched", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--name", required=True)
    args = ap.parse_args()
    if args.cmd == "select":
        args.count = args.count or (150 if args.kind == "image" else 50)
        return select(args)
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
