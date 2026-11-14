#!/usr/bin/env python3
"""G0 provenance check of registry manifests, with anonymous network access only (plan section 10, G0).

    python3 -I tools/check-provenance.py [<manifest.yaml> ...]
        Default: every model-pipeline/registry/*/*/manifest.yaml. For each manifest it checks, and prints one line
        per finding:

        1. Immutable source lock: origin.code and origin.weights pinned by a full 40-hex commit; every origin.files
           entry has a 64-hex SHA-256 and a size; every hosted Starling file (execution.graph asset sources and the
           hosting source_base) is addressed at a full commit and recorded `source_status: public`.
        2. Accessible reference: every origin.files entry is fetched anonymously (no token, no cookies) and verified
           against its pin: first from the Hugging Face weights repository at the pinned commit; when that does not
           serve the pinned bytes, from the recorded Starling hosting (execution.graph.hosting), which mirrors the
           upstream files and carries Starling's own builds (the ortgenai-*, transformers-* and chat_template.jinja
           files and the fp32 ONNX graphs and bundle files upstream never shipped). A graph asset declared
           `source_status: not-hosted` (Starling-built, no public source) is recorded, not fetched. Large (LFS) files
           are checked through the Hub's X-Linked-Etag (the file's SHA-256) and X-Linked-Size headers without
           downloading them; small files are downloaded and hashed. The pinned code and weights commits must resolve
           anonymously.
        3. Rights recorded: rights.code_license, weights_license, tokenizer_terms and redistribution are present.

        Exits 0 only if every check passes. Never writes files; never sends a token.

Reads manifests only (PyYAML, installed with model-pipeline).
"""

from __future__ import annotations

import hashlib
import pathlib
import re
import sys
import urllib.error
import urllib.request

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHA1 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
HF = "https://huggingface.co/"
SMALL = 64 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102, ANN001, ANN201
        return None


OPENER = urllib.request.build_opener(NoRedirect)
FOLLOW = urllib.request.build_opener()
HEADERS = {"User-Agent": "starlings-check-provenance/1"}


def head(url: str) -> tuple[int, dict[str, str]]:
    req = urllib.request.Request(url, method="HEAD", headers=HEADERS)
    try:
        with OPENER.open(req, timeout=60) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}
    except urllib.error.HTTPError as e:
        return e.code, {k.lower(): v for k, v in e.headers.items()}


def get(url: str) -> bytes:
    with FOLLOW.open(urllib.request.Request(url, headers=HEADERS), timeout=120) as r:
        return r.read()


def verify_file(url: str, sha256: str, size: int) -> str | None:
    """None when the anonymous URL serves exactly these bytes, else the reason."""
    status, h = head(url)
    # LFS files: the linked ETag is the file's SHA-256. Plain Git files carry their blob SHA-1 there instead.
    linked = h.get("x-linked-etag", "").strip('"')
    if SHA256.match(linked):
        linked_size = int(h.get("x-linked-size", "-1"))
        if linked != sha256 or linked_size != size:
            return f"Hub reports sha256 {linked} size {linked_size}"
        return None
    if status not in (200, 301, 302, 307, 308):
        return f"HTTP {status}"
    if size > SMALL:
        return f"no X-Linked-Etag for a {size}-byte file (HTTP {status})"
    data = get(url)
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256 or len(data) != size:
        return f"downloaded sha256 {digest} size {len(data)}"
    return None


def commit_resolves(url: str, revision: str) -> str | None:
    if url.startswith(HF):
        repo = url[len(HF) :].strip("/")
        status, _ = head(f"{HF}api/models/{repo}/revision/{revision}")
    elif url.startswith("https://github.com/"):
        status, _ = head(f"{url.rstrip('/')}/commit/{revision}")
    else:
        return f"unsupported host {url}"
    return None if status == 200 else f"HTTP {status}"


def graph_sources(manifest: dict) -> tuple[dict[str, dict], dict | None]:
    graph = (manifest.get("execution") or {}).get("graph") or {}
    assets = list(graph.get("assets") or [])
    if graph.get("asset"):
        assets.append(graph["asset"])
    return {a["path"]: a for a in assets if a.get("source")}, graph.get("hosting")


def check(path: pathlib.Path) -> list[str]:
    m = yaml.safe_load(path.read_text(encoding="utf-8"))
    pid = m["id"]
    problems: list[str] = []
    ok: list[str] = []
    origin = m["origin"]
    for part in ("code", "weights"):
        rev = str(origin[part]["revision"])
        if not SHA1.match(rev):
            problems.append(f"origin.{part}.revision {rev} is not a full commit SHA")
        elif (why := commit_resolves(origin[part]["url"], rev)) is not None:
            problems.append(f"origin.{part} {origin[part]['url']}@{rev} does not resolve anonymously: {why}")
    sources, hosting = graph_sources(m)
    for a in sources.values():
        if a.get("source_status") != "public":
            problems.append(f"{a['path']}: source_status {a.get('source_status')}")
        if not re.search(r"/resolve/[0-9a-f]{40}/", a["source"]):
            problems.append(f"{a['path']}: source not at a full commit: {a['source']}")
    if hosting:
        if hosting.get("source_status") != "public" or not SHA1.match(str(hosting.get("revision", ""))):
            problems.append(f"hosting {hosting.get('repository')}: status {hosting.get('source_status')}, revision {hosting.get('revision')}")
    # A Starling-built graph with no public source (source_status: not-hosted) is recorded
    # with its digests and recipe, but has no URL to verify: skip it in the fetch below.
    graph = (m.get("execution") or {}).get("graph") or {}
    assets = list(graph.get("assets") or [])
    if graph.get("asset"):
        assets.append(graph["asset"])
    unhosted = {a["path"] for a in assets if a.get("source_status") == "not-hosted"}
    weights = origin["weights"]
    base = f"{weights['url'].rstrip('/')}/resolve/{weights['revision']}/"
    # The frozen base model the pipeline reads (origin.base, e.g. the Kev merges): files
    # under base/ are pinned from that repository at its own revision, not the weights
    # repository (the schema has one weights source, so the manifests spell them base/...).
    base_origin = origin.get("base")
    base_url: str | None = None
    base_rev: str | None = None
    if base_origin:
        base_url, base_rev = str(base_origin["url"]), str(base_origin["revision"])
        if not SHA1.match(base_rev):
            problems.append(f"origin.base.revision {base_rev} is not a full commit SHA")
        elif (why := commit_resolves(base_url, base_rev)) is not None:
            problems.append(f"origin.base {base_url}@{base_rev} does not resolve anonymously: {why}")
    base_prefix = "base/"
    checked = 0
    fetchable = [f for f in origin["files"] if f["path"] not in unhosted]
    for f in fetchable:
        fpath, sha, size = f["path"], str(f["sha256"]), int(f["size"])
        if not SHA256.match(sha):
            problems.append(f"{fpath}: sha256 {sha} is not a SHA-256")
            continue
        if fpath in sources:
            url = sources[fpath]["source"]
        else:
            # The pin lives wherever the anonymous Hub serves these exact bytes: the
            # upstream weights repository first, then the recorded Starling hosting
            # (which mirrors the upstream files and carries Starling's own builds —
            # the ortgenai-*/transformers-* bundles and the fp32 graphs upstream
            # never shipped). 404s and digest mismatches upstream therefore fall
            # through to the hosting URL; a file upstream serves with the pinned
            # bytes keeps the upstream URL even when the hosting copy differs.
            if fpath.startswith(base_prefix) and base_url and base_rev:
                url = f"{base_url.rstrip('/')}/resolve/{base_rev}/{fpath[len(base_prefix):]}"
            else:
                url = base + fpath
            if (why := verify_file(url, sha, size)) is not None:
                url = hosting["source_base"] + fpath if hosting else url
        if (why := verify_file(url, sha, size)) is not None:
            problems.append(f"{fpath}: {url}: {why}")
        else:
            checked += 1
    ok.append(f"{checked}/{len(fetchable)} lock files served anonymously with the pinned SHA-256 and size")
    for fpath in sorted(unhosted):
        ok.append(f"{fpath}: not-hosted (Starling-built, digests pinned; no public URL to verify)")
    rights = m.get("rights") or {}
    missing = [k for k in ("code_license", "weights_license", "tokenizer_terms", "redistribution") if not rights.get(k)]
    if missing:
        problems.append(f"rights missing: {', '.join(missing)}")
    else:
        ok.append(
            f"rights: code {rights['code_license']}, weights {rights['weights_license']}, tokenizer {rights['tokenizer_terms']}, "
            f"redistribution {rights['redistribution']}"
        )
    for line in ok:
        print(f"ok:   {pid}: {line}")
    for line in problems:
        print(f"FAIL: {pid}: {line}")
    return problems


def main() -> int:
    paths = [pathlib.Path(p) for p in sys.argv[1:]] or sorted((ROOT / "model-pipeline" / "registry").glob("*/*/manifest.yaml"))
    failed = 0
    for p in paths:
        failed += bool(check(p))
    print("PROVENANCE PASS" if failed == 0 else f"PROVENANCE FAIL ({failed} manifests)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
