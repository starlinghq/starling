"""Upstream change detection (P-020).

Compares each registry manifest's pinned Hugging Face revisions (`origin.code`, `origin.weights`) with the
repository's current default-branch revision from the Hub API, and the tracked source with
`maintenance.latest_known_upstream_revision`. When a revision moved, the file listings at both revisions are
compared by Git blob id, the changed files are classified (card, tokenizer or template, configuration, weights),
and the gates a scoped revalidation must rerun are proposed.

The detector only reads. It never edits a manifest, never moves a pin and never changes a gate: a maintainer (or an
agent in a draft PR) records `latest_known_upstream_revision` and starts the update drill
(docs/plan/maintained-update-drill.md). Sources outside huggingface.co are reported as `unchecked`.

Network access is one injectable function, so tests and offline runs use recorded responses (`--responses`).
"""

from __future__ import annotations

import fnmatch
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from model_pipeline.manifest import find_manifests, load

HF_HOST = "huggingface.co"
DEFAULT_ENDPOINT = "https://huggingface.co"
CURRENT = "main"  # key for the default-branch listing in a responses file

# What a fetch returns: the Hub API model info (`sha`, `siblings` with `rfilename` and `blobId`).
Fetcher = Callable[[str, str | None], dict[str, Any]]

# Gates to rerun per kind of changed locked file (plan section 10). The design note explains the choices.
SCOPE = {
    "card": ("G0",),
    "tokenizer": ("G0", "G1", "G2", "G3", "G6"),
    "config": ("G0", "G1", "G2", "G3", "G6"),
    "weights": ("G0", "G2", "G3", "G4", "G5", "G6"),
}
ALL_GATES = tuple(f"G{i}" for i in range(8))

_TOKENIZER_PATTERNS = (
    "tokenizer*.json",
    "*.model",
    "special_tokens_map.json",
    "vocab*",
    "merges.txt",
    "added_tokens.json",
    "chat_template*",
    "preprocessor_config.json",
    "processor_config.json",
)
_WEIGHT_PATTERNS = ("*.safetensors", "*.onnx", "*.onnx_data", "*.onnx.data", "*.bin", "*.gguf", "*.pt", "*.pth")
_CARD_PATTERNS = ("README.md", "LICENSE*", "NOTICE*", "*.md")


class UpstreamError(Exception):
    """A fetch failed (network, HTTP status or missing recorded response)."""


def classify(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    for kind, patterns in (("tokenizer", _TOKENIZER_PATTERNS), ("weights", _WEIGHT_PATTERNS), ("card", _CARD_PATTERNS)):
        if any(fnmatch.fnmatchcase(name, p) for p in patterns):
            return kind
    return "config"


def hf_repo(url: str) -> str | None:
    """`owner/name` for a huggingface.co model URL, else None."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != HF_HOST:
        return None
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2 or parts[0] in ("datasets", "spaces", "api"):
        return None
    return f"{parts[0]}/{parts[1]}"


def http_fetcher(endpoint: str = DEFAULT_ENDPOINT, token: str | None = None, timeout: float = 30) -> Fetcher:
    """Fetch model info from the Hub API. The token is sent only to the given endpoint."""

    def fetch(repo: str, revision: str | None) -> dict[str, Any]:
        path = f"/api/models/{repo}" + (f"/revision/{revision}" if revision else "")
        request = urllib.request.Request(f"{endpoint}{path}?blobs=true")
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https endpoint
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise UpstreamError(f"{repo}@{revision or CURRENT}: HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise UpstreamError(f"{repo}@{revision or CURRENT}: {exc}") from exc

    return fetch


def responses_fetcher(data: dict[str, Any]) -> Fetcher:
    """Recorded responses: `{repo: {"main" | revision: model info, or {"error": message}}}`. No network."""

    def fetch(repo: str, revision: str | None) -> dict[str, Any]:
        entry = data.get(repo, {}).get(revision or CURRENT)
        if entry is None:
            raise UpstreamError(f"{repo}@{revision or CURRENT}: no recorded response")
        if "error" in entry:
            raise UpstreamError(f"{repo}@{revision or CURRENT}: {entry['error']}")
        return entry

    return fetch


def _cached(fetch: Fetcher) -> Fetcher:
    memo: dict[tuple[str, str | None], dict[str, Any] | UpstreamError] = {}

    def wrapper(repo: str, revision: str | None) -> dict[str, Any]:
        key = (repo, revision)
        if key not in memo:
            try:
                memo[key] = fetch(repo, revision)
            except UpstreamError as exc:
                memo[key] = exc
        value = memo[key]
        if isinstance(value, UpstreamError):
            raise value
        return value

    return wrapper


@dataclass
class SourceCheck:
    profile: str
    role: str  # code or weights
    url: str
    pinned: str
    tracked: bool  # the source maintenance.latest_*_upstream_revision refers to
    status: str = "unchecked"  # current, changed, unchecked, error
    upstream: str | None = None
    latest_known: str | None = None
    new_since_known: bool = False
    detail: str = ""
    changed_files: list[dict[str, Any]] = field(default_factory=list)
    gates: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _listing(info: dict[str, Any]) -> dict[str, str]:
    return {s["rfilename"]: s.get("blobId") or s.get("lfs", {}).get("sha256", "") for s in info.get("siblings", [])}


def _diff(before: dict[str, str], after: dict[str, str], locked: set[str]) -> list[dict[str, Any]]:
    changes = []
    for path in sorted(set(before) | set(after)):
        if path not in after:
            change = "removed"
        elif path not in before:
            change = "added"
        elif before[path] != after[path]:
            change = "modified"
        else:
            continue
        changes.append({"path": path, "change": change, "kind": classify(path), "locked": path in locked})
    return changes


def _scope(changes: list[dict[str, Any]], listings_known: bool) -> list[str]:
    if not listings_known:
        return list(ALL_GATES)  # nothing is known about what changed: rerun everything
    gates = {g for c in changes if c["locked"] for g in SCOPE[c["kind"]]}
    return [g for g in ALL_GATES if g in gates]


def check_manifest(manifest: dict[str, Any], fetch: Fetcher) -> list[SourceCheck]:
    origin = manifest["origin"]
    maintenance = manifest.get("maintenance", {})
    verified = maintenance.get("latest_verified_upstream_revision")
    known = maintenance.get("latest_known_upstream_revision")
    locked = {f["path"] for f in origin.get("files", [])}
    seen: set[tuple[str, str]] = set()
    checks = []
    for role in ("code", "weights"):
        source = origin[role]
        key = (source["url"], source["revision"])
        if key in seen:
            continue
        seen.add(key)
        tracked = source["revision"] == verified
        check = SourceCheck(
            profile=manifest["id"],
            role="code+weights" if origin["code"] == origin["weights"] else role,
            url=source["url"],
            pinned=source["revision"],
            tracked=tracked,
            latest_known=known if tracked else None,
        )
        checks.append(check)
        repo = hf_repo(source["url"])
        if repo is None:
            check.detail = "not a huggingface.co model repository; check by hand"
            continue
        try:
            current = fetch(repo, None)
        except UpstreamError as exc:
            check.status, check.detail = "error", str(exc)
            continue
        check.upstream = current.get("sha")
        if not check.upstream:
            check.status, check.detail = "error", f"{repo}: the Hub response has no sha"
            continue
        if check.upstream == check.pinned:
            check.status = "current"
        else:
            check.status = "changed"
            try:
                pinned_info = fetch(repo, check.pinned)
                check.changed_files = _diff(_listing(pinned_info), _listing(current), locked)
                listings_known = True
            except UpstreamError as exc:
                check.detail = f"file listing at the pinned revision unavailable ({exc}); scope is every gate"
                listings_known = False
            check.gates = _scope(check.changed_files, listings_known)
            if listings_known and not check.gates:
                check.detail = "no locked file changed; the pin can stay, record the revision as known"
        if tracked and check.status == "changed":
            check.new_since_known = check.upstream != known
    return checks


def check_registry(root: Path, fetch: Fetcher, today: str | None = None) -> dict[str, Any]:
    fetch = _cached(fetch)
    checks: list[SourceCheck] = []
    for path in find_manifests(root):
        manifest = load(path)
        if not isinstance(manifest, dict) or manifest.get("synthetic"):
            continue
        checks += check_manifest(manifest, fetch)
    return {
        "format": "starlings-upstream-check/1",
        "checked": today or date.today().isoformat(),
        "sources": [c.as_dict() for c in checks],
        "summary": {
            status: sum(c.status == status for c in checks) for status in ("current", "changed", "unchecked", "error")
        }
        | {"new_since_known": sum(c.new_since_known for c in checks)},
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Upstream check {report['checked']}",
        "",
        "Read-only report (P-020). No manifest, pin or gate was changed. See docs/plan/maintained-update-drill.md.",
        "",
        "| Profile | Source | Pinned | Upstream | Status | New since known | Gates to rerun |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in report["sources"]:
        lines.append(
            f"| {s['profile']} | {s['role']}{' (tracked)' if s['tracked'] else ''} | `{s['pinned'][:12]}` | "
            f"{'`' + s['upstream'][:12] + '`' if s['upstream'] else '-'} | {s['status']} | "
            f"{'yes' if s['new_since_known'] else 'no'} | {', '.join(s['gates']) or '-'} |"
        )
    details = [s for s in report["sources"] if s["detail"] or s["changed_files"]]
    if details:
        lines += ["", "## Details", ""]
        for s in details:
            lines.append(f"- **{s['profile']}** {s['role']} ({s['url']}): {s['detail'] or 'files changed'}")
            for c in s["changed_files"]:
                lock = "locked" if c["locked"] else "not in the lock"
                lines.append(f"  - `{c['path']}` {c['change']} ({c['kind']}, {lock})")
    proposals = [s for s in report["sources"] if s["new_since_known"]]
    if proposals:
        lines += ["", "## Proposed manifest updates (by pull request)", ""]
        for s in proposals:
            lines.append(
                f"- {s['profile']}: set `maintenance.latest_known_upstream_revision` to `{s['upstream']}` and "
                f"`last_checked` to {report['checked']}; keep the pin and `latest_verified_upstream_revision` "
                f"until a scoped revalidation of {', '.join(s['gates']) or 'no gate'} passes."
            )
    return "\n".join(lines) + "\n"


def default_fetcher(responses: Path | None) -> Fetcher:
    if responses is not None:
        return responses_fetcher(json.loads(responses.read_text(encoding="utf-8")))
    return http_fetcher(token=os.environ.get("HF_TOKEN") or None)
