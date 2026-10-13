"""Protected evidence rules (plan 10.1 and 12, backlog P-007).

Two mechanisms, both deterministic:

1. The evaluator reads fixtures and tolerances only from `model-pipeline/protected/`, refuses any
   other source, and checks with Git that the files it read are unchanged against a base commit
   (by default the merge base with `origin/main`). A run on a tree whose protected files differ
   from the base is not gate evidence.
2. `check_changes` classifies the files a pull request changes. Changing protected evidence needs
   the `protected-change` label, and a change to protected evidence may not travel with
   implementation code, so an implementation job cannot move its own thresholds.

Neither replaces code-owner review: they make an unreviewed threshold change visible and failing.
"""

from __future__ import annotations

import fnmatch
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from model_pipeline import fixtures

PROTECTED_ROOT = PurePosixPath("model-pipeline/protected")
PROTECTED_LABEL = "protected-change"

# Paths whose changes need maintainer review as evidence (CODEOWNERS lists the same paths).
PROTECTED_PATTERNS = (
    "model-pipeline/protected/**",
    "**/tolerances.*",
    "tolerances.*",
)

# Implementation code. A pull request that changes protected evidence may not change these.
IMPLEMENTATION_PATTERNS = (
    "ai-dotnet/**",
    "model-pipeline/src/**",
    "model-pipeline/tests/**",
)


class ProtectedEvidenceError(Exception):
    """Fixtures or tolerances come from outside the protected tree, or differ from the base commit."""


def _matches(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(path, p) for p in patterns)


def is_protected(path: str) -> bool:
    return _matches(path, PROTECTED_PATTERNS)


def find_repository_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / PROTECTED_ROOT).is_dir():
            return candidate
    raise ProtectedEvidenceError(f"no repository root with {PROTECTED_ROOT} above {start}")


def resolve_protected(repo_root: Path, relative: str, *, kind: str, must_exist: bool = True) -> Path:
    """Resolve a repository-relative path that must lie inside `model-pipeline/protected/`.

    Rejects absolute paths, `..`, and symbolic links anywhere along the path, so a manifest
    cannot point the evaluator at a tolerance file outside review. With `must_exist=False` a
    missing final file is returned as is (the caller reports it); its directory must still exist.
    """
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ProtectedEvidenceError(f"{kind} path {relative!r} must be relative to the repository root, without '..'")
    if rel.parts[: len(PROTECTED_ROOT.parts)] != PROTECTED_ROOT.parts:
        raise ProtectedEvidenceError(
            f"{kind} path {relative!r} is outside {PROTECTED_ROOT}/; only protected evidence is read"
        )
    current = repo_root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ProtectedEvidenceError(f"{kind} path {relative!r} goes through a symbolic link at {current}")
    if not current.exists():
        if must_exist or not current.parent.is_dir():
            raise ProtectedEvidenceError(f"{kind} path {relative!r} does not exist")
        current = current.parent.resolve() / current.name
    protected = (repo_root / PROTECTED_ROOT).resolve()
    if not current.resolve().is_relative_to(protected):
        raise ProtectedEvidenceError(f"{kind} path {relative!r} resolves outside {PROTECTED_ROOT}/")
    return current


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=True).stdout


@dataclass
class Verification:
    """Outcome of the Git check of the protected files one evaluation read."""

    status: str  # "verified", "changed", or "unverified"
    base: str | None
    detail: str
    changed: list[str]
    digests: dict[str, str]

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "base": self.base,
            "detail": self.detail,
            "changed": self.changed,
            "sha256": self.digests,
        }


def resolve_base(repo_root: Path, base: str) -> tuple[str | None, str]:
    """`auto`: merge base of HEAD and origin/main (or main). `none`: skip. Anything else: a commit-ish."""
    if base == "none":
        return None, "verification disabled by --protected-base none"
    try:
        _git(repo_root, "rev-parse", "--git-dir")
    except (OSError, subprocess.CalledProcessError):
        return None, f"{repo_root} is not a Git work tree"
    if base != "auto":
        try:
            return _git(repo_root, "rev-parse", "--verify", f"{base}^{{commit}}").strip(), f"base {base}"
        except subprocess.CalledProcessError as exc:
            raise ProtectedEvidenceError(f"cannot resolve protected base {base!r}: {exc.stderr.strip()}") from exc
    for ref in ("origin/main", "main"):
        try:
            sha = _git(repo_root, "merge-base", "HEAD", ref).strip()
            return sha, f"merge base of HEAD and {ref}"
        except subprocess.CalledProcessError:
            continue
    return None, "no origin/main or main to compare against"


def verify_unchanged(repo_root: Path, paths: list[Path], base: str = "auto") -> Verification:
    """Check that `paths` (files or directories under the protected tree) equal their state at `base`.

    Compares the working tree, so uncommitted and untracked changes count as changes. A path that
    does not exist (a profile without a tolerances file) has no digest; if it exists at `base`, its
    deletion is reported as a change.
    """
    rels = sorted(str(p.resolve().relative_to(repo_root.resolve())).replace("\\", "/") for p in paths)
    digests = {}
    for p in paths:
        if not p.exists():
            continue
        files = sorted(f for f in p.rglob("*") if f.is_file()) if p.is_dir() else [p]
        for f in files:
            digests[str(f.resolve().relative_to(repo_root.resolve())).replace("\\", "/")] = fixtures.sha256_file(f)
    sha, detail = resolve_base(repo_root, base)
    if sha is None:
        return Verification("unverified", None, detail, [], digests)
    changed = set(_git(repo_root, "diff", "--name-only", sha, "--", *rels).splitlines())
    changed |= set(_git(repo_root, "ls-files", "--others", "--exclude-standard", "--", *rels).splitlines())
    changed.discard("")
    status = "changed" if changed else "verified"
    return Verification(status, sha, detail, sorted(changed), digests)


@dataclass
class ChangeCheck:
    protected: list[str]
    implementation: list[str]
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems


def check_changes(paths: list[str], labels: list[str]) -> ChangeCheck:
    """Rules for a pull request that changes `paths` and carries `labels`."""
    protected = sorted(p for p in paths if is_protected(p))
    implementation = sorted(p for p in paths if not is_protected(p) and _matches(p, IMPLEMENTATION_PATTERNS))
    problems = []
    if protected and PROTECTED_LABEL not in labels:
        problems.append(
            f"changes protected evidence without the '{PROTECTED_LABEL}' label (maintainer review, plan 10.1): "
            + ", ".join(protected)
        )
    if protected and implementation:
        problems.append(
            "changes protected evidence and implementation code in one pull request; propose the evidence change "
            "separately with its reference source (CLAUDE.md, plan 10.1). Implementation files: "
            + ", ".join(implementation)
        )
    return ChangeCheck(protected, implementation, problems)


def changed_files(repo_root: Path, base: str, head: str) -> list[str]:
    return [p for p in _git(repo_root, "diff", "--name-only", f"{base}...{head}").splitlines() if p]
