"""Derived-artifact cache: compile a profile into a base graph once, publish it atomically, reuse it afterwards.

Layout (``docs/interventions/cache.md``)::

    <cache_root>/<model>/interventions/
        <key>/                      one published entry, never modified after the rename that publishes it
            model.onnx              the derived graph
            <external locations>    hard links to the base graph's external data (no copy)
            profile.json            the compiled profile, byte for byte
            <directions files>      the profile's directions files, at their profile-relative paths
            compiler-manifest.json
        <key>.lock                  advisory lock taken by every compile of <key>
        .tmp-<key>-<pid>-<random>/  an entry being built (removed on failure, or by the next compile of <key>)

``<key>`` is the cache key: the SHA-256 of a canonical text over everything the derived graph depends on (base graph
and external weight identities, profile hash, compiler version, site-mapping version, opset, precision, declared
inputs). Every component is in the compiler manifest, so a reader recomputes the key from the manifest alone and checks
it against the directory name (C# ``InterventionCacheKey``). It is 64 hex characters, so the acquisition store, which
treats only 40-hex directories as revisions, never mistakes an entry for a snapshot.

The derived graph reaches the original weights through hard links (the measured ONNX Runtime 1.30 behaviour:
parent-relative, absolute and symlinked locations are rejected). A link cannot cross file systems: the cache must be on
the bundle's file system, and ``ExternalDataLinkError`` is raised otherwise. Weights are never copied and the source
bundle is never written.
"""

from __future__ import annotations

import errno
import fcntl
import os
import re
import secrets
import shutil
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import onnx

from . import external_data
from .compiler import (
    COMPILER_VERSION,
    DERIVED_GRAPH_NAME,
    MANIFEST_NAME,
    STRENGTH_INPUT_NAME,
    STRENGTH_SHAPE,
    InterventionCompileError,
    compile_intervention,
    default_opset,
)
from .contract import (
    ARTIFACT_ROLE_EXTERNAL_DATA,
    ARTIFACT_ROLE_GRAPH,
    DTYPE_FLOAT32,
    ROLE_STRENGTH,
    BaseArtifact,
    GraphInput,
    InterventionContractError,
    InterventionManifest,
    InterventionMismatchError,
    InterventionProfileFile,
    is_relative_path,
    is_sha256,
    load_manifest,
    load_profile,
    resolve_relative,
    sha256_bytes,
    sha256_file,
    verify_artifact_file,
    verify_manifest_profile,
)
from .injection_sites import SiteMapping, lookup_site_mapping

CACHE_DIRECTORY = "interventions"
PROFILE_NAME = "profile.json"
CACHE_KEY_HEADER = "starling-intervention-cache-key 1"
LOCK_SUFFIX = ".lock"
TEMP_PREFIX = ".tmp-"

_MODEL_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")

Verification = Literal["fast", "full"]


class InterventionCacheError(RuntimeError):
    """A cache entry is corrupt, incomplete or does not match what it claims to be; it is never silently replaced."""


@dataclass(frozen=True)
class CacheEntry:
    """A published (or just opened) derived-artifact directory."""

    directory: Path
    key: str
    manifest: InterventionManifest
    profile: InterventionProfileFile
    created: bool

    @property
    def graph_path(self) -> Path:
        return resolve_relative(self.directory, self.manifest.derived_graph.path)


# --------------------------------------------------------------------------------------------------------------------
# Cache key


def _utf8_sorted(items: Iterable[tuple[str, ...]]) -> list[tuple[str, ...]]:
    # Ordinal over UTF-8 bytes: the same order C# gets from comparing the encoded bytes.
    return sorted(items, key=lambda t: tuple(part.encode("utf-8") for part in t))


def cache_key_text(
    *,
    compiler_version: str,
    site_mapping_version: str,
    opset: int,
    precision: str,
    profile_sha256: str,
    base_artifacts: Iterable[BaseArtifact],
    inputs: Iterable[GraphInput],
) -> str:
    """The canonical text whose SHA-256 is the cache key (``docs/interventions/cache.md``, "Cache key").

    One field per line, fields separated by a tab, every line ending in LF. The contract forbids control characters in
    every component, so neither separator can occur inside a value. Base artifacts are sorted by path and inputs by
    name, both ordinally over UTF-8 bytes.
    """
    lines = [
        CACHE_KEY_HEADER,
        f"compiler_version\t{compiler_version}",
        f"site_mapping_version\t{site_mapping_version}",
        f"opset\t{opset}",
        f"precision\t{precision}",
        f"profile_sha256\t{profile_sha256}",
    ]
    for path, role, sha, size in _utf8_sorted((a.path, a.role, a.sha256, str(a.size)) for a in base_artifacts):
        lines.append(f"base\t{role}\t{path}\t{sha}\t{size}")
    for name, role, dtype, shape in _utf8_sorted(
        (i.name, i.role, i.dtype, ",".join(str(d) for d in i.shape)) for i in inputs
    ):
        lines.append(f"input\t{name}\t{role}\t{dtype}\t[{shape}]")
    for line in lines:
        if "\r" in line or "\n" in line:
            raise InterventionContractError(f"cache key component contains a line break: {line!r}.")
    return "".join(line + "\n" for line in lines)


def cache_key(**components) -> str:
    """SHA-256 of ``cache_key_text(**components)`` encoded as UTF-8."""
    return sha256_bytes(cache_key_text(**components).encode("utf-8"))


def manifest_cache_key(manifest: InterventionManifest) -> str:
    """The cache key recomputed from a compiler manifest (every component is recorded there)."""
    return cache_key(
        compiler_version=manifest.compiler_version,
        site_mapping_version=manifest.site_mapping_version,
        opset=manifest.opset,
        precision=manifest.precision,
        profile_sha256=manifest.profile_sha256,
        base_artifacts=manifest.base_artifacts,
        inputs=manifest.inputs,
    )


# --------------------------------------------------------------------------------------------------------------------
# Paths and locking


def _check_model_name(model: str) -> str:
    if not isinstance(model, str) or not _MODEL_NAME.fullmatch(model):
        raise InterventionContractError(f"'{model}' is not a cache model name ([a-z0-9][a-z0-9._-], at most 64).")
    return model


def interventions_directory(cache_root: str | Path, model: str) -> Path:
    return Path(cache_root).absolute() / _check_model_name(model) / CACHE_DIRECTORY


def entry_directory(cache_root: str | Path, model: str, key: str) -> Path:
    if not is_sha256(key):
        raise InterventionContractError(f"'{key}' is not a cache key (64 lower-case hex characters).")
    return interventions_directory(cache_root, model) / key


class _FileLock:
    """An exclusive ``flock`` on ``path``, polled until ``timeout`` seconds. POSIX only."""

    def __init__(self, path: Path, timeout: float) -> None:
        self._path = path
        self._timeout = timeout
        self._fd: int | None = None

    def __enter__(self) -> _FileLock:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as e:
                if e.errno not in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                    os.close(fd)
                    raise
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise TimeoutError(
                        f"timed out after {self._timeout}s waiting for the cache lock '{self._path}'."
                    ) from e
                time.sleep(0.05)
        self._fd = fd
        return self

    def __exit__(self, *exc) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _remove_tree(path: Path) -> None:
    # Removing a hard link only drops a name: the original weights file in the bundle is untouched.
    shutil.rmtree(path, ignore_errors=True)


def _check_disjoint(groups: Mapping[str, Iterable[str]]) -> None:
    """No two relative paths (from any group) are equal, and none is a directory prefix of another."""
    seen: dict[str, str] = {}
    for group, paths in groups.items():
        for path in paths:
            if path in seen and seen[path] != group:
                raise InterventionCompileError(f"'{path}' is both a {seen[path]} path and a {group} path.")
            seen.setdefault(path, group)
    names = sorted(seen)
    for a in names:
        for b in names:
            if a != b and b.startswith(a + "/"):
                raise InterventionCompileError(f"'{a}' ({seen[a]}) is a directory prefix of '{b}' ({seen[b]}).")


# --------------------------------------------------------------------------------------------------------------------
# Open and verify an entry


def open_entry(
    directory: str | Path,
    *,
    verify: Verification = "fast",
    base_dir: str | Path | None = None,
    expected_key: str | None = None,
) -> CacheEntry:
    """Open a derived-artifact directory and verify it; raise ``InterventionCacheError`` when it does not hold.

    ``fast`` (cache hits): the manifest parses; the key recomputed from it matches ``expected_key`` (when given) and,
    for a directory named like a key, the directory name; ``profile.json`` has the manifest's profile SHA-256 and
    matches it (``verify_manifest_profile``) and its directions verify; ``model.onnx`` has the recorded size and
    SHA-256; every other derived file has the recorded size and, when ``base_dir`` is given, is still the same inode as
    the base file of the same identity (a hard link). The weights are not hashed.

    ``full``: additionally hashes every derived file (the linked weights included) and, when ``base_dir`` is given,
    every base artifact under it.
    """
    directory = Path(directory).absolute()
    if verify not in ("fast", "full"):
        raise ValueError(f"verify must be 'fast' or 'full', not {verify!r}.")
    try:
        manifest_path = directory / MANIFEST_NAME
        if not manifest_path.is_file():
            raise InterventionCacheError(f"'{directory}' has no {MANIFEST_NAME}; the entry is incomplete.")
        manifest = load_manifest(manifest_path)
        key = manifest_cache_key(manifest)
        if expected_key is not None and key != expected_key:
            raise InterventionCacheError(f"'{directory}' holds the artifact of cache key {key}, not {expected_key}.")
        if is_sha256(directory.name) and directory.name != key:
            raise InterventionCacheError(f"'{directory}' is named {directory.name} but its manifest gives key {key}.")
        profile_path = directory / PROFILE_NAME
        if not profile_path.is_file():
            raise InterventionCacheError(f"'{directory}' has no {PROFILE_NAME}.")
        profile = load_profile(profile_path)
        verify_manifest_profile(manifest, profile)
        for site in profile.profile.sites:
            profile.load_directions(site)
        graph = manifest.derived_graph
        verify_artifact_file(resolve_relative(directory, graph.path), graph.sha256, graph.size)
        base_by_identity = {
            (a.sha256, a.size): a for a in manifest.base_artifacts if a.role == ARTIFACT_ROLE_EXTERNAL_DATA
        }
        for file in manifest.derived_files:
            path = resolve_relative(directory, file.path)
            if verify == "full":
                verify_artifact_file(path, file.sha256, file.size)
                continue
            if not path.is_file():
                raise InterventionMismatchError(f"'{path}' does not exist.")
            if path.stat().st_size != file.size:
                raise InterventionMismatchError(f"'{path}' is {path.stat().st_size} bytes; expected {file.size}.")
            source = base_by_identity.get((file.sha256, file.size))
            if base_dir is not None and source is not None:
                original = resolve_relative(Path(base_dir), source.path)
                if not original.is_file() or not os.path.samefile(original, path):
                    raise InterventionMismatchError(
                        f"'{path}' is no longer a hard link to '{original}' (the base file was replaced or the link "
                        "swapped); verify with verify='full' or evict the entry."
                    )
        if verify == "full" and base_dir is not None:
            for artifact in manifest.base_artifacts:
                verify_artifact_file(resolve_relative(Path(base_dir), artifact.path), artifact.sha256, artifact.size)
    except InterventionCacheError:
        raise
    except (InterventionContractError, InterventionMismatchError, OSError) as e:
        raise InterventionCacheError(f"cache entry '{directory}' failed verification: {e}") from e
    return CacheEntry(directory, key, manifest, profile, created=False)


# --------------------------------------------------------------------------------------------------------------------
# Compile through the cache


def _profile_files(profile: InterventionProfileFile) -> list[str]:
    return sorted({site.directions_file for site in profile.profile.sites})


def _expected_key(
    *,
    base_dir: Path,
    graph: str,
    graph_bytes: bytes,
    profile: InterventionProfileFile,
    mapping: SiteMapping,
    external_sha256: Mapping[str, str] | None,
    strength_input: str,
) -> tuple[str, list[str]]:
    base = onnx.load_model_from_string(graph_bytes)
    locations = external_data.external_locations(base)
    graph_path = base_dir.joinpath(*graph.split("/"))
    prefix = graph_path.parent.relative_to(base_dir).as_posix()
    artifacts = [BaseArtifact(ARTIFACT_ROLE_GRAPH, graph, sha256_bytes(graph_bytes), len(graph_bytes))]
    for location in locations:
        relative = location if prefix in ("", ".") else f"{prefix}/{location}"
        file = graph_path.parent.joinpath(*location.split("/"))
        if not file.is_file():
            raise InterventionContractError(f"external data file '{file}' does not exist.")
        known = (external_sha256 or {}).get(relative)
        if known is not None and not is_sha256(known):
            raise InterventionContractError(f"external_sha256['{relative}'] is not a SHA-256.")
        sha = known if known is not None else sha256_file(file)
        artifacts.append(BaseArtifact(ARTIFACT_ROLE_EXTERNAL_DATA, relative, sha, file.stat().st_size))
    key = cache_key(
        compiler_version=COMPILER_VERSION,
        site_mapping_version=mapping.version,
        opset=default_opset(base),
        precision=mapping.precision,
        profile_sha256=profile.sha256,
        base_artifacts=artifacts,
        inputs=[GraphInput(strength_input, ROLE_STRENGTH, DTYPE_FLOAT32, STRENGTH_SHAPE)],
    )
    return key, locations


def get_or_compile(
    *,
    cache_root: str | Path,
    model: str,
    base_dir: str | Path,
    graph: str,
    profile: InterventionProfileFile | str | Path,
    mapping: SiteMapping | None = None,
    registry: Mapping[str, SiteMapping] | None = None,
    external_sha256: Mapping[str, str] | None = None,
    verify: Verification = "fast",
    lock_timeout: float = 600.0,
    strength_input: str = STRENGTH_INPUT_NAME,
) -> CacheEntry:
    """Return the cache entry for ``profile`` compiled into ``base_dir/graph``, compiling and publishing it if absent.

    The key is computed before compiling, from the base graph bytes, the external data identities, the profile hash,
    the site mapping and the compiler version. ``external_sha256`` maps base-relative external data paths (as in the
    manifest's ``base_artifacts``, e.g. ``onnx/model.onnx_data``) to SHA-256 values the caller already trusts, for
    example from the model's asset lock; files it does not cover are hashed in full. A compile re-hashes every
    external file anyway, and fails if a supplied value was wrong.

    An existing entry is verified (``open_entry`` with ``verify``) and returned; one that fails verification raises
    ``InterventionCacheError`` and is left in place (``evict`` removes it). A missing entry is compiled into a private
    temporary directory beside it under an exclusive lock, verified, and published with one rename.
    """
    base_dir = Path(base_dir).absolute()
    if not is_relative_path(graph):
        raise InterventionContractError(f"'{graph}' is not a valid relative graph path.")
    profile_file = profile if isinstance(profile, InterventionProfileFile) else load_profile(profile)
    graph_bytes = base_dir.joinpath(*graph.split("/")).read_bytes()
    graph_sha = sha256_bytes(graph_bytes)
    profile_file.verify_base_model(graph_sha)
    if mapping is None:
        mapping = lookup_site_mapping(graph_sha, registry)
    key, locations = _expected_key(
        base_dir=base_dir,
        graph=graph,
        graph_bytes=graph_bytes,
        profile=profile_file,
        mapping=mapping,
        external_sha256=external_sha256,
        strength_input=strength_input,
    )
    parent = interventions_directory(cache_root, model)
    final = parent / key
    if final.exists():
        return open_entry(final, verify=verify, base_dir=base_dir, expected_key=key)

    with _FileLock(parent / f"{key}{LOCK_SUFFIX}", lock_timeout):
        if final.exists():  # published by another compile while we waited for the lock
            return open_entry(final, verify=verify, base_dir=base_dir, expected_key=key)
        # Every compile of this key holds the lock, so any temporary directory of this key left now is from a crash.
        for stale in parent.glob(f"{TEMP_PREFIX}{key}-*"):
            _remove_tree(stale)
        _check_disjoint(
            {
                "derived graph": [DERIVED_GRAPH_NAME, MANIFEST_NAME, PROFILE_NAME],
                "external data": locations,
                "profile": _profile_files(profile_file),
            }
        )
        temporary = parent / f"{TEMP_PREFIX}{key}-{os.getpid()}-{secrets.token_hex(4)}"
        temporary.mkdir(parents=True)
        try:
            result = compile_intervention(
                base_dir=base_dir,
                graph=graph,
                profile=profile_file,
                out_dir=temporary,
                mapping=mapping,
                strength_input=strength_input,
            )
            actual = manifest_cache_key(result.manifest)
            if actual != key:
                raise InterventionMismatchError(
                    f"the compiled artifact has cache key {actual}, expected {key}: an external_sha256 value does not "
                    "match the file, or the base files changed during compilation."
                )
            _copy_profile(profile_file, temporary)
            for path in (temporary / DERIVED_GRAPH_NAME, temporary / MANIFEST_NAME, temporary / PROFILE_NAME):
                _fsync_file(path)
            open_entry(temporary, verify="fast", base_dir=base_dir, expected_key=key)
            _fsync_dir(temporary)
            os.rename(temporary, final)
            _fsync_dir(parent)
        except BaseException:
            _remove_tree(temporary)
            raise
    entry = open_entry(final, verify="fast", base_dir=base_dir, expected_key=key)
    return CacheEntry(entry.directory, entry.key, entry.manifest, entry.profile, created=True)


def _copy_profile(profile: InterventionProfileFile, directory: Path) -> None:
    """Copy ``profile.json`` byte for byte (its hash is its identity) and its directions files at the same layout."""
    source = profile.path.read_bytes()
    if sha256_bytes(source) != profile.sha256:
        raise InterventionMismatchError(f"'{profile.path}' changed after it was loaded.")
    (directory / PROFILE_NAME).write_bytes(source)
    for relative in _profile_files(profile):
        target = resolve_relative(directory, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resolve_relative(profile.directory, relative), target)


def evict(cache_root: str | Path, model: str, key: str, *, lock_timeout: float = 600.0) -> bool:
    """Remove one entry (for example after ``InterventionCacheError``). Returns whether it existed.

    The entry is first renamed to a temporary name under the key's lock, so no reader sees a half-removed directory at
    the published path. The base bundle is not touched: removing a hard link only removes that name.
    """
    final = entry_directory(cache_root, model, key)
    parent = final.parent
    if not final.exists():
        return False
    with _FileLock(parent / f"{key}{LOCK_SUFFIX}", lock_timeout):
        if not final.exists():
            return False
        doomed = parent / f"{TEMP_PREFIX}{key}-evict-{os.getpid()}-{secrets.token_hex(4)}"
        os.rename(final, doomed)
        _remove_tree(doomed)
    return True
