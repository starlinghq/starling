"""Capture mode: a derived graph that exposes the mapped site tensors as extra graph outputs (experimental).

Profile extraction (``extract``) needs the residual-stream values at candidate sites. A capture graph is the base graph
with the site mapping's tensors for the requested ``(layer, target)`` sites appended to ``graph.output``, and nothing
else: no node, initializer or input is added or changed, so every base output is computed exactly as before and each
exposed tensor is the value the base graph computes internally.

Capture graphs follow the derived-artifact rules of the intervention compiler and cache (``docs/interventions/``):

- the site mapping is looked up by the exact SHA-256 of the base graph; unknown graphs are unsupported;
- the base graph is parsed from its bytes with external data left as references; no weight is loaded or copied;
- the derived ``model.onnx`` reaches the original weights through hard links in its own directory (ONNX Runtime 1.30
  rejects parent-relative, absolute and symlinked external-data paths);
- entries are content-addressed, built in a private temporary directory under an exclusive lock, verified, published
  with one rename and verified on every open; the source bundle is never written.

They live in their own directory, ``<cache>/<model>/intervention-captures/<key>/``, with their own manifest
(``capture-manifest.json``) and key header, because they are not intervention artifacts: they have no profile and no
strength input, and the C# reader (which lists ``<cache>/<model>/interventions/``) must never see them. Capture graphs
are a model-pipeline tool; no .NET package loads them.
"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import onnx
from onnx import ModelProto

from . import external_data
from .cache import (
    LOCK_SUFFIX,
    TEMP_PREFIX,
    _check_model_name,
    _FileLock,
    _fsync_dir,
    _fsync_file,
    _remove_tree,
    _utf8_sorted,
)
from .compiler import (
    DERIVED_GRAPH_NAME,
    MIN_IR_VERSION,
    SUPPORTED_SITE_TYPES,
    InterventionCompileError,
    _inside,
    _site_types,
    default_opset,
)
from .contract import (
    ARTIFACT_ROLE_EXTERNAL_DATA,
    ARTIFACT_ROLE_GRAPH,
    TARGET_RESIDUAL_POST,
    TARGETS,
    ArtifactFile,
    BaseArtifact,
    InterventionContractError,
    InterventionMismatchError,
    is_relative_path,
    is_sha256,
    parse_strict_json,
    resolve_relative,
    sha256_bytes,
    sha256_file,
    verify_artifact_file,
)
from .injection_sites import MappedSite, SiteMapping, lookup_site_mapping
from .validate import DerivedGraphError, check_topological_order, checker_view

CAPTURE_VERSION = "0.1.0"
CAPTURE_SCHEMA_VERSION = 1
CAPTURE_KIND = "capture"
CAPTURE_MANIFEST_NAME = "capture-manifest.json"
CAPTURE_DIRECTORY = "intervention-captures"
CAPTURE_KEY_HEADER = "starling-intervention-capture-key 1"

Verification = Literal["fast", "full"]


class InterventionCaptureError(RuntimeError):
    """A capture entry is corrupt, incomplete or not what it claims to be; it is never silently replaced."""


@dataclass(frozen=True)
class CaptureOutput:
    """One exposed site: the graph output named ``tensor`` holds the residual value of ``(layer, target)``."""

    layer: int
    target: str
    tensor: str


@dataclass(frozen=True)
class CaptureManifest:
    capture_version: str
    site_mapping_version: str
    base_artifacts: tuple[BaseArtifact, ...]
    opset: int
    precision: str
    outputs: tuple[CaptureOutput, ...]
    derived_graph: ArtifactFile
    derived_files: tuple[ArtifactFile, ...]

    @property
    def base_graph(self) -> BaseArtifact:
        return next(a for a in self.base_artifacts if a.role == ARTIFACT_ROLE_GRAPH)

    def to_document(self) -> dict[str, Any]:
        def file(f: ArtifactFile) -> dict[str, Any]:
            return {"path": f.path, "sha256": f.sha256, "size": f.size}

        return {
            "schema_version": CAPTURE_SCHEMA_VERSION,
            "kind": CAPTURE_KIND,
            "capture_version": self.capture_version,
            "site_mapping_version": self.site_mapping_version,
            "base_artifacts": [
                {"role": a.role, "path": a.path, "sha256": a.sha256, "size": a.size} for a in self.base_artifacts
            ],
            "opset": self.opset,
            "precision": self.precision,
            "outputs": [{"layer": o.layer, "target": o.target, "tensor": o.tensor} for o in self.outputs],
            "derived_graph": file(self.derived_graph),
            "derived_files": [file(f) for f in self.derived_files],
        }

    def serialize(self) -> bytes:
        return (json.dumps(self.to_document(), indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def parse_capture_manifest(data: bytes) -> CaptureManifest:
    """Strict parse (unknown, missing or duplicate keys rejected), mirroring the compiler manifest's rules."""
    document = parse_strict_json(data, CAPTURE_MANIFEST_NAME)

    def obj(value: Any, where: str, keys: tuple[str, ...]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise InterventionContractError(f"{where}: expected an object.")
        if set(value) != set(keys):
            raise InterventionContractError(f"{where}: keys {sorted(value)}; expected {sorted(keys)}.")
        return value

    def text(value: Any, where: str) -> str:
        if not isinstance(value, str) or not value:
            raise InterventionContractError(f"{where}: expected a non-empty string.")
        return value

    def integer(value: Any, where: str, minimum: int = 0) -> int:
        if type(value) is not int or value < minimum:
            raise InterventionContractError(f"{where}: expected an integer >= {minimum}.")
        return value

    def sha(value: Any, where: str) -> str:
        if not is_sha256(value):
            raise InterventionContractError(f"{where}: expected a SHA-256.")
        return value

    def path(value: Any, where: str) -> str:
        if not is_relative_path(value):
            raise InterventionContractError(f"{where}: expected a relative path.")
        return value

    def file(value: Any, where: str) -> ArtifactFile:
        v = obj(value, where, ("path", "sha256", "size"))
        return ArtifactFile(
            path(v["path"], f"{where}.path"), sha(v["sha256"], f"{where}.sha256"), integer(v["size"], where)
        )

    keys = (
        "schema_version",
        "kind",
        "capture_version",
        "site_mapping_version",
        "base_artifacts",
        "opset",
        "precision",
        "outputs",
        "derived_graph",
        "derived_files",
    )
    if not isinstance(document, dict) or document.get("schema_version") != CAPTURE_SCHEMA_VERSION:
        raise InterventionContractError(f"{CAPTURE_MANIFEST_NAME}: schema_version must be {CAPTURE_SCHEMA_VERSION}.")
    d = obj(document, CAPTURE_MANIFEST_NAME, keys)
    if d["kind"] != CAPTURE_KIND:
        raise InterventionContractError(f"{CAPTURE_MANIFEST_NAME}: kind must be '{CAPTURE_KIND}'.")
    artifacts = []
    for i, a in enumerate(d["base_artifacts"] if isinstance(d["base_artifacts"], list) else []):
        a = obj(a, f"base_artifacts[{i}]", ("role", "path", "sha256", "size"))
        if a["role"] not in (ARTIFACT_ROLE_GRAPH, ARTIFACT_ROLE_EXTERNAL_DATA):
            raise InterventionContractError(f"base_artifacts[{i}].role: unknown role {a['role']!r}.")
        artifacts.append(
            BaseArtifact(a["role"], path(a["path"], "path"), sha(a["sha256"], "sha256"), integer(a["size"], "size"))
        )
    if [a.role for a in artifacts].count(ARTIFACT_ROLE_GRAPH) != 1:
        raise InterventionContractError("base_artifacts: exactly one entry must have role 'graph'.")
    outputs = []
    for i, o in enumerate(d["outputs"] if isinstance(d["outputs"], list) else []):
        o = obj(o, f"outputs[{i}]", ("layer", "target", "tensor"))
        if o["target"] not in TARGETS:
            raise InterventionContractError(f"outputs[{i}].target: unknown target {o['target']!r}.")
        outputs.append(CaptureOutput(integer(o["layer"], "layer"), o["target"], text(o["tensor"], "tensor")))
    if not outputs:
        raise InterventionContractError("outputs: at least one site is required.")
    return CaptureManifest(
        capture_version=text(d["capture_version"], "capture_version"),
        site_mapping_version=text(d["site_mapping_version"], "site_mapping_version"),
        base_artifacts=tuple(artifacts),
        opset=integer(d["opset"], "opset", 1),
        precision=text(d["precision"], "precision"),
        outputs=tuple(outputs),
        derived_graph=file(d["derived_graph"], "derived_graph"),
        derived_files=tuple(
            file(f, f"derived_files[{i}]")
            for i, f in enumerate(d["derived_files"] if isinstance(d["derived_files"], list) else [])
        ),
    )


# --------------------------------------------------------------------------------------------------------------------
# The capture graph


def _resolve_sites(mapping: SiteMapping, layers: Iterable[int], target: str) -> list[MappedSite]:
    layers = list(layers)
    if not layers:
        raise InterventionCompileError("capture needs at least one layer.")
    if len(set(layers)) != len(layers):
        raise InterventionCompileError(f"capture layers {layers} contain duplicates.")
    return [mapping.site(layer, target) for layer in sorted(layers)]


def build_capture_model(base: ModelProto, sites: Sequence[MappedSite]) -> ModelProto:
    """A copy of ``base`` whose outputs are the base outputs followed by each site tensor; ``base`` is not modified.

    A site tensor that is already a graph output is not added twice (it already carries the internal value).
    """
    types = _site_types(base)
    produced = {out for node in base.graph.node for out in node.output if out}
    graph_inputs = {i.name for i in base.graph.input}
    initializers = {t.name for t in base.graph.initializer}
    existing_outputs = {o.name for o in base.graph.output}
    model = ModelProto()
    model.CopyFrom(base)
    for site in sites:
        name = site.tensor
        if name in initializers:
            raise InterventionCompileError(f"site tensor '{name}' is an initializer, not an activation.")
        if name not in produced and name not in graph_inputs:
            raise InterventionCompileError(f"site tensor '{name}' does not exist in the base graph.")
        value_type = types.get(name)
        if value_type is None or not value_type.HasField("tensor_type"):
            raise InterventionCompileError(f"cannot determine the type of site tensor '{name}'.")
        tensor_type = value_type.tensor_type
        if tensor_type.elem_type not in SUPPORTED_SITE_TYPES:
            raise InterventionCompileError(f"site tensor '{name}' has an unsupported element type.")
        if tensor_type.HasField("shape") and tensor_type.shape.dim:
            last = tensor_type.shape.dim[-1]
            if last.HasField("dim_value") and last.dim_value != site.hidden_size:
                raise InterventionCompileError(
                    f"site tensor '{name}' has last dimension {last.dim_value}; the mapping says {site.hidden_size}."
                )
        if name in existing_outputs:
            continue
        output = onnx.ValueInfoProto()
        output.name = name
        output.type.CopyFrom(value_type)
        model.graph.output.append(output)
        existing_outputs.add(name)
    return model


def validate_capture_model(base: ModelProto, capture: ModelProto, sites: Sequence[MappedSite]) -> None:
    """``capture`` is ``base`` with the site tensors appended as outputs, and nothing else changed.

    - onnx.checker passes on the checker view (unless it already rejects the base graph, as for LFM3's ORT-only
      operators) and the top-level graph is topologically ordered;
    - inputs, nodes (in order), initializers, value_info, IR version and opset imports are byte-identical to the base;
    - outputs are the base outputs, unchanged and in order, followed by exactly the site tensors not already exposed.
    """

    def ser(items) -> list[bytes]:
        return [i.SerializeToString(deterministic=True) for i in items]

    try:
        onnx.checker.check_model(checker_view(capture))
    except onnx.checker.ValidationError as error:
        try:
            onnx.checker.check_model(checker_view(base))
        except onnx.checker.ValidationError:
            pass
        else:
            raise DerivedGraphError(f"onnx.checker rejects the capture graph but not its base: {error}") from error
    check_topological_order(capture)
    b, c = base.graph, capture.graph
    for what, left, right in (
        ("inputs", b.input, c.input),
        ("nodes", b.node, c.node),
        ("initializers", b.initializer, c.initializer),
        ("sparse initializers", b.sparse_initializer, c.sparse_initializer),
        ("value_info", b.value_info, c.value_info),
        ("opset imports", base.opset_import, capture.opset_import),
    ):
        if ser(left) != ser(right):
            raise DerivedGraphError(f"the capture graph changed the base graph's {what}.")
    if capture.ir_version != base.ir_version:
        raise DerivedGraphError("the capture graph changed the IR version.")
    base_outputs = ser(b.output)
    if ser(c.output)[: len(base_outputs)] != base_outputs:
        raise DerivedGraphError("the capture graph changed a base graph output.")
    already = {o.name for o in b.output}
    expected = [s.tensor for s in sites if s.tensor not in already]
    added = [o.name for o in c.output[len(base_outputs) :]]
    if added != expected:
        raise DerivedGraphError(f"the capture graph adds outputs {added}; expected exactly {expected}.")


@dataclass(frozen=True)
class CaptureResult:
    derived_dir: Path
    graph_path: Path
    manifest_path: Path | None
    manifest: CaptureManifest
    linked_files: tuple[Path, ...]


def _write_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def compile_capture(
    *,
    base_dir: str | Path,
    graph: str,
    layers: Iterable[int],
    out_dir: str | Path,
    target: str = TARGET_RESIDUAL_POST,
    mapping: SiteMapping | None = None,
    registry: Mapping[str, SiteMapping] | None = None,
    link_external: bool = True,
    write_manifest: bool = True,
) -> CaptureResult:
    """Write the capture graph of ``base_dir/graph`` for ``layers`` (at ``target``) to ``out_dir``.

    Same rules as ``compile_intervention``: ``graph`` is relative to ``base_dir``; the mapping is ``mapping`` or the
    registry entry for the graph's SHA-256 (none: ``UnsupportedGraphError``); ``out_dir`` and ``base_dir`` must be
    disjoint; external data is hard-linked into ``out_dir`` under its original locations and listed in
    ``derived_files``. Written in place; ``get_or_compile_capture`` publishes atomically.
    """
    base_dir = Path(base_dir).absolute()
    out_dir = Path(out_dir).absolute()
    if not is_relative_path(graph):
        raise InterventionContractError(f"'{graph}' is not a valid relative graph path.")
    if target not in TARGETS:
        raise InterventionContractError(f"unknown site target '{target}'.")
    graph_path = base_dir.joinpath(*graph.split("/"))
    if _inside(out_dir, base_dir):
        raise InterventionCompileError(f"output directory '{out_dir}' is inside the base model directory '{base_dir}'.")
    if _inside(base_dir, out_dir):
        raise InterventionCompileError(f"base model directory '{base_dir}' is inside the output directory '{out_dir}'.")

    graph_bytes = graph_path.read_bytes()
    graph_sha = sha256_bytes(graph_bytes)
    if mapping is None:
        mapping = lookup_site_mapping(graph_sha, registry)
    if mapping.base_graph_sha256 != graph_sha:
        raise InterventionMismatchError(
            f"site mapping {mapping.version} is for base graph {mapping.base_graph_sha256}, not {graph_sha}."
        )
    sites = _resolve_sites(mapping, layers, target)
    base = onnx.load_model_from_string(graph_bytes)
    if base.ir_version < MIN_IR_VERSION:
        raise InterventionCompileError(f"IR version {base.ir_version} is below the supported minimum.")
    opset = default_opset(base)
    locations = external_data.external_locations(base)
    reserved = {DERIVED_GRAPH_NAME, CAPTURE_MANIFEST_NAME}
    if any(location in reserved or location.split("/")[0] in reserved for location in locations):
        raise InterventionCompileError(f"an external data location collides with {sorted(reserved)}.")

    capture = build_capture_model(base, sites)
    validate_capture_model(base, capture, sites)

    out_dir.mkdir(parents=True, exist_ok=True)
    derived_path = out_dir / DERIVED_GRAPH_NAME
    derived_bytes = capture.SerializeToString(deterministic=True)
    _write_bytes(derived_path, derived_bytes)
    linked: list[Path] = []
    if link_external:
        linked = external_data.link_external_data(graph_path.parent, out_dir, locations)
    external_artifacts = external_data.base_external_artifacts(graph_path, base_dir, locations)
    for link in linked:
        source = graph_path.parent.joinpath(*link.relative_to(out_dir).parts)
        if not os.path.samefile(source, link):
            raise InterventionMismatchError(f"external data file '{source}' was replaced during compilation.")
    manifest = CaptureManifest(
        capture_version=CAPTURE_VERSION,
        site_mapping_version=mapping.version,
        base_artifacts=(BaseArtifact(ARTIFACT_ROLE_GRAPH, graph, graph_sha, len(graph_bytes)), *external_artifacts),
        opset=opset,
        precision=mapping.precision,
        outputs=tuple(CaptureOutput(s.layer, s.target, s.tensor) for s in sites),
        derived_graph=ArtifactFile(DERIVED_GRAPH_NAME, sha256_bytes(derived_bytes), len(derived_bytes)),
        derived_files=tuple(
            ArtifactFile(location, a.sha256, a.size) for location, a in zip(locations, external_artifacts, strict=True)
        ),
    )
    if sha256_file(graph_path) != graph_sha:
        raise InterventionMismatchError(f"base graph '{graph_path}' changed during compilation.")
    manifest_path = None
    if write_manifest:
        manifest_path = out_dir / CAPTURE_MANIFEST_NAME
        _write_bytes(manifest_path, manifest.serialize())
    return CaptureResult(out_dir, derived_path, manifest_path, manifest, tuple(linked))


# --------------------------------------------------------------------------------------------------------------------
# Cache


def capture_key_text(
    *,
    capture_version: str,
    site_mapping_version: str,
    opset: int,
    precision: str,
    base_artifacts: Iterable[BaseArtifact],
    outputs: Iterable[CaptureOutput],
) -> str:
    """Canonical text whose SHA-256 is the capture cache key (same line rules as the intervention cache key)."""
    lines = [
        CAPTURE_KEY_HEADER,
        f"capture_version\t{capture_version}",
        f"site_mapping_version\t{site_mapping_version}",
        f"opset\t{opset}",
        f"precision\t{precision}",
    ]
    for path, role, sha, size in _utf8_sorted((a.path, a.role, a.sha256, str(a.size)) for a in base_artifacts):
        lines.append(f"base\t{role}\t{path}\t{sha}\t{size}")
    for o in sorted(outputs, key=lambda o: (o.layer, o.target.encode("utf-8"))):
        lines.append(f"output\t{o.layer}\t{o.target}\t{o.tensor}")
    for line in lines:
        if "\r" in line or "\n" in line:
            raise InterventionContractError(f"capture key component contains a line break: {line!r}.")
    return "".join(line + "\n" for line in lines)


def manifest_capture_key(manifest: CaptureManifest) -> str:
    return sha256_bytes(
        capture_key_text(
            capture_version=manifest.capture_version,
            site_mapping_version=manifest.site_mapping_version,
            opset=manifest.opset,
            precision=manifest.precision,
            base_artifacts=manifest.base_artifacts,
            outputs=manifest.outputs,
        ).encode("utf-8")
    )


def captures_directory(cache_root: str | Path, model: str) -> Path:
    return Path(cache_root).absolute() / _check_model_name(model) / CAPTURE_DIRECTORY


@dataclass(frozen=True)
class CaptureEntry:
    directory: Path
    key: str
    manifest: CaptureManifest
    created: bool

    @property
    def graph_path(self) -> Path:
        return resolve_relative(self.directory, self.manifest.derived_graph.path)

    def output(self, layer: int, target: str = TARGET_RESIDUAL_POST) -> str:
        for o in self.manifest.outputs:
            if (o.layer, o.target) == (layer, target):
                return o.tensor
        raise KeyError((layer, target))


def open_capture(
    directory: str | Path,
    *,
    verify: Verification = "fast",
    base_dir: str | Path | None = None,
    expected_key: str | None = None,
) -> CaptureEntry:
    """Open and verify a capture entry; ``InterventionCaptureError`` when it does not hold.

    ``fast``: manifest parses; its recomputed key matches ``expected_key`` and a key-named directory; ``model.onnx`` has
    the recorded size and SHA-256; every derived file is a base external-data artifact's link with its SHA-256 and size,
    has that size, and (with ``base_dir``) is still the same inode as the base file. ``full`` also hashes the links and,
    with ``base_dir``, every base artifact.
    """
    directory = Path(directory).absolute()
    if verify not in ("fast", "full"):
        raise ValueError(f"verify must be 'fast' or 'full', not {verify!r}.")
    try:
        path = directory / CAPTURE_MANIFEST_NAME
        if not path.is_file():
            raise InterventionCaptureError(f"'{directory}' has no {CAPTURE_MANIFEST_NAME}; the entry is incomplete.")
        manifest = parse_capture_manifest(path.read_bytes())
        key = manifest_capture_key(manifest)
        if expected_key is not None and key != expected_key:
            raise InterventionCaptureError(f"'{directory}' holds capture key {key}, not {expected_key}.")
        if is_sha256(directory.name) and directory.name != key:
            raise InterventionCaptureError(f"'{directory}' is named {directory.name} but its manifest gives key {key}.")
        graph = manifest.derived_graph
        verify_artifact_file(resolve_relative(directory, graph.path), graph.sha256, graph.size)
        graph_dir = manifest.base_graph.path.rpartition("/")[0]
        external = {a.path: a for a in manifest.base_artifacts if a.role == ARTIFACT_ROLE_EXTERNAL_DATA}
        if len(external) != len(manifest.derived_files):
            raise InterventionMismatchError("derived files and base external data do not correspond one to one.")
        for file in manifest.derived_files:
            base_path = f"{graph_dir}/{file.path}" if graph_dir else file.path
            source = external.get(base_path)
            if source is None or (source.sha256, source.size) != (file.sha256, file.size):
                raise InterventionMismatchError(f"derived file '{file.path}' is not a link of a base artifact.")
            link = resolve_relative(directory, file.path)
            if verify == "full":
                verify_artifact_file(link, file.sha256, file.size)
            elif not link.is_file() or link.stat().st_size != file.size:
                raise InterventionMismatchError(f"'{link}' is missing or has the wrong size.")
            if base_dir is not None:
                original = resolve_relative(Path(base_dir), source.path)
                if not original.is_file() or not os.path.samefile(original, link):
                    raise InterventionMismatchError(f"'{link}' is no longer a hard link to '{original}'.")
        if verify == "full" and base_dir is not None:
            for artifact in manifest.base_artifacts:
                verify_artifact_file(resolve_relative(Path(base_dir), artifact.path), artifact.sha256, artifact.size)
    except InterventionCaptureError:
        raise
    except (InterventionContractError, InterventionMismatchError, OSError) as e:
        raise InterventionCaptureError(f"capture entry '{directory}' failed verification: {e}") from e
    return CaptureEntry(directory, key, manifest, created=False)


def get_or_compile_capture(
    *,
    cache_root: str | Path,
    model: str,
    base_dir: str | Path,
    graph: str,
    layers: Iterable[int],
    target: str = TARGET_RESIDUAL_POST,
    mapping: SiteMapping | None = None,
    registry: Mapping[str, SiteMapping] | None = None,
    external_sha256: Mapping[str, str] | None = None,
    verify: Verification = "fast",
    lock_timeout: float = 600.0,
) -> CaptureEntry:
    """The capture entry for ``layers`` of ``base_dir/graph``, compiled and published if absent (``get_or_compile``
    rules: key computed before compiling, exclusive lock, private temporary directory, one rename, verified open).

    ``external_sha256`` maps base-relative external data paths to trusted SHA-256 values (e.g. the asset lock's), so
    computing the key does not hash the weights; the compile hashes them anyway and fails on a wrong value.
    """
    base_dir = Path(base_dir).absolute()
    if not is_relative_path(graph):
        raise InterventionContractError(f"'{graph}' is not a valid relative graph path.")
    graph_path = base_dir.joinpath(*graph.split("/"))
    graph_bytes = graph_path.read_bytes()
    graph_sha = sha256_bytes(graph_bytes)
    if mapping is None:
        mapping = lookup_site_mapping(graph_sha, registry)
    sites = _resolve_sites(mapping, layers, target)
    base = onnx.load_model_from_string(graph_bytes)
    locations = external_data.external_locations(base)
    prefix = graph_path.parent.relative_to(base_dir).as_posix()
    artifacts = [BaseArtifact(ARTIFACT_ROLE_GRAPH, graph, graph_sha, len(graph_bytes))]
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
    key = sha256_bytes(
        capture_key_text(
            capture_version=CAPTURE_VERSION,
            site_mapping_version=mapping.version,
            opset=default_opset(base),
            precision=mapping.precision,
            base_artifacts=artifacts,
            outputs=[CaptureOutput(s.layer, s.target, s.tensor) for s in sites],
        ).encode("utf-8")
    )
    parent = captures_directory(cache_root, model)
    final = parent / key
    if final.exists():
        return open_capture(final, verify=verify, base_dir=base_dir, expected_key=key)
    with _FileLock(parent / f"{key}{LOCK_SUFFIX}", lock_timeout):
        if final.exists():
            return open_capture(final, verify=verify, base_dir=base_dir, expected_key=key)
        for stale in parent.glob(f"{TEMP_PREFIX}{key}-*"):
            _remove_tree(stale)
        temporary = parent / f"{TEMP_PREFIX}{key}-{os.getpid()}-{secrets.token_hex(4)}"
        temporary.mkdir(parents=True)
        try:
            result = compile_capture(
                base_dir=base_dir,
                graph=graph,
                layers=[s.layer for s in sites],
                target=target,
                out_dir=temporary,
                mapping=mapping,
            )
            actual = manifest_capture_key(result.manifest)
            if actual != key:
                raise InterventionMismatchError(
                    f"the compiled capture graph has key {actual}, expected {key}: an external_sha256 value does not "
                    "match the file, or the base files changed during compilation."
                )
            for path in (temporary / DERIVED_GRAPH_NAME, temporary / CAPTURE_MANIFEST_NAME):
                _fsync_file(path)
            open_capture(temporary, verify="fast", base_dir=base_dir, expected_key=key)
            _fsync_dir(temporary)
            os.rename(temporary, final)
            _fsync_dir(parent)
        except BaseException:
            _remove_tree(temporary)
            raise
    entry = open_capture(final, verify="fast", base_dir=base_dir, expected_key=key)
    return CaptureEntry(entry.directory, entry.key, entry.manifest, created=True)
