"""External data of a base graph: enumerate, identify, and expose it to a derived graph without copying it.

The derived graph keeps every external-data reference of the base graph unchanged (location, offset, length). ONNX
Runtime 1.30 resolves a location against the directory of the graph it loads and rejects any location that leaves that
directory: parent-relative (``../``) and absolute locations fail, and so do symlinks (file or directory), because ORT
canonicalizes them before the check. A hard link placed next to the derived graph, under the original location, loads
and shares the original bytes, so no second copy of the weights exists. A hard link cannot cross file systems; that
fails with ``ExternalDataLinkError`` (with the remedy in the message) rather than copying the weights.

Original files are only ever opened for reading and hashing; nothing here writes to the base model directory.
"""

from __future__ import annotations

import errno
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import onnx
from onnx import GraphProto, ModelProto, TensorProto

from .contract import (
    ARTIFACT_ROLE_EXTERNAL_DATA,
    BaseArtifact,
    InterventionContractError,
    is_relative_path,
    sha256_file,
)


class ExternalDataLinkError(OSError):
    """The external data could not be hard-linked next to the derived graph (for example across file systems)."""


@dataclass(frozen=True)
class ExternalReference:
    """One external-data entry of one initializer, exactly as the base graph records it."""

    tensor: str
    location: str
    offset: int | None
    length: int | None


def _subgraphs(graph: GraphProto) -> Iterator[GraphProto]:
    yield graph
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.GRAPH:
                yield from _subgraphs(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for g in attribute.graphs:
                    yield from _subgraphs(g)


def _tensors(model: ModelProto) -> Iterator[TensorProto]:
    for graph in _subgraphs(model.graph):
        yield from graph.initializer
        for sparse in graph.sparse_initializer:
            yield sparse.values
            yield sparse.indices
        for node in graph.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    yield attribute.t
                elif attribute.type == onnx.AttributeProto.TENSORS:
                    yield from attribute.tensors


def external_references(model: ModelProto) -> list[ExternalReference]:
    """Every external-data reference in the model, including subgraphs and Constant attributes."""
    references: list[ExternalReference] = []
    for tensor in _tensors(model):
        if tensor.data_location != TensorProto.EXTERNAL:
            continue
        entries = {e.key: e.value for e in tensor.external_data}
        location = entries.get("location")
        if not location:
            raise InterventionContractError(f"initializer '{tensor.name}' is external but has no location.")
        offset = int(entries["offset"]) if "offset" in entries else None
        length = int(entries["length"]) if "length" in entries else None
        references.append(ExternalReference(tensor.name, location, offset, length))
    return references


def external_locations(model: ModelProto) -> list[str]:
    """The distinct external-data locations, in first-use order, each checked to stay inside the graph directory."""
    locations: list[str] = []
    for reference in external_references(model):
        if not is_relative_path(reference.location):
            raise InterventionContractError(
                f"external data location '{reference.location}' of '{reference.tensor}' is not a plain relative path "
                "inside the graph directory; ONNX Runtime would reject it and the compiler does not support it."
            )
        if reference.location not in locations:
            locations.append(reference.location)
    return locations


def _join(prefix: str, relative: str) -> str:
    return f"{prefix}/{relative}" if prefix else relative


def base_external_artifacts(graph_path: Path, base_dir: Path, locations: Iterable[str]) -> list[BaseArtifact]:
    """Manifest entries (role ``external_data``) for the external files, with paths relative to ``base_dir``.

    Each file is hashed in full: this reads the whole weights file once.
    """
    graph_path = Path(graph_path).absolute()
    graph_dir_rel = graph_path.parent.relative_to(Path(base_dir).absolute()).as_posix()
    graph_dir_rel = "" if graph_dir_rel == "." else graph_dir_rel
    artifacts: list[BaseArtifact] = []
    for location in locations:
        file = graph_path.parent.joinpath(*location.split("/"))
        if not file.is_file():
            raise InterventionContractError(f"external data file '{file}' does not exist.")
        artifacts.append(
            BaseArtifact(
                ARTIFACT_ROLE_EXTERNAL_DATA,
                _join(graph_dir_rel, location),
                sha256_file(file),
                file.stat().st_size,
            )
        )
    return artifacts


def _mount_point(path: Path) -> Path:
    path = Path(path).absolute()
    while not os.path.ismount(path) and path.parent != path:
        path = path.parent
    return path


def _existing_ancestor(path: Path) -> Path:
    path = Path(path).absolute()
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def _cross_device_error(source: Path, derived_dir: Path) -> ExternalDataLinkError:
    target_fs = _existing_ancestor(derived_dir)
    return ExternalDataLinkError(
        errno.EXDEV,
        f"cannot hard-link the external data '{source}' into '{derived_dir}': they are on different file systems "
        f"(the weights are on the file system mounted at '{_mount_point(source)}', the output on the one mounted at "
        f"'{_mount_point(target_fs)}'). A derived graph reaches the weights only through a hard link, and the weights "
        "are never copied. Put the output directory (for get_or_compile, cache_root) on the model bundle's file "
        f"system, for example under '{_mount_point(source)}', or acquire the bundle onto the cache's file system.",
    )


def check_same_file_system(source_graph_dir: Path, derived_dir: Path, locations: Iterable[str]) -> None:
    """Fail early, before anything is written, when an external file is on another device than ``derived_dir``.

    Compares ``st_dev`` of each external file with the nearest existing ancestor of ``derived_dir``. Equal devices do
    not guarantee a link succeeds (two bind mounts of one file system share a device and still give ``EXDEV``), so
    ``link_external_data`` handles ``EXDEV`` as well.
    """
    target_device = os.stat(_existing_ancestor(derived_dir)).st_dev
    for location in locations:
        source = Path(source_graph_dir).joinpath(*location.split("/"))
        if os.stat(source).st_dev != target_device:
            raise _cross_device_error(source, Path(derived_dir))


def link_external_data(source_graph_dir: Path, derived_dir: Path, locations: Iterable[str]) -> list[Path]:
    """Hard-link each external file into ``derived_dir`` under the same relative location.

    An existing link to the same inode is kept; any other existing file is an error. Raises ``ExternalDataLinkError``
    when the link cannot be made (``EXDEV`` when the two directories are on different file systems, with the remedy in
    the message). On failure the links this call created are removed again.
    """
    links, _ = link_external_data_tracked(source_graph_dir, derived_dir, locations)
    return links


def link_external_data_tracked(
    source_graph_dir: Path, derived_dir: Path, locations: Iterable[str]
) -> tuple[list[Path], list[Path]]:
    """``link_external_data``, also returning the paths this call created (links, then directories it made), so a
    caller that fails later can remove exactly those."""
    links: list[Path] = []
    created: list[Path] = []
    try:
        for location in locations:
            if not is_relative_path(location):
                raise InterventionContractError(f"'{location}' is not a plain relative external data location.")
            source = Path(source_graph_dir).joinpath(*location.split("/"))
            target = Path(derived_dir).joinpath(*location.split("/"))
            missing = [d for d in (target.parent, *target.parent.parents) if not d.exists()]
            target.parent.mkdir(parents=True, exist_ok=True)
            created.extend(sorted(missing, key=lambda d: len(d.parts)))
            if target.exists() or target.is_symlink():
                if not target.is_symlink() and os.path.samefile(source, target):
                    links.append(target)
                    continue
                raise ExternalDataLinkError(errno.EEXIST, f"'{target}' exists and is not a hard link to '{source}'.")
            try:
                os.link(source, target)
            except OSError as e:
                if e.errno == errno.EXDEV:
                    raise _cross_device_error(source, Path(derived_dir)) from e
                raise ExternalDataLinkError(e.errno, f"cannot hard-link '{source}' to '{target}': {e.strerror}") from e
            links.append(target)
            created.append(target)
    except BaseException:
        remove_created(created)
        raise
    return links, created


def remove_created(paths: Iterable[Path]) -> None:
    """Remove files and (empty) directories a failed write created, newest first. Removing a hard link only drops a
    name: the original weights file is untouched. Errors are ignored; a leftover is reported by verification."""
    for path in reversed(list(paths)):
        try:
            if path.is_dir() and not path.is_symlink():
                path.rmdir()
            else:
                path.unlink()
        except OSError:
            pass
