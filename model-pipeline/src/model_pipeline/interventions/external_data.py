"""External data of a base graph: enumerate, identify, and expose it to a derived graph without copying it.

The derived graph keeps every external-data reference of the base graph unchanged (location, offset, length). ONNX
Runtime 1.30 resolves a location against the directory of the graph it loads and rejects any location that leaves that
directory: parent-relative (``../``) and absolute locations fail, and so do symlinks (file or directory), because ORT
canonicalizes them before the check. A hard link placed next to the derived graph, under the original location, loads
and shares the original bytes, so no second copy of the weights exists. A hard link cannot cross file systems; that
fails with ``ExternalDataLinkError`` rather than silently copying.

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


def link_external_data(source_graph_dir: Path, derived_dir: Path, locations: Iterable[str]) -> list[Path]:
    """Hard-link each external file into ``derived_dir`` under the same relative location.

    An existing link to the same inode is kept; any other existing file is an error. Raises ``ExternalDataLinkError``
    when the link cannot be made (``EXDEV`` when the two directories are on different file systems).
    """
    links: list[Path] = []
    for location in locations:
        if not is_relative_path(location):
            raise InterventionContractError(f"'{location}' is not a plain relative external data location.")
        source = Path(source_graph_dir).joinpath(*location.split("/"))
        target = Path(derived_dir).joinpath(*location.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if not target.is_symlink() and os.path.samefile(source, target):
                links.append(target)
                continue
            raise ExternalDataLinkError(errno.EEXIST, f"'{target}' exists and is not a hard link to '{source}'.")
        try:
            os.link(source, target)
        except OSError as e:
            if e.errno == errno.EXDEV:
                raise ExternalDataLinkError(
                    errno.EXDEV,
                    f"cannot hard-link '{source}' into '{derived_dir}': they are on different file systems. Put the "
                    "derived artifacts on the same file system as the model bundle; the weights are never copied.",
                ) from e
            raise ExternalDataLinkError(e.errno, f"cannot hard-link '{source}' to '{target}': {e.strerror}") from e
        links.append(target)
    return links
