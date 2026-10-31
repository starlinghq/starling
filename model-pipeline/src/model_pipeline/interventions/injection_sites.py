"""Model-specific site mappings: where a profile site lives in one exact base graph.

A mapping is keyed to the SHA-256 of the base graph asset (the ``model.onnx`` bytes the model package's asset lock
records) and carries its own version. A graph hash with no mapping is **unsupported**: the compiler never searches or
pattern-matches an unknown graph for something that looks like a residual stream.

Each binding names the base-graph tensor of a ``(layer, target)`` site, its hidden size, and the consumer nodes that
must read the intervened value. Consumers not listed (for example a graph output or a debugging tap) keep reading the
original tensor, so only the intended downstream residual path is rewired.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .contract import (
    TARGETS,
    InterventionContractError,
    is_precision,
    is_sha256,
    is_tensor_name,
    is_version_label,
)


class UnsupportedGraphError(LookupError):
    """No site mapping exists for this base graph hash (or for a requested site)."""


@dataclass(frozen=True)
class MappedSite:
    """One site of one base graph.

    ``consumers`` are the names of the top-level nodes whose inputs equal to ``tensor`` are rewired to the intervened
    value; every listed node must consume ``tensor``. A node not listed keeps the original value.
    """

    layer: int
    target: str
    tensor: str
    hidden_size: int
    consumers: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.layer) is not int or self.layer < 0:
            raise InterventionContractError(f"site layer must be a non-negative integer, got {self.layer!r}.")
        if self.target not in TARGETS:
            raise InterventionContractError(f"unknown site target '{self.target}'.")
        if not is_tensor_name(self.tensor):
            raise InterventionContractError(f"'{self.tensor}' is not a valid tensor name.")
        if type(self.hidden_size) is not int or self.hidden_size < 1:
            raise InterventionContractError(f"hidden_size must be a positive integer, got {self.hidden_size!r}.")
        if not self.consumers:
            raise InterventionContractError(f"site ({self.layer}, {self.target}) lists no consumers to rewire.")
        if len(set(self.consumers)) != len(self.consumers) or not all(c for c in self.consumers):
            raise InterventionContractError(f"site ({self.layer}, {self.target}) has empty or duplicate consumers.")

    @property
    def key(self) -> tuple[int, str]:
        return (self.layer, self.target)


@dataclass(frozen=True)
class SiteMapping:
    """All supported sites of one exact base graph."""

    model: str
    version: str
    base_graph_sha256: str
    precision: str
    sites: tuple[MappedSite, ...]
    _by_key: dict[tuple[int, str], MappedSite] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not is_version_label(self.version):
            raise InterventionContractError(f"'{self.version}' is not a valid site-mapping version label.")
        if not is_sha256(self.base_graph_sha256):
            raise InterventionContractError(f"'{self.base_graph_sha256}' is not a SHA-256.")
        if not is_precision(self.precision):
            raise InterventionContractError(f"'{self.precision}' is not a valid precision.")
        by_key: dict[tuple[int, str], MappedSite] = {}
        tensors: set[str] = set()
        for site in self.sites:
            if site.key in by_key:
                raise InterventionContractError(f"site mapping {self.version} lists site {site.key} twice.")
            if site.tensor in tensors:
                raise InterventionContractError(f"site mapping {self.version} maps two sites to '{site.tensor}'.")
            by_key[site.key] = site
            tensors.add(site.tensor)
        object.__setattr__(self, "_by_key", by_key)

    def site(self, layer: int, target: str) -> MappedSite:
        try:
            return self._by_key[(layer, target)]
        except KeyError:
            raise UnsupportedGraphError(
                f"site mapping {self.version} for {self.model} has no site (layer {layer}, target '{target}')."
            ) from None


def build_registry(mappings: Iterable[SiteMapping]) -> dict[str, SiteMapping]:
    registry: dict[str, SiteMapping] = {}
    for mapping in mappings:
        if mapping.base_graph_sha256 in registry:
            raise InterventionContractError(f"two site mappings for base graph {mapping.base_graph_sha256}.")
        registry[mapping.base_graph_sha256] = mapping
    return registry


# Supported base graphs, keyed by the SHA-256 of the exact base graph asset. Empty until a model integration adds its
# mapping (LFM3 arrives with its integration); the compiler rejects every other graph as unsupported.
SITE_MAPPINGS: Mapping[str, SiteMapping] = build_registry(())


def lookup_site_mapping(base_graph_sha256: str, registry: Mapping[str, SiteMapping] | None = None) -> SiteMapping:
    """The mapping for this exact base graph hash; ``UnsupportedGraphError`` when there is none."""
    table = SITE_MAPPINGS if registry is None else registry
    mapping = table.get(base_graph_sha256)
    if mapping is None:
        raise UnsupportedGraphError(
            f"base graph {base_graph_sha256} has no intervention site mapping; it is unsupported. "
            "Site mappings are keyed to exact graph hashes and unknown graphs are never pattern-matched."
        )
    return mapping
