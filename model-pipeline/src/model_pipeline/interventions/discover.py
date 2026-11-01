"""Propose residual-stream site tensors of a decoder graph, for a human to review (experimental).

Usage::

    model-pipeline discover-sites <graph.onnx> [--out proposal.json] [--layer-pattern REGEX] [--compare VERSION]
    python -m model_pipeline.interventions.discover <graph.onnx> ...

The command reads the graph structure only (``onnx.load_model_from_string``: external-data references stay references,
no weight is read) and writes a **proposal**: for each decoder layer, the one tensor that looks like the residual stream
after that layer, the nodes that read it, and the evidence for the choice. The proposal is never read at runtime. The
compiler only uses site mappings checked in by hand in ``injection_sites.py`` and keyed to the exact graph SHA-256; a
reviewer turns an accepted proposal into such a mapping (see ``docs/interventions/onboarding.md``). Nothing here loads a
proposal into the registry.

How a site is proposed (top-level nodes only; subgraph bodies are not searched):

1. A node belongs to decoder layer ``L`` when its name matches ``--layer-pattern`` (default ``(?:^|/)layers\\.(\\d+)/``,
   as in the LFM3 export's ``/model/layers.L/...``) with ``L`` as the first group.
2. A tensor *leaves* layer ``L`` when a node of layer ``L`` produces it and a node outside layer ``L`` reads it, or
   it is a graph output. The residual candidate of layer ``L`` is the one leaving tensor produced by an ``Add`` whose
   element type is float32 or float16 and that some node outside the layer reads. A layer index with no such tensor
   (for example a final norm the exporter numbered like a layer) is listed under ``unmatched_layers``; one with
   several is ambiguous and gets no site.
3. The proposed consumers are the readers in the next decoder layer, or, after the last decoder layer, every reader
   outside it. Any other reader (a debugging tap, a reader inside the layer itself) is listed under ``other_readers``
   and keeps the original value unless the reviewer adds it.
4. Evidence: the Add-only path inside the layer from the tensor that enters the layer to the candidate
   (``residual_path``, ``enters_from``). For every layer after the first, the entering tensor should be the previous
   layer's site; anything else is a warning.

``--compare VERSION`` diffs the proposal against a checked-in mapping (exit code 1 on a difference): a regenerated
proposal for a mapped graph must reproduce its mapping exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto

from .compiler import _site_types
from .contract import TARGET_RESIDUAL_POST
from .injection_sites import SITE_MAPPINGS, SiteMapping

PROPOSAL_KIND = "starling-intervention-site-mapping-proposal"
PROPOSAL_FORMAT_VERSION = 1
DISCOVER_VERSION = "0.1.0"
DEFAULT_LAYER_PATTERN = r"(?:^|/)layers\.(\d+)/"
PROPOSAL_STATUS = "unreviewed proposal; not a site mapping and never read at runtime"

_FLOAT_TYPES = {TensorProto.FLOAT: "float32", TensorProto.FLOAT16: "float16"}
_PRECISION_OF = {"float32": "fp32", "float16": "fp16"}

REVIEW_CHECKLIST = (
    "The graph SHA-256 is the one the model's asset lock records for this graph file.",
    "There is one site per decoder layer, numbered 0..N-1 with N the model's layer count; unmatched_layers holds only "
    "indices that are not decoder layers.",
    "Each site tensor is the residual stream after its layer: residual_path goes from the previous site through the "
    "layer's residual adds, and the element type and hidden size match the model config.",
    "consumers are every reader that must see the intervened value; other_readers is empty or each entry is meant to "
    "keep the original value. After the last layer, the consumer leads to the final norm and the LM head.",
    "No warnings remain unexplained.",
)


class SiteDiscoveryError(ValueError):
    """The graph cannot be read, or the layer pattern is unusable."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_info(name: str, types: Mapping[str, onnx.TypeProto]) -> tuple[str | None, list[int | str | None] | None]:
    value_type = types.get(name)
    if value_type is None or not value_type.HasField("tensor_type"):
        return None, None
    tensor_type = value_type.tensor_type
    elem = TensorProto.DataType.Name(tensor_type.elem_type).lower() if tensor_type.elem_type else None
    elem = _FLOAT_TYPES.get(tensor_type.elem_type, elem)
    if not tensor_type.HasField("shape"):
        return elem, None
    dims: list[int | str | None] = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return elem, dims


def _subgraph_inputs(node: onnx.NodeProto) -> set[str]:
    """Every tensor name read inside the subgraph attributes of ``node`` (recursively)."""
    names: set[str] = set()
    for attribute in node.attribute:
        graphs = list(attribute.graphs)
        if attribute.HasField("g"):
            graphs.append(attribute.g)
        for graph in graphs:
            for inner in graph.node:
                names.update(inner.input)
                names.update(_subgraph_inputs(inner))
    return names


def discover_sites(
    graph: str | Path,
    *,
    layer_pattern: str = DEFAULT_LAYER_PATTERN,
    graph_label: str | None = None,
) -> dict[str, Any]:
    """The proposal for ``graph`` (a ``model.onnx`` path) as a JSON-ready dict. Reads no external data."""
    try:
        pattern = re.compile(layer_pattern)
    except re.error as exc:
        raise SiteDiscoveryError(f"layer pattern {layer_pattern!r} is not a regular expression: {exc}") from None
    if pattern.groups < 1:
        raise SiteDiscoveryError(f"layer pattern {layer_pattern!r} needs a group that captures the layer index.")
    path = Path(graph)
    data = path.read_bytes()
    try:
        model = onnx.load_model_from_string(data)
    except Exception as exc:  # noqa: BLE001 - the protobuf parser raises several types
        raise SiteDiscoveryError(f"{path} is not an ONNX model: {exc}") from None
    g = model.graph
    types = _site_types(model)
    warnings: list[str] = []

    def layer_of(node: onnx.NodeProto) -> int | None:
        match = pattern.search(node.name) if node.name else None
        return int(match.group(1)) if match else None

    layers: dict[str, int | None] = {}
    producers: dict[str, onnx.NodeProto] = {}
    readers: dict[str, list[onnx.NodeProto]] = {}
    subgraph_reads: dict[str, list[str]] = {}
    unnamed = 0
    for node in g.node:
        if not node.name:
            unnamed += 1
        layers[node.name] = layer_of(node)
        for name in node.output:
            if name:
                producers[name] = node
        for name in dict.fromkeys(node.input):  # a node reading a tensor twice is one reader
            if name:
                readers.setdefault(name, []).append(node)
        inner = _subgraph_inputs(node)
        for name in inner:
            subgraph_reads.setdefault(name, []).append(node.name or f"<unnamed {node.op_type}>")
        if inner:
            warnings.append(
                f"node '{node.name}' ({node.op_type}) has a subgraph; tensors read inside subgraphs are not "
                "proposed as consumers (the compiler does not rewire subgraph readers)."
            )
    if unnamed:
        warnings.append(
            f"{unnamed} node(s) have no name; a site mapping names its consumers, so they cannot be listed."
        )
    outputs = {o.name for o in g.output}

    def node_layer(node: onnx.NodeProto) -> int | None:
        return layers[node.name] if node.name else layer_of(node)

    # Tensors leaving each layer.
    leaving: dict[int, list[dict[str, Any]]] = {}
    for node in g.node:
        layer = node_layer(node)
        if layer is None:
            continue
        leaving.setdefault(layer, [])
        for name in node.output:
            if not name:
                continue
            outside = [r for r in readers.get(name, []) if node_layer(r) != layer]
            if not outside and name not in outputs:
                continue
            dtype, shape = _tensor_info(name, types)
            leaving[layer].append(
                {
                    "tensor": name,
                    "producer": {"name": node.name, "op_type": node.op_type},
                    "dtype": dtype,
                    "shape": shape,
                    "outside_readers": [r.name for r in outside],
                    "graph_output": name in outputs,
                }
            )

    chosen: dict[int, dict[str, Any]] = {}
    unmatched: list[dict[str, Any]] = []
    for layer in sorted(leaving):
        candidates = [
            c
            for c in leaving[layer]
            if c["producer"]["op_type"] == "Add" and c["dtype"] in _PRECISION_OF and c["outside_readers"]
        ]
        crossing = [
            {k: c[k] for k in ("tensor", "dtype", "outside_readers", "graph_output")}
            | {"op_type": c["producer"]["op_type"]}
            for c in leaving[layer]
        ]
        if len(candidates) == 1:
            chosen[layer] = candidates[0]
        elif not candidates:
            unmatched.append(
                {
                    "layer": layer,
                    "reason": "no float32/float16 Add output leaves this layer for another node",
                    "leaving": crossing,
                }
            )
        else:
            unmatched.append({"layer": layer, "reason": "several residual candidates (ambiguous)", "leaving": crossing})
            warnings.append(
                f"layer {layer}: several Add outputs leave the layer ({', '.join(c['tensor'] for c in candidates)}); "
                "no site proposed."
            )

    decoder = sorted(chosen)
    if decoder and decoder != list(range(decoder[0], decoder[0] + len(decoder))):
        warnings.append(f"decoder layers with a site are not contiguous: {decoder}.")
    if decoder and decoder[0] != 0:
        warnings.append(f"the first decoder layer with a site is {decoder[0]}, not 0.")

    def enters(layer: int, tensor: str) -> tuple[list[str], list[str]]:
        """Walk back from ``tensor`` through Add nodes of ``layer``: the Add path and the tensors entering it."""
        path: list[str] = []
        entering: list[str] = []
        stack = [tensor]
        seen: set[str] = set()
        while stack:
            name = stack.pop()
            if name in seen:
                continue
            seen.add(name)
            node = producers.get(name)
            if node is None or node_layer(node) != layer:
                entering.append(name)
                continue
            if node.op_type != "Add":
                continue
            path.append(node.name)
            stack.extend(reversed([i for i in node.input if i]))
        return list(reversed(path)), entering

    sites: list[dict[str, Any]] = []
    hidden_sizes: set[int | None] = set()
    dtypes: set[str] = set()
    for index, layer in enumerate(decoder):
        c = chosen[layer]
        tensor = c["tensor"]
        last = index == len(decoder) - 1
        next_layer = None if last else decoder[index + 1]
        consumers: list[str] = []
        other: list[str] = []
        for reader in readers.get(tensor, []):
            reader_layer = node_layer(reader)
            wanted = (reader_layer != layer) if last else (reader_layer == next_layer)
            (consumers if wanted and reader.name else other).append(reader.name or f"<unnamed {reader.op_type}>")
        shape = c["shape"]
        hidden = shape[-1] if shape and isinstance(shape[-1], int) else None
        hidden_sizes.add(hidden)
        dtypes.add(c["dtype"])
        add_path, entering = enters(layer, tensor)
        site: dict[str, Any] = {
            "layer": layer,
            "target": TARGET_RESIDUAL_POST,
            "tensor": tensor,
            "hidden_size": hidden,
            "dtype": c["dtype"],
            "shape": shape,
            "consumers": consumers,
            "other_readers": other,
            "graph_output": c["graph_output"],
            "producer": c["producer"],
            "residual_path": add_path,
            "enters_from": entering,
        }
        sites.append(site)
        if hidden is None:
            warnings.append(f"layer {layer}: the last dimension of '{tensor}' is not known from the graph.")
        if other:
            warnings.append(f"layer {layer}: '{tensor}' has readers that are not proposed as consumers: {other}.")
        if c["graph_output"]:
            warnings.append(f"layer {layer}: '{tensor}' is a graph output; it would keep reporting the original value.")
        if tensor in subgraph_reads:
            warnings.append(f"layer {layer}: '{tensor}' is read inside subgraphs of {subgraph_reads[tensor]}.")
        if index > 0:
            previous = sites[index - 1]["tensor"]
            if previous not in entering:
                warnings.append(
                    f"layer {layer}: no Add-only path from the previous site '{previous}' to '{tensor}' "
                    f"(entering: {entering})."
                )
    if len(hidden_sizes) > 1:
        warnings.append(f"the sites do not share one hidden size: {sorted(hidden_sizes, key=str)}.")
    hidden_size = next(iter(hidden_sizes)) if len(hidden_sizes) == 1 else None
    precision = _PRECISION_OF[next(iter(dtypes))] if len(dtypes) == 1 else None
    if len(dtypes) > 1:
        warnings.append(f"the sites do not share one element type: {sorted(dtypes)}.")
    if not sites:
        warnings.append("no site was proposed; check --layer-pattern against the graph's node names.")

    default_opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), None)
    return {
        "kind": PROPOSAL_KIND,
        "format_version": PROPOSAL_FORMAT_VERSION,
        "status": PROPOSAL_STATUS,
        "generated_by": {"tool": "model_pipeline.interventions.discover", "version": DISCOVER_VERSION},
        "graph": {
            "file": graph_label if graph_label is not None else path.name,
            "sha256": _sha256(data),
            "size": len(data),
            "ir_version": model.ir_version,
            "default_opset": default_opset,
            "nodes": len(g.node),
        },
        "layer_pattern": layer_pattern,
        "proposed": {
            "target": TARGET_RESIDUAL_POST,
            "precision": precision,
            "hidden_size": hidden_size,
            "layers": len(sites),
        },
        "sites": sites,
        "unmatched_layers": unmatched,
        "warnings": warnings,
        "review_checklist": list(REVIEW_CHECKLIST),
    }


def compare_with_mapping(proposal: Mapping[str, Any], mapping: SiteMapping) -> list[str]:
    """The differences between a proposal and a checked-in mapping (empty when it reproduces it exactly)."""
    differences: list[str] = []
    graph_sha = proposal["graph"]["sha256"]
    if graph_sha != mapping.base_graph_sha256:
        differences.append(f"graph SHA-256 {graph_sha} != mapping's {mapping.base_graph_sha256}")
    precision = proposal["proposed"]["precision"]
    if precision != mapping.precision:
        differences.append(f"precision {precision} != mapping's {mapping.precision}")
    proposed = {(s["layer"], s["target"]): s for s in proposal["sites"]}
    mapped = {s.key: s for s in mapping.sites}
    for key in sorted(set(proposed) | set(mapped)):
        if key not in mapped:
            differences.append(f"site {key}: proposed, not in the mapping")
            continue
        if key not in proposed:
            differences.append(f"site {key}: in the mapping, not proposed")
            continue
        p, m = proposed[key], mapped[key]
        if p["tensor"] != m.tensor:
            differences.append(f"site {key}: tensor '{p['tensor']}' != '{m.tensor}'")
        if p["hidden_size"] != m.hidden_size:
            differences.append(f"site {key}: hidden size {p['hidden_size']} != {m.hidden_size}")
        if tuple(p["consumers"]) != m.consumers:
            differences.append(f"site {key}: consumers {p['consumers']} != {list(m.consumers)}")
    return differences


def proposal_json(proposal: Mapping[str, Any]) -> str:
    return json.dumps(proposal, indent=2, ensure_ascii=False) + "\n"


def _mapping_by_version(version: str) -> SiteMapping:
    for mapping in SITE_MAPPINGS.values():
        if mapping.version == version:
            return mapping
    known = ", ".join(sorted(m.version for m in SITE_MAPPINGS.values()))
    raise SiteDiscoveryError(f"no checked-in site mapping has version '{version}' (known: {known}).")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("graph", type=Path, help="the base graph (model.onnx); its external data is not read")
    parser.add_argument("--out", type=Path, help="write the proposal here (default: stdout)")
    parser.add_argument(
        "--layer-pattern",
        default=DEFAULT_LAYER_PATTERN,
        help=f"regular expression on node names; group 1 is the decoder layer index (default {DEFAULT_LAYER_PATTERN})",
    )
    parser.add_argument(
        "--compare",
        metavar="VERSION",
        help="diff the proposal against this checked-in site mapping version; exit code 1 on a difference",
    )


def run(args: argparse.Namespace) -> int:
    try:
        proposal = discover_sites(args.graph, layer_pattern=args.layer_pattern)
        mapping = _mapping_by_version(args.compare) if args.compare else None
    except (OSError, SiteDiscoveryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = proposal_json(proposal)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    summary = proposal["proposed"]
    print(
        f"{len(proposal['sites'])} site(s) proposed ({summary['precision']}, hidden {summary['hidden_size']}), "
        f"{len(proposal['warnings'])} warning(s); graph SHA-256 {proposal['graph']['sha256']}. "
        "Review before checking anything in.",
        file=sys.stderr,
    )
    registered = SITE_MAPPINGS.get(proposal["graph"]["sha256"])
    if registered is not None and mapping is None:
        print(f"note: this graph already has the checked-in mapping {registered.version}.", file=sys.stderr)
    if mapping is not None:
        differences = compare_with_mapping(proposal, mapping)
        for line in differences:
            print(f"differs from {mapping.version}: {line}", file=sys.stderr)
        if differences:
            return 1
        print(f"the proposal reproduces {mapping.version} exactly.", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="discover-sites", description=__doc__.split("\n", 1)[0])
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
