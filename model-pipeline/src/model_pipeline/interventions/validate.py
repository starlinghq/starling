"""Structural and numeric checks of a derived intervention graph against its base graph."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import onnx
from onnx import ModelProto, TensorProto

from .contract import (
    CHECKER_PASSED,
    CHECKER_SKIPPED_BASE_REJECTED,
    MAX_VALIDATION_DETAIL,
    ManifestValidation,
    utf16_length,
)

logger = logging.getLogger(__name__)


class DerivedGraphError(ValueError):
    """The derived graph is not a faithful extension of the base graph."""


def check_topological_order(model: ModelProto) -> None:
    """Every node input of the top-level graph is defined before the node (no cycle, valid ONNX order)."""
    graph = model.graph
    defined = {i.name for i in graph.input} | {t.name for t in graph.initializer}
    defined |= {s.values.name for s in graph.sparse_initializer}
    defined.add("")
    for index, node in enumerate(graph.node):
        missing = [name for name in node.input if name not in defined]
        if missing:
            raise DerivedGraphError(
                f"node {index} '{node.name}' ({node.op_type}) reads {missing} before they are defined: "
                "the graph is not topologically ordered or has a cycle."
            )
        defined.update(node.output)


def checker_view(model: ModelProto) -> ModelProto:
    """A copy for onnx.checker without opening external files: top-level external initializers become graph inputs.

    ``onnx.checker.check_model`` on a proto resolves external locations against the current directory. This view keeps
    every node, type and shape the checker looks at. The external files themselves are checked by ONNX Runtime when it
    loads the derived graph (onnx 1.23's path checker refuses hard-linked external data).
    """
    view = ModelProto()
    view.CopyFrom(model)
    graph = view.graph
    inputs = {i.name for i in graph.input}
    kept = []
    for tensor in graph.initializer:
        if tensor.data_location != TensorProto.EXTERNAL:
            kept.append(tensor)
            continue
        if tensor.name not in inputs:
            graph.input.append(onnx.helper.make_tensor_value_info(tensor.name, tensor.data_type, list(tensor.dims)))
    del graph.initializer[:]
    graph.initializer.extend(kept)
    return view


def _tensor_identity(tensor: TensorProto) -> bytes:
    """The serialized tensor: for external tensors this includes location, offset and length, never the weights."""
    return tensor.SerializeToString(deterministic=True)


def _one_line(message: object) -> str:
    """First non-empty line of ``message``, without control characters, at most ``MAX_VALIDATION_DETAIL`` long."""
    lines = [line.strip() for line in str(message).splitlines() if line.strip()]
    text = "".join(c if c.isprintable() else " " for c in (lines[0] if lines else "")) or "no message"
    if utf16_length(text) <= MAX_VALIDATION_DETAIL:
        return text
    kept, used = [], 3  # room for "..."
    for c in text:
        used += utf16_length(c)
        if used > MAX_VALIDATION_DETAIL:
            break
        kept.append(c)
    return "".join(kept) + "..."


def validate_derived_model(base: ModelProto, derived: ModelProto, extra_inputs: Sequence[str]) -> ManifestValidation:
    """Check that ``derived`` is ``base`` plus the intervention and nothing else that matters.

    - onnx.checker passes (unless it already rejects the base graph, e.g. for operators only ONNX Runtime registers),
      and non-strict shape inference runs;
    - top-level nodes are topologically ordered (so the rewiring created no cycle);
    - the inputs are exactly the base inputs, in order, followed by ``extra_inputs``;
    - the outputs are the base outputs, unchanged;
    - every base initializer is present and identical (for external tensors: identical location, offset and length);
    - IR version, opset imports and the base nodes' operators are unchanged.

    Returns what onnx.checker said (recorded in the compiler manifest's ``validation``). When the checker is skipped
    because it rejects the base graph, a warning is logged on this module's logger.
    """
    view = checker_view(derived)
    checker = ManifestValidation(CHECKER_PASSED)
    try:
        onnx.checker.check_model(view)
    except onnx.checker.ValidationError as derived_error:
        # Graphs exported for ONNX Runtime can use operators ORT registers in the default domain that onnx does not
        # know (LFM3: SimplifiedLayerNormalization), so the checker rejects the base graph itself. Then the checker
        # cannot judge the derived graph either; the structural checks below and the ONNX Runtime load stand in for it.
        # A derived graph whose base passes the checker must pass it too.
        try:
            onnx.checker.check_model(checker_view(base))
        except onnx.checker.ValidationError as base_error:
            checker = ManifestValidation(CHECKER_SKIPPED_BASE_REJECTED, _one_line(base_error))
            logger.warning(
                "onnx.checker rejects the base graph (%s); the derived graph is checked structurally only (topological "
                "order, shape inference, inputs, outputs, initializers, nodes) and by the ONNX Runtime load.",
                checker.detail,
            )
        else:
            raise DerivedGraphError(f"onnx.checker rejects the derived graph but not its base: {derived_error}") from (
                derived_error
            )
    check_topological_order(derived)
    try:
        onnx.shape_inference.infer_shapes(view, check_type=True, strict_mode=False)
    except Exception as e:  # noqa: BLE001 - surface the inference failure as a derived-graph error
        raise DerivedGraphError(f"shape inference failed on the derived graph: {e}") from e

    base_inputs = [i.SerializeToString(deterministic=True) for i in base.graph.input]
    derived_inputs = [i.SerializeToString(deterministic=True) for i in derived.graph.input]
    if derived_inputs[: len(base_inputs)] != base_inputs:
        raise DerivedGraphError("the derived graph changed a base graph input.")
    added = [i.name for i in derived.graph.input[len(base_inputs) :]]
    if added != list(extra_inputs):
        raise DerivedGraphError(f"the derived graph adds inputs {added}; expected exactly {list(extra_inputs)}.")

    if [o.SerializeToString(deterministic=True) for o in base.graph.output] != [
        o.SerializeToString(deterministic=True) for o in derived.graph.output
    ]:
        raise DerivedGraphError("the derived graph changed the base graph outputs.")

    derived_initializers = {t.name: t for t in derived.graph.initializer}
    for tensor in base.graph.initializer:
        other = derived_initializers.get(tensor.name)
        if other is None or _tensor_identity(other) != _tensor_identity(tensor):
            raise DerivedGraphError(f"base initializer '{tensor.name}' is missing or changed in the derived graph.")

    if derived.ir_version != base.ir_version:
        raise DerivedGraphError("the derived graph changed the IR version.")
    if [(o.domain, o.version) for o in derived.opset_import] != [(o.domain, o.version) for o in base.opset_import]:
        raise DerivedGraphError("the derived graph changed the opset imports.")
    base_ops = [(n.name, n.op_type, n.domain, list(n.output)) for n in base.graph.node]
    derived_ops = {(n.name, n.op_type, n.domain, tuple(n.output)) for n in derived.graph.node}
    for name, op, domain, outputs in base_ops:
        if (name, op, domain, tuple(outputs)) not in derived_ops:
            raise DerivedGraphError(f"base node '{name}' ({op}) is missing or changed in the derived graph.")
    if len(derived.graph.node) < len(base.graph.node):
        raise DerivedGraphError("the derived graph has fewer nodes than the base graph.")
    return checker


def run_outputs(
    graph_path: str | Path,
    feeds: Mapping[str, np.ndarray],
    output_names: Iterable[str] | None = None,
) -> dict[str, np.ndarray]:
    """Run a graph on CPU with ONNX Runtime (default options, loaded by path) and return its outputs by name."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(graph_path), providers=["CPUExecutionProvider"])
    names = list(output_names) if output_names is not None else [o.name for o in session.get_outputs()]
    values = session.run(names, dict(feeds))
    return dict(zip(names, values, strict=True))


def max_abs_difference(
    base_path: str | Path,
    derived_path: str | Path,
    feeds: Mapping[str, np.ndarray],
    strength_input: str,
    strength: float = 0.0,
) -> dict[str, float]:
    """Per-output maximum absolute difference between the base graph and the derived graph at ``strength``."""
    base = run_outputs(base_path, feeds)
    derived = run_outputs(
        derived_path, {**feeds, strength_input: np.asarray([strength], dtype=np.float32)}, output_names=base.keys()
    )
    return {
        name: float(np.max(np.abs(base[name].astype(np.float64) - derived[name].astype(np.float64))))
        if base[name].size
        else 0.0
        for name in base
    }
