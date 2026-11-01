"""Compile a directional-projection intervention into a derived ONNX graph.

For every profile site the compiler inserts, at the base-graph tensor ``y`` the site mapping names::

    y' = y - alpha * ((y Q^T) * s) Q

as ``MatMul(y, Q^T)``, ``Mul(s, alpha)``, ``Mul(., s*alpha)``, ``MatMul(., Q)`` and ``Sub(y, .)``. The projection is
computed in float32 (the profile dtype); a float16 site gets ``Cast`` nodes and the float32 correction is cast back
before the ``Sub``, so with ``alpha = 0`` the site value is exactly ``y``. ``alpha`` is one float32 graph input of shape
``[1]`` shared by every site; ``Q^T``, ``Q`` and ``s`` are small inline initializers.

Only the consumers the site mapping lists are rewired to ``y'``; the inserted nodes read ``y`` itself and every other
reader (graph outputs included) keeps the original value. The base graph is read with its external-data references
left as they are: no weight is loaded, rewritten or copied, and the derived ``model.onnx`` carries the same location,
offset and length entries as the base graph. See ``external_data`` for how the derived graph reaches those files.
"""

from __future__ import annotations

import os
import secrets
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import ModelProto, TensorProto, helper, numpy_helper

from . import external_data
from .contract import (
    ARTIFACT_ROLE_GRAPH,
    DTYPE_FLOAT32,
    MAX_PROFILES,
    OPERATOR_DIRECTIONAL_ADDITION,
    OPERATOR_DIRECTIONAL_PROJECTION,
    OPERATORS_V2,
    ROLE_STRENGTH,
    ArtifactFile,
    BaseArtifact,
    FileIdentity,
    GraphInput,
    InterventionContractError,
    InterventionManifest,
    InterventionMismatchError,
    InterventionProfileFile,
    InterventionSite,
    ManifestProfile,
    ManifestValidation,
    MultiProfileManifest,
    MultiSiteBinding,
    SiteBinding,
    is_relative_path,
    is_tensor_name,
    load_profile,
    sha256_bytes,
    sha256_file,
)
from .injection_sites import MappedSite, SiteMapping, lookup_site_mapping
from .validate import validate_derived_model

COMPILER_VERSION = "0.1.0"
STRENGTH_INPUT_NAME = "intervention_strength"
STRENGTH_SHAPE = (1,)
DERIVED_GRAPH_NAME = "model.onnx"
MANIFEST_NAME = "compiler-manifest.json"
MIN_OPSET = 7  # multidirectional broadcasting in Mul and Sub
MIN_IR_VERSION = 4  # below IR 4 every initializer must also be a graph input, which would add inputs

# Element types a site tensor may have. The projection itself always runs in float32.
SUPPORTED_SITE_TYPES = {TensorProto.FLOAT: "float32", TensorProto.FLOAT16: "float16"}


class InterventionCompileError(ValueError):
    """The profile cannot be compiled into this graph (bad site, dtype, name collision, unsupported graph)."""


@dataclass(frozen=True)
class CompileResult:
    derived_dir: Path
    graph_path: Path
    manifest_path: Path | None
    manifest: InterventionManifest | MultiProfileManifest
    linked_files: tuple[Path, ...]


@dataclass(frozen=True)
class _PlannedSite:
    site: InterventionSite
    binding: MappedSite
    directions: np.ndarray
    elem_type: int


def default_opset(model: ModelProto) -> int:
    for entry in model.opset_import:
        if entry.domain in ("", "ai.onnx"):
            return entry.version
    raise InterventionCompileError("the graph imports no default-domain (ai.onnx) opset.")


def _all_names(model: ModelProto) -> set[str]:
    graph = model.graph
    names: set[str] = set()
    names.update(i.name for i in graph.input)
    names.update(o.name for o in graph.output)
    names.update(v.name for v in graph.value_info)
    names.update(t.name for t in graph.initializer)
    names.update(s.values.name for s in graph.sparse_initializer)
    for node in graph.node:
        names.add(node.name)
        names.update(node.output)
        names.update(node.input)
    names.discard("")
    return names


def _site_types(model: ModelProto) -> dict[str, onnx.TypeProto]:
    """Value types of the top-level tensors, from the graph and from (non-strict) shape inference."""
    types: dict[str, onnx.TypeProto] = {}
    try:
        inferred = onnx.shape_inference.infer_shapes(model, check_type=False, strict_mode=False, data_prop=False)
        graph = inferred.graph
    except Exception:  # noqa: BLE001 - inference is best effort; the explicit value_info below still counts
        graph = model.graph
    for value in (*graph.input, *graph.value_info, *graph.output, *model.graph.value_info, *model.graph.output):
        types.setdefault(value.name, value.type)
    for value in model.graph.input:
        types[value.name] = value.type
    return types


def _check_site_tensor(name: str, hidden_size: int, types: Mapping[str, onnx.TypeProto], site: InterventionSite) -> int:
    value_type = types.get(name)
    if value_type is None or not value_type.HasField("tensor_type"):
        raise InterventionCompileError(
            f"cannot determine the type of site tensor '{name}' (layer {site.layer}); "
            "the graph must declare or infer it."
        )
    tensor_type = value_type.tensor_type
    if tensor_type.elem_type not in SUPPORTED_SITE_TYPES:
        raise InterventionCompileError(
            f"site tensor '{name}' has element type {TensorProto.DataType.Name(tensor_type.elem_type)}; "
            f"supported: {sorted(SUPPORTED_SITE_TYPES.values())}."
        )
    if tensor_type.HasField("shape"):
        dims = tensor_type.shape.dim
        if len(dims) == 0:
            raise InterventionCompileError(
                f"site tensor '{name}' is a scalar; it needs a last dimension of hidden size."
            )
        last = dims[-1]
        if last.HasField("dim_value") and last.dim_value != hidden_size:
            raise InterventionCompileError(
                f"site tensor '{name}' has last dimension {last.dim_value}; the profile and mapping say {hidden_size}."
            )
    return tensor_type.elem_type


def _plan(model: ModelProto, profile: InterventionProfileFile, mapping: SiteMapping) -> list[_PlannedSite]:
    types = _site_types(model)
    produced = {out for node in model.graph.node for out in node.output if out}
    graph_inputs = {i.name for i in model.graph.input}
    initializers = {t.name for t in model.graph.initializer}
    planned: list[_PlannedSite] = []
    tensors: set[str] = set()
    for site in profile.profile.sites:
        binding = mapping.site(site.layer, site.target)
        if binding.hidden_size != site.hidden_size:
            raise InterventionCompileError(
                f"site (layer {site.layer}, {site.target}): the profile says hidden_size {site.hidden_size}, "
                f"site mapping {mapping.version} says {binding.hidden_size}."
            )
        if binding.tensor in tensors:
            raise InterventionCompileError(f"two sites resolve to tensor '{binding.tensor}'.")
        tensors.add(binding.tensor)
        if binding.tensor in initializers:
            raise InterventionCompileError(f"site tensor '{binding.tensor}' is an initializer, not an activation.")
        if binding.tensor not in produced and binding.tensor not in graph_inputs:
            raise InterventionCompileError(f"site tensor '{binding.tensor}' does not exist in the base graph.")
        elem_type = _check_site_tensor(binding.tensor, site.hidden_size, types, site)
        directions = profile.load_directions(site)
        if directions.shape != (site.rank, site.hidden_size):
            raise InterventionCompileError(f"directions for layer {site.layer} have shape {directions.shape}.")
        planned.append(_PlannedSite(site, binding, directions, elem_type))
    return planned


def _prefix(site: InterventionSite) -> str:
    return f"intervention/layer{site.layer}/{site.target}"


def _site_nodes(plan: _PlannedSite, strength: str) -> tuple[list[onnx.NodeProto], list[TensorProto], str]:
    """The nodes and initializers of one site, and the name of the intervened tensor."""
    p = _prefix(plan.site)
    y = plan.binding.tensor
    q = np.ascontiguousarray(plan.directions, dtype=np.float32)
    initializers = [
        numpy_helper.from_array(np.ascontiguousarray(q.T), f"{p}/directions_t"),
        numpy_helper.from_array(q, f"{p}/directions"),
        numpy_helper.from_array(np.asarray(plan.site.scales, dtype=np.float32), f"{p}/scales"),
    ]
    nodes: list[onnx.NodeProto] = []
    y32 = y
    if plan.elem_type != TensorProto.FLOAT:
        y32 = f"{p}/input_f32"
        nodes.append(helper.make_node("Cast", [y], [y32], name=f"{p}/Cast_input", to=TensorProto.FLOAT))
    nodes += [
        helper.make_node("MatMul", [y32, f"{p}/directions_t"], [f"{p}/coefficients"], name=f"{p}/MatMul_coefficients"),
        helper.make_node("Mul", [f"{p}/scales", strength], [f"{p}/scaled_strength"], name=f"{p}/Mul_strength"),
        helper.make_node(
            "Mul", [f"{p}/coefficients", f"{p}/scaled_strength"], [f"{p}/scaled_coefficients"], name=f"{p}/Mul_scales"
        ),
        helper.make_node(
            "MatMul", [f"{p}/scaled_coefficients", f"{p}/directions"], [f"{p}/delta"], name=f"{p}/MatMul_delta"
        ),
    ]
    delta = f"{p}/delta"
    if plan.elem_type != TensorProto.FLOAT:
        delta = f"{p}/delta_cast"
        nodes.append(helper.make_node("Cast", [f"{p}/delta"], [delta], name=f"{p}/Cast_delta", to=plan.elem_type))
    output = f"{p}/output"
    nodes.append(helper.make_node("Sub", [y, delta], [output], name=f"{p}/Sub"))
    return nodes, initializers, output


def build_derived_model(
    base: ModelProto,
    plans: list[_PlannedSite],
    strength_input: str = STRENGTH_INPUT_NAME,
) -> ModelProto:
    """A copy of ``base`` with the intervention inserted; ``base`` is not modified."""
    if not is_tensor_name(strength_input):
        raise InterventionCompileError(f"'{strength_input}' is not a valid input name.")
    model = ModelProto()
    model.CopyFrom(base)
    graph = model.graph
    existing = _all_names(model)
    if strength_input in existing:
        raise InterventionCompileError(f"the graph already has a value named '{strength_input}'.")

    by_tensor: dict[str, tuple[list[onnx.NodeProto], str, MappedSite]] = {}
    new_initializers: list[TensorProto] = []
    new_names: set[str] = {strength_input}
    for plan in plans:
        nodes, inits, output = _site_nodes(plan, strength_input)
        names = {n.name for n in nodes} | {o for n in nodes for o in n.output} | {t.name for t in inits}
        clash = (names & existing) | (names & new_names)
        if clash:
            raise InterventionCompileError(f"inserted names collide with existing graph names: {sorted(clash)}.")
        new_names |= names
        by_tensor[plan.binding.tensor] = (nodes, output, plan.binding)
        new_initializers += inits

    # Rewire the listed consumers (collected from the base nodes only, so the inserted nodes keep reading y).
    nodes_by_name: dict[str, onnx.NodeProto] = {}
    duplicated: set[str] = set()
    for node in graph.node:
        if node.name:
            if node.name in nodes_by_name:
                duplicated.add(node.name)
            nodes_by_name[node.name] = node
    for tensor, (_, output, binding) in by_tensor.items():
        for consumer in binding.consumers:
            if consumer in duplicated:
                raise InterventionCompileError(
                    f"the graph has several nodes named '{consumer}'; the consumer is ambiguous."
                )
            node = nodes_by_name.get(consumer)
            if node is None:
                raise InterventionCompileError(f"consumer node '{consumer}' of site tensor '{tensor}' does not exist.")
            hits = [i for i, name in enumerate(node.input) if name == tensor]
            if not hits:
                raise InterventionCompileError(f"node '{consumer}' does not consume site tensor '{tensor}'.")
            for i in hits:
                node.input[i] = output

    # Insert each site's nodes right after the producer of its tensor (or first, for a graph input).
    graph_inputs = {i.name for i in graph.input}
    ordered: list[onnx.NodeProto] = []
    for tensor in by_tensor:
        if tensor in graph_inputs:
            ordered += by_tensor[tensor][0]
    for node in graph.node:
        ordered.append(node)
        for out in node.output:
            if out in by_tensor:
                ordered += by_tensor[out][0]
    del graph.node[:]
    graph.node.extend(ordered)
    graph.initializer.extend(new_initializers)
    graph.input.append(helper.make_tensor_value_info(strength_input, TensorProto.FLOAT, list(STRENGTH_SHAPE)))
    return model


def _resolve_profile(profile: InterventionProfileFile | str | Path) -> InterventionProfileFile:
    return profile if isinstance(profile, InterventionProfileFile) else load_profile(profile)


def _inside(path: Path, root: Path) -> bool:
    path, root = path.resolve(), root.resolve()
    return path == root or root in path.parents


def compile_intervention(
    *,
    base_dir: str | Path,
    graph: str,
    profile: InterventionProfileFile | str | Path,
    out_dir: str | Path,
    mapping: SiteMapping | None = None,
    registry: Mapping[str, SiteMapping] | None = None,
    link_external: bool = True,
    write_manifest: bool = True,
    strength_input: str = STRENGTH_INPUT_NAME,
) -> CompileResult:
    """Compile ``profile`` into the base graph ``base_dir/graph`` and write the derived graph to ``out_dir``.

    ``graph`` is relative to ``base_dir`` (the base model directory the manifest's ``base_artifacts`` paths are
    relative to). The site mapping is ``mapping`` when given, else the registry entry for the graph's SHA-256; a hash
    with no mapping is unsupported. ``out_dir`` must not be inside ``base_dir``: the source bundle is never written.

    With ``link_external`` (the default) every external data file is hard-linked into ``out_dir`` under its original
    location, so ``out_dir/model.onnx`` loads by path in ONNX Runtime. The manifest lists those files in
    ``derived_files`` (path, SHA-256 and size of the bytes the derived graph reads) whether or not they were linked
    here, and for each file it linked also its identity on this machine (device, inode, mtime), which Fast
    verification compares. The derived graph, the links and the manifest are written in place; atomic publication
    into a cache is the caller's job. If the compile fails after it started writing, the files, links and directories
    this call created are removed again (a file it overwrote is not restored).

    External data on another file system than ``out_dir`` fails with ``ExternalDataLinkError`` (``EXDEV``) before
    anything is written: the weights are never copied.
    """
    base_dir = Path(base_dir).absolute()
    out_dir = Path(out_dir).absolute()
    if not is_relative_path(graph):
        raise InterventionContractError(f"'{graph}' is not a valid relative graph path.")
    graph_path = base_dir.joinpath(*graph.split("/"))
    if _inside(out_dir, base_dir):
        raise InterventionCompileError(f"output directory '{out_dir}' is inside the base model directory '{base_dir}'.")
    if _inside(base_dir, out_dir):
        # Links are created at out_dir/<location>, which could then land inside the source bundle.
        raise InterventionCompileError(f"base model directory '{base_dir}' is inside the output directory '{out_dir}'.")

    graph_bytes = graph_path.read_bytes()
    graph_sha = sha256_bytes(graph_bytes)
    profile_file = _resolve_profile(profile)
    profile_file.verify_base_model(graph_sha)
    if mapping is None:
        mapping = lookup_site_mapping(graph_sha, registry)
    if mapping.base_graph_sha256 != graph_sha:
        raise InterventionMismatchError(
            f"site mapping {mapping.version} is for base graph {mapping.base_graph_sha256}, not {graph_sha}."
        )
    if profile_file.profile.operator != OPERATOR_DIRECTIONAL_PROJECTION:
        raise InterventionCompileError(
            f"unsupported operator '{profile_file.profile.operator}' for a single-profile (schema 1) graph; compile it "
            "with compile_interventions([profile, ...])."
        )
    if any(s.dtype != DTYPE_FLOAT32 for s in profile_file.profile.sites):
        raise InterventionCompileError("only float32 profiles are supported.")

    # The serialized bytes only: external data stays referenced, never loaded.
    base = onnx.load_model_from_string(graph_bytes)
    opset = default_opset(base)
    if opset < MIN_OPSET:
        raise InterventionCompileError(f"default-domain opset {opset} is below the supported minimum {MIN_OPSET}.")
    if base.ir_version < MIN_IR_VERSION:
        raise InterventionCompileError(
            f"IR version {base.ir_version} is below the supported minimum {MIN_IR_VERSION} (the inserted initializers "
            "would have to become graph inputs)."
        )
    locations = external_data.external_locations(base)
    reserved = {DERIVED_GRAPH_NAME, MANIFEST_NAME}
    if any(location in reserved or location.split("/")[0] in reserved for location in locations):
        raise InterventionCompileError(
            f"an external data location collides with a derived file name {sorted(reserved)}."
        )

    plans = _plan(base, profile_file, mapping)
    derived = build_derived_model(base, plans, strength_input)
    inputs = (GraphInput(strength_input, ROLE_STRENGTH, DTYPE_FLOAT32, STRENGTH_SHAPE),)
    validation = validate_derived_model(base, derived, extra_inputs=[i.name for i in inputs])
    if link_external:
        external_data.check_same_file_system(graph_path.parent, out_dir, locations)

    created: list[Path] = [d for d in (out_dir, *out_dir.parents) if not d.exists()][::-1]
    try:
        return _write_outputs(
            out_dir=out_dir,
            created=created,
            derived=derived,
            graph=graph,
            graph_path=graph_path,
            graph_bytes=graph_bytes,
            graph_sha=graph_sha,
            base_dir=base_dir,
            locations=locations,
            link_external=link_external,
            write_manifest=write_manifest,
            profile_file=profile_file,
            mapping=mapping,
            opset=opset,
            inputs=inputs,
            plans=plans,
            validation=validation,
        )
    except BaseException:
        external_data.remove_created(created)
        raise


def _write_outputs(
    *,
    out_dir: Path,
    created: list[Path],
    derived: ModelProto,
    graph: str,
    graph_path: Path,
    graph_bytes: bytes,
    graph_sha: str,
    base_dir: Path,
    locations: list[str],
    link_external: bool,
    write_manifest: bool,
    profile_file: InterventionProfileFile,
    mapping: SiteMapping,
    opset: int,
    inputs: tuple[GraphInput, ...],
    plans: list[_PlannedSite],
    validation: ManifestValidation,
    build_manifest: Callable[[tuple[BaseArtifact, ...], ArtifactFile, tuple[ArtifactFile, ...]], object] | None = None,
) -> CompileResult:
    """Write the derived graph, links and manifest, appending every path it creates to ``created``.

    ``build_manifest(base_artifacts, derived_graph, derived_files)`` makes the manifest when given (schema 2); otherwise
    the schema 1 manifest of ``profile_file`` is written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    derived_path = out_dir / DERIVED_GRAPH_NAME
    derived_bytes = derived.SerializeToString(deterministic=True)
    _write_bytes(derived_path, derived_bytes)
    created.append(derived_path)
    linked: list[Path] = []
    if link_external:
        linked, made = external_data.link_external_data_tracked(graph_path.parent, out_dir, locations)
        created.extend(made)
        # Not checked by path with onnx.checker: onnx 1.23 refuses external data with several hard links ("potential
        # hardlink attack"), although ONNX Runtime 1.30 loads it. onnx.load(..., load_external_data=True) refuses it
        # too.

    # Taken before hashing and compared after it: a weights file rewritten in place while it was hashed would otherwise
    # be recorded with a digest of stale or torn bytes and the post-rewrite mtime, and pass Fast verification forever.
    before = {link.relative_to(out_dir).as_posix(): FileIdentity.of(link) for link in linked}
    external_artifacts = external_data.base_external_artifacts(graph_path, base_dir, locations)
    base_artifacts = [BaseArtifact(ARTIFACT_ROLE_GRAPH, graph, graph_sha, len(graph_bytes)), *external_artifacts]
    # The derived graph reads its external data from out_dir/<location>, so those files are derived files: recording
    # them lets VerifyDerivedFiles check the bytes ONNX Runtime will actually load (a swapped or stale file next to
    # model.onnx fails). They are listed even with link_external=False; the caller must then place exactly these files.
    # A link shares the original's inode, so its hash is the one just computed, provided it is still the same file.
    identities: dict[str, FileIdentity] = {}
    for link in linked:
        source = graph_path.parent.joinpath(*link.relative_to(out_dir).parts)
        if not os.path.samefile(source, link):
            raise InterventionMismatchError(f"external data file '{source}' was replaced during compilation.")
        location = link.relative_to(out_dir).as_posix()
        if FileIdentity.of(link) != before[location]:
            raise InterventionMismatchError(f"external data file '{source}' was modified during compilation.")
        identities[location] = before[location]
    derived_files = tuple(
        ArtifactFile(location, a.sha256, a.size, identities.get(location))
        for location, a in zip(locations, external_artifacts, strict=True)
    )
    derived_graph_file = ArtifactFile(DERIVED_GRAPH_NAME, sha256_bytes(derived_bytes), len(derived_bytes))
    if build_manifest is not None:
        manifest = build_manifest(tuple(base_artifacts), derived_graph_file, derived_files).validate()
    else:
        manifest = _manifest_v1(
            mapping, base_artifacts, profile_file, opset, inputs, plans, derived_graph_file, derived_files, validation
        )
    # The base graph must not have changed while we compiled.
    if sha256_file(graph_path) != graph_sha:
        raise InterventionMismatchError(f"base graph '{graph_path}' changed during compilation.")
    manifest_path = None
    if write_manifest:
        manifest_path = out_dir / MANIFEST_NAME
        _write_bytes(manifest_path, manifest.serialize())
        created.append(manifest_path)
    return CompileResult(out_dir, derived_path, manifest_path, manifest, tuple(linked))


def _manifest_v1(
    mapping, base_artifacts, profile_file, opset, inputs, plans, derived_graph_file, derived_files, validation
) -> InterventionManifest:
    return InterventionManifest(
        compiler_version=COMPILER_VERSION,
        site_mapping_version=mapping.version,
        base_artifacts=tuple(base_artifacts),
        profile_id=profile_file.profile.id,
        profile_sha256=profile_file.sha256,
        operator=profile_file.profile.operator,
        opset=opset,
        precision=mapping.precision,
        inputs=inputs,
        sites=tuple(SiteBinding(p.site.layer, p.site.target, p.binding.tensor) for p in plans),
        derived_graph=derived_graph_file,
        derived_files=derived_files,
        validation=validation,
    ).validate()


def _write_bytes(path: Path, data: bytes) -> None:
    # Unique per process, thread and call: two compiles into one directory never share a temporary file.
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}-{secrets.token_hex(4)}")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------------------------------------------------
# Several profiles, one strength vector (compiler manifest schema 2)


def _strength_element(index: int) -> str:
    return f"intervention/strength/p{index}"


def _strength_nodes(strength: str, count: int) -> tuple[list[onnx.NodeProto], list[TensorProto]]:
    """``Gather(strength, [i])`` for each profile ``i``: a ``[1]`` tensor, so every site sees the same shape as in a
    single-profile graph. Gather has the same signature at every opset the compiler accepts (Slice changed at 10)."""
    nodes, inits = [], []
    for i in range(count):
        name = _strength_element(i)
        inits.append(numpy_helper.from_array(np.asarray([i], dtype=np.int64), f"{name}/index"))
        nodes.append(helper.make_node("Gather", [strength, f"{name}/index"], [name], name=f"{name}/Gather", axis=0))
    return nodes, inits


def _operator_nodes(
    plan: _PlannedSite, operator: str, alpha: str, y: str, prefix: str
) -> tuple[list[onnx.NodeProto], list[TensorProto], str]:
    """One profile's operator at one site, reading ``y`` (the base tensor or the previous profile's output).

    ``directional_projection``: ``y - alpha * ((y Q^T) * s) Q`` (as in the single-profile graph).
    ``directional_addition``: ``y + alpha * (s Q)``, as ``Mul(s, alpha)``, ``MatMul(., Q)`` (a ``[hidden]`` vector) and
    ``Add(y, .)``; it does not depend on ``y``. A float16 site casts to float32 and back as the projection does.
    """
    p = prefix
    q = np.ascontiguousarray(plan.directions, dtype=np.float32)
    nodes: list[onnx.NodeProto] = []
    if operator == OPERATOR_DIRECTIONAL_PROJECTION:
        initializers = [
            numpy_helper.from_array(np.ascontiguousarray(q.T), f"{p}/directions_t"),
            numpy_helper.from_array(q, f"{p}/directions"),
            numpy_helper.from_array(np.asarray(plan.site.scales, dtype=np.float32), f"{p}/scales"),
        ]
        y32 = y
        if plan.elem_type != TensorProto.FLOAT:
            y32 = f"{p}/input_f32"
            nodes.append(helper.make_node("Cast", [y], [y32], name=f"{p}/Cast_input", to=TensorProto.FLOAT))
        nodes += [
            helper.make_node(
                "MatMul", [y32, f"{p}/directions_t"], [f"{p}/coefficients"], name=f"{p}/MatMul_coefficients"
            ),
            helper.make_node("Mul", [f"{p}/scales", alpha], [f"{p}/scaled_strength"], name=f"{p}/Mul_strength"),
            helper.make_node(
                "Mul",
                [f"{p}/coefficients", f"{p}/scaled_strength"],
                [f"{p}/scaled_coefficients"],
                name=f"{p}/Mul_scales",
            ),
            helper.make_node(
                "MatMul", [f"{p}/scaled_coefficients", f"{p}/directions"], [f"{p}/delta"], name=f"{p}/MatMul_delta"
            ),
        ]
        combine = "Sub"
    elif operator == OPERATOR_DIRECTIONAL_ADDITION:
        initializers = [
            numpy_helper.from_array(q, f"{p}/directions"),
            numpy_helper.from_array(np.asarray(plan.site.scales, dtype=np.float32), f"{p}/scales"),
        ]
        nodes += [
            helper.make_node("Mul", [f"{p}/scales", alpha], [f"{p}/scaled_strength"], name=f"{p}/Mul_strength"),
            helper.make_node(
                "MatMul", [f"{p}/scaled_strength", f"{p}/directions"], [f"{p}/delta"], name=f"{p}/MatMul_delta"
            ),
        ]
        combine = "Add"
    else:
        raise InterventionCompileError(f"unsupported operator '{operator}'.")
    delta = f"{p}/delta"
    if plan.elem_type != TensorProto.FLOAT:
        delta = f"{p}/delta_cast"
        nodes.append(helper.make_node("Cast", [f"{p}/delta"], [delta], name=f"{p}/Cast_delta", to=plan.elem_type))
    output = f"{p}/output"
    nodes.append(helper.make_node(combine, [y, delta], [output], name=f"{p}/{combine}"))
    return nodes, initializers, output


def build_multi_derived_model(
    base: ModelProto,
    plans_by_profile: Sequence[tuple[str, list[_PlannedSite]]],
    strength_input: str = STRENGTH_INPUT_NAME,
) -> ModelProto:
    """A copy of ``base`` with every profile inserted and one strength input of shape ``[len(plans_by_profile)]``.

    ``plans_by_profile`` is ``(operator, plans)`` per profile in strength-vector order. At a tensor several profiles
    share, they are applied in that order: profile ``i`` reads the output of the previous profile at that tensor (the
    base tensor for the first), and the listed consumers read the last output. ``base`` is not modified.
    """
    if not is_tensor_name(strength_input):
        raise InterventionCompileError(f"'{strength_input}' is not a valid input name.")
    model = ModelProto()
    model.CopyFrom(base)
    graph = model.graph
    existing = _all_names(model)
    if strength_input in existing:
        raise InterventionCompileError(f"the graph already has a value named '{strength_input}'.")

    new_names: set[str] = {strength_input}

    def claim(nodes: list[onnx.NodeProto], inits: list[TensorProto]) -> None:
        names = {n.name for n in nodes} | {o for n in nodes for o in n.output} | {t.name for t in inits}
        clash = (names & existing) | (names & new_names)
        if clash:
            raise InterventionCompileError(f"inserted names collide with existing graph names: {sorted(clash)}.")
        new_names.update(names)

    strength_nodes, new_initializers = _strength_nodes(strength_input, len(plans_by_profile))
    claim(strength_nodes, new_initializers)

    # Per tensor: the chain of inserted nodes, the current output, and the binding (same tensor, same consumers).
    chains: dict[str, tuple[list[onnx.NodeProto], str, MappedSite]] = {}
    for index, (operator, plans) in enumerate(plans_by_profile):
        for plan in plans:
            tensor = plan.binding.tensor
            nodes_so_far, current, binding = chains.get(tensor, ([], tensor, plan.binding))
            if binding.consumers != plan.binding.consumers:
                raise InterventionCompileError(f"site tensor '{tensor}' is bound with different consumer lists.")
            prefix = f"intervention/p{index}/layer{plan.site.layer}/{plan.site.target}"
            nodes, inits, output = _operator_nodes(plan, operator, _strength_element(index), current, prefix)
            claim(nodes, inits)
            new_initializers += inits
            chains[tensor] = (nodes_so_far + nodes, output, binding)

    nodes_by_name: dict[str, onnx.NodeProto] = {}
    duplicated: set[str] = set()
    for node in graph.node:
        if node.name:
            if node.name in nodes_by_name:
                duplicated.add(node.name)
            nodes_by_name[node.name] = node
    for tensor, (_, output, binding) in chains.items():
        for consumer in binding.consumers:
            if consumer in duplicated:
                raise InterventionCompileError(
                    f"the graph has several nodes named '{consumer}'; the consumer is ambiguous."
                )
            node = nodes_by_name.get(consumer)
            if node is None:
                raise InterventionCompileError(f"consumer node '{consumer}' of site tensor '{tensor}' does not exist.")
            hits = [i for i, name in enumerate(node.input) if name == tensor]
            if not hits:
                raise InterventionCompileError(f"node '{consumer}' does not consume site tensor '{tensor}'.")
            for i in hits:
                node.input[i] = output

    graph_inputs = {i.name for i in graph.input}
    ordered: list[onnx.NodeProto] = list(strength_nodes)
    for tensor in chains:
        if tensor in graph_inputs:
            ordered += chains[tensor][0]
    for node in graph.node:
        ordered.append(node)
        for out in node.output:
            if out in chains:
                ordered += chains[out][0]
    del graph.node[:]
    graph.node.extend(ordered)
    graph.initializer.extend(new_initializers)
    graph.input.append(helper.make_tensor_value_info(strength_input, TensorProto.FLOAT, [len(plans_by_profile)]))
    return model


def compile_interventions(
    *,
    base_dir: str | Path,
    graph: str,
    profiles: Sequence[InterventionProfileFile | str | Path],
    out_dir: str | Path,
    mapping: SiteMapping | None = None,
    registry: Mapping[str, SiteMapping] | None = None,
    link_external: bool = True,
    write_manifest: bool = True,
    strength_input: str = STRENGTH_INPUT_NAME,
) -> CompileResult:
    """Compile several profiles into one derived graph with a strength vector (compiler manifest schema 2).

    The derived graph takes one float32 input ``strength_input`` of shape ``[len(profiles)]``; element ``i`` is the
    strength of ``profiles[i]``, so profiles combine freely and 0 disables one. The order of ``profiles`` is the
    strength-vector order recorded in the manifest's ``profiles``, and the order in which profiles that share a site
    are applied there (``build_multi_derived_model``). Profile ids must be unique. Operators: ``directional_projection``
    and ``directional_addition``. Everything else (paths, linking, validation, cleanup, file-system rule) is as in
    ``compile_intervention``; one profile gives a schema 2 graph with a ``[1]`` strength vector.
    """
    base_dir = Path(base_dir).absolute()
    out_dir = Path(out_dir).absolute()
    if not is_relative_path(graph):
        raise InterventionContractError(f"'{graph}' is not a valid relative graph path.")
    graph_path = base_dir.joinpath(*graph.split("/"))
    if _inside(out_dir, base_dir):
        raise InterventionCompileError(f"output directory '{out_dir}' is inside the base model directory '{base_dir}'.")
    if _inside(base_dir, out_dir):
        raise InterventionCompileError(f"base model directory '{base_dir}' is inside the output directory '{out_dir}'.")
    profile_files = [_resolve_profile(p) for p in profiles]
    if not 1 <= len(profile_files) <= MAX_PROFILES:
        raise InterventionCompileError(f"expected 1 to {MAX_PROFILES} profiles, got {len(profile_files)}.")
    ids = [p.profile.id for p in profile_files]
    if len(set(ids)) != len(ids):
        raise InterventionCompileError(f"profile ids must be unique: {ids}.")

    graph_bytes = graph_path.read_bytes()
    graph_sha = sha256_bytes(graph_bytes)
    for profile_file in profile_files:
        profile_file.verify_base_model(graph_sha)
        if profile_file.profile.operator not in OPERATORS_V2:
            raise InterventionCompileError(f"unsupported operator '{profile_file.profile.operator}'.")
        if any(s.dtype != DTYPE_FLOAT32 for s in profile_file.profile.sites):
            raise InterventionCompileError("only float32 profiles are supported.")
    if mapping is None:
        mapping = lookup_site_mapping(graph_sha, registry)
    if mapping.base_graph_sha256 != graph_sha:
        raise InterventionMismatchError(
            f"site mapping {mapping.version} is for base graph {mapping.base_graph_sha256}, not {graph_sha}."
        )

    base = onnx.load_model_from_string(graph_bytes)
    opset = default_opset(base)
    if opset < MIN_OPSET:
        raise InterventionCompileError(f"default-domain opset {opset} is below the supported minimum {MIN_OPSET}.")
    if base.ir_version < MIN_IR_VERSION:
        raise InterventionCompileError(
            f"IR version {base.ir_version} is below the supported minimum {MIN_IR_VERSION} (the inserted initializers "
            "would have to become graph inputs)."
        )
    locations = external_data.external_locations(base)
    reserved = {DERIVED_GRAPH_NAME, MANIFEST_NAME}
    if any(location in reserved or location.split("/")[0] in reserved for location in locations):
        raise InterventionCompileError(
            f"an external data location collides with a derived file name {sorted(reserved)}."
        )

    plans_by_profile = [(p.profile.operator, _plan(base, p, mapping)) for p in profile_files]
    derived = build_multi_derived_model(base, plans_by_profile, strength_input)
    inputs = (GraphInput(strength_input, ROLE_STRENGTH, DTYPE_FLOAT32, (len(profile_files),)),)
    validation = validate_derived_model(base, derived, extra_inputs=[i.name for i in inputs])
    if link_external:
        external_data.check_same_file_system(graph_path.parent, out_dir, locations)

    bindings: dict[tuple[int, str], tuple[str, list[int]]] = {}
    for index, (_, plans) in enumerate(plans_by_profile):
        for plan in plans:
            bindings.setdefault(plan.site.key, (plan.binding.tensor, []))[1].append(index)

    def build_manifest(base_artifacts, derived_graph_file, derived_files) -> MultiProfileManifest:
        return MultiProfileManifest(
            compiler_version=COMPILER_VERSION,
            site_mapping_version=mapping.version,
            base_artifacts=base_artifacts,
            profiles=tuple(ManifestProfile(p.profile.id, p.sha256, p.profile.operator) for p in profile_files),
            opset=opset,
            precision=mapping.precision,
            inputs=inputs,
            sites=tuple(
                MultiSiteBinding(layer, target, tensor, tuple(indices))
                for (layer, target), (tensor, indices) in sorted(bindings.items())
            ),
            derived_graph=derived_graph_file,
            derived_files=derived_files,
            validation=validation,
        )

    created: list[Path] = [d for d in (out_dir, *out_dir.parents) if not d.exists()][::-1]
    try:
        return _write_outputs(
            out_dir=out_dir,
            created=created,
            derived=derived,
            graph=graph,
            graph_path=graph_path,
            graph_bytes=graph_bytes,
            graph_sha=graph_sha,
            base_dir=base_dir,
            locations=locations,
            link_external=link_external,
            write_manifest=write_manifest,
            profile_file=profile_files[0],
            mapping=mapping,
            opset=opset,
            inputs=inputs,
            plans=[],
            validation=validation,
            build_manifest=build_manifest,
        )
    except BaseException:
        external_data.remove_created(created)
        raise
