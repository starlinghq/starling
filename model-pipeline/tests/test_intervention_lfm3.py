"""The LFM3-1.7B fp32 site mapping, and (with the real snapshot) the compiled graph on ONNX Runtime.

The mapping tests always run. The graph tests need ``STARLING_LFM3_SNAPSHOT`` (the LFM3-1.7B-ONNX snapshot at
c680759a) and, for the compile and run, ``STARLING_LFM3_INTERVENTION_CACHE`` (a cache root on the snapshot's file
system: the derived graph hard-links the 7 GB of weights). Each run loads one fp32 session (about 7 GB) at a time.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")

from model_pipeline.interventions import (  # noqa: E402
    SITE_MAPPINGS,
    STRENGTH_INPUT_NAME,
    UnsupportedGraphError,
    contract,
    lookup_site_mapping,
)
from model_pipeline.interventions.external_data import link_external_data  # noqa: E402
from model_pipeline.interventions.injection_sites import (  # noqa: E402
    LFM3_FP16_GRAPH_SHA256,
    LFM3_FP16_MAPPING,
    LFM3_FP32_GRAPH_SHA256,
    LFM3_FP32_MAPPING,
    LFM3_HIDDEN_SIZE,
    LFM3_LAYERS,
)
from model_pipeline.interventions.lfm3_synthetic import (  # noqa: E402
    ADDITION_LAYERS,
    ADDITION_PROFILE_ID,
    LFM3_FP16_EXTERNAL_SHA256,
    LFM3_FP16_GRAPH,
    LFM3_FP32_EXTERNAL_SHA256,
    LFM3_FP32_GRAPH,
    SYNTHETIC_LAYERS,
    SYNTHETIC_PROFILE_ID,
    compile_lfm3_synthetic,
    compile_lfm3_synthetic_fp16,
    compile_lfm3_synthetic_multi,
)

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = os.environ.get("STARLING_LFM3_SNAPSHOT")
CACHE = os.environ.get("STARLING_LFM3_INTERVENTION_CACHE")
# "What is 2 + 2? Answer briefly." rendered by the LFM3 chat template without thinking (the template corpus case
# "arithmetic"; the .NET tests use the same ids).
PROMPT_IDS = [124894, 124899, 5922, 207, 2992, 355, 229, 26, 1510, 229, 26, 39, 41774, 21422, 22, 124900, 207, 124899]
PROMPT_IDS += [63514, 207, 124901, 124902]


def _lock_sha(path: str) -> str:
    """The SHA-256 the .NET fp32 asset lock records for ``path`` (read from Lfm3ModelAssets.cs, the source of truth)."""
    source = (REPO / "ai-dotnet/src/Starlings.Models.Lfm3/Lfm3ModelAssets.cs").read_text(encoding="utf-8")
    marker = f'new ModelAssetFile("{path}", '
    start = source.index(marker) + len(marker)
    line = source[start : source.index(")", start)]
    return line.split('"')[1]


# --------------------------------------------------------------------------------------------------------------------
# Mapping (no model needed)


def test_mapping_is_keyed_to_the_fp32_lock_hash():
    assert _lock_sha("onnx/model.onnx") == LFM3_FP32_GRAPH_SHA256
    for path, sha in LFM3_FP32_EXTERNAL_SHA256.items():
        assert _lock_sha(path) == sha
    assert lookup_site_mapping(LFM3_FP32_GRAPH_SHA256) is LFM3_FP32_MAPPING
    assert SITE_MAPPINGS[LFM3_FP32_GRAPH_SHA256].precision == "fp32"
    assert contract.is_version_label(LFM3_FP32_MAPPING.version)


def test_fp16_mapping_is_keyed_to_the_fp16_lock_hash():
    assert _lock_sha(LFM3_FP16_GRAPH) == LFM3_FP16_GRAPH_SHA256
    for path, sha in LFM3_FP16_EXTERNAL_SHA256.items():
        assert _lock_sha(path) == sha
    assert lookup_site_mapping(LFM3_FP16_GRAPH_SHA256) is LFM3_FP16_MAPPING
    assert LFM3_FP16_MAPPING.precision == "fp16"
    assert LFM3_FP16_MAPPING.version == "lfm3-1.7b/fp16/1"
    assert LFM3_FP16_MAPPING.version != LFM3_FP32_MAPPING.version
    # The fp16 export names its nodes like the fp32 one (checked against the real graph below).
    assert LFM3_FP16_MAPPING.sites == LFM3_FP32_MAPPING.sites


def test_another_graph_is_unsupported():
    with pytest.raises(UnsupportedGraphError):
        lookup_site_mapping("0" * 64)


def test_mapping_sites_are_the_residual_stream_after_every_decoder_layer():
    sites = LFM3_FP32_MAPPING.sites
    assert [s.layer for s in sites] == list(range(LFM3_LAYERS))
    assert {s.target for s in sites} == {"residual_post"}
    assert {s.hidden_size for s in sites} == {LFM3_HIDDEN_SIZE}
    assert LFM3_FP32_MAPPING.site(8, "residual_post").tensor == "/model/layers.8/Add_2/output_0"
    assert LFM3_FP32_MAPPING.site(8, "residual_post").consumers == (
        "/model/layers.9/operator_layernorm/LayerNorm",
        "/model/layers.9/Add_1",
    )
    assert LFM3_FP32_MAPPING.site(23, "residual_post").consumers == ("/model/logits_to_keep/Slice",)
    with pytest.raises(UnsupportedGraphError):
        LFM3_FP32_MAPPING.site(24, "residual_post")


# --------------------------------------------------------------------------------------------------------------------
# The real graph

needs_snapshot = pytest.mark.skipif(
    not SNAPSHOT or not Path(SNAPSHOT, LFM3_FP32_GRAPH).is_file(),
    reason="set STARLING_LFM3_SNAPSHOT to the LFM3-1.7B-ONNX snapshot (c680759a)",
)
needs_cache = pytest.mark.skipif(
    not SNAPSHOT or not CACHE, reason="set STARLING_LFM3_SNAPSHOT and STARLING_LFM3_INTERVENTION_CACHE"
)


@needs_snapshot
def test_mapping_matches_the_real_graph_exactly():
    _check_mapping_against_graph(Path(SNAPSHOT, LFM3_FP32_GRAPH), LFM3_FP32_MAPPING, onnx.TensorProto.FLOAT)


needs_fp16_snapshot = pytest.mark.skipif(
    not SNAPSHOT or not Path(SNAPSHOT, LFM3_FP16_GRAPH).is_file(),
    reason="set STARLING_LFM3_SNAPSHOT to the LFM3-1.7B-ONNX snapshot (c680759a) holding onnx/model_fp16.onnx",
)


@needs_fp16_snapshot
def test_fp16_mapping_matches_the_real_fp16_graph_exactly():
    _check_mapping_against_graph(Path(SNAPSHOT, LFM3_FP16_GRAPH), LFM3_FP16_MAPPING, onnx.TensorProto.FLOAT16)


def _check_mapping_against_graph(path: Path, mapping, elem_type: int) -> None:
    assert hashlib.sha256(path.read_bytes()).hexdigest() == mapping.base_graph_sha256
    model = onnx.load(str(path), load_external_data=False)
    readers: dict[str, list[str]] = {}
    producers: dict[str, str] = {}
    for node in model.graph.node:
        for name in node.input:
            readers.setdefault(name, []).append(node.name)
        for name in node.output:
            producers[name] = node.op_type
    value_info = {v.name: v for v in model.graph.value_info}
    for site in mapping.sites:
        assert producers[site.tensor] == "Add", site.tensor
        # Every reader of the residual tensor is rewired: nothing downstream keeps the original value.
        assert sorted(readers[site.tensor]) == sorted(site.consumers), site.tensor
        tensor_type = value_info[site.tensor].type.tensor_type
        assert tensor_type.elem_type == elem_type, site.tensor
        assert tensor_type.shape.dim[-1].dim_value == LFM3_HIDDEN_SIZE


def _feeds(session, ids: list[int]) -> dict[str, np.ndarray]:
    n = len(ids)
    feeds = {
        "input_ids": np.asarray([ids], dtype=np.int64),
        "attention_mask": np.ones((1, n), dtype=np.int64),
        "position_ids": np.arange(n, dtype=np.int64)[None, :],
        "logits_to_keep": np.asarray([1], dtype=np.int64),
    }
    for item in session.get_inputs():
        # The fp16 graph's caches are float16 (its logits stay float32).
        dtype = np.float16 if item.type == "tensor(float16)" else np.float32
        if item.name.startswith("past_conv."):
            feeds[item.name] = np.zeros((1, LFM3_HIDDEN_SIZE, item.shape[2]), dtype=dtype)
        elif item.name.startswith("past_key_values."):
            feeds[item.name] = np.zeros((1, item.shape[1], 0, item.shape[3]), dtype=dtype)
    return feeds


def _run(path: Path, outputs: list[str] | None, strength: float | None) -> dict[str, np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    try:
        feeds = _feeds(session, PROMPT_IDS)
        if strength is not None:
            feeds[STRENGTH_INPUT_NAME] = np.asarray([strength], dtype=np.float32)
        names = outputs or ["logits"]
        return dict(zip(names, session.run(names, feeds), strict=True))
    finally:
        del session


@pytest.fixture(scope="module")
def entry():
    profile_dir = Path(CACHE, "profiles", SYNTHETIC_PROFILE_ID)
    return compile_lfm3_synthetic(SNAPSHOT, CACHE, profile_dir)


@needs_cache
def test_compiled_entry_declares_one_strength_input_and_the_profile_sites(entry):
    manifest = entry.manifest
    assert manifest.base_graph.sha256 == LFM3_FP32_GRAPH_SHA256
    assert [(i.name, i.role, i.dtype, i.shape) for i in manifest.inputs] == [
        (STRENGTH_INPUT_NAME, "strength", "float32", (1,))
    ]
    assert [(s.layer, s.tensor) for s in manifest.sites] == [
        (layer, f"/model/layers.{layer}/Add_2/output_0") for layer in SYNTHETIC_LAYERS
    ]


@needs_cache
def test_strength_zero_matches_the_base_logits_and_strength_one_differs(entry):
    base = _run(Path(SNAPSHOT, LFM3_FP32_GRAPH), None, None)["logits"]
    zero = _run(entry.graph_path, None, 0.0)["logits"]
    one = _run(entry.graph_path, None, 1.0)["logits"]
    diff_zero = float(np.max(np.abs(base - zero)))
    diff_one = float(np.max(np.abs(base - one)))
    print(f"max|base - strength 0| = {diff_zero:.3g}, max|base - strength 1| = {diff_one:.3g}")
    assert diff_zero <= 1e-4
    assert diff_one > 1e-2


@needs_cache
def test_site_tensor_matches_the_numpy_projection(entry, tmp_path_factory):
    """Expose the layer-8 residual tensor and its intervened value as outputs; compare with y - a((yQ^T) * s)Q."""
    layer = SYNTHETIC_LAYERS[0]
    site_tensor = f"/model/layers.{layer}/Add_2/output_0"
    intervened = f"intervention/layer{layer}/residual_post/output"
    derived = onnx.load(str(entry.graph_path), load_external_data=False)
    assert intervened in {o for n in derived.graph.node for o in n.output}
    for name in (site_tensor, intervened):
        derived.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
    # A private copy beside the cache (same file system), the weights hard-linked as the compiler does.
    probe = Path(CACHE, "probe-" + os.urandom(4).hex())
    probe.mkdir()
    try:
        locations = sorted(p.split("/", 1)[1] for p in LFM3_FP32_EXTERNAL_SHA256)
        link_external_data(Path(SNAPSHOT, "onnx"), probe, locations)
        (probe / "model.onnx").write_bytes(derived.SerializeToString())
        profile = entry.profile
        site = next(s for s in profile.profile.sites if s.layer == layer)
        q = profile.load_directions(site).astype(np.float64)
        scales = np.asarray(site.scales, dtype=np.float64)
        for strength in (1.0, 0.5):
            out = _run(probe / "model.onnx", ["logits", site_tensor, intervened], strength)
            y = out[site_tensor].astype(np.float64)
            expected = y - strength * ((y @ q.T) * scales) @ q
            err = float(np.max(np.abs(out[intervened] - expected)))
            scale = float(np.max(np.abs(y)))
            print(f"strength {strength}: max|site - numpy| = {err:.3g} (max|y| = {scale:.3g})")
            assert err <= 1e-5 * max(1.0, scale)
            # The removed component is the span of Q at strength 1 with scales 1.
            if strength == 1.0:
                assert float(np.max(np.abs(out[intervened].astype(np.float64) @ q.T))) <= 1e-4 * max(1.0, scale)
    finally:
        for child in probe.iterdir():
            child.unlink()
        probe.rmdir()


# --------------------------------------------------------------------------------------------------------------------
# Two profiles, one strength vector (manifest schema 2)


def _run_vectors(path: Path, outputs: list[str], vectors: list[tuple[float, ...]]) -> list[dict[str, np.ndarray]]:
    """One session, one run per strength vector (each run gets its own [n] tensor)."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    try:
        feeds = _feeds(session, PROMPT_IDS)
        results = []
        for vector in vectors:
            values = session.run(outputs, {**feeds, STRENGTH_INPUT_NAME: np.asarray(vector, dtype=np.float32)})
            results.append(dict(zip(outputs, values, strict=True)))
        return results
    finally:
        del session


@pytest.fixture(scope="module")
def multi_entry():
    return compile_lfm3_synthetic_multi(SNAPSHOT, CACHE, Path(CACHE, "profiles"))


@needs_cache
def test_multi_entry_declares_a_strength_vector_and_the_shared_site(multi_entry, entry):
    manifest = multi_entry.manifest
    assert manifest.schema_version == 2
    assert [p.id for p in manifest.profiles] == [SYNTHETIC_PROFILE_ID, ADDITION_PROFILE_ID]
    # Profile 0 is byte for byte the profile of the single-profile entry.
    assert manifest.profiles[0].sha256 == entry.manifest.profile_sha256
    assert [(i.name, i.shape) for i in manifest.inputs] == [(STRENGTH_INPUT_NAME, (2,))]
    expected = {layer: [] for layer in sorted({*SYNTHETIC_LAYERS, *ADDITION_LAYERS})}
    for layer in SYNTHETIC_LAYERS:
        expected[layer].append(0)
    for layer in ADDITION_LAYERS:
        expected[layer].append(1)
    assert [(s.layer, s.profiles) for s in manifest.sites] == [(k, tuple(v)) for k, v in expected.items()]


@needs_cache
def test_strength_vector_zero_matches_the_base_and_one_profile_matches_its_single_graph(multi_entry, entry):
    base = _run(Path(SNAPSHOT, LFM3_FP32_GRAPH), None, None)["logits"]
    single = _run(entry.graph_path, None, 1.0)["logits"]
    zero, first, second, both = (
        r["logits"] for r in _run_vectors(multi_entry.graph_path, ["logits"], [(0, 0), (1, 0), (0, 1), (1, 1)])
    )
    diffs = {
        "base - (0,0)": float(np.max(np.abs(base - zero))),
        "single(1) - (1,0)": float(np.max(np.abs(single - first))),
        "base - (0,1)": float(np.max(np.abs(base - second))),
        "(1,0) - (1,1)": float(np.max(np.abs(first - both))),
    }
    print(", ".join(f"max|{k}| = {v:.3g}" for k, v in diffs.items()))
    assert diffs["base - (0,0)"] <= 1e-4
    assert diffs["single(1) - (1,0)"] <= 1e-4
    assert diffs["base - (0,1)"] > 1e-2
    assert diffs["(1,0) - (1,1)"] > 1e-2


@needs_cache
def test_shared_site_matches_the_numpy_composition(multi_entry):
    """At the shared layer-16 site: the projection (profile 0) reads y, the addition (profile 1) reads its output."""
    layer = 16
    assert layer in SYNTHETIC_LAYERS and layer in ADDITION_LAYERS
    site_tensor = f"/model/layers.{layer}/Add_2/output_0"
    out0 = f"intervention/p0/layer{layer}/residual_post/output"
    out1 = f"intervention/p1/layer{layer}/residual_post/output"
    derived = onnx.load(str(multi_entry.graph_path), load_external_data=False)
    for name in (site_tensor, out0, out1):
        derived.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
    probe = Path(CACHE, "probe-" + os.urandom(4).hex())
    probe.mkdir()
    try:
        locations = sorted(p.split("/", 1)[1] for p in LFM3_FP32_EXTERNAL_SHA256)
        link_external_data(Path(SNAPSHOT, "onnx"), probe, locations)
        (probe / "model.onnx").write_bytes(derived.SerializeToString())
        a, b = multi_entry.profiles
        site_a = next(s for s in a.profile.sites if s.layer == layer)
        site_b = next(s for s in b.profile.sites if s.layer == layer)
        qa, sa = a.load_directions(site_a).astype(np.float64), np.asarray(site_a.scales, dtype=np.float64)
        qb, sb = b.load_directions(site_b).astype(np.float64), np.asarray(site_b.scales, dtype=np.float64)
        vectors = [(1.0, 1.0), (0.5, -2.0)]
        for (alpha, beta), out in zip(
            vectors, _run_vectors(probe / "model.onnx", ["logits", site_tensor, out0, out1], vectors), strict=True
        ):
            y = out[site_tensor].astype(np.float64)
            projected = y - alpha * ((y @ qa.T) * sa) @ qa
            expected = projected + beta * (sb @ qb)
            scale = float(np.max(np.abs(y)))
            err0 = float(np.max(np.abs(out[out0] - projected)))
            err1 = float(np.max(np.abs(out[out1] - expected)))
            print(f"strengths ({alpha}, {beta}): max|p0 - numpy| = {err0:.3g}, max|p1 - numpy| = {err1:.3g}")
            assert err0 <= 1e-5 * max(1.0, scale)
            assert err1 <= 1e-5 * max(1.0, scale)
    finally:
        for child in probe.iterdir():
            child.unlink()
        probe.rmdir()


# --------------------------------------------------------------------------------------------------------------------
# The fp16 graph (site mapping lfm3-1.7b/fp16/1): float16 sites, so the compiler casts around the operator

needs_fp16_cache = pytest.mark.skipif(
    not SNAPSHOT or not CACHE or not Path(SNAPSHOT, LFM3_FP16_GRAPH).is_file(),
    reason="set STARLING_LFM3_SNAPSHOT (with onnx/model_fp16.onnx) and STARLING_LFM3_INTERVENTION_CACHE",
)
FP16_EPS = float(np.finfo(np.float16).eps)  # 2**-10


@pytest.fixture(scope="module")
def fp16_entry():
    return compile_lfm3_synthetic_fp16(SNAPSHOT, CACHE, Path(CACHE, "profiles", f"{SYNTHETIC_PROFILE_ID}-fp16"))


@needs_fp16_cache
def test_fp16_entry_records_the_fp16_mapping_and_casts_at_each_site(fp16_entry):
    manifest = fp16_entry.manifest
    assert manifest.base_graph.sha256 == LFM3_FP16_GRAPH_SHA256
    assert manifest.precision == "fp16"
    assert manifest.site_mapping_version == LFM3_FP16_MAPPING.version
    assert [(i.name, i.dtype, i.shape) for i in manifest.inputs] == [(STRENGTH_INPUT_NAME, "float32", (1,))]
    derived = onnx.load(str(fp16_entry.graph_path), load_external_data=False)
    casts = {n.name: n for n in derived.graph.node if n.op_type == "Cast" and n.name.startswith("intervention/")}
    for layer in SYNTHETIC_LAYERS:
        # y -> float32 for the operator, the correction -> float16 before the Sub on the float16 residual stream.
        targets = sorted(
            onnx.helper.get_attribute_value(next(a for a in n.attribute if a.name == "to"))
            for name, n in casts.items()
            if f"/layer{layer}/" in name
        )
        assert targets == sorted([onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16]), layer


@needs_fp16_cache
def test_fp16_strength_zero_matches_the_base_logits_and_strength_one_differs(fp16_entry):
    base = _run(Path(SNAPSHOT, LFM3_FP16_GRAPH), None, None)["logits"]
    zero = _run(fp16_entry.graph_path, None, 0.0)["logits"]
    one = _run(fp16_entry.graph_path, None, 1.0)["logits"]
    assert base.dtype == np.float32  # the fp16 export's logits stay float32
    diff_zero = float(np.max(np.abs(base - zero)))
    diff_one = float(np.max(np.abs(base - one)))
    print(f"fp16: max|base - strength 0| = {diff_zero:.3g}, max|base - strength 1| = {diff_one:.3g}")
    # Strength 0 is y - Cast(0) = y on the float16 stream, so only a different operator fusion around the extra reader
    # could move the logits: allow two float16 units in the last place at the logits' magnitude (measured: 0).
    assert diff_zero <= 2 * FP16_EPS * max(1.0, float(np.max(np.abs(base))))
    assert int(np.argmax(base[0, -1])) == int(np.argmax(zero[0, -1]))
    assert diff_one > 1e-1


@needs_fp16_cache
def test_fp16_site_tensor_matches_the_numpy_projection(fp16_entry):
    layer = SYNTHETIC_LAYERS[0]
    site_tensor = f"/model/layers.{layer}/Add_2/output_0"
    intervened = f"intervention/layer{layer}/residual_post/output"
    derived = onnx.load(str(fp16_entry.graph_path), load_external_data=False)
    for name in (site_tensor, intervened):
        derived.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
    probe = Path(CACHE, "probe-" + os.urandom(4).hex())
    probe.mkdir()
    try:
        locations = sorted(p.split("/", 1)[1] for p in LFM3_FP16_EXTERNAL_SHA256)
        link_external_data(Path(SNAPSHOT, "onnx"), probe, locations)
        (probe / "model_fp16.onnx").write_bytes(derived.SerializeToString())
        profile = fp16_entry.profile
        site = next(s for s in profile.profile.sites if s.layer == layer)
        q = profile.load_directions(site).astype(np.float64)
        scales = np.asarray(site.scales, dtype=np.float64)
        for strength in (1.0, 0.5):
            out = _run(probe / "model_fp16.onnx", ["logits", site_tensor, intervened], strength)
            assert out[site_tensor].dtype == np.float16 and out[intervened].dtype == np.float16
            y = out[site_tensor].astype(np.float64)
            delta = strength * ((y @ q.T) * scales) @ q
            err = float(np.max(np.abs(out[intervened].astype(np.float64) - (y - delta))))
            bound = 2 * FP16_EPS * max(1.0, float(np.max(np.abs(y)) + np.max(np.abs(delta))))
            print(f"fp16 strength {strength}: max|site - numpy| = {err:.3g} (bound {bound:.3g})")
            # The operator runs in float32; the correction and the Sub round to float16.
            assert err <= bound
    finally:
        for child in probe.iterdir():
            child.unlink()
        probe.rmdir()
