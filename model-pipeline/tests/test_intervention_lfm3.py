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
    LFM3_FP32_GRAPH_SHA256,
    LFM3_FP32_MAPPING,
    LFM3_HIDDEN_SIZE,
    LFM3_LAYERS,
)
from model_pipeline.interventions.lfm3_synthetic import (  # noqa: E402
    LFM3_FP32_EXTERNAL_SHA256,
    LFM3_FP32_GRAPH,
    SYNTHETIC_LAYERS,
    SYNTHETIC_PROFILE_ID,
    compile_lfm3_synthetic,
)

REPO = Path(__file__).resolve().parents[2]
LFM3_FP16_GRAPH_SHA256 = "d3093d69f6f954bd74f1dc9e59452ec5e9766e62d9c409456fb8ea5698a830fe"
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


def test_fp16_graph_is_unsupported():
    assert _lock_sha("onnx/model_fp16.onnx") == LFM3_FP16_GRAPH_SHA256
    with pytest.raises(UnsupportedGraphError):
        lookup_site_mapping(LFM3_FP16_GRAPH_SHA256)


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
    path = Path(SNAPSHOT, LFM3_FP32_GRAPH)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == LFM3_FP32_GRAPH_SHA256
    model = onnx.load(str(path), load_external_data=False)
    readers: dict[str, list[str]] = {}
    producers: dict[str, str] = {}
    for node in model.graph.node:
        for name in node.input:
            readers.setdefault(name, []).append(node.name)
        for name in node.output:
            producers[name] = node.op_type
    for site in LFM3_FP32_MAPPING.sites:
        assert producers[site.tensor] == "Add", site.tensor
        # Every reader of the residual tensor is rewired: nothing downstream keeps the original value.
        assert sorted(readers[site.tensor]) == sorted(site.consumers), site.tensor
    value_info = {v.name: v for v in model.graph.value_info}
    dims = value_info[LFM3_FP32_MAPPING.sites[0].tensor].type.tensor_type.shape.dim
    assert dims[-1].dim_value == LFM3_HIDDEN_SIZE


def _feeds(session, ids: list[int]) -> dict[str, np.ndarray]:
    n = len(ids)
    feeds = {
        "input_ids": np.asarray([ids], dtype=np.int64),
        "attention_mask": np.ones((1, n), dtype=np.int64),
        "position_ids": np.arange(n, dtype=np.int64)[None, :],
        "logits_to_keep": np.asarray([1], dtype=np.int64),
    }
    for item in session.get_inputs():
        if item.name.startswith("past_conv."):
            feeds[item.name] = np.zeros((1, LFM3_HIDDEN_SIZE, item.shape[2]), dtype=np.float32)
        elif item.name.startswith("past_key_values."):
            feeds[item.name] = np.zeros((1, item.shape[1], 0, item.shape[3]), dtype=np.float32)
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
