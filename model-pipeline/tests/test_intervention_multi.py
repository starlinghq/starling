"""Profile schema 2 (``directional_addition``) and several profiles in one derived graph with a strength vector.

Synthetic residual-MLP graphs only (seeded random orthonormal directions, no behavioural meaning).
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from test_interventions import (  # noqa: E402
    HIDDEN,
    LAYERS,
    REPO,
    _build_graph,
    _mapping,
    _run,
    _sha,
    _site_arrays,
    _weights,
    _write_base,
    _x,
)

from model_pipeline.interventions import (  # noqa: E402
    COMPILER_VERSION,
    STRENGTH_INPUT_NAME,
    InterventionCacheError,
    InterventionCompileError,
    InterventionContractError,
    InterventionMismatchError,
    MultiProfileManifest,
    compile_intervention,
    compile_interventions,
    contract,
    get_or_compile_profiles,
    load_manifest,
    manifest_cache_key,
    multi_cache_key_text,
    open_entry,
)
from model_pipeline.interventions.contract import (  # noqa: E402
    OPERATOR_DIRECTIONAL_ADDITION,
    OPERATOR_DIRECTIONAL_PROJECTION,
    BaseArtifact,
    GraphInput,
)
from model_pipeline.interventions.synthetic import SyntheticSite, write_synthetic_profile  # noqa: E402

GRAPH = "onnx/model.onnx"


# --------------------------------------------------------------------------------------------------------------------
# Fixtures and the NumPy reference


@pytest.fixture
def two(tmp_path: Path):
    """Bundle plus profile A (projection at layers 0 and 2) and B (addition at layers 1 and 2): layer 2 is shared."""
    base_dir = tmp_path / "bundle"
    graph_sha = _sha(_write_base(base_dir / "onnx", _build_graph()))
    a = write_synthetic_profile(
        tmp_path / "a",
        profile_id="synthetic-a",
        base_model_sha256=graph_sha,
        sites=[SyntheticSite(0, HIDDEN, 3, scales=[1.0, 0.5, 2.0]), SyntheticSite(2, HIDDEN, 2)],
        seed=11,
    )
    b = write_synthetic_profile(
        tmp_path / "b",
        profile_id="synthetic-b",
        base_model_sha256=graph_sha,
        sites=[SyntheticSite(1, HIDDEN, 2, scales=[0.75, -1.5]), SyntheticSite(2, HIDDEN, 1, scales=[2.0])],
        seed=23,
        operator=OPERATOR_DIRECTIONAL_ADDITION,
    )
    return base_dir, graph_sha, a, b


def _compile_multi(base_dir, graph_sha, profiles, out: Path, **kwargs):
    kwargs.setdefault("mapping", _mapping(graph_sha))
    return compile_interventions(base_dir=base_dir, graph=GRAPH, profiles=profiles, out_dir=out, **kwargs)


def _run_vector(path: Path, x: np.ndarray, strengths) -> dict[str, np.ndarray]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {"x": x, STRENGTH_INPUT_NAME: np.asarray(strengths, dtype=np.float32)}
    names = [o.name for o in session.get_outputs()]
    return dict(zip(names, session.run(names, feeds), strict=True))


def _reference(x: np.ndarray, profiles, strengths) -> dict[str, np.ndarray]:
    """NumPy float64 forward pass. ``profiles`` is ``[(operator, {layer: (Q, s)})]`` in strength-vector order; after
    each layer the profiles with a site there are applied in that order, each to the previous one's output:
    projection ``y - a ((y Q^T) * s) Q``, addition ``y + a (s Q)``."""
    w = {k: v.astype(np.float64) for k, v in _weights().items()}
    h = x.astype(np.float64)
    tap = None
    for i in range(LAYERS):
        h = h + np.maximum((h * w[f"g{i}"]) @ w[f"w1_{i}"], 0.0) @ w[f"w2_{i}"]
        if i == 0:
            tap = h  # the graph output reads the original tensor
        for (operator, sites), alpha in zip(profiles, strengths, strict=True):
            if i not in sites:
                continue
            q, s = (v.astype(np.float64) for v in sites[i])
            projection = operator == OPERATOR_DIRECTIONAL_PROJECTION
            h = h - alpha * (((h @ q.T) * s) @ q) if projection else h + alpha * (s @ q)
    return {"logits": h @ w["wout"], "hidden_states_0": tap}


def _ref_profiles(*profiles):
    return [(p.profile.operator, _site_arrays(p)) for p in profiles]


# --------------------------------------------------------------------------------------------------------------------
# Profile schema 2


def _profile_doc(**overrides) -> dict:
    document = {
        "schema_version": 2,
        "id": "synthetic-add",
        "base_model_sha256": "0" * 64,
        "operator": OPERATOR_DIRECTIONAL_ADDITION,
        "sites": [
            {
                "layer": 1,
                "target": "residual_post",
                "hidden_size": 4,
                "rank": 1,
                "dtype": "float32",
                "directions_file": "directions/layer1.f32",
                "directions_sha256": "1" * 64,
                "scales": [0.5],
            }
        ],
    }
    document.update(overrides)
    return document


def test_schema_2_profile_carries_the_addition_operator():
    profile = contract.parse_profile(json.dumps(_profile_doc()).encode())
    assert (profile.schema_version, profile.operator) == (2, OPERATOR_DIRECTIONAL_ADDITION)
    projection = contract.parse_profile(json.dumps(_profile_doc(operator=OPERATOR_DIRECTIONAL_PROJECTION)).encode())
    assert (projection.schema_version, projection.operator) == (2, OPERATOR_DIRECTIONAL_PROJECTION)


@pytest.mark.parametrize(
    ("document", "path"),
    [
        (_profile_doc(schema_version=1), "operator"),  # version 1 knows only directional_projection
        (_profile_doc(operator="directional_scaling"), "operator"),
        (_profile_doc(operator="Directional_Addition"), "operator"),
        (_profile_doc(schema_version=3), "schema_version"),
        (_profile_doc(extra=1), "$"),
    ],
)
def test_schema_2_profile_rejections(document, path):
    with pytest.raises(InterventionContractError) as info:
        contract.parse_profile(json.dumps(document).encode())
    assert info.value.json_path == path


def test_synthetic_addition_profile_is_written_as_schema_2(two):
    _, _, a, b = two
    assert json.loads(a.path.read_bytes())["schema_version"] == 1
    assert json.loads(b.path.read_bytes())["schema_version"] == 2
    assert b.profile.operator == OPERATOR_DIRECTIONAL_ADDITION


# --------------------------------------------------------------------------------------------------------------------
# Numerics


def test_zero_vector_matches_the_base_graph_exactly(two, tmp_path):
    base_dir, graph_sha, a, b = two
    result = _compile_multi(base_dir, graph_sha, [a.path, b.path], tmp_path / "ab")
    x = _x()
    base = _run(base_dir / GRAPH, x)
    derived = _run_vector(result.graph_path, x, [0.0, 0.0])
    for name in base:
        np.testing.assert_array_equal(derived[name], base[name])


def test_one_profile_at_a_time_equals_the_single_profile_graphs(two, tmp_path):
    base_dir, graph_sha, a, b = two
    multi = _compile_multi(base_dir, graph_sha, [a, b], tmp_path / "ab").graph_path
    single_a = compile_intervention(
        base_dir=base_dir, graph=GRAPH, profile=a, out_dir=tmp_path / "a-only", mapping=_mapping(graph_sha)
    ).graph_path
    single_b = _compile_multi(base_dir, graph_sha, [b], tmp_path / "b-only").graph_path
    x = _x()
    for alpha in (1.0, 0.5, -2.0):
        only_a = _run(single_a, x, alpha)
        only_b = _run_vector(single_b, x, [alpha])
        with_a = _run_vector(multi, x, [alpha, 0.0])
        with_b = _run_vector(multi, x, [0.0, alpha])
        for name in only_a:
            # Bit-identical here: a zero-strength link adds an exact zero. Other graphs or optimization levels may
            # only agree within rounding (fusion can differ).
            np.testing.assert_array_equal(with_a[name], only_a[name])
            np.testing.assert_array_equal(with_b[name], only_b[name])


@pytest.mark.parametrize("strengths", [(1.0, 1.0), (0.5, -2.0), (-1.0, 0.25), (2.0, 3.0)])
def test_combined_strengths_match_the_numpy_reference(two, tmp_path, strengths):
    base_dir, graph_sha, a, b = two
    path = _compile_multi(base_dir, graph_sha, [a, b], tmp_path / "ab").graph_path
    x = _x()
    derived = _run_vector(path, x, strengths)
    reference = _reference(x, _ref_profiles(a, b), strengths)
    for name in reference:
        np.testing.assert_allclose(derived[name], reference[name], rtol=1e-4, atol=1e-4)


def test_shared_sites_compose_in_strength_vector_order(two, tmp_path):
    """At layer 2 both profiles apply: [a, b] projects then adds, [b, a] adds then projects. Both match the reference
    for their own order and differ from each other."""
    base_dir, graph_sha, a, b = two
    ab = _compile_multi(base_dir, graph_sha, [a, b], tmp_path / "ab")
    ba = _compile_multi(base_dir, graph_sha, [b, a], tmp_path / "ba")
    x = _x()
    out_ab = _run_vector(ab.graph_path, x, [1.0, 1.0])["logits"]
    out_ba = _run_vector(ba.graph_path, x, [1.0, 1.0])["logits"]
    np.testing.assert_allclose(out_ab, _reference(x, _ref_profiles(a, b), (1.0, 1.0))["logits"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out_ba, _reference(x, _ref_profiles(b, a), (1.0, 1.0))["logits"], rtol=1e-4, atol=1e-4)
    assert np.max(np.abs(out_ab - out_ba)) > 1e-2
    shared = [s for s in ab.manifest.sites if s.layer == 2]
    assert shared[0].profiles == (0, 1)
    assert [p.id for p in ba.manifest.profiles] == ["synthetic-b", "synthetic-a"]


def test_addition_matches_the_numpy_reference(two, tmp_path):
    base_dir, graph_sha, _, b = two
    path = _compile_multi(base_dir, graph_sha, [b], tmp_path / "b").graph_path
    x = _x()
    for alpha in (1.0, -0.5, 4.0):
        derived = _run_vector(path, x, [alpha])
        reference = _reference(x, _ref_profiles(b), (alpha,))
        for name in reference:
            np.testing.assert_allclose(derived[name], reference[name], rtol=1e-4, atol=1e-4)


def test_strength_vectors_are_isolated_per_run(two, tmp_path):
    base_dir, graph_sha, a, b = two
    path = _compile_multi(base_dir, graph_sha, [a, b], tmp_path / "ab").graph_path
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = _x()
    vectors = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, -2.0), (-1.0, 3.0)] * 4
    expected = {
        v: session.run(["logits"], {"x": x, STRENGTH_INPUT_NAME: np.asarray(v, np.float32)})[0] for v in vectors
    }
    results: dict[int, np.ndarray] = {}

    def run(i: int) -> None:
        feeds = {"x": x, STRENGTH_INPUT_NAME: np.asarray(vectors[i], np.float32)}
        results[i] = session.run(["logits"], feeds)[0]

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(vectors))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i, v in enumerate(vectors):
        np.testing.assert_array_equal(results[i], expected[v])


def test_float16_graph_with_both_operators_is_exact_at_zero(tmp_path):
    base_dir = tmp_path / "bundle"
    graph_sha = _sha(_write_base(base_dir / "onnx", _build_graph(dtype=np.float16)))
    a = write_synthetic_profile(
        tmp_path / "a", profile_id="half-a", base_model_sha256=graph_sha, sites=[SyntheticSite(1, HIDDEN, 2)], seed=1
    )
    b = write_synthetic_profile(
        tmp_path / "b",
        profile_id="half-b",
        base_model_sha256=graph_sha,
        sites=[SyntheticSite(1, HIDDEN, 1, scales=[0.5])],
        seed=2,
        operator=OPERATOR_DIRECTIONAL_ADDITION,
    )
    path = _compile_multi(base_dir, graph_sha, [a, b], tmp_path / "out", mapping=_mapping(graph_sha, precision="fp16"))
    x = _x().astype(np.float16)
    base = _run(base_dir / GRAPH, x)
    derived = _run_vector(path.graph_path, x, [0.0, 0.0])
    for name in base:
        np.testing.assert_array_equal(derived[name], base[name])
    model = onnx.load(str(path.graph_path), load_external_data=False)
    assert any(n.op_type == "Cast" and n.name.startswith("intervention/p1/") for n in model.graph.node)


# --------------------------------------------------------------------------------------------------------------------
# Graph structure and manifest


def test_graph_has_one_strength_vector_input_and_a_gather_per_profile(two, tmp_path):
    base_dir, graph_sha, a, b = two
    result = _compile_multi(base_dir, graph_sha, [a, b], tmp_path / "ab")
    model = onnx.load(str(result.graph_path), load_external_data=False)
    extra = model.graph.input[-1]
    assert extra.name == STRENGTH_INPUT_NAME
    assert [d.dim_value for d in extra.type.tensor_type.shape.dim] == [2]
    gathers = [n for n in model.graph.node if n.op_type == "Gather" and n.name.startswith("intervention/strength/")]
    assert [n.output[0] for n in gathers] == ["intervention/strength/p0", "intervention/strength/p1"]
    # The listed consumers of the shared tensor read the last profile's output.
    lm_head = next(n for n in model.graph.node if n.name == "lm_head")
    assert lm_head.input[0] == "intervention/p1/layer2/residual_post/output"


def test_manifest_schema_2_contents_and_round_trip(two, tmp_path):
    base_dir, graph_sha, a, b = two
    result = _compile_multi(base_dir, graph_sha, [a, b], tmp_path / "ab")
    manifest = result.manifest
    assert isinstance(manifest, MultiProfileManifest)
    assert [(p.id, p.sha256, p.operator) for p in manifest.profiles] == [
        ("synthetic-a", a.sha256, OPERATOR_DIRECTIONAL_PROJECTION),
        ("synthetic-b", b.sha256, OPERATOR_DIRECTIONAL_ADDITION),
    ]
    assert manifest.strength_input == GraphInput(STRENGTH_INPUT_NAME, "strength", "float32", (2,))
    assert [(s.layer, s.profiles) for s in manifest.sites] == [(0, (0,)), (1, (1,)), (2, (0, 1))]
    assert manifest.compiler_version == COMPILER_VERSION
    text = result.manifest_path.read_bytes()
    assert list(json.loads(text)) == list(contract._MULTI_MANIFEST_KEYS)
    assert json.loads(text)["schema_version"] == 2
    loaded = load_manifest(result.manifest_path)
    assert loaded == manifest and loaded.serialize() == text
    contract.verify_manifest_profiles(manifest, [a, b])
    with pytest.raises(InterventionMismatchError):
        contract.verify_manifest_profiles(manifest, [b, a])
    with pytest.raises(InterventionMismatchError):
        contract.verify_manifest_profiles(manifest, [a])


def _multi_doc() -> dict:
    return {
        "schema_version": 2,
        "compiler_version": "0.1.0",
        "site_mapping_version": "tiny-mlp/fp32/1",
        "base_artifacts": [{"role": "graph", "path": "onnx/model.onnx", "sha256": "0" * 64, "size": 1}],
        "profiles": [
            {"id": "a", "sha256": "2" * 64, "operator": OPERATOR_DIRECTIONAL_PROJECTION},
            {"id": "b", "sha256": "5" * 64, "operator": OPERATOR_DIRECTIONAL_ADDITION},
        ],
        "opset": 17,
        "precision": "fp32",
        "inputs": [{"name": STRENGTH_INPUT_NAME, "role": "strength", "dtype": "float32", "shape": [2]}],
        "sites": [
            {"layer": 1, "target": "residual_post", "tensor": "t1", "profiles": [0]},
            {"layer": 2, "target": "residual_post", "tensor": "t2", "profiles": [0, 1]},
        ],
        "derived_graph": {"path": "model.onnx", "sha256": "3" * 64, "size": 2},
        "derived_files": [],
    }


def test_schema_2_manifest_rules():
    assert isinstance(contract.parse_manifest(json.dumps(_multi_doc()).encode()), MultiProfileManifest)

    def rejected(mutate, path):
        doc = _multi_doc()
        mutate(doc)
        with pytest.raises(InterventionContractError) as info:
            contract.parse_manifest(json.dumps(doc).encode())
        assert info.value.json_path == path, info.value

    rejected(lambda d: d["inputs"][0].update(shape=[1]), "inputs")
    rejected(lambda d: d["inputs"][0].update(shape=[]), "inputs")
    rejected(lambda d: d["profiles"][1].update(id="a"), "profiles")
    rejected(lambda d: d["profiles"][1].update(operator="directional_scaling"), "profiles[1].operator")
    rejected(lambda d: d["profiles"][1].update(sha256="X" * 64), "profiles[1].sha256")
    rejected(lambda d: d.update(profiles=[]), "profiles")
    rejected(lambda d: d["sites"][1].update(profiles=[1, 0]), "sites[1].profiles")
    rejected(lambda d: d["sites"][1].update(profiles=[0, 0]), "sites[1].profiles")
    rejected(lambda d: d["sites"][1].update(profiles=[]), "sites[1].profiles")
    rejected(lambda d: d["sites"][1].update(profiles=[0, 2]), "sites[1].profiles[1]")
    rejected(lambda d: d["sites"][1].update(profiles=[0]), "sites")  # profile 1 is applied nowhere
    rejected(lambda d: d["sites"][1].pop("profiles"), "sites[1]")
    rejected(lambda d: d["sites"][1].update(tensor="t1"), "sites")
    rejected(lambda d: d.update(profile_id="a"), "$")
    rejected(lambda d: d.update(compiler_version="-bad"), "compiler_version")
    rejected(lambda d: d["derived_graph"].update(path="../model.onnx"), "derived_graph.path")
    rejected(lambda d: d.update(schema_version=1), "$")  # a schema 1 manifest has no "profiles"


def test_profile_ids_must_be_unique_and_projection_only_api_points_to_the_new_one(two, tmp_path):
    base_dir, graph_sha, a, b = two
    with pytest.raises(InterventionCompileError, match="unique"):
        _compile_multi(base_dir, graph_sha, [a, a], tmp_path / "aa")
    with pytest.raises(InterventionCompileError, match="compile_interventions"):
        compile_intervention(
            base_dir=base_dir, graph=GRAPH, profile=b, out_dir=tmp_path / "b-out", mapping=_mapping(graph_sha)
        )
    assert not (tmp_path / "aa").exists() and not (tmp_path / "b-out").exists()


def test_profile_for_another_graph_is_rejected_before_writing(two, tmp_path):
    base_dir, graph_sha, a, _ = two
    other = write_synthetic_profile(
        tmp_path / "other", profile_id="other", base_model_sha256="f" * 64, sites=[SyntheticSite(0, HIDDEN, 1)]
    )
    with pytest.raises(InterventionMismatchError):
        _compile_multi(base_dir, graph_sha, [a, other], tmp_path / "out")
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------------------------------------------------------------
# Cache


# The documented schema 2 key example (docs/interventions/cache.md); the C# suite checks the same value.
V2_KEY_TEXT = (
    "starling-intervention-cache-key 2\n"
    "compiler_version\t0.1.0\n"
    "site_mapping_version\tlfm3/fp32/1\n"
    "opset\t17\n"
    "precision\tfp32\n"
    "profile\t0\t2222222222222222222222222222222222222222222222222222222222222222\n"
    "profile\t1\t5555555555555555555555555555555555555555555555555555555555555555\n"
    "base\tgraph\tonnx/model.onnx\t0000000000000000000000000000000000000000000000000000000000000000\t1234\n"
    "base\texternal_data\tonnx/model.onnx_data\t1111111111111111111111111111111111111111111111111111111111111111\t4096\n"
    "input\tintervention_strength\tstrength\tfloat32\t[2]\n"
)
V2_KEY = "e0f6313abf870678bfcecfaf7734e414b9d8543428204bbf1204782e2e62d37b"


def test_schema_2_cache_key_text_matches_the_documented_example():
    text = multi_cache_key_text(
        compiler_version="0.1.0",
        site_mapping_version="lfm3/fp32/1",
        opset=17,
        precision="fp32",
        profile_sha256s=["2" * 64, "5" * 64],
        base_artifacts=[
            BaseArtifact("external_data", "onnx/model.onnx_data", "1" * 64, 4096),
            BaseArtifact("graph", "onnx/model.onnx", "0" * 64, 1234),
        ],
        inputs=[GraphInput(STRENGTH_INPUT_NAME, "strength", "float32", (2,))],
    )
    assert text == V2_KEY_TEXT
    assert contract.sha256_bytes(text.encode()) == V2_KEY
    doc = (REPO / "docs/interventions/cache.md").read_text(encoding="utf-8")
    assert V2_KEY in doc


def test_cache_entry_with_two_profiles_is_published_and_reused(two, tmp_path):
    base_dir, graph_sha, a, b = two
    cache_root = tmp_path / "cache"
    kwargs = dict(cache_root=cache_root, model="tiny-mlp", base_dir=base_dir, graph=GRAPH, mapping=_mapping(graph_sha))
    entry = get_or_compile_profiles(profiles=[a, b], **kwargs)
    assert entry.created and entry.directory.name == entry.key == manifest_cache_key(entry.manifest)
    assert (entry.directory / "profiles/0/profile.json").read_bytes() == a.path.read_bytes()
    assert (entry.directory / "profiles/1/profile.json").read_bytes() == b.path.read_bytes()
    # Both synthetic profiles use directions/layer2-residual_post.f32; each keeps its own copy.
    assert (entry.directory / "profiles/0/directions/layer2-residual_post.f32").is_file()
    assert (entry.directory / "profiles/1/directions/layer2-residual_post.f32").is_file()
    assert not (entry.directory / "profile.json").exists()
    assert [p.profile.id for p in entry.profiles] == ["synthetic-a", "synthetic-b"]
    again = get_or_compile_profiles(profiles=[a, b], **kwargs)
    assert not again.created and again.key == entry.key
    swapped = get_or_compile_profiles(profiles=[b, a], **kwargs)
    assert swapped.key != entry.key
    x = _x()
    derived = _run_vector(entry.graph_path, x, [0.5, 1.0])
    np.testing.assert_allclose(
        derived["logits"], _reference(x, _ref_profiles(a, b), (0.5, 1.0))["logits"], rtol=1e-4, atol=1e-4
    )
    opened = open_entry(entry.directory, verify="full", base_dir=base_dir)
    assert opened.key == entry.key


def test_cache_entry_with_a_tampered_second_profile_fails_verification(two, tmp_path):
    base_dir, graph_sha, a, b = two
    entry = get_or_compile_profiles(
        cache_root=tmp_path / "cache",
        model="tiny-mlp",
        base_dir=base_dir,
        graph=GRAPH,
        profiles=[a, b],
        mapping=_mapping(graph_sha),
    )
    target = entry.directory / "profiles/1/profile.json"
    target.chmod(0o644)
    target.write_bytes(target.read_bytes().replace(b'"synthetic-b"', b'"synthetic-c"'))
    with pytest.raises(InterventionCacheError, match="failed verification"):
        open_entry(entry.directory)


def _doc_json_blocks() -> list[bytes]:
    text = (REPO / "docs/interventions/profile-format.md").read_text(encoding="utf-8")
    return [(m + "\n").encode() for m in re.findall(r"```json\n(.*?)\n```", text, flags=re.S)]


def test_documented_schema_2_examples_parse_and_reserialize_identically():
    profile_text, manifest_text = _doc_json_blocks()[2:]
    profile = contract.parse_profile(profile_text)
    assert (profile.schema_version, profile.operator) == (2, OPERATOR_DIRECTIONAL_ADDITION)
    manifest = contract.parse_manifest(manifest_text)
    assert isinstance(manifest, MultiProfileManifest)
    assert manifest.serialize() == manifest_text
    # The C# suite embeds the same manifest text (InterventionMultiProfileTests.CanonicalV2, indented by 8 spaces).
    csharp = (REPO / "ai-dotnet/tests/Starlings.Interventions.Tests/InterventionMultiProfileTests.cs").read_text()
    indented = "\n".join(("        " + line) if line else "" for line in manifest_text.decode().split("\n"))
    assert indented in csharp


def test_three_profiles_on_one_site_compose_in_order(two, tmp_path):
    base_dir, graph_sha, a, b = two
    c = write_synthetic_profile(
        tmp_path / "c",
        profile_id="synthetic-c",
        base_model_sha256=graph_sha,
        sites=[SyntheticSite(2, HIDDEN, 4, scales=[0.5, 1.0, 1.5, 2.0])],
        seed=31,
    )
    result = _compile_multi(base_dir, graph_sha, [b, c, a], tmp_path / "bca")
    assert [s.profiles for s in result.manifest.sites if s.layer == 2] == [(0, 1, 2)]
    x = _x()
    for strengths in [(1.0, 1.0, 1.0), (0.0, 2.0, -1.0), (0.5, 0.0, 0.0)]:
        derived = _run_vector(result.graph_path, x, strengths)
        reference = _reference(x, _ref_profiles(b, c, a), strengths)
        for name in reference:
            np.testing.assert_allclose(derived[name], reference[name], rtol=1e-4, atol=1e-4)
