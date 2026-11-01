"""Capture mode (model_pipeline.interventions.capture) on a synthetic residual MLP with external data."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from model_pipeline.interventions import InterventionCompileError, MappedSite, SiteMapping  # noqa: E402
from model_pipeline.interventions import contract as contract_module  # noqa: E402
from model_pipeline.interventions.capture import (  # noqa: E402
    CAPTURE_DIRECTORY,
    CAPTURE_MANIFEST_NAME,
    InterventionCaptureError,
    build_capture_model,
    capture_key_text,
    compile_capture,
    get_or_compile_capture,
    manifest_capture_key,
    open_capture,
    parse_capture_manifest,
    validate_capture_model,
)
from model_pipeline.interventions.injection_sites import UnsupportedGraphError  # noqa: E402
from model_pipeline.interventions.validate import DerivedGraphError, run_outputs  # noqa: E402

HIDDEN = 16
FFN = 24
VOCAB = 7
LAYERS = 3


def _weights(seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    w: dict[str, np.ndarray] = {}
    for i in range(LAYERS):
        w[f"w1_{i}"] = (rng.standard_normal((HIDDEN, FFN)) / np.sqrt(HIDDEN)).astype(np.float32)
        w[f"w2_{i}"] = (rng.standard_normal((FFN, HIDDEN)) / np.sqrt(FFN) * 0.3).astype(np.float32)
    w["wout"] = (rng.standard_normal((HIDDEN, VOCAB)) / np.sqrt(HIDDEN)).astype(np.float32)
    return w


def site_tensor(layer: int) -> str:
    return f"layer{layer}/residual_post"


def build_graph(weights: dict[str, np.ndarray] | None = None) -> onnx.ModelProto:
    """h_{i+1} = h_i + relu(h_i W1_i) W2_i; logits = h_L Wout. Tensor ``layer{i}/residual_post`` is h_{i+1}."""
    w = weights or _weights()
    nodes = []
    h = "x"
    for i in range(LAYERS):
        p = f"layer{i}"
        nodes += [
            helper.make_node("MatMul", [h, f"w1_{i}"], [f"{p}/up"], name=f"{p}/up_proj"),
            helper.make_node("Relu", [f"{p}/up"], [f"{p}/act"], name=f"{p}/relu"),
            helper.make_node("MatMul", [f"{p}/act", f"w2_{i}"], [f"{p}/down"], name=f"{p}/down_proj"),
            helper.make_node("Add", [h, f"{p}/down"], [site_tensor(i)], name=f"{p}/residual_add"),
        ]
        h = site_tensor(i)
    nodes.append(helper.make_node("MatMul", [h, "wout"], ["logits"], name="lm_head"))
    graph = helper.make_graph(
        nodes,
        "capture-mlp",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", "sequence", HIDDEN])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", "sequence", VOCAB])],
        [numpy_helper.from_array(v, k) for k, v in w.items()],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    return model


def reference(x: np.ndarray, weights: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    w = {k: v.astype(np.float64) for k, v in (weights or _weights()).items()}
    h = x.astype(np.float64)
    out = {}
    for i in range(LAYERS):
        h = h + np.maximum(h @ w[f"w1_{i}"], 0.0) @ w[f"w2_{i}"]
        out[site_tensor(i)] = h
    out["logits"] = h @ w["wout"]
    return out


def write_bundle(directory: Path, model: onnx.ModelProto | None = None) -> tuple[Path, str]:
    """``directory/onnx/model.onnx`` with every initializer in ``model.onnx_data``; returns (path, SHA-256)."""
    graph_dir = directory / "onnx"
    graph_dir.mkdir(parents=True)
    path = graph_dir / "model.onnx"
    onnx.save_model(
        model or build_graph(),
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx_data",
        size_threshold=0,
    )
    return path, contract_module.sha256_file(path)


def mapping_for(sha: str) -> SiteMapping:
    sites = []
    for i in range(LAYERS):
        consumers = ("lm_head",) if i == LAYERS - 1 else (f"layer{i + 1}/up_proj", f"layer{i + 1}/residual_add")
        sites.append(MappedSite(i, "residual_post", site_tensor(i), HIDDEN, consumers))
    return SiteMapping("capture-mlp", "capture-mlp/fp32/1", sha, "fp32", tuple(sites))


@pytest.fixture
def bundle(tmp_path):
    base_dir = tmp_path / "bundle"
    path, sha = write_bundle(base_dir)
    return base_dir, path, sha, mapping_for(sha)


def _x(seed: int = 0, sequence: int = 5) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((1, sequence, HIDDEN)).astype(np.float32)


# --------------------------------------------------------------------------------------------------------------------
# The capture graph


def test_capture_outputs_equal_the_base_graphs_internal_tensors(bundle, tmp_path):
    base_dir, path, sha, mapping = bundle
    result = compile_capture(base_dir=base_dir, graph="onnx/model.onnx", layers=[0, 2], out_dir=tmp_path / "cap",
                             mapping=mapping)  # fmt: skip
    assert result.manifest.outputs[0].tensor == site_tensor(0)
    x = _x()
    captured = run_outputs(result.graph_path, {"x": x})
    assert list(captured) == ["logits", site_tensor(0), site_tensor(2)]

    # Independently: the base graph with the internal tensors appended as outputs by hand (separate directory, the
    # weights hard-linked), and the NumPy forward pass.
    probe_model = onnx.load(str(path), load_external_data=False)
    for layer in (0, 2):
        probe_model.graph.output.append(helper.make_empty_tensor_value_info(site_tensor(layer)))
    probe = tmp_path / "probe"
    probe.mkdir()
    os.link(base_dir / "onnx" / "model.onnx_data", probe / "model.onnx_data")
    (probe / "model.onnx").write_bytes(probe_model.SerializeToString())
    internal = run_outputs(probe / "model.onnx", {"x": x})
    base = run_outputs(path, {"x": x})
    expected = reference(x)
    np.testing.assert_array_equal(captured["logits"], base["logits"])
    for layer in (0, 2):
        name = site_tensor(layer)
        np.testing.assert_array_equal(captured[name], internal[name])
        np.testing.assert_allclose(captured[name], expected[name], rtol=1e-5, atol=1e-5)


def test_capture_graph_changes_nothing_but_the_outputs(bundle, tmp_path):
    base_dir, path, sha, mapping = bundle
    result = compile_capture(base_dir=base_dir, graph="onnx/model.onnx", layers=[1], out_dir=tmp_path / "cap",
                             mapping=mapping)  # fmt: skip
    base = onnx.load(str(path), load_external_data=False)
    capture = onnx.load(str(result.graph_path), load_external_data=False)
    assert [n.SerializeToString() for n in capture.graph.node] == [n.SerializeToString() for n in base.graph.node]
    assert [t.SerializeToString() for t in capture.graph.initializer] == [
        t.SerializeToString() for t in base.graph.initializer
    ]
    assert [o.name for o in capture.graph.output] == ["logits", site_tensor(1)]
    # The weights are a hard link, not a copy, and the bundle is untouched.
    assert os.path.samefile(base_dir / "onnx" / "model.onnx_data", tmp_path / "cap" / "model.onnx_data")
    assert contract_module.sha256_file(path) == sha
    manifest = parse_capture_manifest((tmp_path / "cap" / CAPTURE_MANIFEST_NAME).read_bytes())
    assert manifest == result.manifest
    assert manifest.base_graph.sha256 == sha
    assert [f.path for f in manifest.derived_files] == ["model.onnx_data"]


def test_validate_capture_model_rejects_any_other_change(bundle):
    _, path, _, mapping = bundle
    base = onnx.load(str(path), load_external_data=False)
    sites = [mapping.site(1, "residual_post")]
    capture = build_capture_model(base, sites)
    validate_capture_model(base, capture, sites)

    extra = onnx.ModelProto()
    extra.CopyFrom(capture)
    extra.graph.output.append(helper.make_tensor_value_info("layer0/up", TensorProto.FLOAT, ["batch", "sequence", FFN]))
    with pytest.raises(DerivedGraphError, match="adds outputs"):
        validate_capture_model(base, extra, sites)

    rewired = onnx.ModelProto()
    rewired.CopyFrom(capture)
    rewired.graph.node[-1].input[0] = site_tensor(0)
    with pytest.raises(DerivedGraphError, match="nodes"):
        validate_capture_model(base, rewired, sites)

    dropped = onnx.ModelProto()
    dropped.CopyFrom(capture)
    del dropped.graph.output[0]
    with pytest.raises(DerivedGraphError):
        validate_capture_model(base, dropped, sites)


def test_a_site_already_exposed_is_not_added_twice(bundle):
    _, path, _, mapping = bundle
    base = onnx.load(str(path), load_external_data=False)
    base.graph.output.append(
        helper.make_tensor_value_info(site_tensor(0), TensorProto.FLOAT, ["batch", "sequence", HIDDEN])
    )
    sites = [mapping.site(0, "residual_post"), mapping.site(1, "residual_post")]
    capture = build_capture_model(base, sites)
    assert [o.name for o in capture.graph.output] == ["logits", site_tensor(0), site_tensor(1)]
    validate_capture_model(base, capture, sites)


def test_unknown_graphs_layers_and_bad_directories_are_rejected(bundle, tmp_path):
    base_dir, _, _, mapping = bundle
    with pytest.raises(UnsupportedGraphError):
        compile_capture(base_dir=base_dir, graph="onnx/model.onnx", layers=[0], out_dir=tmp_path / "a")
    with pytest.raises(UnsupportedGraphError):
        compile_capture(base_dir=base_dir, graph="onnx/model.onnx", layers=[LAYERS], out_dir=tmp_path / "b",
                        mapping=mapping)  # fmt: skip
    with pytest.raises(InterventionCompileError, match="duplicates"):
        compile_capture(base_dir=base_dir, graph="onnx/model.onnx", layers=[0, 0], out_dir=tmp_path / "c",
                        mapping=mapping)  # fmt: skip
    with pytest.raises(InterventionCompileError, match="inside the base"):
        compile_capture(base_dir=base_dir, graph="onnx/model.onnx", layers=[0], out_dir=base_dir / "cap",
                        mapping=mapping)  # fmt: skip
    other = mapping_for("0" * 64)
    with pytest.raises(contract_module.InterventionMismatchError):
        compile_capture(base_dir=base_dir, graph="onnx/model.onnx", layers=[0], out_dir=tmp_path / "d",
                        mapping=other)  # fmt: skip


def test_wrong_hidden_size_in_the_mapping_is_rejected(bundle, tmp_path):
    base_dir, _, sha, _ = bundle
    wrong = SiteMapping(
        "capture-mlp", "capture-mlp/fp32/9", sha, "fp32", (MappedSite(0, "residual_post", site_tensor(0), 8, ("x",)),)
    )
    with pytest.raises(InterventionCompileError, match="last dimension"):
        compile_capture(base_dir=base_dir, graph="onnx/model.onnx", layers=[0], out_dir=tmp_path / "w", mapping=wrong)


# --------------------------------------------------------------------------------------------------------------------
# Cache


def test_capture_cache_compiles_once_and_reuses(bundle, tmp_path):
    base_dir, _, _, mapping = bundle
    root = tmp_path / "cache"
    first = get_or_compile_capture(cache_root=root, model="mlp", base_dir=base_dir, graph="onnx/model.onnx",
                                   layers=[2, 0], mapping=mapping)  # fmt: skip
    assert first.created
    assert first.directory.parent == root / "mlp" / CAPTURE_DIRECTORY
    assert first.directory.name == first.key == manifest_capture_key(first.manifest)
    assert [o.layer for o in first.manifest.outputs] == [0, 2]
    assert first.output(2) == site_tensor(2)
    assert not (root / "mlp" / "interventions").exists()  # never where the C# reader lists intervention entries
    second = get_or_compile_capture(cache_root=root, model="mlp", base_dir=base_dir, graph="onnx/model.onnx",
                                    layers=[0, 2], mapping=mapping)  # fmt: skip
    assert not second.created and second.key == first.key
    other = get_or_compile_capture(cache_root=root, model="mlp", base_dir=base_dir, graph="onnx/model.onnx",
                                   layers=[1], mapping=mapping)  # fmt: skip
    assert other.key != first.key
    assert not [p for p in (root / "mlp" / CAPTURE_DIRECTORY).iterdir() if p.name.startswith(".tmp-")]
    full = open_capture(first.directory, verify="full", base_dir=base_dir)
    assert full.key == first.key
    np.testing.assert_allclose(
        run_outputs(first.graph_path, {"x": _x(1)}, [site_tensor(0)])[site_tensor(0)],
        reference(_x(1))[site_tensor(0)],
        rtol=1e-5,
        atol=1e-5,
    )


def test_capture_cache_detects_tampering(bundle, tmp_path):
    base_dir, _, _, mapping = bundle
    root = tmp_path / "cache"
    entry = get_or_compile_capture(cache_root=root, model="mlp", base_dir=base_dir, graph="onnx/model.onnx",
                                   layers=[1], mapping=mapping)  # fmt: skip
    graph = entry.graph_path
    data = bytearray(graph.read_bytes())
    data[-1] ^= 1
    graph.write_bytes(bytes(data))
    with pytest.raises(InterventionCaptureError):
        get_or_compile_capture(cache_root=root, model="mlp", base_dir=base_dir, graph="onnx/model.onnx", layers=[1],
                               mapping=mapping)  # fmt: skip


def test_capture_cache_detects_a_replaced_weights_file(bundle, tmp_path):
    base_dir, _, _, mapping = bundle
    root = tmp_path / "cache"
    entry = get_or_compile_capture(cache_root=root, model="mlp", base_dir=base_dir, graph="onnx/model.onnx",
                                   layers=[1], mapping=mapping)  # fmt: skip
    link = entry.directory / "model.onnx_data"
    copy = link.read_bytes()
    link.unlink()
    link.write_bytes(copy)  # same bytes, but no longer the base file
    with pytest.raises(InterventionCaptureError, match="hard link"):
        open_capture(entry.directory, base_dir=base_dir)
    assert open_capture(entry.directory, verify="full").key == entry.key


def test_wrong_external_sha256_fails_the_compile_and_leaves_nothing(bundle, tmp_path):
    base_dir, _, _, mapping = bundle
    root = tmp_path / "cache"
    with pytest.raises(contract_module.InterventionMismatchError, match="external_sha256"):
        get_or_compile_capture(cache_root=root, model="mlp", base_dir=base_dir, graph="onnx/model.onnx", layers=[1],
                               mapping=mapping, external_sha256={"onnx/model.onnx_data": "a" * 64})  # fmt: skip
    assert [p.name for p in (root / "mlp" / CAPTURE_DIRECTORY).iterdir() if not p.name.endswith(".lock")] == []


def test_capture_key_text_is_canonical():
    from model_pipeline.interventions.capture import CaptureOutput
    from model_pipeline.interventions.contract import BaseArtifact

    text = capture_key_text(
        capture_version="0.1.0",
        site_mapping_version="m/fp32/1",
        opset=17,
        precision="fp32",
        base_artifacts=[
            BaseArtifact("external_data", "onnx/b", "1" * 64, 2),
            BaseArtifact("graph", "onnx/a", "0" * 64, 1),
        ],
        outputs=[CaptureOutput(3, "residual_post", "t3"), CaptureOutput(1, "residual_post", "t1")],
    )
    assert text == (
        "starling-intervention-capture-key 1\ncapture_version\t0.1.0\nsite_mapping_version\tm/fp32/1\nopset\t17\n"
        f"precision\tfp32\nbase\tgraph\tonnx/a\t{'0' * 64}\t1\nbase\texternal_data\tonnx/b\t{'1' * 64}\t2\n"
        "output\t1\tresidual_post\tt1\noutput\t3\tresidual_post\tt3\n"
    )


def test_capture_manifest_parse_is_strict(bundle, tmp_path):
    base_dir, _, _, mapping = bundle
    result = compile_capture(base_dir=base_dir, graph="onnx/model.onnx", layers=[0], out_dir=tmp_path / "cap",
                             mapping=mapping)  # fmt: skip
    data = result.manifest.serialize()
    assert parse_capture_manifest(data) == result.manifest
    for bad in (
        data.replace(b'"kind": "capture"', b'"kind": "intervention"'),
        data.replace(b'"opset"', b'"extra": 1, "opset"'),
        data.replace(b'"schema_version": 1', b'"schema_version": 2'),
        data.replace(b'"outputs": [', b'"outputs": [], "unused": ['),
    ):
        with pytest.raises(contract_module.InterventionContractError):
            parse_capture_manifest(bad)
