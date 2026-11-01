"""Intervention graph compiler (model_pipeline.interventions) on synthetic residual-MLP graphs with external data."""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import re
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from model_pipeline.interventions import (  # noqa: E402
    COMPILER_VERSION,
    STRENGTH_INPUT_NAME,
    DerivedGraphError,
    ExternalDataLinkError,
    InterventionCompileError,
    InterventionContractError,
    InterventionMismatchError,
    MappedSite,
    SiteMapping,
    UnsupportedGraphError,
    compile_intervention,
    compiler,
    contract,
    external_data,
    load_manifest,
    lookup_site_mapping,
)
from model_pipeline.interventions.synthetic import (  # noqa: E402
    SyntheticSite,
    random_orthonormal,
    write_synthetic_profile,
)
from model_pipeline.interventions.validate import (  # noqa: E402
    check_topological_order,
    max_abs_difference,
    validate_derived_model,
)

REPO = Path(__file__).resolve().parents[2]
HIDDEN = 16
FFN = 32
VOCAB = 11
LAYERS = 3


# --------------------------------------------------------------------------------------------------------------------
# Synthetic residual MLP: h_{i+1} = h_i + W2_i relu(W1_i (h_i * g_i)); logits = h_L Wout.
# Tensor "layer{i}/residual_post" is h_{i+1}; it feeds layer i+1's norm and residual Add (two consumers), and layer 0's
# is also the graph output "hidden_states_0".


def _weights(seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    w: dict[str, np.ndarray] = {}
    for i in range(LAYERS):
        w[f"g{i}"] = (1.0 + 0.1 * rng.standard_normal(HIDDEN)).astype(np.float32)
        w[f"w1_{i}"] = (rng.standard_normal((HIDDEN, FFN)) / np.sqrt(HIDDEN)).astype(np.float32)
        w[f"w2_{i}"] = (rng.standard_normal((FFN, HIDDEN)) / np.sqrt(FFN)).astype(np.float32)
    w["wout"] = (rng.standard_normal((HIDDEN, VOCAB)) / np.sqrt(HIDDEN)).astype(np.float32)
    return w


def _consumers(layer: int) -> tuple[str, ...]:
    return ("lm_head",) if layer == LAYERS - 1 else (f"layer{layer + 1}/norm", f"layer{layer + 1}/residual_add")


def _build_graph(dtype=np.float32, opset: int = 17) -> onnx.ModelProto:
    elem = TensorProto.FLOAT if dtype == np.float32 else TensorProto.FLOAT16
    w = _weights()
    nodes = []
    initializers = [numpy_helper.from_array(v.astype(dtype), k) for k, v in w.items()]
    h = "x"
    for i in range(LAYERS):
        p = f"layer{i}"
        nodes += [
            helper.make_node("Mul", [h, f"g{i}"], [f"{p}/normed"], name=f"{p}/norm"),
            helper.make_node("MatMul", [f"{p}/normed", f"w1_{i}"], [f"{p}/up"], name=f"{p}/up_proj"),
            helper.make_node("Relu", [f"{p}/up"], [f"{p}/act"], name=f"{p}/relu"),
            helper.make_node("MatMul", [f"{p}/act", f"w2_{i}"], [f"{p}/down"], name=f"{p}/down_proj"),
            helper.make_node("Add", [h, f"{p}/down"], [f"{p}/residual_post"], name=f"{p}/residual_add"),
        ]
        h = f"{p}/residual_post"
    nodes.append(helper.make_node("MatMul", [h, "wout"], ["logits"], name="lm_head"))
    nodes.append(helper.make_node("Identity", ["layer0/residual_post"], ["hidden_states_0"], name="tap0"))
    graph = helper.make_graph(
        nodes,
        "tiny-residual-mlp",
        [helper.make_tensor_value_info("x", elem, ["batch", "sequence", HIDDEN])],
        [
            helper.make_tensor_value_info("logits", elem, ["batch", "sequence", VOCAB]),
            helper.make_tensor_value_info("hidden_states_0", elem, ["batch", "sequence", HIDDEN]),
        ],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], ir_version=8)
    onnx.checker.check_model(model)
    return model


def _reference(
    x: np.ndarray, sites: dict[int, tuple[np.ndarray, np.ndarray]], strength: float
) -> dict[str, np.ndarray]:
    """NumPy float64 forward pass with y' = y - a ((y Q^T) * s) Q after each listed layer."""
    w = {k: v.astype(np.float64) for k, v in _weights().items()}
    h = x.astype(np.float64)
    tap = None
    for i in range(LAYERS):
        h_next = h + np.maximum((h * w[f"g{i}"]) @ w[f"w1_{i}"], 0.0) @ w[f"w2_{i}"]
        if i == 0:
            tap = h_next  # the graph output reads the original tensor
        if i in sites:
            q, s = (a.astype(np.float64) for a in sites[i])
            h_next = h_next - strength * (((h_next @ q.T) * s) @ q)
        h = h_next
    return {"logits": h @ w["wout"], "hidden_states_0": tap}


def _write_base(directory: Path, model: onnx.ModelProto, graph_name: str = "model.onnx") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / graph_name
    onnx.save_model(
        model,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{graph_name}_data",
        size_threshold=0,
    )
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(graph_sha: str, layers=range(LAYERS), precision: str = "fp32") -> SiteMapping:
    return SiteMapping(
        model="tiny-residual-mlp",
        version=f"tiny-mlp/{precision}/1",
        base_graph_sha256=graph_sha,
        precision=precision,
        sites=tuple(MappedSite(i, "residual_post", f"layer{i}/residual_post", HIDDEN, _consumers(i)) for i in layers),
    )


@pytest.fixture
def bundle(tmp_path: Path):
    """Base bundle at tmp/bundle/onnx/model.onnx (+ model.onnx_data), profile with sites at layers 0 and 2."""
    base_dir = tmp_path / "bundle"
    graph = _write_base(base_dir / "onnx", _build_graph())
    graph_sha = _sha(graph)
    profile = write_synthetic_profile(
        tmp_path / "profile",
        profile_id="synthetic-tiny",
        base_model_sha256=graph_sha,
        sites=[SyntheticSite(0, HIDDEN, 3, scales=[1.0, 0.5, 2.0]), SyntheticSite(2, HIDDEN, 2)],
        seed=11,
    )
    return base_dir, graph, graph_sha, profile


def _compile(bundle, out: Path, **kwargs):
    base_dir, _, graph_sha, profile = bundle
    kwargs.setdefault("mapping", _mapping(graph_sha))
    return compile_intervention(base_dir=base_dir, graph="onnx/model.onnx", profile=profile.path, out_dir=out, **kwargs)


def _site_arrays(profile) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    return {
        s.layer: (profile.load_directions(s), np.asarray(s.scales, dtype=np.float32)) for s in profile.profile.sites
    }


def _x(seed: int = 3, batch: int = 2, seq: int = 5) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((batch, seq, HIDDEN)).astype(np.float32)


def _run(path: Path, x: np.ndarray, strength: float | None = None) -> dict[str, np.ndarray]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {"x": x}
    if strength is not None:
        feeds[STRENGTH_INPUT_NAME] = np.asarray([strength], dtype=np.float32)
    names = [o.name for o in session.get_outputs()]
    return dict(zip(names, session.run(names, feeds), strict=True))


# --------------------------------------------------------------------------------------------------------------------
# Numerics


def test_strength_zero_matches_original_exactly(bundle, tmp_path):
    result = _compile(bundle, tmp_path / "derived")
    x = _x()
    base = _run(bundle[1], x)
    derived = _run(result.graph_path, x, 0.0)
    for name in base:
        np.testing.assert_array_equal(derived[name], base[name])


@pytest.mark.parametrize("strength", [1.0, 0.5, -0.75])
def test_strength_matches_numpy_projection_reference(bundle, tmp_path, strength):
    result = _compile(bundle, tmp_path / "derived")
    x = _x()
    derived = _run(result.graph_path, x, strength)
    reference = _reference(x, _site_arrays(bundle[3]), strength)
    np.testing.assert_allclose(derived["logits"], reference["logits"], rtol=1e-5, atol=1e-5)
    # The unlisted consumer (graph output) keeps reading the original tensor.
    np.testing.assert_allclose(derived["hidden_states_0"], reference["hidden_states_0"], rtol=1e-5, atol=1e-5)
    # And the intervention actually changes the result.
    base = _run(bundle[1], x)
    assert np.max(np.abs(derived["logits"] - base["logits"])) > 1e-2


def test_strength_one_removes_the_span_at_the_site(tmp_path):
    """With orthonormal Q, s = 1 and alpha = 1 the site tensor has no component left in span(Q)."""
    base_dir = tmp_path / "bundle"
    model = _build_graph()
    model.graph.output.append(
        helper.make_tensor_value_info("layer2/residual_post", TensorProto.FLOAT, ["batch", "sequence", HIDDEN])
    )
    graph = _write_base(base_dir, model)
    profile = write_synthetic_profile(
        tmp_path / "p", profile_id="span", base_model_sha256=_sha(graph), sites=[SyntheticSite(2, HIDDEN, 4)]
    )
    mapping = _mapping(_sha(graph), layers=[2])
    result = compile_intervention(
        base_dir=base_dir, graph="model.onnx", profile=profile.path, out_dir=tmp_path / "d", mapping=mapping
    )
    # The graph output reads the original site tensor (unlisted consumer); logits read the intervened one.
    session = ort.InferenceSession(str(result.graph_path), providers=["CPUExecutionProvider"])
    x = _x()
    feeds = {"x": x, STRENGTH_INPUT_NAME: np.asarray([1.0], dtype=np.float32)}
    original, logits = session.run(["layer2/residual_post", "logits"], feeds)
    q = profile.load_directions(profile.profile.sites[0]).astype(np.float64)
    intervened = original - ((original @ q.T) @ q)
    np.testing.assert_allclose(logits, intervened @ _weights()["wout"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(intervened @ q.T, 0.0, atol=1e-5)
    assert np.max(np.abs(original @ q.T)) > 1e-2


def test_strength_input_is_isolated_per_run(bundle, tmp_path):
    result = _compile(bundle, tmp_path / "derived")
    session = ort.InferenceSession(str(result.graph_path), providers=["CPUExecutionProvider"])
    x = _x()
    a = session.run(["logits"], {"x": x, STRENGTH_INPUT_NAME: np.asarray([1.0], np.float32)})[0]
    b = session.run(["logits"], {"x": x, STRENGTH_INPUT_NAME: np.asarray([0.0], np.float32)})[0]
    np.testing.assert_array_equal(b, _run(bundle[1], x)["logits"])
    assert not np.array_equal(a, b)


def test_float16_graph_casts_and_strength_zero_is_exact(tmp_path):
    base_dir = tmp_path / "bundle"
    graph = _write_base(base_dir, _build_graph(dtype=np.float16))
    profile = write_synthetic_profile(
        tmp_path / "p", profile_id="fp16", base_model_sha256=_sha(graph), sites=[SyntheticSite(1, HIDDEN, 2)]
    )
    result = compile_intervention(
        base_dir=base_dir,
        graph="model.onnx",
        profile=profile.path,
        out_dir=tmp_path / "d",
        mapping=_mapping(_sha(graph), layers=[1], precision="fp16"),
    )
    assert result.manifest.precision == "fp16"
    assert result.manifest.strength_input.dtype == "float32"
    x = _x().astype(np.float16)
    base = _run(graph, x)
    np.testing.assert_array_equal(_run(result.graph_path, x, 0.0)["logits"], base["logits"])
    derived = _run(result.graph_path, x, 1.0)["logits"].astype(np.float64)
    reference = _reference(x.astype(np.float32), _site_arrays(profile), 1.0)["logits"]
    np.testing.assert_allclose(derived, reference, rtol=2e-2, atol=2e-2)
    assert np.max(np.abs(derived - base["logits"])) > 1e-2
    ops = [n.op_type for n in onnx.load(str(result.graph_path), load_external_data=False).graph.node]
    assert ops.count("Cast") == 2


def test_max_abs_difference_helper(bundle, tmp_path):
    result = _compile(bundle, tmp_path / "derived")
    diffs = max_abs_difference(bundle[1], result.graph_path, {"x": _x()}, STRENGTH_INPUT_NAME, 0.0)
    assert diffs == {"logits": 0.0, "hidden_states_0": 0.0}


# --------------------------------------------------------------------------------------------------------------------
# Weights and external data


def test_original_files_unchanged_and_weights_not_copied(bundle, tmp_path):
    base_dir, graph, _, _ = bundle
    data = graph.with_name("model.onnx_data")
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(base_dir.rglob("*")) if p.is_file()}
    result = _compile(bundle, tmp_path / "derived")
    _run(result.graph_path, _x(), 1.0)
    after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in sorted(base_dir.rglob("*")) if p.is_file()}
    assert after == before

    linked = tmp_path / "derived" / "model.onnx_data"
    assert result.linked_files == (linked,)
    assert os.path.samefile(linked, data)
    assert not linked.is_symlink()
    # The derived graph holds only the small intervention initializers inline; base weights stay external.
    derived = onnx.load(str(result.graph_path), load_external_data=False)
    base = onnx.load(str(graph), load_external_data=False)
    base_refs = external_data.external_references(base)
    assert external_data.external_references(derived) == base_refs
    assert len(base_refs) == len(base.graph.initializer)
    inline = [t for t in derived.graph.initializer if t.data_location != TensorProto.EXTERNAL]
    assert {t.name.split("/")[-1] for t in inline} == {"directions", "directions_t", "scales"}
    assert all(t.name.startswith("intervention/") for t in inline)
    assert result.graph_path.stat().st_size < 4 * (3 * HIDDEN * HIDDEN + 2 * HIDDEN * 2) + 8192


def test_compile_is_deterministic(bundle, tmp_path):
    a = _compile(bundle, tmp_path / "a")
    b = _compile(bundle, tmp_path / "b")
    assert a.graph_path.read_bytes() == b.graph_path.read_bytes()
    assert a.manifest_path.read_bytes() == b.manifest_path.read_bytes()


def test_recompile_into_same_directory_reuses_link(bundle, tmp_path):
    _compile(bundle, tmp_path / "d")
    result = _compile(bundle, tmp_path / "d")
    assert os.path.samefile(result.linked_files[0], bundle[1].with_name("model.onnx_data"))


def test_existing_foreign_file_at_link_location_is_rejected(bundle, tmp_path):
    out = tmp_path / "d"
    out.mkdir()
    (out / "model.onnx_data").write_bytes(b"not the weights")
    with pytest.raises(ExternalDataLinkError, match="not a hard link"):
        _compile(bundle, out)


def test_cross_device_link_fails_clearly(bundle, tmp_path, monkeypatch):
    def refuse(src, dst):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(external_data.os, "link", refuse)
    with pytest.raises(ExternalDataLinkError, match="different file systems") as info:
        _compile(bundle, tmp_path / "d")
    assert info.value.errno == errno.EXDEV
    assert "never copied" in str(info.value) and "cache_root" in str(info.value)
    # Nothing is left behind: not the derived graph, not the directory the compile created.
    assert not (tmp_path / "d").exists()
    assert (bundle[0] / "onnx" / "model.onnx_data").is_file()


def test_cross_device_is_detected_before_anything_is_written(bundle, tmp_path, monkeypatch):
    source = (bundle[0] / "onnx" / "model.onnx_data").absolute()
    real_stat = os.stat

    class _Other:
        st_dev = -1

    def stat(path, *args, **kwargs):
        if Path(path).absolute() == source:
            return _Other()
        return real_stat(path, *args, **kwargs)

    writes = []
    monkeypatch.setattr(compiler, "_write_bytes", lambda *a: writes.append(a))
    monkeypatch.setattr(external_data.os, "stat", stat)
    with pytest.raises(ExternalDataLinkError, match="different file systems") as info:
        _compile(bundle, tmp_path / "d")
    monkeypatch.undo()
    assert info.value.errno == errno.EXDEV
    assert writes == [] and not (tmp_path / "d").exists()


def test_failed_link_of_a_later_file_removes_the_earlier_links(tmp_path, monkeypatch):
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.bin").write_bytes(b"a")
    (src / "sub" / "b.bin").write_bytes(b"b")
    real_link = os.link

    def link(a, b):
        if Path(b).name == "b.bin":
            raise OSError(errno.EACCES, "denied")
        real_link(a, b)

    monkeypatch.setattr(external_data.os, "link", link)
    with pytest.raises(ExternalDataLinkError, match="denied"):
        external_data.link_external_data(src, tmp_path / "out", ["a.bin", "sub/b.bin"])
    assert sorted(p.relative_to(tmp_path).as_posix() for p in (tmp_path / "out").rglob("*")) == []
    assert (src / "a.bin").read_bytes() == b"a"


def test_failure_after_writing_removes_partial_output(bundle, tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("simulated failure after model.onnx and the link exist")

    monkeypatch.setattr(external_data, "base_external_artifacts", boom)
    with pytest.raises(RuntimeError, match="simulated"):
        _compile(bundle, tmp_path / "parent" / "d")
    assert not (tmp_path / "parent").exists()
    assert (bundle[0] / "onnx" / "model.onnx_data").is_file()
    # An existing output directory is kept; only what the compile created goes.
    (tmp_path / "keep").mkdir()
    (tmp_path / "keep" / "other.txt").write_text("x")
    with pytest.raises(RuntimeError, match="simulated"):
        _compile(bundle, tmp_path / "keep")
    assert sorted(p.name for p in (tmp_path / "keep").iterdir()) == ["other.txt"]


def test_weights_rewritten_while_hashed_fail_the_compile(bundle, tmp_path, monkeypatch):
    real = external_data.sha256_file

    def rewrite_while_hashing(path):
        digest = real(path)
        data = Path(path).read_bytes()
        stat = os.stat(path)
        with open(path, "r+b") as f:  # in place: same inode, same size, new bytes and mtime
            f.write(bytes([data[0] ^ 0xFF]))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return digest

    monkeypatch.setattr(external_data, "sha256_file", rewrite_while_hashing)
    with pytest.raises(InterventionMismatchError, match="modified during compilation"):
        _compile(bundle, tmp_path / "d")
    assert not (tmp_path / "d").exists()


def test_write_bytes_leaves_no_temporary_file_on_failure(tmp_path, monkeypatch):
    def refuse(src, dst):
        raise OSError(errno.EIO, "simulated")

    monkeypatch.setattr(compiler.os, "replace", refuse)
    with pytest.raises(OSError, match="simulated"):
        compiler._write_bytes(tmp_path / "model.onnx", b"x")
    monkeypatch.undo()
    assert list(tmp_path.iterdir()) == []


def test_write_bytes_from_many_threads_into_one_directory(tmp_path):
    import threading

    errors = []

    def write(i):
        try:
            for _ in range(20):
                compiler._write_bytes(tmp_path / "model.onnx", bytes([i]) * 64)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert [p.name for p in tmp_path.iterdir()] == ["model.onnx"]


def test_no_link_mode_writes_graph_only(bundle, tmp_path):
    result = _compile(bundle, tmp_path / "d", link_external=False)
    assert result.linked_files == ()
    assert sorted(p.name for p in (tmp_path / "d").iterdir()) == ["compiler-manifest.json", "model.onnx"]
    # The manifest still names the external data the derived graph needs; until it is placed, verification fails.
    (file,) = result.manifest.derived_files
    assert file.path == "model.onnx_data"
    assert file.identity is None  # nothing was linked, so there is no identity to record
    with pytest.raises(InterventionMismatchError, match="does not exist"):
        contract.verify_artifact_file(result.derived_dir / file.path, file.sha256, file.size)


def test_swapped_external_data_next_to_derived_graph_fails_verification(bundle, tmp_path):
    result = _compile(bundle, tmp_path / "d")
    (file,) = result.manifest.derived_files
    link = result.derived_dir / file.path
    data = bytearray(link.read_bytes())
    data[0] ^= 1
    link.unlink()  # never write through the link: that would change the base bundle
    link.write_bytes(bytes(data))
    with pytest.raises(InterventionMismatchError, match="SHA-256"):
        contract.verify_artifact_file(link, file.sha256, file.size)


def test_output_inside_base_directory_is_rejected(bundle):
    base_dir = bundle[0]
    with pytest.raises(InterventionCompileError, match="inside the base model directory"):
        _compile(bundle, base_dir / "derived")
    with pytest.raises(InterventionCompileError, match="inside the base model directory"):
        _compile(bundle, base_dir)
    assert not (base_dir / "derived").exists()
    # An output directory that contains the bundle is rejected too: links at out_dir/<location> could land inside it.
    with pytest.raises(InterventionCompileError, match="inside the output directory"):
        _compile(bundle, base_dir.parent)
    assert not (base_dir.parent / "model.onnx").exists()


def test_escaping_external_location_is_rejected(tmp_path):
    base_dir = tmp_path / "bundle"
    graph = _write_base(base_dir, _build_graph())
    model = onnx.load(str(graph), load_external_data=False)
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location":
                entry.value = "../elsewhere/model.onnx_data"
    graph.write_bytes(model.SerializeToString())
    profile = write_synthetic_profile(
        tmp_path / "p", profile_id="esc", base_model_sha256=_sha(graph), sites=[SyntheticSite(0, HIDDEN, 1)]
    )
    with pytest.raises(InterventionContractError, match="not a plain relative path"):
        compile_intervention(
            base_dir=base_dir,
            graph="model.onnx",
            profile=profile.path,
            out_dir=tmp_path / "d",
            mapping=_mapping(_sha(graph)),
        )


# --------------------------------------------------------------------------------------------------------------------
# Graph rewrite


def test_only_listed_consumers_are_rewired_and_no_cycle(bundle, tmp_path):
    result = _compile(bundle, tmp_path / "derived")
    derived = onnx.load(str(result.graph_path), load_external_data=False)
    check_topological_order(derived)
    by_name = {n.name: n for n in derived.graph.node}
    assert list(by_name["layer1/norm"].input) == ["intervention/layer0/residual_post/output", "g1"]
    assert list(by_name["layer1/residual_add"].input) == ["intervention/layer0/residual_post/output", "layer1/down"]
    assert list(by_name["lm_head"].input) == ["intervention/layer2/residual_post/output", "wout"]
    # Unlisted readers of the site tensors keep the original value.
    assert list(by_name["tap0"].input) == ["layer0/residual_post"]
    assert list(by_name["layer2/norm"].input) == ["layer1/residual_post", "g2"]
    # The inserted nodes read the original tensor.
    assert by_name["intervention/layer0/residual_post/MatMul_coefficients"].input[0] == "layer0/residual_post"
    assert by_name["intervention/layer0/residual_post/Sub"].input[0] == "layer0/residual_post"
    assert [i.name for i in derived.graph.input] == ["x", STRENGTH_INPUT_NAME]
    strength = derived.graph.input[1].type.tensor_type
    assert strength.elem_type == TensorProto.FLOAT
    assert [d.dim_value for d in strength.shape.dim] == [1]
    assert [o.name for o in derived.graph.output] == ["logits", "hidden_states_0"]


def test_partial_consumer_list_rewires_only_that_consumer(bundle, tmp_path):
    base_dir, graph, graph_sha, profile = bundle
    sites = [MappedSite(0, "residual_post", "layer0/residual_post", HIDDEN, ("layer1/residual_add",))]
    sites.append(MappedSite(2, "residual_post", "layer2/residual_post", HIDDEN, _consumers(2)))
    mapping = SiteMapping("tiny", "tiny/partial/1", graph_sha, "fp32", tuple(sites))
    result = _compile(bundle, tmp_path / "d", mapping=mapping)
    by_name = {n.name: n for n in onnx.load(str(result.graph_path), load_external_data=False).graph.node}
    assert by_name["layer1/norm"].input[0] == "layer0/residual_post"
    assert by_name["layer1/residual_add"].input[0] == "intervention/layer0/residual_post/output"


def _with_ort_only_op(model: onnx.ModelProto) -> onnx.ModelProto:
    """A copy whose first node is an operator ONNX Runtime registers in the default domain but onnx does not know."""
    copy = onnx.ModelProto()
    copy.CopyFrom(model)
    copy.graph.node[0].op_type = "SimplifiedLayerNormalization"
    return copy


def test_checker_is_skipped_only_when_it_rejects_the_base_graph(caplog):
    base = _with_ort_only_op(_build_graph())
    with pytest.raises(onnx.checker.ValidationError):
        onnx.checker.check_model(base)
    derived = onnx.ModelProto()
    derived.CopyFrom(base)
    with caplog.at_level(logging.WARNING, logger="model_pipeline.interventions.validate"):
        report = validate_derived_model(base, derived, [])
    assert report.onnx_checker == contract.CHECKER_SKIPPED_BASE_REJECTED
    assert "SimplifiedLayerNormalization" in report.detail and "\n" not in report.detail
    assert any("checked structurally only" in r.getMessage() for r in caplog.records)
    assert validate_derived_model(_build_graph(), _build_graph(), []).onnx_checker == contract.CHECKER_PASSED
    # The structural checks still run when the checker cannot judge the graph.
    nodes = list(derived.graph.node)
    del derived.graph.node[:]
    derived.graph.node.extend([nodes[1], nodes[0], *nodes[2:]])
    with pytest.raises(DerivedGraphError, match="topologically ordered"):
        validate_derived_model(base, derived, [])


def test_checker_failure_of_the_derived_graph_alone_is_an_error():
    base = _build_graph()
    onnx.checker.check_model(base)
    with pytest.raises(DerivedGraphError, match="rejects the derived graph but not its base"):
        validate_derived_model(base, _with_ort_only_op(base), [])


def test_topological_order_check_detects_misordered_nodes():
    model = _build_graph()
    nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend([nodes[1], nodes[0], *nodes[2:]])
    with pytest.raises(DerivedGraphError, match="topologically ordered"):
        check_topological_order(model)


@pytest.mark.parametrize(
    ("site", "message"),
    [
        (MappedSite(0, "residual_post", "missing/tensor", HIDDEN, ("lm_head",)), "does not exist"),
        (MappedSite(0, "residual_post", "w1_0", HIDDEN, ("layer0/up_proj",)), "initializer"),
        (MappedSite(0, "residual_post", "layer0/residual_post", HIDDEN, ("nope",)), "consumer node 'nope'"),
        (MappedSite(0, "residual_post", "layer0/residual_post", HIDDEN, ("lm_head",)), "does not consume"),
        (MappedSite(0, "residual_post", "layer0/residual_post", HIDDEN + 1, ("lm_head",)), "hidden_size"),
    ],
)
def test_bad_site_bindings_are_rejected(bundle, tmp_path, site, message):
    graph_sha = bundle[2]
    sites = (site, MappedSite(2, "residual_post", "layer2/residual_post", HIDDEN, _consumers(2)))
    with pytest.raises(InterventionCompileError, match=message):
        _compile(bundle, tmp_path / "d", mapping=SiteMapping("tiny", "tiny/bad/1", graph_sha, "fp32", sites))
    assert not (tmp_path / "d" / "model.onnx").exists()


def test_site_last_dimension_must_match(tmp_path):
    base_dir = tmp_path / "bundle"
    graph = _write_base(base_dir, _build_graph())
    profile = write_synthetic_profile(
        tmp_path / "p", profile_id="dims", base_model_sha256=_sha(graph), sites=[SyntheticSite(0, VOCAB, 1)]
    )
    mapping = SiteMapping(
        "tiny",
        "tiny/dims/1",
        _sha(graph),
        "fp32",
        (MappedSite(0, "residual_post", "layer0/up", VOCAB, ("layer0/relu",)),),
    )
    with pytest.raises(InterventionCompileError, match=f"last dimension {FFN}"):
        compile_intervention(
            base_dir=base_dir, graph="model.onnx", profile=profile.path, out_dir=tmp_path / "d", mapping=mapping
        )


def test_strength_name_collision_is_rejected(bundle, tmp_path):
    with pytest.raises(InterventionCompileError, match="already has a value named"):
        _compile(bundle, tmp_path / "d", strength_input="logits")


def test_opset_below_minimum_is_rejected(tmp_path):
    base_dir = tmp_path / "bundle"
    model = _build_graph()
    model.opset_import[0].version = 6
    graph = _write_base(base_dir, model)
    profile = write_synthetic_profile(
        tmp_path / "p", profile_id="old", base_model_sha256=_sha(graph), sites=[SyntheticSite(0, HIDDEN, 1)]
    )
    with pytest.raises(InterventionCompileError, match="opset 6"):
        compile_intervention(
            base_dir=base_dir,
            graph="model.onnx",
            profile=profile.path,
            out_dir=tmp_path / "d",
            mapping=_mapping(_sha(graph)),
        )


def test_ir_version_below_four_is_rejected(tmp_path):
    base_dir = tmp_path / "bundle"
    model = _build_graph()
    model.ir_version = 3
    for init in model.graph.initializer:
        model.graph.input.append(helper.make_tensor_value_info(init.name, init.data_type, list(init.dims)))
    graph = _write_base(base_dir, model)
    profile = write_synthetic_profile(
        tmp_path / "p", profile_id="ir3", base_model_sha256=_sha(graph), sites=[SyntheticSite(0, HIDDEN, 1)]
    )
    with pytest.raises(InterventionCompileError, match="IR version 3"):
        compile_intervention(
            base_dir=base_dir,
            graph="model.onnx",
            profile=profile.path,
            out_dir=tmp_path / "d",
            mapping=_mapping(_sha(graph)),
        )
    assert not (tmp_path / "d").exists()


# --------------------------------------------------------------------------------------------------------------------
# Identities and site mappings


def test_wrong_base_graph_is_rejected_before_anything_is_written(bundle, tmp_path):
    base_dir, graph, graph_sha, _ = bundle
    profile = write_synthetic_profile(
        tmp_path / "other", profile_id="other", base_model_sha256="0" * 64, sites=[SyntheticSite(0, HIDDEN, 1)]
    )
    with pytest.raises(InterventionMismatchError, match="targets base graph 0000"):
        compile_intervention(
            base_dir=base_dir,
            graph="onnx/model.onnx",
            profile=profile.path,
            out_dir=tmp_path / "d",
            mapping=_mapping(graph_sha),
        )
    assert not (tmp_path / "d").exists()


def test_mapping_for_another_graph_is_rejected(bundle, tmp_path):
    with pytest.raises(InterventionMismatchError, match="site mapping"):
        _compile(bundle, tmp_path / "d", mapping=_mapping("1" * 64))


def test_unknown_graph_hash_is_unsupported(bundle, tmp_path):
    with pytest.raises(UnsupportedGraphError, match="unsupported"):
        _compile(bundle, tmp_path / "d", mapping=None)
    with pytest.raises(UnsupportedGraphError):
        lookup_site_mapping(bundle[2])
    assert lookup_site_mapping(bundle[2], {bundle[2]: _mapping(bundle[2])}).version == "tiny-mlp/fp32/1"


def test_registry_lookup_is_used_when_no_mapping_is_given(bundle, tmp_path):
    graph_sha = bundle[2]
    result = _compile(bundle, tmp_path / "d", mapping=None, registry={graph_sha: _mapping(graph_sha)})
    assert result.manifest.site_mapping_version == "tiny-mlp/fp32/1"


def test_profile_site_missing_from_mapping_is_unsupported(bundle, tmp_path):
    with pytest.raises(UnsupportedGraphError, match="no site"):
        _compile(bundle, tmp_path / "d", mapping=_mapping(bundle[2], layers=[0]))


def test_site_mapping_rejects_duplicates():
    a = MappedSite(0, "residual_post", "t", HIDDEN, ("n",))
    with pytest.raises(InterventionContractError, match="twice"):
        SiteMapping("m", "m/1", "0" * 64, "fp32", (a, a))
    with pytest.raises(InterventionContractError, match="two sites"):
        SiteMapping("m", "m/1", "0" * 64, "fp32", (a, MappedSite(1, "residual_post", "t", HIDDEN, ("n",))))
    with pytest.raises(InterventionContractError, match="no consumers"):
        MappedSite(0, "residual_post", "t", HIDDEN, ())
    with pytest.raises(InterventionContractError, match="unknown site target"):
        MappedSite(0, "residual_pre", "t", HIDDEN, ("n",))


# --------------------------------------------------------------------------------------------------------------------
# Manifest


def test_manifest_contents(bundle, tmp_path):
    base_dir, graph, graph_sha, profile = bundle
    result = _compile(bundle, tmp_path / "derived")
    manifest = load_manifest(result.manifest_path)
    assert manifest == result.manifest
    assert manifest.compiler_version == COMPILER_VERSION
    assert manifest.profile_id == "synthetic-tiny"
    assert manifest.profile_sha256 == _sha(profile.path)
    assert manifest.opset == 17
    assert manifest.precision == "fp32"
    data = graph.with_name("model.onnx_data")
    assert [(a.role, a.path, a.sha256, a.size) for a in manifest.base_artifacts] == [
        ("graph", "onnx/model.onnx", graph_sha, graph.stat().st_size),
        ("external_data", "onnx/model.onnx_data", _sha(data), data.stat().st_size),
    ]
    assert [(i.name, i.role, i.dtype, i.shape) for i in manifest.inputs] == [
        (STRENGTH_INPUT_NAME, "strength", "float32", (1,))
    ]
    assert [(s.layer, s.target, s.tensor) for s in manifest.sites] == [
        (0, "residual_post", "layer0/residual_post"),
        (2, "residual_post", "layer2/residual_post"),
    ]
    assert manifest.derived_graph.path == "model.onnx"
    assert manifest.derived_graph.sha256 == _sha(result.graph_path)
    assert manifest.derived_graph.size == result.graph_path.stat().st_size
    # The hard-linked external data is a derived file: what ONNX Runtime reads next to model.onnx is hash-checked.
    assert [(f.path, f.sha256, f.size) for f in manifest.derived_files] == [
        ("model.onnx_data", _sha(data), data.stat().st_size)
    ]
    st = data.stat()
    assert manifest.derived_files[0].identity == contract.FileIdentity(st.st_dev, st.st_ino, st.st_mtime_ns)
    assert manifest.validation == contract.ManifestValidation(contract.CHECKER_PASSED)
    contract.verify_manifest_profile(manifest, profile)
    contract.verify_artifact_file(result.graph_path, manifest.derived_graph.sha256, manifest.derived_graph.size)
    for file in manifest.derived_files:
        contract.verify_artifact_file(result.derived_dir / file.path, file.sha256, file.size)
    for artifact in manifest.base_artifacts:
        contract.verify_artifact_file(base_dir / artifact.path, artifact.sha256, artifact.size)

    text = result.manifest_path.read_bytes()
    assert text.endswith(b"}\n") and b"\r" not in text and not text.startswith(b"\xef\xbb\xbf")
    keys = list(json.loads(text))
    assert keys == list(contract._MANIFEST_KEYS)
    assert contract.parse_manifest(text).serialize() == text


def _doc_json_blocks() -> list[bytes]:
    text = (REPO / "docs/interventions/profile-format.md").read_text(encoding="utf-8")
    return [(m + "\n").encode() for m in re.findall(r"```json\n(.*?)\n```", text, flags=re.S)]


def test_documented_examples_parse_and_reserialize_identically():
    profile_text, manifest_text = _doc_json_blocks()[:2]
    profile = contract.parse_profile(profile_text)
    assert profile.id == "synthetic-demo" and profile.sites[0].scales == (1.0, 0.5)
    assert contract.parse_manifest(manifest_text).serialize() == manifest_text


def test_manifest_validation_matches_csharp_rules():
    _, manifest_text = _doc_json_blocks()[:2]
    document = json.loads(manifest_text)

    def rejected(mutate, path):
        doc = json.loads(json.dumps(document))
        mutate(doc)
        with pytest.raises(InterventionContractError) as info:
            contract.parse_manifest(json.dumps(doc).encode())
        assert info.value.json_path == path

    rejected(lambda d: d.update(schema_version=3), "schema_version")
    rejected(lambda d: d.update(schema_version=2), "$")  # a schema 2 manifest has "profiles", not "profile_id"
    rejected(lambda d: d.update(extra=1), "$")
    rejected(lambda d: d["inputs"][0].update(shape=[1, 1]), "inputs")
    rejected(lambda d: d["inputs"][0].update(dtype="float16"), "inputs[0].dtype")
    rejected(lambda d: d.update(opset=17.0), "opset")
    rejected(lambda d: d["base_artifacts"][1].update(role="graph"), "base_artifacts")
    rejected(
        lambda d: d["sites"].append({"layer": 2, "target": "residual_post", "tensor": d["sites"][0]["tensor"]}), "sites"
    )
    rejected(lambda d: d["derived_files"].append(dict(d["derived_graph"])), "derived_files")
    rejected(lambda d: d["derived_graph"].update(path="../model.onnx"), "derived_graph.path")
    rejected(lambda d: d.update(compiler_version="-bad"), "compiler_version")
    identity = {"device": 1, "inode": 2, "mtime_ns": 3}
    rejected(lambda d: d["derived_graph"].update(identity=identity), "derived_graph")
    rejected(lambda d: d["derived_files"][0]["identity"].update(inode=-1), "derived_files[0].identity.inode")
    rejected(lambda d: d["derived_files"][0]["identity"].update(device=1.0), "derived_files[0].identity.device")
    rejected(lambda d: d["derived_files"][0]["identity"].pop("mtime_ns"), "derived_files[0].identity")
    rejected(lambda d: d["derived_files"][0]["identity"].update(ctime_ns=1), "derived_files[0].identity")
    rejected(lambda d: d["validation"].update(onnx_checker="skipped"), "validation.onnx_checker")
    rejected(lambda d: d["validation"].update(detail="two\nlines"), "validation.detail")
    rejected(lambda d: d["validation"].update(detail=""), "validation.detail")
    rejected(lambda d: d["validation"].update(detail="x" * 513), "validation.detail")
    rejected(lambda d: d["validation"].pop("onnx_checker"), "validation")
    rejected(lambda d: d["validation"].update(extra=1), "validation")
    rejected(lambda d: d.update(validation="passed"), "validation")


def test_identity_and_validation_are_optional_and_round_trip():
    _, manifest_text = _doc_json_blocks()[:2]
    manifest = contract.parse_manifest(manifest_text)
    assert manifest.derived_files[0].identity == contract.FileIdentity(66307, 1234567, 1761000000123456789)
    assert manifest.validation.onnx_checker == contract.CHECKER_SKIPPED_BASE_REJECTED
    document = json.loads(manifest_text)
    del document["validation"]
    del document["derived_files"][0]["identity"]
    document["derived_files"][0]["identity"] = {"device": 0, "inode": 0, "mtime_ns": -(2**63)}
    text = contract.serialize_manifest_document(document)
    assert contract.parse_manifest(text).serialize() == text
    del document["derived_files"][0]["identity"]
    text = contract.serialize_manifest_document(document)
    bare = contract.parse_manifest(text)
    assert bare.validation is None and bare.derived_files[0].identity is None
    assert bare.serialize() == text


# --------------------------------------------------------------------------------------------------------------------
# Profile and directions reading


def _profile_doc(**overrides) -> dict:
    site = {
        "layer": 1,
        "target": "residual_post",
        "hidden_size": 4,
        "rank": 2,
        "dtype": "float32",
        "directions_file": "d.f32",
        "directions_sha256": "1" * 64,
        "scales": [1, 0.5],
    }
    site.update(overrides.pop("site", {}))
    doc = {
        "schema_version": 1,
        "id": "p",
        "base_model_sha256": "0" * 64,
        "operator": "directional_projection",
        "sites": [site],
    }
    doc.update(overrides)
    return doc


@pytest.mark.parametrize(
    ("text", "path"),
    [
        (json.dumps(_profile_doc(schema_version=3)), "schema_version"),
        (json.dumps(_profile_doc(operator="directional_addition")), "operator"),
        (json.dumps(_profile_doc(schema_version=1.0)), "schema_version"),
        (json.dumps(_profile_doc(schema_version="1")), "schema_version"),
        (json.dumps(_profile_doc(extra=True)), "$"),
        (json.dumps(_profile_doc(operator="steer")), "operator"),
        (json.dumps(_profile_doc(id="Upper")), "id"),
        (json.dumps(_profile_doc(base_model_sha256="A" * 64)), "base_model_sha256"),
        (json.dumps(_profile_doc(sites=[])), "sites"),
        (json.dumps(_profile_doc(site={"layer": 1.0})), "sites[0].layer"),
        (json.dumps(_profile_doc(site={"layer": True})), "sites[0].layer"),
        (json.dumps(_profile_doc(site={"rank": 5})), "sites[0].rank"),
        (json.dumps(_profile_doc(site={"target": "residual_pre"})), "sites[0].target"),
        (json.dumps(_profile_doc(site={"dtype": "float16"})), "sites[0].dtype"),
        (json.dumps(_profile_doc(site={"scales": [1]})), "sites[0].scales"),
        (json.dumps(_profile_doc(site={"scales": [1, 1e39]})), "sites[0].scales[1]"),
        (json.dumps(_profile_doc(site={"scales": [1, "2"]})), "sites[0].scales[1]"),
        (json.dumps(_profile_doc(site={"directions_file": "../d.f32"})), "sites[0].directions_file"),
        (json.dumps(_profile_doc(site={"directions_file": "a\\d.f32"})), "sites[0].directions_file"),
        (json.dumps(_profile_doc(site={"directions_file": "/d.f32"})), "sites[0].directions_file"),
        (json.dumps(_profile_doc(site={"directions_file": "c:d.f32"})), "sites[0].directions_file"),
        (json.dumps({k: v for k, v in _profile_doc().items() if k != "operator"}), "$"),
    ],
)
def test_profile_rejections(text, path):
    with pytest.raises(InterventionContractError) as info:
        contract.parse_profile(text.encode())
    assert info.value.json_path == path


@pytest.mark.parametrize(
    "data",
    [
        b"\xef\xbb\xbf" + json.dumps(_profile_doc()).encode(),
        json.dumps(_profile_doc()).replace('"rank": 2', '"rank": 2, "rank": 2').encode(),
        json.dumps(_profile_doc()).replace("0.5", "NaN").encode(),
        json.dumps(_profile_doc()).encode() + b",",
    ],
)
def test_profile_rejects_nonstandard_json(data):
    with pytest.raises(InterventionContractError):
        contract.parse_profile(data)


def test_profile_duplicate_site_rejected():
    doc = _profile_doc()
    doc["sites"].append(dict(doc["sites"][0]))
    with pytest.raises(InterventionContractError, match="two sites"):
        contract.parse_profile(json.dumps(doc).encode())


def test_profile_hash_is_over_stored_bytes(tmp_path):
    path = tmp_path / "profile.json"
    data = json.dumps(_profile_doc(), separators=(",", ":")).encode()
    path.write_bytes(data)
    assert contract.load_profile(path).sha256 == hashlib.sha256(data).hexdigest()


def test_synthetic_profile_is_canonical_and_orthonormal(tmp_path):
    profile = write_synthetic_profile(
        tmp_path,
        profile_id="canon",
        base_model_sha256="0" * 64,
        sites=[SyntheticSite(3, 8, 2, scales=[0.1, 1 / 3]), SyntheticSite(1, 8, 1)],
    )
    text = profile.path.read_text()
    assert '"scales": [\n        0.1,\n        0.33333334\n      ]' in text
    assert [s.layer for s in profile.profile.sites] == [1, 3]
    assert list(json.loads(text)) == ["schema_version", "id", "description", "base_model_sha256", "operator", "sites"]
    q = profile.load_directions(profile.profile.sites[1])
    assert contract.max_orthonormality_error(q) < 1e-6
    assert q.dtype == np.float32 and q.shape == (2, 8)
    assert profile.profile.sites[1].scales == (float(np.float32(0.1)), float(np.float32(1 / 3)))


def test_directions_checks_size_then_hash_then_finite():
    site = contract.InterventionSite(0, "residual_post", 4, 2, "float32", "d.f32", "0" * 64, (1.0, 1.0))
    good = contract.encode_directions(random_orthonormal(2, 4, 0))
    with pytest.raises(InterventionContractError, match="bytes"):
        contract.decode_directions(good[:-1], site)  # size is checked before the hash
    with pytest.raises(InterventionMismatchError, match="SHA-256"):
        contract.decode_directions(good, site)
    bad = np.frombuffer(good, dtype="<f4").copy()
    bad[3] = np.nan
    bad_bytes = bad.astype("<f4").tobytes()
    nan_site = contract.InterventionSite(
        0, "residual_post", 4, 2, "float32", "d.f32", hashlib.sha256(bad_bytes).hexdigest(), (1.0, 1.0)
    )
    with pytest.raises(InterventionContractError, match="NaN"):
        contract.decode_directions(bad_bytes, nan_site)
    ok_site = contract.InterventionSite(
        0, "residual_post", 4, 2, "float32", "d.f32", hashlib.sha256(good).hexdigest(), (1.0, 1.0)
    )
    np.testing.assert_array_equal(contract.decode_directions(good, ok_site), random_orthonormal(2, 4, 0))


def test_directions_file_is_little_endian_row_major():
    q = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert contract.encode_directions(q) == np.arange(6, dtype="<f4").tobytes()
    assert contract.encode_directions(q)[4:8] == bytes.fromhex("0000803f")


def test_changed_directions_file_is_rejected_at_compile(bundle, tmp_path):
    profile = bundle[3]
    path = profile.directory / profile.profile.sites[0].directions_file
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(bytes(data))
    with pytest.raises(InterventionMismatchError, match="directions file"):
        _compile(bundle, tmp_path / "d")


def test_graph_input_site_is_inserted_first(bundle, tmp_path):
    base_dir, graph, graph_sha, _ = bundle
    profile = write_synthetic_profile(
        tmp_path / "p", profile_id="input-site", base_model_sha256=graph_sha, sites=[SyntheticSite(0, HIDDEN, 2)]
    )
    site = MappedSite(0, "residual_post", "x", HIDDEN, ("layer0/norm", "layer0/residual_add"))
    result = compile_intervention(
        base_dir=base_dir,
        graph="onnx/model.onnx",
        profile=profile.path,
        out_dir=tmp_path / "d",
        mapping=SiteMapping("tiny", "tiny/input/1", graph_sha, "fp32", (site,)),
    )
    derived = onnx.load(str(result.graph_path), load_external_data=False)
    assert derived.graph.node[0].name == "intervention/layer0/residual_post/MatMul_coefficients"
    x = _x()
    q = profile.load_directions(profile.profile.sites[0]).astype(np.float64)
    shifted = (x - 0.5 * ((x @ q.T) @ q)).astype(np.float32)
    np.testing.assert_allclose(
        _run(result.graph_path, x, 0.5)["logits"], _run(graph, shifted)["logits"], rtol=1e-5, atol=1e-5
    )
