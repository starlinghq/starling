"""Derived-artifact cache (model_pipeline.interventions.cache) on the synthetic residual-MLP bundle."""

from __future__ import annotations

import errno
import os
import threading
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from test_interventions import (  # noqa: E402
    HIDDEN,
    _build_graph,
    _mapping,
    _reference,
    _run,
    _sha,
    _site_arrays,
    _write_base,
    _x,
)

from model_pipeline.interventions import (  # noqa: E402
    COMPILER_VERSION,
    ExternalDataLinkError,
    InterventionCacheError,
    InterventionContractError,
    InterventionMismatchError,
    UnsupportedGraphError,
    cache,
    cache_key_text,
    compiler,
    contract,
    entry_directory,
    evict,
    get_or_compile,
    manifest_cache_key,
    open_entry,
)
from model_pipeline.interventions.contract import ArtifactFile, BaseArtifact, GraphInput, SiteBinding  # noqa: E402
from model_pipeline.interventions.synthetic import SyntheticSite, write_synthetic_profile  # noqa: E402

MODEL = "tiny-mlp"
GRAPH = "onnx/model.onnx"
DATA = "onnx/model.onnx_data"


@pytest.fixture
def setup(tmp_path: Path):
    """Bundle at tmp/bundle/onnx/model.onnx (+ model.onnx_data), profile with sites 0 and 2, cache at tmp/cache."""
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
    return {
        "base_dir": base_dir,
        "graph": graph,
        "graph_sha": graph_sha,
        "profile": profile,
        "mapping": _mapping(graph_sha),
        "cache_root": tmp_path / "cache",
    }


def _get(s, **kwargs):
    kwargs.setdefault("mapping", s["mapping"])
    kwargs.setdefault("profile", s["profile"].path)
    return get_or_compile(cache_root=s["cache_root"], model=MODEL, base_dir=s["base_dir"], graph=GRAPH, **kwargs)


def _tree(directory: Path) -> dict[str, tuple[int, int, int]]:
    """Relative path -> (size, mtime_ns, inode) of every file below ``directory``."""
    return {
        p.relative_to(directory).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino)
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


# --------------------------------------------------------------------------------------------------------------------
# Key


# The documented example in docs/interventions/cache.md; the C# test (InterventionCacheKeyTests) uses the same values.
_EXAMPLE_ARGS = {
    "compiler_version": "0.1.0",
    "site_mapping_version": "lfm3/fp32/1",
    "opset": 17,
    "precision": "fp32",
    "profile_sha256": "2" * 64,
    "base_artifacts": [
        BaseArtifact("external_data", "onnx/model.onnx_data", "1" * 64, 4096),
        BaseArtifact("graph", "onnx/model.onnx", "0" * 64, 1234),
    ],
    "inputs": [GraphInput("intervention_strength", "strength", "float32", (1,))],
}
_EXAMPLE_TEXT = (
    "starling-intervention-cache-key 1\n"
    "compiler_version\t0.1.0\n"
    "site_mapping_version\tlfm3/fp32/1\n"
    "opset\t17\n"
    "precision\tfp32\n"
    f"profile_sha256\t{'2' * 64}\n"
    f"base\tgraph\tonnx/model.onnx\t{'0' * 64}\t1234\n"
    f"base\texternal_data\tonnx/model.onnx_data\t{'1' * 64}\t4096\n"
    "input\tintervention_strength\tstrength\tfloat32\t[1]\n"
)
EXAMPLE_KEY = "560773f98c03601aa3bd35112beebdbd3b91198e8dbac03f94eb1134499e6579"


def test_cache_key_text_matches_documented_example():
    assert cache_key_text(**_EXAMPLE_ARGS) == _EXAMPLE_TEXT
    assert cache.cache_key(**_EXAMPLE_ARGS) == contract.sha256_bytes(_EXAMPLE_TEXT.encode("utf-8"))
    assert cache.cache_key(**_EXAMPLE_ARGS) == EXAMPLE_KEY


def test_cache_key_is_order_independent_and_sensitive_to_every_component():
    base = cache.cache_key(**_EXAMPLE_ARGS)
    reordered = dict(_EXAMPLE_ARGS, base_artifacts=list(reversed(_EXAMPLE_ARGS["base_artifacts"])))
    assert cache.cache_key(**reordered) == base
    variants = [
        {"compiler_version": "0.1.1"},
        {"site_mapping_version": "lfm3/fp32/2"},
        {"opset": 18},
        {"precision": "fp16"},
        {"profile_sha256": "3" * 64},
        {"base_artifacts": [BaseArtifact("graph", "onnx/model.onnx", "0" * 64, 1234)]},
        {
            "base_artifacts": [
                BaseArtifact("external_data", "onnx/model.onnx_data", "4" * 64, 4096),
                BaseArtifact("graph", "onnx/model.onnx", "0" * 64, 1234),
            ]
        },
        {
            "base_artifacts": [
                BaseArtifact("external_data", "onnx/model.onnx_data", "1" * 64, 4097),
                BaseArtifact("graph", "onnx/model.onnx", "0" * 64, 1234),
            ]
        },
        {"inputs": [GraphInput("alpha", "strength", "float32", (1,))]},
        {"inputs": [GraphInput("intervention_strength", "strength", "float32", ())]},
    ]
    keys = {cache.cache_key(**dict(_EXAMPLE_ARGS, **v)) for v in variants}
    assert len(keys) == len(variants)
    assert base not in keys


def test_manifest_cache_key_uses_only_manifest_fields():
    manifest = contract.InterventionManifest(
        compiler_version="0.1.0",
        site_mapping_version="lfm3/fp32/1",
        base_artifacts=tuple(reversed(_EXAMPLE_ARGS["base_artifacts"])),
        profile_id="p",
        profile_sha256="2" * 64,
        operator="directional_projection",
        opset=17,
        precision="fp32",
        inputs=tuple(_EXAMPLE_ARGS["inputs"]),
        sites=(SiteBinding(0, "residual_post", "t"),),
        derived_graph=ArtifactFile("model.onnx", "5" * 64, 10),
        derived_files=(),
    ).validate()
    assert manifest_cache_key(manifest) == cache.cache_key(**_EXAMPLE_ARGS)


# --------------------------------------------------------------------------------------------------------------------
# Miss, publish, hit


def test_miss_compiles_publishes_and_hit_reuses(setup, monkeypatch):
    first = _get(setup)
    assert first.created
    assert first.directory == entry_directory(setup["cache_root"], MODEL, first.key)
    assert first.directory.parent == setup["cache_root"] / MODEL / "interventions"
    assert first.key == manifest_cache_key(first.manifest)
    names = {p.relative_to(first.directory).as_posix() for p in first.directory.rglob("*") if p.is_file()}
    assert names == {
        "model.onnx",
        "model.onnx_data",
        "profile.json",
        "compiler-manifest.json",
        "directions/layer0-residual_post.f32",
        "directions/layer2-residual_post.f32",
    }
    # profile.json is copied byte for byte; the weights are a hard link, not a copy.
    assert (first.directory / "profile.json").read_bytes() == setup["profile"].path.read_bytes()
    assert os.path.samefile(first.directory / "model.onnx_data", setup["base_dir"] / DATA)
    assert not (first.directory / "model.onnx_data").is_symlink()
    # No temporary directory is left behind.
    assert [p.name for p in first.directory.parent.iterdir() if p.name.startswith(".tmp-")] == []

    before = _tree(first.directory)
    monkeypatch.setattr(cache, "compile_intervention", _fail("a hit must not compile"))
    second = _get(setup)
    assert not second.created
    assert second.directory == first.directory and second.key == first.key
    assert _tree(first.directory) == before


def _fail(message):
    def fail(*args, **kwargs):
        raise AssertionError(message)

    return fail


def test_cached_graph_loads_and_matches_reference(setup):
    entry = _get(setup)
    x = _x()
    base = _run(setup["graph"], x)
    derived0 = _run(entry.graph_path, x, 0.0)
    np.testing.assert_array_equal(derived0["logits"], base["logits"])
    derived1 = _run(entry.graph_path, x, 1.0)
    reference = _reference(x, _site_arrays(setup["profile"]), 1.0)
    np.testing.assert_allclose(derived1["logits"], reference["logits"], atol=1e-5)


def test_source_bundle_and_profile_never_written(setup):
    bundle_before = _tree(setup["base_dir"])
    profile_before = _tree(setup["profile"].directory)
    data_sha = _sha(setup["base_dir"] / DATA)
    entry = _get(setup)
    _get(setup)
    open_entry(entry.directory, verify="full", base_dir=setup["base_dir"])
    evict(setup["cache_root"], MODEL, entry.key)
    assert _tree(setup["base_dir"]) == bundle_before
    assert _tree(setup["profile"].directory) == profile_before
    assert _sha(setup["base_dir"] / DATA) == data_sha


def test_trusted_external_hash_skips_weight_hashing_on_hit(setup, monkeypatch):
    data_sha = _sha(setup["base_dir"] / DATA)
    entry = _get(setup, external_sha256={DATA: data_sha})
    hashed: list[Path] = []
    real = cache.sha256_file
    monkeypatch.setattr(cache, "sha256_file", lambda p: hashed.append(Path(p)) or real(p))
    monkeypatch.setattr(contract, "sha256_file", lambda p: hashed.append(Path(p)) or real(p))
    again = _get(setup, external_sha256={DATA: data_sha})
    assert again.key == entry.key and not again.created
    assert all(p.name != "model.onnx_data" for p in hashed), hashed


def test_wrong_trusted_external_hash_fails_and_publishes_nothing(setup):
    with pytest.raises(InterventionMismatchError, match="external_sha256"):
        _get(setup, external_sha256={DATA: "0" * 64})
    parent = setup["cache_root"] / MODEL / "interventions"
    assert [p for p in parent.iterdir() if p.is_dir()] == []


def test_different_profile_gets_a_different_entry(setup, tmp_path):
    first = _get(setup)
    other = write_synthetic_profile(
        tmp_path / "profile2",
        profile_id="synthetic-other",
        base_model_sha256=setup["graph_sha"],
        sites=[SyntheticSite(1, HIDDEN, 2)],
        seed=5,
    )
    second = _get(setup, profile=other.path)
    assert second.created and second.key != first.key
    assert first.directory.exists() and second.directory.exists()


def test_compiler_version_is_part_of_the_key(setup, monkeypatch):
    first = _get(setup)
    monkeypatch.setattr(cache, "COMPILER_VERSION", "0.1.0-next")
    monkeypatch.setattr(compiler, "COMPILER_VERSION", "0.1.0-next")
    second = _get(setup)
    assert second.created and second.key != first.key
    assert second.manifest.compiler_version == "0.1.0-next"
    assert first.manifest.compiler_version == COMPILER_VERSION


# --------------------------------------------------------------------------------------------------------------------
# Rejections before anything is written


def test_wrong_base_hash_rejected_before_writing(setup, tmp_path):
    other = write_synthetic_profile(
        tmp_path / "wrong", profile_id="wrong", base_model_sha256="0" * 64, sites=[SyntheticSite(0, HIDDEN, 1)]
    )
    with pytest.raises(InterventionMismatchError):
        _get(setup, profile=other.path)
    assert not setup["cache_root"].exists()


def test_unknown_graph_is_unsupported(setup):
    with pytest.raises(UnsupportedGraphError):
        get_or_compile(
            cache_root=setup["cache_root"],
            model=MODEL,
            base_dir=setup["base_dir"],
            graph=GRAPH,
            profile=setup["profile"].path,
            registry={},
        )
    assert not setup["cache_root"].exists()


@pytest.mark.parametrize("model", ["", "Upper", "../x", "a/b", ".hidden", "x" * 65])
def test_bad_model_name_rejected(setup, model):
    with pytest.raises(InterventionContractError):
        get_or_compile(
            cache_root=setup["cache_root"],
            model=model,
            base_dir=setup["base_dir"],
            graph=GRAPH,
            profile=setup["profile"].path,
            mapping=setup["mapping"],
        )


def test_cross_device_link_fails_clearly_and_cleans_up(setup, monkeypatch):
    def exdev(src, dst):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(os, "link", exdev)
    with pytest.raises(ExternalDataLinkError, match="different file systems"):
        _get(setup)
    parent = setup["cache_root"] / MODEL / "interventions"
    assert [p.name for p in parent.iterdir() if p.is_dir()] == []


def test_failure_mid_compile_leaves_no_entry_and_no_temp(setup, monkeypatch):
    real = cache._copy_profile

    def crash(*args, **kwargs):
        real(*args, **kwargs)
        raise RuntimeError("simulated crash after the compile")

    monkeypatch.setattr(cache, "_copy_profile", crash)
    with pytest.raises(RuntimeError, match="simulated"):
        _get(setup)
    parent = setup["cache_root"] / MODEL / "interventions"
    assert [p.name for p in parent.iterdir() if p.is_dir()] == []
    monkeypatch.undo()
    assert _get(setup).created


def test_stale_temp_dir_from_a_crashed_process_is_removed(setup):
    entry = _get(setup)
    evict(setup["cache_root"], MODEL, entry.key)
    parent = entry.directory.parent
    stale = parent / f".tmp-{entry.key}-99999-deadbeef"
    (stale / "sub").mkdir(parents=True)
    (stale / "model.onnx").write_bytes(b"partial")
    other_key = parent / f".tmp-{'f' * 64}-1-00000000"
    other_key.mkdir()
    assert _get(setup).created
    assert not stale.exists()
    assert other_key.exists()  # another key's build may be running; it is not ours to remove


def test_profile_path_colliding_with_external_location_rejected(setup, tmp_path):
    # A directions file at "model.onnx_data" would overwrite the hard link.
    profile_dir = tmp_path / "collide"
    p = write_synthetic_profile(
        profile_dir, profile_id="collide", base_model_sha256=setup["graph_sha"], sites=[SyntheticSite(0, HIDDEN, 1)]
    )
    text = p.path.read_text()
    old = "directions/layer0-residual_post.f32"
    (profile_dir / "model.onnx_data").write_bytes((profile_dir / old).read_bytes())
    p.path.write_text(text.replace(old, "model.onnx_data"))
    with pytest.raises(compiler.InterventionCompileError, match="model.onnx_data"):
        _get(setup, profile=p.path)


# --------------------------------------------------------------------------------------------------------------------
# Corruption and mismatch on a hit: always a clear failure, never a silent recompile


def _corrupt_cases():
    def flip_graph(d: Path, s):
        data = bytearray((d / "model.onnx").read_bytes())
        data[len(data) // 2] ^= 0x01
        (d / "model.onnx").write_bytes(bytes(data))

    def truncate_graph(d: Path, s):
        (d / "model.onnx").write_bytes((d / "model.onnx").read_bytes()[:-1])

    def missing_manifest(d: Path, s):
        (d / "compiler-manifest.json").unlink()

    def edit_profile(d: Path, s):
        (d / "profile.json").write_bytes((d / "profile.json").read_bytes() + b" ")

    def tamper_directions(d: Path, s):
        path = d / "directions/layer0-residual_post.f32"
        data = bytearray(path.read_bytes())
        data[0] ^= 0x01
        path.write_bytes(bytes(data))

    def swap_data_file(d: Path, s):
        link = d / "model.onnx_data"
        data = link.read_bytes()
        link.unlink()
        link.write_bytes(data)  # same bytes, but a copy: no longer the bundle's inode

    def missing_data_file(d: Path, s):
        (d / "model.onnx_data").unlink()

    return {
        "flip_graph": flip_graph,
        "truncate_graph": truncate_graph,
        "missing_manifest": missing_manifest,
        "edit_profile": edit_profile,
        "tamper_directions": tamper_directions,
        "swap_data_file": swap_data_file,
        "missing_data_file": missing_data_file,
    }


@pytest.mark.parametrize("case", sorted(_corrupt_cases()))
def test_corrupt_entry_fails_clearly_and_is_kept(setup, case, monkeypatch):
    entry = _get(setup)
    _corrupt_cases()[case](entry.directory, setup)
    monkeypatch.setattr(cache, "compile_intervention", _fail("corruption must not trigger a recompile"))
    with pytest.raises(InterventionCacheError):
        _get(setup)
    assert entry.directory.exists()
    # After eviction the next call compiles a fresh, valid entry.
    monkeypatch.undo()
    assert evict(setup["cache_root"], MODEL, entry.key)
    assert _get(setup).created


def test_base_replaced_by_rename_leaves_a_stale_link_that_is_detected(setup):
    entry = _get(setup)
    data = setup["base_dir"] / DATA
    replacement = data.with_name("replacement")
    replacement.write_bytes(data.read_bytes())
    os.replace(replacement, data)  # same bytes, new inode: the cached link still points at the old inode
    with pytest.raises(InterventionCacheError, match="hard link"):
        open_entry(entry.directory, base_dir=setup["base_dir"])
    # Without the base directory the fast check cannot see it; the full check hashes and accepts identical bytes.
    open_entry(entry.directory)
    open_entry(entry.directory, verify="full", base_dir=setup["base_dir"])


def test_base_rewritten_in_place_is_detected_by_full_verification(setup):
    entry = _get(setup)
    data = setup["base_dir"] / DATA
    raw = bytearray(data.read_bytes())
    raw[-1] ^= 0x01
    with open(data, "r+b") as f:  # in place: the link shares the inode and sees the new bytes
        f.write(bytes(raw))
    open_entry(entry.directory, base_dir=setup["base_dir"])  # fast: same inode, same size
    with pytest.raises(InterventionCacheError, match="SHA-256"):
        open_entry(entry.directory, verify="full")


def test_directory_name_must_match_manifest_key(setup):
    entry = _get(setup)
    wrong = entry.directory.with_name("e" * 64)
    os.rename(entry.directory, wrong)
    with pytest.raises(InterventionCacheError, match="named"):
        open_entry(wrong)
    with pytest.raises(InterventionCacheError, match="cache key"):
        open_entry(wrong, expected_key="d" * 64)


def test_open_entry_rejects_unknown_verification(setup):
    entry = _get(setup)
    with pytest.raises(ValueError):
        open_entry(entry.directory, verify="none")


# --------------------------------------------------------------------------------------------------------------------
# Concurrency


def test_concurrent_compiles_of_one_key_compile_once(setup, monkeypatch):
    calls = []
    real = cache.compile_intervention
    gate = threading.Barrier(4)

    def counting(*args, **kwargs):
        calls.append(threading.get_ident())
        return real(*args, **kwargs)

    monkeypatch.setattr(cache, "compile_intervention", counting)
    results, errors = [], []

    def worker():
        try:
            gate.wait()
            results.append(_get(setup))
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(calls) == 1
    assert sum(r.created for r in results) == 1
    assert len({r.directory for r in results}) == 1


def test_lock_timeout_is_reported(setup):
    key_dir = setup["cache_root"] / MODEL / "interventions"
    key = cache._expected_key(
        base_dir=setup["base_dir"].absolute(),
        graph=GRAPH,
        graph_bytes=setup["graph"].read_bytes(),
        profile=setup["profile"],
        mapping=setup["mapping"],
        external_sha256=None,
        strength_input=compiler.STRENGTH_INPUT_NAME,
    )[0]
    with cache._FileLock(key_dir / f"{key}.lock", 1.0), pytest.raises(TimeoutError, match="cache lock"):
        _get(setup, lock_timeout=0.2)
