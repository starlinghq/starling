"""Capture and extraction on the real LFM3-1.7B fp32 graph (env-gated; see test_intervention_lfm3.py for the variables).

- ``STARLING_LFM3_SNAPSHOT``: the LFM3-1.7B-ONNX snapshot at c680759a. With it (and the ``tokenizers`` package), the
  extraction's renderer and encoder are checked against the LFM3 template corpus, and the checked-in profile compiles.
- ``STARLING_LFM3_INTERVENTION_CACHE``: a cache root on the snapshot's file system, for the capture graph checks.
- ``STARLING_LFM3_EXTRACT_REPRODUCE=1`` additionally reruns the "concise" extraction (128 prefills, about 2 minutes on
  12 CPU threads) and requires the same selected layers, directions within cosine 0.99999 and the same separability
  numbers within rounding.

Each check loads at most one fp32 session (about 7 GB) at a time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from model_pipeline.interventions.capture import get_or_compile_capture  # noqa: E402
from model_pipeline.interventions.injection_sites import LFM3_FP32_GRAPH_SHA256, LFM3_FP32_MAPPING  # noqa: E402
from model_pipeline.interventions.lfm3_synthetic import LFM3_FP32_EXTERNAL_SHA256, LFM3_FP32_GRAPH  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONCISE = REPO / "model-pipeline/interventions/profiles/lfm3-1.7b/concise"
SNAPSHOT = os.environ.get("STARLING_LFM3_SNAPSHOT")
CACHE = os.environ.get("STARLING_LFM3_INTERVENTION_CACHE")

needs_snapshot = pytest.mark.skipif(
    not SNAPSHOT or not Path(SNAPSHOT, LFM3_FP32_GRAPH).is_file(),
    reason="set STARLING_LFM3_SNAPSHOT to the LFM3-1.7B-ONNX snapshot (c680759a)",
)
needs_cache = pytest.mark.skipif(
    not SNAPSHOT or not CACHE, reason="set STARLING_LFM3_SNAPSHOT and STARLING_LFM3_INTERVENTION_CACHE"
)


@needs_snapshot
def test_renderer_and_encoder_reproduce_the_lfm3_template_corpus():
    pytest.importorskip("tokenizers")
    from model_pipeline.interventions.extract import Example, lfm3_encoder, render_lfm3, tokenize

    encoder = lfm3_encoder(Path(SNAPSHOT))
    corpus = REPO / "ai-dotnet/tests/Starlings.Models.Lfm3.Tests/Data/template-reference.jsonl"
    cases = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
    for case in cases:
        assert encoder(case["text"]) == case["prompt_ids"], case["id"]
    single = [c for c in cases if not c["enable_thinking"] and [m["role"] for m in c["messages"]] == ["user"]]
    assert single
    for case in single:
        prefix, _ = render_lfm3(case["messages"][0]["content"], "")
        assert prefix == case["text"], case["id"]
    token = tokenize(encoder, Example("x", "What is a river?", "A river is flowing water."))
    assert token.response_tokens > 3


@needs_snapshot
def test_the_checked_in_concise_profile_compiles_into_the_fp32_graph(tmp_path):
    from model_pipeline.interventions import compile_intervention

    result = compile_intervention(
        base_dir=SNAPSHOT,
        graph=LFM3_FP32_GRAPH,
        profile=CONCISE / "profile.json",
        out_dir=tmp_path / "derived",
        link_external=False,
    )
    assert result.manifest.profile_id == "lfm3-1.7b-concise"
    assert result.manifest.base_graph.sha256 == LFM3_FP32_GRAPH_SHA256


def _feeds(session, ids):
    n = len(ids)
    feeds = {
        "input_ids": np.asarray([ids], dtype=np.int64),
        "attention_mask": np.ones((1, n), dtype=np.int64),
        "position_ids": np.arange(n, dtype=np.int64)[None, :],
        "logits_to_keep": np.asarray([1], dtype=np.int64),
    }
    for item in session.get_inputs():
        if item.name.startswith("past_conv."):
            feeds[item.name] = np.zeros((1, item.shape[1], item.shape[2]), dtype=np.float32)
        elif item.name.startswith("past_key_values."):
            feeds[item.name] = np.zeros((1, item.shape[1], 0, item.shape[3]), dtype=np.float32)
    return feeds


def _run(path: Path, names: list[str], ids: list[int]) -> dict[str, np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    try:
        return dict(zip(names, session.run(names, _feeds(session, ids)), strict=True))
    finally:
        del session


@needs_cache
def test_capture_graph_outputs_equal_the_base_graphs_internal_tensors():
    """Layers 8 and 23 through the capture graph vs the same tensors exposed by hand on the base graph, and logits."""
    from test_intervention_lfm3 import PROMPT_IDS

    from model_pipeline.interventions.external_data import link_external_data

    layers = [8, 23]
    entry = get_or_compile_capture(
        cache_root=CACHE,
        model="lfm3",
        base_dir=SNAPSHOT,
        graph=LFM3_FP32_GRAPH,
        layers=layers,
        external_sha256=LFM3_FP32_EXTERNAL_SHA256,
    )
    names = [LFM3_FP32_MAPPING.site(layer, "residual_post").tensor for layer in layers]
    assert [entry.output(layer) for layer in layers] == names
    captured = _run(entry.graph_path, ["logits", *names], PROMPT_IDS)

    model = onnx.load(str(Path(SNAPSHOT, LFM3_FP32_GRAPH)), load_external_data=False)
    for name in names:
        model.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
    probe = Path(CACHE, "capture-probe-" + os.urandom(4).hex())
    probe.mkdir()
    try:
        link_external_data(Path(SNAPSHOT, "onnx"), probe, sorted(p.split("/", 1)[1] for p in LFM3_FP32_EXTERNAL_SHA256))
        (probe / "model.onnx").write_bytes(model.SerializeToString())
        internal = _run(probe / "model.onnx", ["logits", *names], PROMPT_IDS)
    finally:
        for child in probe.iterdir():
            child.unlink()
        probe.rmdir()
    base = _run(Path(SNAPSHOT, LFM3_FP32_GRAPH), ["logits"], PROMPT_IDS)
    np.testing.assert_array_equal(captured["logits"], base["logits"])
    for name in names:
        assert captured[name].shape == (1, len(PROMPT_IDS), 2048)
        np.testing.assert_array_equal(captured[name], internal[name])


@needs_cache
@pytest.mark.skipif(
    os.environ.get("STARLING_LFM3_EXTRACT_REPRODUCE") != "1", reason="set STARLING_LFM3_EXTRACT_REPRODUCE=1"
)
def test_concise_extraction_reproduces_the_checked_in_profile(tmp_path):
    pytest.importorskip("tokenizers")
    from model_pipeline.interventions.extract import REPORT_JSON, load_spec, run_extraction

    spec = load_spec(CONCISE / "spec.yaml")
    _, profile = run_extraction(
        spec,
        snapshot=Path(SNAPSHOT),
        cache_root=Path(CACHE),
        out_dir=tmp_path,
        external_sha256=LFM3_FP32_EXTERNAL_SHA256,
        log=lambda _: None,
    )
    # ONNX Runtime's CPU reductions may round differently with another thread count, so compare numerically.
    from model_pipeline.interventions import load_profile

    committed_profile = load_profile(CONCISE / "profile.json")
    assert [s.layer for s in profile.profile.sites] == [s.layer for s in committed_profile.profile.sites]
    for fresh_site, site in zip(profile.profile.sites, committed_profile.profile.sites, strict=True):
        q_fresh, q = profile.load_directions(fresh_site), committed_profile.load_directions(site)
        assert float(np.min(np.sum(q_fresh * q, axis=1))) > 0.99999
    fresh = json.loads((tmp_path / REPORT_JSON).read_text())
    committed = json.loads((CONCISE / REPORT_JSON).read_text())
    assert fresh["selected_layers"] == committed["selected_layers"]
    for a, b in zip(fresh["layers"], committed["layers"], strict=True):
        assert a["heldout_auc"] == pytest.approx(b["heldout_auc"], abs=1e-3)
        assert a["heldout_cohens_d"] == pytest.approx(b["heldout_cohens_d"], rel=1e-3)
