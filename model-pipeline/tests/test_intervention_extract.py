"""Profile extraction (model_pipeline.interventions.extract): spec validation, rendering, the analysis, and a planted
direction recovered end to end through a synthetic capture graph. No real model is needed."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from test_intervention_capture import HIDDEN, _weights, build_graph, mapping_for, write_bundle  # noqa: E402

from model_pipeline.cli import main as cli_main  # noqa: E402
from model_pipeline.interventions import compile_intervention, load_profile  # noqa: E402
from model_pipeline.interventions.capture import get_or_compile_capture  # noqa: E402
from model_pipeline.interventions.contract import max_orthonormality_error  # noqa: E402
from model_pipeline.interventions.extract import (  # noqa: E402
    REPORT_JSON,
    REPORT_MD,
    Example,
    ExtractionError,
    ExtractionSpecError,
    analyze,
    auc,
    cohens_d,
    collect,
    directions,
    extract_from_activations,
    fold_indices,
    load_pairs,
    load_spec,
    pool,
    render_lfm3,
    tokenize,
)
from model_pipeline.interventions.validate import run_outputs  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONCISE = REPO / "model-pipeline/interventions/profiles/lfm3-1.7b/concise"


def _unit(v):
    return v / np.linalg.norm(v)


# --------------------------------------------------------------------------------------------------------------------
# Spec


def _write_spec(directory: Path, pairs: int = 16, **overrides) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "prompts").mkdir(exist_ok=True)
    for name, style in (("pos", "short"), ("neg", "long long long")):
        lines = [
            json.dumps({"id": f"p{i}", "prompt": f"Question {i}?", "response": f"{style} {i}"}) for i in range(pairs)
        ]
        (directory / "prompts" / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    spec = {
        "schema_version": 1,
        "profile_id": "test-profile",
        "model": "mlp",
        "base_model_sha256": "a" * 64,
        "graph": "onnx/model.onnx",
        "chat_template": "lfm3",
        "operator": "directional_projection",
        "target": "residual_post",
        "prompt_sets": {
            "positive": {"label": "short", "path": "prompts/pos.jsonl"},
            "negative": {"label": "long", "path": "prompts/neg.jsonl"},
        },
        "pooling": "response_mean",
        "candidate_layers": [0, 1, 2],
        "rank": 1,
        "direction_method": "mean_difference_pca",
        "scale": 1,
        "evaluation": {"folds": 4},
        "selection": {"max_layers": 1, "min_auc": 0.9},
        "seed": 7,
    }
    for key, value in overrides.items():
        if value is None:
            spec.pop(key)
        else:
            spec[key] = value
    path = directory / "spec.yaml"
    path.write_text(json.dumps(spec), encoding="utf-8")  # JSON is YAML
    return path


def test_valid_spec_loads(tmp_path):
    spec = load_spec(_write_spec(tmp_path))
    assert spec.profile_id == "test-profile"
    assert spec.positive.path == tmp_path / "prompts" / "pos.jsonl"
    assert spec.candidate_layers == (0, 1, 2)
    positive, negative = load_pairs(spec)
    assert len(positive) == len(negative) == 16


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"extra": 1}, "Additional properties"),
        ({"seed": None}, "'seed' is a required property"),
        ({"base_model_sha256": "A" * 64}, "base_model_sha256"),
        ({"rank": 0}, "rank"),
        ({"pooling": "max"}, "pooling"),
        ({"schema_version": 2}, "schema_version"),
        ({"candidate_layers": [1, 1]}, "candidate_layers"),
        ({"selection": {"max_layers": 1, "min_auc": 0.4}}, "min_auc"),
        ({"selection": {"max_layers": 4, "min_auc": 0.9}}, "max_layers exceeds"),
        ({"evaluation": {"folds": 1}}, "folds"),
        ({"operator": "steer"}, "operator"),
        ({"profile_id": "Bad Id"}, "profile_id"),
        ({"graph": "../model.onnx"}, "graph"),
        (
            {
                "prompt_sets": {
                    "positive": {"label": "a", "path": "prompts/pos.jsonl"},
                    "negative": {"label": "a", "path": "prompts/neg.jsonl"},
                }
            },
            "same label",
        ),
        (
            {
                "prompt_sets": {
                    "positive": {"label": "a", "path": "prompts/pos.jsonl"},
                    "negative": {"label": "b", "path": "../neg.jsonl"},
                }
            },
            "relative path",
        ),
    ],
)
def test_invalid_specs_are_rejected(tmp_path, overrides, match):
    with pytest.raises(ExtractionSpecError, match=match):
        load_spec(_write_spec(tmp_path, **overrides))


def test_prompt_sets_must_pair_up(tmp_path):
    spec = load_spec(_write_spec(tmp_path))
    neg = tmp_path / "prompts" / "neg.jsonl"
    lines = neg.read_text().splitlines()

    neg.write_text("\n".join(lines[1:] + lines[:1]) + "\n")
    with pytest.raises(ExtractionSpecError, match="same ids"):
        load_pairs(spec)
    neg.write_text("\n".join([lines[0].replace("Question 0", "Other")] + lines[1:]) + "\n")
    with pytest.raises(ExtractionSpecError, match="prompts differ"):
        load_pairs(spec)
    neg.write_text("\n".join([json.dumps({"id": "p0", "prompt": "Question 0?", "response": "short 0"})] + lines[1:]))
    with pytest.raises(ExtractionSpecError, match="identical"):
        load_pairs(spec)
    neg.write_text("\n".join([json.dumps({"id": "p0", "prompt": "Question 0?", "response": ""})] + lines[1:]))
    with pytest.raises(ExtractionSpecError, match="non-empty"):
        load_pairs(spec)
    neg.write_text("\n".join([json.dumps({"id": "p0", "prompt": "Q", "response": "x", "x": 1})] + lines[1:]))
    with pytest.raises(ExtractionSpecError, match="exactly the keys"):
        load_pairs(spec)
    (tmp_path / "prompts" / "neg.jsonl").unlink()
    with pytest.raises(ExtractionSpecError, match="does not exist"):
        load_pairs(spec)


def test_too_few_pairs_for_the_folds(tmp_path):
    spec = load_spec(_write_spec(tmp_path, pairs=7))
    with pytest.raises(ExtractionSpecError, match="too few"):
        load_pairs(spec)


def test_the_checked_in_concise_spec_and_prompt_sets_are_valid():
    spec = load_spec(CONCISE / "spec.yaml")
    assert spec.profile_id == "lfm3-1.7b-concise"
    assert spec.pooling == "response_mean"
    positive, negative = load_pairs(spec)
    assert len(positive) >= 64
    # The concise answer is shorter than the verbose one for every prompt.
    assert all(len(p.response) < len(n.response) for p, n in zip(positive, negative, strict=True))
    from model_pipeline.interventions.injection_sites import LFM3_FP32_GRAPH_SHA256, LFM3_FP32_MAPPING

    assert spec.base_model_sha256 == LFM3_FP32_GRAPH_SHA256
    for layer in spec.candidate_layers:
        LFM3_FP32_MAPPING.site(layer, spec.target)


def test_the_checked_in_concise_profile_matches_its_spec_and_report():
    profile = load_profile(CONCISE / "profile.json")
    spec = load_spec(CONCISE / "spec.yaml")
    report = json.loads((CONCISE / REPORT_JSON).read_text(encoding="utf-8"))
    assert profile.profile.id == spec.profile_id
    assert profile.profile.base_model_sha256 == spec.base_model_sha256
    assert report["spec"]["sha256"] == spec.sha256
    assert report["profile"]["sha256"] == profile.sha256
    assert [s.layer for s in profile.profile.sites] == report["selected_layers"]
    # The markdown report is generated from the JSON one; no hand edits.
    from model_pipeline.interventions.extract import render_report_markdown

    assert render_report_markdown(report) == (CONCISE / REPORT_MD).read_text(encoding="utf-8")
    from model_pipeline.interventions.contract import sha256_file

    for role, s in (("positive", spec.positive), ("negative", spec.negative)):
        assert report["prompt_sets"][role]["sha256"] == sha256_file(s.path)
    for site in profile.profile.sites:
        q = profile.load_directions(site)
        assert site.rank == spec.rank and site.hidden_size == 2048
        assert max_orthonormality_error(q) < 1e-5
        assert list(site.scales) == [spec.scale] * spec.rank


# --------------------------------------------------------------------------------------------------------------------
# Rendering, tokenization, pooling


def _char_encoder(text: str) -> list[int]:
    """A stand-in tokenizer: special tokens are single ids, every other character is its code point + 1000."""
    specials = ["<|startoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
    ids, i = [], 0
    while i < len(text):
        for k, s in enumerate(specials):
            if text.startswith(s, i):
                ids.append(k)
                i += len(s)
                break
        else:
            ids.append(1000 + ord(text[i]))
            i += 1
    return ids


def test_render_matches_the_lfm3_template_reference():
    prefix, full = render_lfm3("What is 2 + 2? Answer briefly.", "4.")
    reference = (REPO / "ai-dotnet/tests/Starlings.Models.Lfm3.Tests/Data/template-reference.jsonl").read_text()
    arithmetic = next(json.loads(line) for line in reference.splitlines() if '"id": "arithmetic"' in line)
    assert prefix == arithmetic["text"]
    assert full == prefix + "4.<|im_end|>"


def test_tokenize_locates_the_response_tokens():
    token = tokenize(_char_encoder, Example("a", "Hi", "Yes."))
    assert [chr(i - 1000) for i in token.ids[token.response_start : token.response_end]] == list("Yes.")
    assert token.ids[token.response_end] == 2  # <|im_end|>
    assert token.response_tokens == 4
    with pytest.raises(ExtractionSpecError):
        tokenize(_char_encoder, Example("a", "Hi", "Yes."), template="other")


def test_pool_response_mean_and_last_token():
    token = tokenize(_char_encoder, Example("a", "Hi", "abc"))
    acts = np.arange(len(token.ids) * 2, dtype=np.float32).reshape(-1, 2)
    s, e = token.response_start, token.response_end
    np.testing.assert_array_equal(pool(acts, token, "response_mean"), acts[s:e].mean(axis=0))
    np.testing.assert_array_equal(pool(acts, token, "last_token"), acts[e - 1])
    with pytest.raises(ValueError):
        pool(acts[:-1], token, "response_mean")


# --------------------------------------------------------------------------------------------------------------------
# Statistics and directions


def test_auc_and_cohens_d():
    assert auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert auc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0
    assert auc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5
    assert auc(np.array([2.0, 0.0]), np.array([1.0, 1.0])) == 0.5
    assert cohens_d(np.array([1.0, 3.0]), np.array([0.0, 2.0])) == pytest.approx(1 / np.sqrt(2))


def test_folds_partition_the_pairs_deterministically():
    folds = fold_indices(10, 3, seed=5)
    assert sorted(np.concatenate(folds).tolist()) == list(range(10))
    assert [f.tolist() for f in folds] == [f.tolist() for f in fold_indices(10, 3, seed=5)]
    assert [f.tolist() for f in folds] != [f.tolist() for f in fold_indices(10, 3, seed=6)]


def _planted(pairs=64, hidden=32, layers=4, planted_layer=2, strength=8.0, seed=0, extra=None):
    """Positive = negative + strength * v at ``planted_layer`` (+ noise); other layers differ by noise only."""
    rng = np.random.default_rng(seed)
    v = _unit(rng.standard_normal(hidden))
    negative = rng.standard_normal((pairs, layers, hidden)) * 2.0 + rng.standard_normal(hidden)
    positive = negative + 0.5 * rng.standard_normal((pairs, layers, hidden))
    positive[:, planted_layer] += strength * v
    if extra is not None:
        positive[:, planted_layer] += rng.standard_normal((pairs, 1)) * 2.0 * extra[None, :]
    return positive, negative, v


def test_a_planted_direction_is_recovered_and_its_layer_selected():
    positive, negative, v = _planted()
    analysis = analyze(positive, negative, [10, 11, 12, 13], rank=1, folds=4, seed=1, max_layers=1, min_auc=0.9)
    assert analysis.selected == [12]
    q = analysis.directions[12]
    assert q.shape == (1, 32)
    assert float(q[0] @ v) > 0.99  # sign towards positive
    by_layer = {m.layer: m for m in analysis.layers}
    assert by_layer[12].heldout_auc > 0.99
    # Noise-only layers: held-out separability is near chance even though in-sample looks better.
    for layer in (10, 11, 13):
        assert by_layer[layer].heldout_auc < 0.8
        assert by_layer[layer].insample_auc > by_layer[layer].heldout_auc


def test_rank_two_spans_the_mean_difference_and_the_main_paired_variation():
    rng = np.random.default_rng(3)
    w = _unit(rng.standard_normal(32))
    positive, negative, v = _planted(extra=w)
    w_perp = _unit(w - (w @ v) * v)
    q = directions(positive[:, 2], negative[:, 2], 2)
    assert max_orthonormality_error(q.astype(np.float32)) < 1e-6
    assert float(q[0] @ v) > 0.98
    projection = q.T @ q
    assert np.linalg.norm(projection @ w_perp) > 0.97  # the second direction picks up the planted variation


def test_no_layer_above_the_threshold_writes_no_profile(tmp_path):
    spec = load_spec(_write_spec(tmp_path, pairs=64))
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((64, 3, 8))
    with pytest.raises(ExtractionError, match="no candidate layer"):
        extract_from_activations(spec, noise + 0.01 * rng.standard_normal((64, 3, 8)), noise,
                                 response_tokens={"short": [1], "long": [2]}, out_dir=tmp_path / "out",
                                 environment={})  # fmt: skip
    assert not (tmp_path / "out" / "profile.json").exists()
    assert "no profile was written" in (tmp_path / "out" / REPORT_MD).read_text()


# --------------------------------------------------------------------------------------------------------------------
# End to end through a synthetic capture graph


def test_planted_direction_through_the_capture_graph_compiles_into_a_working_profile(tmp_path):
    """Inputs carry a planted direction v in the response positions of positive examples; the capture graph's residual
    stream carries it to every layer. The extraction recovers v, writes a profile and report, and the profile compiles
    and removes v's component at strength 1."""
    # Small feed-forward contributions, so the residual stream carries the planted input direction nearly unchanged.
    weights = {k: (v * 0.02 if k.startswith("w2_") else v) for k, v in _weights().items()}
    base_dir = tmp_path / "bundle"
    graph_path, sha = write_bundle(base_dir, build_graph(weights))
    mapping = mapping_for(sha)
    spec_path = _write_spec(tmp_path / "spec", pairs=32, base_model_sha256=sha, selection={"max_layers": 2,
                            "min_auc": 0.9})  # fmt: skip
    spec = load_spec(spec_path)
    positive, negative = load_pairs(spec)
    entry = get_or_compile_capture(cache_root=tmp_path / "cache", model="mlp", base_dir=base_dir,
                                   graph="onnx/model.onnx", layers=spec.candidate_layers, mapping=mapping)  # fmt: skip

    rng = np.random.default_rng(11)
    v = _unit(rng.standard_normal(HIDDEN)).astype(np.float32)
    # The two response texts differ, so their token embeddings differ too; keep that lexical difference small.
    embedding = rng.standard_normal((2000, HIDDEN)).astype(np.float32) * 0.05
    import onnxruntime

    session = onnxruntime.InferenceSession(str(entry.graph_path), providers=["CPUExecutionProvider"])
    names = [entry.output(layer) for layer in spec.candidate_layers]

    def runner_for(planted: bool):
        def run(ids):
            x = embedding[np.asarray(ids)][None]
            if planted:
                x = x + 1.5 * v
            out = session.run(names, {"x": x.astype(np.float32)})
            return {layer: o[0] for layer, o in zip(spec.candidate_layers, out, strict=True)}

        return run

    pos_tokens = [tokenize(_char_encoder, e) for e in positive]
    neg_tokens = [tokenize(_char_encoder, e) for e in negative]
    pos = collect(runner_for(True), pos_tokens, spec.candidate_layers, spec.pooling)
    neg = collect(runner_for(False), neg_tokens, spec.candidate_layers, spec.pooling)
    out = tmp_path / "out"
    analysis, profile = extract_from_activations(
        spec, pos, neg, response_tokens={"short": [1], "long": [1]}, out_dir=out, environment={"Runtime": "test"}
    )
    assert len(analysis.selected) == 2
    for layer in analysis.selected:
        assert float(analysis.directions[layer][0] @ v) > 0.99
    assert profile.profile.base_model_sha256 == sha
    report = json.loads((out / REPORT_JSON).read_text())
    assert report["selected_layers"] == analysis.selected
    assert report["profile"]["sha256"] == profile.sha256
    assert "Per-layer separability" in (out / REPORT_MD).read_text()

    # Deterministic: the same activations give the same bytes.
    again = tmp_path / "again"
    extract_from_activations(spec, pos, neg, response_tokens={"short": [1], "long": [1]}, out_dir=again,
                             environment={"Runtime": "test"})  # fmt: skip
    for name in ("profile.json", REPORT_JSON, *(s.directions_file for s in profile.profile.sites)):
        assert (out / name).read_bytes() == (again / name).read_bytes()

    # The written profile is a valid compiler input; at strength 1 the site output has no component along Q.
    derived = compile_intervention(base_dir=base_dir, graph="onnx/model.onnx", profile=out / "profile.json",
                                   out_dir=tmp_path / "derived", mapping=mapping)  # fmt: skip
    layer = analysis.selected[0]
    site = next(s for s in profile.profile.sites if s.layer == layer)
    q = profile.load_directions(site)
    model = onnx.load(str(derived.graph_path), load_external_data=False)
    model.graph.output.append(
        onnx.helper.make_empty_tensor_value_info(f"intervention/layer{layer}/residual_post/output")
    )
    probe = tmp_path / "probe"
    probe.mkdir()
    shutil.copy(base_dir / "onnx" / "model.onnx_data", probe / "model.onnx_data")
    (probe / "model.onnx").write_bytes(model.SerializeToString())
    x = (embedding[:5][None] + 1.5 * v).astype(np.float32)
    outputs = run_outputs(probe / "model.onnx", {"x": x, "intervention_strength": np.asarray([1.0], np.float32)})
    intervened = outputs[f"intervention/layer{layer}/residual_post/output"]
    assert float(np.max(np.abs(intervened @ q.T))) < 1e-4


def test_cli_extract_requires_the_snapshot_and_cache(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("STARLING_LFM3_SNAPSHOT", raising=False)
    monkeypatch.delenv("STARLING_LFM3_INTERVENTION_CACHE", raising=False)
    assert cli_main(["interventions", "extract", str(_write_spec(tmp_path))]) == 2
    assert "--snapshot and --cache are required" in capsys.readouterr().out
