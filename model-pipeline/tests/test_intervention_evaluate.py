"""The profile evaluation harness without a model: spec, prompt sets, scorers, KL, recommendation, the whole run
with fake sessions, and the checked-in "concise" evaluation report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("jsonschema")

from model_pipeline.interventions import evaluate as ev  # noqa: E402
from model_pipeline.interventions.extract import Tokenized, load_prompt_set  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONCISE = REPO / "model-pipeline/interventions/profiles/lfm3-1.7b/concise"
SPEC = CONCISE / "evaluation/spec.yaml"
EVIDENCE = REPO / "docs/evidence/lfm3-concise-evaluation-2026-10-31"


# --------------------------------------------------------------------------------------------------------------------
# Spec and prompt sets


def test_the_checked_in_spec_parses():
    spec = ev.load_spec(SPEC)
    assert spec.profile == (CONCISE / "profile.json").absolute()
    assert 0.0 in spec.strengths and list(spec.strengths) == sorted(spec.strengths)
    assert any(s < 0 for s in spec.strengths) and any(s > 0 for s in spec.strengths)
    assert 0.0 in spec.control_strengths and set(spec.control_strengths) <= set(spec.strengths)
    assert spec.control_scaling == "matched_rms"
    assert (spec.positive_label, spec.negative_label) == ("concise", "verbose")


def _spec_text() -> str:
    return SPEC.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("strengths: [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 2]", "strengths: [-1, 1]", "strengths"),
        ("  scaling: matched_rms\n", "", "scaling"),
        ("profile: ../profile.json", "profile: /abs/profile.json", "profile"),
        ("  effect: shorter_responses", "  effect: longer_responses", "effect"),
        ("schema_version: 1", "schema_version: 1\nextra: 1", "extra"),
    ],
)
def test_invalid_specs_are_rejected(tmp_path, old, new, message):
    text = _spec_text()
    assert old in text
    path = tmp_path / "spec.yaml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(ev.EvaluationSpecError, match=message):
        ev.load_spec(path)


def test_the_target_prompts_are_held_out_from_the_extraction_sets():
    spec = ev.load_spec(SPEC)
    target = ev.load_target(spec.target_prompts)
    train = load_prompt_set(CONCISE / "prompts/concise.jsonl")
    assert len(target) == 16
    assert not {t.id for t in target} & {e.id for e in train}
    normalized = {e.prompt.lower().rstrip("?") for e in train}
    assert not {t.prompt.lower().rstrip("?") for t in target} & normalized
    neutral = ev.load_neutral(spec.neutral)
    assert not {n.prompt for n in neutral} & ({t.prompt for t in target} | {e.prompt for e in train})


def test_every_reference_answer_passes_the_keyword_and_repetition_checks():
    # The quality check must accept a good answer of either style, or it would measure style instead of quality.
    for t in ev.load_target(ev.load_spec(SPEC).target_prompts):
        for answer in (t.reference_positive, t.reference_negative):
            assert ev.keyword_hits(t.keywords, answer), t.id
            assert not ev.is_degenerate(answer), t.id


def test_every_capability_item_accepts_its_ideal_answer():
    ideal = {
        "exact": lambda c: str(c["value"]),
        "last_integer": lambda c: f"So the answer is {c['value']}.",
        "uppercase_contains": lambda c: c["value"] + ".",
        "bullets": lambda c: "\n".join(f"- item {i}" for i in range(c["count"])),
        "one_word": lambda c: c["value"].capitalize() + ".",
        "json_object": lambda c: "```json\n" + json.dumps(c["value"]) + "\n```",
        "starts_with": lambda c: c["value"] + " roll onto the beach.",
        "max_words": lambda c: "Clear bright blue.",
    }
    items = ev.load_capability(ev.load_spec(SPEC).capability)
    assert len(items) == 24 and {i.kind for i in items} == {"arithmetic", "instruction", "code"}
    for item in items:
        assert ev.score_capability(item.check, ideal[item.check["type"]](item.check)), item.id


def test_prompt_set_errors(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"id": "a", "prompt": "p", "response": "r"}\n{"id": "a", "prompt": "q", "response": "r"}\n')
    with pytest.raises(ev.EvaluationSpecError, match="duplicate"):
        ev.load_neutral(path)
    path.write_text('{"id": "a", "kind": "k", "prompt": "p", "check": {"type": "regex"}}\n')
    with pytest.raises(ev.EvaluationSpecError, match="check.type"):
        ev.load_capability(path)
    path.write_text(
        '{"id": "a", "prompt": "p", "keywords": [], "reference_positive": "x", "reference_negative": "y"}\n'
    )
    with pytest.raises(ev.EvaluationSpecError, match="keywords"):
        ev.load_target(path)


# --------------------------------------------------------------------------------------------------------------------
# Scorers


@pytest.mark.parametrize(
    ("check", "text", "expected"),
    [
        ({"type": "exact", "value": "8"}, " 8\n", True),
        ({"type": "exact", "value": "8"}, "`8`", True),
        ({"type": "exact", "value": "8"}, "A spider has 8 legs.", False),
        ({"type": "exact", "value": "yes"}, "Yes.", True),
        ({"type": "exact", "value": "[1, 2, 3]"}, "```\n[1, 2, 3]\n```", True),
        ({"type": "last_integer", "value": 633}, "347 + 286 = 633", True),
        ({"type": "last_integer", "value": 633}, "The sum is 633. Check: 633 - 286 = 347", False),
        ({"type": "last_integer", "value": 1200}, "That makes 1,200 in total.", True),
        ({"type": "uppercase_contains", "value": "THE TRAIN"}, "THE TRAIN LEAVES", True),
        ({"type": "uppercase_contains", "value": "THE TRAIN"}, "THE TRAIN leaves", False),
        ({"type": "bullets", "count": 3}, "- a\n- b\n- c\n", True),
        ({"type": "bullets", "count": 3}, "Here:\n- a\n- b\n- c", False),
        ({"type": "bullets", "count": 3}, "- a\n- b", False),
        ({"type": "one_word", "value": "yellow"}, "Yellow", True),
        ({"type": "one_word", "value": "yellow"}, "It is yellow.", False),
        ({"type": "json_object", "value": {"a": 1}}, '{"a": 1}', True),
        ({"type": "json_object", "value": {"a": 1}}, '{"a": "1"}', False),
        ({"type": "json_object", "value": {"a": 1}}, 'Sure: {"a": 1}', False),
        ({"type": "starts_with", "value": "Waves"}, "Waves crash.", True),
        ({"type": "starts_with", "value": "Waves"}, "The waves crash.", False),
        ({"type": "max_words", "value": 5}, "Bright, clear and blue.", True),
        ({"type": "max_words", "value": 5}, "It is a bright clear blue colour.", False),
    ],
)
def test_score_capability(check, text, expected):
    assert ev.score_capability(check, text) is expected


def test_keywords_and_degeneracy():
    assert ev.keyword_hits([["starch"], ["water", "dry"]], "Starch pushes WATER out")
    assert not ev.keyword_hits([["starch"], ["water", "dry"]], "starch only")
    loop = "the cat sat on the mat and " * 3
    assert ev.is_degenerate(loop)
    assert not ev.is_degenerate("the cat sat on the mat and the dog sat on the rug and then they slept")


def test_kl_per_position():
    rng = np.random.default_rng(0)
    p = rng.standard_normal((5, 50)).astype(np.float32)
    q = rng.standard_normal((5, 50)).astype(np.float32)
    assert np.allclose(ev.kl_per_position(p, p), 0, atol=1e-12)
    kl = ev.kl_per_position(p, q)
    pp = np.exp(p - p.max(1, keepdims=True))
    pp /= pp.sum(1, keepdims=True)
    qq = np.exp(q - q.max(1, keepdims=True))
    qq /= qq.sum(1, keepdims=True)
    assert np.allclose(kl, (pp * np.log(pp / qq)).sum(1), rtol=1e-5)
    assert (kl > 0).all()
    # Shift invariance of logits.
    assert np.allclose(ev.kl_per_position(p + 3.0, q - 1.0), kl, rtol=1e-5)


def test_compare_ids():
    assert ev.compare_ids([1, 2, 3, 4], [1, 2, 3], 3) == {"compared": 3, "equal": True, "first_difference": None}
    assert ev.compare_ids([1, 2, 3], [1, 9, 3], 3)["first_difference"] == 1
    assert ev.compare_ids([1, 2], [1, 2, 3], 3) == {"compared": 2, "equal": False, "first_difference": 2}


# --------------------------------------------------------------------------------------------------------------------
# Recommendation


def _entry(s, *, reduction, finished=10, quality=0.9, correct=20, kl=0.1):
    return {
        "strength": s,
        "target": {"length_reduction": reduction, "finished": finished, "quality_pass_rate": quality},
        "capability": {"correct": correct},
        "drift": {"mean_kl": kl},
    }


RULE = ev.Recommendation(0.25, 0.125, 2, 0.5, 0.15)


def test_recommend_applies_every_rule():
    profile = {
        -3.0: _entry(-3.0, reduction=0.7, kl=0.9),  # drift
        -2.0: _entry(-2.0, reduction=0.6),
        -1.0: _entry(-1.0, reduction=0.4, quality=0.75),  # quality drop 0.15
        -0.5: _entry(-0.5, reduction=0.1),  # too little
        0.0: _entry(0.0, reduction=0.0),
        1.0: _entry(1.0, reduction=0.3, correct=17),  # capability
    }
    control = {
        -3.0: _entry(-3.0, reduction=0.1),
        -2.0: _entry(-2.0, reduction=0.1),
        -1.0: _entry(-1.0, reduction=0.0),
        1.0: _entry(1.0, reduction=0.0),
    }
    rec = ev.recommend(RULE, profile, control)
    by = {p["strength"]: p for p in rec["per_strength"]}
    assert not by[-3.0]["checks"]["drift"]
    assert by[-2.0]["recommended"]
    assert not by[-1.0]["checks"]["quality"]
    assert not by[-0.5]["checks"]["length_reduction"] and not by[-0.5]["checks"]["specificity"]
    assert by[-0.5]["specificity"] is None
    assert not by[1.0]["checks"]["capability"]
    assert rec["recommended_strengths"] == [-2.0]
    assert rec["recommended_range"] == [-2.0, -2.0]


def test_recommend_needs_specificity_and_a_contiguous_range():
    profile = {s: _entry(s, reduction=0.5) for s in (-3.0, -2.0, -1.0, 0.0)}
    profile[0.0] = _entry(0.0, reduction=0.0)
    # The control shortens as much at -2: not specific there.
    control = {-3.0: _entry(-3.0, reduction=0.0), -2.0: _entry(-2.0, reduction=0.45), -1.0: _entry(-1.0, reduction=0.0)}
    rec = ev.recommend(RULE, profile, control)
    assert rec["recommended_strengths"] == [-3.0, -1.0]
    assert rec["recommended_range"] is None
    assert ev.recommend(RULE, profile, {})["recommended_strengths"] == []


# --------------------------------------------------------------------------------------------------------------------
# The whole run with fake sessions

HIDDEN = 8
VOCAB = 40
END = 39


class FakeModel:
    """Answers get shorter as the strength goes negative, but only when the direction is the profile's."""

    def __init__(self, effective: bool) -> None:
        self.effective = effective
        self.calls = 0

    def generate(self, prompt_ids, strength, max_new_tokens):
        self.calls += 1
        base = 30 + len(prompt_ids) % 5
        n = max(2, int(round(base * (1 + 0.3 * strength)))) if self.effective else base
        out = [(i * 7 + len(prompt_ids)) % 30 + 1 for i in range(min(n, max_new_tokens))]
        return out, n <= max_new_tokens

    def logits(self, ids, strength):
        rng = np.random.default_rng(len(ids))
        base = rng.standard_normal((len(ids), VOCAB)).astype(np.float32)
        return base + (0.2 if self.effective else 0.02) * strength * np.linspace(-1, 1, VOCAB, dtype=np.float32)


def fake_capture(ids):
    # Activation along axis 0 grows with the response length, so shorter answers project differently.
    n = len(ids)
    acts = np.zeros((n, HIDDEN), dtype=np.float32)
    acts[:, 0] = np.linspace(0, 1, n) * (60.0 / max(n, 1))
    acts[:, 1] = np.cos(np.arange(n))
    return {2: acts, 4: acts * 2.0}


def _encoder_with_end(text):
    # Special tokens become single ids; other characters map to small ids.
    parts = text.split("<|im_end|>")
    ids = []
    for i, part in enumerate(parts):
        ids += [ord(c) % 30 + 1 for c in part.replace("<|startoftext|>", "\x01").replace("<|im_start|>", "\x02")]
        if i < len(parts) - 1:
            ids.append(END)
    return ids


def run_fake(tmp_path, spec=None):
    spec = spec or ev.load_spec(SPEC)
    prepared = ev.prepare(spec, _encoder_with_end)
    directions = {2: np.eye(HIDDEN, dtype=np.float32)[:1], 4: np.eye(HIDDEN, dtype=np.float32)[:1]}
    control_dirs = {2: np.eye(HIDDEN, dtype=np.float32)[1:2], 4: np.eye(HIDDEN, dtype=np.float32)[1:2]}
    seen = {}

    def prepare_control(scales):
        seen["scales"] = scales

    report = ev.evaluate_with(
        spec,
        prepared,
        open_model=lambda name: FakeModel(effective=name == ev.ARM_PROFILE),
        open_capture=lambda: fake_capture,
        directions=directions,
        scales={2: np.ones(1), 4: np.ones(1)},
        control_directions=control_dirs,
        prepare_control=prepare_control,
        decode=lambda ids: " ".join(f"w{i}" for i in ids),
        open_cpu_model=lambda: FakeModel(effective=True),
        out_dir=tmp_path,
        profile_sha256="0" * 64,
        profile_id="fake",
        environment={"Runtime": "fake"},
        log=lambda _: None,
    )
    return report, seen


def test_the_whole_run_with_fake_sessions(tmp_path):
    report, seen = run_fake(tmp_path)
    spec = ev.load_spec(SPEC)
    profile = {e["strength"]: e for e in report["arms"]["profile"]}
    control = {e["strength"]: e for e in report["arms"]["control"]}
    assert sorted(profile) == list(spec.strengths) and sorted(control) == list(spec.control_strengths)
    assert profile[0.0]["drift"]["mean_kl"] == 0 and profile[-2.0]["drift"]["mean_kl"] > 0
    assert profile[-2.0]["target"]["length_reduction"] > 0.5 and control[-2.0]["target"]["length_reduction"] == 0
    assert profile[1.0]["target"]["length_reduction"] < 0
    assert profile[-2.0]["target"]["shorter_than_strength_0"] == 16
    # Control scales: the profile scale times the RMS ratio (profile direction over control direction).
    assert set(seen["scales"]) == {2, 4} and all(
        np.isfinite(v).all() and (v > 0).all() for v in seen["scales"].values()
    )
    sites = report["internal"]["control"]["sites"]
    assert sites[0]["control_scales"][0] == pytest.approx(
        sites[0]["rms_projection_profile"][0] / sites[0]["rms_projection_control"][0], rel=1e-5
    )
    # Internal: references sit at 0 (verbose) and 1 (concise) by construction.
    assert report["internal"]["reference_position"] == {"concise": pytest.approx(1.0), "verbose": pytest.approx(0.0)}
    assert report["cross_check"]["cases"] and all(c["equal"] for c in report["cross_check"]["cases"])
    assert [p["strength"] for p in report["recommendation"]["per_strength"]] == [s for s in spec.strengths if s != 0]
    # Outputs: the markdown is the JSON rendered, and generations hold every answer.
    data = json.loads((tmp_path / ev.REPORT_JSON).read_text(encoding="utf-8"))
    assert (tmp_path / ev.REPORT_MD).read_text(encoding="utf-8") == ev.render_report_markdown(data)
    rows = [json.loads(line) for line in (tmp_path / ev.GENERATIONS).read_text(encoding="utf-8").splitlines()]
    configs = len(spec.strengths) + len(spec.control_strengths)
    assert len(rows) == configs * (16 + 24)
    assert all("ids" not in r for r in rows)


def test_the_run_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    run_fake(a)
    run_fake(b)
    for name in (ev.REPORT_JSON, ev.REPORT_MD, ev.GENERATIONS):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_generated_tokens_and_projection():
    t = ev.generated_tokens([5, 6, 7], [8, 9])
    assert t == Tokenized((5, 6, 7, 8, 9, ev.LFM3_IM_END_ID), 3, 5)
    acts = {2: np.arange(12, dtype=np.float32).reshape(6, 2)}
    out = ev.projections(lambda ids: acts, [t], {2: np.asarray([[1.0, 0.0]], dtype=np.float32)}, "response_mean")
    assert out.shape == (1, 1) and out[0, 0] == pytest.approx((6 + 8) / 2)
    rms = ev.token_rms(lambda ids: acts, [t], {2: np.asarray([[1.0, 0.0]], dtype=np.float32)})
    assert rms[2][0] == pytest.approx(np.sqrt((36 + 64) / 2))


def test_the_control_directions_are_the_written_profiles_directions(tmp_path):
    from model_pipeline.interventions.contract import load_profile

    profile = load_profile(CONCISE / "profile.json")
    dirs = ev.control_directions(profile, 7)
    written = ev.write_control_profile(tmp_path, profile, 7, {s.layer: [2.5] for s in profile.profile.sites})
    for site in written.profile.sites:
        assert np.array_equal(written.load_directions(site), dirs[site.layer])
        assert list(site.scales) == [2.5]
    assert [s.layer for s in written.profile.sites] == [s.layer for s in profile.profile.sites]


def test_only_supported_profiles_are_evaluated():
    import dataclasses

    from model_pipeline.interventions.contract import load_profile

    profile = load_profile(CONCISE / "profile.json").profile
    ev.check_supported_profile(profile)
    with pytest.raises(ev.EvaluationSpecError, match="operator"):
        ev.check_supported_profile(dataclasses.replace(profile, operator="additive"))
    first = profile.sites[0]
    with pytest.raises(ev.EvaluationSpecError, match="one site per layer"):
        ev.check_supported_profile(dataclasses.replace(profile, sites=(first, first)))
    other = dataclasses.replace(profile.sites[1], target="mlp_out")
    with pytest.raises(ev.EvaluationSpecError, match="one target"):
        ev.check_supported_profile(dataclasses.replace(profile, sites=(first, other)))


# --------------------------------------------------------------------------------------------------------------------
# The checked-in evidence


@pytest.mark.skipif(not (EVIDENCE / ev.REPORT_JSON).is_file(), reason="no checked-in evaluation report")
def test_the_checked_in_report_matches_its_inputs():
    report = json.loads((EVIDENCE / ev.REPORT_JSON).read_text(encoding="utf-8"))
    assert (EVIDENCE / ev.REPORT_MD).read_text(encoding="utf-8") == ev.render_report_markdown(report)
    spec = ev.load_spec(SPEC)
    assert report["spec"]["sha256"] == spec.sha256
    assert (EVIDENCE / report["spec"]["path"]).resolve() == SPEC.resolve()
    from model_pipeline.interventions.contract import sha256_file

    assert report["profile"]["sha256"] == sha256_file(CONCISE / "profile.json")
    for name, path in (("target", spec.target_prompts), ("neutral", spec.neutral), ("capability", spec.capability)):
        assert report["prompt_sets"][name]["sha256"] == sha256_file(path), name
        assert (EVIDENCE / report["prompt_sets"][name]["path"]).resolve() == path.resolve()
    rows = [json.loads(line) for line in (EVIDENCE / ev.GENERATIONS).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == (len(spec.strengths) + len(spec.control_strengths)) * (16 + 24)
    # The recommendation follows from the report's own numbers and the spec's rules.
    profile = {e["strength"]: e for e in report["arms"]["profile"]}
    control = {e["strength"]: e for e in report["arms"]["control"]}
    assert ev.recommend(spec.recommendation, profile, control) == report["recommendation"]
    # Summaries follow from the generations.
    for arm, entries in report["arms"].items():
        for e in entries:
            target = [r for r in rows if r["arm"] == arm and r["set"] == "target" and r["strength"] == e["strength"]]
            assert e["target"]["finished"] == sum(r["finished"] for r in target)
            assert e["target"]["median_tokens"] == pytest.approx(float(np.median([r["tokens"] for r in target])))
            assert e["target"]["keywords"] == sum(r["keywords"] for r in target)
            assert e["target"]["degenerate"] == sum(r["degenerate"] for r in target)
            capability = [
                r for r in rows if r["arm"] == arm and r["set"] == "capability" and r["strength"] == e["strength"]
            ]
            assert e["capability"]["correct"] == sum(r["correct"] for r in capability)
    # Every stored score follows from the stored text under the current scorers.
    target_by_id = {t.id: t for t in ev.load_target(spec.target_prompts)}
    checks = {c.id: c.check for c in ev.load_capability(spec.capability)}
    for r in rows:
        if r["set"] == "target":
            assert r["keywords"] == ev.keyword_hits(target_by_id[r["id"]].keywords, r["text"]), r["id"]
            assert r["degenerate"] == ev.is_degenerate(r["text"]), r["id"]
        else:
            assert r["correct"] == ev.score_capability(checks[r["id"]], r["text"]), (r["arm"], r["strength"], r["id"])
