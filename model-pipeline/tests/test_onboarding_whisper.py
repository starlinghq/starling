"""Weight-free checks of the Whisper ASR route: contract allow-list, corpus, evaluator and orchestration."""

from __future__ import annotations

import array
import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from model_pipeline import fixtures, onboarding_bert, onboarding_plugins
from model_pipeline import onboarding_whisper as whisper
from model_pipeline.cli import main
from model_pipeline.onboarding_candidate import load_candidate
from model_pipeline.onboarding_runner import PipelineError

REPO = Path(__file__).resolve().parents[2]
PROFILE = REPO / "model-pipeline/candidates/whisper-large-v3-turbo/profile.json"
RECIPE = REPO / "model-pipeline/recipes/whisper-large-v3-turbo-cpu.yaml"
CORPUS = REPO / "model-pipeline/reference/corpus/asr-corpus-v2.jsonl"
GRAPHS = "onnx-community/whisper-large-v3-turbo-ONNX"


def profile() -> dict:
    return json.loads(PROFILE.read_text(encoding="utf-8"))


# --- contract -------------------------------------------------------------------------------------------------------


def test_shipped_candidate_passes_schema_and_exact_contract():
    candidate = load_candidate(PROFILE)
    contract = whisper.validate_candidate_behavior(candidate)
    assert candidate["id"] == contract["id"] == "whisper-large-v3-turbo/onnx-fp16-cpu"
    assert candidate["task"] == "transcription" and candidate["runtime"]["precision"] == "fp16"
    behavior = candidate["behavior"]
    assert (behavior["num_mel_bins"], behavior["num_frames"], behavior["encoder_positions"]) == (128, 3000, 1500)
    assert (behavior["vocabulary_size"], behavior["hidden_size"]) == (51866, 1280)
    assert whisper.prompt(candidate, "en") == [50258, 50259, 50360, 50364]
    assert whisper.prompt(candidate, "en", timestamps=True) == [50258, 50259, 50360]
    assert behavior["timestamps"]["timestamp_begin"] == behavior["special_tokens"]["no_timestamps"] + 1


def test_timestamp_constants_match_the_dotnet_decoder():
    source = (REPO / "ai-dotnet/src/Starlings.Models.Whisper/WhisperModel.cs").read_text(encoding="utf-8")
    timestamps = whisper.TIMESTAMPS
    assert f"TimestampBegin = {timestamps['timestamp_begin']};" in source
    assert f"MaxInitialTimestampIndex = {timestamps['max_initial_timestamp_index']};" in source
    assert f"TimePrecision = {timestamps['time_precision']};" in source
    segmenter = (REPO / "ai-dotnet/src/Starlings.Models.Whisper/WhisperTranscription.cs").read_text(encoding="utf-8")
    assert f"InputStride = {timestamps['input_stride']};" in segmenter
    assert f"FeaturePrecision = {timestamps['feature_precision']}f;" in segmenter


def test_graph_pins_match_the_dotnet_asset_lock():
    source = (REPO / "ai-dotnet/src/Starlings.Models.Whisper/WhisperModelAssets.cs").read_text(encoding="utf-8")
    contract = whisper.CONTRACTS[GRAPHS]
    assert f'Revision = "{contract["revision"]}"' in source
    for path, (size, sha256) in contract["pins"].items():
        assert f'"{sha256}"' in source and f"{size}, " in source, path


def test_suppression_lists_match_the_dotnet_decoder():
    source = (REPO / "ai-dotnet/src/Starlings.Models.Whisper/WhisperModel.cs").read_text(encoding="utf-8")
    listed = source.split("SuppressTokens =", 1)[1].split("];", 1)[0]
    assert [int(value) for value in listed.replace("[", "").split(",") if value.strip()] == whisper.SUPPRESS_TOKENS
    assert "BeginSuppressTokens = [220, 50257]" in source


NEAR_MISSES = [
    ("task", lambda c: c.update(task="generation"), "task"),
    ("identifier", lambda c: c.update(id="whisper-large-v3-turbo/onnx-fp16-cuda"), "id"),
    ("repository", lambda c: c["origin"].update(repo="onnx-community/whisper-large-v3-ONNX"), "origin.repo"),
    ("tag revision", lambda c: c["origin"].update(revision="main"), "origin.revision"),
    ("base repository", lambda c: c["origin"]["base"].update(repo="openai/whisper-large-v3"), "origin.base.repo"),
    ("base revision", lambda c: c["origin"]["base"].update(revision="0" * 40), "origin.base.revision"),
    ("missing graph", lambda c: c["origin"]["files"].pop(3), "origin.files"),
    ("other encoder", lambda c: c["origin"]["files"][4].update(sha256="f" * 64), "origin.files"),
    ("missing weights", lambda c: c["origin"]["base"]["files"].pop(4), "origin.base.files"),
    ("fp32 runtime", lambda c: c["runtime"].update(precision="fp32"), "runtime"),
    ("cuda", lambda c: c["runtime"].update(provider="CUDAExecutionProvider"), "runtime"),
    ("mel bins", lambda c: c["behavior"].update(num_mel_bins=80), "behavior.num_mel_bins"),
    ("hop", lambda c: c["behavior"].update(hop_length=320), "behavior.hop_length"),
    ("vocabulary", lambda c: c["behavior"].update(vocabulary_size=51865), "behavior.vocabulary_size"),
    ("special token", lambda c: c["behavior"]["special_tokens"].update(transcribe=50359), "behavior.special_tokens"),
    ("other language", lambda c: c["behavior"]["languages"].update(fr=50265), "behavior.languages"),
    ("timestamps off", lambda c: c["behavior"].update(timestamps=False), "behavior.timestamps"),
    ("word timestamps", lambda c: c["behavior"]["timestamps"].update(level="word"), "behavior.timestamps"),
    (
        "initial timestamp",
        lambda c: c["behavior"]["timestamps"].update(max_initial_timestamp_index=1),
        "behavior.timestamps",
    ),
    ("translation", lambda c: c["behavior"].update(task="translate"), "behavior.task"),
    ("sampling", lambda c: c["behavior"].update(decoding="sample"), "behavior.decoding"),
    ("suppression", lambda c: c["behavior"]["suppress_tokens"].pop(), "behavior.suppress_tokens"),
    ("extra behavior", lambda c: c["behavior"].update(beams=5), "behavior"),
    ("tight mel bound loosened", lambda c: c["evaluation"]["policy"].update(mel_max_abs=1e-3), "mel_max_abs"),
    ("canary loosened", lambda c: c["evaluation"]["policy"].update(same_backend_max_abs=1e-2), "same_backend"),
    ("ids made fuzzy", lambda c: c["evaluation"]["policy"].update(token_ids="wer"), "token_ids"),
    ("transcript made fuzzy", lambda c: c["evaluation"]["policy"].update(transcript="wer"), "transcript"),
    ("dropped path", lambda c: c["evaluation"]["policy"]["required_paths"].pop(), "required_paths"),
    ("hidden over cap", lambda c: c["evaluation"]["policy"].update(hidden_max_abs=1.5), "hidden_max_abs"),
    (
        "padding over cap",
        lambda c: c["evaluation"]["policy"].update(hidden_padding_max_abs=3.0),
        "hidden_padding_max_abs",
    ),
    ("logits over cap", lambda c: c["evaluation"]["policy"].update(logits_max_abs=2.0), "logits_max_abs"),
    ("wer over cap", lambda c: c["evaluation"]["policy"].update(reference_wer_max=0.5), "reference_wer_max"),
    ("negative wer", lambda c: c["evaluation"]["policy"].update(reference_wer_max=-0.1), "reference_wer_max"),
    ("boolean bound", lambda c: c["evaluation"]["policy"].update(hidden_max_abs=True), "hidden_max_abs"),
    ("unknown policy", lambda c: c["evaluation"]["policy"].update(beam_max=1), "evaluation.policy"),
]


@pytest.mark.parametrize(("name", "mutate", "field"), NEAR_MISSES, ids=[case[0] for case in NEAR_MISSES])
def test_contract_refuses_near_misses_field_by_field(name, mutate, field):
    candidate = profile()
    assert candidate["origin"]["files"][3]["path"] == whisper.DECODER_FILE
    assert candidate["origin"]["files"][4]["path"] == whisper.ENCODER_FILE
    assert candidate["origin"]["base"]["files"][4]["path"] == "model.safetensors"
    mutate(candidate)
    with pytest.raises(PipelineError, match=field.replace(".", r"\.")):
        whisper.validate_candidate_behavior(candidate)


def test_fidelity_bounds_may_be_proposed_within_caps():
    candidate = profile()
    candidate["evaluation"]["policy"].update(
        hidden_max_abs=0.1, hidden_padding_max_abs=2.0, logits_max_abs=0.5, reference_wer_max=0.1
    )
    whisper.validate_candidate_behavior(candidate)


TOLERANCES = REPO / "model-pipeline/protected/whisper-large-v3-turbo/onnx-fp16/tolerances.yaml"


PROFILE_ID = "whisper-large-v3-turbo/onnx-fp16-cpu"


def test_shipped_candidate_equals_the_binding_tolerances():
    # Against the candidate, not a copy in code: the maintainer may move a fidelity bound in protected/ together with
    # the candidate (neither is implementation code) without touching the plugin.
    assert whisper.protected_policy(TOLERANCES, PROFILE_ID) == profile()["evaluation"]["policy"]
    whisper.validate_candidate_behavior(load_candidate(PROFILE), REPO)


def test_numeric_bounds_may_be_written_as_integers_or_exponents(tmp_path):
    text = TOLERANCES.read_text(encoding="utf-8")
    text = text.replace("hidden_padding_max_abs: 1.0", "hidden_padding_max_abs: 1")
    text = text.replace("same_backend_max_abs: 1.0e-5", "same_backend_max_abs: 1e-5")  # a string in YAML 1.1
    (tmp_path / "tolerances.yaml").write_text(text, encoding="utf-8")
    bound = whisper.protected_policy(tmp_path / "tolerances.yaml", PROFILE_ID)
    assert bound == profile()["evaluation"]["policy"]
    assert type(bound["hidden_padding_max_abs"]) is float and type(bound["same_backend_max_abs"]) is float


def test_a_policy_within_caps_but_off_the_binding_tolerances_is_refused():
    candidate = profile()
    candidate["evaluation"]["policy"]["hidden_max_abs"] = 0.1
    whisper.validate_candidate_behavior(candidate)  # within the route's caps
    with pytest.raises(PipelineError, match="hidden_max_abs \\(binding .*tolerances.yaml\\)"):
        whisper.validate_candidate_behavior(candidate, REPO)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda text: text.replace("onnx-fp16-cpu", "onnx-fp16-cuda"), "profile"),
        (lambda text: text.replace("  logits_max_abs: 0.1\n", ""), "policy fields"),
        (lambda text: text + "extra: 1\n", "profile, reference, policy"),
        (lambda text: text.replace("logits_max_abs: 0.1", "logits_max_abs: small"), "must be a number"),
        (lambda text: text.replace("logits_max_abs: 0.1", "logits_max_abs: .inf"), "finite number"),
        (lambda text: "[unclosed", "invalid YAML"),
    ],
    ids=["other profile", "missing bound", "unknown field", "non-numeric bound", "infinite bound", "not yaml"],
)
def test_unexpected_tolerance_layouts_are_refused(tmp_path, change, message):
    (tmp_path / "tolerances.yaml").write_text(change(TOLERANCES.read_text(encoding="utf-8")), encoding="utf-8")
    with pytest.raises(PipelineError, match=message):
        whisper.protected_policy(tmp_path / "tolerances.yaml", PROFILE_ID)


def test_the_evaluator_itself_applies_the_binding(tmp_path, monkeypatch):
    """evaluate_development refuses an off-binding policy when it is given the repository, before reading evidence."""
    candidate = profile()
    candidate["evaluation"]["policy"]["logits_max_abs"] = 0.2
    with pytest.raises(PipelineError, match="logits_max_abs \\(binding"):
        whisper.evaluate_development(candidate, PROFILE, tmp_path, tmp_path, tmp_path, [], REPO)


def test_other_routes_still_refuse_fp16_and_transcription():
    bert = json.loads((REPO / "model-pipeline/candidates/minilm/profile.json").read_text(encoding="utf-8"))
    bert["runtime"]["precision"] = "fp16"
    with pytest.raises(PipelineError, match="runtime"):
        onboarding_bert.validate_candidate_behavior(bert)


def test_plugin_is_registered_and_recipe_plans(capsys):
    assert onboarding_plugins.PLUGINS["whisper-onnx-cpu-v1"] == "model_pipeline.onboarding_whisper"
    assert main(["plan", str(RECIPE)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["plugin"] == "whisper-onnx-cpu-v1"
    assert plan["stages"] == ["assets", "environment", "reference", "onnx", "dotnet", "evaluate", "review"]
    assert plan["base"]["repo"] == "openai/whisper-large-v3-turbo"
    assert plan["qualification_status"] == "unapproved" and plan["reference_status"] == "proposed"
    assert plan["policy_source"] == "model-pipeline/protected/whisper-large-v3-turbo/onnx-fp16/tolerances.yaml"


# --- configs and corpus ---------------------------------------------------------------------------------------------


def write_configs(directory: Path, behavior: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": [behavior["architecture"]],
        "d_model": behavior["hidden_size"],
        "encoder_layers": behavior["encoder_layers"],
        "decoder_layers": behavior["decoder_layers"],
        "num_mel_bins": behavior["num_mel_bins"],
        "vocab_size": behavior["vocabulary_size"],
        "max_source_positions": behavior["encoder_positions"],
        "max_target_positions": behavior["max_target_positions"],
        "decoder_start_token_id": 50258,
        "eos_token_id": 50257,
    }
    generation = {
        "suppress_tokens": behavior["suppress_tokens"],
        "begin_suppress_tokens": behavior["begin_suppress_tokens"],
        "no_timestamps_token_id": 50364,
        "max_initial_timestamp_index": 50,
        "task_to_id": {"transcribe": 50360, "translate": 50359},
        "lang_to_id": {"<|en|>": 50259, "<|fr|>": 50265},
        "max_length": 448,
    }
    preprocessor = {
        "feature_size": 128,
        "sampling_rate": 16000,
        "n_fft": 400,
        "hop_length": 160,
        "n_samples": 480000,
        "nb_max_frames": 3000,
        "padding_value": 0.0,
    }
    for name, value in (
        ("config.json", config),
        ("generation_config.json", generation),
        ("preprocessor_config.json", preprocessor),
    ):
        (directory / name).write_text(json.dumps(value), encoding="utf-8")


def test_config_contract_checks_both_repositories(tmp_path):
    candidate = profile()
    write_configs(tmp_path / "graphs", candidate["behavior"])
    write_configs(tmp_path / "base", candidate["behavior"])
    checked = whisper.check_configs(candidate, tmp_path / "graphs", tmp_path / "base")
    assert "generation.suppress_tokens" in checked["snapshot"] and checked["snapshot"] == checked["base"]
    generation = json.loads((tmp_path / "base/generation_config.json").read_text())
    generation["begin_suppress_tokens"] = [220, 50256]
    (tmp_path / "base/generation_config.json").write_text(json.dumps(generation))
    with pytest.raises(PipelineError, match="base generation.begin_suppress_tokens"):
        whisper.check_configs(candidate, tmp_path / "graphs", tmp_path / "base")


def test_config_contract_checks_the_timestamp_settings(tmp_path):
    candidate = profile()
    write_configs(tmp_path / "graphs", candidate["behavior"])
    write_configs(tmp_path / "base", candidate["behavior"])
    generation = json.loads((tmp_path / "graphs/generation_config.json").read_text())
    generation["max_initial_timestamp_index"] = 1
    (tmp_path / "graphs/generation_config.json").write_text(json.dumps(generation))
    with pytest.raises(PipelineError, match="snapshot generation.max_initial_timestamp_index"):
        whisper.check_configs(candidate, tmp_path / "graphs", tmp_path / "base")


def test_shipped_corpus_is_pinned_and_differently_sized():
    rows = whisper.load_corpus(CORPUS, profile())
    assert len(rows) >= 5 and len({row["samples"] for row in rows}) == len(rows)
    assert all(row["language"] == "en" and row["license"] == "CC-BY-4.0" for row in rows)
    assert all(row["source"]["revision"] == "5be91486e11a2d616f4ec5db8d3fd248585ac07a" for row in rows)


def test_corpus_v2_extends_v1_unchanged_with_a_composite_clip():
    first = fixtures.read_jsonl(CORPUS.parent / "asr-corpus-v1.jsonl")
    second = fixtures.read_jsonl(CORPUS)
    assert second[: len(first)] == first
    composite = [row for row in second if "utterances" in row["source"]]
    assert composite and all(len(row["source"]["utterances"]) > 1 for row in composite)
    assert all(row["audio"].startswith("asr-v2/") and row["samples"] <= 480000 for row in composite)


def corpus_copy(tmp_path: Path, mutate=None) -> Path:
    rows = fixtures.read_jsonl(CORPUS)
    (tmp_path / "asr-v1").mkdir()
    (tmp_path / "asr-v2").mkdir()
    for row in rows:
        (tmp_path / row["audio"]).write_bytes((CORPUS.parent / row["audio"]).read_bytes())
    if mutate:
        mutate(tmp_path, rows)
    fixtures.write_jsonl(tmp_path / "asr.jsonl", rows)
    return tmp_path / "asr.jsonl"


@pytest.mark.parametrize(
    ("name", "mutate", "message"),
    [
        ("changed audio", lambda d, rows: (d / rows[0]["audio"]).write_bytes(b"RIFF"), "SHA-256"),
        ("escaping path", lambda d, rows: rows[0].update(audio="../asr-v1/x.wav"), "unsafe"),
        ("not wav", lambda d, rows: rows[0].update(audio="asr-v1/x.flac"), "non-WAV"),
        ("over 30 s", lambda d, rows: rows[0].update(samples=480001), "30 s"),
        ("other rate", lambda d, rows: rows[0].update(sample_rate=8000), "16 kHz"),
        ("other language", lambda d, rows: rows[0].update(language="fr"), "language"),
        ("duplicate id", lambda d, rows: rows[1].update(id=rows[0]["id"]), "unique"),
        ("wrong sample count", lambda d, rows: rows[0].update(samples=rows[0]["samples"] + 1), "indexed clip"),
        (
            "one size",
            lambda d, rows: [row.update({key: rows[0][key] for key in ("audio", "sha256", "samples")}) for row in rows],
            "differently sized",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_corpus_refusals(tmp_path, name, mutate, message):
    with pytest.raises(PipelineError, match=message):
        whisper.load_corpus(corpus_copy(tmp_path, mutate), profile())


def test_word_error_rate():
    assert whisper.word_error_rate(" Mister Quilter, the apostle.", "MISTER QUILTER THE APOSTLE") == 0
    assert whisper.word_error_rate("The 20s.", "THE TWENTIES") == 0.5
    assert whisper.word_error_rate("he doesn't work", "HE DOESN'T WORK AT ALL") == 0.4
    assert whisper.word_error_rate("a b c", "") == 1.0


# --- evaluator ------------------------------------------------------------------------------------------------------

TINY = {"num_mel_bins": 2, "num_frames": 3, "encoder_positions": 4, "hidden_size": 3}
# clip-a covers two of the four encoder positions (the rest is padding); clip-b covers all four.
SAMPLES = {"clip-a": 1, "clip-b": 640}
GENERATED = {"clip-a": [440, 1029, 13], "clip-b": [2221, 50257 - 1]}


@pytest.fixture
def tiny(monkeypatch):
    """Shrink the tensor geometry so the evaluator runs on small synthetic evidence; the vocabulary stays real."""
    for key, value in TINY.items():
        monkeypatch.setitem(whisper.BEHAVIOR, key, value)
    candidate = profile()
    candidate["behavior"] = copy.deepcopy(whisper.BEHAVIOR)
    return candidate


def tensor_values(seed: int, count: int) -> list[float]:
    return [((seed * 7919 + index * 104729) % 2001 - 1000) / 250.0 for index in range(count)]


def timestamped(candidate: dict, generated: list[int], text: str, redecoded: bool = False) -> dict:
    """One window: <|0.00|> text <|1.00|> end of text, a single segment. `redecoded`: the window closes that segment
    with a timestamp pair and ends inside the next, so a second window starts at its closing timestamp."""
    start = whisper.prompt(candidate, "en", timestamps=True)
    tokens = [50365, *generated, 50415]
    if not redecoded:
        return {
            "windows": [{"seek": 0, "ids": start + tokens + [50257]}],
            "segments": [{"start": 0.0, "end": 1.0, "ids": tokens, "text": text}],
            "transcript": text,
        }
    return {
        "windows": [
            {"seek": 0, "ids": start + tokens + [50415, 440, 50257]},
            {"seek": 100, "ids": start + [50365, 1029, 50380, 50257]},
        ],
        "segments": [
            {"start": 0.0, "end": 1.0, "ids": [*tokens, 50415], "text": text},
            {"start": 1.0, "end": 1.3, "ids": [50365, 1029, 50380], "text": " more"},
        ],
        "transcript": text + " more",
    }


def write_stage(
    directory: Path, stage: str, candidate: dict, digest: str, reference_sha: str | None, edit=None
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    behavior = candidate["behavior"]
    if stage == "reference":
        names = ("model.safetensors", "preprocessor_config.json", "tokenizer.json")
        files = candidate["origin"]["base"]["files"]
    else:
        names = (whisper.ENCODER_FILE, whisper.DECODER_FILE, whisper.TOKENIZER_FILE)
        files = candidate["origin"]["files"]
    pins = {record["path"]: record["sha256"] for record in files}
    cases, layout, blob = [], [], array.array("f")
    shapes = {
        "mel": [behavior["num_mel_bins"], behavior["num_frames"]],
        "mel_torch": [behavior["num_mel_bins"], behavior["num_frames"]],
        "hidden": [behavior["encoder_positions"], behavior["hidden_size"]],
    }
    for seed, (identifier, generated) in enumerate(GENERATED.items()):
        ids = whisper.prompt(candidate, "en") + generated + [50257]
        row = {
            "id": identifier,
            "language": "en",
            "audio_sha256": hashlib.sha256(identifier.encode()).hexdigest(),
            "ids": ids,
            "transcript": f" Words of {identifier}.",
        }
        if stage == "reference":
            row.update(text=f"WORDS OF {identifier.upper()}", samples=SAMPLES[identifier], default_path_ids=ids)
        else:
            row["forced_ids"] = ids
        row["timestamped"] = timestamped(candidate, generated, row["transcript"], identifier == "clip-b")
        if stage == "dotnet":
            row["audio_transcript"] = row["transcript"]
            row["audio_timestamped"] = timestamped(candidate, generated, row["transcript"], identifier == "clip-b")
        cases.append(row)
        for name in whisper.TENSORS[stage]:
            shape = shapes.get(name) or [len(ids) - 1, behavior["vocabulary_size"]]
            layout.append({"id": identifier, "name": name, "offset": len(blob), "shape": shape})
            blob.extend(tensor_values(seed + len(name.removesuffix("_torch")), shape[0] * shape[1]))
    document = {
        "format": whisper.FORMAT,
        "format_version": 1,
        "profile": candidate["id"],
        "task": "transcription",
        "stage": stage,
        "candidate_sha256": digest,
        "assets": [{"path": name, "sha256": pins[name]} for name in names],
        "environment": {"machine": "test-host", "provider": "CPUExecutionProvider", "onnxruntime": "1.30.0"},
        "cases": cases,
    }
    if reference_sha:
        document["reference_sha256"] = reference_sha
    if stage == "dotnet":
        document["checks"] = {"pre_cancelled_request": True}
    if edit:
        edit(document, blob)
    sidecar = directory / f"{stage}-tensors.f32"
    sidecar.write_bytes(blob.tobytes())
    document["tensors"] = {"file": sidecar.name, "sha256": fixtures.sha256_file(sidecar), "cases": layout}
    path = directory / f"{stage}.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def synthetic_corpus() -> list[dict]:
    """The pinned-corpus rows write_stage's reference cases describe."""
    return [
        {
            "id": identifier,
            "sha256": hashlib.sha256(identifier.encode()).hexdigest(),
            "samples": SAMPLES[identifier],
            "language": "en",
            "text": f"WORDS OF {identifier.upper()}",
        }
        for seed, identifier in enumerate(GENERATED)
    ]


def evidence(tmp_path: Path, candidate: dict, edits: dict | None = None, corpus: list[dict] | None = None) -> dict:
    edits = edits or {}
    candidate_path = tmp_path / "profile.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    digest = fixtures.sha256_file(candidate_path)
    reference = write_stage(tmp_path / "reference", "reference", candidate, digest, None, edits.get("reference"))
    reference_sha = fixtures.sha256_file(reference)
    onnx = write_stage(tmp_path / "onnx", "onnx", candidate, digest, reference_sha, edits.get("onnx"))
    dotnet = write_stage(tmp_path / "dotnet", "dotnet", candidate, digest, reference_sha, edits.get("dotnet"))
    return {
        "candidate": candidate,
        "candidate_path": candidate_path,
        "reference_path": reference,
        "onnx_path": onnx,
        "dotnet_path": dotnet,
        "corpus": synthetic_corpus() if corpus is None else corpus,
    }


def test_complete_evidence_passes_and_never_adopts_protected_gates(tmp_path, tiny):
    report = whisper.evaluate_development(**evidence(tmp_path, tiny))
    assert report["status"] == "passed", report["failed_checks"]
    assert report["protected_release_qualification"] == {"status": "unapproved", "G1": "not-run", "G2": "not-run"}
    assert report["evaluation_status"] == "proposed" and report["case_count"] == 2
    # Synthetic evidence runs without the repository, so its (tiny-geometry) policy is checked against the caps only.
    assert (report["policy_status"], report["policy_source"]) == ("proposed", "unbound (route caps only)")
    per_clip = {check["id"].replace("/clip-a/", "/*/") for check in report["checks"] if "/clip-a/" in check["id"]}
    assert per_clip == {
        "reference/*/mel_torch_vs_numpy",
        "reference/*/default_path_ids",
        "dotnet-audio/*/mel_vs_reference",
        "onnx/*/hidden_vs_reference",
        "onnx/*/hidden_padding_vs_reference",
        "dotnet-mel/*/hidden_vs_python",
        "onnx/*/logits_vs_reference",
        "dotnet-mel/*/logits_vs_python",
        "dotnet-mel/*/ids_vs_python",
        "dotnet-mel/*/transcript_vs_python",
        "dotnet-audio/*/transcript_vs_python",
        "onnx/*/wer_vs_reference",
        "dotnet-audio/*/wer_vs_reference",
        "onnx/*/timestamp_ids_vs_reference",
        "onnx/*/segment_times_vs_reference",
        "onnx/*/timestamp_wer_vs_reference",
        "dotnet-mel/*/timestamp_ids_vs_python",
        "dotnet-mel/*/segments_vs_python",
        "dotnet-mel/*/timestamp_transcript_vs_python",
        "dotnet-audio/*/segments_vs_python",
        "dotnet-audio/*/timestamp_transcript_vs_python",
    }
    assert [check["id"] for check in report["checks"][-3:]] == [
        "reference/corpus/wer_vs_ground_truth",
        "reference/corpus/timestamp_pair_coverage",
        "reference/corpus/redecode_coverage",
    ]


def nudge(name: str, delta: float, clip: str = "clip-a", index: int = 0):
    def edit(document, blob):
        entry = next(layout for layout in _layout(document, blob) if (layout["id"], layout["name"]) == (clip, name))
        blob[entry["offset"] + index] += delta

    return edit


def _layout(document, blob):
    """Recompute the layout write_stage is about to record (tensors are appended in case, then name order)."""
    position, layout = 0, []
    behavior = whisper.BEHAVIOR
    shapes = {
        "mel": behavior["num_mel_bins"] * behavior["num_frames"],
        "mel_torch": behavior["num_mel_bins"] * behavior["num_frames"],
        "hidden": behavior["encoder_positions"] * behavior["hidden_size"],
    }
    for row in document["cases"]:
        for name in whisper.TENSORS[document["stage"]]:
            size = shapes.get(name) or (len(row["ids"]) - 1) * behavior["vocabulary_size"]
            layout.append({"id": row["id"], "name": name, "offset": position})
            position += size
    return layout


@pytest.mark.parametrize(
    ("stage", "edit", "failed"),
    [
        ("dotnet", nudge("mel", 1e-5), "dotnet-audio/clip-a/mel_vs_reference"),
        ("dotnet", nudge("hidden", 1e-4), "dotnet-mel/clip-a/hidden_vs_python"),
        ("dotnet", nudge("logits", 1e-4, "clip-b"), "dotnet-mel/clip-b/logits_vs_python"),
        ("onnx", nudge("hidden", 1.0), "onnx/clip-a/hidden_vs_reference"),
        ("onnx", nudge("hidden", 1.5, index=2 * TINY["hidden_size"]), "onnx/clip-a/hidden_padding_vs_reference"),
        ("reference", nudge("mel_torch", 1e-2), "reference/clip-a/mel_torch_vs_numpy"),
        (
            "dotnet",
            lambda document, blob: document["cases"][0].update(ids=document["cases"][0]["ids"][:-2] + [50257]),
            "dotnet-mel/clip-a/ids_vs_python",
        ),
        (
            "dotnet",
            lambda document, blob: document["cases"][1].update(transcript=" Word of clip-b."),
            "dotnet-mel/clip-b/transcript_vs_python",
        ),
        (
            "dotnet",
            lambda document, blob: document["cases"][1].update(audio_transcript=" Words of clip b"),
            "dotnet-audio/clip-b/transcript_vs_python",
        ),
        (
            "reference",
            lambda document, blob: document["cases"][0].update(
                default_path_ids=document["cases"][0]["ids"][:-2] + [50257]
            ),
            "reference/clip-a/default_path_ids",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) and "/" in value else "",
)
def test_differences_produce_named_failed_checks(tmp_path, tiny, stage, edit, failed):
    report = whisper.evaluate_development(**evidence(tmp_path, tiny, {stage: edit}))
    assert report["status"] == "failed"
    assert failed in report["failed_checks"]


def _segment(stage_key: str, index: int, **values):
    def edit(document, blob):
        document["cases"][0][stage_key]["segments"][index].update(values)

    return edit


def _window(stage_key: str, ids: list[int]):
    def edit(document, blob):
        window = document["cases"][0][stage_key]["windows"][0]
        window["ids"] = window["ids"][:3] + ids

    return edit


@pytest.mark.parametrize(
    ("stage", "edit", "failed"),
    [
        ("onnx", _segment("timestamped", 0, end=1.02), "onnx/clip-a/segment_times_vs_reference"),
        ("onnx", _window("timestamped", [50365, 440, 50416, 50257]), "onnx/clip-a/timestamp_ids_vs_reference"),
        ("dotnet", _window("timestamped", [50365, 440, 50416, 50257]), "dotnet-mel/clip-a/timestamp_ids_vs_python"),
        ("dotnet", _segment("timestamped", 0, start=0.02), "dotnet-mel/clip-a/segments_vs_python"),
        ("dotnet", _segment("timestamped", 0, text=" Other."), "dotnet-mel/clip-a/segments_vs_python"),
        ("dotnet", _segment("audio_timestamped", 0, ids=[50365, 50415]), "dotnet-audio/clip-a/segments_vs_python"),
        (
            "dotnet",
            lambda document, blob: document["cases"][0]["audio_timestamped"].update(transcript=" Words."),
            "dotnet-audio/clip-a/timestamp_transcript_vs_python",
        ),
        (
            "reference",
            lambda document, blob: document["cases"][0]["timestamped"].update(transcript=" Other words entirely."),
            "onnx/clip-a/timestamp_wer_vs_reference",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) and "/" in value else "",
)
def test_timestamp_differences_produce_named_failed_checks(tmp_path, tiny, stage, edit, failed):
    report = whisper.evaluate_development(**evidence(tmp_path, tiny, {stage: edit}))
    assert report["status"] == "failed"
    assert failed in report["failed_checks"]


@pytest.mark.parametrize(
    ("stage", "edit", "message"),
    [
        ("onnx", lambda document, blob: document["cases"][0].pop("timestamped"), "incomplete or malformed"),
        ("dotnet", lambda document, blob: document["cases"][0].pop("audio_timestamped"), "incomplete or malformed"),
        ("reference", _window("timestamped", [50364, 50365, 50257]), "invalid timestamped window"),
        ("onnx", _window("timestamped", [50365, 50257, 50365, 50257]), "invalid timestamped window"),
        (
            "onnx",
            lambda document, blob: document["cases"][0]["timestamped"]["windows"][0].update(seek=160),
            "frame 0",
        ),
        ("dotnet", _segment("timestamped", 0, start=2.0), "invalid segment"),
        ("dotnet", _segment("timestamped", 0, end=float("nan")), "invalid segment"),
        ("dotnet", _segment("timestamped", 0, ids=[]), "invalid segment"),
        (
            "reference",
            lambda document, blob: document["cases"][0]["timestamped"].update(segments=[]),
            "at least one window and one segment",
        ),
    ],
    ids=[
        "no timestamped decode",
        "no audio timestamped decode",
        "no-timestamps prompt",
        "eos inside a window",
        "first window not at 0",
        "start after end",
        "non-finite end",
        "empty segment",
        "no segments",
    ],
)
def test_malformed_timestamp_evidence_is_refused(tmp_path, tiny, stage, edit, message):
    with pytest.raises(PipelineError, match=message):
        whisper.evaluate_development(**evidence(tmp_path, tiny, {stage: edit}))


def test_transcript_drift_against_the_reference_fails_its_wer_bound(tmp_path, tiny):
    def drift(document, blob):
        document["cases"][0]["transcript"] = " Other words entirely."

    report = whisper.evaluate_development(**evidence(tmp_path, tiny, {"reference": drift}))
    assert "onnx/clip-a/wer_vs_reference" in report["failed_checks"]
    assert "dotnet-audio/clip-a/wer_vs_reference" in report["failed_checks"]


def test_ground_truth_wer_bound_catches_agreeing_but_wrong_transcripts(tmp_path, tiny):
    corpus = synthetic_corpus()
    for row in corpus:
        row["text"] = "SOMETHING ELSE WAS SAID HERE"

    report = whisper.evaluate_development(**evidence(tmp_path, tiny, corpus=corpus))
    assert report["failed_checks"] == ["reference/corpus/wer_vs_ground_truth"]


def test_ground_truth_comes_from_the_pinned_corpus_not_the_reference(tmp_path, tiny):
    def rewrite(document, blob):
        for row in document["cases"]:
            row["text"] = row["transcript"]

    corpus = synthetic_corpus()
    for row in corpus:
        row["text"] = "SOMETHING ELSE WAS SAID HERE"
    report = whisper.evaluate_development(**evidence(tmp_path, tiny, {"reference": rewrite}, corpus))
    assert report["failed_checks"] == ["reference/corpus/wer_vs_ground_truth"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda corpus: corpus.pop(), "exactly the pinned corpus"),
        (lambda corpus: corpus[0].update(sha256="0" * 64), "pinned corpus"),
        (lambda corpus: corpus[1].update(samples=999), "pinned corpus"),
    ],
    ids=["missing clip", "other audio", "other sample count"],
)
def test_reference_must_describe_the_pinned_corpus(tmp_path, tiny, change, message):
    corpus = synthetic_corpus()
    change(corpus)
    with pytest.raises(PipelineError, match=message):
        whisper.evaluate_development(**evidence(tmp_path, tiny, corpus=corpus))


@pytest.mark.parametrize(
    ("samples", "positions"),
    [(1, 2), (26239, 84), (93680, 295), (480000, 1500)],
)
def test_audio_positions_follow_the_encoder_receptive_field(samples, positions):
    # 26239 samples: the last frame touching audio is (26238 + 200) // 160 = 165, so positions 0..83 see audio.
    assert whisper.audio_positions(samples, whisper.BEHAVIOR) == positions


def _set(path: tuple, value):
    def edit(document, blob):
        target = document
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return edit


@pytest.mark.parametrize(
    ("stage", "edit", "message"),
    [
        ("dotnet", _set(("checks", "pre_cancelled_request"), False), "cancellation"),
        ("onnx", _set(("reference_sha256",), "0" * 64), "different original reference"),
        ("onnx", _set(("environment", "machine"), "other-host"), "same machine"),
        ("onnx", _set(("environment", "onnxruntime"), "1.29.0"), "runtime/provider"),
        ("dotnet", _set(("stage",), "onnx"), "stage identity"),
        ("onnx", _set(("assets", 0, "sha256"), "f" * 64), "asset identities"),
        ("reference", _set(("cases", 0, "ids", 1), 50260), "invalid prompt"),
        ("reference", _set(("cases", 0, "ids", 5), 51866), "invalid prompt"),
        ("dotnet", _set(("cases", 0, "forced_ids"), [50258]), "teacher-forced ids"),
        ("onnx", _set(("cases", 1, "audio_sha256"), "0" * 64), "audio or teacher-forced"),
        ("dotnet", lambda document, blob: document["cases"].pop(), "missing required case rows"),
        ("dotnet", lambda document, blob: blob.append(0.0), "unreferenced values"),
        ("onnx", lambda document, blob: blob.__setitem__(0, float("nan")), "non-finite"),
    ],
    ids=lambda value: value if isinstance(value, str) and " " in value else "",
)
def test_incomplete_or_wrong_evidence_is_refused(tmp_path, tiny, stage, edit, message):
    with pytest.raises(PipelineError, match=message):
        whisper.evaluate_development(**evidence(tmp_path, tiny, {stage: edit}))


def test_tampered_tensor_sidecar_is_refused(tmp_path, tiny):
    paths = evidence(tmp_path, tiny)
    sidecar = tmp_path / "dotnet/dotnet-tensors.f32"
    sidecar.write_bytes(sidecar.read_bytes()[:-4] + b"\0\0\0\0")
    with pytest.raises(PipelineError, match="identity mismatch"):
        whisper.evaluate_development(**paths)


# --- orchestration --------------------------------------------------------------------------------------------------


def test_build_requires_a_base_snapshot_and_refuses_distribution(tmp_path):
    recipe = {"recipe_version": 1, "id": "w", "plugin": whisper.PLUGIN}
    with pytest.raises(PipelineError, match="base-snapshot"):
        whisper.build_stages(recipe, repo_root=REPO, snapshot=tmp_path, reference_python=tmp_path / "python")
    with pytest.raises(PipelineError, match="distribution"):
        whisper.build_stages(
            recipe, repo_root=REPO, snapshot=tmp_path, reference_python=tmp_path / "python", distribution=True
        )


def test_stage_graph_declares_audio_inputs_and_order(tmp_path, monkeypatch):
    from model_pipeline import onboarding_candidate

    monkeypatch.setattr(
        onboarding_candidate,
        "environment_identity",
        lambda python, root, timeout: {"python": {"status": "available"}, "dotnet": {"status": "available"}},
    )
    monkeypatch.setattr(onboarding_candidate, "_source", lambda snapshot, relative: snapshot / relative)
    recipe = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    (tmp_path / "graphs").mkdir()
    (tmp_path / "base").mkdir()
    python = tmp_path / "python"
    python.write_text("")
    stages = whisper.build_stages(
        recipe,
        repo_root=REPO,
        snapshot=tmp_path / "graphs",
        reference_python=python,
        base_snapshot=tmp_path / "base",
    )
    assert [stage.name for stage in stages] == whisper.STAGES
    by_name = {stage.name: stage for stage in stages}
    assert by_name["onnx"].dependencies == ("reference", "assets", "environment")
    audio = {key for key in by_name["reference"].inputs if key.startswith("corpus_audio:")}
    assert len(audio) == len(fixtures.read_jsonl(CORPUS))
    assert "ai-dotnet/src/Starlings.Models.Whisper/WhisperModel.cs" in by_name["dotnet"].inputs
    assert "model-pipeline/reference/whisper/run_whisper.py" in by_name["reference"].inputs


@pytest.mark.parametrize(
    ("edit", "failed"),
    [
        (
            lambda document, blob: document["cases"][1].update(
                timestamped=timestamped(profile(), GENERATED["clip-b"], document["cases"][1]["transcript"])
            ),
            "reference/corpus/redecode_coverage",
        ),
        (
            lambda document, blob: [
                row["timestamped"]["windows"][0]["ids"].__setitem__(-3, 440) for row in document["cases"][1:2]
            ],
            "reference/corpus/timestamp_pair_coverage",
        ),
    ],
    ids=["no re-decoded window", "no timestamp pair"],
)
def test_the_corpus_must_cover_the_timestamp_branches(tmp_path, tiny, edit, failed):
    report = whisper.evaluate_development(**evidence(tmp_path, tiny, {"reference": edit}))
    assert failed in report["failed_checks"]


def test_a_truncated_clip_is_left_out_of_the_ground_truth_bound(tmp_path, tiny):
    corpus = synthetic_corpus()
    corpus[0].update(text="SOMETHING ELSE WAS SAID HERE", source={"truncated_to_samples": 1})
    report = whisper.evaluate_development(**evidence(tmp_path, tiny, corpus=corpus))
    check = next(check for check in report["checks"] if check["id"] == "reference/corpus/wer_vs_ground_truth")
    assert check["status"] == "passed" and check["excluded"] == ["clip-a"]
