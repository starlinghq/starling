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
CORPUS = REPO / "model-pipeline/reference/corpus/asr-corpus-v1.jsonl"
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
    ("timestamps", lambda c: c["behavior"].update(timestamps=True), "behavior.timestamps"),
    ("translation", lambda c: c["behavior"].update(task="translate"), "behavior.task"),
    ("sampling", lambda c: c["behavior"].update(decoding="sample"), "behavior.decoding"),
    ("suppression", lambda c: c["behavior"]["suppress_tokens"].pop(), "behavior.suppress_tokens"),
    ("extra behavior", lambda c: c["behavior"].update(beams=5), "behavior"),
    ("tight mel bound loosened", lambda c: c["evaluation"]["policy"].update(mel_max_abs=1e-3), "mel_max_abs"),
    ("canary loosened", lambda c: c["evaluation"]["policy"].update(same_backend_max_abs=1e-2), "same_backend"),
    ("ids made fuzzy", lambda c: c["evaluation"]["policy"].update(token_ids="wer"), "token_ids"),
    ("transcript made fuzzy", lambda c: c["evaluation"]["policy"].update(transcript="wer"), "transcript"),
    ("dropped path", lambda c: c["evaluation"]["policy"]["required_paths"].pop(), "required_paths"),
    ("hidden over cap", lambda c: c["evaluation"]["policy"].update(hidden_max_abs=0.6), "hidden_max_abs"),
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
    candidate["evaluation"]["policy"].update(hidden_max_abs=0.5, logits_max_abs=1.0, reference_wer_max=0.1)
    whisper.validate_candidate_behavior(candidate)


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


def test_shipped_corpus_is_pinned_and_differently_sized():
    rows = whisper.load_corpus(CORPUS, profile())
    assert len(rows) >= 4 and len({row["samples"] for row in rows}) == len(rows)
    assert all(row["language"] == "en" and row["license"] == "CC-BY-4.0" for row in rows)
    assert all(row["source"]["revision"] == "5be91486e11a2d616f4ec5db8d3fd248585ac07a" for row in rows)


def corpus_copy(tmp_path: Path, mutate=None) -> Path:
    rows = fixtures.read_jsonl(CORPUS)
    (tmp_path / "asr-v1").mkdir()
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
        ("one size", lambda d, rows: [row.update(samples=1000) for row in rows], "differently sized"),
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

TINY = {"num_mel_bins": 2, "num_frames": 3, "encoder_positions": 2, "hidden_size": 3}
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
            row.update(text=f"WORDS OF {identifier.upper()}", samples=16000 * (seed + 1), default_path_ids=ids)
        else:
            row["forced_ids"] = ids
        if stage == "dotnet":
            row["audio_transcript"] = row["transcript"]
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


def evidence(tmp_path: Path, candidate: dict, edits: dict | None = None) -> dict:
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
    }


def test_complete_evidence_passes_and_never_adopts_protected_gates(tmp_path, tiny):
    report = whisper.evaluate_development(**evidence(tmp_path, tiny))
    assert report["status"] == "passed", report["failed_checks"]
    assert report["protected_release_qualification"] == {"status": "unapproved", "G1": "not-run", "G2": "not-run"}
    assert report["evaluation_status"] == "proposed" and report["case_count"] == 2
    per_clip = {check["id"].replace("/clip-a/", "/*/") for check in report["checks"] if "/clip-a/" in check["id"]}
    assert per_clip == {
        "reference/*/mel_torch_vs_numpy",
        "reference/*/default_path_ids",
        "dotnet-audio/*/mel_vs_reference",
        "onnx/*/hidden_vs_reference",
        "dotnet-mel/*/hidden_vs_python",
        "onnx/*/logits_vs_reference",
        "dotnet-mel/*/logits_vs_python",
        "dotnet-mel/*/ids_vs_python",
        "dotnet-mel/*/transcript_vs_python",
        "dotnet-audio/*/transcript_vs_python",
        "onnx/*/wer_vs_reference",
        "dotnet-audio/*/wer_vs_reference",
    }
    assert report["checks"][-1]["id"] == "reference/corpus/wer_vs_ground_truth"


def nudge(name: str, delta: float, clip: str = "clip-a"):
    def edit(document, blob):
        entry = next(layout for layout in _layout(document, blob) if (layout["id"], layout["name"]) == (clip, name))
        blob[entry["offset"]] += delta

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


def test_transcript_drift_against_the_reference_fails_its_wer_bound(tmp_path, tiny):
    def drift(document, blob):
        document["cases"][0]["transcript"] = " Other words entirely."

    report = whisper.evaluate_development(**evidence(tmp_path, tiny, {"reference": drift}))
    assert "onnx/clip-a/wer_vs_reference" in report["failed_checks"]
    assert "dotnet-audio/clip-a/wer_vs_reference" in report["failed_checks"]


def test_ground_truth_wer_bound_catches_agreeing_but_wrong_transcripts(tmp_path, tiny):
    def wrong(document, blob):
        for row in document["cases"]:
            row["text"] = "SOMETHING ELSE WAS SAID HERE"

    report = whisper.evaluate_development(**evidence(tmp_path, tiny, {"reference": wrong}))
    assert report["failed_checks"] == ["reference/corpus/wer_vs_ground_truth"]


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
