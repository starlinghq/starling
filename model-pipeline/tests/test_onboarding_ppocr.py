"""Weight-free checks of the PP-OCRv5 route: contract, configurations, corpus, evaluator and orchestration."""

from __future__ import annotations

import array
import copy
import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

from model_pipeline import fixtures, onboarding_bert, onboarding_plugins, onboarding_whisper
from model_pipeline import onboarding_ppocr as ppocr
from model_pipeline.cli import main
from model_pipeline.onboarding_candidate import load_candidate
from model_pipeline.onboarding_runner import PipelineError

REPO = Path(__file__).resolve().parents[2]
PROFILE = REPO / "model-pipeline/candidates/ppocrv5-mobile/profile.json"
RECIPE = REPO / "model-pipeline/recipes/ppocrv5-mobile-cpu.yaml"
CORPUS = REPO / "model-pipeline/reference/corpus/ocr-corpus-v1.jsonl"
SNAPSHOT = REPO / ".models/snapshots/ppocrv5-mobile"
DETECTOR = "PaddlePaddle/PP-OCRv5_mobile_det_onnx"


def profile() -> dict:
    return json.loads(PROFILE.read_text(encoding="utf-8"))


# --- contract -------------------------------------------------------------------------------------------------------


def test_shipped_candidate_passes_schema_and_exact_contract():
    candidate = load_candidate(PROFILE)
    contract = ppocr.validate_candidate_behavior(candidate)
    assert candidate["id"] == contract["id"] == "ppocrv5-mobile/onnx-fp32-cpu"
    assert candidate["task"] == "ocr" and candidate["runtime"]["precision"] == "fp32"
    assert [entry["role"] for entry in candidate["origin"]["components"]] == ["rec", "det-paddle", "rec-paddle"]
    ctc = candidate["behavior"]["ctc"]
    assert ctc["classes"] == ctc["dictionary_size"] + 2 == 18385


def test_policy_keeps_correctness_bounds_fixed():
    policy = profile()["evaluation"]["policy"]
    assert policy["preprocess_max_abs"] == policy["crop_max_abs"] == 0.0
    assert policy["line_text"] == policy["pipeline_match"] == "exact"
    assert set(ppocr.FIXED_POLICY_KEYS) | set(ppocr.FIDELITY_CAPS) | {"box_iou_min"} == set(ppocr.POLICY)


def components(candidate: dict, role: str) -> dict:
    return next(entry for entry in candidate["origin"]["components"] if entry["role"] == role)


NEAR_MISSES = [
    ("task", lambda c: c.update(task="transcription"), "task"),
    ("identifier", lambda c: c.update(id="ppocrv5-server/onnx-fp32-cpu"), "id"),
    ("server detector", lambda c: c["origin"].update(repo="PaddlePaddle/PP-OCRv5_server_det_onnx"), "origin.repo"),
    ("tag revision", lambda c: c["origin"].update(revision="main"), "origin.revision"),
    ("other graph", lambda c: c["origin"]["files"][0].update(sha256="f" * 64), "origin.files"),
    ("missing config", lambda c: c["origin"]["files"].pop(), "origin.files"),
    ("extra origin field", lambda c: c["origin"].update(base=c["origin"]["components"][1]), "origin"),
    ("component order", lambda c: c["origin"]["components"].reverse(), "origin.components"),
    ("missing component", lambda c: c["origin"]["components"].pop(), "origin.components"),
    (
        "server recognizer",
        lambda c: components(c, "rec").update(repo="PaddlePaddle/PP-OCRv5_server_rec_onnx"),
        "origin.components[rec].repo",
    ),
    (
        "recognizer revision",
        lambda c: components(c, "rec").update(revision="0" * 40),
        "origin.components[rec].revision",
    ),
    (
        "paddle weights",
        lambda c: components(c, "det-paddle")["files"][2].update(size=1),
        "origin.components[det-paddle].files",
    ),
    ("fp16 runtime", lambda c: c["runtime"].update(precision="fp16"), "runtime"),
    ("cuda", lambda c: c["runtime"].update(provider="CUDAExecutionProvider"), "runtime"),
    ("rgb", lambda c: c["behavior"].update(color_order="RGB"), "behavior.color_order"),
    ("limit side", lambda c: c["behavior"]["detector"]["resize"].update(limit_side_len=736), "behavior.detector"),
    ("limit type", lambda c: c["behavior"]["detector"]["resize"].update(limit_type="min"), "behavior.detector"),
    ("pillow resize", lambda c: c["behavior"]["detector"]["resize"].update(interpolation="pil"), "behavior.detector"),
    ("mean", lambda c: c["behavior"]["detector"]["normalize"].update(mean=[0.5, 0.5, 0.5]), "behavior.detector"),
    ("db threshold", lambda c: c["behavior"]["db"].update(thresh=0.2), "behavior.db"),
    ("unclip", lambda c: c["behavior"]["db"].update(unclip_ratio=2.0), "behavior.db"),
    ("polygons", lambda c: c["behavior"]["db"].update(box_type="poly"), "behavior.db"),
    ("dilation", lambda c: c["behavior"]["db"].update(use_dilation=True), "behavior.db"),
    ("slow score", lambda c: c["behavior"]["db"].update(score_mode="slow"), "behavior.db"),
    ("tall ratio", lambda c: c["behavior"]["crop"].update(tall_ratio=2.0), "behavior.crop"),
    ("rec height", lambda c: c["behavior"]["recognizer"].update(image_shape=[3, 32, 320]), "behavior.recognizer"),
    ("batching", lambda c: c["behavior"]["recognizer"].update(batch_size=6), "behavior.recognizer"),
    ("dictionary", lambda c: c["behavior"]["ctc"].update(dictionary_sha256="0" * 64), "behavior.ctc"),
    ("no space", lambda c: c["behavior"]["ctc"].update(use_space_char=False), "behavior.ctc"),
    ("beam search", lambda c: c["behavior"]["ctc"].update(decoding="beam"), "behavior.ctc"),
    (
        "textline orientation",
        lambda c: c["behavior"]["pipeline"].update(textline_orientation=True),
        "behavior.pipeline",
    ),
    ("extra behavior", lambda c: c["behavior"].update(language="en"), "behavior"),
    ("approved", lambda c: c["evaluation"].update(status="approved"), "evaluation.status"),
    ("bounded crop", lambda c: c["evaluation"]["policy"].update(crop_max_abs=0.01), "evaluation.policy.crop_max_abs"),
    (
        "loose same backend",
        lambda c: c["evaluation"]["policy"].update(same_backend_max_abs=1e-3),
        "evaluation.policy.same_backend_max_abs",
    ),
    ("fuzzy text", lambda c: c["evaluation"]["policy"].update(line_text="cer"), "evaluation.policy.line_text"),
    ("prob cap", lambda c: c["evaluation"]["policy"].update(prob_map_max_abs=0.02), "prob_map_max_abs"),
    ("iou floor", lambda c: c["evaluation"]["policy"].update(box_iou_min=0.5), "box_iou_min"),
    ("flip cap", lambda c: c["evaluation"]["policy"].update(box_flip_margin=0.2), "box_flip_margin"),
    ("missing bound", lambda c: c["evaluation"]["policy"].pop("box_score_max_abs"), "evaluation.policy"),
]


@pytest.mark.parametrize(("name", "mutate", "field"), NEAR_MISSES, ids=[case[0] for case in NEAR_MISSES])
def test_contract_refuses_near_misses_field_by_field(name, mutate, field):
    candidate = profile()
    mutate(candidate)
    with pytest.raises(PipelineError, match=f"field \\S*{re.escape(field)}|incomplete PP-OCR"):
        ppocr.validate_candidate_behavior(candidate)


def test_fidelity_bounds_may_be_proposed_within_caps():
    candidate = profile()
    candidate["evaluation"]["policy"].update(prob_map_max_abs=0.005, box_iou_min=1.0, box_flip_margin=0.05)
    ppocr.validate_candidate_behavior(candidate)


def test_other_routes_refuse_ocr():
    candidate = profile()
    with pytest.raises(PipelineError):
        onboarding_whisper.validate_candidate_behavior(candidate)
    with pytest.raises(PipelineError):
        onboarding_bert.validate_candidate_behavior(candidate)


def test_plugin_is_registered_and_recipe_plans(capsys):
    assert onboarding_plugins.PLUGINS[ppocr.PLUGIN] == "model_pipeline.onboarding_ppocr"
    assert main(["plan", str(RECIPE)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["plugin"] == ppocr.PLUGIN and plan["stages"] == ppocr.STAGES
    assert plan["reference_status"] == "proposed" and plan["qualification_status"] == "unapproved"
    assert set(plan["components"]) == {"rec", "det-paddle", "rec-paddle"}


# --- configurations -------------------------------------------------------------------------------------------------


def write_configs(directory: Path, candidate: dict, dictionary: list[str]) -> None:
    """The four inference.yml files as the pinned repositories ship them, with a small dictionary."""
    behavior = candidate["behavior"]
    detector, db = behavior["detector"], behavior["db"]
    det = {
        "Global": {"model_name": behavior["detector_model"]},
        "Hpi": {"backend_configs": {}},
        "PreProcess": {
            "transform_ops": [
                {"DecodeImage": {"channel_first": False, "img_mode": "BGR"}},
                {"DetLabelEncode": None},
                {"DetResizeForTest": {"resize_long": 960}},
                {"NormalizeImage": {**detector["normalize"]}},
                {"ToCHWImage": None},
                {"KeepKeys": {"keep_keys": ["image", "shape", "polys", "ignore_tags"]}},
            ]
        },
        "PostProcess": {
            "name": "DBPostProcess",
            "thresh": db["thresh"],
            "box_thresh": db["box_thresh"],
            "max_candidates": db["max_candidates"],
            "unclip_ratio": db["unclip_ratio"],
        },
    }
    rec = {
        "Global": {"model_name": behavior["recognizer_model"]},
        "Hpi": {"backend_configs": {}},
        "PreProcess": {
            "transform_ops": [
                {"DecodeImage": {"channel_first": False, "img_mode": "BGR"}},
                {"MultiLabelEncode": {"gtc_encode": "NRTRLabelEncode"}},
                {"RecResizeImg": {"image_shape": [3, 48, 320]}},
                {"KeepKeys": {"keep_keys": ["image", "label_ctc", "label_gtc", "length", "valid_ratio"]}},
            ]
        },
        "PostProcess": {"name": "CTCLabelDecode", "character_dict": dictionary},
    }
    for role, config in (("det", det), ("det-paddle", det), ("rec", rec), ("rec-paddle", rec)):
        (directory / role).mkdir(parents=True, exist_ok=True)
        (directory / role / "inference.yml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")


def small_dictionary_candidate(dictionary: list[str]) -> dict:
    candidate = profile()
    candidate["behavior"]["ctc"].update(
        dictionary_size=len(dictionary), dictionary_sha256=ppocr.dictionary_digest(dictionary)
    )
    return candidate


def test_config_contract_checks_every_repository(tmp_path):
    dictionary = ["a", "'", "文", "🇩🇪"]
    candidate = small_dictionary_candidate(dictionary)
    write_configs(tmp_path, candidate, dictionary)
    assert set(ppocr.check_configs(candidate, tmp_path)) == set(ppocr.ROLES)


@pytest.mark.parametrize(
    ("role", "change", "message"),
    [
        ("det", lambda text: text.replace("resize_long: 960", "resize_long: 736"), "PreProcess"),
        ("det-paddle", lambda text: text.replace("unclip_ratio: 1.5", "unclip_ratio: 2.0"), "PostProcess"),
        ("det", lambda text: text.replace("img_mode: BGR", "img_mode: RGB"), "PreProcess"),
        ("rec", lambda text: text.replace("- 48", "- 32"), "PreProcess"),
        ("rec-paddle", lambda text: text.replace("- a\n", "- b\n"), "character_dict.sha256"),
        ("rec", lambda text: text.replace("- a\n", "- a\n  - a\n"), "character_dict.size"),
        ("rec", lambda text: text.replace("- a\n", "- ' '\n"), "character_dict"),
        ("det", lambda text: text.replace("model_name: PP-OCRv5_mobile_det", "model_name: other"), "model_name"),
    ],
)
def test_config_contract_refuses_upstream_drift(tmp_path, role, change, message):
    dictionary = ["a", "'", "文", "🇩🇪"]
    candidate = small_dictionary_candidate(dictionary)
    write_configs(tmp_path, candidate, dictionary)
    path = tmp_path / role / "inference.yml"
    path.write_text(change(path.read_text(encoding="utf-8")), encoding="utf-8")
    with pytest.raises(PipelineError, match=message):
        ppocr.check_configs(candidate, tmp_path)


@pytest.mark.skipif(not (SNAPSHOT / "rec-paddle/inference.yml").is_file(), reason="pinned snapshot not present")
def test_pinned_configurations_match_the_contract():
    candidate = profile()
    from model_pipeline.onboarding_candidate import verify_files

    verify_files(candidate["origin"]["files"], SNAPSHOT / "det")
    ppocr.verify_components(candidate, SNAPSHOT)
    assert set(ppocr.check_configs(candidate, SNAPSHOT)) == set(ppocr.ROLES)


# --- corpus ---------------------------------------------------------------------------------------------------------


def test_shipped_corpus_is_pinned_and_varied():
    rows = ppocr.load_corpus(CORPUS)
    assert len(rows) >= 8
    assert any(not row["lines"] for row in rows), "a text-free image exercises the zero-box path"
    assert any(any(ord(char) > 0x2E80 for line in row["lines"] for char in line) for row in rows)
    assert any(row["height"] > 2 * row["width"] for row in rows), "a tall image exercises the rotation rule"


def corpus_copy(tmp_path: Path, mutate=None) -> Path:
    rows = fixtures.read_jsonl(CORPUS)
    (tmp_path / "ocr-v1").mkdir()
    for row in rows:
        (tmp_path / row["image"]).write_bytes((CORPUS.parent / row["image"]).read_bytes())
    if mutate:
        mutate(tmp_path, rows)
    fixtures.write_jsonl(tmp_path / "ocr.jsonl", rows)
    return tmp_path / "ocr.jsonl"


def _with_chunk(path: Path) -> None:
    import struct
    import zlib

    data = path.read_bytes()
    chunk = b"gAMA" + struct.pack(">I", 45455)
    extra = struct.pack(">I", 4) + chunk + struct.pack(">I", zlib.crc32(chunk))
    path.write_bytes(data[:33] + extra + data[33:])


@pytest.mark.parametrize(
    ("name", "mutate", "message"),
    [
        ("digest", lambda d, rows: rows[0].update(sha256="0" * 64), "pinned SHA-256"),
        ("duplicate", lambda d, rows: rows[1].update(id=rows[0]["id"]), "unique"),
        ("traversal", lambda d, rows: rows[0].update(image="../x.png"), "unsafe"),
        ("jpeg", lambda d, rows: rows[0].update(image="ocr-v1/x.jpg"), "non-PNG"),
        ("size", lambda d, rows: rows[0].update(width=rows[0]["width"] + 1), "indexed size"),
        ("lines", lambda d, rows: rows[0].update(lines=[""]), "ground-truth"),
        (
            "gamma chunk",
            lambda d, rows: (
                _with_chunk(d / rows[0]["image"]),
                rows[0].update(sha256=fixtures.sha256_file(d / rows[0]["image"])),
            ),
            "ancillary",
        ),
    ],
)
def test_corpus_refusals(tmp_path, name, mutate, message):
    with pytest.raises(PipelineError, match=message):
        ppocr.load_corpus(corpus_copy(tmp_path, mutate))


# --- metrics --------------------------------------------------------------------------------------------------------


def test_quad_iou_and_matching():
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    shifted = [[5, 0], [15, 0], [15, 10], [5, 10]]
    assert ppocr.quad_iou(square, square) == 1.0
    assert ppocr.quad_iou(square, shifted) == pytest.approx(50 / 150)
    assert ppocr.quad_iou(square, list(reversed(shifted))) == pytest.approx(50 / 150)
    assert ppocr.quad_iou(square, [[20, 20], [30, 20], [30, 30], [20, 30]]) == 0.0
    diamond = [[5, 0], [10, 5], [5, 10], [0, 5]]
    assert ppocr.quad_iou(square, diamond) == pytest.approx(0.5)
    matches = ppocr.match_boxes([{"box": shifted}, {"box": square}], [{"box": square}, {"box": shifted}])
    assert [(i, j) for i, j, _ in matches] == [(1, 0), (0, 1)]


def test_box_comparison_allows_only_near_threshold_flips():
    policy = profile()["evaluation"]["policy"]
    square = {"box": [[0, 0], [10, 0], [10, 10], [0, 10]], "box_score": 0.9}
    marginal = {"box": [[50, 0], [60, 0], [60, 10], [50, 10]], "box_score": 0.605}
    confident = {**marginal, "box_score": 0.8}
    assert ppocr.compare_boxes([square], [square, marginal], policy, 0.6, "x")["status"] == "passed"
    failed = ppocr.compare_boxes([square], [square, confident], policy, 0.6, "x")
    assert failed["status"] == "failed" and "dropped" in failed["violations"][0]
    moved = {**square, "box": [[2, 0], [12, 0], [12, 10], [2, 10]]}
    assert ppocr.compare_boxes([moved], [square], policy, 0.6, "x")["status"] == "failed"
    rescored = {**square, "box_score": 0.95}
    assert ppocr.compare_boxes([rescored], [square], policy, 0.6, "x")["status"] == "failed"


def test_edit_distance():
    assert ppocr.edit_distance("kitten", "sitting") == 3
    assert ppocr.edit_distance("", "abc") == 3
    assert ppocr.edit_distance("文字", "文字") == 0


# --- evaluator ------------------------------------------------------------------------------------------------------

IMAGES = {
    "text": [
        {"box": [[2, 2], [30, 2], [30, 10], [2, 10]], "box_score": 0.9, "text": "Hello", "score": 0.98},
        {"box": [[2, 14], [20, 14], [20, 22], [2, 22]], "box_score": 0.8, "text": "文字", "score": 0.95},
    ],
    "empty": [],
}
GROUND_TRUTH = {"text": ["Hello", "文字"], "empty": []}


def values(seed: int, count: int) -> list[float]:
    return [((seed * 7919 + index * 104729) % 2001) / 2000.0 for index in range(count)]


def write_stage(directory: Path, stage: str, candidate: dict, digest: str, reference_sha: str | None, edit=None):
    directory.mkdir(parents=True, exist_ok=True)
    origin = candidate["origin"]
    files = {"det": origin["files"], **{entry["role"]: entry["files"] for entry in origin["components"]}}
    if stage == "reference":
        wanted = [("det-paddle", n) for n in ("inference.pdiparams", "inference.json", "inference.yml")]
        wanted += [("rec-paddle", n) for n in ("inference.pdiparams", "inference.json", "inference.yml")]
    else:
        wanted = [(role, n) for role in ("det", "rec") for n in ("inference.onnx", "inference.yml")]
    pins = {f"{role}/{entry['path']}": entry["sha256"] for role, entries in files.items() for entry in entries}
    cases, layout, blob = [], [], array.array("f")

    def add(identifier: str, name: str, shape: list[int], seed: int, line: int | None = None) -> None:
        row = {"id": identifier, "name": name, "offset": len(blob), "shape": shape}
        if line is not None:
            row["line"] = line
        layout.append(row)
        count = shape[0] * shape[1] * shape[2] * shape[3]
        blob.extend(values(seed, count))

    for seed, (identifier, lines) in enumerate(IMAGES.items()):
        row = {
            "id": identifier,
            "image_sha256": hashlib.sha256(identifier.encode()).hexdigest(),
            "lines": copy.deepcopy(lines),
        }
        check = {"boxes": True, "texts": True, "scores_max_abs": 0.0}
        if stage in ("reference", "onnx"):
            row["pipeline"] = check
        if stage in ("onnx", "dotnet"):
            row["forced"] = [{"text": line["text"], "score": line["score"]} for line in lines]
        if stage == "dotnet":
            row["postprocess_lines"] = copy.deepcopy(lines)
        cases.append(row)
        for name in ppocr.STAGE_TENSORS[stage]:
            add(identifier, name, [1, 3 if name == "det_input" else 1, 4, 6], seed + len(name))
        for index in range(len(lines)):
            for name in ppocr.LINE_TENSORS[stage]:
                shape = [1, 3, 2, 5] if name == "rec_input" else [1, 2, 7, 1]
                add(identifier, name, shape, seed + index + len(name), index)
    document = {
        "format": ppocr.FORMAT,
        "format_version": 1,
        "profile": candidate["id"],
        "task": "ocr",
        "stage": stage,
        "candidate_sha256": digest,
        "assets": [{"path": key, "sha256": pins[key]} for key in (f"{role}/{name}" for role, name in wanted)],
        "environment": {"machine": "test-host", "provider": "CPUExecutionProvider", "onnxruntime": "1.30.0"},
        "cases": cases,
    }
    if stage == "reference":
        document["engine"] = "paddle"
        document["environment"].update(provider="paddle-cpu", paddle_onednn=False, **ppocr.REFERENCE_PACKAGES)
    elif stage == "onnx":
        document["engine"] = "onnxruntime"
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
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


def synthetic_corpus() -> list[dict]:
    return [
        {"id": identifier, "sha256": hashlib.sha256(identifier.encode()).hexdigest(), "lines": GROUND_TRUTH[identifier]}
        for identifier in IMAGES
    ]


def evidence(tmp_path: Path, edits: dict | None = None, corpus: list[dict] | None = None) -> dict:
    edits = edits or {}
    candidate = profile()
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


def test_complete_evidence_passes_and_never_adopts_protected_gates(tmp_path):
    report = ppocr.evaluate_development(**evidence(tmp_path))
    assert report["status"] == "passed", report["failed_checks"]
    assert report["protected_release_qualification"] == {"status": "unapproved", "G1": "not-run", "G2": "not-run"}
    assert report["policy_status"] == "proposed" and report["line_count"] == 2
    ids = {check["id"] for check in report["checks"]}
    for expected in (
        "dotnet-image/text/det_input_vs_reference",
        "onnx/text/prob_map_vs_reference",
        "dotnet-tensors/text/prob_map_vs_python",
        "dotnet-postprocess/text/boxes_vs_reference",
        "dotnet-image/text/boxes_vs_reference",
        "dotnet-crop/text/1/rec_input_vs_reference",
        "dotnet-tensors/text/1/rec_probs_vs_python",
        "dotnet-image/text/1/text_vs_reference",
        "reference/text/stepwise_vs_paddleocr_pipeline",
        "reference/corpus/cer_vs_ground_truth",
    ):
        assert expected in ids
    assert "# Development parity" in ppocr.render_report(report, "p")


def nudge(name: str, delta: float, identifier: str = "text", line: int | None = None, position: int = 0):
    def edit(document, blob):
        offset = _offset(document, blob, identifier, name, line)
        blob[offset + position] += delta

    return edit


def _offset(document, blob, identifier, name, line) -> int:
    # write_stage records the layout after edits run, so recompute it from IMAGES (same order as write_stage).
    position = 0
    for image, lines in IMAGES.items():
        for tensor in ppocr.STAGE_TENSORS[document["stage"]]:
            if (image, tensor, None) == (identifier, name, line):
                return position
            position += 1 * (3 if tensor == "det_input" else 1) * 4 * 6
        for index in range(len(lines)):
            for tensor in ppocr.LINE_TENSORS[document["stage"]]:
                if (image, tensor, index) == (identifier, name, line):
                    return position
                position += 30 if tensor == "rec_input" else 14
    raise KeyError(name)


def _set(path: tuple, value):
    def edit(document, blob):
        target = document
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return edit


def _swap_lines(key: str):
    def edit(document, blob):
        lines = document["cases"][0][key]
        lines[0], lines[1] = lines[1], lines[0]

    return edit


@pytest.mark.parametrize(
    ("stage", "edit", "failed"),
    [
        ("dotnet", nudge("det_input", 1e-6), "dotnet-image/text/det_input_vs_reference"),
        ("onnx", nudge("prob_map", 0.01), "onnx/text/prob_map_vs_reference"),
        ("dotnet", nudge("prob_map", 1e-4), "dotnet-tensors/text/prob_map_vs_python"),
        ("dotnet", nudge("rec_input", 1e-6, line=0), "dotnet-crop/text/0/rec_input_vs_reference"),
        ("onnx", nudge("rec_probs", 1e-3, line=1), "onnx/text/1/rec_probs_vs_reference"),
        ("dotnet", nudge("rec_probs", 1e-4, line=1), "dotnet-tensors/text/1/rec_probs_vs_python"),
        ("dotnet", _set(("cases", 0, "lines", 1, "text"), "文学"), "dotnet-image/text/1/text_vs_reference"),
        ("onnx", _set(("cases", 0, "lines", 0, "text"), "Hel1o"), "onnx/text/0/text_vs_reference"),
        ("dotnet", _set(("cases", 0, "forced", 0, "text"), "Hallo"), "dotnet-tensors/text/0/text_vs_python"),
        ("dotnet", _set(("cases", 0, "forced", 0, "score"), 0.97), "dotnet-tensors/text/0/confidence_vs_python"),
        (
            "dotnet",
            _set(("cases", 0, "postprocess_lines", 0, "box"), [[4, 2], [32, 2], [32, 10], [4, 10]]),
            "dotnet-postprocess/text/boxes_vs_reference",
        ),
        ("dotnet", _set(("cases", 0, "lines"), []), "dotnet-image/text/boxes_vs_reference"),
        ("reference", _set(("cases", 0, "pipeline", "texts"), False), "reference/text/stepwise_vs_paddleocr_pipeline"),
        ("onnx", _set(("cases", 1, "pipeline", "scores_max_abs"), 1e-9), "onnx/empty/stepwise_vs_paddleocr_pipeline"),
        ("reference", _set(("cases", 0, "lines", 1, "text"), "文本"), "reference/corpus/cer_vs_ground_truth"),
        ("reference", _set(("cases", 0, "lines", 1, "text"), "文本"), "reference/text/cer_vs_ground_truth"),
        ("dotnet", _swap_lines("lines"), "dotnet-image/text/boxes_vs_reference"),
        ("onnx", _swap_lines("lines"), "onnx/text/boxes_vs_reference"),
        ("onnx", _set(("cases", 0, "lines", 1, "score"), 0.951), "onnx/text/1/score_vs_reference"),
        ("dotnet", _set(("cases", 0, "lines", 0, "score"), 0.9801), "dotnet-image/text/0/score_vs_python"),
        ("onnx", _set(("cases", 0, "forced", 1, "score"), 0.951), "onnx/text/1/forced_score_vs_reference"),
    ],
)
def test_differences_produce_named_failed_checks(tmp_path, stage, edit, failed):
    report = ppocr.evaluate_development(**evidence(tmp_path, {stage: edit}))
    assert report["status"] == "failed"
    assert failed in report["failed_checks"]


@pytest.mark.parametrize(
    ("stage", "edit", "message"),
    [
        ("dotnet", _set(("checks", "pre_cancelled_request"), False), "cancellation"),
        ("onnx", _set(("reference_sha256",), "0" * 64), "different original reference"),
        ("onnx", _set(("environment", "machine"), "other-host"), "same machine"),
        ("onnx", _set(("environment", "onnxruntime"), "1.29.0"), "runtime/provider"),
        ("dotnet", _set(("stage",), "onnx"), "stage identity"),
        ("reference", _set(("assets", 0, "sha256"), "f" * 64), "asset identities"),
        ("dotnet", _set(("cases", 0, "image_sha256"), "0" * 64), "image differs"),
        ("dotnet", _set(("cases", 0, "forced"), []), "every reference line"),
        ("dotnet", _set(("cases", 0, "forced", 0), {"text": "Hello"}), "forced"),
        ("onnx", _set(("cases", 0, "forced", 0, "score"), "high"), "forced"),
        ("onnx", _set(("cases", 0, "forced", 1, "score"), float("inf")), "forced"),
        ("reference", _set(("engine",), "onnxruntime"), "engine identity"),
        ("reference", _set(("environment", "paddle_onednn"), True), "engine identity"),
        ("reference", _set(("environment", "paddlex"), "3.7.1"), "engine identity"),
        ("reference", _set(("environment", "provider"), "CPUExecutionProvider"), "engine identity"),
        ("onnx", _set(("engine",), "paddle"), "runtime/provider"),
        ("dotnet", _set(("environment", "provider"), "CUDAExecutionProvider"), "runtime/provider"),
        ("reference", _set(("cases", 0, "lines", 0, "box"), [[0, 0], [1, 1]]), "invalid line"),
        ("reference", _set(("cases", 0, "lines", 0, "score"), float("nan")), "invalid line"),
        ("reference", lambda document, blob: document["cases"][0].pop("pipeline"), "pipeline"),
        ("dotnet", lambda document, blob: document["cases"].pop(), "missing required case rows"),
        ("dotnet", lambda document, blob: blob.append(0.0), "unreferenced values"),
        ("onnx", lambda document, blob: blob.__setitem__(0, float("nan")), "non-finite"),
    ],
)
def test_incomplete_or_wrong_evidence_is_refused(tmp_path, stage, edit, message):
    with pytest.raises(PipelineError, match=message):
        ppocr.evaluate_development(**evidence(tmp_path, {stage: edit}))


def test_each_image_is_checked_against_its_own_ground_truth(tmp_path):
    # An image with drawn text must give reference lines; a text-free image must give none.
    corpus = synthetic_corpus()
    corpus[1]["lines"] = ["Hello"]
    (tmp_path / "b").mkdir()
    report = ppocr.evaluate_development(**evidence(tmp_path / "b", corpus=corpus))
    assert "reference/empty/cer_vs_ground_truth" in report["failed_checks"]
    corpus = synthetic_corpus()
    corpus[0]["lines"] = []
    (tmp_path / "c").mkdir()
    report = ppocr.evaluate_development(**evidence(tmp_path / "c", corpus=corpus))
    assert "reference/text/no_lines_without_text" in report["failed_checks"]


def test_every_required_path_must_produce_checks(tmp_path, monkeypatch):
    monkeypatch.setitem(ppocr.REQUIRED_PATH_PREFIXES, "dotnet-crop", "missing/")
    with pytest.raises(PipelineError, match="dotnet-crop"):
        ppocr.evaluate_development(**evidence(tmp_path))


def test_reference_packages_match_the_lock():
    lock = (REPO / "model-pipeline/reference/ppocr/requirements.txt").read_text(encoding="utf-8")
    names = {"paddle": "paddlepaddle", "paddleocr": "paddleocr", "paddlex": "paddlex"}
    for key, version in ppocr.REFERENCE_PACKAGES.items():
        assert f"\n{names[key]}=={version} " in lock


def test_reference_must_cover_exactly_the_corpus(tmp_path):
    corpus = synthetic_corpus()[:1]
    with pytest.raises(PipelineError, match="exactly the pinned corpus"):
        ppocr.evaluate_development(**evidence(tmp_path, corpus=corpus))


def test_tampered_tensor_sidecar_is_refused(tmp_path):
    paths = evidence(tmp_path)
    sidecar = tmp_path / "dotnet/dotnet-tensors.f32"
    sidecar.write_bytes(sidecar.read_bytes()[:-4] + b"\0\0\0\0")
    with pytest.raises(PipelineError, match="identity mismatch"):
        ppocr.evaluate_development(**paths)


# --- orchestration --------------------------------------------------------------------------------------------------


def test_build_refuses_distribution_and_an_incomplete_snapshot(tmp_path):
    recipe = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    with pytest.raises(PipelineError, match="distribution"):
        ppocr.build_stages(
            recipe, repo_root=REPO, snapshot=tmp_path, reference_python=tmp_path / "python", distribution=True
        )
    (tmp_path / "det").mkdir()
    with pytest.raises(PipelineError, match="missing rec/"):
        ppocr.build_stages(recipe, repo_root=REPO, snapshot=tmp_path, reference_python=tmp_path / "python")


def test_build_refuses_a_base_snapshot(tmp_path):
    recipe = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    with pytest.raises(PipelineError, match="no base snapshot"):
        onboarding_plugins.build_stages(
            recipe, repo_root=REPO, snapshot=tmp_path, reference_python=tmp_path / "python", base_snapshot=tmp_path
        )


@pytest.mark.skipif(
    not (REPO / "ai-dotnet/tools/Starlings.PaddleOcr.Fidelity.Tool").is_dir(), reason="the .NET fidelity tool is absent"
)
def test_stage_graph_declares_image_inputs_and_order(tmp_path, monkeypatch):
    from model_pipeline import onboarding_candidate

    monkeypatch.setattr(
        onboarding_candidate,
        "environment_identity",
        lambda python, root, timeout: {"python": {"status": "available"}, "dotnet": {"status": "available"}},
    )
    monkeypatch.setattr(onboarding_candidate, "_source", lambda snapshot, relative: snapshot / relative)
    for role in ppocr.ROLES:
        (tmp_path / role).mkdir()
    python = tmp_path / "python"
    python.write_text("")
    recipe = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    stages = ppocr.build_stages(recipe, repo_root=REPO, snapshot=tmp_path, reference_python=python)
    assert [stage.name for stage in stages] == ppocr.STAGES
    by_name = {stage.name: stage for stage in stages}
    assert by_name["onnx"].dependencies == ("reference", "assets", "environment")
    images = {key for key in by_name["reference"].inputs if key.startswith("corpus_image:")}
    assert len(images) == len(fixtures.read_jsonl(CORPUS))
    assert "model-pipeline/reference/ppocr/run_ppocr.py" in by_name["reference"].inputs
    assert "ai-dotnet/src/Starlings.Models.PaddleOcr/PaddleOcrModelAssets.cs" in by_name["dotnet"].inputs
