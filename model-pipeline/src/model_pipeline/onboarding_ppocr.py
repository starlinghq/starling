"""PP-OCRv5 text detection and recognition on ONNX Runtime CPU fp32: OCR parity against a fresh Paddle reference.

Explicit allow-list: the pinned official PaddlePaddle ONNX exports of the mobile detector and recognizer, the pinned
Paddle inference models they were exported from, and the preprocessing/post-processing contract (every constant read
from the pinned `inference.yml` files) are checked field by field, here and in Starlings.Models.PaddleOcr. Anything
else is refused. Stages:

    assets -> environment -> reference -> onnx -> dotnet -> evaluate -> review

`--snapshot` is one directory holding the four pinned repositories as subdirectories named by role (`det`, `rec`,
`det-paddle`, `rec-paddle`). The truth is PaddleOCR (paddlex) on the Paddle inference engine; PaddleOCR with
`engine="onnxruntime"` on the ONNX graphs is the same-backend Python run. Correctness is checked at four levels:
preprocessing tensors, graph outputs (detector probability map, recognizer probabilities), detection geometry (boxes
matched by IoU), and text (exact on matched boxes; character error rate against the drawn ground truth as a sanity
bound). Reference outputs and the policy are proposals; nothing here adopts protected fixtures or changes tolerances.
"""

from __future__ import annotations

import array
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import zlib
from dataclasses import replace
from pathlib import Path, PurePosixPath

from model_pipeline import fixtures
from model_pipeline.onboarding_candidate import guard_output, repository_input, write_json
from model_pipeline.onboarding_runner import PipelineError, Stage, run_command

PLUGIN = "ppocr-onnx-cpu-v1"
# Distribution is appended by onboarding_plugins from onboarding_distribution (and refused below).
SHARED_DISTRIBUTION = True
RUNNER = "model-pipeline/reference/ppocr/run_ppocr.py"
TOOL = "ai-dotnet/tools/Starlings.PaddleOcr.Fidelity.Tool/Starlings.PaddleOcr.Fidelity.Tool.csproj"
TOOL_ASSEMBLY = "Starlings.PaddleOcr.Fidelity.Tool.dll"
FORMAT = "starlings.ppocr-candidate-results"
STAGES = ["assets", "environment", "reference", "onnx", "dotnet", "evaluate", "review"]
RUNTIME = {"backend": "onnx", "version": "1.30.0", "provider": "CPUExecutionProvider", "precision": "fp32"}
GRAPH_FILE = "inference.onnx"
CONFIG_FILE = "inference.yml"
PADDLE_FILES = ("config.json", "inference.json", "inference.pdiparams", CONFIG_FILE)
# Snapshot subdirectory per role. "det" is the candidate's origin; the others are origin.components.
ROLES = ("det", "rec", "det-paddle", "rec-paddle")
COMPONENT_ROLES = ROLES[1:]

DETECTOR_MODEL = "PP-OCRv5_mobile_det"
RECOGNIZER_MODEL = "PP-OCRv5_mobile_rec"
DICTIONARY_SIZE = 18383
# SHA-256 of the recognizer dictionary (PostProcess.character_dict of the pinned inference.yml), entries joined by "\n".
DICTIONARY_SHA256 = "17665d27ed39f0deb82007859992d626d3105d0ee4578c120b7c72138dc04d05"
BEHAVIOR = {
    "detector_model": DETECTOR_MODEL,
    "recognizer_model": RECOGNIZER_MODEL,
    "graph_file": GRAPH_FILE,
    "config_file": CONFIG_FILE,
    # DecodeImage img_mode of both inference.yml files: every tensor is built from the BGR image.
    "color_order": "BGR",
    "detector": {
        "input": "x",
        "output": "fetch_name_0",
        # paddlex maps the yml's DetResizeForTest {resize_long: 960} of the PP-OCRv5 detectors to limit_side_len 960
        # with limit_type "max" (text_detection/predictor.py, _get_text_det_resize_defaults), and max_side_limit 4000.
        "resize": {
            "limit_side_len": 960,
            "limit_type": "max",
            "max_side_limit": 4000,
            "multiple": 32,
            "rounding": "half-even",
            "interpolation": "opencv-inter-linear-u8",
        },
        "normalize": {"scale": "1./255.", "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "order": "hwc"},
    },
    "db": {
        "thresh": 0.3,
        "box_thresh": 0.6,
        "max_candidates": 1000,
        "unclip_ratio": 1.5,
        "min_size": 3,
        "score_mode": "fast",
        "box_type": "quad",
        "use_dilation": False,
    },
    "crop": {
        "method": "min-area-rect-perspective",
        "interpolation": "opencv-inter-cubic-u8",
        "border": "replicate",
        "tall_ratio": 1.5,
    },
    "recognizer": {
        "input": "x",
        "output": "fetch_name_0",
        "image_shape": [3, 48, 320],
        "max_width": 3200,
        "batch_size": 1,
    },
    "ctc": {
        "blank": 0,
        "dictionary_size": DICTIONARY_SIZE,
        "dictionary_sha256": DICTIONARY_SHA256,
        "use_space_char": True,
        "classes": DICTIONARY_SIZE + 2,
        "decoding": "greedy",
        "confidence": "mean-max-probability",
    },
    "pipeline": {
        "sort_row_tolerance": 10,
        "document_preprocessing": False,
        "textline_orientation": False,
        "rec_score_thresh": 0.0,
    },
}
CONTRACTS = {
    "PaddlePaddle/PP-OCRv5_mobile_det_onnx": {
        "id": "ppocrv5-mobile/onnx-fp32-cpu",
        "revision": "e6f4fa85f00e168c862bc462aebca69eef9b3d3d",
        # The same pins as PaddleOcrModelAssets.cs: the .NET library refuses any other graph or configuration.
        "pins": {
            GRAPH_FILE: (4826518, "a431985659dc921974177a95adcfbb90fd9e51989a5e04d70d0b75f597b6e61d"),
            CONFIG_FILE: (903, "98069072e1b6b37d727fd9d9f11725faa46d6ea0de012f2ed26caea011c37699"),
        },
        "components": {
            "rec": (
                "PaddlePaddle/PP-OCRv5_mobile_rec_onnx",
                "ed152b8b495f84de93cda5709d768548a9127622",
                {
                    GRAPH_FILE: (16534782, "da72dc72ca4dc220df0dfde68c1dedc31c58d3e76a25871122e5056227d50092"),
                    CONFIG_FILE: (148345, "5dfeb2777f6d0db8177d8128a8acfcf6e6276dc4ac73ea3bf0dc06d6a5e85d8e"),
                },
            ),
            "det-paddle": (
                "PaddlePaddle/PP-OCRv5_mobile_det",
                "0d63e78e2b680928f6b1747d76a08db6e645efb7",
                {
                    "config.json": (2871, "7ac1c33f377ba58561f4b89d3180b6add2b7c5c60a4edac90fa4e0ceccdc6665"),
                    "inference.json": (229777, "05feef1acb00aa4cd7362b15f7f501fc4f99d7b1fa73c1c871e0c7b1504b0f5c"),
                    "inference.pdiparams": (
                        4692937,
                        "afa1820cb16c1fd0dad589d0f8b389139061c1ef6d68019685fd07be997dda5b",
                    ),
                    CONFIG_FILE: (903, "98069072e1b6b37d727fd9d9f11725faa46d6ea0de012f2ed26caea011c37699"),
                },
            ),
            "rec-paddle": (
                "PaddlePaddle/PP-OCRv5_mobile_rec",
                "682f20538d8c086cb2128e5cfac775e6c4904e85",
                {
                    "config.json": (352253, "6df1587ae056f0b987125fc894b17f62ae3a12644fa2488d108f19377da9c1a8"),
                    "inference.json": (217724, "24587345250c7332d0fc6f9a44e794d078cdaeb64c302fef906f325619de2569"),
                    "inference.pdiparams": (
                        16458665,
                        "2460da90875937c94db97eba74ae3d9e5d4c4c57c42f1f41531c09a26bcc771a",
                    ),
                    CONFIG_FILE: (148345, "5dfeb2777f6d0db8177d8128a8acfcf6e6276dc4ac73ea3bf0dc06d6a5e85d8e"),
                },
            ),
        },
        "behavior": BEHAVIOR,
    },
}
POLICY = {
    "preprocess_max_abs": 0.0,
    "crop_max_abs": 0.0,
    "same_backend_max_abs": 1e-5,
    "prob_map_max_abs": 1e-3,
    "rec_probs_max_abs": 2e-4,
    "rec_score_max_abs": 2e-4,
    "box_iou_min": 0.9,
    "box_score_max_abs": 1e-3,
    "box_flip_margin": 0.01,
    "line_text": "exact",
    "pipeline_match": "exact",
    "ground_truth_cer_max": 0.05,
    "non_finite": "fail",
    "required_paths": [
        "reference",
        "python-onnx",
        "dotnet-tensors",
        "dotnet-postprocess",
        "dotnet-crop",
        "dotnet-image",
    ],
}
# Check-id prefix of each required evidence path; every one must produce checks (`required_paths`).
REQUIRED_PATH_PREFIXES = {
    "reference": "reference/",
    "python-onnx": "onnx/",
    "dotnet-tensors": "dotnet-tensors/",
    "dotnet-postprocess": "dotnet-postprocess/",
    "dotnet-crop": "dotnet-crop/",
    "dotnet-image": "dotnet-image/",
}
# The reference engine identity the evaluator requires (the pins of reference/ppocr/requirements.txt).
REFERENCE_PACKAGES = {"paddle": "3.3.1", "paddleocr": "3.7.0", "paddlex": "3.7.2"}
# Correctness bounds are fixed: preprocessing and cropping restate OpenCV's fixed-point uint8 kernels (exact), .NET
# against Python ONNX Runtime runs the same graphs, and line text must be identical. Only the bounds that cover the
# Paddle-to-ONNX export (graph outputs and line confidences against the Paddle inference engine) and the owned DB
# geometry (IoU, box score, the threshold margin within which a box may appear on one side only) may be proposed per
# model, up to these caps. line_text, pipeline_match, non_finite and required_paths are enforced by the evaluator.
FIXED_POLICY_KEYS = {
    key: POLICY[key]
    for key in (
        "preprocess_max_abs",
        "crop_max_abs",
        "same_backend_max_abs",
        "line_text",
        "pipeline_match",
        "ground_truth_cer_max",
        "non_finite",
        "required_paths",
    )
}
FIDELITY_CAPS = {
    "prob_map_max_abs": 1e-2,
    "rec_probs_max_abs": 1e-2,
    "rec_score_max_abs": 1e-2,
    "box_score_max_abs": 1e-2,
    "box_flip_margin": 0.05,
}
# box_iou_min is a lower bound: it may be proposed from this floor up to 1.
IOU_FLOOR = 0.8
STAGE_TENSORS = {
    "reference": ("det_input", "prob_map"),
    "onnx": ("prob_map",),
    "dotnet": ("det_input", "prob_map"),
}
LINE_TENSORS = {
    "reference": ("rec_input", "rec_probs"),
    "onnx": ("rec_probs",),
    "dotnet": ("rec_input", "rec_probs"),
}


def _refuse(field: str, detail: str) -> None:
    raise PipelineError(f"unsupported PP-OCR contract field {field}: {detail}")


def _same(field: str, actual, expected) -> None:
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        _refuse(field, f"expected {expected!r}, found {actual!r}")


def _pinned_files(field: str, records, pins: dict) -> None:
    paths = [record.get("path") for record in records] if isinstance(records, list) else None
    if paths is None or sorted(paths) != sorted(pins) or len(set(paths)) != len(paths):
        _refuse(field, f"pinned files must be exactly {sorted(pins)}")
    for record in records:
        path = PurePosixPath(record["path"])
        if path.is_absolute() or len(path.parts) != 1:
            _refuse(field, f"unsafe path {record['path']!r}")
        _same(f"{field}[{record['path']}]", [record["size"], record["sha256"]], list(pins[record["path"]]))


def _validate_policy(policy: dict) -> None:
    if not isinstance(policy, dict) or set(policy) != set(POLICY):
        _refuse("evaluation.policy", f"fields must be exactly {sorted(POLICY)}")
    for key, value in FIXED_POLICY_KEYS.items():
        _same(f"evaluation.policy.{key}", policy[key], value)
    for key, cap in FIDELITY_CAPS.items():
        value = policy[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= cap:
            _refuse(f"evaluation.policy.{key}", f"must be a finite number above 0 and up to {cap}")
    value = policy["box_iou_min"]
    if type(value) not in (int, float) or not math.isfinite(value) or not IOU_FLOOR <= value <= 1:
        _refuse("evaluation.policy.box_iou_min", f"must be a finite number from {IOU_FLOOR} to 1")


def validate_candidate_behavior(candidate: dict) -> dict:
    """Refuse every contract this route does not implement; returns the matching allow-list entry."""
    try:
        origin = candidate["origin"]
        repo = origin["repo"]
        if repo not in CONTRACTS:
            _refuse("origin.repo", f"{repo!r} is not an allow-listed PP-OCR detector repository")
        contract = CONTRACTS[repo]
        _same("format_version", candidate["format_version"], 1)
        _same("id", candidate["id"], contract["id"])
        _same("task", candidate["task"], "ocr")
        _same("origin.revision", origin["revision"], contract["revision"])
        _pinned_files("origin.files", origin["files"], contract["pins"])
        extra = set(origin) - {"repo", "revision", "files", "components"}
        if extra:
            _refuse("origin", f"unknown fields {sorted(extra)}")
        components = origin["components"]
        if not isinstance(components, list) or [entry.get("role") for entry in components] != list(COMPONENT_ROLES):
            _refuse("origin.components", f"roles must be exactly {list(COMPONENT_ROLES)}, in that order")
        for entry in components:
            role = entry["role"]
            expected_repo, expected_revision, pins = contract["components"][role]
            if set(entry) != {"role", "repo", "revision", "files"}:
                _refuse(f"origin.components[{role}]", "fields must be exactly role, repo, revision and files")
            _same(f"origin.components[{role}].repo", entry["repo"], expected_repo)
            _same(f"origin.components[{role}].revision", entry["revision"], expected_revision)
            _pinned_files(f"origin.components[{role}].files", entry["files"], pins)
        _same("runtime", candidate["runtime"], RUNTIME)
        behavior, expected = candidate["behavior"], contract["behavior"]
        if set(behavior) != set(expected):
            _refuse("behavior", f"fields must be exactly {sorted(expected)}")
        for key, value in expected.items():
            _same(f"behavior.{key}", behavior[key], value)
        _same("evaluation.status", candidate["evaluation"]["status"], "proposed")
        _validate_policy(candidate["evaluation"]["policy"])
        return contract
    except (KeyError, TypeError, AttributeError) as exc:
        raise PipelineError(f"unsupported or incomplete PP-OCR contract: {exc!r}") from exc


def dictionary_digest(characters: list[str]) -> str:
    return hashlib.sha256("\n".join(characters).encode("utf-8")).hexdigest()


def _load_yaml(path: Path) -> dict:
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PipelineError(f"{path}: unreadable pinned configuration: {exc}") from exc
    if not isinstance(data, dict):
        raise PipelineError(f"{path}: configuration must be a mapping")
    return data


def check_configs(candidate: dict, snapshot: Path) -> dict:
    """Every preprocessing and post-processing constant of the pinned inference.yml files equals the contract.

    No upstream code runs. The ONNX and Paddle repositories carry byte-identical configurations; each is checked.
    """
    behavior = candidate["behavior"]
    detector, db = behavior["detector"], behavior["db"]
    recognizer, ctc = behavior["recognizer"], behavior["ctc"]
    expected_detector_ops = [
        {"DecodeImage": {"channel_first": False, "img_mode": behavior["color_order"]}},
        {"DetLabelEncode": None},
        {"DetResizeForTest": {"resize_long": detector["resize"]["limit_side_len"]}},
        {
            "NormalizeImage": {
                "mean": detector["normalize"]["mean"],
                "order": detector["normalize"]["order"],
                "scale": detector["normalize"]["scale"],
                "std": detector["normalize"]["std"],
            }
        },
        {"ToCHWImage": None},
        {"KeepKeys": {"keep_keys": ["image", "shape", "polys", "ignore_tags"]}},
    ]
    expected_detector_post = {
        "name": "DBPostProcess",
        "thresh": db["thresh"],
        "box_thresh": db["box_thresh"],
        "max_candidates": db["max_candidates"],
        "unclip_ratio": db["unclip_ratio"],
    }
    expected_recognizer_ops = [
        {"DecodeImage": {"channel_first": False, "img_mode": behavior["color_order"]}},
        {"MultiLabelEncode": {"gtc_encode": "NRTRLabelEncode"}},
        {"RecResizeImg": {"image_shape": recognizer["image_shape"]}},
        {"KeepKeys": {"keep_keys": ["image", "label_ctc", "label_gtc", "length", "valid_ratio"]}},
    ]
    checked = {}
    for role, model, kind in (
        ("det", behavior["detector_model"], "detector"),
        ("det-paddle", behavior["detector_model"], "detector"),
        ("rec", behavior["recognizer_model"], "recognizer"),
        ("rec-paddle", behavior["recognizer_model"], "recognizer"),
    ):
        config = _load_yaml(snapshot / role / behavior["config_file"])
        facts = {
            "keys": (sorted(config), ["Global", "Hpi", "PostProcess", "PreProcess"]),
            "Global.model_name": (config.get("Global", {}).get("model_name"), model),
            "PreProcess": (
                config.get("PreProcess"),
                {"transform_ops": expected_detector_ops if kind == "detector" else expected_recognizer_ops},
            ),
        }
        post = config.get("PostProcess", {})
        if kind == "detector":
            facts["PostProcess"] = (post, expected_detector_post)
        else:
            characters = post.get("character_dict")
            if not isinstance(characters, list) or not all(
                isinstance(item, str) and item and "\n" not in item for item in characters
            ):
                raise PipelineError(f"{role}: character_dict must be a list of nonempty single-line strings")
            facts["PostProcess.keys"] = (sorted(post), ["character_dict", "name"])
            facts["PostProcess.name"] = (post.get("name"), "CTCLabelDecode")
            facts["character_dict.size"] = (len(characters), ctc["dictionary_size"])
            facts["character_dict.unique"] = (len(set(characters)), ctc["dictionary_size"])
            facts["character_dict.sha256"] = (dictionary_digest(characters), ctc["dictionary_sha256"])
            # paddlex CTCLabelDecode: blank first, the dictionary, then the space character (use_space_char=True).
            facts["character_dict.space_absent"] = (" " in characters, False)
        for field, (actual, expected) in facts.items():
            if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
                raise PipelineError(f"{role} {behavior['config_file']} {field} is {actual!r}, contract {expected!r}")
        checked[role] = sorted(facts)
    return checked


def verify_components(candidate: dict, snapshot: Path) -> list[dict]:
    """Pinned size and SHA-256 of every component file, each read only inside its role's directory."""
    from model_pipeline.onboarding_candidate import verify_files

    records = []
    for entry in candidate["origin"]["components"]:
        role = entry["role"]
        for record in verify_files(entry["files"], snapshot / role, role):
            records.append({"role": role, **record})
    return records


# --- corpus -----------------------------------------------------------------------------------------------------------


def png_header(data: bytes) -> tuple[int, int, int, int, list[str]]:
    """(width, height, bit depth, color type, chunk types) of a PNG, refusing anything but a well-formed file."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise PipelineError("not a PNG file")
    position, chunks, header = 8, [], None
    while position < len(data):
        if position + 12 > len(data):
            raise PipelineError("truncated PNG chunk")
        (length,) = struct.unpack(">I", data[position : position + 4])
        kind = data[position + 4 : position + 8]
        body = data[position + 8 : position + 8 + length]
        (crc,) = struct.unpack(">I", data[position + 8 + length : position + 12 + length])
        if len(body) != length or zlib.crc32(kind + body) != crc:
            raise PipelineError("corrupt PNG chunk")
        chunks.append(kind.decode("latin-1"))
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        position += 12 + length
    if header is None or chunks[0] != "IHDR" or chunks[-1] != "IEND":
        raise PipelineError("PNG lacks IHDR or IEND")
    return header[0], header[1], header[2], header[3], chunks


def load_corpus(corpus: Path) -> list[dict]:
    """The OCR corpus index and its images, each verified by SHA-256 and PNG header against the index."""
    rows, seen = [], set()
    for row in fixtures.read_jsonl(corpus):
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise PipelineError("OCR corpus rows require unique nonempty ids")
        seen.add(identifier)
        relative = PurePosixPath(row.get("image", ""))
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".png" or not relative.parts:
            raise PipelineError(f"{identifier}: unsafe or non-PNG image path {row.get('image')!r}")
        path = corpus.parent / relative
        if path.is_symlink() or not path.is_file() or fixtures.sha256_file(path) != row.get("sha256"):
            raise PipelineError(f"{identifier}: image file is missing or differs from its pinned SHA-256")
        lines = row.get("lines")
        if not isinstance(lines, list) or not all(isinstance(line, str) and line for line in lines):
            raise PipelineError(f"{identifier}: needs its ground-truth lines")
        try:
            width, height, depth, color, chunks = png_header(path.read_bytes())
        except PipelineError as exc:
            raise PipelineError(f"{identifier}: {exc}") from exc
        # 8-bit truecolor only, and no chunk that tells a decoder to transform the stored samples (gamma, ICC, sRGB,
        # chromaticity, transparency): every decoder then reads the same pixels.
        if (width, height, depth, color) != (row.get("width"), row.get("height"), 8, 2) or set(chunks) - {
            "IHDR",
            "IDAT",
            "IEND",
        }:
            raise PipelineError(f"{identifier}: PNG must be the indexed size, 8-bit RGB, without ancillary chunks")
        rows.append({**row, "path": path})
    if len(rows) < 2 or not any(row["lines"] for row in rows):
        raise PipelineError("OCR corpus must hold at least two images, some with text")
    return rows


# --- metrics ----------------------------------------------------------------------------------------------------------


def edit_distance(hypothesis, reference) -> int:
    previous = list(range(len(reference) + 1))
    for row, item in enumerate(hypothesis, 1):
        current = [row]
        for column, expected in enumerate(reference, 1):
            current.append(
                min(previous[column] + 1, current[column - 1] + 1, previous[column - 1] + (item != expected))
            )
        previous = current
    return previous[-1]


def _area(polygon: list[tuple[float, float]]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(polygon, polygon[1:] + polygon[:1], strict=True):
        total += x0 * y1 - x1 * y0
    return total / 2


def _clip(subject: list[tuple[float, float]], clipper: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sutherland-Hodgman: the part of `subject` inside the convex, counterclockwise `clipper`."""

    def inside(point, a, b) -> bool:
        return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) >= 0

    def cross(p, q, a, b):
        x1, y1, x2, y2 = p[0], p[1], q[0], q[1]
        x3, y3, x4, y4 = a[0], a[1], b[0], b[1]
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    output = subject
    for a, b in zip(clipper, clipper[1:] + clipper[:1], strict=True):
        points, output = output, []
        for p, q in zip(points, points[1:] + points[:1], strict=True) if points else ():
            if inside(q, a, b):
                if not inside(p, a, b):
                    output.append(cross(p, q, a, b))
                output.append(q)
            elif inside(p, a, b):
                output.append(cross(p, q, a, b))
    return output


def quad_iou(first, second) -> float:
    """Intersection over union of two convex quadrilaterals given as four [x, y] points."""
    polygons = []
    for quad in (first, second):
        polygon = [(float(x), float(y)) for x, y in quad]
        if _area(polygon) < 0:
            polygon.reverse()
        polygons.append(polygon)
    areas = [abs(_area(polygon)) for polygon in polygons]
    if min(areas) == 0:
        return 1.0 if [list(map(float, p)) for p in first] == [list(map(float, p)) for p in second] else 0.0
    intersection = abs(_area(_clip(polygons[0], polygons[1])))
    return intersection / (areas[0] + areas[1] - intersection)


def match_boxes(actual: list[dict], expected: list[dict]) -> list[tuple[int, int, float]]:
    """One-to-one greedy matching by descending IoU (ties by index): (actual index, expected index, IoU)."""
    pairs = sorted(
        (
            (-quad_iou(left["box"], right["box"]), i, j)
            for i, left in enumerate(actual)
            for j, right in enumerate(expected)
        ),
    )
    used_actual, used_expected, matches = set(), set(), []
    for negative, i, j in pairs:
        if -negative <= 0 or i in used_actual or j in used_expected:
            continue
        used_actual.add(i)
        used_expected.add(j)
        matches.append((i, j, -negative))
    return sorted(matches, key=lambda match: match[1])


# --- evidence ------------------------------------------------------------------------------------------------------


def _line(row, stage: str, where: str) -> dict:
    box = row["box"]
    if (
        not isinstance(box, list)
        or len(box) != 4
        or any(not isinstance(point, list) or len(point) != 2 for point in box)
        or any(type(value) is not int for point in box for value in point)
        or not isinstance(row["text"], str)
        or type(row["score"]) not in (int, float)
        or not math.isfinite(row["score"])
        or type(row["box_score"]) not in (int, float)
        or not math.isfinite(row["box_score"])
    ):
        raise PipelineError(f"{stage}/{where}: invalid line (box, box score, text or score)")
    return row


def _lines(rows, stage: str, where: str) -> list[dict]:
    if not isinstance(rows, list):
        raise PipelineError(f"{stage}/{where}: lines must be a list")
    return [_line(row, stage, f"{where}[{index}]") for index, row in enumerate(rows)]


def _forced(rows, stage: str, identifier: str) -> list[dict]:
    """Forced recognition rows: one {text, score} per reference line, text a string, score a finite number."""
    if not isinstance(rows, list):
        raise PipelineError(f"{stage}/{identifier}: forced recognition rows missing")
    for index, row in enumerate(rows):
        if (
            not isinstance(row, dict)
            or set(row) != {"text", "score"}
            or not isinstance(row["text"], str)
            or type(row["score"]) not in (int, float)
            or not math.isfinite(row["score"])
        ):
            raise PipelineError(f"{stage}/{identifier}/forced[{index}]: needs a text string and a finite score")
    return rows


def _document(path: Path, candidate: dict, digest: str, stage: str) -> tuple[dict, dict[str, dict]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"missing or malformed required {stage} evidence: {path}: {exc}") from exc
    if not isinstance(data, dict) or type(data.get("format_version")) is not int:
        raise PipelineError(f"{stage}: evidence must be an object with an integer format version")
    if (
        data.get("format"),
        data.get("format_version"),
        data.get("profile"),
        data.get("task"),
        data.get("candidate_sha256"),
        data.get("stage"),
    ) != (FORMAT, 1, candidate["id"], candidate["task"], digest, stage):
        raise PipelineError(f"{stage}: wrong format, profile, task, contract digest, or stage identity")
    roles = ("det-paddle", "rec-paddle") if stage == "reference" else ("det", "rec")
    origin = candidate["origin"]
    files = {"det": origin["files"], **{entry["role"]: entry["files"] for entry in origin["components"]}}
    names = (
        ("inference.pdiparams", "inference.json", CONFIG_FILE) if stage == "reference" else (GRAPH_FILE, CONFIG_FILE)
    )
    expected_assets = {
        f"{role}/{entry['path']}": entry["sha256"] for role in roles for entry in files[role] if entry["path"] in names
    }
    try:
        assets = data["assets"]
        if len(assets) != len(expected_assets) or {e["path"]: e["sha256"] for e in assets} != expected_assets:
            raise PipelineError(f"{stage}: asset identities differ from the candidate")
        environment = data["environment"]
        if stage == "reference":
            # The Paddle inference engine on the CPU with oneDNN off, at the pinned paddle/paddleocr/paddlex versions.
            if (
                data.get("engine") != "paddle"
                or environment["provider"] != "paddle-cpu"
                or environment["paddle_onednn"] is not False
                or {key: environment[key] for key in REFERENCE_PACKAGES} != REFERENCE_PACKAGES
            ):
                raise PipelineError(f"{stage}: engine identity differs from the Paddle reference contract")
        elif (
            (stage == "onnx" and data.get("engine") != "onnxruntime")
            or environment["provider"] != "CPUExecutionProvider"
            or environment["onnxruntime"] != candidate["runtime"]["version"]
        ):
            raise PipelineError(f"{stage}: runtime/provider identity differs from the candidate")
        records = data["cases"]
        if not isinstance(records, list) or len(records) < 2:
            raise PipelineError(f"{stage}: missing required case rows")
        cases = {}
        for row in records:
            identifier = row["id"]
            if not isinstance(identifier, str) or not identifier or identifier in cases:
                raise PipelineError(f"{stage}: duplicate or invalid case id {identifier!r}")
            if not isinstance(row["image_sha256"], str):
                raise PipelineError(f"{stage}/{identifier}: image digest missing")
            row["lines"] = _lines(row["lines"], stage, f"{identifier}/lines")
            if stage == "reference":
                pipeline = row["pipeline"]
                if not isinstance(pipeline, dict) or set(pipeline) != {"boxes", "texts", "scores_max_abs"}:
                    raise PipelineError(f"{stage}/{identifier}: the PaddleOCR pipeline cross-check is missing")
            if stage == "onnx":
                row["forced"] = _forced(row["forced"], stage, identifier)
                pipeline = row["pipeline"]
                if not isinstance(pipeline, dict) or set(pipeline) != {"boxes", "texts", "scores_max_abs"}:
                    raise PipelineError(f"{stage}/{identifier}: the PaddleOCR pipeline cross-check is missing")
            if stage == "dotnet":
                row["postprocess_lines"] = _lines(row["postprocess_lines"], stage, f"{identifier}/postprocess")
                row["forced"] = _forced(row["forced"], stage, identifier)
            cases[identifier] = row
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError(f"{stage}: incomplete or malformed required evidence: {exc}") from exc
    return data, cases


def _tensors(
    path: Path, document: dict, cases: dict[str, dict], stage: str, line_counts: dict[str, int], non_finite: str
) -> dict:
    """The stage's float32 sidecar split into named tensors (shape checked), with no missing or unreferenced value."""
    try:
        specification = document["tensors"]
        relative = Path(specification["file"])
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name in {".", ".."}:
            raise PipelineError("unsafe tensor sidecar path")
        tensor_path = path.parent / relative
        if tensor_path.is_symlink() or fixtures.sha256_file(tensor_path) != specification["sha256"]:
            raise PipelineError("tensor sidecar identity mismatch")
        blob = tensor_path.read_bytes()
        if len(blob) % 4:
            raise PipelineError("tensor sidecar has an incomplete float32 value")
        values = array.array("f")
        values.frombytes(blob)
        if sys.byteorder != "little":
            values.byteswap()
        # policy non_finite "fail" (the only value the route accepts): any NaN or infinity refuses the evidence.
        if non_finite == "fail" and any(not math.isfinite(value) for value in values):
            raise PipelineError("tensor sidecar contains non-finite values")
        expected_keys = {(identifier, name, None) for identifier in cases for name in STAGE_TENSORS[stage]}
        expected_keys |= {
            (identifier, name, index)
            for identifier, count in line_counts.items()
            for name in LINE_TENSORS[stage]
            for index in range(count)
        }
        output, position = {}, 0
        for row in specification["cases"]:
            key = (row["id"], row["name"], row.get("line"))
            if key not in expected_keys or key in output:
                raise PipelineError(f"unknown or duplicate tensor {key}")
            shape = row["shape"]
            if (
                not isinstance(shape, list)
                or len(shape) not in (3, 4)
                or any(type(value) is not int or value < 1 for value in shape)
                or row["offset"] != position
            ):
                raise PipelineError(f"tensor {key} has shape {shape} at {row['offset']}")
            end = position + math.prod(shape)
            if end > len(values):
                raise PipelineError("tensor sidecar is truncated")
            output[key] = (shape, values[position:end])
            position = end
        if set(output) != expected_keys:
            raise PipelineError("tensor sidecar is missing required tensors")
        if position != len(values):
            raise PipelineError("tensor sidecar has unreferenced values")
        return output
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise PipelineError(f"{stage}: missing or malformed tensor evidence: {exc}") from exc


def _difference(actual, expected, bound: float, where: str) -> dict:
    (actual_shape, actual_values), (expected_shape, expected_values) = actual, expected
    if actual_shape != expected_shape:
        return {"id": where, "status": "failed", "bound": bound, "detail": f"shape {actual_shape} != {expected_shape}"}
    maximum, total, unequal = 0.0, 0.0, 0
    for left, right in zip(actual_values, expected_values, strict=True):
        difference = abs(left - right)
        total += difference
        unequal += difference != 0
        maximum = max(maximum, difference)
    return {
        "id": where,
        "status": "passed" if maximum <= bound else "failed",
        "bound": bound,
        "max_abs": maximum,
        "mean_abs": total / len(expected_values) if len(expected_values) else 0.0,
        "unequal": unequal,
    }


def _exact(actual, expected, where: str) -> dict:
    return {
        "id": where,
        "status": "passed" if actual == expected else "failed",
        "bound": "exact",
        "actual": actual,
        "expected": expected,
    }


def _rate(value: float, bound: float, where: str, **details) -> dict:
    return {"id": where, "status": "passed" if value <= bound else "failed", "bound": bound, "value": value, **details}


def compare_boxes(actual: list[dict], expected: list[dict], policy: dict, box_thresh: float, where: str) -> dict:
    """Set-of-boxes comparison: every box matched one-to-one by IoU, except a box whose own detection score lies within
    `box_flip_margin` of the box threshold (a near-threshold flip either way). Matched boxes must reach `box_iou_min`
    and their box scores must agree within `box_score_max_abs`."""
    matches = match_boxes(actual, expected)
    violations = []
    iou = min((match[2] for match in matches), default=1.0)
    score = max((abs(actual[i]["box_score"] - expected[j]["box_score"]) for i, j, _ in matches), default=0.0)
    for _, j, value in matches:
        if value < policy["box_iou_min"]:
            violations.append(f"box {j} IoU {value:.4f} < {policy['box_iou_min']}")
    if score > policy["box_score_max_abs"]:
        violations.append(f"box score difference {score:.3g} > {policy['box_score_max_abs']}")
    # Reading order: with no one-sided box the matched actual indices must be exactly 0..n-1 in reference order;
    # with flips they must still increase with the reference index.
    order = [i for i, _, _ in matches]
    if order != sorted(order):
        violations.append(f"line order differs from the reference: {order}")
    flips = []
    for side, rows, used in (
        ("dropped", expected, {j for _, j, _ in matches}),
        ("added", actual, {i for i, _, _ in matches}),
    ):
        for index, row in enumerate(rows):
            if index in used:
                continue
            margin = abs(row["box_score"] - box_thresh)
            flips.append({"side": side, "index": index, "box_score": row["box_score"], "margin": margin})
            if margin > policy["box_flip_margin"]:
                violations.append(f"box {index} {side}: score {row['box_score']:.4f} is {margin:.3g} from threshold")
    if not flips and order != list(range(len(expected))):
        violations.append(f"line order differs from the reference: {order}")
    return {
        "id": where,
        "status": "failed" if violations else "passed",
        "bound": {
            "box_iou_min": policy["box_iou_min"],
            "box_score_max_abs": policy["box_score_max_abs"],
            "box_flip_margin": policy["box_flip_margin"],
        },
        "matched": len(matches),
        "actual_count": len(actual),
        "expected_count": len(expected),
        "min_iou": iou,
        "max_box_score_abs": score,
        "flips": flips,
        "violations": violations,
        "matches": [[i, j] for i, j, _ in matches],
    }


def _text_checks(actual: list[dict], expected: list[dict], matches, where: str, line_text: str) -> list[dict]:
    if line_text != "exact":
        raise PipelineError(f"unsupported line_text comparison {line_text!r}")
    return [_exact(actual[i]["text"], expected[j]["text"], f"{where}/{j}/text_vs_reference") for i, j in matches]


def evaluate_development(
    candidate: dict,
    candidate_path: Path,
    reference_path: Path,
    onnx_path: Path,
    dotnet_path: Path,
    corpus: list[dict],
) -> dict:
    """Check complete proposed evidence without reading or promoting protected fixtures.

    `corpus` is the pinned corpus as `load_corpus` verified it: it, not the reference evidence, supplies each image's
    digest and ground-truth lines.
    """
    validate_candidate_behavior(candidate)
    policy = candidate["evaluation"]["policy"]
    box_thresh = candidate["behavior"]["db"]["box_thresh"]
    digest = fixtures.sha256_file(candidate_path)
    reference, expected = _document(reference_path, candidate, digest, "reference")
    onnx, python_cases = _document(onnx_path, candidate, digest, "onnx")
    dotnet, dotnet_cases = _document(dotnet_path, candidate, digest, "dotnet")
    images = {row["id"]: row for row in corpus}
    if set(expected) != set(images):
        raise PipelineError("reference evidence does not cover exactly the pinned corpus images")
    for identifier, row in expected.items():
        if row["image_sha256"] != images[identifier]["sha256"]:
            raise PipelineError(f"reference/{identifier}: image differs from the pinned corpus")
    machines = [document.get("environment", {}).get("machine") for document in (reference, onnx, dotnet)]
    if any(not isinstance(machine, str) or not machine for machine in machines) or len(set(machines)) != 1:
        raise PipelineError("same-backend comparison requires all stages to name the same machine")
    reference_digest = fixtures.sha256_file(reference_path)
    if onnx.get("reference_sha256") != reference_digest or dotnet.get("reference_sha256") != reference_digest:
        raise PipelineError("backend evidence names a different original reference artifact")
    if set(expected) != set(python_cases) or set(expected) != set(dotnet_cases):
        raise PipelineError("backend evidence has missing or unknown required case rows")
    line_counts = {identifier: len(row["lines"]) for identifier, row in expected.items()}
    for stage, cases in (("onnx", python_cases), ("dotnet", dotnet_cases)):
        for identifier, row in cases.items():
            if row["image_sha256"] != expected[identifier]["image_sha256"]:
                raise PipelineError(f"{stage}/{identifier}: image differs from the reference")
            if len(row["forced"]) != line_counts[identifier]:
                raise PipelineError(f"{stage}/{identifier}: forced recognition must cover every reference line")
    tensors = {
        "reference": _tensors(reference_path, reference, expected, "reference", line_counts, policy["non_finite"]),
        "onnx": _tensors(onnx_path, onnx, python_cases, "onnx", line_counts, policy["non_finite"]),
        "dotnet": _tensors(dotnet_path, dotnet, dotnet_cases, "dotnet", line_counts, policy["non_finite"]),
    }
    reference_tensors, python_tensors, dotnet_tensors = (tensors[name] for name in ("reference", "onnx", "dotnet"))
    checks = []
    for identifier, row in expected.items():
        python_row, dotnet_row = python_cases[identifier], dotnet_cases[identifier]
        lines = row["lines"]
        if policy["pipeline_match"] != "exact":
            raise PipelineError(f"unsupported pipeline_match comparison {policy['pipeline_match']!r}")
        for stage, pipeline in (("reference", row["pipeline"]), ("onnx", python_row["pipeline"])):
            checks.append(
                _exact(
                    [pipeline["boxes"], pipeline["texts"], pipeline["scores_max_abs"] == 0],
                    [True, True, True],
                    f"{stage}/{identifier}/stepwise_vs_paddleocr_pipeline",
                )
            )
        checks += [
            _difference(
                dotnet_tensors[identifier, "det_input", None],
                reference_tensors[identifier, "det_input", None],
                policy["preprocess_max_abs"],
                f"dotnet-image/{identifier}/det_input_vs_reference",
            ),
            _difference(
                python_tensors[identifier, "prob_map", None],
                reference_tensors[identifier, "prob_map", None],
                policy["prob_map_max_abs"],
                f"onnx/{identifier}/prob_map_vs_reference",
            ),
            _difference(
                dotnet_tensors[identifier, "prob_map", None],
                python_tensors[identifier, "prob_map", None],
                policy["same_backend_max_abs"],
                f"dotnet-tensors/{identifier}/prob_map_vs_python",
            ),
        ]
        matched = {}
        for stage, actual in (
            ("dotnet-postprocess", dotnet_row["postprocess_lines"]),
            ("onnx", python_row["lines"]),
            ("dotnet-image", dotnet_row["lines"]),
        ):
            comparison = compare_boxes(actual, lines, policy, box_thresh, f"{stage}/{identifier}/boxes_vs_reference")
            checks.append(comparison)
            matched[stage] = {j: actual[i] for i, j in comparison["matches"]}
            if stage != "dotnet-postprocess":
                checks += _text_checks(
                    actual, lines, comparison["matches"], f"{stage}/{identifier}", policy["line_text"]
                )
        # End-to-end line confidences: Python ONNX Runtime against Paddle; .NET against Python where both matched.
        for j, python_line in matched["onnx"].items():
            checks.append(
                _rate(
                    abs(python_line["score"] - lines[j]["score"]),
                    policy["rec_score_max_abs"],
                    f"onnx/{identifier}/{j}/score_vs_reference",
                )
            )
            if j in matched["dotnet-image"]:
                checks.append(
                    _rate(
                        abs(matched["dotnet-image"][j]["score"] - python_line["score"]),
                        policy["same_backend_max_abs"],
                        f"dotnet-image/{identifier}/{j}/score_vs_python",
                    )
                )
        # Per image: an image with drawn text must yield reference lines within the CER bound; a text-free one none.
        truth = images[identifier]["lines"]
        if truth:
            text = "\n".join(line["text"] for line in lines)
            reference_errors = edit_distance(text, "\n".join(truth))
            checks.append(
                _rate(
                    reference_errors / len("\n".join(truth)),
                    policy["ground_truth_cer_max"],
                    f"reference/{identifier}/cer_vs_ground_truth",
                    lines=len(lines),
                    errors=reference_errors,
                )
            )
        else:
            checks.append(_exact(len(lines), 0, f"reference/{identifier}/no_lines_without_text"))
        for index, line in enumerate(lines):
            python_forced, dotnet_forced = python_row["forced"][index], dotnet_row["forced"][index]
            checks += [
                _difference(
                    dotnet_tensors[identifier, "rec_input", index],
                    reference_tensors[identifier, "rec_input", index],
                    policy["crop_max_abs"],
                    f"dotnet-crop/{identifier}/{index}/rec_input_vs_reference",
                ),
                _difference(
                    python_tensors[identifier, "rec_probs", index],
                    reference_tensors[identifier, "rec_probs", index],
                    policy["rec_probs_max_abs"],
                    f"onnx/{identifier}/{index}/rec_probs_vs_reference",
                ),
                _difference(
                    dotnet_tensors[identifier, "rec_probs", index],
                    python_tensors[identifier, "rec_probs", index],
                    policy["same_backend_max_abs"],
                    f"dotnet-tensors/{identifier}/{index}/rec_probs_vs_python",
                ),
                _exact(python_forced["text"], line["text"], f"onnx/{identifier}/{index}/forced_text_vs_reference"),
                _exact(
                    dotnet_forced["text"], python_forced["text"], f"dotnet-tensors/{identifier}/{index}/text_vs_python"
                ),
                _rate(
                    abs(dotnet_forced["score"] - python_forced["score"]),
                    policy["same_backend_max_abs"],
                    f"dotnet-tensors/{identifier}/{index}/confidence_vs_python",
                ),
                _rate(
                    abs(python_forced["score"] - line["score"]),
                    policy["rec_score_max_abs"],
                    f"onnx/{identifier}/{index}/forced_score_vs_reference",
                ),
            ]
    errors = sum(
        edit_distance("\n".join(line["text"] for line in row["lines"]), "\n".join(images[identifier]["lines"]))
        for identifier, row in expected.items()
    )
    total = sum(len("\n".join(image["lines"])) for image in images.values())
    checks.append(
        _rate(
            errors / total if total else 1.0,
            policy["ground_truth_cer_max"],
            "reference/corpus/cer_vs_ground_truth",
            errors=errors,
            characters=total,
        )
    )
    for path in policy["required_paths"]:
        prefix = REQUIRED_PATH_PREFIXES.get(path)
        if prefix is None or not any(check["id"].startswith(prefix) for check in checks):
            raise PipelineError(f"required evidence path {path!r} produced no checks")
    if dotnet.get("checks", {}).get("pre_cancelled_request") is not True:
        raise PipelineError(".NET cancellation evidence is missing or failed")
    failed = [check["id"] for check in checks if check["status"] != "passed"]
    return {
        "format": "starlings.ppocr-development-parity",
        "format_version": 1,
        "profile": candidate["id"],
        "task": candidate["task"],
        "status": "failed" if failed else "passed",
        "failed_checks": failed,
        "checks": checks,
        "case_count": len(expected),
        "line_count": sum(line_counts.values()),
        "texts": {
            identifier: {
                "ground_truth": images[identifier]["lines"],
                "reference": [line["text"] for line in row["lines"]],
                "python-onnx": [line["text"] for line in python_cases[identifier]["lines"]],
                "dotnet-image": [line["text"] for line in dotnet_cases[identifier]["lines"]],
            }
            for identifier, row in expected.items()
        },
        # References and the comparison policy stay proposed: no protected tolerances exist for this profile.
        "evaluation_status": "proposed",
        "policy": policy,
        "policy_source": "candidate (proposed)",
        "policy_status": "proposed",
        "protected_release_qualification": {"status": "unapproved", "G1": "not-run", "G2": "not-run"},
        "inputs": {
            "candidate_sha256": digest,
            "reference_sha256": reference_digest,
            "onnx_sha256": fixtures.sha256_file(onnx_path),
            "dotnet_sha256": fixtures.sha256_file(dotnet_path),
        },
    }


def render_report(report: dict, profile: str) -> str:
    def cell(lines: list[str]) -> str:
        return "<br>".join(line.replace("|", "\\|") for line in lines) or "(none)"

    rows = [
        f"| {identifier} | {cell(row['ground_truth'])} | {cell(row['reference'])} | {cell(row['dotnet-image'])} |"
        for identifier, row in report["texts"].items()
    ]
    return "\n".join(
        [
            f"# Development parity: {profile}",
            "",
            f"Status: **{report['status']}**; {report['case_count']} images, {report['line_count']} reference lines, "
            f"{len(report['checks'])} checks. Original references and the comparison policy are **proposed**. "
            "Protected G1/G2 release qualification is **unapproved**. No protected reference fixtures were adopted.",
            "",
            "| Image | Ground truth | Reference (Paddle) | .NET (image, ONNX) |",
            "| --- | --- | --- | --- |",
            *rows,
            "",
        ]
    )


# --- orchestration --------------------------------------------------------------------------------------------------


def _profile(recipe: dict, root: Path) -> tuple[Path, dict, dict]:
    from model_pipeline.onboarding_candidate import load_candidate

    if recipe.get("plugin") != PLUGIN or recipe.get("recipe_version") != 1:
        raise PipelineError(f"expected recipe_version 1 and plugin {PLUGIN}")
    path = repository_input(root, recipe["manifest"])
    candidate = load_candidate(path)
    return path, candidate, validate_candidate_behavior(candidate)


def describe_plan(recipe: dict, *, repo_root: Path) -> dict:
    root = repo_root.resolve()
    _, candidate, _ = _profile(recipe, root)
    origin = candidate["origin"]
    return {
        "plugin": PLUGIN,
        "recipe": recipe["id"],
        "profile": candidate["id"],
        "task": candidate["task"],
        "revision": origin["revision"],
        "components": {
            entry["role"]: {"repo": entry["repo"], "revision": entry["revision"]} for entry in origin["components"]
        },
        "asset_count": len(origin["files"]) + sum(len(entry["files"]) for entry in origin["components"]),
        "stages": list(STAGES),
        "inputs": {
            "snapshot": "local directory holding the pinned repositories as " + ", ".join(f"{r}/" for r in ROLES),
            "reference_python": "reference/ppocr/requirements.txt",
        },
        "reference_status": "proposed",
        "qualification_status": "unapproved",
        "policy": candidate["evaluation"]["policy"],
        "policy_source": "candidate (proposed)",
    }


def build_stages(
    recipe: dict,
    *,
    repo_root: Path,
    snapshot: Path,
    reference_python: Path,
    threads: int = 2,
    timeout: float = 1800,
    distribution: bool = False,
) -> list[Stage]:
    from model_pipeline.onboarding_candidate import candidate_stages, code_inputs, verify_assets

    if distribution:
        raise PipelineError("PP-OCR distribution is not implemented by this plugin")
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or not 0 < timeout <= 86400:
        raise PipelineError("timeout must be finite, positive, and at most 86400 seconds")
    root, snapshot = repo_root.resolve(), snapshot.resolve()
    for role in ROLES:
        if not (snapshot / role).is_dir() or (snapshot / role).is_symlink():
            raise PipelineError(f"snapshot must hold the pinned repositories as subdirectories; missing {role}/")
    manifest, candidate, _ = _profile(recipe, root)
    corpus = repository_input(root, recipe["reference_corpus"])
    images = {f"corpus_image:{row['image']}": row["path"] for row in load_corpus(corpus)}
    _, initial, shared = candidate_stages(
        recipe,
        repo_root=root,
        snapshot=snapshot / "det",
        reference_python=reference_python,
        threads=threads,
        timeout=timeout,
        source_files=(RUNNER, "model-pipeline/src/model_pipeline/onboarding_ppocr.py"),
    )
    python = Path(os.path.abspath(reference_python))
    env = {
        "PYTHONPATH": str(root / "model-pipeline/src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        # paddlex: no model-hoster connectivity check, no downloads (models come from the pinned snapshot).
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "DOTNET_PROCESSOR_COUNT": str(threads),
        "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
    }
    parameters = {"plugin": PLUGIN, "candidate": candidate, "threads": threads, "timeout": timeout}

    def guard(out: Path) -> None:
        guard_output(
            out,
            (snapshot, root / "model-pipeline/protected", corpus.parent, manifest),
            "stage output overlaps original source or protected evidence",
        )

    def verified(stage: Stage) -> Stage:
        """Recheck the pins of every repository before the stage reads any of them."""
        original = stage.execute

        def checked(out: Path, deps: dict) -> dict:
            guard(out)
            lock = verify_assets(candidate, snapshot / "det", out)
            records = verify_components(candidate, snapshot)
            result = original(out, deps)
            return {
                **result,
                **lock,
                "component_files_verified": len(records),
                "original_source_pins": "verified-before-read",
            }

        return replace(stage, inputs={**stage.inputs, "original_snapshot": snapshot}, execute=checked)

    assets_stage, environment_stage = initial
    original_assets = assets_stage.execute

    def assets(out: Path, deps: dict) -> dict:
        guard(out)
        result = original_assets(out, deps)
        lock = json.loads((out / "source-lock.json").read_text(encoding="utf-8"))
        lock["components"] = verify_components(candidate, snapshot)
        lock["contract_checks"] = check_configs(candidate, snapshot)
        write_json(out / "source-lock.json", lock)
        return {**result, "component_files_verified": len(lock["components"])}

    def runner(out: Path, kind: str, extra: list[str]) -> dict:
        guard(out)
        return run_command(
            [
                str(python),
                str(root / RUNNER),
                "--profile",
                str(manifest),
                "--stage",
                kind,
                "--snapshot",
                str(snapshot),
                "--corpus",
                str(corpus),
                "--out",
                str(out),
                "--threads",
                str(threads),
                *extra,
            ],
            cwd=root,
            log_path=out / "command.log",
            env=env,
            timeout=timeout,
        )

    def reference(out: Path, deps: dict) -> dict:
        return runner(out, "reference", [])

    def onnx(out: Path, deps: dict) -> dict:
        return runner(out, "onnx", ["--reference", str(deps["reference"] / "reference.json")])

    def dotnet(out: Path, deps: dict) -> dict:
        guard(out)
        with tempfile.TemporaryDirectory(prefix="dotnet-build-", dir=out) as build:
            dotnet_executable = shutil.which("dotnet") or "dotnet"
            published = Path(build) / "published"
            build_result = run_command(
                [
                    dotnet_executable,
                    "publish",
                    str(root / TOOL),
                    "-c",
                    "Release",
                    "-m:1",
                    "--disable-build-servers",
                    "--artifacts-path",
                    build,
                    "--output",
                    str(published),
                    "-p:RestoreLockedMode=true",
                ],
                cwd=root,
                log_path=out / "build.log",
                env=env,
                timeout=timeout,
            )
            assembly = published / TOOL_ASSEMBLY
            if not assembly.is_file():
                raise PipelineError("PP-OCR fidelity publish succeeded without producing the required tool assembly")
            result = run_command(
                [
                    dotnet_executable,
                    str(assembly),
                    "--profile",
                    str(manifest),
                    "--snapshot",
                    str(snapshot),
                    "--corpus",
                    str(corpus),
                    "--reference",
                    str(deps["reference"] / "reference.json"),
                    "--out",
                    str(out),
                    "--threads",
                    str(threads),
                ],
                cwd=root,
                log_path=out / "command.log",
                env=env,
                timeout=timeout,
            )
            return {**result, "build": build_result}

    def evaluation(out: Path, deps: dict) -> dict:
        guard(out)
        report = evaluate_development(
            candidate,
            manifest,
            deps["reference"] / "reference.json",
            deps["onnx"] / "onnx.json",
            deps["dotnet"] / "dotnet.json",
            load_corpus(corpus),
        )
        write_json(out / "report.json", report)
        (out / "report.md").write_text(render_report(report, candidate["id"]), encoding="utf-8")
        if report["status"] != "passed":
            raise PipelineError(
                f"PP-OCR development parity failed: {', '.join(report['failed_checks'])}; see {out / 'report.json'}"
            )
        return {"status": report["status"], "qualification_status": "unapproved", "reference_status": "proposed"}

    def review(out: Path, deps: dict) -> dict:
        guard(out)
        report = json.loads((deps["evaluate"] / "report.json").read_text(encoding="utf-8"))
        if report["status"] != "passed":
            raise PipelineError("review requires complete passed development parity")
        for name in ("report.json", "report.md"):
            shutil.copyfile(deps["evaluate"] / name, out / name)
        document = {
            "profile": candidate["id"],
            "task": candidate["task"],
            "development_parity": report,
            "candidate_commit": fixtures.git_head(root),
            "source_lock": json.loads((deps["assets"] / "source-lock.json").read_text(encoding="utf-8")),
            "support_status": "experimental",
            "release_qualification": "unapproved",
            "reference_status": "proposed",
            "policy_status": report["policy_status"],
            "protected_fixture_adoption": False,
            "distribution": "not-run",
        }
        write_json(out / "review.json", document)
        (out / "review.md").write_text(
            f"# Candidate review: {candidate['id']}\n\n"
            "Development OCR parity passed. References and the comparison policy remain proposed. "
            "Protected G1/G2 and full release qualification are unapproved; distribution is not run.\n",
            encoding="utf-8",
        )
        return {"development_parity": "passed", "release_qualification": "unapproved"}

    def stage(name: str, dependencies: tuple[str, ...], execute, inputs: dict | None = None, **extra) -> Stage:
        return Stage(name, dependencies, {**shared, **images, **(inputs or {})}, {**parameters, **extra}, execute)

    dotnet_inputs = code_inputs(
        root,
        files=(
            "global.json",
            "Directory.Build.props",
            "ai-dotnet/Directory.Build.props",
            "ai-dotnet/Directory.Build.targets",
            "ai-dotnet/Directory.Packages.props",
            *(str(path.relative_to(root)) for path in (root / "Directory.Build.targets",) if path.is_file()),
        ),
        directories=(
            "ai-dotnet/tools/Starlings.PaddleOcr.Fidelity.Tool",
            "ai-dotnet/src/Starlings.Models.PaddleOcr",
            "ai-dotnet/src/Starlings.Models.EmbeddingGemma2",
            "ai-dotnet/src/Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp",
            "ai-dotnet/tools/Shared",
            "ai-dotnet/src/Starlings.Runtime.Onnx",
            "ai-dotnet/src/Starlings.Acquisition",
        ),
    )
    return [
        replace(assets_stage, inputs={**assets_stage.inputs, "snapshot": snapshot}, execute=assets),
        environment_stage,
        verified(stage("reference", ("assets", "environment"), reference, {"corpus": corpus})),
        verified(stage("onnx", ("reference", "assets", "environment"), onnx)),
        verified(stage("dotnet", ("reference", "assets", "environment"), dotnet, dotnet_inputs)),
        stage("evaluate", ("reference", "onnx", "dotnet"), evaluation),
        stage("review", ("assets", "reference", "evaluate"), review, candidate_commit=fixtures.git_head(root)),
    ]
