"""Pinned PP-OCRv5 reference (Paddle inference engine) or ONNX Runtime execution, into proposed artifacts only.

Both stages run PaddleOCR's own building blocks (paddlex 3.x) step by step, so every intermediate is recorded, and then
check that the steps reproduce the PaddleOCR pipeline (`PaddleOCR.predict`) exactly on the same image:

reference: Paddle inference engine on the pinned Paddle models (oneDNN off; see below). Per image: the detector input
           tensor (BGR, resized and normalized), the probability map, DB boxes in pipeline order with their box
           scores, and per box the recognizer input tensor, its probabilities, and the CTC text and confidence.
onnx:      PaddleOCR with engine="onnxruntime" on the pinned ONNX exports. The probability map is computed from the
           reference detector input (which this stage also rebuilds and requires to be byte-identical), and the
           recognizer probabilities from each reference recognizer input ("forced" rows); the end-to-end lines come
           from this stage's own boxes.

oneDNN is disabled on the Paddle engine: paddlepaddle 3.3.1 with the PIR executor fails these graphs under oneDNN
(`ConvertPirAttribute2RuntimeAttribute not support [pir::ArrayAttribute<pir::DoubleAttribute>]`), so the reference is
Paddle's native CPU kernels.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np

from model_pipeline import fixtures
from model_pipeline.onboarding_ppocr import FORMAT, load_corpus, validate_candidate_behavior

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


class Tensors:
    """One little-endian float32 sidecar holding every named tensor, with its layout."""

    def __init__(self, path: Path) -> None:
        self.path, self.layout = path, []
        self.handle = path.open("wb")

    def add(self, identifier: str, name: str, values: np.ndarray, line: int | None = None) -> None:
        values = np.ascontiguousarray(values, dtype="<f4")
        if not np.isfinite(values).all():
            raise ValueError(f"{identifier}/{name}: non-finite values")
        row = {"id": identifier, "name": name, "offset": self.handle.tell() // 4, "shape": list(values.shape)}
        if line is not None:
            row["line"] = line
        self.layout.append(row)
        self.handle.write(values.tobytes())

    def close(self) -> dict:
        self.handle.close()
        return {"file": self.path.name, "sha256": fixtures.sha256_file(self.path), "cases": self.layout}


def sort_order(boxes: np.ndarray, tolerance: int) -> list[int]:
    """Indices of paddlex SortQuadBoxes: by (top-left y, x), then a bubble pass swapping same-row boxes by x."""
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][0][1], boxes[i][0][0]))
    for i in range(len(order) - 1):
        for j in range(i, -1, -1):
            a, b = boxes[order[j]], boxes[order[j + 1]]
            if abs(b[0][1] - a[0][1]) < tolerance and b[0][0] < a[0][0]:
                order[j], order[j + 1] = order[j + 1], order[j]
            else:
                break
    return order


def read_tensor(document: dict, blob: np.ndarray, identifier: str, name: str, line: int | None = None) -> np.ndarray:
    for entry in document["tensors"]["cases"]:
        if (entry["id"], entry["name"], entry.get("line")) == (identifier, name, line):
            count = int(np.prod(entry["shape"]))
            return blob[entry["offset"] : entry["offset"] + count].reshape(entry["shape"])
    raise ValueError(f"reference tensor {identifier}/{name}/{line} is missing")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("reference", "onnx"), required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.threads <= 64:
        parser.error("threads must be 1 through 64")
    if args.stage == "onnx" and args.reference is None:
        parser.error("onnx requires --reference")
    candidate = json.loads(args.profile.read_text(encoding="utf-8"))
    validate_candidate_behavior(candidate)
    behavior = candidate["behavior"]
    resize, db, pipeline_contract = behavior["detector"]["resize"], behavior["db"], behavior["pipeline"]
    args.out.mkdir(parents=True, exist_ok=True)
    profile_hash = fixtures.sha256_file(args.profile)
    corpus = load_corpus(args.corpus)

    import cv2
    from paddleocr import PaddleOCR, TextDetection, TextRecognition
    from paddlex.inference.pipelines.components import CropByPolys, SortQuadBoxes

    cv2.setNumThreads(1)
    if args.stage == "reference":
        engine, roles = "paddle", ("det-paddle", "rec-paddle")
        options = {"engine": "paddle", "enable_mkldnn": False, "cpu_threads": args.threads}
    else:
        engine, roles = "onnxruntime", ("det", "rec")
        options = {
            "engine": "onnxruntime",
            "engine_config": {"intra_op_num_threads": args.threads, "inter_op_num_threads": 1},
        }
    det_dir, rec_dir = (str(args.snapshot / role) for role in roles)
    detector = TextDetection(
        model_name=behavior["detector_model"],
        model_dir=det_dir,
        device="cpu",
        limit_side_len=resize["limit_side_len"],
        limit_type=resize["limit_type"],
        thresh=db["thresh"],
        box_thresh=db["box_thresh"],
        unclip_ratio=db["unclip_ratio"],
        **options,
    ).paddlex_predictor
    recognizer = TextRecognition(model_name=behavior["recognizer_model"], model_dir=rec_dir, device="cpu", **options)
    recognizer = recognizer.paddlex_predictor
    pipeline = PaddleOCR(
        text_detection_model_name=behavior["detector_model"],
        text_detection_model_dir=det_dir,
        text_recognition_model_name=behavior["recognizer_model"],
        text_recognition_model_dir=rec_dir,
        use_doc_orientation_classify=pipeline_contract["document_preprocessing"],
        use_doc_unwarping=pipeline_contract["document_preprocessing"],
        use_textline_orientation=pipeline_contract["textline_orientation"],
        text_recognition_batch_size=behavior["recognizer"]["batch_size"],
        text_det_limit_side_len=resize["limit_side_len"],
        text_det_limit_type=resize["limit_type"],
        text_det_thresh=db["thresh"],
        text_det_box_thresh=db["box_thresh"],
        text_det_unclip_ratio=db["unclip_ratio"],
        text_rec_score_thresh=pipeline_contract["rec_score_thresh"],
        device="cpu",
        **options,
    )
    sorter, cropper = SortQuadBoxes(), CropByPolys(det_box_type=behavior["db"]["box_type"])

    def detect(image: np.ndarray) -> np.ndarray:
        resized, shapes = detector.pre_tfs["Resize"](
            imgs=[image],
            limit_side_len=resize["limit_side_len"],
            limit_type=resize["limit_type"],
            max_side_limit=resize["max_side_limit"],
        )
        normalized = detector.pre_tfs["Normalize"](imgs=resized)
        return detector.pre_tfs["ToBatch"](imgs=detector.pre_tfs["ToCHW"](imgs=normalized))[0], shapes

    def boxes_of(probability: np.ndarray, shapes) -> tuple[np.ndarray, list[float]]:
        polys, scores = detector.post_op(
            [probability], shapes, thresh=db["thresh"], box_thresh=db["box_thresh"], unclip_ratio=db["unclip_ratio"]
        )
        return np.asarray(polys[0]), [float(score) for score in scores[0]]

    def recognize_input(crop: np.ndarray) -> np.ndarray:
        return recognizer.pre_tfs["ToBatch"](imgs=recognizer.pre_tfs["ReisizeNorm"](imgs=[crop]))[0]

    def decode(probabilities: np.ndarray) -> tuple[str, float]:
        texts, scores = recognizer.post_op([probabilities])
        return texts[0], float(scores[0])

    reference = blob = None
    if args.reference is not None:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        if reference["candidate_sha256"] != profile_hash or reference["profile"] != candidate["id"]:
            raise ValueError("reference identity differs from the selected candidate")
        blob = np.fromfile(args.reference.parent / reference["tensors"]["file"], dtype="<f4")
        reference_cases = {row["id"]: row for row in reference["cases"]}
    tensors = Tensors(args.out / f"{args.stage}-tensors.f32")
    cases = []
    for row in corpus:
        identifier = row["id"]
        image = detector.pre_tfs["Read"](imgs=[str(row["path"])])[0]
        det_input, shapes = detect(image)
        if reference is not None and not np.array_equal(
            det_input, read_tensor(reference, blob, identifier, "det_input")
        ):
            raise ValueError(f"{identifier}: this stage's detector input differs from the reference input")
        probability = detector.runner(x=[det_input])[0]
        polys, box_scores = boxes_of(probability, shapes)
        order = sort_order(polys, pipeline_contract["sort_row_tolerance"]) if len(polys) else []
        ordered = [polys[index] for index in order]
        if len(polys) and not all(
            np.array_equal(left, right) for left, right in zip(ordered, sorter(polys), strict=True)
        ):
            raise ValueError(f"{identifier}: the restated box order differs from SortQuadBoxes")
        crops = list(cropper(image, ordered)) if ordered else []
        lines, line_inputs, line_probs = [], [], []
        for index, (poly, crop) in enumerate(zip(ordered, crops, strict=True)):
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                raise ValueError(f"{identifier}: empty crop for box {index}; the contract has no rule for it")
            rec_input = recognize_input(crop)
            probabilities = recognizer.runner(x=[rec_input])[0]
            text, score = decode(probabilities)
            lines.append(
                {
                    "box": poly.astype(int).tolist(),
                    "box_score": box_scores[order[index]],
                    "text": text,
                    "score": score,
                    "crop": [int(crop.shape[0]), int(crop.shape[1])],
                }
            )
            line_inputs.append(rec_input)
            line_probs.append(probabilities)
        result = pipeline.predict(str(row["path"]))[0]
        pipeline_boxes = [np.asarray(poly).astype(int).tolist() for poly in result["rec_polys"]]
        check = {
            "boxes": pipeline_boxes == [line["box"] for line in lines],
            "texts": list(result["rec_texts"]) == [line["text"] for line in lines],
            "scores_max_abs": max(
                (abs(float(a) - line["score"]) for a, line in zip(result["rec_scores"], lines, strict=False)),
                default=0.0,
            )
            if len(result["rec_scores"]) == len(lines)
            else float("inf"),
        }
        if not (check["boxes"] and check["texts"] and check["scores_max_abs"] == 0):
            raise ValueError(f"{identifier}: the stepwise run differs from the PaddleOCR pipeline: {check}")
        case = {
            "id": identifier,
            "image_sha256": row["sha256"],
            "det": {"resized": list(det_input.shape[2:]), "shape": [float(value) for value in shapes[0]]},
            "lines": lines,
            "pipeline": check,
        }
        if reference is None:
            tensors.add(identifier, "det_input", det_input)
            tensors.add(identifier, "prob_map", probability)
            for index, (rec_input, probabilities) in enumerate(zip(line_inputs, line_probs, strict=True)):
                tensors.add(identifier, "rec_input", rec_input, index)
                tensors.add(identifier, "rec_probs", probabilities, index)
        else:
            # Same inputs as the reference: its detector input (checked equal above) and its recognizer inputs.
            tensors.add(identifier, "prob_map", probability)
            forced = []
            for index in range(len(reference_cases[identifier]["lines"])):
                rec_input = read_tensor(reference, blob, identifier, "rec_input", index)
                probabilities = recognizer.runner(x=[np.ascontiguousarray(rec_input)])[0]
                text, score = decode(probabilities)
                forced.append({"text": text, "score": score})
                tensors.add(identifier, "rec_probs", probabilities, index)
            case["forced"] = forced
        cases.append(case)
    import paddle
    import paddleocr
    import paddlex

    roles_files = (
        ("det-paddle", "inference.pdiparams"),
        ("det-paddle", "inference.json"),
        ("rec-paddle", "inference.pdiparams"),
        ("rec-paddle", "inference.json"),
    )
    if args.stage == "onnx":
        roles_files = (("det", "inference.onnx"), ("rec", "inference.onnx"))
    roles_files += tuple((role, "inference.yml") for role in roles)
    import onnxruntime

    document = {
        "format": FORMAT,
        "format_version": 1,
        "profile": candidate["id"],
        "task": candidate["task"],
        "stage": args.stage,
        "engine": engine,
        "candidate_sha256": profile_hash,
        "reference_status": "proposed",
        "assets": [
            {"path": f"{role}/{name}", "sha256": fixtures.sha256_file(args.snapshot / role / name)}
            for role, name in roles_files
        ],
        "environment": {
            **fixtures.environment(
                ["paddlepaddle", "paddleocr", "paddlex", "onnxruntime", "numpy", "opencv-contrib-python"]
            ),
            "paddle": paddle.__version__,
            "paddleocr": paddleocr.__version__,
            "paddlex": paddlex.__version__,
            "opencv": cv2.__version__,
            "machine": platform.node(),
            "provider": "CPUExecutionProvider",
            "threads": args.threads,
            "onnxruntime": onnxruntime.__version__,
            "paddle_onednn": False if args.stage == "reference" else None,
        },
        "cases": cases,
        "tensors": tensors.close(),
    }
    if args.reference is not None:
        document["reference_sha256"] = fixtures.sha256_file(args.reference)
    fixtures.write_json(args.out / f"{args.stage}.json", document)
    print(f"{candidate['id']}: {args.stage}, {len(cases)} images, original references remain proposed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
