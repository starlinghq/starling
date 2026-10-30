"""Write OpenCV and Clipper conformance fixtures for the owned geometry of Starlings.Models.PaddleOcr.

The .NET library restates the OpenCV and Clipper operations PaddleOCR's pipeline uses, so each is checked here against
the pinned implementations (opencv-contrib-python and pyclipper of `requirements.txt`) on seeded random inputs:

- `cv2.resize` INTER_LINEAR on uint8 BGR images (the detector and recognizer resize);
- `cv2.getPerspectiveTransform` + `cv2.warpPerspective` INTER_CUBIC with BORDER_REPLICATE (the crop);
- `cv2.findContours` RETR_LIST / CHAIN_APPROX_SIMPLE on binary maps (DB post-processing);
- `cv2.minAreaRect` + `cv2.boxPoints`, `cv2.contourArea`, `cv2.arcLength` (DB boxes, unclip distance);
- `cv2.fillPoly` masks and `cv2.mean` (DB box score);
- `pyclipper.PyclipperOffset` JT_ROUND / ET_CLOSEDPOLYGON (unclip).

Images are produced by a xorshift32 generator both sides implement, so only digests are stored for image outputs.
Run inside the hash-locked reference environment:

    .models/venv-ppocr/bin/python -I model-pipeline/reference/ppocr/make_geometry_fixtures.py \\
        --out ai-dotnet/tests/Starlings.Models.PaddleOcr.Tests/TestData/opencv-geometry-v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pyclipper


def xorshift_bytes(seed: int, count: int) -> np.ndarray:
    """xorshift32 (13, 17, 5) from a nonzero seed; each output byte is the state's top 8 bits."""
    state = seed & 0xFFFFFFFF or 1
    out = np.empty(count, np.uint8)
    for index in range(count):
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        out[index] = state >> 24
    return out


def image(seed: int, height: int, width: int, channels: int = 3) -> np.ndarray:
    return xorshift_bytes(seed, height * width * channels).reshape(height, width, channels)


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def resize_cases(rng: np.random.Generator) -> list[dict]:
    cases = []
    sizes = [((48, 300), (48, 961)), ((300, 820), (288, 832)), ((420, 70), (416, 64)), ((64, 64), (32, 32))]
    sizes += [((int(h), int(w)), (int(th), int(tw))) for h, w, th, tw in rng.integers(1, 260, (24, 4))]
    for index, ((height, width), (target_h, target_w)) in enumerate(sizes):
        source = image(1000 + index, height, width)
        output = cv2.resize(source, (target_w, target_h))
        cases.append(
            {"seed": 1000 + index, "size": [height, width], "target": [target_h, target_w], "sha256": digest(output)}
        )
    return cases


def quad(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    return np.float32(
        [
            [rng.uniform(-3, width / 3), rng.uniform(-3, height / 3)],
            [rng.uniform(2 * width / 3, width + 3), rng.uniform(-3, height / 3)],
            [rng.uniform(2 * width / 3, width + 3), rng.uniform(2 * height / 3, height + 3)],
            [rng.uniform(-3, width / 3), rng.uniform(2 * height / 3, height + 3)],
        ]
    )


def warp_cases(rng: np.random.Generator) -> list[dict]:
    cases = []
    for index in range(30):
        height, width = (int(value) for value in rng.integers(8, 140, 2))
        if index == 0:
            height, width = 60, 700  # wider than one remap block row, so block origins matter
        if index == 1:
            height, width = 520, 300  # more than 65,536 output pixels: several parallel stripes
        source = image(2000 + index, height, width)
        points = quad(rng, height, width)
        if index % 5 == 0:
            points = np.float32([[3, 4], [width - 5, 4], [width - 5, height - 6], [3, height - 6]])
        crop_w = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
        crop_h = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
        target = np.float32([[0, 0], [crop_w, 0], [crop_w, crop_h], [0, crop_h]])
        matrix = cv2.getPerspectiveTransform(points, target)
        output = cv2.warpPerspective(
            source, matrix, (crop_w, crop_h), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC
        )
        cases.append(
            {
                "seed": 2000 + index,
                "size": [height, width],
                "points": points.tolist(),
                "crop": [crop_h, crop_w],
                "matrix": matrix.tolist(),
                "sha256": digest(output),
            }
        )
    return cases


def blob_map(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    field = cv2.GaussianBlur(rng.random((height, width)).astype(np.float32), (0, 0), float(rng.uniform(1.0, 3.0)))
    return (field > np.quantile(field, float(rng.uniform(0.55, 0.8)))).astype(np.uint8)


def rows(bitmap: np.ndarray) -> list[str]:
    return ["".join("1" if value else "0" for value in row) for row in bitmap]


def contour_cases(rng: np.random.Generator) -> list[dict]:
    cases = []
    for index in range(20):
        height, width = (int(value) for value in rng.integers(6, 48, 2))
        bitmap = blob_map(rng, height, width)
        if index == 0:
            bitmap = np.zeros((12, 16), np.uint8)
            bitmap[2:10, 2:14] = 1
            bitmap[4:8, 5:11] = 0  # a ring: one outer and one hole contour
            bitmap[0, 15] = 1  # a single pixel touching the border
        contours, _ = cv2.findContours(bitmap * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cases.append({"bitmap": rows(bitmap), "contours": [contour.reshape(-1, 2).tolist() for contour in contours]})
    return cases


def rect_case(points: np.ndarray) -> dict:
    center, size, angle = cv2.minAreaRect(points)
    return {
        "points": points.tolist(),
        "dtype": "int32" if points.dtype == np.int32 else "float32",
        "center": [float(value) for value in center],
        "size": [float(value) for value in size],
        "angle": float(angle),
        "box": cv2.boxPoints((center, size, angle)).tolist(),
    }


def min_area_rect_cases(rng: np.random.Generator) -> list[dict]:
    cases = [
        rect_case(np.int32([[18, 26], [379, 26], [379, 72], [18, 72]])),
        rect_case(np.int32([[0, 0], [5, 0], [5, 5], [0, 5]])),
        rect_case(np.float32([[1.5, 2.25], [40.75, 9.5], [38.0, 20.125], [-1.0, 13.0]])),
    ]
    for _ in range(25):
        count = int(rng.integers(3, 30))
        points = rng.integers(-5, 200, (count, 2)).astype(np.int32)
        if rng.random() < 0.5:
            points = (points + rng.random((count, 2)) * 3).astype(np.float32)
        cases.append(rect_case(points))
    # Text-line boxes: long, thin, slightly rotated integer quads (the DB and crop inputs), and their float corners.
    for _ in range(60):
        length, height = float(rng.uniform(20, 800)), float(rng.uniform(8, 60))
        angle = float(rng.normal(0, 0.05)) if rng.random() < 0.8 else float(rng.uniform(-0.6, 0.6))
        cx, cy = float(rng.uniform(0, 900)), float(rng.uniform(0, 900))
        corners = cv2.boxPoints(((cx, cy), (length, height), np.degrees(angle)))
        jitter = rng.integers(-1, 2, (4, 2))
        cases.append(rect_case((np.rint(corners) + jitter).astype(np.int32)))
        cases.append(rect_case(corners.astype(np.float32)))
    for bitmap in (blob_map(rng, 40, 60) for _ in range(5)):
        for contour in cv2.findContours(bitmap * 255, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0][:3]:
            if len(contour) >= 3:
                cases.append(rect_case(contour.reshape(-1, 2).astype(np.int32)))
    return cases


def measure_cases(rng: np.random.Generator) -> list[dict]:
    cases = []
    for _ in range(20):
        points = quad(rng, float(rng.uniform(10, 60)), float(rng.uniform(20, 300)))
        cases.append(
            {
                "points": points.tolist(),
                "area": float(cv2.contourArea(points)),
                "length": float(cv2.arcLength(points, True)),
            }
        )
    return cases


def fill_cases(rng: np.random.Generator) -> list[dict]:
    cases = []
    for index in range(60):
        height, width = (int(value) for value in rng.integers(5, 60, 2))
        points = quad(rng, height, width).astype(np.int32)
        if index % 6 == 0:
            points = np.int32([[1, 1], [width - 2, 1], [width - 2, height - 2], [1, height - 2]])
        if index % 4 == 1:
            points = (points + rng.integers(-4, 5, (4, 2))).astype(np.int32)  # edges leaving the mask: clipped
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, points.reshape(1, -1, 2), 1)
        values = xorshift_bytes(3000 + index, height * width).reshape(height, width).astype(np.float32) / 255
        cases.append(
            {
                "size": [height, width],
                "points": points.tolist(),
                "mask": rows(mask),
                "values_seed": 3000 + index,
                "mean": float(cv2.mean(values, mask)[0]),
            }
        )
    return cases


def offset_cases(rng: np.random.Generator) -> list[dict]:
    cases = []
    for _ in range(60):
        points = quad(rng, float(rng.uniform(8, 50)), float(rng.uniform(20, 400)))
        distance = float(cv2.contourArea(points) * 1.5 / cv2.arcLength(points, True))
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        cases.append({"points": points.tolist(), "distance": distance, "paths": offset.Execute(distance)})
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    rng = np.random.default_rng(20261030)
    fixture = {
        "format": "starlings.ppocr-geometry-fixtures",
        "format_version": 1,
        "generator": "model-pipeline/reference/ppocr/make_geometry_fixtures.py",
        "opencv": cv2.__version__,
        "pyclipper": pyclipper.__version__,
        "numpy": np.__version__,
        "image_generator": "xorshift32 (13, 17, 5), top byte of each state, row-major HWC",
        "resize": resize_cases(rng),
        "warp": warp_cases(rng),
        "contours": contour_cases(rng),
        "min_area_rect": min_area_rect_cases(rng),
        "measure": measure_cases(rng),
        "fill": fill_cases(rng),
        "offset": offset_cases(rng),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{args.out}: {sum(len(value) for value in fixture.values() if isinstance(value, list))} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
