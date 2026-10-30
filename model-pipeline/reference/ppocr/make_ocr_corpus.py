"""Write the OCR reference corpus: rendered text images with known ground truth, plus a JSONL index.

Every image is drawn here from text this script holds, with system Noto fonts (SIL Open Font License 1.1); no
third-party image is redistributed. Each font file is pinned by SHA-256 and the run refuses a different file, so the
images regenerate byte-identically only with the same fonts, Pillow and FreeType (the index records all three). The
committed PNGs, pinned by SHA-256 in the index, are the corpus; this script is its provenance.

The ground truth is the drawn text, one entry per line in reading order (top to bottom, left to right). It is recorded
only for a corpus-level character error rate sanity bound; parity is judged against the fresh Paddle reference, never
against this text.

Run inside the hash-locked reference environment (Pillow and numpy are pinned there):

    .models/venv-ppocr/bin/python -I model-pipeline/reference/ppocr/make_ocr_corpus.py \\
        --out model-pipeline/reference/corpus
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, features

FONTS = {
    "sans": ("/usr/share/fonts/noto/NotoSans-Regular.ttf", 0),
    "sans-bold": ("/usr/share/fonts/noto/NotoSans-Bold.ttf", 0),
    "serif": ("/usr/share/fonts/noto/NotoSerif-Regular.ttf", 0),
    "mono": ("/usr/share/fonts/noto/NotoSansMono-Regular.ttf", 0),
    # Face 2 of the collection is Noto Sans CJK SC (Simplified Chinese).
    "cjk": ("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc", 2),
}
NOTO, CJK = "/usr/share/fonts/noto/", "/usr/share/fonts/noto-cjk/"
FONT_SHA256 = {
    "/usr/share/fonts/noto/NotoSans-Regular.ttf": "478c558ea716033cd60c03438f628dfa75694dcf6b5f6d505a2f05fd2b4f3823",
    "/usr/share/fonts/noto/NotoSans-Bold.ttf": "1df075a380fc7cb898acf64c1f7b3b4dd780de3caa860178bf929de35817a913",
    "/usr/share/fonts/noto/NotoSerif-Regular.ttf": "19e72cd8d595fae5bd74a5206f5d938512e1183d4fed7abb1ec1be1d7efa5f88",
    NOTO + "NotoSansMono-Regular.ttf": "65b5e2b2c4a1fba9ae8be1f026cb35b03dcb8886d9b2a4147054fde12f7e767d",
    CJK + "NotoSansCJK-Regular.ttc": "b76b0433203017ca80401b2ee0dd69350349871c4b19d504c34dbdd80541690a",
}
FONT_PACKAGES = "Arch Linux noto-fonts 1:2026.10.01-1, noto-fonts-cjk (system fonts; SIL Open Font License 1.1)"
DIRECTORY = "ocr-v1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path, index = FONTS[name]
    return ImageFont.truetype(path, size, index=index, layout_engine=ImageFont.Layout.BASIC)


def canvas(width: int, height: int, color=(255, 255, 255)) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), color)
    return image, ImageDraw.Draw(image)


def lines(draw: ImageDraw.ImageDraw, rows: list[tuple[int, int, str, str, int, tuple]]) -> list[str]:
    for x, y, text, face, size, color in rows:
        draw.text((x, y), text, font=font(face, size), fill=color)
    return [row[2] for row in rows]


def document() -> tuple[Image.Image, list[str]]:
    image, draw = canvas(820, 300)
    text = lines(
        draw,
        [
            (32, 24, "Quarterly report: shipping volumes", "sans-bold", 30, (0, 0, 0)),
            (32, 86, "Orders rose 12% to 48,210 in the third quarter.", "sans", 24, (20, 20, 20)),
            (32, 136, "Returns fell from 3.4% to 2.9% after the change.", "sans", 24, (20, 20, 20)),
            (32, 186, "Contact: ops-team@example.org, ext. 5521", "serif", 24, (40, 40, 40)),
            (32, 236, "Next review: 2026-11-15 (draft v3)", "serif", 24, (40, 40, 40)),
        ],
    )
    return image, text


def invoice() -> tuple[Image.Image, list[str]]:
    image, draw = canvas(700, 260, (250, 248, 240))
    draw.rectangle((20, 20, 680, 240), outline=(90, 90, 90), width=2)
    text = lines(
        draw,
        [
            (44, 40, "INVOICE #4821", "sans-bold", 30, (30, 30, 120)),
            (44, 100, "Widget A x3", "mono", 22, (0, 0, 0)),
            (470, 100, "$1,299.00", "mono", 22, (0, 0, 0)),
            (44, 150, "Service fee", "mono", 22, (0, 0, 0)),
            (470, 150, "$45.50", "mono", 22, (0, 0, 0)),
            (44, 196, "Total due", "sans-bold", 24, (150, 20, 20)),
            (470, 196, "$1,344.50", "sans-bold", 24, (150, 20, 20)),
        ],
    )
    return image, text


def chinese() -> tuple[Image.Image, list[str]]:
    image, draw = canvas(640, 240)
    text = lines(
        draw,
        [
            (30, 24, "文字识别测试", "cjk", 40, (0, 0, 0)),
            (30, 96, "今天的天气很好，我们去公园散步。", "cjk", 30, (10, 10, 10)),
            (30, 160, "价格：128元，数量：3件", "cjk", 30, (60, 30, 0)),
        ],
    )
    return image, text


def small_print() -> tuple[Image.Image, list[str]]:
    image, draw = canvas(600, 260)
    rows = [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "Sphinx of black quartz, judge my vow.",
        "How vexingly quick daft zebras jump!",
        "Waltz, bad nymph, for quick jigs vex.",
        "Bright vixens jump; dozy fowl quack.",
        "Jackdaws love my big sphinx of quartz.",
        "Five quacking zephyrs jolt my wax bed.",
    ]
    text = lines(draw, [(18, 12 + 30 * index, row, "sans", 16, (0, 0, 0)) for index, row in enumerate(rows)])
    return image, text


def interface() -> tuple[Image.Image, list[str]]:
    """A synthetic application window: title bar, buttons and low-contrast labels on colored panels."""
    image, draw = canvas(760, 320, (236, 239, 244))
    draw.rectangle((0, 0, 760, 48), fill=(38, 50, 72))
    draw.rounded_rectangle((40, 210, 250, 270), radius=10, fill=(52, 120, 246))
    draw.rounded_rectangle((300, 210, 510, 270), radius=10, fill=(220, 224, 230))
    draw.rectangle((560, 80, 740, 300), fill=(255, 255, 255), outline=(200, 205, 212))
    text = lines(
        draw,
        [
            (20, 10, "Settings - Account", "sans-bold", 22, (240, 240, 240)),
            (40, 80, "Display name", "sans", 20, (90, 96, 110)),
            (40, 130, "Email notifications enabled", "sans", 20, (90, 96, 110)),
            (80, 226, "Save changes", "sans-bold", 20, (255, 255, 255)),
            (350, 226, "Cancel", "sans", 20, (40, 44, 52)),
            (580, 100, "Storage", "sans-bold", 18, (30, 30, 30)),
            (580, 140, "7.2 GB of 15 GB", "sans", 16, (80, 80, 80)),
        ],
    )
    # Reading order: the side panel's lines sort by their top edge between the left column's lines.
    order = [0, 1, 5, 2, 6, 3, 4]
    return image, [text[index] for index in order]


def rotated() -> tuple[Image.Image, list[str]]:
    """Two lines rotated by 7 degrees: quadrilateral boxes, so the crop is a true perspective warp."""
    image, draw = canvas(640, 220)
    text = lines(
        draw,
        [
            (60, 50, "Rotated text line one", "sans", 30, (0, 0, 0)),
            (60, 120, "Second line, 2026", "serif", 30, (0, 0, 0)),
        ],
    )
    turned = image.rotate(7, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    return turned, text


def perspective() -> tuple[Image.Image, list[str]]:
    """A line under a mild perspective transform (keystone), as photographed paper."""
    image, draw = canvas(620, 160)
    text = lines(draw, [(40, 50, "Perspective sample 4471", "sans-bold", 36, (10, 10, 10))])
    # Pillow's coefficients map output to input pixels.
    coefficients = (1.0, 0.06, -10.0, 0.02, 1.0, -6.0, 0.00008, 0.0002)
    warped = image.transform(
        (620, 160), Image.Transform.PERSPECTIVE, coefficients, Image.Resampling.BICUBIC, fillcolor=(255, 255, 255)
    )
    return warped, text


def vertical() -> tuple[Image.Image, list[str]]:
    """A horizontal line turned 90 degrees clockwise: its crop is tall, so the pipeline rotates it back."""
    image, draw = canvas(420, 70)
    text = lines(draw, [(16, 12, "Sideways label 90", "sans", 30, (0, 0, 0))])
    return image.transpose(Image.Transpose.ROTATE_270), text


def noisy() -> tuple[Image.Image, list[str]]:
    """Seeded Gaussian noise and a slight blur over two lines."""
    image, draw = canvas(520, 170, (245, 245, 245))
    text = lines(
        draw,
        [
            (24, 30, "Noisy scan, page 12", "serif", 30, (25, 25, 25)),
            (24, 100, "Signal over noise", "sans", 28, (25, 25, 25)),
        ],
    )
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(0.8)), dtype=np.float64)
    noise = np.random.default_rng(20261030).normal(0.0, 9.0, blurred.shape)
    return Image.fromarray(np.clip(np.rint(blurred + noise), 0, 255).astype(np.uint8), "RGB"), text


def blank() -> tuple[Image.Image, list[str]]:
    """No text: a soft horizontal gradient. The detector must return no box."""
    ramp = np.linspace(200, 250, 400, dtype=np.float64)
    pixels = np.repeat(np.repeat(ramp[None, :, None], 120, axis=0), 3, axis=2)
    return Image.fromarray(np.rint(pixels).astype(np.uint8), "RGB"), []


IMAGES = {
    "document": document,
    "invoice": invoice,
    "chinese": chinese,
    "small-print": small_print,
    "interface": interface,
    "rotated": rotated,
    "perspective": perspective,
    "vertical": vertical,
    "noisy": noisy,
    "blank": blank,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fonts = {}
    for path in FONT_SHA256:
        digest = sha256(Path(path).read_bytes())
        if FONT_SHA256[path] is not None and FONT_SHA256[path] != digest:
            raise SystemExit(f"{path}: SHA-256 {digest} differs from the pinned font")
        fonts[path] = digest
    directory = args.out / DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    index = []
    for identifier, draw in IMAGES.items():
        image, text = draw()
        buffer = io.BytesIO()
        # No ancillary chunks (gamma, ICC, text, time): decoders must read the stored 8-bit RGB samples as they are.
        image.save(buffer, format="PNG", optimize=False, compress_level=9)
        data = buffer.getvalue()
        name = f"{identifier}.png"
        (directory / name).write_bytes(data)
        index.append(
            {
                "id": identifier,
                "image": f"{DIRECTORY}/{name}",
                "sha256": sha256(data),
                "width": image.width,
                "height": image.height,
                "lines": text,
                "license": "MIT OR Apache-2.0",
                "source": {
                    "generator": "model-pipeline/reference/ppocr/make_ocr_corpus.py",
                    "fonts": "SIL Open Font License 1.1",
                },
            }
        )
    manifest = {
        "fonts": fonts,
        "font_packages": FONT_PACKAGES,
        "pillow": Image.__version__,
        "freetype": features.version("freetype2"),
        "numpy": np.__version__,
    }
    (args.out / "ocr-corpus-v1.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in index), encoding="utf-8"
    )
    (directory / "GENERATOR.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(index)} images in {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
