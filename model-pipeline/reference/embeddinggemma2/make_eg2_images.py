"""Image set for the EmbeddingGemma 2 image-path spike: generated from a seed, plus pinned public-domain photos.

Nothing produced here is committed. The script writes every image into `--out` and an `images.json` manifest with,
per image: id, file, how it was made, source and license, PIL mode and size as decoded, the file's SHA-256 and the
SHA-256 of the decoded pixels (`np.asarray(Image.open(f))`, which pins the decoder output independently of the
encoder: PNG files written by another zlib build differ in bytes but not in pixels).

Sources:

- Synthetic images (gradients, shapes, a text render with Pillow's built-in bitmap font, checkerboard, noise, size
  and mode edge cases) are generated with NumPy's PCG64 from `SEED`. They are original works of this repository
  (MIT, like the rest of `model-pipeline/`).
- Photos are fetched from the scikit-image repository pinned by full commit SHA (`SKIMAGE_COMMIT`) and checked
  against the SHA-256 below. Licenses as stated by scikit-image (`src/_skimage2/data/_fetchers.py` at that commit):
  chelsea and coffee CC0 (photographers Stefan van der Walt and Rachel Michetti), camera CC0 (Lav Varshney),
  astronaut public domain (NASA), rocket public domain (SpaceX), hubble_deep_field public domain (NASA).
- Derived images (EXIF-rotated, RGBA, palette, CMYK, 16-bit, tiny, progressive JPEG) are made here from those
  photos or from synthetic images; they keep the license of their source.

Usage:
    python make_eg2_images.py --out <dir> [--photos <cache dir>]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SEED = 20261011
SKIMAGE_COMMIT = "b700edc8005e4ee9e74fc83259f09e8df3213f15"
SKIMAGE_URL = "https://raw.githubusercontent.com/scikit-image/scikit-image/{commit}/src/_skimage2/data/{name}"
PHOTOS = {
    "chelsea.png": ("CC0-1.0", "Stefan van der Walt (scikit-image data)"),
    "coffee.png": ("CC0-1.0", "Rachel Michetti, courtesy of Pikolo Espresso Bar (scikit-image data)"),
    "camera.png": ("CC0-1.0", "Lav Varshney (scikit-image data)"),
    "astronaut.png": ("public domain", "NASA Great Images database (scikit-image data)"),
    "rocket.jpg": ("public domain", "SpaceX launch photo of DSCOVR (scikit-image data)"),
    "hubble_deep_field.jpg": ("public domain", "NASA/HubbleSite (scikit-image data)"),
}
PHOTO_SHA256 = {
    "astronaut.png": "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5",
    "camera.png": "b0793d2adda0fa6ae899c03989482bff9a42d3d5690fc7e3648f2795d730c23a",
    "chelsea.png": "596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb",
    "coffee.png": "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7",
    "hubble_deep_field.jpg": "3a19c5dd8a927a9334bb1229a6d63711b1c0c767fb27e2286e7c84a3e2c2f5f4",
    "rocket.jpg": "c2dd0de7c538df8d111e479619b129464d0269d0ae5fd18ca91d33a7fdfea95c",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_photo(name: str, cache: Path) -> Path:
    path = cache / name
    if not path.exists():
        url = SKIMAGE_URL.format(commit=SKIMAGE_COMMIT, name=name)
        with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 (pinned https URL)
            path.write_bytes(r.read())
    expected = PHOTO_SHA256.get(name)
    digest = sha256_bytes(path.read_bytes())
    if expected is not None and digest != expected:
        raise SystemExit(f"{name}: SHA-256 {digest} differs from the pinned {expected}")
    return path


def rng(stream: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64([SEED, stream]))


def gradient(w: int, h: int) -> Image.Image:
    x = np.linspace(0.0, 1.0, w)[None, :]
    y = np.linspace(0.0, 1.0, h)[:, None]
    r = 255 * x * np.ones_like(y)
    g = 255 * y * np.ones_like(x)
    b = 255 * (0.5 + 0.5 * np.sin(6.0 * x + 4.0 * y))
    return Image.fromarray(np.stack([r, g, b], -1).round().astype(np.uint8), "RGB")


def shapes(w: int, h: int) -> Image.Image:
    im = Image.new("RGB", (w, h), (235, 228, 210))
    d = ImageDraw.Draw(im)
    d.ellipse((w * 0.1, h * 0.15, w * 0.45, h * 0.7), fill=(200, 30, 40), outline=(20, 20, 20), width=6)
    d.rectangle((w * 0.55, h * 0.2, w * 0.9, h * 0.55), fill=(30, 90, 200))
    d.polygon([(w * 0.5, h * 0.95), (w * 0.7, h * 0.6), (w * 0.9, h * 0.95)], fill=(40, 160, 70))
    d.line((0, h - 1, w - 1, 0), fill=(0, 0, 0), width=3)
    return im


def text_render(w: int, h: int) -> Image.Image:
    im = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default_imagefont()  # Pillow's built-in bitmap font (no FreeType, no font file)
    lines = ["EMBEDDING GEMMA 2 IMAGE PATH", "The quick brown fox jumps over the lazy dog 0123456789"]
    y = 10
    for i, line in enumerate(lines):
        small = Image.new("L", (d.textlength(line, font=font).__ceil__() + 2, 14), 255)
        ImageDraw.Draw(small).text((1, 1), line, font=font, fill=0)
        scale = 4 if i == 0 else 2
        big = small.resize((small.width * scale, small.height * scale), Image.Resampling.NEAREST)
        im.paste(Image.merge("RGB", [big] * 3), (10, y))
        y += big.height + 12
    return im


def checker(w: int, h: int, cell: int) -> Image.Image:
    yy, xx = np.mgrid[0:h, 0:w]
    a = (((xx // cell) + (yy // cell)) % 2).astype(np.uint8) * 255
    return Image.fromarray(np.stack([a, 255 - a, a // 2], -1), "RGB")


def noise(w: int, h: int, stream: int) -> Image.Image:
    return Image.fromarray(rng(stream).integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")


def smooth_large(w: int, h: int) -> Image.Image:
    x = np.linspace(0, 8 * np.pi, w, dtype=np.float32)[None, :]
    y = np.linspace(0, 6 * np.pi, h, dtype=np.float32)[:, None]
    r = 127.5 + 127.5 * np.sin(x) * np.cos(y)
    g = 127.5 + 127.5 * np.cos(0.5 * x + y)
    b = 127.5 + 127.5 * np.sin(0.3 * x - 0.7 * y)
    return Image.fromarray(np.stack([r, g, b], -1).round().astype(np.uint8), "RGB")


def save(im: Image.Image, path: Path, **kw) -> None:
    buf = io.BytesIO()
    im.save(buf, **kw)
    path.write_bytes(buf.getvalue())


def build(out: Path, cache: Path) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    photos = {name: fetch_photo(name, cache) for name in PHOTOS}
    rows: list[dict] = []

    def add(case_id: str, filename: str, made: str, source: str, license_: str, writer) -> None:
        path = out / filename
        writer(path)
        rows.append({"id": case_id, "file": filename, "made": made, "source": source, "license": license_})

    def photo(name: str) -> Image.Image:
        return Image.open(photos[name])

    for name, (lic, src) in PHOTOS.items():
        stem = name.rsplit(".", 1)[0].replace("_", "-")
        add(
            f"photo-{stem}", name, "copied unchanged", src, lic, lambda p, n=name: p.write_bytes(photos[n].read_bytes())
        )

    mit = "MIT (generated by make_eg2_images.py)"
    gen = "this repository"
    add(
        "synth-gradient",
        "synth-gradient.png",
        "gradient 640x480",
        gen,
        mit,
        lambda p: save(gradient(640, 480), p, format="PNG"),
    )
    add(
        "synth-shapes",
        "synth-shapes.png",
        "shapes 800x600",
        gen,
        mit,
        lambda p: save(shapes(800, 600), p, format="PNG"),
    )
    add(
        "synth-text",
        "synth-text.png",
        "text render 1024x256",
        gen,
        mit,
        lambda p: save(text_render(1024, 256), p, format="PNG"),
    )
    add(
        "synth-checker",
        "synth-checker.png",
        "2-px checkerboard 517x333",
        gen,
        mit,
        lambda p: save(checker(517, 333, 2), p, format="PNG"),
    )
    add(
        "synth-noise",
        "synth-noise.png",
        "uniform RGB noise 300x200",
        gen,
        mit,
        lambda p: save(noise(300, 200, 1), p, format="PNG"),
    )
    add(
        "size-1x1",
        "size-1x1.png",
        "single pixel (upscaled to 768x768)",
        gen,
        mit,
        lambda p: save(Image.new("RGB", (1, 1), (12, 200, 99)), p, format="PNG"),
    )
    add(
        "size-tiny-8x6",
        "size-tiny-8x6.png",
        "8x6 noise (upscaled)",
        gen,
        mit,
        lambda p: save(noise(8, 6, 2), p, format="PNG"),
    )
    add(
        "size-exact-864x672",
        "size-exact-864x672.png",
        "already at the target size (no resize)",
        gen,
        mit,
        lambda p: save(shapes(864, 672), p, format="PNG"),
    )
    add(
        "size-huge-4000x3000",
        "size-huge-4000x3000.png",
        "smooth 4000x3000 (large downscale)",
        gen,
        mit,
        lambda p: save(smooth_large(4000, 3000), p, format="PNG", compress_level=9),
    )
    add(
        "aspect-wide-2000x40",
        "aspect-wide-2000x40.png",
        "2000x40 (height rounds to 0: 48-pixel floor)",
        gen,
        mit,
        lambda p: save(gradient(2000, 40), p, format="PNG"),
    )
    add(
        "aspect-tall-30x1500",
        "aspect-tall-30x1500.png",
        "30x1500 (width rounds to 0)",
        gen,
        mit,
        lambda p: save(checker(30, 1500, 3), p, format="PNG"),
    )
    add(
        "aspect-extreme-4000x8",
        "aspect-extreme-4000x8.png",
        "4000x8 (clamped to the 13,440-pixel max side)",
        gen,
        mit,
        lambda p: save(gradient(4000, 8), p, format="PNG"),
    )

    chelsea = photo("chelsea.png").convert("RGB")
    coffee = photo("coffee.png").convert("RGB")
    cc0 = "CC0-1.0"
    add(
        "mode-gray-L",
        "mode-gray-L.png",
        "camera.png is already L; this is chelsea converted to L",
        PHOTOS["chelsea.png"][1],
        cc0,
        lambda p: save(chelsea.convert("L"), p, format="PNG"),
    )

    def rgba(p: Path) -> None:
        a = np.asarray(coffee).copy()
        alpha = np.tile(np.linspace(0, 255, a.shape[1]).round().astype(np.uint8), (a.shape[0], 1))
        alpha[:, : a.shape[1] // 4] = 0  # fully transparent band whose RGB is still the photo
        save(Image.fromarray(np.dstack([a, alpha]), "RGBA"), p, format="PNG")

    add(
        "mode-rgba",
        "mode-rgba.png",
        "coffee with an alpha ramp and a fully transparent band",
        PHOTOS["coffee.png"][1],
        cc0,
        rgba,
    )

    def la(p: Path) -> None:
        g = np.asarray(chelsea.convert("L"))
        alpha = np.full_like(g, 128)
        save(Image.fromarray(np.dstack([g, alpha]), "LA"), p, format="PNG")

    add("mode-la", "mode-la.png", "chelsea as LA with alpha 128", PHOTOS["chelsea.png"][1], cc0, la)
    add(
        "mode-palette",
        "mode-palette.png",
        "chelsea quantized to a 64-colour adaptive palette (P)",
        PHOTOS["chelsea.png"][1],
        cc0,
        lambda p: save(chelsea.quantize(64, method=Image.Quantize.MEDIANCUT), p, format="PNG"),
    )

    def palette_transparent(p: Path) -> None:
        q = shapes(400, 300).quantize(16, method=Image.Quantize.MEDIANCUT)
        save(q, p, format="PNG", transparency=0)

    add(
        "mode-palette-transparent",
        "mode-palette-transparent.png",
        "shapes, 16-colour palette, index 0 transparent (tRNS)",
        gen,
        mit,
        palette_transparent,
    )
    add(
        "mode-1bit",
        "mode-1bit.png",
        "chelsea dithered to mode 1",
        PHOTOS["chelsea.png"][1],
        cc0,
        lambda p: save(chelsea.convert("1"), p, format="PNG"),
    )

    def gray16(p: Path) -> None:
        g = (np.asarray(chelsea.convert("L")).astype(np.uint16) * 257).astype(np.uint16)
        save(Image.fromarray(g), p, format="PNG")  # uint16 array gives mode I;16

    add(
        "mode-gray16",
        "mode-gray16.png",
        "chelsea as a 16-bit grayscale PNG (I;16)",
        PHOTOS["chelsea.png"][1],
        cc0,
        gray16,
    )
    add(
        "jpeg-cmyk",
        "jpeg-cmyk.jpg",
        "chelsea as a CMYK JPEG (quality 92)",
        PHOTOS["chelsea.png"][1],
        cc0,
        lambda p: save(chelsea.convert("CMYK"), p, format="JPEG", quality=92),
    )
    add(
        "jpeg-420-progressive",
        "jpeg-420-progressive.jpg",
        "shapes as a progressive 4:2:0 JPEG (quality 85)",
        gen,
        mit,
        lambda p: save(shapes(800, 600), p, format="JPEG", quality=85, progressive=True, subsampling=2),
    )
    add(
        "jpeg-444",
        "jpeg-444.jpg",
        "coffee as a baseline 4:4:4 JPEG (quality 95)",
        PHOTOS["coffee.png"][1],
        cc0,
        lambda p: save(coffee, p, format="JPEG", quality=95, subsampling=0),
    )

    def exif_rotated(p: Path) -> None:
        # Stored pixels are the upright photo rotated 90 degrees counter-clockwise; Orientation 6 tells a viewer to
        # rotate them 90 degrees clockwise. Upright display is 451x300, stored is 300x451.
        stored = chelsea.transpose(Image.Transpose.ROTATE_90)
        exif = Image.Exif()
        exif[0x0112] = 6
        save(stored, p, format="JPEG", quality=92, exif=exif.tobytes())

    add(
        "jpeg-exif-orientation",
        "jpeg-exif-orientation.jpg",
        "chelsea stored rotated with EXIF Orientation 6",
        PHOTOS["chelsea.png"][1],
        cc0,
        exif_rotated,
    )

    for row in rows:
        path = out / row["file"]
        data = path.read_bytes()
        im = Image.open(path)
        pixels = np.ascontiguousarray(np.asarray(im))
        row.update(
            {
                "format": im.format,
                "mode": im.mode,
                "width": im.width,
                "height": im.height,
                "exif_orientation": im.getexif().get(0x0112),
                "bytes": len(data),
                "sha256": sha256_bytes(data),
                "decoded_sha256": sha256_bytes(pixels.tobytes()),
                "decoded_dtype": str(pixels.dtype),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--photos", type=Path, help="cache directory for the fetched photos (default: <out>/photos)")
    args = parser.parse_args()
    rows = build(args.out, args.photos or args.out / "photos")
    manifest = {
        "seed": SEED,
        "skimage_commit": SKIMAGE_COMMIT,
        "pillow": Image.__version__,
        "numpy": np.__version__,
        "images": rows,
    }
    (args.out / "images.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    for r in rows:
        print(f"{r['id']:28s} {r['mode']:5s} {r['width']:5d}x{r['height']:<5d} {r['bytes']:9d} {r['license']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
