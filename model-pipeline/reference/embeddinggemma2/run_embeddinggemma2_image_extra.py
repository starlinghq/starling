"""EmbeddingGemma 2 image path: reference evidence for the capabilities the accepted image fixtures leave untested.

Two parts, both on the accepted image set of `make_eg2_images.py` (each file checked against `images.json`):

1. Formats (`image_decode_gif_bmp_webp`, decoder only, no model). GIF, BMP and WebP files are written with Pillow from
   fixture images: palette, transparent, grayscale and two-frame animated GIF; 24-bit, 32-bit with alpha, 8-bit
   palette, 8-bit grayscale and 1-bit BMP; lossless, lossy, lossy with alpha and lossless with alpha WebP. Upstream
   opens each with `Image.open` and the processor converts it with `convert("RGB")`, so the reference is the SHA-256
   of `np.asarray(Image.open(f).convert("RGB"))` (frame 0 for the animated GIF, which is what `Image.open` gives).
   Writes the files into `--formats-out` (not committed; they keep the license of their source image) and
   `image-formats.json` (per file: id, source, how made, file SHA-256, PIL format, mode, size, frames, RGB digest).

2. Budgets (`image_budget_other`). Images embedded alone (no prompt) by sentence-transformers 6.1.0 with the
   processor's `max_soft_tokens` set to 70, 140, 560 and 1,120 (`Gemma4ImageProcessor`, `_SUPPORTED_SOFT_TOKENS`;
   the accepted fixtures cover 280). Rows have the layout of the protected `image-embeddings.jsonl` plus
   `max_soft_tokens`: rendered ids, per image the decoded, resized and soft-token sizes with the digests of the
   resized pixels, pixel values and positions (valid rows), and the fp32 embedding.

Environment: requirements-video.txt (sentence-transformers 6.1.0, transformers 5.19.0, torch 2.14.1 CPU, Pillow 12.3.0).

Usage:
    python run_embeddinggemma2_image_extra.py --model-dir <snapshot> --revision <sha> --images <dir>
        --formats-out <dir> --out <dir> [--threads N] [--skip-budgets]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import torch
from eg2_vision import patchify
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision.transforms.v2 import functional as tvF
from transformers.models.gemma4.image_processing_gemma4 import get_aspect_ratio_preserving_size

REPO = "google/embeddinggemma-2"
PACKAGES = ["torch", "torchvision", "transformers", "sentence-transformers", "tokenizers", "numpy", "pillow"]
BUDGETS = [70, 140, 560, 1120]
BUDGET_IMAGES = [
    "photo-astronaut",
    "photo-coffee",
    "photo-camera",
    "aspect-wide-2000x40",
    "size-tiny-8x6",
    "size-huge-4000x3000",
]
POOL = 3


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_set(images: Path) -> dict[str, dict]:
    manifest = json.loads((images / "images.json").read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in manifest["images"]}
    for r in by_id.values():
        if file_sha(images / r["file"]) != r["sha256"]:
            raise SystemExit(f"{r['file']}: not the fixture image set")
    return by_id


def write_formats(images: Path, by_id: dict[str, dict], out: Path) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)

    def src(image_id: str) -> Image.Image:
        im = Image.open(images / by_id[image_id]["file"])
        im.load()
        return im

    made: list[tuple[str, str, str, str]] = []  # id, file, source, how

    def save(image_id: str, file: str, source: str, how: str, im: Image.Image, **kw) -> None:
        im.save(out / file, **kw)
        made.append((image_id, file, source, how))

    chelsea, coffee, astronaut, rocket = (
        src("photo-chelsea"),
        src("photo-coffee"),
        src("photo-astronaut"),
        src("photo-rocket"),
    )
    save("gif-palette", "gif-palette.gif", "photo-chelsea", "quantize(256, median cut), GIF", chelsea.quantize(256))
    save(
        "gif-transparent",
        "gif-transparent.gif",
        "mode-palette-transparent",
        "palette with transparency index, GIF",
        src("mode-palette-transparent"),
    )
    save("gif-grayscale", "gif-grayscale.gif", "photo-camera", "mode L, GIF", src("photo-camera"))
    frame0, frame1 = coffee.quantize(256), coffee.transpose(Image.Transpose.FLIP_LEFT_RIGHT).quantize(256)
    save(
        "gif-animated-2frames",
        "gif-animated-2frames.gif",
        "photo-coffee",
        "two frames (second mirrored), GIF; frame 0 is used",
        frame0,
        save_all=True,
        append_images=[frame1],
        duration=100,
        loop=0,
    )
    save("bmp-rgb24", "bmp-rgb24.bmp", "photo-astronaut", "mode RGB, 24-bit BMP", astronaut)
    save("bmp-rgba32", "bmp-rgba32.bmp", "mode-rgba", "mode RGBA, 32-bit BMP", src("mode-rgba"))
    save("bmp-palette8", "bmp-palette8.bmp", "mode-palette", "mode P, 8-bit BMP", src("mode-palette"))
    save("bmp-gray8", "bmp-gray8.bmp", "mode-gray-L", "mode L, 8-bit BMP", src("mode-gray-L"))
    save("bmp-1bit", "bmp-1bit.bmp", "mode-1bit", "mode 1, 1-bit BMP", src("mode-1bit"))
    save("webp-lossless", "webp-lossless.webp", "photo-chelsea", "lossless WebP", chelsea, lossless=True, method=4)
    save("webp-lossy", "webp-lossy.webp", "photo-rocket", "lossy WebP quality 80", rocket, quality=80, method=4)
    save(
        "webp-lossy-alpha",
        "webp-lossy-alpha.webp",
        "mode-rgba",
        "lossy WebP with alpha, quality 80",
        src("mode-rgba"),
        quality=80,
        method=4,
    )
    save(
        "webp-lossless-alpha",
        "webp-lossless-alpha.webp",
        "mode-la",
        "mode LA as RGBA, lossless WebP",
        src("mode-la").convert("RGBA"),
        lossless=True,
        method=4,
    )

    records = []
    for image_id, file, source, how in made:
        im = Image.open(out / file)
        rgb = np.asarray(im.convert("RGB"))
        records.append(
            {
                "id": image_id,
                "file": file,
                "source": source,
                "license": by_id[source].get("license"),
                "made": how,
                "sha256": file_sha(out / file),
                "format": im.format,
                "mode": im.mode,
                "width": im.width,
                "height": im.height,
                "frames": getattr(im, "n_frames", 1),
                "rgb_sha256": sha(rgb),
            }
        )
    return records


def budget_rows(model_dir: Path, images: Path, by_id: dict[str, dict]) -> list[dict]:
    st = SentenceTransformer(str(model_dir), device="cpu", model_kwargs={"dtype": torch.float32})
    st.eval()
    processor = st[0].processor
    ip = processor.image_processor
    assert type(ip).__name__ == "Gemma4ImageProcessor" and ip.backend == "torchvision" and ip.max_soft_tokens == 280
    rows = []
    for budget in BUDGETS:
        ip.max_soft_tokens = budget
        for image_id in BUDGET_IMAGES:
            image = Image.open(images / by_id[image_id]["file"])
            features = st.preprocess([image])
            with torch.inference_mode():
                emb = st(features)["sentence_embedding"][0].numpy()
            ids = features["input_ids"][0].tolist()
            pv = features["pixel_values"][0].numpy()
            pos = features["image_position_ids"][0].numpy()
            n = (
                int(features["num_soft_tokens_per_image"][0])
                if "num_soft_tokens_per_image" in features
                else ids.count(258880)
            )
            valid = n * POOL * POOL
            if pv.shape[0] != budget * POOL * POOL:
                raise SystemExit(f"{image_id}@{budget}: padded to {pv.shape[0]} patches, expected {budget * 9}")
            rgb = image if image.mode == "RGB" else image.convert("RGB")
            # The processor restated (as run_embeddinggemma2_image.restate_preprocess) at this budget: must equal it.
            chw = tvF.pil_to_tensor(rgb)
            th, tw = get_aspect_ratio_preserving_size(chw.shape[-2], chw.shape[-1], 16, budget * POOL * POOL, POOL)
            resized = (
                chw
                if (th, tw) == tuple(chw.shape[-2:])
                else tvF.resize(chw, [th, tw], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
            )
            restated, restated_pos, count = patchify(resized * 0.00392156862745098, budget * POOL * POOL, 16)
            if count != n or not (
                np.array_equal(restated.numpy()[:valid], pv[:valid])
                and np.array_equal(restated_pos.numpy()[:valid], pos[:valid])
            ):
                raise SystemExit(f"{image_id}@{budget}: restated preprocessing differs from the processor")
            rows.append(
                {
                    "id": f"{image_id}@{budget}",
                    "kind": "image",
                    "prompt": "none",
                    "prompt_text": "",
                    "text": None,
                    "max_soft_tokens": budget,
                    "images": [
                        {
                            "image": image_id,
                            "decoded": {"mode": image.mode, "width": image.width, "height": image.height},
                            "rgb_sha256": sha(np.asarray(rgb)),
                            "resized": {"height": th, "width": tw},
                            "soft_tokens": n,
                            "patches": valid,
                            "digests": {
                                "resized_uint8": sha(resized.permute(1, 2, 0).numpy()),
                                "pixel_values": sha(pv[:valid]),
                                "positions": sha(pos[:valid]),
                            },
                        }
                    ],
                    "input_ids": ids,
                    "embedding": [float(x) for x in emb.astype(np.float32)],
                    "norm": float(np.linalg.norm(emb.astype(np.float64))),
                }
            )
            print(f"{image_id}@{budget}: {n} soft tokens, {len(ids)} ids, resized {tw}x{th}", flush=True)
    ip.max_soft_tokens = 280
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--formats-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--skip-budgets", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    by_id = load_set(args.images)

    formats = write_formats(args.images, by_id, args.formats_out)
    (args.out / "image-formats.json").write_text(json.dumps({"images": formats}, indent=1) + "\n", encoding="utf-8")
    print(f"{len(formats)} format files", flush=True)

    if not args.skip_budgets:
        rows = budget_rows(args.model_dir, args.images, by_id)
        with (args.out / "image-budget-embeddings.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")

    env = {
        "runner": "model-pipeline/reference/embeddinggemma2/run_embeddinggemma2_image_extra.py",
        "model": {"repo": REPO, "revision": args.revision},
        "assets": {
            a: file_sha(args.model_dir / a)
            for a in ["config.json", "processor_config.json", "tokenizer.json", "model.safetensors"]
        },
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in PACKAGES},
        "cpu": platform.processor() or platform.machine(),
        "threads": args.threads,
        "budgets": BUDGETS,
        "budget_images": BUDGET_IMAGES,
    }
    (args.out / "image-extra-environment.json").write_text(json.dumps(env, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
