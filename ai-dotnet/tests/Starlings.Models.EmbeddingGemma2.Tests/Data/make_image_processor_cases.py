"""Small, weight-free reference cases for the EmbeddingGemma 2 image preprocessing port (C# unit tests, run in CI).

Writes `image-processor-cases.json` next to this script:

- `resize`: random uint8 RGB images (NumPy PCG64, fixed seed) resized by
  `torchvision.transforms.v2.functional.resize(uint8 CHW tensor, [h, w], BICUBIC, antialias=True)` on CPU, the call
  the transformers 5.19.0 torchvision image backend makes. Each case stores the input bytes (base64, HWC), the target
  size and the SHA-256 of the output (HWC uint8). Sizes cover downscaling up to 9x, upscaling, one-axis resizes and
  thin images.
- `target_size`: `transformers.models.gemma4.image_processing_gemma4.get_aspect_ratio_preserving_size` for image sizes
  and every supported budget (70, 140, 280, 560, 1,120 soft tokens).
- `patches`: for two small resized images, the SHA-256 of the processor's `pixel_values` (float32, `uint8 *
  0.00392156862745098`, patch order) and positions (int64).

Environment: `model-pipeline/reference/embeddinggemma2/requirements.txt` (torch 2.14.1 CPU, torchvision 0.29.1,
transformers 5.19.0). No model weights are read.

Usage:
    python make_image_processor_cases.py
"""

from __future__ import annotations

import base64
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch
import torchvision
import transformers
from torchvision.transforms.v2 import functional as tvF
from transformers.models.gemma4.image_processing_gemma4 import get_aspect_ratio_preserving_size

SEED = 20261012
RESIZE = [
    # (width, height, target width, target height)
    (37, 23, 11, 7),
    (64, 48, 7, 5),
    (90, 60, 10, 6),
    (31, 29, 48, 48),
    (8, 6, 96, 48),
    (5, 3, 61, 40),
    (40, 17, 40, 9),
    (13, 50, 13, 77),
    (200, 3, 48, 48),
    (3, 150, 48, 96),
    (101, 99, 48, 48),
    (17, 19, 144, 96),
    (255, 64, 96, 48),
    (1, 1, 48, 48),
    (2, 2, 3, 5),
]
TARGET = [
    (451, 300), (600, 400), (512, 512), (640, 427), (1000, 872), (1, 1), (8, 6), (864, 672), (4000, 3000),
    (2000, 40), (30, 1500), (4000, 8), (8, 4000), (7, 1), (1, 7), (1920, 1080), (1080, 1920), (100000, 2),
    (3, 100000), (48, 48), (47, 49), (12345, 678), (2520, 1), (803, 803),
]
BUDGETS = [70, 140, 280, 560, 1120]


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main() -> int:
    rng = np.random.Generator(np.random.PCG64(SEED))
    resize = []
    for w, h, tw, th in RESIZE:
        img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        chw = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        out = tvF.resize(chw, [th, tw], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
        out = out.permute(1, 2, 0).contiguous().numpy()
        resize.append(
            {
                "width": w,
                "height": h,
                "target_width": tw,
                "target_height": th,
                "input": base64.b64encode(img.tobytes()).decode(),
                "output_sha256": sha(out),
            }
        )
    target = []
    for w, h in TARGET:
        for budget in BUDGETS:
            try:
                th, tw = get_aspect_ratio_preserving_size(h, w, 16, budget * 9, 3)
                target.append({"width": w, "height": h, "budget": budget, "target_width": tw, "target_height": th})
            except ValueError as exc:
                target.append({"width": w, "height": h, "budget": budget, "error": str(exc)})
    patches = []
    for w, h in ((48, 48), (96, 48)):
        img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        chw = torch.from_numpy(img).permute(2, 0, 1)
        rescaled = chw * 0.00392156862745098
        c = 3
        ph, pw = h // 16, w // 16
        pv = rescaled.reshape(c, ph, 16, pw, 16).permute(1, 3, 2, 4, 0).reshape(ph * pw, -1)
        gx, gy = torch.meshgrid(torch.arange(pw), torch.arange(ph), indexing="xy")
        pos = torch.stack([gx, gy], dim=-1).reshape(ph * pw, 2)
        patches.append(
            {
                "width": w,
                "height": h,
                "input": base64.b64encode(img.tobytes()).decode(),
                "pixel_values_dtype": str(pv.numpy().dtype),
                "pixel_values_sha256": sha(pv.numpy()),
                "positions_sha256": sha(pos.numpy()),
            }
        )
    doc = {
        "generator": "make_image_processor_cases.py",
        "seed": SEED,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "machine": platform.machine(),
        "resize": resize,
        "target_size": target,
        "patches": patches,
    }
    out = Path(__file__).with_name("image-processor-cases.json")
    out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(out, len(resize), len(target), len(patches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
