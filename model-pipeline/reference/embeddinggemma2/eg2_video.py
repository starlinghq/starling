"""Video path of EmbeddingGemma 2, restated (transformers 5.19.0, sentence-transformers 6.1.0; video-path spike).

Upstream flow for a video file (path string, or a `{"video": path}` dict entry):

1. Decoding: `BaseVideoProcessor.fetch_videos` uses torchcodec (`VideoDecoder(path, seek_mode="exact",
   num_ffmpeg_threads=0)`); if torchcodec is missing it falls back to torchvision, whose video API no longer exists in
   torchvision 0.29 (ImportError). Metadata: `num_frames`, `average_fps`, `duration_seconds` of the stream.
   Frames: `get_frames_at(indices)`, RGB uint8 [T, 3, H, W], rotated by the display matrix.
2. Sampling (`EmbeddingGemma2VideoProcessor.sample_frames`, fps 1, max_frames 32, overflow "uniform"):
   `sample_indices` below. Without fps or duration (a decoded array passed without metadata) every frame is kept and
   only the 32-frame cap applies.
3. Per video (all frames share one size): the image rule for the target size with `max_soft_tokens` 140
   (`max_patches` 1,260), torchvision `resize(bicubic, antialias=True)` on the uint8 [T, 3, H, W] tensor (skipped
   when the size already matches), x 0.00392156862745098, 16x16 patches padded to 1,260 rows per frame.
4. Placeholders: each `<|video|>` becomes, per frame, `<|image>` + n x `<|video|>` + `<image|>`, concatenated
   with no separator (timestamps off).
5. Model: `get_video_features` runs the vision tower on all frames as one batch and `embed_vision`; the soft tokens
   replace the token embedding at the `<|video|>` positions (id 258884), as for images.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch

VIDEO_TOKEN, BOI, EOI = 258884, 255999, 258882
FPS, MAX_FRAMES, MAX_SOFT_TOKENS, PATCH, POOL = 1, 32, 140, 16, 3
MAX_PATCHES = MAX_SOFT_TOKENS * POOL * POOL  # 1,260
RESCALE = 0.00392156862745098


def pin_torchcodec_ffmpeg(major: int | None = None) -> None:
    """Make torchcodec load the FFmpeg major version given (env `EG2_TORCHCODEC_FFMPEG_MAJOR`, default 8).

    torchcodec 0.17 tries FFmpeg 9, 8, ..., 4 in that order and takes the first whose libraries load, so a system
    FFmpeg 9 wins over the pinned FFmpeg 8 on LD_LIBRARY_PATH. This refuses the other majors' core libraries before
    torchcodec is imported. Must run before importing torchcodec or sentence_transformers.
    """
    major = int(major or os.environ.get("EG2_TORCHCODEC_FFMPEG_MAJOR", "8"))
    original = torch.ops.load_library

    def load_library(path):
        name = os.path.basename(str(path))
        if name.startswith(("libtorchcodec_core", "libtorchcodec_custom_ops")) and not name.split(".")[0].endswith(
            str(major)
        ):
            raise OSError(f"skipped {name}: FFmpeg {major} pinned")
        return original(path)

    torch.ops.load_library = load_library
    import torchcodec._core as core

    torch.ops.load_library = original
    versions = core.get_ffmpeg_library_versions()
    if core.ffmpeg_major_version != major:
        raise SystemExit(f"torchcodec loaded FFmpeg {versions}, expected major {major}")


def sample_indices(total_num_frames: int, fps: float | None, duration: float | None) -> list[int]:
    """`EmbeddingGemma2VideoProcessor.sample_frames` with fps 1, max_frames 32, overflow "uniform", restated."""
    if fps is None or duration is None:
        indices = list(range(total_num_frames))
    else:
        step = fps / FPS
        num_sampled = max(1, int(duration * FPS))
        indices = [min(total_num_frames - 1, int(i * step)) for i in range(num_sampled)]
    if len(indices) > MAX_FRAMES:
        # np.linspace(0, n - 1, 32, dtype=int): i * ((n - 1) / 31) in float64, last value exactly n - 1, floored.
        n = len(indices)
        step = (n - 1) / (MAX_FRAMES - 1)
        picks = [math.floor(i * step) for i in range(MAX_FRAMES - 1)] + [n - 1]
        indices = [indices[p] for p in picks]
    return indices


def target_size(height: int, width: int) -> tuple[int, int]:
    from transformers.models.embedding_gemma2.video_processing_embedding_gemma2 import (
        get_aspect_ratio_preserving_size,
    )

    return get_aspect_ratio_preserving_size(height, width, PATCH, MAX_PATCHES, POOL)


def preprocess(frames_tchw: torch.Tensor) -> dict:
    """Steps 3 of the module docstring for one video: uint8 [T, 3, H, W] -> resized uint8, patches, positions."""
    from torchvision.transforms.v2 import functional as tvF

    t, _, h, w = frames_tchw.shape
    th, tw = target_size(h, w)
    resized = (
        frames_tchw
        if (th, tw) == (h, w)
        else tvF.resize(frames_tchw, [th, tw], interpolation=tvF.InterpolationMode.BICUBIC, antialias=True)
    )
    x = resized * RESCALE
    ph, pw = th // PATCH, tw // PATCH
    patches = x.reshape(t, 3, ph, PATCH, pw, PATCH).permute(0, 2, 4, 3, 5, 1).reshape(t, ph * pw, -1)
    gx, gy = torch.meshgrid(torch.arange(pw), torch.arange(ph), indexing="xy")
    pos = torch.stack([gx, gy], dim=-1).reshape(ph * pw, 2)[None].repeat(t, 1, 1)
    count = ph * pw // (POOL * POOL)
    pad = MAX_PATCHES - ph * pw
    patches_p = torch.nn.functional.pad(patches, (0, 0, 0, pad))
    pos_p = torch.nn.functional.pad(pos, (0, 0, 0, pad), value=-1)
    return {
        "resized": resized,
        "target": (th, tw),
        "pixel_values": patches_p,
        "positions": pos_p,
        "count": count,
    }


def expand(frames: int, count: int) -> list[int]:
    """Token ids that replace one `<|video|>` placeholder (timestamps off)."""
    return ([BOI] + [VIDEO_TOKEN] * count + [EOI]) * frames


def numpy_linspace_indices(n: int) -> list[int]:
    """The upstream expression itself, for checking `sample_indices`."""
    return np.linspace(0, n - 1, MAX_FRAMES, dtype=int).tolist()
