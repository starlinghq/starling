"""torchcodec's decoded RGB of the EmbeddingGemma 2 fixture videos on one FFmpeg major (decoder reference, no model).

The decoder profile of `embeddinggemma-2/video-fp32-*` binds each FFmpeg adapter to torchcodec's RGB on the same FFmpeg
major (`decoder_ffmpeg.*` for FFmpeg 8; `decoder_ffmpeg9.*` proposed for FFmpeg 9). The protected `video-frames.json`
holds torchcodec on FFmpeg 8. This runs torchcodec 0.17.0 exactly as upstream (and `run_embeddinggemma2_video.py`'s
`decode_video`) calls it on the major `EG2_TORCHCODEC_FFMPEG_MAJOR` selects (`eg2_video.pin_torchcodec_ffmpeg`): per
video the metadata, upstream's sampled indices and the SHA-256 of every decoded RGB frame (H x W x 3 uint8), in the
format of `video-frames.json`, and compares each value with the protected file.

Environment: requirements-video.txt (torchcodec 0.17.0, torch 2.14.1 CPU). FFmpeg 9: the machine's FFmpeg 9 shared
libraries on the loader path (EG2_TORCHCODEC_FFMPEG_MAJOR=9). Videos: the set of `make_eg2_videos.py`, each checked
against the SHA-256 in the protected `videos.json`.

Writes `--out` (the frames file, `video-frames.json` layout plus the FFmpeg versions) and `--summary` (the
comparison with the protected file: per video equal or the differing fields and frames).

With `--manifest <json>` (a `videos.json`-layout file of other videos, for example `edge-videos.json` of
`make_eg2_edge_videos.py`), the videos of that manifest are decoded and written in the same layout; there is no
protected file to compare with, so the summary lists each video's frame count only.

Usage:
    EG2_TORCHCODEC_FFMPEG_MAJOR=9 python torchcodec_eg2_frames.py --videos <dir> --out <json> --summary <json>
        [--manifest <json>]
"""

from __future__ import annotations

import eg2_video

eg2_video.pin_torchcodec_ffmpeg()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchcodec  # noqa: E402
import torchcodec._core as tc_core  # noqa: E402
from torchcodec.decoders import VideoDecoder  # noqa: E402

PROTECTED = Path(__file__).resolve().parents[2] / "protected" / "embeddinggemma-2" / "video-fp32-cpu"


def sha(a) -> str:
    a = a.numpy() if isinstance(a, torch.Tensor) else a
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def frames_of(path: Path) -> dict:
    """As `run_embeddinggemma2_video.decode_video`: metadata, sampled indices and every frame's digest."""
    d = VideoDecoder(str(path), seek_mode="exact", num_ffmpeg_threads=0)
    m = d.metadata
    indices = eg2_video.sample_indices(m.num_frames, m.average_fps, m.duration_seconds)
    digests = []
    for s in range(0, m.num_frames, 64):
        batch = d.get_frames_in_range(s, min(s + 64, m.num_frames)).data
        digests += [sha(f.permute(1, 2, 0)) for f in batch]
    return {
        "metadata": {
            "num_frames": m.num_frames,
            "average_fps": m.average_fps,
            "duration_seconds": m.duration_seconds,
            "begin_stream_seconds": m.begin_stream_seconds,
            "end_stream_seconds": m.end_stream_seconds,
            "width": m.width,
            "height": m.height,
            "codec": m.codec,
            "rotation": getattr(m, "rotation", None),
        },
        "sampled_indices": indices,
        "all_frame_sha256": digests,
    }


def decode_manifest(args) -> int:
    """The frames file for the videos of `--manifest` (each checked against its SHA-256), with no comparison."""
    manifest = json.loads(args.manifest.read_text())["videos"]
    frames = {}
    for v in manifest:
        path = args.videos / v["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != v["sha256"]:
            raise SystemExit(f"{path}: SHA-256 {digest}, expected {v['sha256']} ({args.manifest.name})")
        frames[v["id"]] = frames_of(path)
        print(v["id"], len(frames[v["id"]]["all_frame_sha256"]), frames[v["id"]]["metadata"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(frames, indent=1) + "\n")
    summary = {
        "decoder": f"torchcodec {torchcodec.__version__} (VideoDecoder seek_mode exact, num_ffmpeg_threads 0, CPU)",
        "ffmpeg": tc_core.get_ffmpeg_library_versions(),
        "ffmpeg_major_pinned": int(os.environ.get("EG2_TORCHCODEC_FFMPEG_MAJOR", "8")),
        "manifest": args.manifest.name,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "frames_file_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
        "videos": {k: len(v["all_frame_sha256"]) for k, v in frames.items()},
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=1) + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--manifest", type=Path, help="decode the videos of this manifest instead (no comparison)")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.manifest:
        return decode_manifest(args)
    manifest = json.loads((PROTECTED / "videos.json").read_text())["videos"]
    protected = json.loads((PROTECTED / "video-frames.json").read_text())
    versions = tc_core.get_ffmpeg_library_versions()
    frames, per_video = {}, {}
    total = equal_frames = 0
    for v in manifest:
        path = args.videos / v["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != v["sha256"]:
            raise SystemExit(f"{path}: SHA-256 {digest}, expected {v['sha256']} (videos.json)")
        got = frames_of(path)
        frames[v["id"]] = got
        want = protected[v["id"]]
        differing = [
            i for i, (a, b) in enumerate(zip(got["all_frame_sha256"], want["all_frame_sha256"], strict=False)) if a != b
        ]
        same_count = len(got["all_frame_sha256"]) == len(want["all_frame_sha256"])
        total += len(want["all_frame_sha256"])
        equal_frames += len(want["all_frame_sha256"]) - len(differing) if same_count else 0
        per_video[v["id"]] = {
            "frames": len(got["all_frame_sha256"]),
            "metadata_equal": got["metadata"] == want["metadata"],
            "sampled_indices_equal": got["sampled_indices"] == want["sampled_indices"],
            "frame_count_equal": same_count,
            "differing_frames": differing,
        }
        print(v["id"], per_video[v["id"]], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(frames, indent=1) + "\n")
    out_sha = hashlib.sha256(args.out.read_bytes()).hexdigest()
    summary = {
        "decoder": f"torchcodec {torchcodec.__version__} (VideoDecoder seek_mode exact, num_ffmpeg_threads 0, CPU)",
        "ffmpeg": versions,
        "ffmpeg_major_pinned": int(os.environ.get("EG2_TORCHCODEC_FFMPEG_MAJOR", "8")),
        "compared_with": "model-pipeline/protected/embeddinggemma-2/video-fp32-cpu/video-frames.json "
        "(torchcodec 0.17.0 on FFmpeg n8.1.3-17-gc389840894)",
        "protected_sha256": hashlib.sha256((PROTECTED / "video-frames.json").read_bytes()).hexdigest(),
        "frames_file_sha256": out_sha,
        "videos": len(per_video),
        "frames": total,
        "equal_frames": equal_frames,
        "all_equal": all(
            p["metadata_equal"] and p["sampled_indices_equal"] and p["frame_count_equal"] and not p["differing_frames"]
            for p in per_video.values()
        ),
        "per_video": per_video,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_video"}, indent=1))
    return 0 if summary["all_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
