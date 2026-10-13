"""Decode the spike's videos with one decoder and compare the RGB frames with the reference (torchcodec + FFmpeg 8.1).

Decoders (`--decoder`):

- `torchcodec`: torchcodec 0.17.0 as upstream calls it, on whatever FFmpeg major `EG2_TORCHCODEC_FFMPEG_MAJOR` selects
  (8 with the pinned LGPL build on LD_LIBRARY_PATH; 9 for the system FFmpeg 9.0.1 of this machine).
- `pyav`: PyAV 18.1.0 (bundled FFmpeg), `frame.to_ndarray(format="rgb24")`.
- `opencv`: OpenCV 5.0.0 headless (bundled FFmpeg), `cv2.VideoCapture`, BGR to RGB.
- `ffmpeg-cli`: the pinned LGPL `ffmpeg` CLI, `-pix_fmt rgb24 -f rawvideo` (autorotate on, as by default).
- `numpy-nearest`, `numpy-bilinear`: decoded Y'CbCr from PyAV (native pixel format; checked against the generator's
  `framemd5`), converted in float64 with the tagged matrix and range (BT.601 when untagged), chroma replicated per 2x2
  block (nearest) or interpolated (bilinear, centred), rounded; the display matrix applied with `np.rot90`. This is
  what a managed C# converter would compute.
- `external:<dir>`: frames written by another program (the C# prototype) as `<dir>/<video id>.rgb` (uint8, T x H x W
  x 3 at the reference indices) with `<dir>/index.json` (`{video id: {"width", "height", "frames"}}`).

Writes `<out>/<name>/<video id>.npy` (frames at the reference's sampled indices) and `<out>/<name>.json`: per video
the decoder's own metadata and sampled indices, and per frame the largest level difference and the fraction of
differing bytes against the reference frames.

Usage:
    python decode_eg2_videos.py --decoder <name> --videos <dir> --reference-tensors <dir> --reference <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

MATRIX = {"bt709": (0.2126, 0.0722)}  # everything else (bt470bg, smpte170m, unknown/untagged): BT.601


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def decode_torchcodec(path: Path, indices: list[int]) -> tuple[np.ndarray, dict]:
    import eg2_video

    eg2_video.pin_torchcodec_ffmpeg()
    import torchcodec._core as core
    from torchcodec.decoders import VideoDecoder

    d = VideoDecoder(str(path), seek_mode="exact", num_ffmpeg_threads=0)
    m = d.metadata
    frames = d.get_frames_at(indices=indices).data.permute(0, 2, 3, 1).numpy()
    meta = {"num_frames": m.num_frames, "fps": m.average_fps, "duration": m.duration_seconds}
    meta["ffmpeg"] = core.get_ffmpeg_library_versions()["ffmpeg_version"]
    return frames, meta


def pyav_frames(path: Path):
    import av

    container = av.open(str(path))
    stream = container.streams.video[0]
    meta = {
        "num_frames": stream.frames,
        "fps": float(stream.average_rate) if stream.average_rate else None,
        "duration": float(stream.frames / stream.average_rate) if stream.average_rate else None,
        "ffmpeg": ".".join(map(str, av.library_versions["libavcodec"])),
    }
    return container, stream, meta


def decode_pyav(path: Path, indices: list[int]) -> tuple[np.ndarray, dict]:
    container, _, meta = pyav_frames(path)
    want, out = set(indices), {}
    for i, frame in enumerate(container.decode(video=0)):
        if i in want:
            out[i] = frame.to_ndarray(format="rgb24")
        if i >= max(indices):
            break
    container.close()
    return np.stack([out[i] for i in indices]), meta


def decode_opencv(path: Path, indices: list[int]) -> tuple[np.ndarray, dict]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    meta = {
        "num_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "orientation_auto": cap.get(cv2.CAP_PROP_ORIENTATION_AUTO),
        "opencv": cv2.__version__,
    }
    want, out, i = set(indices), {}, 0
    while i <= max(indices):
        ok, bgr = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = bgr[..., ::-1].copy()
        i += 1
    cap.release()
    return np.stack([out[i] for i in indices]), meta


def decode_cli(path: Path, indices: list[int], ffmpeg: str, width: int, height: int) -> tuple[np.ndarray, dict]:
    raw = subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(path), "-map", "0:v:0", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        check=True,
        capture_output=True,
    ).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)
    return frames[indices].copy(), {"num_frames": frames.shape[0]}


def yuv_planes(frame) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    fmt = frame.format.name
    bits = 10 if "10" in fmt else 8
    dtype = np.uint16 if bits > 8 else np.uint8
    planes = []
    for k, p in enumerate(frame.planes):
        w = frame.width if k == 0 else (frame.width + 1) // 2
        h = frame.height if k == 0 else (frame.height + 1) // 2
        a = np.frombuffer(p, dtype).reshape(h, p.line_size // np.dtype(dtype).itemsize)[:, :w]
        planes.append(a)
    return planes[0], planes[1], planes[2], bits


def convert(y, cb, cr, bits: int, matrix: str, full: bool, mode: str) -> np.ndarray:
    kr, kb = MATRIX.get(matrix, (0.299, 0.114))
    kg = 1 - kr - kb
    s = float(1 << (bits - 8))
    h, w = y.shape
    if mode == "nearest":
        cbu = np.repeat(np.repeat(cb, 2, 0), 2, 1)[:h, :w]
        cru = np.repeat(np.repeat(cr, 2, 0), 2, 1)[:h, :w]
    else:  # bilinear, chroma sample centred on its 2x2 block

        def up(c):
            c = c.astype(np.float64)
            ys = np.clip((np.arange(h) + 0.5) / 2 - 0.5, 0, c.shape[0] - 1)
            xs = np.clip((np.arange(w) + 0.5) / 2 - 0.5, 0, c.shape[1] - 1)
            y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
            y1, x1 = np.minimum(y0 + 1, c.shape[0] - 1), np.minimum(x0 + 1, c.shape[1] - 1)
            fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
            top = c[y0][:, x0] * (1 - fx) + c[y0][:, x1] * fx
            bot = c[y1][:, x0] * (1 - fx) + c[y1][:, x1] * fx
            return top * (1 - fy) + bot * fy

        cbu, cru = up(cb), up(cr)
    yf = y.astype(np.float64) / s
    cbf, crf = cbu.astype(np.float64) / s - 128, cru.astype(np.float64) / s - 128
    if full:
        yn, pb, pr = yf / 255, cbf / 255, crf / 255
    else:
        yn, pb, pr = (yf - 16) / 219, cbf / 224, crf / 224
    r = yn + 2 * (1 - kr) * pr
    b = yn + 2 * (1 - kb) * pb
    g = (yn - kr * r - kb * b) / kg
    rgb = np.stack([r, g, b], -1) * 255
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8)


def decode_numpy(path: Path, indices: list[int], mode: str, record: dict) -> tuple[np.ndarray, dict]:
    container, stream, meta = pyav_frames(path)
    probe = record["probe"]
    matrix = probe.get("color_space", "unknown")
    full = probe.get("color_range") == "pc"
    rot = int(probe.get("display_rotation", 0) or 0)
    want, out, md5_ok = set(indices), {}, True
    for i, frame in enumerate(container.decode(video=0)):
        y, cb, cr, bits = yuv_planes(frame)
        digest = hashlib.md5(b"".join(np.ascontiguousarray(p).tobytes() for p in (y, cb, cr))).hexdigest()
        md5_ok &= digest == record["decoded_native_framemd5"][i]
        if i in want:
            rgb = convert(y, cb, cr, bits, matrix, full, mode)
            # FFmpeg's display rotation is counter-clockwise; np.rot90 with k = rotation / 90 rotates the same way.
            out[i] = np.ascontiguousarray(np.rot90(rgb, k=rot // 90)) if rot else rgb
    container.close()
    meta.update({"matrix": matrix, "full_range": full, "rotation": rot, "native_framemd5_equal": md5_ok})
    return np.stack([out[i] for i in indices]), meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--name", help="output name (default: the decoder, or torchcodec-ffmpeg<major>)")
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="runner --out (video-frames.json)")
    parser.add_argument("--reference-tensors", type=Path, required=True, help="runner --tensors")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.videos / "videos.json").read_text())
    ref = json.loads((args.reference / "video-frames.json").read_text())
    name = args.name or args.decoder
    if args.decoder == "torchcodec" and not args.name:
        name = "torchcodec-ffmpeg" + os.environ.get("EG2_TORCHCODEC_FFMPEG_MAJOR", "8")
    if args.decoder.startswith("external:"):
        ext_dir = Path(args.decoder.split(":", 1)[1])
        ext_index = json.loads((ext_dir / "index.json").read_text())
        name = args.name or ext_dir.name
    out_dir = args.out / name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"decoder": args.decoder, "videos": {}}
    for record in manifest["videos"]:
        vid = record["id"]
        path = args.videos / record["file"]
        indices = ref[vid]["sampled_indices"]
        reference = np.load(args.reference_tensors / f"{vid}.npz")["video0_frames_uint8"]
        entry: dict = {}
        try:
            if args.decoder == "torchcodec":
                frames, meta = decode_torchcodec(path, indices)
            elif args.decoder == "pyav":
                frames, meta = decode_pyav(path, indices)
            elif args.decoder == "opencv":
                frames, meta = decode_opencv(path, indices)
            elif args.decoder == "ffmpeg-cli":
                h, w = reference.shape[1:3]
                frames, meta = decode_cli(path, indices, args.ffmpeg, w, h)
            elif args.decoder in ("numpy-nearest", "numpy-bilinear"):
                frames, meta = decode_numpy(path, indices, args.decoder.split("-")[1], record)
            elif args.decoder.startswith("external:"):
                info = ext_index[vid]
                raw = np.fromfile(ext_dir / f"{vid}.rgb", np.uint8)
                frames = raw.reshape(info["frames"], info["height"], info["width"], 3)
                meta = {k: v for k, v in info.items() if k not in ("width", "height", "frames")}
            else:
                raise SystemExit(f"unknown decoder {args.decoder}")
        except Exception as e:  # noqa: BLE001 - record and continue
            summary["videos"][vid] = {"error": f"{type(e).__name__}: {e}"}
            print(vid, "ERROR", e, flush=True)
            continue
        entry["metadata"] = meta
        if "num_frames" in meta and meta.get("fps") and meta.get("duration") is not None:
            import eg2_video

            own = eg2_video.sample_indices(int(meta["num_frames"]), float(meta["fps"]), float(meta["duration"]))
            entry["own_sampled_indices_equal"] = own == indices
        if frames.shape != reference.shape:
            entry["shape"] = list(frames.shape)
            entry["reference_shape"] = list(reference.shape)
            entry["equal"] = False
            print(vid, "shape", frames.shape, "vs", reference.shape, flush=True)
        else:
            d = np.abs(frames.astype(np.int16) - reference.astype(np.int16))
            entry.update(
                {
                    "equal": bool(d.max() == 0),
                    "max_level_diff": int(d.max()),
                    "mean_level_diff": float(d.mean()),
                    "differing_fraction": float((d > 0).mean()),
                    "per_frame_max": [int(x) for x in d.reshape(d.shape[0], -1).max(axis=1)],
                    "frame_sha256_equal_reference": [
                        sha(f) == ref[vid]["all_frame_sha256"][i] for f, i in zip(frames, indices, strict=True)
                    ],
                }
            )
            np.save(out_dir / f"{vid}.npy", frames)
            print(vid, "max", entry["max_level_diff"], "frac", round(entry["differing_fraction"], 4), flush=True)
        summary["videos"][vid] = entry
    (args.out / f"{name}.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
