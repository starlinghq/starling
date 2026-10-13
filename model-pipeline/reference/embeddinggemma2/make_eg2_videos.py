"""Synthetic video set for the EmbeddingGemma 2 video-path spike, generated from a seed.

Nothing produced here is committed. Every frame is drawn with NumPy (PCG64 from `SEED`) and Pillow's built-in bitmap
font, converted to Y'CbCr in NumPy with exact float64 formulas (so the encoder input does not depend on swscale or the
CPU), and piped as raw `yuv420p` or `yuv420p10le` to a pinned FFmpeg CLI (`fetch_eg2_ffmpeg.sh`, GPL build:
libx264, libx265, libvpx-vp9, libaom-av1), single-threaded for deterministic output. The videos are original works
of this repository (MIT, like the rest of `model-pipeline/`).

Scenes: colour bars with a moving square and a frame counter (`bars`); moving circles and rectangles on a background
whose hue changes every second (`shapes`); scrolling text over a gradient (`text`). Every frame carries its index in
large digits, so a wrong sampled index shows in the pixels.

`videos.json` records per video: id, file, the generation parameters, the exact FFmpeg command, the file's SHA-256,
the SHA-256 of the raw encoder input, `ffprobe` stream facts, and the decoder ground truth: FFmpeg's `framemd5` of the
decoded frames in their native pixel format (H.264, HEVC, VP9 and AV1 decoding is bit-exact by specification, so this
pins the decoded Y'CbCr independently of any RGB conversion).

Usage:
    python make_eg2_videos.py --ffmpeg-bin <dir with ffmpeg/ffprobe of the GPL build> --out <dir>
        [--decoder-bin <dir with the LGPL ffmpeg for framemd5>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SEED = 20261011
FFMPEG_BUILD = "BtbN/FFmpeg-Builds autobuild-2026-10-11-13-04, FFmpeg n8.1.3-17-gc389840894"

# Kr, Kb of the two matrices used here.
MATRICES = {"bt601": (0.299, 0.114), "bt709": (0.2126, 0.0722)}
# FFmpeg colour tags per matrix (none = untagged stream).
TAGS = {
    "bt601": ["-colorspace", "bt470bg", "-color_primaries", "bt470bg", "-color_trc", "smpte170m"],
    "bt709": ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"],
    "none": [],
}


@dataclass
class Spec:
    id: str
    codec: str  # h264 | hevc | vp9 | av1
    width: int
    height: int
    fps: str  # rational, e.g. "30" or "30000/1001"
    frames: int
    scene: str = "shapes"
    matrix: str = "bt601"  # bt601 | bt709 | none (untagged; encoded with BT.601)
    full_range: bool = False
    bits: int = 8
    rotation: int = 0  # display rotation side data, degrees counter-clockwise (FFmpeg -display_rotation)
    note: str = ""
    extra: list = field(default_factory=list)

    @property
    def container(self) -> str:
        return "webm" if self.codec == "vp9" else "mp4"

    @property
    def file(self) -> str:
        return f"{self.id}.{self.container}"


SPECS = [
    Spec("h264-bars-640x360-30fps-5s", "h264", 640, 360, "30", 150, "bars", note="baseline; BT.601 limited"),
    Spec("h264-shapes-1280x720-25fps-10s", "h264", 1280, 720, "25", 250, "shapes", "bt709", note="BT.709 tagged"),
    Spec("h264-long-320x240-30fps-40s", "h264", 320, 240, "30", 1200, "shapes", note="> 32 s: uniform overflow"),
    Spec("h264-32s-160x120-30fps", "h264", 160, 120, "30", 960, "bars", note="exactly 32 sampled frames"),
    Spec("h264-33s-160x120-30fps", "h264", 160, 120, "30", 990, "bars", note="33 -> 32 (uniform)"),
    Spec("h264-short-320x240-24fps-0.5s", "h264", 320, 240, "24", 12, "shapes", note="< 1 s"),
    Spec("h264-1s-320x240-30fps", "h264", 320, 240, "30", 30, "shapes", note="exactly 1 s"),
    Spec("h264-1.9s-320x240-30fps", "h264", 320, 240, "30", 57, "shapes", note="int(duration) truncation"),
    Spec("h264-ntsc-640x480-29.97fps-12s", "h264", 640, 480, "30000/1001", 360, "text", note="fractional rate"),
    Spec("h264-60fps-480x270-3s", "h264", 480, 270, "60", 180, "shapes"),
    Spec("h264-1fps-64x48-8s", "h264", 64, 48, "1", 8, "bars", note="1 fps source, tiny (upscaled)"),
    Spec("h264-portrait-360x640-30fps-4s", "h264", 360, 640, "30", 120, "text", note="width not a multiple of 32"),
    Spec("h264-330x186-30fps-3s", "h264", 330, 186, "30", 90, "shapes", note="width not a multiple of 32"),
    Spec("h264-rot90-640x360-30fps-3s", "h264", 640, 360, "30", 90, "text", rotation=90, note="display matrix"),
    Spec("h264-fullrange-640x360-30fps-3s", "h264", 640, 360, "30", 90, "bars", full_range=True, note="pc range"),
    Spec("h264-untagged-640x360-30fps-3s", "h264", 640, 360, "30", 90, "bars", "none", note="no colour tags"),
    Spec("h264-10bit-640x360-30fps-3s", "h264", 640, 360, "30", 90, "bars", bits=10, note="High 10"),
    Spec("h264-1080p-1920x1080-30fps-4s", "h264", 1920, 1080, "30", 120, "shapes", "bt709", note="downscaled"),
    Spec("vp9-640x360-30fps-6s", "vp9", 640, 360, "30", 180, "shapes", note="WebM"),
    Spec("vp9-odd-321x179-30fps-3s", "vp9", 321, 179, "30", 90, "bars", note="odd width and height"),
    Spec("hevc-640x360-30fps-4s", "hevc", 640, 360, "30", 120, "shapes", "bt709", note="HEVC Main"),
    Spec("av1-640x360-30fps-3s", "av1", 640, 360, "30", 90, "text", note="AV1 in MP4"),
]

_FONT = None


def font() -> ImageFont.ImageFont:
    global _FONT
    if _FONT is None:
        _FONT = ImageFont.load_default_imagefont()  # Pillow's built-in bitmap font: no FreeType, no font file
    return _FONT


def label(text: str, scale: int) -> np.ndarray:
    """Black-on-white bitmap text, upscaled by nearest neighbour; returns a bool mask (True = ink)."""
    d = ImageDraw.Draw(Image.new("L", (1, 1)))
    w = int(np.ceil(d.textlength(text, font=font()))) + 2
    small = Image.new("L", (w, 13), 0)
    ImageDraw.Draw(small).text((1, 1), text, font=font(), fill=255)
    big = small.resize((w * scale, 13 * scale), Image.Resampling.NEAREST)
    return np.asarray(big) > 127


def stamp(frame: np.ndarray, mask: np.ndarray, x: int, y: int, color) -> None:
    h, w = frame.shape[:2]
    mh, mw = mask.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + mw, w), min(y + mh, h)
    if x1 <= x0 or y1 <= y0:
        return
    sub = mask[y0 - y : y1 - y, x0 - x : x1 - x]
    frame[y0:y1, x0:x1][sub] = color


def hsv(h: float, s: float, v: float) -> tuple[int, int, int]:
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def scene_params(spec: Spec) -> dict:
    g = np.random.Generator(np.random.PCG64([SEED, sum(map(ord, spec.id))]))
    n = 6
    return {
        "pos": g.uniform(0, 1, (n, 2)),
        "vel": g.uniform(-0.35, 0.35, (n, 2)),
        "size": g.uniform(0.06, 0.22, n),
        "hue": g.uniform(0, 1, n),
        "kind": g.integers(0, 2, n),
        "words": " ".join(g.choice(["red", "green", "blue", "video", "frame", "embed", "gemma", "clip"], 24)),
    }


def draw(spec: Spec, params: dict, i: int, fps: float) -> np.ndarray:
    w, h = spec.width, spec.height
    t = i / fps
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    if spec.scene == "bars":
        bars = np.array(
            [[192, 192, 192], [192, 192, 0], [0, 192, 192], [0, 192, 0], [192, 0, 192], [192, 0, 0], [0, 0, 192]],
            np.uint8,
        )
        frame = bars[np.minimum((xx * 7 // w).astype(int), 6)]
        low = yy >= h * 2 // 3
        ramp = np.clip(xx / max(w - 1, 1) * 255, 0, 255).astype(np.uint8)
        frame[low] = np.stack([ramp, ramp, ramp], -1)[low]
        s = max(4, h // 6)
        x = int((t * 0.4 % 1.0) * (w - s))
        frame[h // 3 : h // 3 + s, x : x + s] = (255, 255, 255)
    elif spec.scene == "shapes":
        frame = np.empty((h, w, 3), np.uint8)
        frame[:] = hsv((int(t) * 0.13) % 1.0, 0.5, 0.35)
        for k in range(len(params["size"])):
            p = (params["pos"][k] + params["vel"][k] * t) % 1.0
            cx, cy, r = p[0] * w, p[1] * h, params["size"][k] * min(w, h)
            if params["kind"][k] == 0:
                m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            else:
                m = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r * 0.6)
            frame[m] = hsv(params["hue"][k], 0.9, 0.95)
    elif spec.scene == "text":
        frame = np.stack(
            [(xx / max(w - 1, 1) * 200 + 30), (yy / max(h - 1, 1) * 150 + 50), np.full_like(xx, 110)], -1
        ).astype(np.uint8)
        scale = max(1, h // 120)
        words = label(params["words"][:40], scale)
        stamp(frame, words, int(w - (t * 60 * scale) % (w + words.shape[1])), h // 2, (250, 250, 250))
    else:
        raise ValueError(spec.scene)
    digits = label(f"{i:04d}", max(1, min(w, h) // 40))
    stamp(frame, digits, 2, 2, (0, 0, 0))
    stamp(frame, digits, 1, 1, (255, 255, 0))
    return frame


def to_yuv420(rgb: np.ndarray, matrix: str, full_range: bool, bits: int) -> bytes:
    """Exact float64 R'G'B' -> Y'CbCr (BT.601 or BT.709), chroma = mean of each 2x2 block (edge pixels repeated)."""
    kr, kb = MATRICES["bt601" if matrix == "none" else matrix]
    kg = 1.0 - kr - kb
    x = rgb.astype(np.float64) / 255.0
    h, w = rgb.shape[:2]
    y = kr * x[..., 0] + kg * x[..., 1] + kb * x[..., 2]
    ph, pw = h + (h & 1), w + (w & 1)
    xp = np.pad(x, ((0, ph - h), (0, pw - w), (0, 0)), mode="edge")
    avg = xp.reshape(ph // 2, 2, pw // 2, 2, 3).mean(axis=(1, 3))
    ya = kr * avg[..., 0] + kg * avg[..., 1] + kb * avg[..., 2]
    cb = (avg[..., 2] - ya) / (2 * (1 - kb))
    cr = (avg[..., 0] - ya) / (2 * (1 - kr))
    scale = (1 << (bits - 8)) if bits > 8 else 1
    maxv = (1 << bits) - 1
    if full_range:
        yq = np.round(y * maxv)
        cbq, crq = np.round(cb * maxv + (128 * scale)), np.round(cr * maxv + (128 * scale))
    else:
        yq = np.round((16 + 219 * y) * scale)
        cbq, crq = np.round((128 + 224 * cb) * scale), np.round((128 + 224 * cr) * scale)
    dtype = np.uint16 if bits > 8 else np.uint8
    planes = [np.clip(p, 0, maxv).astype(dtype) for p in (yq, cbq, crq)]
    return b"".join(p.astype(dtype.__name__ if bits <= 8 else "<u2").tobytes() for p in planes)


def encoder_args(spec: Spec) -> list[str]:
    if spec.codec == "h264":
        a = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-g", "60", "-threads", "1"]
        a += ["-profile:v", "high10" if spec.bits == 10 else "high"]
    elif spec.codec == "hevc":
        a = ["-c:v", "libx265", "-preset", "fast", "-crf", "22", "-tag:v", "hvc1"]
        a += ["-x265-params", "pools=1:frame-threads=1:wpp=0:log-level=error"]
    elif spec.codec == "vp9":
        a = ["-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0", "-deadline", "good", "-cpu-used", "4"]
        a += ["-row-mt", "0", "-threads", "1", "-g", "60"]
    elif spec.codec == "av1":
        a = ["-c:v", "libaom-av1", "-crf", "32", "-cpu-used", "8", "-row-mt", "0", "-threads", "1", "-g", "60"]
    else:
        raise ValueError(spec.codec)
    return a


def command(spec: Spec, ffmpeg: str, out: Path) -> list[str]:
    pix = "yuv420p10le" if spec.bits == 10 else "yuv420p"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix]
    cmd += ["-s", f"{spec.width}x{spec.height}", "-framerate", spec.fps]
    cmd += ["-i", "pipe:0", *encoder_args(spec), "-pix_fmt", pix]
    cmd += TAGS[spec.matrix]
    if spec.matrix != "none":
        cmd += ["-color_range", "pc" if spec.full_range else "tv"]
    elif spec.full_range:
        raise ValueError("full range needs a tag")
    cmd += ["-fflags", "+bitexact", "-flags:v", "+bitexact", "-map_metadata", "-1"]
    if spec.container == "mp4":
        cmd += ["-movflags", "+faststart"]
    cmd += [*spec.extra, str(out.with_suffix(".tmp." + spec.container) if spec.rotation else out)]
    return cmd


def remux_command(spec: Spec, ffmpeg: str, out: Path) -> list[str]:
    """Stream copy that adds a display matrix (an encode with -display_rotation would rotate the pixels instead)."""
    src = out.with_suffix(".tmp." + spec.container)
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-display_rotation", str(spec.rotation)] + [
        "-i",
        str(src),
        "-c",
        "copy",
        "-fflags",
        "+bitexact",
        "-map_metadata",
        "-1",
        "-movflags",
        "+faststart",
        str(out),
    ]


def probe(ffprobe: str, path: Path) -> dict:
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_streams", "-show_format"]
        + ["-print_format", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    data = json.loads(out)
    s = data["streams"][0]
    keep = [
        "codec_name",
        "profile",
        "pix_fmt",
        "width",
        "height",
        "color_range",
        "color_space",
        "color_transfer",
        "color_primaries",
        "r_frame_rate",
        "avg_frame_rate",
        "time_base",
        "nb_frames",
        "nb_read_frames",
        "duration",
        "start_time",
    ]
    rec = {k: s.get(k) for k in keep if k in s}
    rot = [sd.get("rotation") for sd in s.get("side_data_list", []) if "rotation" in sd]
    if rot:
        rec["display_rotation"] = rot[0]
    rec["format_duration"] = data["format"].get("duration")
    return rec


def framemd5(ffmpeg: str, path: Path) -> list[str]:
    """MD5 of every decoded frame in its native pixel format (decoder ground truth)."""
    out = subprocess.run(
        [ffmpeg, "-v", "error", "-noautorotate", "-i", str(path), "-map", "0:v:0", "-f", "framemd5", "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line.rsplit(",", 1)[1].strip() for line in out.splitlines() if line and not line.startswith("#")]


def build(out: Path, ffmpeg_bin: Path, decoder_bin: Path) -> list[dict]:
    ffmpeg, ffprobe = str(ffmpeg_bin / "ffmpeg"), str(ffmpeg_bin / "ffprobe")
    records = []
    for spec in SPECS:
        path = out / spec.file
        fps = float(Fraction(spec.fps))
        params = scene_params(spec)
        cmd = command(spec, ffmpeg, path)
        raw = hashlib.sha256()
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i in range(spec.frames):
            data = to_yuv420(draw(spec, params, i, fps), spec.matrix, spec.full_range, spec.bits)
            raw.update(data)
            proc.stdin.write(data)
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit(f"{spec.id}: ffmpeg failed")
        commands = [["ffmpeg"] + cmd[1:-1] + [Path(cmd[-1]).name]]
        if spec.rotation:
            remux = remux_command(spec, ffmpeg, path)
            subprocess.run(remux, check=True)
            Path(remux[remux.index("-i") + 1]).unlink()
            commands.append(["ffmpeg"] + [Path(a).name if a.startswith(str(out)) else a for a in remux[1:]])
        file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        md5s = framemd5(str(decoder_bin / "ffmpeg"), path)
        rec = asdict(spec) | {
            "file": spec.file,
            "container": spec.container,
            "sha256": file_sha,
            "bytes": path.stat().st_size,
            "encoder_input_sha256": raw.hexdigest(),
            "commands": commands,
            "probe": probe(ffprobe, path),
            "decoded_native_framemd5": md5s,
            "license": "MIT (generated by this script; original work of this repository)",
        }
        if len(md5s) != spec.frames:
            raise SystemExit(f"{spec.id}: decoded {len(md5s)} frames, expected {spec.frames}")
        records.append(rec)
        print(spec.id, rec["bytes"], file_sha[:12], rec["probe"].get("pix_fmt"), flush=True)
    return records


def tool_version(binary: Path) -> str:
    return subprocess.run([str(binary), "-version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg-bin", type=Path, required=True, help="directory with ffmpeg and ffprobe (GPL)")
    parser.add_argument("--decoder-bin", type=Path, help="directory with the LGPL ffmpeg (default: --ffmpeg-bin)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="*", help="generate only these ids")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LC_ALL", "C")
    if args.only:
        SPECS[:] = [s for s in SPECS if s.id in args.only]
    decoder_bin = args.decoder_bin or args.ffmpeg_bin
    records = build(args.out, args.ffmpeg_bin, decoder_bin)
    manifest = {
        "generator": "model-pipeline/reference/embeddinggemma2/make_eg2_videos.py",
        "seed": SEED,
        "ffmpeg_build": FFMPEG_BUILD,
        "encoder": tool_version(args.ffmpeg_bin / "ffmpeg"),
        "framemd5_decoder": tool_version(decoder_bin / "ffmpeg"),
        "numpy": np.__version__,
        "pillow": Image.__version__,
        "videos": records,
    }
    (args.out / "videos.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
