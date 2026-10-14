"""Edge-case videos for the EmbeddingGemma 2 video capabilities the accepted video set leaves untested.

Capabilities: `video_decode_rotation_other` (display rotation -90 and 180), `video_hdr` (HEVC Main 10 tagged BT.2020
with the PQ and HLG transfer functions), `video_interlaced` (H.264 coded as interlaced, top and bottom field first)
and `video_variable_frame_rate` (H.264 in MP4 with varying frame durations).

The frames, the encoder settings and the pinned FFmpeg 8.1 GPL build are those of `make_eg2_videos.py`, which this
script imports. Its own list of specs is kept separate, so regenerating the accepted set is unchanged. The videos are
original works of this repository (MIT). They are not committed.

HDR here means only the stream signalling. The pixels are the same synthetic SDR drawings, converted with the BT.2020
non-constant-luminance matrix and tagged with the PQ (SMPTE ST 2084) or HLG (ARIB STD-B67) transfer. What is
checked is that a decoder produces upstream's RGB for such a stream. That is torchcodec's conversion, with no tone
mapping, so it does not test whether the result looks right.

`edge-videos.json` has the layout of `videos.json` plus a per-video `checks` record from `ffprobe -show_frames`: the
distinct frame durations (VFR), the frames' `interlaced_frame` and `top_field_first` flags and the stream's
`field_order`, and the colour tags.

Usage:
    python make_eg2_edge_videos.py --ffmpeg-bin <GPL build bin> --decoder-bin <LGPL build bin> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import make_eg2_videos as mv
import numpy as np
from PIL import Image

# BT.2020 non-constant luminance (Kr, Kb) and the stream tags of each HDR variant.
mv.MATRICES["bt2020-pq"] = (0.2627, 0.0593)
mv.MATRICES["bt2020-hlg"] = (0.2627, 0.0593)
mv.TAGS["bt2020-pq"] = ["-colorspace", "bt2020nc", "-color_primaries", "bt2020", "-color_trc", "smpte2084"]
mv.TAGS["bt2020-hlg"] = ["-colorspace", "bt2020nc", "-color_primaries", "bt2020", "-color_trc", "arib-std-b67"]

# Timestamps in ticks of 1/600 s (1/15, 1/20, 1/30 and 1/60 s are whole ticks) through the filter, the encoder and
# the MP4 track, so no stage rounds them to the 1/30 s input rate.
VFR_COMMON = ["-fps_mode", "passthrough", "-enc_time_base", "1/600", "-video_track_timescale", "600"]
INTERLACED = ["-flags", "+ilme+ildct"]
# libx265 writes the colour description into the VUI only from its own parameters; this replaces the -x265-params of
# make_eg2_videos.encoder_args (same pools/threads settings) and adds the HDR10 static metadata SEI for PQ.
X265_BASE = "pools=1:frame-threads=1:wpp=0:log-level=error:colorprim=bt2020:colormatrix=bt2020nc:range=limited"
HDR10_SEI = (
    ":hdr10=1:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=1000,400"
)

EDGE_SPECS = [
    mv.Spec(
        "h264-rot180-640x360-30fps-3s", "h264", 640, 360, "30", 90, "text", rotation=180, note="display matrix 180"
    ),
    mv.Spec(
        "h264-rot-90-640x360-30fps-3s", "h264", 640, 360, "30", 90, "text", rotation=-90, note="display matrix -90"
    ),
    mv.Spec(
        "h264-rot180-portrait-360x640-30fps-2s",
        "h264",
        360,
        640,
        "30",
        60,
        "bars",
        rotation=180,
        note="display matrix 180, portrait",
    ),
    mv.Spec(
        "hevc-hdr-pq-640x360-30fps-3s",
        "hevc",
        640,
        360,
        "30",
        90,
        "shapes",
        "bt2020-pq",
        bits=10,
        note="HEVC Main 10, BT.2020, PQ",
        extra=["-x265-params", X265_BASE + ":transfer=smpte2084" + HDR10_SEI],
    ),
    mv.Spec(
        "hevc-hdr-hlg-640x360-30fps-3s",
        "hevc",
        640,
        360,
        "30",
        90,
        "bars",
        "bt2020-hlg",
        bits=10,
        note="HEVC Main 10, BT.2020, HLG",
        extra=["-x265-params", X265_BASE + ":transfer=arib-std-b67"],
    ),
    mv.Spec(
        "h264-interlaced-tff-720x480-29.97fps-3s",
        "h264",
        720,
        480,
        "30000/1001",
        90,
        "text",
        note="interlaced coding, top field first",
        extra=[*INTERLACED, "-x264-params", "tff=1", "-field_order", "tt"],
    ),
    mv.Spec(
        "h264-interlaced-bff-720x576-25fps-3s",
        "h264",
        720,
        576,
        "25",
        75,
        "shapes",
        note="interlaced coding, bottom field first",
        extra=[*INTERLACED, "-x264-params", "bff=1", "-field_order", "bb"],
    ),
    mv.Spec(
        "h264-vfr-30-then-15fps-640x360",
        "h264",
        640,
        360,
        "30",
        90,
        "bars",
        note="frames 0-44 at 1/30 s, 45-89 at 1/15 s (4.5 s)",
        extra=["-vf", "settb=1/600,setpts='if(lt(N,45),N*20,900+(N-45)*40)'", *VFR_COMMON],
    ),
    mv.Spec(
        "h264-vfr-alternating-320x240",
        "h264",
        320,
        240,
        "30",
        120,
        "shapes",
        note="frame durations alternate 1/20 and 1/60 s (4 s)",
        extra=["-vf", "settb=1/600,setpts='trunc(N/2)*40+mod(N,2)*30'", *VFR_COMMON],
    ),
]


def framemd5_passthrough(ffmpeg: str, path: Path) -> list[str]:
    """`make_eg2_videos.framemd5` with `-fps_mode passthrough`: every decoded frame once, also for VFR streams (the
    default output frame-rate mode duplicates or drops frames to a constant rate)."""
    out = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-noautorotate",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-fps_mode",
            "passthrough",
            "-f",
            "framemd5",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [line.rsplit(",", 1)[1].strip() for line in out.splitlines() if line and not line.startswith("#")]


def frame_checks(ffprobe: str, path: Path) -> dict:
    out = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_streams",
            "-print_format",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    data = json.loads(out)
    frames = data["frames"]
    stream = data["streams"][0]
    pts = [float(f["pts_time"]) for f in frames if "pts_time" in f]
    durations = sorted({round(b - a, 6) for a, b in zip(pts, pts[1:], strict=False)})
    return {
        "frames": len(frames),
        "distinct_frame_durations": durations,
        "interlaced_frames": sum(int(f.get("interlaced_frame", 0)) for f in frames),
        "top_field_first_frames": sum(int(f.get("top_field_first", 0)) for f in frames),
        "field_order": stream.get("field_order"),
        "color_space": stream.get("color_space"),
        "color_transfer": stream.get("color_transfer"),
        "color_primaries": stream.get("color_primaries"),
        "pix_fmt": stream.get("pix_fmt"),
        "display_rotation": next(
            (sd.get("rotation") for sd in stream.get("side_data_list", []) if "rotation" in sd), None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg-bin", type=Path, required=True, help="directory with ffmpeg and ffprobe (GPL)")
    parser.add_argument("--decoder-bin", type=Path, help="directory with the LGPL ffmpeg (default: --ffmpeg-bin)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LC_ALL", "C")
    mv.SPECS[:] = EDGE_SPECS
    mv.framemd5 = framemd5_passthrough
    decoder_bin = args.decoder_bin or args.ffmpeg_bin
    records = mv.build(args.out, args.ffmpeg_bin, decoder_bin)
    for rec in records:
        rec["checks"] = frame_checks(str(args.ffmpeg_bin / "ffprobe"), args.out / rec["file"])
        print(rec["id"], rec["checks"], flush=True)
    manifest = {
        "generator": "model-pipeline/reference/embeddinggemma2/make_eg2_edge_videos.py",
        "seed": mv.SEED,
        "ffmpeg_build": mv.FFMPEG_BUILD,
        "encoder": mv.tool_version(args.ffmpeg_bin / "ffmpeg"),
        "framemd5_decoder": mv.tool_version(decoder_bin / "ffmpeg"),
        "numpy": np.__version__,
        "pillow": Image.__version__,
        "videos": records,
    }
    (args.out / "edge-videos.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"{len(records)} videos", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
