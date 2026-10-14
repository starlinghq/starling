# Proposed: EmbeddingGemma 2 edge-case videos (rotation, HDR, interlaced, variable frame rate)

Status: **proposed** (2026-10-14). Not accepted and not under `model-pipeline/protected/`. If the maintainer accepts it, the intended destination is `model-pipeline/protected/embeddinggemma-2/video-edge/`, byte-identical. `DecoderTests.EdgeCaseVideosEqualTorchcodec` (FFmpeg 8 adapter tests, and the same source compiled for the FFmpeg 9 adapter) reads the protected copy when it exists and this proposal otherwise.

These files cover four video capabilities that the 22 accepted videos leave untested: `video_decode_rotation_other`, `video_hdr`, `video_interlaced` and `video_variable_frame_rate`.

## Files

| File | Contents |
| --- | --- |
| `edge-videos.json` | Nine videos from `model-pipeline/reference/embeddinggemma2/make_eg2_edge_videos.py`, in the layout of `videos.json`: generation parameters, exact FFmpeg commands, file SHA-256, `ffprobe` facts, decoded-frame `framemd5`. A `checks` record from `ffprobe -show_frames` shows each property is really present. |
| `edge-video-frames.json` | torchcodec 0.17.0 on the pinned LGPL FFmpeg n8.1.3-17-gc389840894 (`torchcodec_eg2_frames.py --manifest`): per video the metadata, upstream's sampled indices and the SHA-256 of every decoded RGB frame. This is the layout of `video-frames.json`. |
| `edge-video-frames-summary.json` | Decoder versions and digests of the two files above |

## Videos

The frames are the synthetic scenes of `make_eg2_videos.py`, encoded by the pinned FFmpeg 8.1 GPL build. The videos are original works of this repository and are not committed.

| Video | Property (from `ffprobe`) |
| --- | --- |
| `h264-rot180-640x360-30fps-3s`, `h264-rot180-portrait-360x640-30fps-2s` | display matrix rotation 180 |
| `h264-rot-90-640x360-30fps-3s` | display matrix rotation -90 |
| `hevc-hdr-pq-640x360-30fps-3s` | HEVC Main 10, `bt2020nc` / `bt2020` / `smpte2084`, HDR10 mastering-display and content-light SEI |
| `hevc-hdr-hlg-640x360-30fps-3s` | HEVC Main 10, `bt2020nc` / `bt2020` / `arib-std-b67` |
| `h264-interlaced-tff-720x480-29.97fps-3s` | 90 of 90 frames interlaced, `field_order` tt |
| `h264-interlaced-bff-720x576-25fps-3s` | 75 of 75 frames interlaced, `field_order` bb |
| `h264-vfr-30-then-15fps-640x360` | frame durations 1/30 s, then 1/15 s (4.47 s; average 20.15 fps) |
| `h264-vfr-alternating-320x240` | frame durations alternate 1/20 s and 1/60 s |

## What upstream does, and the bound

Upstream decodes with torchcodec and then samples frames by index from torchcodec's `average_fps` and duration. It does not deinterlace, does not tone-map HDR (the 10-bit BT.2020 code values are converted to 8-bit RGB with torchcodec's swscale settings), and applies the display rotation. The bound is the existing decoder bound, `decoder_ffmpeg.decoded_rgb_vs_torchcodec: exact`. Equal decoded frames and sampled indices mean the embedding path receives upstream's input, and that path is parity-tested on the accepted videos. HDR here means only the signalling of the stream. A real HDR camera clip and its perceptual result are not covered.

## Result at the time of the proposal

| Adapter | Result |
| --- | --- |
| `Starlings.Models.EmbeddingGemma2.Video.FFmpeg` on the pinned LGPL FFmpeg 8.1 | 9 of 9 videos: metadata, sampled indices and all 795 frames equal to torchcodec |
| `Starlings.Models.EmbeddingGemma2.Video.FFmpeg9` on the machine's FFmpeg 9.0.1 (GPL, opt-in) | 9 of 9, all 795 frames equal to the same digests |

Logs are in `docs/evidence/capability-closure-2026-10-14/logs/`.
