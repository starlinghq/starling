# EmbeddingGemma 2: video-path spike

Status: spike result, 2026-10-11. Brief: maintainer approval (Mark, 2026-10-11) to extend the EmbeddingGemma 2 contract to images, then video; the image path is merged (PR #45). Template: the image spike (`docs/m0/embeddinggemma2-image-spike.md`, section 11 "What carries over to video"). Upstream code, the model card and repository files were read as data.

Pins: checkpoint `google/embeddinggemma-2` at `914f7f89142e33e77833254d9c9b90c3cef7303b`; transformers 5.19.0, sentence-transformers 6.1.0, torch 2.14.1 (CPU), torchvision 0.29.1, ONNX Runtime 1.30.0, as for images. New for video: torchcodec 0.17.0 (the decoder upstream uses), PyAV 18.1.0 and OpenCV 5.0.0.93 (compared only), environment `model-pipeline/reference/embeddinggemma2/requirements-video.txt` (the image environment plus those three). FFmpeg: BtbN/FFmpeg-Builds `autobuild-2026-10-11-13-04`, FFmpeg `n8.1.3-17-gc389840894` (commit `c3898408946f6cbfe8e244372c250366d5639d00`), shared LGPL and GPL builds, archives checked by SHA-256 (`fetch_eg2_ffmpeg.sh`). torchcodec source read at `v0.17.0` = `d9dc54238e173000bc3969eb3959777026ff9ae8`. Graphs: the released image-path graphs, unchanged (`embeddinggemma2-vision-fp32.onnx` `6c97f477...`, `embeddinggemma2-text-mm-fp32.onnx` `a2c22e07...`).

Contract under test: video-only and interleaved text-and-video embeddings (mean pooling over every token, L2 normalization, 768 dimensions) with the checkpoint's video processor settings (1 frame per second, at most 32 frames, 140 soft tokens per frame, no timestamps).

## 1. Summary

| Question | Answer |
| --- | --- |
| How does a video reach the embedding? | torchcodec decodes (FFmpeg; metadata from a packet scan); 1 fps sampling by frame index from the average frame rate, `int(duration)` frames (at least one), then 32 frames picked by `np.linspace(..., dtype=int)`; every frame resized with the image rule at a 140-token budget (same torchvision uint8 antialiased bicubic kernel), x 1/255, 16x16 patches; all frames through the vision tower as one batch; per frame `<\|image>` + n x `<\|video\|>` + `<image\|>`, no separator, no timestamps; soft tokens at the `<\|video\|>` positions (id 258884); mean pooling over every token. Section 2. |
| Sampling formula | Section 2.2, exact: a C# port equals upstream on 20,849 cases. |
| Graphs | The released vision graph and multimodal text graph take video frames unchanged (1,260-row frames, any multiple of 9 patches; mask at `<\|video\|>`). Python ORT CPU: embedding max abs 1.7e-6, padded and unpadded frames bit-identical; CUDA (`use_tf32=0`): 1.1e-6 (TF32 on: 4.9e-4). Section 4. |
| C# on decoded frames | The image resampler at 140 tokens gives the reference's resized pixels, patches and positions **exactly** for every frame of all 31 video inputs; the owned tokenizer gives the reference ids on all 35 rows; embedding 1.7e-6 (CPU), 4.5e-7 (CUDA). Section 5. |
| Decoding | FFmpeg-based decoding with FFmpeg's default conversion gives torchcodec's RGB **bit for bit** on all 22 videos: a C# decoder on FFmpeg.AutoGen 8.1.0 (MIT) that replicates torchcodec's calls, the `ffmpeg` CLI (`-pix_fmt rgb24`), and torchcodec on the system FFmpeg 9.0.1. PyAV differs on 10-bit and odd-size video, OpenCV on 10-bit (and cannot open AV1); a managed Y'CbCr-to-RGB conversion differs by up to 3 levels on every video (embedding up to 9.3e-3 on 8-bit video, cosine down to 0.9977). No managed H.264/HEVC/VP9/AV1 decoder exists for .NET. Section 6. |
| Recommendation | Model package: API on decoded frames (exact, no native dependency). Optional adapter on FFmpeg.AutoGen with user-supplied LGPL FFmpeg shared libraries (exact against upstream here; no codec binaries shipped). Decisions in section 11. |
| Upstream quirks | Sub-second clips and `int(duration)` truncation (a 1.9 s clip gives one frame); decoded arrays without metadata keep all frames and are capped uniformly (a different embedding, cosine 0.965 here); 10-bit and odd-size video are converted by a different swscale path (the large differences against a textbook conversion suggest interpolated chroma); `.webm` and `.m4v` path strings are embedded as text by sentence-transformers. Section 2.5. |
| Estimate | 11 to 17 working days for a video-embedding preview on decoded frames plus the FFmpeg adapter (section 10). |

## 2. Upstream flow (transformers 5.19.0, sentence-transformers 6.1.0, torchcodec 0.17.0)

### 2.1 Inputs and decoding

- sentence-transformers accepts a path or URL string (detected as video only for an existing file ending in `.mp4`, `.avi`, `.mov`, `.wmv`, `.flv` or `.mkv`, or a YouTube URL), a `{"video": ...}` entry of a multimodal dict, a torchcodec `VideoDecoder`, a frame array or tensor (3 to 5 dimensions), a list of PIL frames, or `{"array": frames, "video_metadata": {...}}`. The chat template renders a video as `<|video|>`, after the text in dict key order unless the text already contains placeholders (as for images).
- A path goes to `BaseVideoProcessor.fetch_videos`, which uses torchcodec (`VideoDecoder(path, seek_mode="exact", num_ffmpeg_threads=0)`); without torchcodec it falls back to torchvision's video API, which torchvision 0.29 no longer has (ImportError). The pinned image environment therefore cannot embed video files at all; `requirements-video.txt` adds torchcodec. PyAV, decord and OpenCV backends exist in `transformers.video_utils` but are used only when called explicitly.
- Metadata (torchcodec, exact mode): a packet scan without decoding; `num_frames` = packets of the stream (discarded packets skipped); `duration` = max(pts + duration) - min(pts), in seconds as `pts * num / den`; `average_fps` = `num_frames / duration`.
- A `VideoDecoder` object is decoded completely by sentence-transformers (`get_frames_in_range(0, len)`) with the same metadata; a frame array carries no metadata unless given.
- Frames: `get_frames_at(indices)`, uint8 RGB [T, 3, H, W]. On CPU torchcodec converts each decoded frame with swscale when the width is a multiple of 32 and no rotation applies (one `sws_scale` to RGB24 with flags 0, which FFmpeg 8.1 turns into bicubic; source matrix from `sws_getCoefficients(frame colorspace)`, so BT.601 when untagged; source range from the frame when tagged), otherwise with a filter graph (`buffer` -> `copy`, or `format=rgb24,transpose=cclock|clock` or `hflip,vflip` for a display matrix, -> `buffersink` RGB24). The display matrix is applied (a 640x360 stream with rotation 90 gives 360x640 frames).

### 2.2 Sampling (`EmbeddingGemma2VideoProcessor.sample_frames`, fps 1, max_frames 32, overflow "uniform")

```text
if fps or duration unknown:   indices = 0 .. total-1                      (decoded array without metadata)
else:                         step = fps / 1
                              n = max(1, int(duration * 1))               (truncation; at least one frame)
                              indices[i] = min(total - 1, int(i * step))  for i in 0 .. n-1   (float64, truncated)
if len(indices) > 32:         picks[i] = floor(i * ((m - 1) / 31)) for i < 31, picks[31] = m - 1   (m = len(indices);
                              this is np.linspace(0, m - 1, 32, dtype=int) in float64)
                              indices = indices[picks]
```

Consequences: sampling is by frame index from the average rate, not by timestamp (variable-frame-rate video samples frames, not seconds); a clip shorter than 2 s gives one frame (0.5 s, 1.0 s and 1.9 s all give frame 0); 29.97 fps gives 0, 29, 59, 89, ...; a 40 s clip gives 40 candidates thinned to 32 (0, 30, 60, 90, 150, ...); 32 s gives exactly 32; 33 s gives 32 of 33. The restatement (`eg2_video.sample_indices`) and the C# port (`VideoSampling.SampleIndices`) equal upstream on 20,849 cases (`make_eg2_sampling_cases.py`: common and random rates, exact and jittered durations, the 1 s and 32/33 boundaries, metadata-less inputs).

### 2.3 Preprocessing and placeholders

- All frames of one video share a size. Target size: the image rule (`get_aspect_ratio_preserving_size`) with `max_patches` = 140 x 9 = 1,260: largest 48-multiple size within 322,560 pixels. Examples: 640x360, 1280x720, 1920x1080 and 330x186 to 720x384 (120 tokens); 320x240, 160x120, 64x48 and 640x480 to 624x480 (130); 360x640 to 384x720 (120). The full 140 tokens need a shape such as 480x672 (10 x 14 blocks of 48).
- Resize: `torchvision.transforms.v2.functional.resize` on the uint8 [T, 3, H, W] tensor (bicubic, antialias): per frame identical to the image kernel (the C# image resampler reproduces it, section 5); skipped when the size already matches. Rescale x 0.00392156862745098; no normalization. Patches per frame padded with zeros and (-1, -1) positions to 1,260 rows.
- Placeholders: each `<|video|>` becomes, per frame, `<|image>` (255999) + n x `<|video|>` (258884) + `<image|>` (258882), concatenated with no separator; with timestamps on (not the default) frames would be `mm:ss <|image>...<image|>` joined by spaces. `<bos>`/`<eos>` around the whole input. Tokens per frame n + 2, so a 32-frame video at 120 tokens is 3,904 + 2 tokens and at the 140-token maximum 4,544 + 2.
- `<|video|>` is in `tokenizer.json`; the processor re-adds it as a special token. The owned tokenizer gives the reference ids for every row (section 5).

### 2.4 Model

`get_video_features` runs the vision tower on all frames of all videos as one batch ([total frames, 1,260, 768]), `embed_vision` on the pooled tokens, and splits them per video by `non_pad // 9` per frame; the soft tokens replace the token embedding at the `<|video|>` positions (`masked_scatter`), images first when an input has both. The rest is the image path (section 2.3 of the image spike).

### 2.5 Upstream behaviour that a C# contract has to choose about

| Case | Upstream (observed) | Note |
| --- | --- | --- |
| Clip under 2 s | One frame (index 0); 1.9 s also one frame | `int(duration)`; keep for parity |
| Decoded frames without metadata | All frames kept, then uniform to 32: for the 150-frame bars clip, 32 frames instead of 5; embedding cosine 0.965 against the file | C# should require a frame rate and duration (or pre-sampled frames) rather than silently diverge |
| Decoded frames with metadata, or a `VideoDecoder` | Equal to the file path (embedding difference 0) | |
| Variable frame rate | Index-based sampling from the average rate | untested here |
| 10-bit video | Converted by swscale without the 8-bit fast path (inferred: chroma is interpolated); up to 64 levels from a textbook conversion at sharp chroma edges | Reproduced exactly through FFmpeg; a pre-decoded-frames API takes 8-bit RGB only |
| Display matrix | Applied | |
| Odd sizes, widths not a multiple of 32 | Filter-graph conversion | Exact through FFmpeg |
| `.webm`, `.m4v` path strings | Classified as text and embedded as the file name (sentence-transformers' extension list) | A typed C# API has no such case |
| Audio track | Ignored | |
| Input above 8,192 tokens | Not tested upstream; two 32-frame videos at 140 tokens are 9,092 tokens | Reject, as for text and images |

### 2.6 Reference stability (`video-reference.json`)

Batch of 4 videos vs single 8.9e-8; eager vs sdpa 4.8e-7; file path vs decoded array with metadata 0; file path vs `VideoDecoder` object 0; restated vision path vs upstream soft tokens: all frames padded 1.9e-6, unpadded 1.5e-4, one frame at a time 1.6e-4 (PyTorch); no NaN/Inf. Retrieval sanity: the three scene queries rank a video of their scene first (bars 0.79, shapes 0.78, scrolling text 0.74 cosine).

## 3. Video set and reference runner

- `make_eg2_videos.py`: 22 videos from seed 20261011, original works (MIT). Frames drawn with NumPy and Pillow's bitmap font (colour bars with a moving square, moving shapes on a background whose hue changes each second, scrolling text; every frame carries its index), converted to Y'CbCr in NumPy (exact float64 BT.601 or BT.709, limited or full range, 8 or 10 bit) and piped raw to the pinned FFmpeg CLI (GPL build), single-threaded. Codecs: H.264 High and High 10 (libx264), HEVC Main (libx265), VP9 in WebM (libvpx-vp9), AV1 in MP4 (libaom). Cases: 0.5 s, 1 s, 1.9 s, 5 to 12 s, 32 s, 33 s, 40 s; 1, 24, 25, 29.97, 30, 60 fps; 64x48 to 1920x1080, portrait, widths not a multiple of 32, odd 321x179; display rotation 90; BT.709, full range, untagged, 10-bit. 4.2 MB in all, not committed. Two generations gave byte-identical files. `videos.json` records the exact commands, file SHA-256, encoder-input SHA-256, ffprobe facts, and the decoder ground truth: FFmpeg's `framemd5` of every decoded frame in its native pixel format (decoding is bit-exact by specification; PyAV's bundled FFmpeg gives the same digests for all frames of all 22 videos).
- `run_embeddinggemma2_video.py`: sentence-transformers fp32 CPU, torchcodec on the pinned LGPL FFmpeg 8.1 (`eg2_video.pin_torchcodec_ffmpeg`: torchcodec would otherwise load the machine's FFmpeg 9 first). 36 rows: 22 video-only (path strings; the two WebM files as `{"video": path}`), 6 multimodal (caption then video, placeholder inside text, two videos, image and video, `document` prompt, `query` prompt), 3 input forms (array with metadata, array without metadata, `VideoDecoder`), 5 text queries. It restates sampling and preprocessing and stops if any index, pixel or position differs (none did). Per video: torchcodec metadata, sampled indices, SHA-256 of each sampled RGB frame and of every decoded frame (`video-frames.json`), and digests of every intermediate. Tensors (sampled frames, resized frames, soft tokens, token embeddings) outside Git (732 MB).
- `decode_eg2_videos.py`, `compare_eg2_video_decoders.py`: other decoders and their embedding impact. `parity_embeddinggemma2_video_ort.py`: ORT on the released graphs. `compare_embeddinggemma2_video_csharp.py`: C# prototype runs.

## 4. Graphs and ORT parity

No new graph. The vision graph accepts any multiple of 9 patches, so a frame is a 1,260-row (padded, as upstream) or `9 x n`-row (unpadded) image; the multimodal text graph takes the soft tokens and a mask that is true at `<|video|>` (and `<|image|>`) ids. Script: `parity_embeddinggemma2_video_ort.py` (frames from the reference's sampled RGB, preprocessing checked against the reference digests).

Reference: PyTorch fp32 CPU (`ref-clean`, 36 rows, 276 sampled frames). Row-relative = max abs difference / max abs reference value of the row (as for images). Cosines from the Python script are computed on float32 vectors (the float64 values are in section 5).

| Candidate | Soft tokens, row-relative | Token embeddings, row-relative | Embedding max abs | Min cosine | Layouts and batches | NaN/Inf |
| --- | --- | --- | --- | --- | --- | --- |
| ORT CPU, 12 threads, frame batches of 8 | 5.8e-5 | 2.1e-4 | 1.7e-6 (`av1-640x360-30fps-3s`) | 0.99999991 | unpadded (`9 x n` rows) vs padded (1,260): 0; text batches of 4 vs single 8.9e-8 | none |
| ORT CUDA, `use_tf32=0`, frame batches of 4 | 6.6e-5 (padded), 4.6e-5 (unpadded), 3.1e-5 (one frame per call) | 1.8e-4 | 1.1e-6 (`mm-caption-then-video`) | 0.99999996 | soft tokens unpadded vs padded 2.5e-4 abs, one frame vs batch 3.0e-4 abs; text batches 1.5e-7 | none |
| ORT CUDA, `use_tf32=1` (4 rows) | 6.1e-2 | | 4.9e-4 | 0.999995 | | none |

- The vision graph needs no change for video: a frame is an image of at most 1,260 patches. Video rows are longer (up to 4,226 tokens), which raises the token-embedding row-relative error from 4.5e-5 (images) to 2.1e-4; the embedding error stays at the image level.
- TF32 fails as for images; pin `use_tf32: 0`.
- GPU memory: frame batches of 4 (1,260 patches each) ran on GPU 1 with about 15 GB free (the other workload was not resident during the run). The image spike measured 610 MB of attention buffer per 2,520-patch image; a 1,260-patch frame needs about a quarter of that per layer. Batch sizes with the other workload loaded were not measured.

## 5. C# prototype (outside the repository)

`/home/mark/Projects/AI-models/embeddinggemma2/csharp-video` (.NET 10, ProjectReference to `Starlings.Models.EmbeddingGemma2` for the image processor, resampler and tokenizer; ONNX Runtime 1.30.0; FFmpeg.AutoGen 8.1.0):

- `VideoSampling.SampleIndices(totalFrames, fps?, duration?)`: section 2.2; 0 mismatches on 20,849 cases.
- Preprocessing: `EmbeddingGemma2ImageProcessor.Preprocess(frame, maxSoftTokens: 140)` per frame, no change to product code.
- Expansion of `<|video|>` and `<|image|>` placeholders in the input text, the owned tokenizer, vision graph in frame batches (unpadded), scatter, multimodal text graph, mean pooling.

| C# run (35 rows: 22 video, 6 multimodal, 2 input forms, 5 text; 31 video inputs) | Owned-tokenizer ids | Frames, resized frames, pixel values, positions (SHA-256 vs reference) | Embedding max abs | Min cosine (float64) |
| --- | --- | --- | --- | --- |
| CPU, 12 threads, frame batches of 8 (unpadded), frames from the C# FFmpeg decoder | 35 of 35 exact | 31 of 31 exact | 1.7e-6 (`av1-640x360-30fps-3s`) | 0.99999999992 |
| CUDA (`use_tf32=0`), frame batches of 4, frames from the reference (torchcodec) | 35 of 35 exact | 31 of 31 exact | 4.5e-7 | 0.999999999993 |
| CUDA, frames from the C# FFmpeg decoder | 35 of 35 exact | 31 of 31 exact | 4.5e-7 (identical to the previous row) | 0.999999999993 |

The row without metadata (`vid-array-no-metadata`) is not part of the proposed C# contract and was skipped. Every sampled frame of every video went through the existing `TorchAntialiasResampler` at the 140-token target size with no code change and matched the reference bit for bit (upscaling 64x48 to 624x480, downscaling 1920x1080 to 720x384, portrait, odd 321x179).

## 6. Video decoding for .NET

### 6.1 Decoded RGB against the reference (torchcodec + FFmpeg 8.1, 22 videos, sampled frames)

| Decoder | Videos with identical RGB (of 22) | Differences |
| --- | --- | --- |
| torchcodec 0.17.0 on the machine's FFmpeg 9.0.1 (distribution build) | 22 | none |
| **C# prototype: FFmpeg.AutoGen 8.1.0 + the pinned LGPL FFmpeg 8.1 libraries** (torchcodec's scan, rotation and conversion calls restated) | **22** | none; metadata (frame count, duration, average rate) and sampled indices equal too |
| `ffmpeg` CLI 8.1 (LGPL build), `-pix_fmt rgb24 -f rawvideo`, autorotate on | 22 | none |
| PyAV 18.1.0 (bundled FFmpeg, libavcodec 62.28.102), `to_ndarray("rgb24")` | 19 | 10-bit: up to 67 levels (mean 0.57); odd 321x179: up to 27 (mean 0.39); display matrix not applied; WebM frame count 0 in the header (its own sampling would fail) |
| OpenCV 5.0.0.93 (bundled FFmpeg) | 20 | 10-bit: up to 59 levels (mean 1.42); AV1 does not open ("hardware accelerated AV1 decoding" only) |
| Managed conversion (NumPy, float64 textbook matrix, nearest chroma) of FFmpeg's decoded Y'CbCr | 0 | 8-bit: up to 3 levels on 21 to 97% of bytes (mean 0.2 to 1.3); 10-bit: 64; odd 321x179: 164 (a different swscale path; chroma apparently interpolated) |
| Managed conversion, bilinear chroma | 0 | up to 161 levels at sharp chroma edges (mean 0.8 to 5.2) |

Reading: decoding to Y'CbCr is identical everywhere (bit-exact codecs, same FFmpeg); every difference comes from the Y'CbCr-to-RGB step. FFmpeg's default conversion (swscale, bicubic flag, frame colour tags, BT.601 when untagged; or the CLI's auto-inserted scale filter) is what upstream runs, and anything that calls it the same way matches bit for bit. The swscale x86 fast path uses fixed-point SIMD arithmetic; the reference and all matches above are on one AVX2 machine (other CPUs, ARM in particular, are untested and may differ by a level: the same holds for upstream itself).

### 6.2 Options and licenses

| Option | License | Codecs | Fidelity | Notes |
| --- | --- | --- | --- | --- |
| **FFmpeg.AutoGen 8.1.0** (bindings) + user-supplied FFmpeg shared libraries | Bindings MIT (changed from LGPL in recent versions; no native code shipped). FFmpeg: LGPL 2.1+ by default; the BtbN LGPL build is configured `--enable-version3`, so LGPL 3.0; GPL builds (`--enable-gpl`: libx264, libx265 encoders, some filters) make the whole library GPL; `--enable-nonfree` builds are not redistributable | Everything FFmpeg decodes natively (H.264, HEVC, VP9, AV1 via libdav1d, ...) | **Exact** (C# prototype, all 22 videos) | ABI is per FFmpeg major (8.x needs `libavcodec.so.62`; FFmpeg.AutoGen 9.0.1 exists for FFmpeg 9). LGPL duties fall on whoever distributes the FFmpeg binaries: dynamic linking, source offer, notices, no renamed libraries, replaceability. If Starling ships no FFmpeg binary, the package itself carries no LGPL obligation |
| `ffmpeg` CLI as a subprocess (`-f rawvideo -pix_fmt rgb24`), e.g. through FFMpegCore 5.5.0 (MIT) | Wrapper MIT; FFmpeg as above | as above | **Exact** (Python test, all 22 videos, rotation and 10-bit included) | No ABI coupling; needs `ffprobe` or a scan for torchcodec's metadata; process overhead and raw-frame piping |
| Sdcb.FFmpeg 7.0.0 | LGPL-3.0 bindings; runtime packages bundle FFmpeg binaries | as FFmpeg | not run | LGPL on the binding itself and bundled binaries: avoid |
| FFMediaToolkit 4.8.1 | MIT, on FFmpeg.AutoGen | as FFmpeg | not run | Own conversion settings; would need checking |
| GStreamer (gstreamer-sharp) | LGPL 2.1 core; plugins good (LGPL), bad/ugly (mixed, patents) | via `gst-libav` (FFmpeg) or openh264 plugins | not run | Heavy dependency; conversion differs from swscale |
| Media Foundation (Windows), AVFoundation (macOS, needs the `net-macos` workload) | OS components; codec licences paid by the OS vendor (HEVC on Windows needs a Store extension) | OS-dependent | not run | Platform-specific; own YUV-to-RGB; not on Linux |
| Pure managed decoders | n/a | none maintained for H.264, HEVC, VP9 or AV1 | n/a | Only MJPEG/GIF-class formats are realistic |
| Cisco OpenH264 (BSD-2 source) | Cisco pays the AVC pool royalty only for its own binary, downloaded at run time by the end user and not bundled; attribution text required | H.264 Constrained Baseline/Main decoding | not run | Partial profile coverage (no High 10); exactness would depend on the RGB conversion done afterwards |
| NVDEC (NVIDIA Video Codec SDK) | Proprietary SDK licence (redistribution only inside a product, with flow-down terms); FFmpeg loads the driver's `libnvcuvid` through MIT-licensed `nv-codec-headers` | NVIDIA GPUs only | not run | GPU colour conversion (NPP or CUDA kernels) is not swscale: not expected to match |

Patents: H.264 and HEVC are covered by licensing pools (Via LA; Access Advance for HEVC); VP9 and AV1 are offered royalty-free by their developers (AOMedia patent licence) though third-party pools assert claims. FFmpeg's own decoders come with no patent licence. A library that ships no codec binaries and decodes with the user's FFmpeg leaves codec licensing to the deployer; the package README must say so. This is an engineering summary, not legal advice; the maintainer may want counsel for the distribution question (decision 3).

### 6.3 Embedding impact of other conversions

Each video embedded from the other decoder's frames (as a decoded array; a control with the reference's own frames gives 0 difference):

| Conversion | Videos differing | Embedding max abs (worst) | Min cosine |
| --- | --- | --- | --- |
| PyAV | 2 (10-bit, odd size) | 9.1e-3 | 0.99885 |
| OpenCV | 1 (10-bit) | 5.4e-3 | 0.99888 |
| Managed, nearest chroma (8-bit videos only) | 19 | 9.3e-3 (`h264-rot90`) | 0.99770 (`h264-ntsc`) |
| Managed, nearest chroma (all 22) | 22 | 1.7e-2 (`vp9-odd`) | 0.99625 |
| Managed, bilinear chroma | 22 | 1.7e-2 | 0.98996 (`h264-1fps-64x48`) |

Video is about three times more sensitive than images to this kind of error (images: 1 to 2 levels of resampling difference gave up to 2.6e-3): one to three levels on most pixels of every frame, over up to 32 frames, move the embedding by up to 9e-3. A managed converter would have to reproduce swscale's fixed-point x86 arithmetic (and its general-scaler path for 10-bit and odd sizes); possible in principle, platform-dependent upstream as well, and not attempted here.

### 6.4 Recommendation

1. Model package: `EmbeddingGemma2Video` built from decoded 8-bit RGB frames, in two forms: all decoded frames with the frame rate and duration (the package samples exactly as upstream, section 2.2), or frames the caller already sampled (at most 32, used as given). Exact against the reference, no native dependency, same tolerance profile as images.
2. Optional adapter `Starlings.Models.EmbeddingGemma2.Video.FFmpeg` on FFmpeg.AutoGen (MIT), loading FFmpeg shared libraries the user installs (LGPL build recommended; none shipped), replicating torchcodec's scan, rotation and conversion: exact on this set. Its fidelity depends on the user's FFmpeg major (8.x tested; 9.0.1 through torchcodec also exact) and CPU.
3. Alternative to 2 if the maintainer prefers no FFmpeg-facing package: a documented recipe (`ffmpeg -i in.mp4 -f rawvideo -pix_fmt rgb24 -` plus ffprobe metadata), which was also exact.

## 7. Proposed contract

| Feature | Status proposed | Evidence |
| --- | --- | --- |
| Video embedding from decoded 8-bit RGB frames with frame rate and duration (upstream sampling), 140 tokens per frame, at most 32 frames | experimental | sections 2.2, 4, 5 |
| Pre-sampled frames (caller's choice, at most 32) | experimental | `vid-array-*` rows (equal to the file when the frames are upstream's) |
| Interleaved text, images and videos (`<\|video\|>` and `<\|image\|>` placeholders, or appended after the text) | experimental | 6 multimodal rows, owned tokenizer exact |
| Prompts with video | experimental for the rendering (card: no prompts for media) | `vid-query-prompt`, `mm-document-prompt` |
| FFmpeg adapter (file or stream to frames and metadata, display matrix, 8- and 10-bit, H.264/HEVC/VP9/AV1 as the user's FFmpeg supports) | experimental with the decoder profile | section 6 |
| Timestamps (`add_timestamps`), other fps, `max_frames` or budgets (70, 280, 560, 1,120) | untested | |
| Variable frame rate, interlaced, HDR (beyond 10-bit SDR), audio track | untested (audio unsupported) | |
| Frames without metadata (upstream: keep all, uniform to 32) | unsupported by default (divergent); available as an explicit option if wanted | section 2.5 |
| Input over 8,192 tokens (e.g. two 32-frame videos at 140 tokens) | rejected | |
| Decoded size limit | reject frames above 50 megapixels; limit decoded frames held in memory (the adapter decodes sequentially and keeps only sampled frames) | |
| CUDA | experimental, `use_tf32: 0` pinned | section 4 |

Budget inside 8,192 tokens: one video costs `frames x (n + 2)` tokens with n <= 140 and frames <= 32, at most 4,544, plus `<bos>`, `<eos>`, prompt and text; about 3,640 tokens remain for text and images next to a maximal video.

## 8. Proposed tolerances (values only; no tolerances file)

| Key | Proposed bound | Observed worst (Python ORT CPU / CUDA TF32 off / C# CPU / C# CUDA) |
| --- | --- | --- |
| Sampled indices from (frames, fps, duration) | exact | exact (20,849 cases; 22 videos) |
| Rendered ids (owned tokenizer on expanded placeholders) | exact | exact (35 rows) |
| Resized uint8 frames, pixel values, positions from decoded RGB | exact (SHA-256) | exact |
| Soft tokens, row-relative | 1e-3 | 5.8e-5 / 6.6e-5 / not dumped / not dumped |
| Token embeddings, row-relative | 1e-3 | 2.1e-4 / 1.8e-4 / not dumped / not dumped |
| Embedding max abs | 1e-5 | 1.7e-6 / 1.1e-6 / 1.7e-6 / 4.5e-7 |
| Embedding min cosine | >= 0.9999995 | 0.99999991 (float32) / 0.99999996 (float32) / 0.99999999992 / 0.999999999993 |
| Frame batches vs single frames, padded vs unpadded | 1e-5 on the embedding | text batches 8.9e-8 / 1.5e-7; PyTorch batch of 4 videos 8.9e-8; vision layouts 0 (CPU) and 3.0e-4 on soft tokens (CUDA, inside the soft-token bound) |
| NaN/Inf | none | none |
| Provider options | CUDA `use_tf32: 0` | |
| Decoder profile (FFmpeg adapter): decoded RGB vs torchcodec on the fixture videos, same FFmpeg major, x86-64 | exact | exact (22 of 22) |
| Other conversions | reported, not gated | PyAV 9.1e-3, OpenCV 5.4e-3, managed nearest 1.7e-2 / 0.996: none would pass a useful bound; reported only |

Same values as the image profile: the video path adds no new numerical stage.

## 9. Timing (indicative; shared workstation)

| Stage | CPU (i7-5820K, 12 threads) | CUDA (RTX 4060 Ti) |
| --- | --- | --- |
| C# FFmpeg decode (scan, full sequential decode, conversion of the sampled frames) | 4 ms (8 frames of 64x48) to 350 ms (250 frames of 1280x720); 330 ms for 1,200 frames of 320x240; 290 ms for 120 frames of 1920x1080 | |
| C# preprocessing (scalar resampler), per frame | 6 to 40 ms | |
| Vision graph, per frame (up to 1,260 patches) | 2.3 s (Python), 2.7 s (C#) | 0.08 s (Python), 0.07 s (C#) |
| Text graph, 4,226-token row (32 frames) | 16 s (C#) | 0.48 s |
| One 32-frame video end to end | about 100 s | about 3 s |

CPU video embedding is slow (the vision tower dominates; 32 frames cost as much as 16 full-budget images); the README should steer video users to CUDA. The PyTorch reference took about an hour for 36 rows on CPU.

## 10. Implementation estimate

| Work item | Days |
| --- | --- |
| `EmbeddingGemma2Video` input type (frames, stride, frame rate, duration or pre-sampled), sampling, per-frame preprocessing reuse, placeholder expansion, 8,192-token and size limits, unit tests on the sampling cases and fixture digests | 2 to 3 |
| Encoder: frame batching through the vision session (memory-aware on GPU), scatter at `<\|video\|>`, interleaving with images, CUDA profile | 2 to 3 |
| FFmpeg adapter: FFmpeg.AutoGen loading and version check, torchcodec-equivalent scan, sequential decode keeping only sampled frames, swscale and filter-graph conversion, display matrix, errors for unsupported streams, tests on the generated videos, README on FFmpeg licensing and installation per OS | 4 to 6 |
| Fixtures and evaluator: video rows in the image task (or a video task), C# stage results, generator in CI-free docs | 2 to 3 |
| Extensions.AI (`DataContent` video) and capability statuses, README, sample | 1 to 2 |
| **Total** | **11 to 17** |

## 11. Decisions for the maintainer

1. Contract: video-only and interleaved text/image/video at Experimental with the checkpoint's settings (1 fps, 32 frames, 140 tokens, no timestamps), from decoded frames with frame rate and duration (section 7).
2. Frames without metadata: reject (recommended), or offer upstream's "keep all, uniform to 32" as an explicit option.
3. Decoding: (a) ship the FFmpeg.AutoGen adapter package with user-supplied FFmpeg (no binaries shipped; README states LGPL and codec-patent responsibilities), (b) document a recipe only, or (c) neither. Legal review of (a) is the maintainer's call.
4. FFmpeg versions the adapter supports (8.x tested; 9.x via FFmpeg.AutoGen 9) and whether the decoder profile is gated (exact on x86-64) or reported.
5. Upstream quirks kept for parity: `int(duration)` sampling (one frame under 2 s), index-based sampling for variable frame rate, 10-bit through FFmpeg's default path.
6. Tolerances: set the values in section 8; accept the proposed fixtures (stacked pull request, `model-pipeline/protected/embeddinggemma-2/video-fp32-cpu/`).
7. Environment: `requirements-video.txt` (torchcodec 0.17.0 + the pinned FFmpeg build) as the video reference environment.

## 12. Not done

- No product code; no evaluator change; no tolerances file.
- No PyTorch CUDA reference; no bfloat16; no task-quality (MTEB/MIEB video) run.
- Variable frame rate, interlaced, HDR, timestamps and budgets other than 140 not run.
- One CPU (AVX2) for the reference and the decoder comparison; ARM and non-AVX2 swscale paths untested; Windows/macOS untested.
- GStreamer, Media Foundation, AVFoundation, OpenH264 and NVDEC not run (license and platform analysis only).
- Timing is indicative (P-008 protocol not run).
