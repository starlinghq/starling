# Starlings.Models.EmbeddingGemma2.Video.FFmpeg

Decodes video files and streams into `EmbeddingGemma2Video` for `Starlings.Models.EmbeddingGemma2` exactly as upstream sees them: sentence-transformers decodes videos with torchcodec 0.17 on FFmpeg, and this adapter makes the same FFmpeg calls (packet-scan metadata, 1 frame per second sampling up to 32 frames, the display matrix applied, FFmpeg's default RGB conversion). The model package itself takes decoded frames and has no native dependency; this adapter is optional.

**This package contains no FFmpeg.** It binds FFmpeg through its own bindings, generated from the FFmpeg 8.1 headers by `tools/generate-ffmpeg-bindings.py` (no third-party bindings package), and loads **your own FFmpeg 8.x shared libraries** at run time. For FFmpeg 9.x use `Starlings.Models.EmbeddingGemma2.Video.FFmpeg9` (one adapter package per FFmpeg major; the decoding code is shared). You choose, install and distribute FFmpeg (see [Licenses and patents](#licenses-and-patents)).

```csharp
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.EmbeddingGemma2.Video.FFmpeg;

var decoder = new EmbeddingGemma2VideoDecoder(new EmbeddingGemma2FFmpegOptions
{
    LibraryDirectory = "/opt/ffmpeg-8/lib",   // or $STARLING_FFMPEG_DIR, or the system library path
});
EmbeddingGemma2DecodedVideo clip = decoder.DecodeFile("clip.mp4");   // metadata, sampled indices, sampled frames
using var embedder = EmbeddingGemma2MultimodalEmbedder.Create(EmbeddingGemma2ModelAssets.OpenLocalImageAssets());
float[] embedding = embedder.Embed(clip.Video);                       // no prompt for media (the model card)
```

## Maturity

<!-- catalog:maturity profiles=embeddinggemma-2/video-* -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `embeddinggemma-2/video-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: untested |
| `embeddinggemma-2/video-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
<!-- /catalog:maturity -->

## Installing FFmpeg 8

You need the shared libraries `libavcodec.so.62`, `libavformat.so.62`, `libavutil.so.60`, `libswscale.so.9` and `libavfilter.so.11` (FFmpeg 8.x; on Windows `avcodec-62.dll` and so on, on macOS `libavcodec.62.dylib`). Use an **LGPL build**: configured without `--enable-gpl` and `--enable-nonfree`. Examples: a distribution package of FFmpeg 8, or an LGPL shared build such as the BtbN/FFmpeg-Builds one the evidence used (release `autobuild-2026-10-11-13-04`, FFmpeg `n8.1.3-17-gc389840894`, linux64 lgpl shared; fetched and checked by SHA-256 with `model-pipeline/reference/embeddinggemma2/fetch_eg2_ffmpeg.sh`). Point the adapter at the `lib` directory with `EmbeddingGemma2FFmpegOptions.LibraryDirectory` or `STARLING_FFMPEG_DIR`; without either, the operating system's library search is used (on Linux `LD_LIBRARY_PATH` and the loader cache). The runtime keeps the libraries it loaded for the process, so all decoders in a process use one location.

Errors are typed (`EmbeddingGemma2FFmpegException` and subclasses), checked when the decoder is created:

| Situation | Exception |
| --- | --- |
| Libraries not found (missing, wrong directory) | `EmbeddingGemma2FFmpegNotFoundException`; names other majors it finds (FFmpeg 9 is `libavcodec.so.63`) |
| A library with another major than FFmpeg 8 | `EmbeddingGemma2FFmpegVersionException` |
| A GPL or nonfree build (`avcodec_license()` and the other libraries' license strings) | `EmbeddingGemma2FFmpegLicenseException`, unless `AllowNonLgplBuild = true` (your deliberate choice; decoding is identical) |
| File does not open, no video stream, decode failure, unsupported rotation | `EmbeddingGemma2VideoDecodeException` |
| Frame above `MaxFramePixels` (50 MP) | `EmbeddingGemma2ImageTooLargeException` |

## What it reproduces

| Step | Behaviour (torchcodec 0.17.0, CPU, `seek_mode="exact"`) | Status |
| --- | --- | --- |
| Metadata (`Probe`) | Packet scan of the best video stream: frame count = packets (discard-flagged skipped), duration = max(pts + duration) - min(pts) in seconds, average rate = frames / duration, rotation from the display matrix; width and height after rotation | supported (Stable `video-fp32-*` contracts): equal on the 22 fixture videos and the 4 release videos (G1) |
| Sampling (`DecodeFile`, `Decode(Stream)`, `Decode(ReadOnlyMemory<byte>)`) | `EmbeddingGemma2VideoSampling.SampleIndices` on that metadata (upstream's 1 fps, at most 32, one frame under 2 s); torchcodec's `get_frames_at` on the scanning demuxer: frames reached by decoding on or by seeking to the scanned keyframe (`avformat_seek_file`, `avcodec_flush_buffers`) with torchcodec's seek-avoidance rule; only the sampled frames are kept. The frame after a seek is upstream's even where it differs from a sequential decode (the Ogg Theora video `commons-video-02` of the release set, frame 24) | supported (Stable `video-fp32-*` contracts): sampled frames equal torchcodec's on the 22 fixture videos and the 4 release videos (G1, AVX2 and AVX-512 hosts); the edge-case videos too (experimental rows) |
| RGB conversion | Width a multiple of 32 and no rotation: one `sws_scale` to RGB24, flags 0, coefficients of the frame's colourspace (BT.601 when untagged), source range from the frame. Otherwise a filter graph `buffer` -> `copy` (or `format=rgb24,transpose=cclock`, `transpose=clock`, `hflip,vflip`) -> `buffersink` rgb24. FFmpeg's default 10-bit conversion is kept (upstream quirk, for parity) | supported (Stable `video-fp32-*` contracts): every frame of the 22 fixture videos equal to torchcodec's (H.264 High and High 10, HEVC, VP9, AV1; BT.709, full range, untagged; odd 321x179; rotation 90) |
| Rotation -90 and 180 | torchcodec's transforms | experimental: every frame of the generated edge-case videos equal to torchcodec's (proposed `video-edge`) |
| All frames (`DecodeAllFrames`), frames by index (`DecodeFramesAt`) | Every frame (with timing: the package then samples like the file), or `get_frames_at` indices | experimental |
| Streams and memory | Seekable streams through a custom AVIO context; equal to the file on the fixtures | experimental |
| FFmpeg 9 | Other ABI; not loadable by this package (typed error naming `libavcodec.so.63`). Use `Starlings.Models.EmbeddingGemma2.Video.FFmpeg9`: the same code with bindings generated from the FFmpeg 9.0.1 headers, exact against torchcodec on FFmpeg 9.0.1 (every frame of the 22 fixture videos) | unsupported here; supported in the FFmpeg 9 package |
| Other CPUs (ARM64, x86-64 without AVX2) | swscale's SIMD paths may differ by a level (for upstream as well); AVX2 and AVX-512 x86-64 are measured (G1 exact on both) | untested |
| Variable frame rate, interlaced, HDR (PQ, HLG) | As upstream: index-based sampling from the average rate, no deinterlacing, no tone mapping | experimental: every frame, metadata and sampled indices of the generated edge-case videos equal to torchcodec's |
| Audio | Ignored, as upstream | unsupported |

The model package rejects decoded frames without timing by default (`EmbeddingGemma2VideoTimingRequiredException`); this adapter always provides the timing.

## Licenses and patents

The package is MIT OR Apache-2.0. Its FFmpeg bindings are generated from FFmpeg's headers and contain only interface facts (function signatures, the used struct fields' offsets, constants), not FFmpeg code; see `THIRD-PARTY-NOTICES.md`. It restates the FFmpeg calls of torchcodec v0.17.0 (BSD 3-Clause, Meta; no source copied). Summary: `THIRD-PARTY-NOTICES.md` in this package.

**FFmpeg is user-supplied and not distributed by Starlings.** FFmpeg is LGPL 2.1 or later (LGPL 3.0 with `--enable-version3`); a GPL build makes the program that loads it subject to the GPL, and a nonfree build is not redistributable. Whoever ships FFmpeg with an application carries FFmpeg's obligations (for the LGPL: dynamic linking, license text, source or an offer, unmodified library names, replaceability).

**Codec patents (information, not legal advice).** H.264/AVC and H.265/HEVC are covered by patent licensing pools (Via LA; Access Advance for HEVC). VP9 and AV1 are offered royalty-free by their developers (the AOMedia patent license for AV1), though third parties assert claims. FFmpeg's decoders come with no patent license. Because this package ships no codec, codec licensing is the deployer's responsibility; seek legal advice for your distribution.

**Publishable.** The legal review of the FFmpeg adapters (decision D2) was cleared on 2026-10-15. This package is publishable from the next release; it is not on nuget.org yet.

Deployments of the model must adhere to the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) (model card).
