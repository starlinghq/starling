# Starlings.Models.EmbeddingGemma2.Video.FFmpeg9

The FFmpeg 9 counterpart of `Starlings.Models.EmbeddingGemma2.Video.FFmpeg`: decodes video files and streams into `EmbeddingGemma2Video` for `Starlings.Models.EmbeddingGemma2` exactly as upstream sees them (sentence-transformers decodes videos with torchcodec 0.17 on FFmpeg; this adapter makes the same FFmpeg calls). One adapter package per FFmpeg major: use this one with FFmpeg 9.x, the `...Video.FFmpeg` package with FFmpeg 8.x. Both are optional; the model package takes decoded frames and references neither.

**This package contains no FFmpeg.** It binds FFmpeg through its own bindings, generated from the FFmpeg 9.0.1 headers (commit `bf1b838f2ab88b4f8fd83443325c782ea0e0f7fa`) by `tools/generate-ffmpeg-bindings.py --major 9` (no third-party bindings package), and loads **your own FFmpeg 9.x shared libraries** at run time. You choose, install and distribute FFmpeg (see [Licenses and patents](#licenses-and-patents)). The decoding code is the FFmpeg 8 adapter's, shared as source; only the bindings differ.

```csharp
using Starlings.Models.EmbeddingGemma2;
using Starlings.Models.EmbeddingGemma2.Video.FFmpeg9;

var decoder = new EmbeddingGemma2VideoDecoder(new EmbeddingGemma2FFmpegOptions
{
    LibraryDirectory = "/opt/ffmpeg-9/lib",   // or $STARLING_FFMPEG9_DIR, or the system library path
});
EmbeddingGemma2DecodedVideo clip = decoder.DecodeFile("clip.mp4");   // metadata, sampled indices, sampled frames
using var embedder = EmbeddingGemma2MultimodalEmbedder.Create(EmbeddingGemma2ModelAssets.OpenLocalImageAssets());
float[] embedding = embedder.Embed(clip.Video);                       // no prompt for media (the model card)
```

The types have the same names as in the FFmpeg 8 package, in the namespace `Starlings.Models.EmbeddingGemma2.Video.FFmpeg9`. A process loads one FFmpeg; reference the package of the major you install.

## Maturity

<!-- catalog:maturity profiles=embeddinggemma-2/video-* -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `embeddinggemma-2/video-fp32-cpu` | experimental | linux-x64 cpu: experimental; windows-x64 cpu: untested |
| `embeddinggemma-2/video-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
<!-- /catalog:maturity -->

## Installing FFmpeg 9

You need the shared libraries `libavcodec.so.63`, `libavformat.so.63`, `libavutil.so.61`, `libswscale.so.10` and `libavfilter.so.12` (FFmpeg 9.x; on Windows `avcodec-63.dll` and so on, on macOS `libavcodec.63.dylib`). Use an **LGPL build**: configured without `--enable-gpl` and `--enable-nonfree`. Many distribution packages are GPL builds (the evidence below used one, with `AllowNonLgplBuild`); check `ffmpeg -version` for `--enable-gpl`. Point the adapter at the `lib` directory with `EmbeddingGemma2FFmpegOptions.LibraryDirectory` or `STARLING_FFMPEG9_DIR`; without either, the operating system's library search is used. The runtime keeps the libraries it loaded for the process, so all decoders in a process use one location.

Errors are typed (`EmbeddingGemma2FFmpegException` and subclasses), checked when the decoder is created:

| Situation | Exception |
| --- | --- |
| Libraries not found (missing, wrong directory) | `EmbeddingGemma2FFmpegNotFoundException`; names other majors it finds (FFmpeg 8 is `libavcodec.so.62`, which the FFmpeg 8 package loads) |
| A library with another major than FFmpeg 9 | `EmbeddingGemma2FFmpegVersionException` |
| A GPL or nonfree build (`avcodec_license()` and the other libraries' license strings) | `EmbeddingGemma2FFmpegLicenseException`, unless `AllowNonLgplBuild = true` (your deliberate choice; decoding is identical) |
| File does not open, no video stream, decode failure, unsupported rotation | `EmbeddingGemma2VideoDecodeException` |
| Frame above `MaxFramePixels` (50 MP) | `EmbeddingGemma2ImageTooLargeException` |

## What it reproduces

As the FFmpeg 8 package (its README has the full table): torchcodec 0.17.0's packet-scan metadata, upstream's sampling, the display matrix, and FFmpeg's default RGB conversion as torchcodec calls it.

| Check (2026-10-13, x86-64, FFmpeg n9.0.1, libavcodec 63.1.101) | Result | Status |
| --- | --- | --- |
| Metadata and sampled indices, 22 fixture videos | equal to torchcodec 0.17.0 on FFmpeg 9.0.1 | experimental |
| Decoded RGB of every frame, 22 videos (5,367 frames; H.264 High and High 10, HEVC, VP9, AV1; BT.601, BT.709, full range, untagged; rotation 90; odd 321x179) | exact against torchcodec 0.17.0 on FFmpeg 9.0.1 | experimental |
| Seekable stream and in-memory input vs file | identical metadata and frames | experimental |
| Loader: missing directory, other major, license classification, GPL build refused without `AllowNonLgplBuild` | typed errors | experimental |
| LGPL FFmpeg 9 build | the BtbN LGPL shared build n9.0.2-28-g669f5ab083 (2026-10-15): every frame of the 22 fixture videos equal to torchcodec's digests (adapter tests: 26 passed, the edge-case and FFmpeg-8-only release tests skipped), and the sampled frames of the 4 release videos (G1) | experimental |
| Rotation -90 and 180, variable frame rate, interlaced, HDR | every frame of the generated edge-case videos equal to torchcodec's digests on FFmpeg 9.0.1 | experimental |
| Other CPUs | as for FFmpeg 8 | untested |

torchcodec 0.17.0 on FFmpeg 9.0.1 decodes every frame of the fixture videos to the same RGB as on FFmpeg 8.1 (the protected digests), so the FFmpeg 9 gate is the same digests (`model-pipeline/reference/embeddinggemma2/torchcodec_eg2_frames.py`). The binding tolerance section `decoder_ffmpeg9` (exact) is protected since 2026-10-15 (`model-pipeline/protected/embeddinggemma-2/decoder-ffmpeg9/`, appended to both video profiles' `tolerances.yaml`): the tests read it, and `model-pipeline evaluate` checks the FFmpeg 9 frames in G1 next to the FFmpeg 8 ones.

## Licenses and patents

The package is MIT OR Apache-2.0. Its FFmpeg bindings are generated from FFmpeg's headers and contain only interface facts (function signatures, the used struct fields' offsets, constants), not FFmpeg code; see `THIRD-PARTY-NOTICES.md`. It restates the FFmpeg calls of torchcodec v0.17.0 (BSD 3-Clause, Meta; no source copied).

**FFmpeg is user-supplied and not distributed by Starlings.** FFmpeg is LGPL 2.1 or later (LGPL 3.0 with `--enable-version3`); a GPL build makes the program that loads it subject to the GPL, and a nonfree build is not redistributable. Whoever ships FFmpeg with an application carries FFmpeg's obligations (for the LGPL: dynamic linking, license text, source or an offer, unmodified library names, replaceability).

**Codec patents (information, not legal advice).** H.264/AVC and H.265/HEVC are covered by patent licensing pools (Via LA; Access Advance for HEVC). VP9 and AV1 are offered royalty-free by their developers (the AOMedia patent license for AV1), though third parties assert claims. FFmpeg's decoders come with no patent license. Because this package ships no codec, codec licensing is the deployer's responsibility; seek legal advice for your distribution.

**Publishable.** The legal review of the FFmpeg adapters (decision D2) was cleared on 2026-10-15. Like the FFmpeg 8 adapter, this package is publishable from the next release; it is not on nuget.org yet.

Deployments of the model must adhere to the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) (model card).
