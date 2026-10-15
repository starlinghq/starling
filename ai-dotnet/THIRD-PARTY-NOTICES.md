# Third-party notices (ai-dotnet)

The ai-dotnet packages are licensed under the MIT License or the Apache License, Version 2.0, at your option. This file lists third-party material that they incorporate (ported source, see [`PROVENANCE.md`](PROVENANCE.md)) or bring in as package dependencies with notices that users should know about. Model weights and graphs are covered by their own notices (for EmbeddingGemma 2: `model-pipeline/reference/embeddinggemma2/NOTICE-embeddinggemma2-onnx.txt` and `NOTICE-embeddinggemma2-vision-onnx.txt`).

## PyTorch (ported algorithm)

`Starlings.Models.EmbeddingGemma2` (`TorchAntialiasResampler.cs`) reproduces PyTorch's uint8 antialiased bicubic resize (PyTorch v2.14.1, commit `5c4886908584029761b579af026dcfb627c84070`). PyTorch is licensed under the BSD 3-Clause license:

```
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

The full PyTorch notice (including the Caffe2 contributors) is the `LICENSE` file at that commit.

## Qwen3.5-0.8B chat template and transformers (ported source)

`Starlings.Models.Qwen35` (`Qwen35ChatTemplate.cs`) is a C# port of the `chat_template.jinja` of `Qwen/Qwen3.5-0.8B` at commit `2fc06364715b967f1860aea9cf38778875588b17`, Copyright 2026 Alibaba Cloud, licensed under the Apache License, Version 2.0. The tool-calling instructions are reproduced verbatim. `Qwen35ToolCallParser.cs` ports the tool-call argument casting of transformers (`src/transformers/utils/chat_parsing/response_parser.py`, `_coerce` and `_schema_types`, commit `a906d3c4b65095f2308b6a6a193e934d03b8eb5d`, tag v5.18.0), Copyright 2026 The HuggingFace Team, licensed under the Apache License, Version 2.0. Both files carry a modification notice. The Apache License 2.0 text is at https://www.apache.org/licenses/LICENSE-2.0.

## transformers and Pillow (restated arithmetic)

`EmbeddingGemma2ImageProcessor.GetTargetSize` and the patch order restate `get_aspect_ratio_preserving_size` and `convert_image_to_patches` of transformers (Apache-2.0, commit `c4c4605f961877852b89b97478536bb3a26b6fd1`, tag v5.19.0). `EmbeddingGemma2ImageDecoder.ApplyOrientation` follows the orientation table of Pillow's `ImageOps.exif_transpose` (MIT-CMU, commit `bb1d8e8ab8d29048624d96e3ee53cecf7c13d13d`, tag 12.3.0). No source text was copied.

## SkiaSharp (package dependency of Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp)

The decoding adapter depends on `SkiaSharp` 4.153.1 and `SkiaSharp.NativeAssets.Linux.NoDependencies` 4.153.1 (MIT, Copyright (c) 2015-2016 Xamarin, Inc., Copyright (c) 2017-2018 Microsoft Corporation). On Windows and macOS, `SkiaSharp` brings `SkiaSharp.NativeAssets.Win32` and `SkiaSharp.NativeAssets.macOS`. The native library `libSkiaSharp` incorporates third-party code listed in the `THIRD-PARTY-NOTICES.txt` file that ships inside each natives package (SHA-256 `21504c46c4c58aa64c1055bd2dcbc5f9a136b4b8c412ed3cc6740e22c5b127f5` for the Linux package): Skia, ANGLE, HarfBuzz, etc1, libpng, expat, FreeType, ICU, imgui, jsoncpp, libjpeg-turbo, libwebp, libmicrohttpd, piex, SDL, sfntly, SPIR-V Headers and Tools, zlib, and two entries that deserve attention:

- **GIF decoder (mozilla.org code, Netscape Communications Corporation).** Tri-licensed MPL 1.1 / GPL 2.0 or later / LGPL 2.1 or later; a recipient may use it under any one of the three. StarlingHq uses it under the MPL 1.1, which is a file-level copyleft on the unmodified upstream files inside the native library. StarlingHq does not modify or redistribute that source; the binary comes from Microsoft's natives package unchanged. The license block, verbatim from the notices file:

  ```
  Version: MPL 1.1/GPL 2.0/LGPL 2.1

  The contents of this file are subject to the Mozilla Public License Version
  1.1 (the "License"); you may not use this file except in compliance with
  the License. You may obtain a copy of the License at
  http://www.mozilla.org/MPL/

  Software distributed under the License is distributed on an "AS IS" basis,
  WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
  for the specific language governing rights and limitations under the
  License.

  The Original Code is mozilla.org code.

  The Initial Developer of the Original Code is
  Netscape Communications Corporation.
  Portions created by the Initial Developer are Copyright (C) 1998
  the Initial Developer. All Rights Reserved.

  Contributor(s):
    Chris Saari <saari@netscape.com>
    Apple Computer

  Alternatively, the contents of this file may be used under the terms of
  either the GNU General Public License Version 2 or later (the "GPL"), or
  the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
  in which case the provisions of the GPL or the LGPL are applicable instead
  of those above. [...] If you do not delete the provisions above, a recipient
  may use your version of this file under the terms of any one of the MPL, the
  GPL or the LGPL.
  ```

- **Adobe DNG SDK.** "This product includes DNG technology under license by Adobe Systems Incorporated." The DNG SDK License Agreement grants a non-exclusive, worldwide, royalty-free license to use, reproduce, prepare derivative works from, distribute and sublicense the software, provided copyright and other notices are kept and included in distributed copies. The adapter rejects DNG input (only PNG, JPEG, GIF, BMP and WebP are accepted), but the code is part of the native library.

Applications that ship the adapter ship the natives package and should carry its `THIRD-PARTY-NOTICES.txt`.

## torchcodec and transformers (restated video behaviour)

`EmbeddingGemma2VideoSampling` (in `Starlings.Models.EmbeddingGemma2`) restates `EmbeddingGemma2VideoProcessor.sample_frames` of transformers (Apache-2.0, commit `c4c4605f961877852b89b97478536bb3a26b6fd1`, tag v5.19.0) and NumPy's `linspace(..., dtype=int)` arithmetic. `EmbeddingGemma2VideoDecoder` (in `Starlings.Models.EmbeddingGemma2.Video.FFmpeg` and, from the same shared source, `Starlings.Models.EmbeddingGemma2.Video.FFmpeg9`) makes the FFmpeg calls that torchcodec makes on CPU (packet-scan metadata, `get_frames_at`'s seeks and seek-avoidance rule, display-matrix rotation, swscale or filter-graph RGB24 conversion) as read in torchcodec v0.17.0, commit `d9dc54238e173000bc3969eb3959777026ff9ae8` (BSD 3-Clause License, Copyright 2024 Meta). No source text was copied; `PROVENANCE.md` lists what was restated.

## FFmpeg headers (the interface of the user-supplied FFmpeg)

Each video adapter calls FFmpeg through its own bindings, `FFmpegNative.g.cs`, generated by `tools/generate-ffmpeg-bindings.py` from the headers of one FFmpeg release: `Starlings.Models.EmbeddingGemma2.Video.FFmpeg` from FFmpeg n8.1.3-17-gc389840894 (commit `c3898408946f6cbfe8e244372c250366d5639d00`, `--major 8`), `Starlings.Models.EmbeddingGemma2.Video.FFmpeg9` from FFmpeg n9.0.1 (commit `bf1b838f2ab88b4f8fd83443325c782ea0e0f7fa`, `--major 9`; headers installed from the source tree at that commit by `model-pipeline/reference/embeddinggemma2/fetch_eg2_ffmpeg9_headers.sh`). Each generated file contains no FFmpeg source code, only the interface facts needed to call the libraries: function names and parameter types, the sizes of the structs used and the offsets of the fields the adapter reads or writes, and constant values. The C compiler supplies the sizes, offsets and values (`sizeof`, `offsetof`) and checks the declarations against the headers.

FFmpeg's headers are part of FFmpeg (LGPL 2.1 or later). Section 5 of the LGPL 2.1 provides that a work using only numerical parameters, data structure layouts and accessors, and small macros and inline functions from a library's header files may be used without restriction. The adapters' code, the generated bindings included, is licensed MIT OR Apache-2.0. They depend on no third-party FFmpeg bindings package.

## FFmpeg (user-supplied, not distributed)

The video adapters load FFmpeg shared libraries (`libavcodec`, `libavformat`, `libavutil`, `libswscale`, `libavfilter`; FFmpeg 8.x for `Starlings.Models.EmbeddingGemma2.Video.FFmpeg`, FFmpeg 9.x for `Starlings.Models.EmbeddingGemma2.Video.FFmpeg9`) that **the user installs**; no Starlings package contains or downloads an FFmpeg binary. FFmpeg is licensed under the LGPL 2.1 or later by default (LGPL 3.0 with `--enable-version3`); builds configured with `--enable-gpl` are GPL as a whole, and `--enable-nonfree` builds are not redistributable. Each adapter expects an LGPL build and rejects GPL and nonfree builds unless `EmbeddingGemma2FFmpegOptions.AllowNonLgplBuild` is set. Whoever distributes FFmpeg binaries with an application carries FFmpeg's license obligations (for the LGPL: dynamic linking, the license text, the corresponding source or an offer for it, unmodified library names, the user's ability to replace the libraries).

Codec patents, for information and not legal advice: H.264/AVC and H.265/HEVC are covered by patent licensing pools (Via LA for AVC and HEVC; Access Advance for HEVC); VP9 and AV1 are offered royalty-free by their developers (the AOMedia patent license for AV1), although third parties assert claims. FFmpeg's decoders come with no patent license. Because the adapters ship no codec binaries and decodes with the user's FFmpeg, codec licensing is the deployer's responsibility; seek legal advice for a distribution decision.

## Excluded

`SixLabors.ImageSharp` is not referenced by any ai-dotnet project (Six Labors Split License, not OSI-approved; Release builds need a license key; less accurate than SkiaSharp against PIL). CI fails if it appears in any project's dependency graph.

`Sdcb.FFmpeg` (LGPL-3.0 bindings) and any package that bundles FFmpeg binaries (for example the `Sdcb.FFmpeg.runtime.*` packages) are not referenced by any ai-dotnet project; CI fails if a package whose name contains `Sdcb.FFmpeg`, or an FFmpeg native or runtime package, appears in any project's dependency graph, and if any packed Starlings package contains an FFmpeg library file.
