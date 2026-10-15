# G0 provenance, 2026-10-14

Plan section 10, G0: an immutable source and asset lock, an accessible reference, and rights recorded. Release rule: no public bundle without a clear distribution route.

This record checks all 13 registry profiles against that definition. G0 is recorded as `passed` for 11 of them. The two EmbeddingGemma 2 video profiles stay `checklist`, because the FFmpeg adapters that decode their input are held pending legal review D2.

## How it was checked

| Clause | Check | Evidence |
| --- | --- | --- |
| Immutable source lock | `origin.code` and `origin.weights` are pinned by full 40-hex commit SHAs. Every `origin.files` entry has a SHA-256 and a size. Every file Starling hosts is addressed at a full commit (`/resolve/<sha>/`) and recorded `source_status: public`. | `tools/check-provenance.py` (new), [check-provenance.log](check-provenance.log) |
| Accessible reference | With no token and no cookies: each pinned code and weights commit resolves, and every file of each lock is served from where the packages download it (the upstream Hub repository at the pinned commit, or Starling's hosting). Large files are checked through the Hub's `X-Linked-Etag` (the LFS SHA-256) and `X-Linked-Size`; small files are downloaded and hashed. | [check-provenance.log](check-provenance.log): `PROVENANCE PASS`, every lock file of all 13 manifests served with its pinned SHA-256 and size |
| Accessible reference, full download | The packages' own download step fetched the files Starling hosts into empty caches with no token and no seed, and verified size and SHA-256 of each. | preview.6 drill ([rc-drill.log](../../releases/ai-0.1.0-preview.6/rc-drill.log), step 11): GLiNER2 graph, EmbeddingGemma 2 text graph, Qwen 3.5 fp32x bundle. This record's drills ([G6](../g6-distribution-2026-10-14/README.md)): the two **EmbeddingGemma 2 vision graphs** (671,256,072 and 1,084,327,898 bytes; `eg2-image-cpu-drill.log` lines 335-336), the **Qwen 3.5 fp16x** bundle (1,572,143,104-byte data file; `cuda-drill.log` line 312), and again the GLiNER2 graph and Qwen fp32x bundle (`cpu-drill.log`) |
| Rights recorded | `rights.code_license`, `weights_license`, `tokenizer_terms` and `redistribution` are present in every manifest, with notes on the hosting route and attribution. | [check-provenance.log](check-provenance.log); the manifests' `rights.notes` |
| Distribution route | The route is "download": the packages carry no weights, and an explicit acquisition step fetches the pinned files from the upstream Hub or from Starling's public Hugging Face repositories, which carry the license, attribution and notice files (`model-pipeline/reference/*/NOTICE-*.txt`). The packages that run each profile are in the publishable set of the preview.6 release record. | [ai-0.1.0-preview.6.md](../../releases/ai-0.1.0-preview.6.md), "Hosting" and "Packages" |

The provenance check is anonymous: `HF_TOKEN` was unset and no credentials were sent. Run it again with `model-pipeline/.venv/bin/python -I tools/check-provenance.py`.

## Per profile

| Profile | Code and weights pins | Files (SHA-256 and size, anonymous) | Starling hosting | Rights | G0 |
| --- | --- | --- | --- | --- | --- |
| `multilingual-e5-small/onnx-fp32-cpu` | intfloat/multilingual-e5-small@`614241f6` | 12/12 (upstream Hub) | none (upstream ships the ONNX graph) | MIT / MIT / MIT, download | passed (already, 2026-10-09; unchanged) |
| `multilingual-e5-small/onnx-fp32-cuda` | same | 12/12 | none | same | **passed** |
| `gliner2-base-v1/onnx-fp32-cpu` | fastino-ai/GLiNER2@`55656fbf`, fastino/gliner2-base-v1@`f9634218` | 10/10 (9 upstream, graph from Starlinghq/gliner2-base-v1-onnx@`c02c3350`) | public | Apache-2.0 ×3, download | **passed** |
| `gliner2-base-v1/onnx-fp32-cuda` | same | 10/10 | public | same | **passed** |
| `embeddinggemma-2/onnx-fp32-cpu` | google/embeddinggemma-2@`914f7f89` | 15/15 (text graph from Starlinghq/embeddinggemma-2-onnx@`92a02ae2`) | public | Apache-2.0 ×3, download; Gemma Prohibited Use Policy sentence carried | **passed** |
| `embeddinggemma-2/onnx-fp32-cuda` | same | 15/15 | public | same | **passed** |
| `embeddinggemma-2/image-fp32-cpu` | same | 17/17 (vision and multimodal graphs from Starlinghq/embeddinggemma-2-vision-onnx@`1a22d40b`, also downloaded in full) | public | same; SkiaSharp (MIT, bundled native notices) for decoding | **passed** |
| `embeddinggemma-2/image-fp32-cuda` | same | 17/17 | public | same | **passed** |
| `embeddinggemma-2/video-fp32-cpu` | same | 17/17 | public | same; decoding needs the user's FFmpeg through `Starlings.Models.EmbeddingGemma2.Video.FFmpeg`/`FFmpeg9` | **passed** (2026-10-15, [below](#video-2026-10-15)) |
| `embeddinggemma-2/video-fp32-cuda` | same | 17/17 | public | same | **passed** (2026-10-15, [below](#video-2026-10-15)) |
| `qwen3.5-0.8b/ortgenai-fp32x-cpu` | microsoft/onnxruntime-genai@`83de55a1` (builder), Qwen/Qwen3.5-0.8B@`2fc06364` | 9/9 (5 upstream, 4 from Starlinghq/qwen3.5-0.8b-ortgenai@`95a690c1`) | public | MIT (builder) / Apache-2.0 / Apache-2.0, download | **passed** |
| `qwen3.5-0.8b/ortgenai-fp32x-cuda` | same | 9/9 | public | same | **passed** |
| `qwen3.5-0.8b/ortgenai-fp16x-cuda` | same | 9/9 (fp16x bundle, also downloaded in full) | public | same | **passed** |

## What stays open

- ~~**EmbeddingGemma 2 video (CPU and CUDA).**~~ Closed on 2026-10-15; see below. On 2026-10-14 the lock, the anonymous reference and the rights all checked out (17/17 files), but both FFmpeg adapters were held from publishing until legal review D2, so the profiles had no clear distribution route.

## Video, 2026-10-15

G0 is now `passed` for `embeddinggemma-2/video-fp32-cpu` and `video-fp32-cuda`. The model lock is unchanged from the table above: the 17 files are served anonymously with their pinned SHA-256, and the vision graphs were downloaded again with no token on this date ([G6](../g6-distribution-2026-10-14/README.md#video-2026-10-15)). What changed is the decoding path.

| Clause | Check | Result |
| --- | --- | --- |
| Distribution route | Legal review D2 cleared on 2026-10-15 (`docs/kickoff.md`, EmbeddingGemma 2 admission and package versions rows). `.github/workflows/release.yml` holds nothing (`HELD: ""`). Both adapter READMEs say "Publishable"; the FFmpeg 9 project file's stale "held" comment is corrected here. | both adapters publishable from the next release |
| Generated bindings, FFmpeg 8 | `tools/generate-ffmpeg-bindings.py --major 8 --check` against `include/` of the pinned BtbN LGPL shared build `ffmpeg-n8.1.3-17-gc389840894-linux64-lgpl-shared-8.1.tar.xz` (archive SHA-256 `bf3db18a4a4bdc066f6b74ee5ec3d7b7be7d6738dcfd7162eae3e09cb857c6b2`, as `fetch_eg2_ffmpeg.sh` pins it; FFmpeg commit `c3898408946f6cbfe8e244372c250366d5639d00`) | `ok: …Video.FFmpeg/FFmpegNative.g.cs matches the headers` (byte-identical regeneration; SHA-256 `32cfe97129c71d2197c7bdd4c2580948b7c187529637cae8cdf514c1775e71e2`) |
| Generated bindings, FFmpeg 9 | `fetch_eg2_ffmpeg9_headers.sh` installed the headers of FFmpeg n9.0.1 at commit `bf1b838f2ab88b4f8fd83443325c782ea0e0f7fa` into an empty directory (shallow fetch from `git.ffmpeg.org`, `configure` and `make install-headers`). Then `--major 9 --check` against them, and against the distribution's `/usr/include` (Arch `ffmpeg 2:9.0.1-4`) | `ok: …Video.FFmpeg9/FFmpegNative.g.cs matches the headers` for both (SHA-256 `e31fd244df3e45997ff790f830cf0b8d67a1408f211531fe461b387ed899b424`) |
| Notices | `ai-dotnet/THIRD-PARTY-NOTICES.md` ("FFmpeg headers", "FFmpeg (user-supplied, not distributed)", codec patents; torchcodec, BSD-3-Clause) and `ai-dotnet/PROVENANCE.md` (the restated torchcodec calls at `d9dc5423…`). Each adapter package carries `THIRD-PARTY-NOTICES.md` and its README; the packed files are `lib/…dll`, `…xml`, `README.md` and `THIRD-PARTY-NOTICES.md`, with no FFmpeg library (the drill's step 4 fails on one) | present in both packed adapters |
| User-supplied FFmpeg, LGPL expected | The drill's clean consumers loaded the user's LGPL builds through `--ffmpeg-dir`: n8.1.3-17-gc389840894 and the BtbN LGPL build n9.0.2-28-g669f5ab083 (`ffmpeg-n9.0.2-28-g669f5ab083-linux64-lgpl-shared-9.0.tar.xz` from BtbN/FFmpeg-Builds release `autobuild-2026-10-15-13-03`, SHA-256 `5838264a2cff06ddc6b7b2fc43edeb51358ead38fd56d82f83f49c270fe0f5ff`). Every library reports "LGPL version 3 or later" | `ffmpeg: n8.1.3… (LGPL)`, `ffmpeg: n9.0.2… (LGPL)` ([eg2-video-cpu-drill.log](../g6-distribution-2026-10-14/eg2-video-cpu-drill.log) lines 353, 371) |
| GPL and nonfree rejected unless opted in | The same packed samples pointed at GPL builds (BtbN GPL shared n8.1.3 for the FFmpeg 8 adapter; the distribution's GPL FFmpeg 9.0.1 in `/usr/lib` for the FFmpeg 9 adapter), with no `AllowNonLgplBuild` | both refused with `EmbeddingGemma2FFmpegLicenseException`, exit 3 (lines 387 to 390) |

The FFmpeg builds stay outside the repository and outside every package; the drill's checks find no FFmpeg library in the package cache or any build output.

## Also changed

- The EmbeddingGemma 2 image manifests listed `origin.code` and `origin.weights` twice. The copies were identical, and YAML keeps the last one. The duplicates are removed.
- The stale "hosting addresses are placeholders" known issue in the image and video manifests is replaced with the public hosting record.
