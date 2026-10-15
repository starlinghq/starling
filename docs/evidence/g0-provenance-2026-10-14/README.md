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
| `embeddinggemma-2/video-fp32-cpu` | same | 17/17 | public | same; decoding needs the user's FFmpeg through `Starlings.Models.EmbeddingGemma2.Video.FFmpeg`/`FFmpeg9` | **open** (checklist) |
| `embeddinggemma-2/video-fp32-cuda` | same | 17/17 | public | same | **open** (checklist) |
| `qwen3.5-0.8b/ortgenai-fp32x-cpu` | microsoft/onnxruntime-genai@`83de55a1` (builder), Qwen/Qwen3.5-0.8B@`2fc06364` | 9/9 (5 upstream, 4 from Starlinghq/qwen3.5-0.8b-ortgenai@`95a690c1`) | public | MIT (builder) / Apache-2.0 / Apache-2.0, download | **passed** |
| `qwen3.5-0.8b/ortgenai-fp32x-cuda` | same | 9/9 | public | same | **passed** |
| `qwen3.5-0.8b/ortgenai-fp16x-cuda` | same | 9/9 (fp16x bundle, also downloaded in full) | public | same | **passed** |

## What stays open

- **EmbeddingGemma 2 video (CPU and CUDA).** The lock, the anonymous reference and the rights all check out (17/17 files). But a video profile cannot run without one of the FFmpeg adapters. Both are held from publishing until legal review D2 (`HELD` in `.github/workflows/release.yml`; the preview.6 release record). Until D2 settles them, these profiles have no clear distribution route, and the release rule keeps G0 open. The manifests say so above their G0 entries.

## Also changed

- The EmbeddingGemma 2 image manifests listed `origin.code` and `origin.weights` twice. The copies were identical, and YAML keeps the last one. The duplicates are removed.
- The stale "hosting addresses are placeholders" known issue in the image and video manifests is replaced with the public hosting record.
