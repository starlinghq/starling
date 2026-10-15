# embeddinggemma-2 `image-fp32-cuda` on the P-030 image release set

Run on 2026-10-14 at `472e187` on the RTX 3090, GPU 0 only. The set, media and reference tensors are those of the CPU profile ([`../embeddinggemma2-image-fp32-cpu/`](../embeddinggemma2-image-fp32-cpu/README.md)): 6-thread regenerated `.npz` files whose arrays all equal `tensors.json`. The bounds come from `protected/embeddinggemma-2/image-fp32-cuda/tolerances.yaml`. Both graphs ran on `CUDAExecutionProvider` with `use_tf32: 0`.

| Gate | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- |
| G1 | **passed** | ids exact 19/19; decoded RGB, resized pixels, pixel values and positions exact on 20/20 images | [`g1-g2-image-release/report.md`](g1-g2-image-release/report.md) |
| G2 | **passed**, first divergence none | embedding max abs 1.53e-6 (1e-5); min cosine 0.999999999896 (0.9999995); soft tokens row-relative 1.06e-4 (1e-3); token embeddings row-relative 6.27e-5 (1e-3); batches vs single within bounds | same |
| G3 (2026-10-15) | **passed** | `image-retrieval-v1`, 149 queries, 149 images: nDCG@10 **0.9222** (minimum 0.916), MRR@10 **0.9008** (minimum 0.893); recall@1 0.8456, recall@10 0.9866. The upstream reference scores 0.9222 / 0.9008. | [`g3-retrieval/report.md`](g3-retrieval/report.md) |

The same finding as on CPU applies: `commons-image-14` (79 MP) needed `--max-image-pixels 80000000`, because the package default (50 MP) refuses it. The default stays at 50 MP (coordinator decision), and the capability `image_over_default_pixel_limit_with_option` is `experimental`. Wall clock: .NET dump 48.3 s.

## G3 on the image and video retrieval sets (2026-10-15)

The set is `embeddinggemma-2/release/image-retrieval-v1`, added in #57 (protected-change): 149 queries, 149 images from Wikimedia Commons. Each query is a file's own Commons description, and that file is the query's only relevant document. The set's `SOURCE.md` gives the sources, licenses and selection. The budget comes from `protected/embeddinggemma-2/image-fp32-cuda/task-quality.yaml`. It is the upstream reference's score on this exact set minus a stated margin, and it was committed before any .NET run on the set.

| Metric | .NET (CUDA) | Upstream reference (PyTorch CPU fp32) | Minimum |
| --- | ---: | ---: | ---: |
| nDCG@10 | 0.9222 | 0.9222 | 0.916 |
| MRR@10 | 0.9008 | 0.9008 | 0.893 |
| recall@1 | 0.8456 | 0.8456 | |
| recall@10 | 0.9866 | 0.9866 | |

**G3 passed, exit 0.** The evaluator verified the budget against Git, and the budget matched the data+metadata digest `43b3a8d41907ce5fda042554eb8485378b5cacacfa4d994cfebd6bc2631f0607`. Every metric equals the reference's exactly. Compared with the reference embeddings, the .NET embeddings measure max abs 5.3e-6 per element, with min cosine 0.999999999. That is measured, not a G3 bound.

- **Producer.** `Starlings.Fidelity.Tool retrieval --model embeddinggemma2-image` ran at `1aa26e5` (main after #57) on CUDA (`Starlings.Fidelity.Tool.Cuda`, `--device cuda`, RTX 3090, `CUDA_VISIBLE_DEVICES=0`, `use_tf32: 0`, CUDA 13 and cuDNN 9 from the pinned `eg2-cuda` wheels). Before embedding anything, it checked every media file against `corpus.jsonl`'s SHA-256. Queries used the `query` prompt through the multimodal embedder. Each image was embedded alone, with no prompt, through the SkiaSharp decoder. Everything ran at the package defaults. Wall clock: 78 s.
- **Reference.** `reference/embeddinggemma2/reference_eg2_media_retrieval.py` (#53) ran on sentence-transformers 6.1.0 and PyTorch 2.14.1+cpu at 14 threads, in 471 s, with embeddings SHA-256 `18694dbb61e80e7ca10b5280f67c2905f00dee98ba524a2afec1185d71130d15`.
- **Machine.** Both ran on RunPod pod `starling-q-eg2g3`: secure cloud EU-CZ-1, AMD EPYC 7H12 (31-CPU quota), RTX 3090, .NET SDK 10.0.100. Wall clock is not a benchmark.
- **Outside Git.** The candidate embeddings (298 rows) are `de94bc7a20c8574245728dd4261d10c9af675a40801b7fe648e1c69ace138c6d`. The media are fetched with the set's `SOURCE.md` commands.

```sh
SET=model-pipeline/protected/embeddinggemma-2/release/image-retrieval-v1
dotnet <Starlings.Fidelity.Tool.Cuda build>/Starlings.Fidelity.Tool.dll retrieval --model embeddinggemma2-image --dataset $SET \
  --media <media dir> --out <run>/embeddings.jsonl --device cuda --threads 16
model-pipeline evaluate model-pipeline/registry/embeddinggemma-2/image-fp32-cuda/manifest.yaml --g3 retrieval \
  --dataset $SET --results <run>/embeddings.jsonl --out <run>/g3-retrieval
```
