# embeddinggemma-2 `image-fp32-cuda` on the P-030 image release set

Run on 2026-10-14 at `472e187` on the RTX 3090, GPU 0 only. The set, media and reference tensors are those of the CPU profile ([`../embeddinggemma2-image-fp32-cpu/`](../embeddinggemma2-image-fp32-cpu/README.md)): 6-thread regenerated `.npz` files whose arrays all equal `tensors.json`. The bounds come from `protected/embeddinggemma-2/image-fp32-cuda/tolerances.yaml`. Both graphs ran on `CUDAExecutionProvider` with `use_tf32: 0`.

| Gate | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- |
| G1 | **passed** | ids exact 19/19; decoded RGB, resized pixels, pixel values and positions exact on 20/20 images | [`g1-g2-image-release/report.md`](g1-g2-image-release/report.md) |
| G2 | **passed**, first divergence none | embedding max abs 1.53e-6 (1e-5); min cosine 0.999999999896 (0.9999995); soft tokens row-relative 1.06e-4 (1e-3); token embeddings row-relative 6.27e-5 (1e-3); batches vs single within bounds | same |

The same finding as on CPU applies: `commons-image-14` (79 MP) needed `--max-image-pixels 80000000`, because the package default (50 MP) refuses it. The default stays at 50 MP (coordinator decision), and the capability `image_over_default_pixel_limit_with_option` is `experimental`. Wall clock: .NET dump 48.3 s.
