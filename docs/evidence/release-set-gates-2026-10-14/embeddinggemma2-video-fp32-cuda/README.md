# embeddinggemma-2 `video-fp32-cuda` on the P-030 video release set

Run on 2026-10-14 at `472e187` on the RTX 3090, GPU 0 only. Decoding used the same pinned FFmpeg 8.1.3 LGPL build as the CPU run ([`../embeddinggemma2-video-fp32-cpu/`](../embeddinggemma2-video-fp32-cpu/README.md)), and the graphs ran on `CUDAExecutionProvider` with `use_tf32: 0`.

| Gate | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- |
| G1 | **failed** | The same mismatch as on CPU: decoded frames, resized pixels and pixel values differ on the 3/12 video inputs that sample `commons-video-02` (Ogg Theora), and on the still image taken from it. Ids, frame indices and positions are exact. | [`g1-g2-video-release/report.md`](g1-g2-video-release/report.md) |
| G2 | **failed** | embedding max abs 8.88e-3 (1e-5), min cosine 0.99793, worst `mm-image-and-video`. Rows without `-02` measure 2.0e-7 to 2.8e-6 per row; rows with it measure 2.6e-3 to 8.9e-3. Soft-token and token-embedding checks not run (4-thread reference tensors). | same |

The failure comes from decoding, which runs on the CPU, so the provider does not change it. Wall clock: .NET dump 61.5 s.
