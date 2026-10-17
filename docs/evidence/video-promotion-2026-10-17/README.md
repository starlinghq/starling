# EmbeddingGemma 2 video profiles to Stable (2026-10-17)

This folder holds the evidence for promoting `embeddinggemma-2/video-fp32-cpu` and `embeddinggemma-2/video-fp32-cuda` to Stable (`docs/kickoff.md`, "Stable video profiles"). The source is the full qualification run [39023915026](https://github.com/starlinghq/starling/actions/runs/39023915026), dispatched on `main` at `6ea1bfa` (includes #90, #93 and #94). Every job of the run passed. The files under `run-39023915026/` are copied unchanged from the run's artifacts (`qualification-release-gates-4`, `qualification-soak-embeddinggemma2-video`, `qualification-cuda`).

## Why the profiles were held back, and what closed it

| Blocker | Closed by |
| --- | --- |
| The hosted one-hour CPU video soak failed the growth rule in every run up to 2026-10-16 (about 280 MB/h). | Library changes: no ONNX Runtime CPU arena (#83), dedicated run threads (#86), and on the image and video path heap trims after load and after each run plus reused request buffers (#93). Three consecutive full runs passed with every soak twice (#93). This run passed too (below). |
| Release-set G2 on AVX-512 hosted runners: token embeddings 1.83e-3 against 1e-3. | Diagnosed in #89: ONNX Runtime's AVX-512 Softmax kernels in the vision graph, and the PyTorch reference itself moves by up to 2.09e-3 between CPUs. The maintainer recalibrated that one intermediate bound to 4e-3 from the reference's own spread (#90, protected change). The final-embedding bound (1e-5) and every other bound are unchanged. |
| Video CUDA had no green per-release run on the contract GPU: the first run (38857860186) landed on an RTX 4090 and failed G2 on the final embedding (1.009e-5 against 1e-5). | The `cuda (runpod)` job is pinned to the RTX 3090. Every video CUDA check passed there in run 39012176735, which then timed out on an unrelated stage (fixed in #94), and again in this run. |

## Run 39023915026: `embeddinggemma-2/video-fp32-cpu` (hosted)

| Check | Runner | Result |
| --- | --- | --- |
| Release gates G1, G2 (video release set, 11 rows; both FFmpeg decoder checks exact) | AMD EPYC 9V74, **AVX-512** (`run-39023915026/cpu/gates/machine.txt`) | passed. Embedding max abs **5.72e-6** (bound 1e-5); min cosine 1.0 (0.9999995); soft tokens 2.63e-4 row-relative (1e-3); token embeddings **1.83e-3** row-relative (4e-3; the old 1e-3 would have failed, as #89 predicted for this CPU); padded batches 0 (1e-5). G2 read the hosted reference tensors, 11/11 verified against the protected `hosted-reference.json`. |
| G3, `video-retrieval-v1` | same | passed. nDCG@10 0.8126 (minimum 0.797), MRR@10 0.7714 (minimum 0.751). |
| One-hour soak | AMD EPYC 7763, AVX2 | PASS: every gating series plateaus, with no wrong output or error. The loaded working set's fitted growth is 15.7 MiB against a limit of 43.6, and the unloaded working set's is 5.2 MiB against 32.0. |

Reports: `run-39023915026/cpu/gates/g1-g2-video-release/report.md`, `run-39023915026/cpu/gates/g3-retrieval/report.md`, `run-39023915026/cpu/soak/`.

Between them, the hosted runners have now passed release-set G2 on an AVX2 CPU (EPYC 7763: token embeddings 3.81e-4, #89) and on AVX-512 CPUs (Xeon 8370C and 8573C, EPYC 9V45/9V74: 1.83e-3). The CPU contract covers both.

## Run 39023915026: `embeddinggemma-2/video-fp32-cuda` (RunPod, NVIDIA GeForce RTX 3090, driver 580.126.20)

| Check | Result |
| --- | --- |
| G1, G2 (video release set; FFmpeg 8 and 9 decoder checks exact) | passed. Embedding max abs **2.77e-6** (bound 1e-5); min cosine 1.0; soft tokens 2.50e-4 row-relative (1e-3); token embeddings **7.97e-4** row-relative (4e-3); padded batches at most 4.64e-7 (1e-5). |
| G3, `video-retrieval-v1` | passed. nDCG@10 0.8126, MRR@10 0.7714. |
| 10-minute soak smoke test with `MALLOC_MMAP_THRESHOLD_=65536` | passed. 86 load and unload cycles with no wrong output or error. The growth verdict (not binding in a short run) is PASS. |
| CUDA test projects | passed. The CUDA video parity tests and the multimodal generator over CUDA are skipped on the pod, because they need the video fixture set (see the contract). |

All 31 checks of the pod run passed (`run-39023915026/cuda/summary.md`, `summary.tsv`); the pod run took 106 minutes. The same numbers were measured in run 39012176735 on another RTX 3090 pod.

The RTX 4090 stays outside the contract. In run 38857860186 its final embedding was 1.009e-5 against the unchanged 1e-5 bound. Its token embeddings (3.34e-3) are within #90's 4e-3, so the final-embedding bound is the one it misses (`docs/evidence/gpu-regression-2026-10-16/README.md`).

## Pod reference stages (#94)

Run 39012176735 hit the pod-run limit because the PyTorch reference stages ran 128 threads under the container's CPU quota. This run reads that quota from cgroup v1: `cpu.cfs_quota_us` 3112500 / `cpu.cfs_period_us` 100000, which is 31 CPUs; `nproc` reports 256. It runs those stages with 31 threads (`run-39023915026/cuda/reference-threads.txt`):

| Stage | 38977347513 (128 threads) | 39012176735 (128 threads) | 39023915026 (31 threads) |
| --- | --- | --- | --- |
| EmbeddingGemma 2 text python-stage2 | 428 s | 3058 s | 163 s (CFS throttled 131 periods, 31 s) |
| e5 python-stage2 | 58 s | 279 s | 33 s (5 periods, 0 s) |
| GLiNER2 python-stage2 (`OMP_NUM_THREADS=4` throughout) | 29 s | 30 s | 32 s |

Every gate stays within its bound, and the final embeddings are unchanged:

- EmbeddingGemma 2 text: final embedding 9.8e-7, C# against Python 2.6e-7, both unchanged. The Python ONNX Runtime CUDA token embeddings against the 31-thread PyTorch CPU reference are 4.27e-6 relative per case, against 2.14e-6 with 128 threads (bound 1e-4).
- e5: every metric is unchanged to the printed digit (last hidden state 2.4e-6, embedding 1.3e-7).
