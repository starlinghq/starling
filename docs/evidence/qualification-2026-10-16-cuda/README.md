# Binding CUDA qualification round, 2026-10-16: G4 soaks and G5 matched benchmarks

This is the binding CUDA round at the final code: binding commit `e404702fa0fdbf91fd506757f1a5c15522d023dc` (`origin/main` after #61, #63 and #66). It covers all seven CUDA profiles. The rule is the one fixed in [`../qualification-2026-10-15/README.md`](../qualification-2026-10-15/README.md): these results are binding, and nothing was rerun to chase a pass. The EmbeddingGemma 2 text and both Qwen CUDA soaks, and every CUDA G5 run, were repeated because #54, #55, #61, #63 and #66 changed the code they measure. Their 2026-10-15 results described older code.

The rules were not changed for this round: the soak growth rule in `Starlings.Soak.Tool` (`Analysis.cs`), and the G5 budgets in `model-pipeline/protected/performance-budgets.yaml`, checked by `model-pipeline bench g5`.

## Environment (fixed before any run)

- **Code.** A fresh checkout on the network volume at the binding commit, clean. The CUDA harnesses (`Starlings.Benchmarks.*Harness.Cuda`) and the soak tool (`-p:UseCuda=true`) were built from it in Release with `RestoreLockedMode=true`. Every pod's `env.txt` records the commit and `dirty: 0 paths`, and each G5 verdict records `qualifying: yes (run counts and clean checkout)`.
- **Hosts.** RunPod secure cloud EU-CZ-1, one RTX 3090 per pod. Pods `starling-q-b1` to `starling-q-b4`, at most four at once. Each pod ran one job at a time, so no other Starling job was running on it. The host CPUs were shared with other tenants. The load average at each job's start is in the pod's `status.txt` (6 to 60 on 256 logical CPUs).
- **CUDA.** CUDA 13 and cuDNN 9 from the volume's pinned nvidia-* wheels on `LD_LIBRARY_PATH`. The soaks use the `eg2-cuda` venv (`model-pipeline/reference/embeddinggemma2/requirements-cuda.txt`). In G5, both sides use their runner's venv, as `run-g5-matched.sh` and `run-e5-matched.sh` do. `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0`, `NVIDIA_TF32_OVERRIDE=0`, and TF32 off in every session.
- **`MALLOC_MMAP_THRESHOLD_=65536`** was set only where the package README makes it a Linux requirement:
  - `embeddinggemma-2/video-fp32-cuda`: required, per the `Starlings.Models.EmbeddingGemma2` README, "Memory on Linux".
  - `gliner2-base-v1/onnx-fp32-cuda`: the `Starlings.Models.Gliner2` README requires it "in the environment of a Linux process that runs this profile", and the 2026-10-15 binding run and the CI soak set it too.

  It was not set for `image-fp32-cuda` (recommended only), e5, EmbeddingGemma 2 text or Qwen. For the two profiles that need it, it was set for the soak and for both sides of the G5 run. Each soak report's `allocator_and_runtime_env` records it.
- **Transparent huge pages.** `/sys/kernel/mm/transparent_hugepage/enabled` was already `madvise` on every pod (`always [madvise] never`). The container mounts `/sys` read-only, so the workers' attempt to write `madvise` failed. It was not needed, because the setting was already `madvise`. Each `env.txt` records the setting before and after the attempt, and every soak report records it as `transparent_hugepage`.
- **Library defaults at this commit.** These apply in every run: the container-aware intra-op thread count (#63), the cancel-safe CUDA runs (#61), and the heap trim on session release (#66).

## G4: one-hour CUDA soaks

Reports are in [`soak/`](soak/) (`<model>-<start>.json` raw, `.md` summary). Each soak used `--cuda --duration 1h`, with the runtime-default threads and 2 callers. The table gives settled `rss_anon` after warmup: the median, and the fitted growth over the post-warmup window with its first and second halves, in MiB. The limit is 32 MiB unless the report gives a higher one.

| Profile | Pod, GPU, driver | THP | `MALLOC_MMAP_THRESHOLD_` | Cycles / requests | Wrong / errors | Unloaded rss_anon | Loaded rss_anon | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multilingual-e5-small `onnx-fp32-cuda` | b1, RTX 3090, 580.126.20 | madvise | not set | 1,541 / 277,380 | 0 / 0 | 534; 67.8 (42.0, -1.7) | 637; 61.9 (39.4, -1.8) | **PASS** |
| gliner2-base-v1 `onnx-fp32-cuda` | b2, RTX 3090, 580.126.20 | madvise | 65536 | 857 / 77,130 | 0 / 0 | 445; 4.1 (3.1, 0.7) | 520; 4.0 (2.6, 0.7) | **PASS** |
| embeddinggemma-2 `onnx-fp32-cuda` (text) | b1, RTX 3090, 580.126.20 | madvise | not set | 760 / 41,040 | 0 / 0 | 760; -1.8 (6.9, 3.5) | 1,360; -2.0 (9.3, 9.9) | **PASS** |
| embeddinggemma-2 `image-fp32-cuda` | b4, RTX 3090, 580.126.20 | madvise | not set | 799 / 3,196 | 0 / 0 | 847; 12.6 (9.9, 14.5) | 1,747; 29.4 (36.1, 16.9) | **PASS** |
| embeddinggemma-2 `video-fp32-cuda` | b3, RTX 3090, 580.159.03 | madvise | 65536 | 553 / 1,659 | 0 / 0 | 711; 4.1 (3.4, 2.2) | 862; 6.0 (3.3, 5.7) | **PASS** |
| qwen3.5-0.8b `ortgenai-fp32x-cuda` | b2, RTX 3090, 580.126.20 | madvise | not set | 710 / 7,810 | 0 / 0 | 760; 27.8 (18.4, 10.5) | 892; 29.6 (22.3, 7.5) | **PASS** |
| qwen3.5-0.8b `ortgenai-fp16x-cuda` | b4, RTX 3090, 580.126.20 | madvise | not set | 1,043 / 11,473 | 0 / 0 | 824; 17.2 (21.8, -10.1) | 893; 24.4 (23.3, -1.6) | **PASS** |

Notes:

- **e5.** The fitted growth over the whole window exceeds the limit (67.8 and 61.9 MiB). The rule fails a series only when the growth continues in both halves, and here the second half is flat (-1.7 and -1.8 MiB). This is a one-time rise to a high-water mark in the first half, which the rule classes as a plateau. The 2026-10-15 run of the older code grew +120 MiB/h, with 82 MiB in its first half and 12 MiB in its second.
- **GLiNER2 CUDA.** No wrong output in 77,130 requests. The 2026-10-15 binding run had 1 wrong output in 80,100 requests, fixed by #61 (`../gliner2-cuda-wrong-output/`).
- **`private` (VmData).** On image, video and e5, `private` grows. It is address space, reported but not gating on Linux (column Gates of each report).
- **Idle build server.** On b1, the build ran just before the e5 soak. The idle .NET build server (`VBCSCompiler`) was still resident when the soak started (`pods/starling-q-b1-run1/env.txt`) and exits on its own after an idle timeout. It ran no build during the soak.

## G5: matched CUDA benchmarks (C# against the Python same-backend run)

Commands: `run-g5-matched.sh <model> cuda <out> 3 6 10`, or `run-e5-matched.sh cuda <out> 3 6 10` for e5. That is 3 rounds, 6 threads and 10 cold processes per round, with both sides on the same pod. Each run was then checked with `model-pipeline bench g5 <out>`. Raw files are in [`g5/`](g5/): `<model>-cuda.jsonl`, the verdict `<model>-cuda.jsonl.verdict.txt`, and the load log `<model>-cuda.load.tsv`, which samples the load average and GPU every 5 s. Every verdict notes that the C# side (NuGet) and the Python side (pip) load different native ONNX Runtime 1.30.0 builds. That is context, not a verdict input.

| Profile | Pod, GPU, driver | THP | Verdict | Key numbers (C# / Python, median ratio [95% CI]) |
| --- | --- | --- | --- | --- |
| e5 `onnx-fp32-cuda` | b1, RTX 3090, 580.126.20 | madvise | **PASS**, qualifying | warm 0.915 to 1.028 (budget 1.05) |
| EmbeddingGemma 2 `onnx-fp32-cuda` (text) | b1, RTX 3090, 580.126.20 | madvise | **PASS**, qualifying | warm 0.414 to 1.095 (budget 1.10); short-b1 1.095 [0.997, 1.119] |
| EmbeddingGemma 2 `image-fp32-cuda` | b4, RTX 3090, 580.126.20 | madvise | **PASS**, qualifying | preprocessing 1.2 to 3.8% of embed (budget 5%); warm 0.785 to 1.029 (photo-b1 0.991) |
| EmbeddingGemma 2 `video-fp32-cuda` | b3, RTX 3090, 580.159.03 | madvise | **PASS**, qualifying | preprocessing 1.2 to 2.5% of embed (budget 5%); warm 0.710 to 0.885 (1080p 0.710) |
| GLiNER2 `onnx-fp32-cuda` | b2, RTX 3090, 580.126.20 | madvise | **PASS**, qualifying | prepare + decode 0.6 to 1.5% of inference (budget 5%); inference 0.959 to 0.996 |
| Qwen `ortgenai-fp32x-cuda` | b2, RTX 3090, 580.126.20 | madvise | **PASS**, qualifying | decode 0.975 to 1.023 (min 0.95); time to first token 0.981 to 1.012 (budget 1.10) |
| Qwen `ortgenai-fp16x-cuda` | b4, RTX 3090, 580.126.20 | madvise | **PASS**, qualifying | decode 1.042 to 1.058; time to first token 0.922 to 0.997 |

The three profiles that failed G5 on 2026-10-15 now pass:

| Profile | 2026-10-15 | Now |
| --- | --- | --- |
| image | preprocessing 8 to 48% of embed; photo warm ratio 1.118 | preprocessing at most 3.8%; photo 0.991 |
| video | preprocessing 11 to 36%; 1080p warm ratio 1.229 | preprocessing at most 2.5%; 1080p 0.710 |
| GLiNER2 | prepare + decode 8 to 13% of inference | at most 1.5% |

EmbeddingGemma 2 text `long-b4` at 0.414 has a wide interval, [0.392, 0.954], and the Python side was slow there. It passes either way.

## Interrupted first attempt (infrastructure, not a result)

At about 02:10:47 UTC, every process running from the network volume stopped at the same moment on all four pods. Writes to the volume stopped persisting: new files were created with zero length, and `echo … >` returned 0. The interrupted jobs had no verdict and wrote no report:

- the EmbeddingGemma 2 text soak (b1, cycle 269);
- the Qwen fp32x soak (b2, cycle 220);
- the video G5 run (b3, round 3 of 3);
- the image G5 run (b4, round 3, cold starts).

The volume (50 GB) held about 42 GB at the time, including about 2.4 GB of CI cache written by the separate CI pods that share it. Its quota was the likely cause, but this was not established. The workers' `status.txt` for that attempt is in `pods/starling-q-b*-run1/`.

Six jobs had finished before the interruption, with their reports complete, and they stand: the soaks of e5, GLiNER2, video and image, and the e5 and GLiNER2 G5 runs. Each interrupted job was then run once, from the start, with outputs on the pods' container disks (`pods/starling-q-b*-run2/`). Each interrupted job therefore has exactly one completed run, and that run is the one reported here. The Qwen fp16x soak and G5 run had not started before the interruption; they ran in the second attempt.

## Files

- `soak/`: the seven soak reports.
- `g5/`: the seven G5 raw files, verdicts and load logs. `gliner2-cuda.gliner2-inputs.jsonl` holds the exact tensors the C# `Prepare` built, which the Python inference-stage runner consumed.
- `pods/<pod>-run<n>/env.txt`: commit, THP before and after, `/sys` mount, GPU, driver, CPU, glibc, load and top processes at worker start.
- `pods/<pod>-run<n>/status.txt`: each job's start and end, exit code, wall time and load at start.
