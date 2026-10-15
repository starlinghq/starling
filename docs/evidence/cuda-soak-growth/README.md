# CUDA soak growth: diagnosis (e5, EmbeddingGemma 2 image and video)

The one-hour CUDA G4 soaks of 2026-10-15 (RunPod, source `2624963`, CUDA 13 and cuDNN 9 from the pinned wheels, TF32 off, raw reports kept outside Git) had no wrong output and a flat managed heap, but `rss_anon` grew:

| Profile | GPU | Unloaded rss_anon | Post-warmup thirds, unloaded (MiB/h) | Verdict |
| --- | --- | --- | --- | --- |
| `multilingual-e5-small/onnx-fp32-cuda` | RTX 3090 | 588 to 670 MiB (120 MiB/h) | 112, 79, -5 (saturates) | FAIL (both halves above a quarter of the limit) |
| `embeddinggemma-2/image-fp32-cuda` | RTX 3090 | 85 MiB/h | 90, 79, 115 | FAIL |
| `embeddinggemma-2/video-fp32-cuda` | RTX 4090 | 105 MiB/h | 127, 109, 201 | FAIL |

**Conclusion: no leak in our code, in ONNX Runtime's session lifetime or in the CUDA libraries. Two causes:**

1. **A product defect, fixed here: the intra-op thread pool was sized from the host, not the container.** ONNX Runtime's default intra-op count is one thread per physical core of the host and ignores the cgroup CPU quota. On the soak pods (256 logical CPUs, a quota of about 31 CPUs) one e5 CUDA session ran **147 threads and 147 glibc malloc arenas** (2-minute smoke run, `smaps` sampler). Each arena keeps its own resident high-water mark, so the retained memory grows with the arena count. `Starlings.Runtime.Onnx` (and the GenAI decoder session) now use `Environment.ProcessorCount`, which honours the quota and the affinity mask, whenever the caller sets no count and the process is constrained. With the fix the same session ran **52 threads and 50 arenas** (e5) and the multimodal embedder at most 82 (two sessions).
2. **glibc allocator retention, a deployment setting.** What remains is freed memory that glibc keeps resident, chiefly in the main heap (`[heap]`, about 1.1 GiB for EmbeddingGemma 2). `malloc_trim` returns it, the trimmed series is flat, and with `MALLOC_MMAP_THRESHOLD_=65536` every profile's footprint is about half as large and flat. As for GLiNER2 on CPU (`../gliner2-soak-growth/`), this is documented as a deployment setting, and the soak growth rule is unchanged.

## Controlled runs (fixed build `a2b5e8f`)

12 minutes each (`--duration 12m`), default threads (now the container-aware count), CUDA libraries from the pinned wheels, TF32 off, RunPod EU-CZ-1 secure cloud, one model per pod: e5 on an RTX 4090 (`e5/`), image on an RTX 3090 (`embeddinggemma2-image/`), video on an RTX 4090 (`embeddinggemma2-video/`). Each run directory holds the soak report (`.json`, `.md`), `smaps.jsonl` (every 20 s: `/proc/<pid>/status` and resident memory per mapping class, by `scripts/smaps-sum.py`) and `gpu.txt` (device `memory.used`, MiB). Driver: `scripts/diag.sh`. Twelve minutes sits partly on the warmup ramp, so runs are compared with each other and by shape.

Median rss_anon in MiB (fitted slope in MiB/h, per-series verdict). In (d) the model is loaded once, so both series are loaded samples.

| Profile | Run | Cycles | Requests | Wrong | Loaded | Unloaded | Trimmed | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| e5 | (a) `a-baseline` | 342 | 61,560 | 0 | 616 (114, plateau) | 518 (107, plateau) | | PASS |
| e5 | (b) `b-mmap-threshold-64k` | 299 | 53,820 | 0 | 487 (40, plateau) | 443 (14, plateau) | | PASS |
| e5 | (d) `d-requests-only` (`--keep-loaded`) | 1,281 | 230,580 | 0 | 466 (33, plateau) | | | PASS |
| e5 | (f) `f-malloc-trim` | 326 | 58,680 | 0 | 585 (46, plateau) | 488 (10, plateau) | 480 (32, plateau) | PASS |
| image | (a) | 179 | 716 | 0 | 1,754 (385, plateau) | 1,511 (965, plateau) | | PASS |
| image | (b) | 125 | 500 | 0 | 886 (-268, plateau) | 690 (-235, plateau) | | PASS |
| image | (d) | 1,025 | 4,100 | 0 | 1,094 (29, plateau) | | | PASS |
| image | (f) | 161 | 644 | 0 | 1,646 (-26, plateau) | 1,502 (141, plateau) | 773 (5, plateau) | PASS |
| video | (a) | 214 | 642 | 0 | 1,648 (316, **growth**) | 1,566 (246, plateau) | | **FAIL** |
| video | (b) | 122 | 366 | 0 | 857 (55, plateau) | 704 (66, plateau) | | PASS |
| video | (d) | 2,085 | 6,255 | 0 | 1,065 (272, plateau) | | | PASS |
| video | (f) | 183 | 549 | 0 | 1,602 (84, plateau) | 1,501 (30, plateau) | 769 (77, plateau) | PASS |

Unloaded medians per 2-minute bin (`scripts/bins.py`): image (a) 1,409, 1,422, 1,495, 1,500, 1,527, 1,558 against (b) 687, 690, 693, 693, 663, 664; video (a) 1,462, 1,554, 1,546, 1,566, 1,576, 1,576 against (b) 695, 698, 700, 702, 705, 707; e5 (a) 490 to 524 against (b) 442 to 445.

## Diagnosis

- **Per-request buffers in our code.** Ruled out. Requests alone (d) plateau for all three profiles, and the managed heap is flat. Code review: every input and output `OrtValue` in `E5Encoder`, `EmbeddingGemma2Encoder` and `EmbeddingGemma2MultimodalEmbedder` is in a `using` declaration and outputs are bound to pooled managed arrays returned in `finally`; there is no IOBinding; `OrtMemoryInfo.DefaultInstance` is shared.
- **Session native memory not freed on dispose (session, options, CUDA provider options, arenas).** Ruled out. `OnnxModelSession` releases the session, the session options and the CUDA provider options at the last reference, and the environment is ONNX Runtime's default (no shared allocators, no global thread pool). With the mmap threshold fixed (b), the unloaded footprint holds; the remaining slow drift (14 to 66 MiB/h over 12 minutes, plateau) is within the band.
- **CUDA device arena and pinned host memory (`arena_extend_strategy`, `gpu_mem_limit`).** Ruled out for host growth. These govern the per-session device arena and the pinned-host arena; both belong to the session. Device `memory.used` falls back to 292 to 446 MiB at every unload in every run (`gpu.txt`), and the `/dev/nvidia*` mappings hold at 62 to 63 MiB resident throughout. The settings are left at ONNX Runtime's defaults; no session option change is needed.
- **cuDNN and cuBLAS handles and workspaces.** Ruled out. The anonymous pages inside `libcudnn*`, `libcublasLt` and `libcuda` are constant across each run (`smaps.jsonl`); workspaces are device allocations, covered above.
- **glibc retention.** Confirmed. In the smaps samples the main heap (`[heap]`) holds about 1.1 GiB in image and video (a) and 145 MiB in video (b); `malloc_trim` (f) returns about 730 MiB per unload for EmbeddingGemma 2, and the trimmed series is flat (5 and 77 MiB/h, plateau). The per-thread arenas grew in every default run (e5 57 to 72 MiB, image 26 to 50 MiB, video 26 to 49 MiB), which is why the arena count matters.
- **Thread count.** Confirmed as a defect and fixed. It multiplies the arenas (147 before, 50 to 82 after). The baseline runs here use the fix; no same-pod run of the old build completed, so the change in the one-hour slope that the fix alone gives is measured by the binding reruns, not here.

## Decision

- **Fix (library):** the container-aware default intra-op thread count in `Starlings.Runtime.Onnx` (`OnnxSessionOptions.IntraOpNumThreads`, reported in `OnnxSessionDiagnostics`) and in `Starlings.Generation.OnnxGenAI` (decoder `intra_op_num_threads` overlay), with unit tests. Unconstrained hosts keep the ONNX Runtime default the CPU profiles were qualified with. The inter-op count is unchanged: execution is sequential, so ONNX Runtime creates no inter-op pool. CPU fidelity with the constrained branch active (`DOTNET_PROCESSOR_COUNT=4` on the 12-thread workstation, so 4 intra-op threads): `Starlings.Models.E5.Tests` 50 of 50 passed, CPU parity included; `Starlings.Models.EmbeddingGemma2.Tests` 137 passed, 0 failed, 19 skipped.
- **Deployment setting:** `MALLOC_MMAP_THRESHOLD_=65536` is the Linux requirement of `embeddinggemma-2/video-fp32-cuda` (its default run fails the growth rule even with the fix) and is recommended for `image-fp32-cuda` and e5 CUDA, whose default 12-minute runs passed but were still on the ramp (image unloaded slope 965 MiB/h). Documented in the `Starlings.Models.EmbeddingGemma2` README. A library-level `malloc_trim` mitigation is a separate change.

## Binding one-hour reruns (coordinator; not run here)

On a `starling-q-*` pod with the volume's environment (`runpod-ops/worker.sh` sets `STARLING_*`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0`, `NVIDIA_TF32_OVERRIDE=0` and `LD_LIBRARY_PATH` from the eg2-cuda venv's `nvidia/*/lib`), with the soak tool built at the merge commit of this change:

```sh
dotnet build ai-dotnet/tools/Starlings.Soak.Tool -c Release -p:RestoreLockedMode=true -p:UseCuda=true -o /workspace/build/soak-cuda-<sha8>
MALLOC_MMAP_THRESHOLD_=65536 dotnet /workspace/build/soak-cuda-<sha8>/Starlings.Soak.Tool.dll e5 --cuda --duration 1h --out <dir> --note "G4 one-hour qualification soak, cuda, MALLOC_MMAP_THRESHOLD_=65536"
MALLOC_MMAP_THRESHOLD_=65536 dotnet /workspace/build/soak-cuda-<sha8>/Starlings.Soak.Tool.dll embeddinggemma2-image --cuda --duration 1h --out <dir> --note "G4 one-hour qualification soak, cuda, MALLOC_MMAP_THRESHOLD_=65536"
MALLOC_MMAP_THRESHOLD_=65536 dotnet /workspace/build/soak-cuda-<sha8>/Starlings.Soak.Tool.dll embeddinggemma2-video --cuda --duration 1h --out <dir> --note "G4 one-hour qualification soak, cuda, MALLOC_MMAP_THRESHOLD_=65536"
```

## Reproduce the diagnostic runs

`scripts/build.sh <sha>` builds the CUDA soak tool on a pod; `scripts/diag.sh <out> <model> 12m a b d f` runs the legs with the smaps sampler (`a` baseline, `b` `MALLOC_MMAP_THRESHOLD_=65536`, `d` `--keep-loaded`, `f` `--malloc-trim`; also `c` `MALLOC_ARENA_MAX=2` and `e` `--requests-per-cycle 1 --deadline-every 0`, not run here). Summaries: `python3 -I scripts/table.py <run>/*.json`, `scripts/bins.py` (binned medians and thirds) and `scripts/smaps.py <run>/smaps.jsonl` (per-class trend).
