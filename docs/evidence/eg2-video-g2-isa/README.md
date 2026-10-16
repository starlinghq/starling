# EmbeddingGemma 2 video G2 on AVX-512 hosts and the RTX 4090: diagnosis

`embeddinggemma-2/video-fp32-cpu` fails release-set G2 on GitHub-hosted runners with AVX-512 and passes on the AVX2
runner (EPYC 7763) and the workstation (i7-5820K). `video-fp32-cuda` failed G2 on an RTX 4090 pod (run 38857860186) and
passed on an RTX 3090. Both failures are the same check on the same row: `dotnet-ids.token_embeddings` on
`commons-video-04`. This directory traces the CPU failure down to an operator. The hypothesis was that CPU-side video
preprocessing differs on AVX-512 hosts. That hypothesis is refuted. **No Starling code change makes the check
ISA-independent, and the PyTorch reference cannot meet the check itself when it runs on another CPU.** No tolerance,
protected file, release set, budget or maturity is changed here.

## Result

1. **Preprocessing is not the cause.** Every failing run has stage 1 exact: decoded frames, resized frames,
   `pixel_values`, positions and ids all match the reference digests (`tokenizer.pixel_values_sha256` and the other
   stage 1 rows in `run-38977402003/*/g1-g2-report.md`, in `../eg2-hosted-soak/release-gates-video/` and in
   `../gpu-regression-2026-10-16/run-38857860186/gates/embeddinggemma-2--video-fp32-cuda/`). The RTX 4090 pod's host
   was an **AMD EPYC 7763 (AVX2, no AVX-512)**, as `cpu_model` in that run's video soak report shows. So the CUDA
   failure did not happen on an AVX-512 host.
2. **The text graph and .NET are not the cause.** With the reference soft tokens as input, the released multimodal
   text graph matches the reference token embeddings to 2.63e-5 (AVX2) and 2.80e-5 (AVX-512) on `commons-video-04`.
   Python ORT run on the C# soft tokens of the same machine reproduces the C# token embeddings bit for bit
   (`vs_stage_te_max_abs: 0.0` on all six runners). When the AVX-512 runner's soft tokens go through the text graph
   on the AVX2 workstation, the result is the AVX-512 failure (1.827e-3,
   `local/textgraph-avx512-soft-on-avx2.json`). The difference is in the soft tokens, which come from the vision graph.
3. **The vision graph's soft tokens depend on the ISA at ULP level. The first ISA-dependent node is the layer-0
   attention `Softmax`.** Pixels are identical. The soft tokens on AVX-512 differ from those on AVX2 by 2.6e-5
   row-relative. Both are about 2.4e-4 from the PyTorch reference, which is the ORT-against-PyTorch gap and inside the
   1e-3 soft-token bound. The per-node digests of the unoptimized vision graph (frame 0) are bit-identical between
   AVX2 and AVX-512 for the first 124 node outputs: patch embedding, RMS norms, rotary `Cos`/`Sin` and the Q/K/V
   `MatMul`s. They first differ at `node_Softmax_229` (`[1, 12, 1260, 1260]`). Downstream of it the first differences
   are 3e-7 relative (`local/nodes.json`, `run-38980345606/*/nodes.json`, `analysis/cmp_nodes.py`). ONNX Runtime 1.30.0
   MLAS selects AVX-512F kernels for `ComputeExpF32Kernel`, `ComputeSumExpF32Kernel` and `ReduceMaximumF32Kernel`
   (softmax), and for GEMM, erf and GELU, whenever CPUID and XCR0 report AVX-512F
   (`onnxruntime/core/mlas/lib/platform.cpp` at v1.30.0, lines 605-627). No session option or environment variable
   selects the ISA on x86-64. The only MLAS override is `ORT_MLAS_RISCV_FORCE_SCALAR`, for RISC-V. Graph optimizations
   are not involved: `ORT_DISABLE_ALL` against `ORT_ENABLE_ALL` changes the soft tokens by 2e-7 on both ISAs.
4. **The row is ill-conditioned: one element in channel 115 drives the check.** The row-relative metric divides by
   `max|ref row|` = 4,352.8, which is Gemma's massive-activation channel 115. On AVX-512, a single element exceeds the
   bound: token 349 (a video soft token), channel 115, reference value 512.2, error 7.95 (1.83e-3 of the row maximum).
   The next-worst token is at 3.3e-4 and the median token is at 7e-7. Adding Gaussian noise to the reference soft
   tokens shows the amplification (`analysis/noise.sh`, the text graph on the workstation):

   | Noise sigma (x max\|soft\|) | `commons-video-04` token embeddings | `commons-video-05` | `commons-video-00` |
   | --- | --- | --- | --- |
   | 1e-5, seeds 1 / 2 | 3.8e-4 / 7.8e-4 | 2.9e-5 / 3.4e-5 | 1.4e-4 / 4.7e-5 |
   | 1e-4, seeds 1 / 2 | 3.4e-3 / 9.2e-3 | 2.9e-4 / 3.5e-4 | 1.4e-3 / 4.7e-4 |

5. **The PyTorch reference cannot reproduce itself within this bound on other CPUs.** `analysis/ref_rerun.py` reruns
   the reference runner's own `load`/`forward`/`Capture` (PyTorch 2.14.1 CPU fp32, the pinned reference environment)
   on `commons-video-04` and compares the result with the accepted reference tensors:

   | Machine | Torch threads | ISA used | Soft tokens | Token embeddings | Embedding max abs |
   | --- | --- | --- | --- | --- | --- |
   | i7-5820K (the reference machine), default threads | 6 | AVX2 | bit-identical | bit-identical | 0 |
   | i7-5820K, `torch.set_num_threads(2)` | 2 | AVX2 | bit-identical | 1.19e-5 | 4.5e-8 |
   | i7-5820K, `torch.set_num_threads(6)` (same count as the default) | 6 | AVX2 | 3.4e-5 | **7.3e-4** | 2.4e-6 |
   | Xeon Platinum 8573C (hosted), native | 2 | AVX-512 | 3.3e-4 | 4.9e-4 | 1.9e-6 |
   | Xeon Platinum 8573C, capped at AVX2 (`ATEN_CPU_CAPABILITY`, `MKL_ENABLE_INSTRUCTIONS`, `ONEDNN_MAX_CPU_ISA`) | 2 | AVX2 | bit-identical | 1.19e-5 | 6e-8 |
   | EPYC 9V45 / 9V74 (hosted), native | 2 | AVX-512 | 2.7e-5 | **1.95e-3** | 5.7e-6 |
   | EPYC 9V45 / 9V74, capped at AVX2 | 2 | AVX2 | 3.3e-5 | **2.09e-3** | 6.1e-6 |
   | EPYC 7763 (hosted) | 2 | AVX2 | 3.3e-5 | **2.09e-3** | 6.1e-6 |

   On Intel, the reference reproduces only under AVX2; native AVX-512 on the 8573C moves it by 4.9e-4. On every AMD
   host, AVX2 included, the reference misses the 1e-3 token-embedding bound against its own accepted output (2.09e-3 on
   the EPYC 7763, the runner where the C# path passes). The thread count does not explain this: two threads on the
   reference machine stay within 1.2e-5. An explicit `torch.set_num_threads(6)` call on the reference machine, at the
   default thread count, already moves the row by 7.3e-4. The embedding bound (1e-5) holds in every row of this table.

## G2 on every hosted runner (C# path, release set, unchanged code at main)

Runs [38977402003](https://github.com/starlinghq/starling/actions/runs/38977402003) (six runners) and
[38980345606](https://github.com/starlinghq/starling/actions/runs/38980345606) (six runners). Both ran on the
diagnostic branch `video-avx512-diag`, which is not merged. It holds a push-triggered workflow and `diag/run.sh` /
`diag/run2.sh`, and adds no code to the model path.

| CPU | AVX-512 | Soft tokens (bound 1e-3) | Token embeddings (bound 1e-3) | Embedding max abs (bound 1e-5) | Text graph on reference soft tokens, `-04` | G2 |
| --- | --- | --- | --- | --- | --- | --- |
| AMD EPYC 7763 | no | 2.37e-4 | 3.81e-4 (`-04`: 2.74e-4) | 1.29e-6 | 2.63e-5 | passed |
| Intel Xeon Platinum 8370C | yes | 2.63e-4 | **1.83e-3** (`-04`) | 5.72e-6 | 2.80e-5 | failed |
| Intel Xeon Platinum 8573C | yes | 2.63e-4 | **1.83e-3** (`-04`) | 5.72e-6 | 2.80e-5 | failed |
| AMD EPYC 9V45 (three runners) | yes | 2.63e-4 | **1.83e-3** (`-04`) | 5.72e-6 | 2.80e-5 | failed |
| i7-5820K (workstation, AVX2; C# soft tokens and token embeddings identical to the 7763's) | no | 2.37e-4 | `-04`: 2.74e-4 | | 2.63e-5 | (same tensors as the 7763) |

All four AVX-512 CPU models, two Intel and two AMD generations, give identical digits. That pattern points to kernel
dispatch, not to the silicon. The same holds for the earlier runs 38874143326 (EPYC 9V45) and 38882433616 (EPYC 9V74).

## What this means for the contract (maintainer decision; nothing changed here)

The `commons-video-04` token-embedding check sits at the fp32 recomputation noise floor of the model. The C# path
passes it on AVX2 x86-64 hosts deterministically (2.74e-4 on that row, bit-identical on the EPYC 7763 and the
i7-5820K). Any change of fp32 kernel order breaks it: the AVX-512 softmax, a different GPU architecture (RTX 4090:
3.34e-3; RTX 3090: 7.97e-4), or even the PyTorch reference rerun on another CPU. Options, for the maintainer:

- (a) State the CPU contract as AVX2 x86-64. This matches every passing run, but the CI hosts cannot be selected by
  ISA, so a hosted G2 would need a skip or a retry on AVX-512 runners, or a self-hosted or pinned machine.
- (b) Reconsider the row or the metric for this check. `commons-video-04`'s row maximum is a single massive-activation
  channel. The release set's `SOURCE.md` already lists row choices as reviewer decisions. This is a protected change.
- (c) Rely on the bounds that hold everywhere: the embedding max abs (worst 5.72e-6 on CPU against 1e-5) and the soft
  tokens (worst 2.63e-4 against 1e-3). The RTX 4090 embedding (1.009e-5) is also just over its bound, so for CUDA
  (c) alone does not cover that GPU.

A Starling-side workaround is possible in principle: rewrite the vision graph's softmax into operators that MLAS does
not dispatch by ISA. That changes the locked, SHA-256-checked graph. It would match the AVX2 result, not the
reference, which the table above shows is itself CPU-dependent at this level. It was not attempted.

## The RTX 4090 failure

Same check, same row (`commons-video-04`): token embeddings 3.34e-3 with the soft tokens in bound (2.08e-4), stage 1
exact, and a host without AVX-512 (EPYC 7763). This fits the mechanism above: CUDA kernels on Ada order the vision
graph's fp32 reductions differently from Ampere, and channel 115 amplifies the difference. The run kept no tensor
sidecars, so the token-level localization was not repeated for CUDA.

## Files

| Path | Contents |
| --- | --- |
| `analysis/textgraph_diag.py` | Multimodal text graph (Python ORT) on the reference or a stage's soft tokens, against the reference token embeddings; `--noise` for the sensitivity sweep |
| `analysis/vision_diag.py` | Vision graph on the sampled frames of one video (torchcodec plus the reference preprocessing), session variants, per-node digests of the unoptimized graph |
| `analysis/ref_rerun.py` | PyTorch reference rerun (the runner's own functions) against the accepted tensors |
| `analysis/ref_layouts.py` | The reference's restated vision path (padded, unpadded, one frame at a time) through the text graph |
| `analysis/cmp_soft.py`, `analysis/cmp_nodes.py`, `analysis/noise.sh` | Comparisons |
| `run-38977402003/diag-<n>-full/` | Per runner: `cpu.txt`, the G1/G2 report, text graph results on the reference soft tokens (4 threads, 1 thread, optimizations off) and on the runner's C# soft tokens |
| `run-38980345606/diag-<n>-r2/` | Per runner: `cpu.txt`, PyTorch reference reruns (native, capped at AVX2), vision variants (`vision.json`), per-node digests (`nodes.json`) |
| `local/` | The same measurements on the i7-5820K |

Tensors (`.f32` sidecars, `.npy`) are not committed. They regenerate with the commands below.

## Reproduce

```sh
# Text graph on the reference soft tokens (needs the hosted references, tools/fetch-release-references.py)
python -I analysis/textgraph_diag.py <snapshot>/onnx/embeddinggemma2-text-mm-fp32.onnx \
  model-pipeline/protected/embeddinggemma-2/release/video-fp32-release/video-embeddings.jsonl <refs> <out> --rows commons-video-04
# Vision graph per-node digests (pinned FFmpeg 8 on LD_LIBRARY_PATH); compare two machines with cmp_nodes.py
python -I analysis/vision_diag.py model-pipeline/reference/embeddinggemma2 <media>/commons-video-04.webm \
  <snapshot>/onnx/embeddinggemma2-vision-fp32.onnx <out> --frames 32 --variants all,disable --dump-nodes 150
# PyTorch reference rerun (google/embeddinggemma-2 at 914f7f89142e33e77833254d9c9b90c3cef7303b); last argument 0 = default threads
python -I analysis/ref_rerun.py model-pipeline/reference/embeddinggemma2 <hf snapshot> <media>/commons-video-04.webm <refs> \
  model-pipeline/protected/embeddinggemma-2/release/video-fp32-release/video-embeddings.jsonl <out.json> 0
```
