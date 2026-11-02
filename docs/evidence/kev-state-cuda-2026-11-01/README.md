# KevState state reuse on the CUDA execution provider, 2026-11-01

Issue #140, step 1 of the triage plan. `KevDecisionModel.CreateState(state)` runs the state pass once and returns a `KevState`; `Decide(state, questions)` answers question sets against it. On CPU the capability `state_reuse_across_calls` is experimental from `ApiTests.StateReuseEqualsDecideBitForBit` and `ApiTests.StateIsSharedByConcurrentCallsAndRefusedAfterDispose` (Kev-0.8B). `state_reuse_across_calls.cuda` was untested because those tests had never run on a GPU: the CPU test project references the CPU-only native package.

This run ports the two tests to the local CUDA test project (`ai-dotnet/tests/Starlings.Models.Kev.Tests.Cuda`, `CudaParityTests`, same method names and assertions) and runs them on the profile `kev-0.8b/onnx-fp32-cuda`.

## What is compared

Within one CUDA session: decisions from a `KevState` against `Decide(request)` on **the same CUDA session**, with exact float equality (`Assert.Equal` on `ProbabilityList`). This is the claim the CPU evidence makes, on the CUDA provider. It is not a cross-provider claim. CUDA and CPU outputs differ by up to about 1e-5 (see [kev-cuda-2026-10-17](../kev-cuda-2026-10-17/README.md); bound 2e-4 against PyTorch), and this run does not compare them.

## Result

Passed in 3 of 3 separate processes. No difference in any compared answer.

| Test (Kev-0.8B, CUDA) | Per process | Processes |
| --- | --- | --- |
| `StateReuseEqualsDecideBitForBit`: the 9 request goldens; for each, `CreateState` token count equals `Decide(request).StateTokens`; `Decide(state, questions)` twice in a row equal to `Decide(request)` (state and branch tokens, answer ids, probabilities); then each question alone on the same state | 60 answers bit-identical | 3 of 3 passed |
| `StateIsSharedByConcurrentCallsAndRefusedAfterDispose`: one state, 8 decisions on 4 parallel callers (serialized, the CUDA default), equal to `Decide(request)`; a cancelled call throws and the state still answers bit-identically; double dispose is safe and a disposed state is refused | 8 decisions bit-identical | 3 of 3 passed |
| `SessionUsesCudaWithTf32DisabledAndSerializes`: `CUDAExecutionProvider` first, `use_tf32=0`, locked graph digest, serialized native calls | passed | 3 of 3 passed |

Log: [logs/kev-0.8b-state-cuda-3x.log](logs/kev-0.8b-state-cuda-3x.log). The Kev-4B cases skip (bundle variable unset, see below).

**Status change:** `state_reuse_across_calls.cuda` moves from untested to experimental (`KevCapabilities`, package README, release note).

## Environment

- Intel i7-5820K (12 threads), Linux 7.2.8.
- **NVIDIA GeForce RTX 4060 Ti 16 GB** (Ada), GPU 1 in PCI order (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`; device 0 of the visible set). Driver 615.78.08.
- CUDA 13 runtime from `/opt/cuda`, cuDNN 9.24 from the `nvidia-cudnn-cu13` wheel on `LD_LIBRARY_PATH`.
- ONNX Runtime 1.30.0 (`Microsoft.ML.OnnxRuntime.Gpu.Linux`), .NET runtime 10.0.12, Release build.
- Bundle: the locked Kev-0.8B bundle (`jaredpalmer/kev-0.8b` at `9a45d25eb2ab761841196625383fa1dff0e56c1e`) from the Starlings model cache, size-checked against the lock, graph digest enforced by the session.

```sh
export STARLING_CUDA_TESTS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
export STARLING_KEV_08B_BUNDLE=<kev-0.8b bundle>
dotnet build ai-dotnet/tests/Starlings.Models.Kev.Tests.Cuda -c Release
ai-dotnet/tests/Starlings.Models.Kev.Tests.Cuda/bin/Release/net10.0/Starlings.Models.Kev.Tests.Cuda \
  -method '*SessionUsesCudaWithTf32DisabledAndSerializes' -method '*StateReuseEqualsDecideBitForBit' \
  -method '*StateIsSharedByConcurrentCallsAndRefusedAfterDispose' -showLiveOutput -noColor
```

## Not covered

- **This is not the RTX 3090** that the Kev CUDA profiles' evidence and Corvid's measurements use. A workstation RTX 4060 Ti run supports experimental only. During the run the RTX 3090 was in use by another workload and was not touched.
- **Kev-4B on CUDA:** not run. It needs about 17 GB of device memory and the RTX 4060 Ti has 16 GB. The tests cover both sizes and will run it when `STARLING_KEV_4B_BUNDLE` is set on a larger GPU.
- **Latency:** not measured. On this bundle `GroupQueryAttention` stays on the CPU provider (issue #140), so the state cache still passes through host memory.
- The rest of issue #140 stays open: ask 1 (a GPU-resident graph from the builder's `-e cuda` export) and `state_prefix_reuse_across_requests` (unsupported).
