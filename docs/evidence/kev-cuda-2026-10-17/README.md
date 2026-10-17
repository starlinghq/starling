# Kev on the CUDA execution provider: development evidence, 2026-10-17

Development evidence for the profiles `kev-0.8b/onnx-fp32-cuda` and `kev-4b/onnx-fp32-cuda`: the qualified CPU-built
fp32 bundles (`KevModelAssets` locks) on ONNX Runtime 1.30.0's CUDA execution provider with `use_tf32=0`. Not a gate
result: the references are still proposed, no protected CUDA tolerances or release sets exist for Kev, and the run is on
a workstation, not the qualification pod. Bounds below are the CPU profile's (`model-pipeline/protected/kev-0.8b/onnx-fp32/tolerances.yaml`,
which states it applies to the CUDA EP with `use_tf32=0`; the same values are proposed for 4B).

Machine: Intel i7-5820K, NVIDIA GeForce RTX 3090 (24 GB, PCI order device 0, driver 610.57.04), Linux 7.2.5, system
CUDA 13.3 runtime (`/opt/cuda`), cuDNN 9 from the `nvidia-cudnn` wheel on `LD_LIBRARY_PATH`. .NET 10.0.12 with
`Microsoft.ML.OnnxRuntime.Gpu.Linux` 1.30.0; Python `onnxruntime-gpu` 1.30.0. Both stages read the provider options
back from the session: `CUDAExecutionProvider` first, `use_tf32` `0`, `device_id` `0`.

## C# tests (`ai-dotnet/tests/Starlings.Models.Kev.Tests.Cuda`)

The locked bundles (`$STARLING_KEV_08B_BUNDLE`, `$STARLING_KEV_4B_BUNDLE`; graph digest enforced by the session), 12 of
12 passing in 65 s, against Kev's PyTorch fp32 probabilities in `tests/Starlings.Models.Kev.Tests/Data/`:

| Check (bound) | Kev-0.8B | Kev-4B |
| --- | --- | --- |
| 9 request goldens vs PyTorch fp32 (2e-4), argmax | 1.33e-5, no flip | 4.17e-5, no flip |
| Cached vs uncached rows (1e-5) | 7.75e-7 | 7.15e-7 |
| 4 callers x 2 rounds of the goldens, serialized (default on CUDA) | 72 of 72 bit-identical to sequential | 72 of 72 |
| Cancelled calls: device work left queued (`CudaQueueProbe`, 2 ms) | 0 of 10, drain at most 0.014 ms | 0 of 10, at most 0.019 ms |
| One 8-question request (`many-questions`), uncancelled | 105 ms | 299 ms |

## Onboarding corpus: Python and .NET on CUDA, the pipeline's decision evaluator

The `kev-onnx-cpu-v1` route's own stage outputs for the reference (PyTorch fp32 on CPU, 30 records of
`model-pipeline/reference/corpus/kev-decisions-v1.jsonl`, 58 questions, 44 labelled, states up to 1,531 tokens) and the
build, with the `onnx` and `dotnet` stages rerun on CUDA:

```sh
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python -I model-pipeline/reference/kev/run_kev.py onnx --provider cuda \
  --profile model-pipeline/candidates/<model>/profile.json --reference <reference stage>/reference.json \
  --bundle <build stage>/bundle --build <build stage>/build.json --threads 4 --out <dir>/onnx
dotnet run --project ai-dotnet/tools/Starlings.Kev.Fidelity.Tool -c Release -p:UseCuda=true -- --device cuda \
  --profile model-pipeline/candidates/<model>/profile.json --snapshot <build stage>/bundle \
  --reference <reference stage>/reference.json --threads 4 --out <dir>/dotnet
# then model_pipeline.evaluate_decision.evaluate(...) with the candidate's predeclared policy
```

The evaluator's report names the candidate id (`onnx-fp32-cpu`); the provider is in each stage's `environment`. The
build stage's bundle is a rebuild of the locked one: graph and head identical, `model.onnx.data` within the fp32 LoRA
merge's non-reproducibility (package README, "The bundle"). Reports: [kev-0.8b](kev-0.8b/report.md), [kev-4b](kev-4b/report.md).

| Check (bound) | Kev-0.8B CUDA | Kev-0.8B CPU (2026-10-16) | Kev-4B CUDA | Kev-4B CPU (2026-10-16) |
| --- | --- | --- | --- | --- |
| Every backend path vs PyTorch fp32 (2e-4) | 2.85e-5 | 2.96e-5 | 6.57e-5 | 6.49e-5 |
| Cached vs uncached, .NET (1e-5) | 1.94e-6 | 2.7e-7 | 2.03e-6 | 2.5e-7 |
| Padded batch vs single rows, Python (1e-5) | 2.15e-6 | n/a | 2.50e-6 | n/a |
| .NET vs Python ONNX Runtime, same provider (1e-6) | 1.56e-7 | 1.24e-7 | 1.49e-7 | 1.81e-7 |
| Argmax flips, accuracy on 44, token ids | 0, identical, identical | 0, identical, identical | 0, identical, identical | 0, identical, identical |

CPU: `tests/Starlings.Models.Kev.Tests` with both bundles, 110 of 111 passing in one run; the failure was the old contract
test that expected CUDA to be refused, updated for the CUDA profiles (its class then 14 of 14). `ConcurrentCallsEqualSequential` (4 callers x 2 rounds, not serialized) 72 of 72 bit-identical to
sequential for each size (59 s for 0.8B, 405 s for 4B on 12 threads).

## Not covered

- Upstream's development partitions (decision-v7, transfer-v4) on CUDA, and any release set (none is accepted for Kev).
- The per-release GPU regression (`tools/ci/runpod-cuda-pod.sh` has no Kev steps yet; the bundles are hosted since
  2026-10-17, so it can acquire them).
- A one-hour G4 soak and the G5 matched benchmark (no Kev budget exists).
- The builder's own CUDA graph (`-e cuda`), fp16, bf16 and TF32: not these profiles (admission, "GPU results").
