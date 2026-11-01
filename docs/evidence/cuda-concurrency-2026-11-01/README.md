# Concurrent CUDA EmbedBatch calls on one e5 or EmbeddingGemma 2 session, 2026-11-01

Issue #167. GLiNER2 serializes its CUDA runs because unserialized concurrent runs on one session returned wrong entities in 6 of 8 runs ([capability closure](../capability-closure-2026-10-14/README.md)). e5 and EmbeddingGemma 2 share the session code (`Starlings.Runtime.Onnx.OnnxModelSession`) and still claim concurrent calls as supported on CUDA:

- `multilingual-e5-small/onnx-fp32-cuda`: `concurrent_calls` is `Supported` in `E5Capabilities.Profiles` and in the package README's "Stable contract: CUDA". The manifest has no `concurrent_calls` key.
- `embeddinggemma-2/onnx-fp32-cuda`: `concurrent_calls: supported` in the manifest.

The existing CUDA concurrency tests of both models used single-input `Embed` calls on four threads. This run adds a stronger test to each local CUDA test project: concurrent multi-row `EmbedBatch` calls on eight threads.

## Result

Both passed. In 8 separate processes per model, every vector was bit-for-bit equal to the sequential result.

| Model | Test | Per process | Processes | Vectors compared | Max abs |
| --- | --- | --- | --- | --- | --- |
| multilingual-e5-small | `E5.Tests.Cuda` `CudaParityTests.ConcurrentEmbedBatchCallsGiveTheSequentialResultsBitForBit` | 8 threads x 8 inputs x 5 rounds | 8 of 8 passed | 2,560 | 0 |
| EmbeddingGemma 2 (text) | `EmbeddingGemma2.Tests.Cuda` `CudaParityTests.ConcurrentEmbedBatchCallsGiveTheSequentialResultsBitForBit` | 8 threads x 4 rows x 5 rounds | 8 of 8 passed | 1,280 | 0 |

Logs: [e5](logs/e5-cuda-concurrency-8x.log), [EmbeddingGemma 2](logs/eg2-cuda-concurrency-8x.log). Each process ran two methods: `SessionUsesCudaWithTf32Disabled`, which records the ONNX Runtime version and provider options, and the new concurrency test.

**No status change.** `concurrent_calls` stays supported on both CUDA profiles, and neither encoder serializes CUDA runs. If the test fails in a later run, follow the issue's plan: serialize as GLiNER2 does, and move the status to unsupported in a maintainer change with a release note.

## Method

- One encoder (one ONNX Runtime session) per process, CUDA device 0 of the visible set, `use_tf32=0`.
- 8 batches, each a different rotation of the fixture inputs, so their lengths and padding differ. e5 uses all 46 protected cases (8 inputs per batch). EmbeddingGemma 2 uses the 132 protected text rows under 512 tokens (4 rows per batch).
- The baseline is a sequential `EmbedBatch` of each batch on the same session, so the concurrent and sequential calls see the same shapes and padding.
- Each round starts 8 threads, released together by a `Barrier`. Each thread runs `EmbedBatch` on its own batch. Every output vector must equal the baseline bit for bit (`SequenceEqual` on the floats). 5 rounds per process.
- On CUDA, `OnnxModelSession` runs on the calling thread unless dedicated run threads are requested, and it does not serialize runs. The 8 calls therefore overlap on the device.

## Environment

- Intel i7-5820K (12 threads). NVIDIA RTX 4060 Ti 16 GB (Ada) as GPU 1 (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`). Driver 615.78.08.
- CUDA 13 runtime from `/opt/cuda`. cuDNN 9.24 from the `nvidia-cudnn-cu13` wheel.
- ONNX Runtime 1.30.0 (`Microsoft.ML.OnnxRuntime.Gpu.Linux`). .NET runtime 10.0.14, Release build.
- Models from the Starlings model cache, with the pinned SHA-256 enforced by the fixtures.

This is not the RTX 3090 that the issue proposed and that the CUDA contract covers. During the run, GPU 0 (the 3090) had about 23 of 24.6 GB in use by another workload. The machine's CPU was saturated by other jobs (load 12 on 12 threads) during the EmbeddingGemma 2 runs, which adds scheduling variation to the thread interleaving.

This is one machine and one GPU, so it is not qualification. Before relying on these statuses for the 3090 contract, rerun the two tests there. Commands, from the repository root:

```sh
export STARLING_CUDA_TESTS=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu>
dotnet build ai-dotnet/tests/Starlings.Models.E5.Tests.Cuda -c Release
ai-dotnet/tests/Starlings.Models.E5.Tests.Cuda/bin/Release/net10.0/Starlings.Models.E5.Tests.Cuda \
  -method Starlings.Models.E5.Tests.Cuda.CudaParityTests.ConcurrentEmbedBatchCallsGiveTheSequentialResultsBitForBit -showLiveOutput
dotnet build ai-dotnet/tests/Starlings.Models.EmbeddingGemma2.Tests.Cuda -c Release
ai-dotnet/tests/Starlings.Models.EmbeddingGemma2.Tests.Cuda/bin/Release/net10.0/Starlings.Models.EmbeddingGemma2.Tests.Cuda \
  -method Starlings.Models.EmbeddingGemma2.Tests.Cuda.CudaParityTests.ConcurrentEmbedBatchCallsGiveTheSequentialResultsBitForBit -showLiveOutput
```

The full e5 `CudaParityTests` class (7 tests, including the new one) also passed on the same GPU. Paths in the logs are shortened: `<repo>` is the checkout and `<model-cache>` the Starlings model cache.
