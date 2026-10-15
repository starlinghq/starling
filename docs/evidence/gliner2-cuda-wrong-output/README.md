# GLiNER2 CUDA wrong output: cancelled CUDA runs left device work queued

## The failure

The one-hour G4 soak of `gliner2-base-v1/onnx-fp32-cuda` on RunPod returned 1 wrong output in 80,100 checked requests. The run used an RTX 3090 with driver 580.126.20, commit `26249631`, `use_tf32=0` and `MALLOC_MMAP_THRESHOLD_=65536`, with 2 callers and a new extractor every cycle. The wrong output was cycle 388, request 14, input 2 ("Marie Curie worked at the University of Paris and won two Nobel Prizes."). It returned an **empty** entity list instead of `person:0-11;organization:26-45;location:40-45`. Request 9 of that cycle was a deadline request: a 16-text batch cancelled after 5 to 150 ms. That run had 4,680 cancelled deadline requests. Memory plateaued, and CPU soaks of the same model gave 0 wrong in more than 73,000 requests.

## Root cause

`OnnxModelSession` cancelled a running call by setting ONNX Runtime's `RunOptions.Terminate`, which the executor checks between graph nodes. On the CUDA execution provider, ORT 1.30.0 returns from such a terminated run while kernels and copies it already queued are still pending on the device. The run's buffers have already gone back to ORT's allocators at that point. The next run can then compute on memory that the abandoned work is still writing.

GLiNER2 serializes CUDA calls (PR #19), so the request waiting on its run gate starts exactly when this happens. Its logits come out as garbage, for example count logits near -1e26, all-NaN span logits, or span logits near -1e8. The count gate then closes and the result is empty.

The source fits this. The CUDA provider's `OnRunEnd` synchronizes only the provider's own stream (`cudaStreamSynchronize(stream_)`), not the run's pooled compute stream. The output copy that synchronizes a successful run never happens on the terminated path. This explains the measurements below; it was not traced instruction by instruction.

Hypotheses ruled out:

- **A leaked terminate flag.** Each call already gets its own `RunOptions`.
- **A race in `SerializeNativeCalls`.** The gate covers the run and the output read, and `OrtValue`s are per call.
- **Nondeterminism without cancellation.** There were 0 wrong outputs without cancellation, below.

## Measurements

These ran on the workstation: RTX 3090 (GPU 0 only), driver 610.57.04, CUDA 13 and cuDNN 9 wheels, ORT 1.30.0, .NET 10. The reproducer is `ai-dotnet/tools/Starlings.Gliner2.CudaStress` (local only; `run.sh`).

"Before" means commit `1275049`, with terminated CUDA runs. "Before, drained" is the same code with a diagnostic `cudaDeviceSynchronize` or a 300 ms sleep inside the run gate after a cancelled run. That diagnostic was not committed.

| Run | Before | Before, drained | After |
| --- | --- | --- | --- |
| Soak pattern, 2 callers, deadline every 2nd request, seed 1 | **7 wrong / 10,000** (8,340 cancelled) | | |
| Same, seed 7 | **8 wrong / 4,000** (3,440 cancelled) | 0 / 4,000 (sync), 0 / 4,000 (sleep) | 0 / 4,000 (3,719 cancelled) |
| Same, no deadline requests | 0 / 20,000 | | |
| Same, deadline requests that never fire (16-text batches complete) | 0 / 4,000 | | |
| Cancelling thread plus 1 waiting thread, 2 other sessions loading the GPU | **2 wrong / 1,724** (2,994 cancelled) | | 0 / 1,634 (3,000 cancelled) |

Every wrong output had an empty result, and each one was served immediately after a terminated run ([log with logits and request history](logs/stress-soak-pattern-before-seed7.txt)). The hit rate depends on GPU load. A later "before" rerun of the seed-7 row on an idle GPU gave 0 wrong in 4,000, and the soak-pattern rows above ran while another process also used GPU 0.

### Load-independent check

This check measures the mechanism directly rather than the symptom, so it does not depend on load. After each cancelled call returns, it times `cudaDeviceSynchronize`, which measures the work the call left queued. The probe is `tests/Shared/Cuda/CudaQueueProbe.cs`, used by the new `ACancelledCallLeavesNoDeviceWorkQueued` test in each CUDA test project. Queued work counts if draining takes more than 2 ms; a drained device answers in about 10 µs.

| Model (CUDA) | Before: cancelled calls with work queued | Before: max drain | After |
| --- | --- | --- | --- |
| GLiNER2, 16 texts in batches of 8 | 78 of 300 | 3.5 ms (33 ms in the stress tool) | 0 of 300, max 0.05 ms |
| e5, 32 x 512 tokens | 24 of 25 (most calls finished before their deadline) | 76 ms | 0 of 300, max 0.02 ms |
| EmbeddingGemma 2, 4 x 1,750 tokens | 150 of 150 | 114 ms | 0 of 150, max 0.02 ms |

Completed (uncancelled) calls never left work queued.

## Fix

In `Starlings.Runtime.Onnx` (`OnnxModelSession`), a CUDA run is no longer terminated mid-graph. The run completes. If the token was cancelled meanwhile, the outputs are disposed and the call throws `OperationCanceledException`. No state with released buffers and pending device work can exist, so the fix also holds for sessions that are not serialized: e5 and EmbeddingGemma 2 allow concurrent calls on CUDA. A CUDA run that fails for another reason drains the device (the I/O binding's synchronize calls reach the CUDA provider's `Sync`, a device synchronization) before the exception propagates. CPU cancellation is unchanged.

**Trade-off.** On CUDA, cancellation latency is now bounded by the current run rather than by one node. Every library splits work into chunks (`MaxBatchSize`) and checks the token between them, so this is one chunk:

- **GLiNER2:** about 50 ms for 8 long texts. The 4,096-text CUDA cancellation test still throws 52 ms after a 50 ms deadline.
- **e5:** about 125 ms for 32 x 512 tokens.
- **EmbeddingGemma 2:** up to about 1.2 s at its default 16 rows of 2,048 tokens. A caller who needs faster cancellation can lower `MaxBatchSize`.

The capability statuses are unchanged (`cancellation: experimental`), and the documented semantics are updated: `OnnxModelSession`, the model and adapter remarks, and the READMEs.

## Other models

The pattern was in the shared session code, so every ONNX Runtime library on CUDA was affected:

- **e5** and **EmbeddingGemma 2** (text, image and video graphs) leave the same queued work. The probe above shows it, larger than in GLiNER2. Both are fixed by the same change and covered by the new probe test. Wrong outputs were not separately reproduced for them. They are not serialized on CUDA, so concurrent runs were exposed as well as the next run.
- **Qwen3.5** runs on ONNX Runtime GenAI and does not use `OnnxModelSession`. It cancels only between prefill chunks and decode steps, so it is not affected.

Also fixed: `tests/Starlings.Models.E5.Tests.Cuda` did not compile on `main`. Its `CancelAfter` helper was in the wrong class. The project is local only, so CI never built it.

## Tests

`ACancelledCallLeavesNoDeviceWorkQueued` is in the GLiNER2, e5 and EmbeddingGemma 2 CUDA test projects. Each one fails on `1275049` and passes after the fix. All three CUDA test projects pass after the fix:

- **GLiNER2:** 8 of 8.
- **e5:** 9, plus 1 opt-in skip.
- **EmbeddingGemma 2:** 10 of 10 in `CudaParityTests`.

`dotnet test --solution ai-dotnet/AI.slnx` passes (712 passed, 532 skipped without models or GPUs).
