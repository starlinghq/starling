# G4 resource correctness: soak harness and operational tests (R1)

Workstream R1 of the production readiness plan (plan section 10.2, gate G4). This directory holds **short development soak runs (5 minutes per model, 8 minutes for images) on CPU. They are not the one-hour qualification runs of plan 10.2.** They exercise the harness and screen for gross growth. They do not qualify gate G4, and no manifest gate status changes with them. The one-hour runs, the CUDA runs and the gate records stay with the maintainer.

- Tool: `ai-dotnet/tools/Starlings.Soak.Tool`. Usage is in `ai-dotnet/README.md` ("Operational qualification") and in the header of `Program.cs`.
- Tests: an `OperationalTests` class in each model test project, built on `ai-dotnet/tests/Shared/Operational/OperationalChecks.cs`.

## Runs in this directory (2026-10-13)

The machine was a shared workstation: i7-5820K (6 cores, 12 threads, AVX2), 94 GiB RAM, Linux 7.2.5, .NET 10.0.12, ONNX Runtime 1.30.0 CPU, ONNX Runtime GenAI 0.17.1 CPU. Other agents' builds, tests and model runs used the CPU at the same time (load average 12 to 23 during the runs). Every report records this. Latencies are therefore pessimistic and noisy. Each run used default threads, 2 concurrent callers, default glibc allocator settings, and source `fb27658` plus this change (uncommitted at run time, so `git_dirty: true`). After the runs, the verdicts were recomputed with the final analysis (`Starlings.Soak.Tool reanalyze`); the samples are unchanged.

| Profile | Cycles (load, serve, unload) | Requests checked | Wrong outputs / errors | Deadline requests cancelled | Cancellation latency p50 / p95 / max (ms) | Loaded RssAnon median (range), MiB | Unloaded RssAnon median (range), MiB | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multilingual-e5-small/onnx-fp32-cpu | 62 | 11,160 | 0 / 0 | 1,240 | 4.8 / 32 / 80 | 831 (785 to 886), growth | 371 (328 to 429), growth | **FAIL (screening)**: about 33 to 44 MiB fitted growth over 4 minutes. See below. |
| gliner2-base-v1/onnx-fp32-cpu | 41 | 3,690 | 0 / 0 | 410 | 3.1 / 23 / 45 | 1,804 (1,639 to 1,870), plateau | 1,133 (1,024 to 1,182), retention | PASS, with allocator retention between loads |
| embeddinggemma-2/onnx-fp32-cpu (text) | 46 | 2,484 | 0 / 0 | 276 | 285 / 559 / 759 | 2,151 (2,116 to 2,186), plateau | 1,342 (1,279 to 1,382), plateau | PASS |
| embeddinggemma-2/image-fp32-cpu | 12 | 48 | 0 / 0 | 12 | 19 / 48 / 69 | 5,061 (4,579 to 5,266), plateau | 2,166 (2,000 to 2,320), retention | PASS, with retention; only 10 post-warmup cycles |
| qwen3.5-0.8b/ortgenai-fp32x-cpu | 10 | 110 | 0 / 0 | 10 | 3,817 / 5,289 / 6,039 | 4,248 (4,211 to 4,261), plateau | 1,305 (282 to 2,283), retention | PASS, with retention; only 8 post-warmup cycles |

The managed heap has no trend for any model: a plateau verdict in every series, moving by at most about 8 MiB for e5 and about 30 MiB for EmbeddingGemma 2, whose tokenizer is large. In every request, every fixed input gave the output of the first cycle, through the cancellations, the concurrency and the reloads: no state leakage.

### How to read the memory verdicts

- `unloaded` is a settled sample (full GC) after disposing everything, once per cycle. `loaded` is a settled sample after the cycle's requests, with the model still loaded. The tool fits a slope over the post-warmup window. Growth counts only if it continues in both halves of the window (`Analysis.cs`).
- **Retention:** the unloaded series grows but the loaded series plateaus. The memory kept between loads is reused by the next load. If it were leaked, every loaded sample would be higher by the same amount. This is the glibc allocator keeping freed heap memory, and it is not a leak. The tool reports it and does not fail on it. Evidence for GLiNER2 is in `diagnostics/`:
  - `gliner2-malloc-trim-*`: `--malloc-trim`. After each unload, `malloc_trim(0)` returns about 90% of the unloaded RssAnon (about 1,000 MiB to 50 to 200 MiB).
  - `gliner2-static-mmap-threshold-*`: run with `MALLOC_MMAP_THRESHOLD_=65536 MALLOC_TRIM_THRESHOLD_=131072`. The unloaded RssAnon is flat at 57 to 78 MiB over 21 cycles, so unloading releases the model. With these settings the loaded series rises over the 5 minutes (914 to 1,165 MiB), while the unloaded one stays flat. That is the per-session peak of ONNX Runtime's CPU arena varying from cycle to cycle, which every unload releases. The run's own `FAIL` verdict is the per-series screening rule, which does not cross-check the unloaded series.
  - Deployments that load and unload models repeatedly in one process can bound the resident footprint between loads with those glibc settings. Whether retention plateaus under the default settings is a question for the one-hour run.
  - The one-hour CPU run of GLiNER2 (2026-10-14) failed on about 65 MiB of growth in both series under default glibc settings. The diagnosis is in [`../gliner2-soak-growth/`](../gliner2-soak-growth/README.md): glibc's resident high-water mark, which saturates after about 30 minutes; no leak. The deployment recommendation is `MALLOC_MMAP_THRESHOLD_=65536`.
- **Private bytes** on Linux is .NET's `VmData` address space (malloc arenas reserve 64 MiB each without touching it), not resident memory. It is reported but does not gate. On Windows it gates.

### e5: open screening failure

In the 5-minute run, e5's loaded and unloaded RssAnon both rise by about 33 to 44 MiB. That is the pattern of a cumulative leak, so the tool fails the run. The evidence does not confirm a leak:

- The first harness pass (a dedicated thread per caller per cycle) plateaued for e5. A 20-minute repeat on the final harness was killed from outside at 7 minutes (SIGKILL, exit 137, while the full test suite was also running; `systemd-oomd` is active on the machine, but the readable journal has no entry; the soak itself used under 1 GiB). Over its 29 cycles it stayed in a band, loaded 622 to 803 MiB and unloaded 117 to 345 MiB, with no monotone trend.
- A `--malloc-trim` diagnostic run grew 1.8 MiB per cycle after trimming.

This needs the one-hour run on a quiet machine (maintainer action M6) before a conclusion. The tool now writes a checkpoint every 60 s, so a killed run keeps its data.

### Other findings

- **Qwen3.5 on CPU: a deadline does not interrupt the prefill.** In the soak, a 256-token generation from a prompt of about 330 tokens with a 5 to 150 ms deadline threw 3.8 s (p50) to 6.0 s (max) after the deadline. On CPU the prefill is one native call: `Qwen35ChatOptions.PrefillChunkTokens` defaults to chunking only on CUDA, and cancellation is checked between chunks and steps. This matches the documented contract ("cancelled before the prefill or between decode steps"). For latency-sensitive callers on CPU, set `PrefillChunkTokens`. With the short `en-list` prompt the latency was 47 to 118 ms (`OperationalTests.TimedOutRequestsReportCancellationLatencyThenReuse`).
- **EmbeddingGemma 2 text: cancellation granularity is one graph node.** With 4 x 1,500 to 2,048-token inputs, latency was 285 ms (p50) to 759 ms (max) in the soak and 440 to 580 ms in the test. On a CPU shared with the whole test suite it reached 4.9 s once. The test bound is now 10 s, and the observed values are printed.
- **Killed runs.** Four soak processes were killed from outside (SIGKILL, exit 137) while other jobs were using memory: in the first pass, EmbeddingGemma 2 text and Qwen (both repeated in the runs above), plus a diagnostic run and the 20-minute e5 run.

## Plan 10.2 coverage map

New tests are in each project's `OperationalTests.cs`. Existing tests are named by class. "Soak" means every soak run exercises the item.

| Item | e5 | GLiNER2 | EmbeddingGemma 2 | Qwen3.5 |
| --- | --- | --- | --- | --- |
| Queue saturation / concurrency limits | new `SaturatedCallersOnOneEncoderAllComplete` (4 x cores callers, weight-free), `RealModelSaturatedCallersMatchTheirSingleCallResults`; existing `CpuParityTests`/`EncoderBehaviorTests` concurrency | new `SaturatedCallersOnOneExtractorAllComplete`; existing concurrency tests | new `SaturatedCallersOnOneEncoderAllComplete`; existing concurrency tests | new `SaturatedRequestsAllCompleteWithTheirSingleRequestIds` (12 concurrent generators); existing `ConcurrentRequestsAreIndependent` |
| Timed-out requests | existing `CpuParityTests.CancellationInterruptsALargeBatchAndEncoderStaysUsable` (`CancelAfter`); soak | new `RealModelTimedOutRequestsReportCancellationLatencyThenReuse`; soak | new `RealModelTimedOutRequestsReportCancellationLatencyThenReuse`; soak | new `TimedOutRequestsReportCancellationLatencyThenReuse`; soak |
| Failed model initialization | new `FailedInitializationIsTypedAndTheNextLoadWorks` (missing, garbage, truncated, empty, wrong graph, wrong digest, corrupt tokenizer) | new, same name | new, same name, plus `ImageEmbedderWithAMissingOrCorruptGraphFailsTypedAndReleasesWhatItLoaded` | new `MalformedConfigurationOrMissingGraphFailsTyped`, `UnsupportedVariantAndCorruptTemplateFailBeforeAnyNativeWork` |
| Malformed lock or manifest | new `MalformedLocksAndSnapshotsAreRejected` (9 lock corruptions, empty and wrong-size snapshots); existing `AcquisitionTests` | new, same | new, same, text and image locks | new, same, both variants; malformed `genai_config.json` above |
| Unsupported provider rejected, not ignored | new `UnsupportedProvidersAreRejectedNotIgnored` (CUDA on the CPU package, undefined provider, invalid device) | new, plus out-of-profile settings | new, plus unsupported dimension | new (undefined provider, CUDA on the CPU package); fp16x on CPU above |
| Cancellation then reuse | existing `CpuParityTests`, `EncoderBehaviorTests`; soak | existing `ExtractorBehaviorTests`, `CpuParityTests`; new timed-out test; soak | existing `CpuParityTests`, `EncoderBehaviorTests`; new timed-out test; soak | existing `CpuModelTests.CancellationBetweenStepsThenReuse`; new timed-out test; soak |
| Disposal during in-flight work / shutdown | new `DisposalDuringInFlightWorkFinishesOrThrowsObjectDisposed` (weight-free), `RealModelDisposalDuringALargeBatch`; existing runtime `ResourceTests.DisposeDuringRunLetsTheRunFinishThenReleases` | new, same two | new, same two | new `DisposalDuringAnInFlightRequestLetsItFinishThenRejectsNewOnes` (the in-flight request completes correctly, new requests get `ObjectDisposedException`) |
| Cancellation latency measured and reported | existing `CpuParityTests` (reported); soak p50/p95/max | new timed-out test; soak | new timed-out test; soak | new timed-out test; soak |
| Repeated load/unload, bounded memory | existing `ModelMemoryTests`; soak | soak | existing `ModelMemoryTests`; soak | soak |

There is no request queue or admission limit in the libraries (plan 5.3 adds one only when a benchmark justifies it). "Queue saturation" is therefore tested as oversubscription: more concurrent callers than cores on one shared instance. Every call must complete, with no deadlock and no rejection, and must return its single-call result. The tests print the latency spread.

Not covered here: EmbeddingGemma 2 video (the FFmpeg decoder), the Microsoft.Extensions.AI and ML.NET adapters under soak, and every CUDA profile. The tool supports CUDA (`-p:UseCuda=true`, `--cuda`, GPU memory from `nvidia-smi`), but it was not run on a GPU.

## Reproduce

```sh
STARLING_MODELS=<cache> dotnet run --project ai-dotnet/tools/Starlings.Soak.Tool -c Release -- all --duration 5m --out <dir>
# qualification length (default): --duration 1h; diagnostics: --malloc-trim, or MALLOC_MMAP_THRESHOLD_=65536 in the environment
dotnet run --project ai-dotnet/tools/Starlings.Soak.Tool -c Release -- reanalyze <dir>/*.json
```
