# e5 one-hour CPU soak on the hosted runner: thread-rotation retention and dedicated run threads; Qwen 3.5 cancellation outlier

Two intermittent failures on Stable CPU profiles in the hosted qualification workflow (`ubuntu-latest`, 4 vCPUs, glibc 2.39) after #86:

1. `multilingual-e5-small/onnx-fp32-cpu`, one-hour soak, run [38913307558](https://github.com/starlinghq/starling/actions/runs/38913307558) (default variant, EPYC 7763): unloaded `rss_anon` growth 58.0 MiB and loaded 55.7 MiB against a limit of 32. Its `smaps5` copy in the same run passed, and so did both copies in runs 38924489084 and 38933567784.
2. `Starlings.Models.Qwen35.Tests.OperationalTests.TimedOutRequestsReportCancellationLatencyThenReuse`, run [38933567784](https://github.com/starlinghq/starling/actions/runs/38933567784): one cancelled request took 11.8 s against a 5 s bound (the other two 131 and 52 ms).

## 1. e5: the same mechanism as EmbeddingGemma 2

### Diagnosis from the dumps

The failing soak had no memory dumps, so the diagnosis uses the three `smaps5` soaks of runs 38913307558, 38924489084 and 38933567784 (same code, dumps every 5 cycles) and an A/B run.

`../eg2-hosted-soak/analysis/arenas.py <dump> unloaded` (cycle, number of 64 MiB-aligned glibc arena heaps with more than 2 MiB resident, their resident MiB, each heap's MiB) on run 38913307558:

```
  0 3  13 [5, 4, 4]
195 8 101 [22, 21, 19, 16, 13, 4, 3, 3]
475 4  72 [23, 23, 22, 4]
955 7 117 [26, 23, 21, 16, 16, 11, 4]
```

After unloading, with no session alive, between 1 and 8 per-thread glibc heaps hold 4 to 139 MiB (`mallinfo2` `uordblks` stays at 10 to 12 MiB, so this is free memory, not live allocations). The soak has 2 callers; the extra heaps belong to thread-pool threads that ran a request earlier. This is the shape #86 found for EmbeddingGemma 2 (`../eg2-soak-reliability/`): each run allocates and frees memory with `malloc` on the thread that calls ONNX Runtime; glibc keeps freed memory at the top of a non-main heap resident (`malloc_trim` never returns a non-main heap's top), and callers from the rotating .NET thread pool spread it over more and more heaps. e5 keeps ONNX Runtime's CPU arena on, so the arena's own regions, allocated on whichever thread extends them and freed at unload, go the same way.

Per mapping class (`analysis/classes.py`, fitted growth over the post-warmup window, then the two halves, then the range; unloaded series, MiB), the heap class is what swings: in the passing `smaps5` soaks it ranges 8 to 135 MiB, with half-window fits of -36 to +20 MiB. A swing of that size in one half of the window, in the same direction as the other half, is what the growth rule reads as growth; the failing soak's unloaded series moves by up to 100 MiB within each 3-minute bin and its mean rises from about 155 to 215 MiB.

GLiNER2 CPU is not exposed: its contract sets `MALLOC_MMAP_THRESHOLD_=65536`, so run buffers are mmapped and returned, and in all three of its `smaps5` soaks one arena heap of 4 MiB stays constant for the hour (unloaded `rss_anon` growth 2.6 to 4.3 MiB). It is unchanged.

### A/B on the hosted runner (run [38948732068](https://github.com/starlinghq/starling/actions/runs/38948732068), `main` at 2b7e742)

Three soaks with the library default (caller threads) and three with `--run-threads`, all with dumps every 5 cycles. Unloaded series (MiB):

| Variant | CPU | Verdict | Unloaded `rss_anon` growth / 32 [halves] | Loaded growth / 32 [halves] | Arena heaps after unloading: range, fit [halves] | Request p50 ms |
| --- | --- | --- | --- | --- | --- | --- |
| `smaps5` | EPYC 7763 | pass | -40.6 [-33.3, 10.4] | -43.8 [-37.4, 6.1] | 16-128, -32.0 [-18.5, 7.9] | 17.4 |
| `default+smaps5` | EPYC 9V45 | pass | 4.8 [-23.0, 24.1] | 0.0 [-19.0, 23.7] | 13-127, 7.2 [-30.9, 34.0] | 8.0 |
| `smaps5+default` | EPYC 9V74 | pass | -23.1 [26.0, -4.0] | -21.3 [29.0, -7.6] | 6-139, -24.5 [19.3, 7.2] | 13.8 |
| `run-threads+smaps5` | EPYC 7763 | pass | 6.4 [7.2, 6.0] | 7.2 [7.5, 8.5] | 16-60, -1.7 [-1.6, -1.7] | 17.2 |
| `default+run-threads+smaps5` | EPYC 7763 | pass | -7.0 [-0.6, -8.6] | -8.1 [-3.1, -7.6] | 14-57, -3.9 [5.2, -3.6] | 16.9 |
| `smaps5+run-threads` | Xeon 8573C | pass | 1.0 [0.0, 8.5] | -29.3 [-4.2, -19.3] | 16-62, -2.8 [-4.2, -5.1] | 10.0 |

All six passed this time (the failure rate is about one soak in six), but the distributions differ: on dedicated run threads the arena heaps after unloading stay within 14 to 62 MiB with half-window fits of at most 5 MiB, against 6 to 139 MiB and half fits of up to 34 MiB on caller threads. The unloaded `rss_anon` range narrows from about 130 MiB (116 to 252) to 70 MiB (105 to 180 on EPYC 7763). Request latency on the same CPU model is unchanged (EPYC 7763 p50: 17.4 ms on caller threads; 17.2 and 16.9 ms on run threads).

The other anonymous mappings (not glibc heaps; 4 MiB-aligned, consistent with .NET GC regions) range 70 to 190 MiB and did not trend in these runs (fits -10 to +7 MiB).

### Fix

`Starlings.Models.E5`: `E5Encoder.Create` gives CPU sessions `OnnxSessionOptions.DedicatedRunThreads = true` unless the caller set it (`E5Encoder.SessionDefaults`, the same pattern as EmbeddingGemma 2). ONNX Runtime's CPU arena keeps its default. CUDA sessions and sessions passed to the `E5Encoder` constructor are unchanged. The ML.NET and `Microsoft.Extensions.AI` adapters create their encoders through `E5Encoder.Create`, so they get the default. No `mallopt`, no `MALLOC_*` setting, no change to the growth rule, budgets or limits.

Embeddings are bit-identical on dedicated threads, on the calling thread and with the default (`RunThreadsTests.EmbeddingsAreBitIdenticalOnDedicatedRunThreadsAndOnTheCaller`, every case of the protected `embeddings.jsonl`, single and batched).

## 2. Qwen 3.5: a stall of the whole runner, not a cancellation defect

- The test cancels a streaming request on the `en-list` prompt, which is **22 tokens**: prefill is one short native call, so prefill chunking cannot be the cause.
- On CPU no lock serializes native calls (`GenAIModelOptions.EffectiveSerializeNativeCalls` is false for CPU), and the decode loop checks the token before each step. A latency of 11.8 s therefore means one native call (generator creation, the prefill or one decode step) took about 11.8 s.
- That call was the first request on a freshly loaded model, and the whole runner was slow at that moment. Durations of the Qwen 3.5 model tests from the TRX results of every downloaded run (`analysis/qwen_durations.py`; seconds, `@` the start offset from the module's first test):

| Run | TimedOut | CancellationBetweenSteps | CachedDecoding | Streaming | Concurrent | Saturated |
| --- | --- | --- | --- | --- | --- | --- |
| 11 earlier runs (range) | 4.9-10.1 @3-8 | 1.7-4.0 @3-9 | 2.9-8.0 @8-17 | 3.1-5.9 @5-13 | 6.5-11.6 | 7.1-13.6 @8-18 |
| 38933567784 (failed) | **17.5 @25** | **15.7 @23** | 8.7 @44 | 4.7 @39 | 9.1 | 9.8 @42 |

  In the failing run the module's first tests started 23 to 25 s after its start instead of 3 to 9 s (the two class fixtures each load the 3.1 GB model in parallel), and the first test of each class, running concurrently, took 4 to 8 times its usual time. Tests that started after about 40 s took their usual time. The other class's first test (`CancellationBetweenStepsThenReuse`, 15.7 s instead of 1.7 to 4.0) was slowed the same way, so a single cancelled request is not the cause. The previous test module (Qwen 3.5 `Extensions.AI`) had read the same model files just before, as in the passing runs.
- Locally (4 pinned CPUs, two models loaded and generating concurrently, page cache dropped before load), the first request's first update takes 0.7 s and a decode step 0.33 s; the test's first-iteration latency was 0.28 to 0.40 s.

The documented contract (cancellation checked between decode steps and prefill chunks; a running step is not interrupted) held: the request threw at the first check after the stalled call returned. The library is not changed and the 5 s bound is not changed. What stalled the runner (memory reclaim during the two concurrent 3.1 GB loads on a 16 GB runner, or a noisy neighbour) cannot be told from the existing artifacts, so the shared check now reports, next to each cancellation latency on Linux, the process's major page faults and the kernel's pressure stall totals for CPU, memory and I/O during the call (`OperationalChecks.TimedOutRequestsThenReuse`). A repeat will show which resource stalled.

## Reproduce

```sh
gh workflow run qualification.yml --ref <branch> -f soak_only=true -f skip_cuda=true -f soak_duration=1h -f soak_workloads=e5 \
  -f 'soak_variants=["smaps5","default+smaps5","smaps5+default","no-run-threads+smaps5","default+no-run-threads+smaps5","smaps5+no-run-threads"]'
python3 -I docs/evidence/eg2-hosted-soak/analysis/arenas.py <x.smaps.jsonl> unloaded
python3 -I docs/evidence/e5-soak-reliability/analysis/classes.py <x.smaps.jsonl>
python3 -I docs/evidence/e5-soak-reliability/analysis/qwen_durations.py <dir with <run>/qualification-model-tests/*.trx>
```

After this change the library default is dedicated run threads, so the A/B uses `no-run-threads` for the caller-thread arm.
