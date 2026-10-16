# EmbeddingGemma 2 one-hour CPU soaks on the hosted runner: thread-rotation retention and dedicated run threads

After #83 (ONNX Runtime's CPU arena off for EmbeddingGemma 2 CPU sessions), two Stable profiles were still not reliable on the hosted qualification runner (`ubuntu-latest`, 4 vCPUs, glibc 2.39, THP `madvise`):

- `embeddinggemma-2/image-fp32-cpu` failed the one-hour soak intermittently, with the arena on or off (run 38874143326: unloaded `rss_anon` +105 MiB, loaded +146 MiB).
- `embeddinggemma-2/onnx-fp32-cpu` (text) passed, but with a loaded growth of 35.7 MiB against a limit of 36.0 MiB in run 38882433616.

## 1. Diagnosis (run 38903310736, `soak_only`, the code of `main` at f43a648)

Two copies of each soak with `/proc/self/smaps` and `mallinfo2` dumps every 5 cycles (`smaps5`, `default+smaps5`, `smaps5+default`) and one with the model kept loaded (`keep-loaded+smaps5`). Reports and dumps: `diagnosis-38903310736/`.

| Soak | Runner | Result | Unloaded `rss_anon` growth (limit 32) | Loaded `rss_anon` growth (limit) |
| --- | --- | --- | --- | --- |
| text `smaps5` | EPYC 7763 | **FAIL** | 62.5 | 70.1 (36.2) |
| text `default+smaps5` | EPYC 7763 | pass | 14.3 | 18.4 (36.2) |
| text `smaps5+default` | EPYC 9V45 | pass | 14.0 | 16.0 (36.3) |
| text `keep-loaded+smaps5` | EPYC 7763 | pass | 17.1 | 17.1 (32.3) |
| image `smaps5` | EPYC 7763 | **FAIL** | 164.8 | 76.3 (54.7) |
| image `default+smaps5` | EPYC 7763 | pass | -4.9 | -28.8 (52.8) |
| image `smaps5+default` | EPYC 7763 | pass (unloaded 94.4 fitted, halves 76.9 and -27.8) | 94.4 | 39.8 (53.6) |
| image `keep-loaded+smaps5` | EPYC 9V74 | pass | 2.5 | 2.5 (55.5) |

Both failures are on EPYC 7763, the CPU most passing runs had, so the runner's CPU is not the cause. Keeping one session loaded is flat for both, so the growth is not inside a live session: it is memory left behind between loads.

What grew (`../eg2-hosted-soak/analysis/arenas.py <dump> unloaded`; columns: cycle, number of 64 MiB-aligned glibc arena heaps with more than 2 MiB resident, their resident MiB, each heap's resident MiB):

```
text smaps5 (FAIL)                       image smaps5 (FAIL)
  0 3  30 [15, 12, 3]                      0 2   9 [5, 4]
 60 4  72 [19, 19, 18, 16]                20 5  99 [49, 37, 5, 4, 4]
180 5  84 [19, 19, 16, 15, 15]            50 5  86 [43, 29, 7, 4, 3]
300 6 109 [25, 19, 19, 19, 15, 12]        60 8 149 [43, 42, 35, 13, 7, 4, 3, 2]
360 7 127 [24, 19, 19, 19, 19, 15, 12]    70 8 226 [59, 43, 40, 36, 34, 7, 4, 3]
480 7 127 [24, 19, 19, 19, 19, 15, 12]    80 8 183 [59, 40, 36, 22, 12, 7, 4, 3]
```

In the failing image run the whole unloaded growth is in these heaps (`smaps_summary.py`: arena class 9 to 182-226 MiB; the main heap stays at 11 MiB and the other anonymous mappings, which include the managed heap, stay at 290-340 MiB). In the failing text run the heaps hold 30 then 127 MiB. The **number** of heaps that hold a run's worth of buffers grows: the soak has 2 concurrent callers, but text ends with 5 to 6 heaps of about 19 MiB and images with 5 heaps of 34 to 59 MiB.

Mechanism:

1. With the arena off, ONNX Runtime allocates each run's intermediate buffers with `malloc` on the thread that calls `Run` and frees them at the end of the run. glibc serves that thread from its own arena. After the first free of a large mmapped chunk, glibc's dynamic mmap threshold rises (up to 32 MiB), so buffers up to that size come from the arena's heap, and freeing them leaves the memory at the heap's top.
2. A non-main arena's top is returned only by `free`, and only when it exceeds the dynamic trim threshold (twice the mmap threshold, up to 64 MiB; glibc 2.39 `heap_trim` returns early below it). `malloc_trim` releases free chunks inside every arena but never a non-main arena's top (`mtrim` calls `systrim` for the main arena only). So each heap that served a run keeps up to 64 MiB resident.
3. The callers are .NET thread-pool threads. Over an hour the pool starts and retires threads, and a new thread takes a free arena or creates one. A thread that calls ONNX Runtime on an arena that has not held a run before adds another top's worth of resident memory. The count grows step by step, which is the shape in the reports (steps of about 19 MiB for text, 30 to 60 MiB for images) and why the result varies from run to run.

A local 15-minute text soak on 4 CPUs (`taskset -c 0-3`, glibc 2.44) reproduced the shape: 2 callers, 6 heaps of 8 to 26 MiB, 96 to 106 MiB in total.

## 2. Fix: dedicated run threads (library level)

- `Starlings.Runtime.Onnx`: new `OnnxSessionOptions.DedicatedRunThreads` (`bool?`) and `OnnxSessionDiagnostics.DedicatedRunThreads`. When on, `OnnxModelSession.Run` executes the ONNX Runtime call on a small process-wide set of long-lived threads (`RunThreads`) and the caller blocks until it completes. A thread is started only when every existing one is busy, up to `Environment.ProcessorCount`; further concurrent runs queue. A run started from one of these threads runs inline. The threads start without the caller's `ExecutionContext`. `null`, the default, keeps the calling thread for sessions created directly.
- `Starlings.Models.EmbeddingGemma2`: CPU sessions (text, vision, multimodal text) use them unless the caller sets the option. CUDA sessions are unchanged.
- No `mallopt`, no `MALLOC_*` setting, no change to the contract environment, the growth rule, the budgets or the limits.

The same few threads, and so the same few arenas, now serve every run, so the memory at their tops is reused by the next run instead of accumulating. The per-arena top is still kept (up to 64 MiB per run thread); with 2 callers that is a fixed high-water mark of about 2 × 16-20 MiB for text and 2 × up to 64 MiB for images, which the growth rule reads as a plateau. `MALLOC_ARENA_MAX` would additionally cap the other threads' arenas; it is not needed for these soaks and is not added to any contract.

Local check (15 minutes, 4 CPUs): text heaps 2 × 19-20 MiB plus small ones, 57 MiB in total instead of 96 to 106; 2 `starlings-onnx-run` threads in `/proc/<pid>/task/*/comm`. A first version started a third thread when a caller queued its next run before the previous thread was back in its wait; completing the run under the lock and counting threads already woken removed it.

Behavior changes, all documented in the package READMEs:

- More concurrent runs than processors queue instead of oversubscribing the CPU (each run already uses every core through ONNX Runtime's intra-op threads).
- A token cancelled while a run is queued takes effect when the run reaches a thread, at its first node boundary.
- Embeddings are bit-identical on dedicated threads and on the caller (`CpuArenaTests.TextEmbeddingsAreBitIdenticalOnDedicatedRunThreadsAndOnTheCaller`). Request latency in the soaks is unchanged (image p50 on EPYC 7763: about 18.1 s before, 18.0 s after).

## 3. Verification

Full qualification workflow on this branch, `skip_cuda=true`, `soak_variants=["default","smaps5"]` (every soak twice: the qualification variant and one with dumps). Growth is the fitted `rss_anon` growth in MiB against the limit.

Verified commit: 47f1a52 (later commits on the branch change documentation and evidence only, plus the merge of `main`). Format: unloaded growth / limit; loaded growth / limit; halves of the loaded series in brackets.

| Run | Soak | Runner | Verdict | Unloaded | Loaded [1st half, 2nd half] |
| --- | --- | --- | --- | --- | --- |
| [38913307558](https://github.com/starlinghq/starling/actions/runs/38913307558) | text | EPYC 9V45 | pass | -0.7 / 32 | 0.1 / 34.9 [39.7, -12.2] |
| | text `smaps5` | Xeon 8573C | pass | 2.0 / 32 | 4.0 / 34.9 [3.9, 8.1] |
| | image | EPYC 7763 | pass (`retention`) | 42.8 / 32 | 44.5 / 53.9 [48.8, -28.3] |
| | image `smaps5` | Xeon 8573C | pass | -2.6 / 32 | 62.5 / 54.2 [13.4, -6.4] |
| | video | Xeon 8573C | pass | 49.9 / 32 (plateau) | 47.0 / 56.0 [40.1, -10.0] |
| | video `smaps5` | EPYC 7763 | pass | -31.9 / 32 | -15.6 / 54.6 [-5.4, -9.8] |
| [38924489084](https://github.com/starlinghq/starling/actions/runs/38924489084) | text | EPYC 7763 | pass | 5.2 / 32 | 4.1 / 34.6 [1.3, -2.2] |
| | text `smaps5` | EPYC 7763 | pass | -11.3 / 32 | -8.5 / 34.8 [-8.5, -3.1] |
| | image | EPYC 9V45 | pass | 2.7 / 32 | 9.8 / 53.5 [-33.6, 5.9] |
| | image `smaps5` | EPYC 9V45 | pass | -11.8 / 32 | -17.5 / 53.3 [-37.9, 18.9] |
| | video | EPYC 9V45 | pass | -7.2 / 32 | -7.8 / 54.6 [-0.9, 0.4] |
| | video `smaps5` | Xeon 8573C | pass (`retention`) | 40.4 / 32 | 22.0 / 56.8 [36.4, 52.5] |
| [38933567784](https://github.com/starlinghq/starling/actions/runs/38933567784) | text | EPYC 7763 | pass | 0.9 / 32 | 2.0 / 34.9 [3.6, -1.2] |
| | text `smaps5` | EPYC 9V74 | pass | 8.2 / 32 | 9.4 / 34.7 [10.8, 0.2] |
| | image | EPYC 7763 | pass | 15.3 / 32 | 52.6 / 55.1 [78.5, -4.0] |
| | image `smaps5` | EPYC 9V74 | pass | 12.8 / 32 | 15.1 / 54.3 [9.7, -9.3] |
| | video | EPYC 9V74 | pass | 6.4 / 32 | 16.8 / 54.1 [11.0, 11.8] |
| | video `smaps5` | EPYC 7763 | pass | 7.5 / 32 | 22.1 / 55.0 [-17.0, 10.5] |

Text: the largest loaded growth in six soaks is 9.4 MiB against a limit of about 35 (before: 35.7 against 36.0, and 70.1 in the failing diagnosis soak). After unloading, the text arenas settle at two heaps of 15 to 18 MiB (`verification-38913307558/embeddinggemma2-smaps5`: 38 MiB in all arena heaps at cycle 0, 50 at cycle 600), against 3 to 7 heaps and 127 MiB before.

Images: the unloaded arena heaps stay at about 33 MiB for the whole hour (`verification-38913307558/embeddinggemma2-image-smaps5`), against 9 to 226 MiB in the failing diagnosis soak. Unloaded growth is at most 15.3 MiB in five soaks; in the sixth (`retention`, 42.8) the loaded series plateaus.

**Left open: the loaded image series is noisy.** Two of six loaded image fits come close to or above the limit (52.6/55.1 and 62.5/54.2). In both, the second half is negative, so the growth rule reads a plateau. The raw series of run 38933567784 is flat at 2,700 to 2,810 MiB from cycle 4 to 88. The dumps show two parts. The run threads' arena heaps reach their high-water mark once, at most 64 MiB each (arena class 245 to 310 MiB). The main heap varies by plus or minus 50 MiB between loads (1,540 to 1,630 MiB) while `uordblks` stays at 1,193 MiB, which points to memory freed during load (for example graph optimization copies) that stays resident until the release trim. A `malloc_trim` after session creation would likely tighten this. It is not in this change, because it would apply to every `OnnxModelSession`, not only EmbeddingGemma 2, and the series does not grow.

Other jobs in these runs, unrelated to this change:

- Run 38913307558: the `multilingual-e5-small/onnx-fp32-cpu` soak (default variant) failed on EPYC 7763 (unloaded 58.0, loaded 55.7 against 32; its `smaps5` copy passed). e5 does not use dedicated run threads; its sessions are created as before.
- Run 38933567784: `model tests` failed on `Starlings.Models.Qwen35.Tests.OperationalTests.TimedOutRequestsReportCancellationLatencyThenReuse`. The first of three timed-out requests took 11.8 s against a 5 s bound; the other two took 131 and 52 ms. Qwen 3.5 runs on ONNX Runtime GenAI and does not reference `Starlings.Runtime.Onnx`.
- Run 38924489084 passed every job.

## Reproduce

```sh
gh workflow run qualification.yml --ref <branch> -f soak_only=true -f soak_duration=1h \
  -f "soak_workloads=embeddinggemma2 embeddinggemma2-image" \
  -f 'soak_variants=["smaps5","no-run-threads+smaps5"]'
python3 -I docs/evidence/eg2-hosted-soak/analysis/arenas.py <x.smaps.jsonl> unloaded
python3 -I docs/evidence/eg2-hosted-soak/analysis/smaps_summary.py <x.smaps.jsonl>
```

The soak tool's `--run-threads` and `--no-run-threads` (workflow variant parts `run-threads`, `no-run-threads`) override the library default for A/B runs.
