# GLiNER2 CPU soak growth: diagnosis

The one-hour CPU G4 soak of `gliner2-base-v1/onnx-fp32-cpu` failed (`one-hour-default/`, source `69f00f7`, default glibc and threads): 811 load/serve/unload cycles, 72,990 requests checked, no wrong output. The loaded and unloaded `rss_anon` series both grew by about 65 MiB (81 to 82 MiB/h fitted over the post-warmup window). This directory holds six short controlled runs that locate the growth, and the conclusion.

**Conclusion: no leak in our code, in the tokenizer or in ONNX Runtime's session lifetime. The growth is glibc malloc's resident high-water mark (retention and fragmentation of freed native memory), which saturates. It is outside the library's control. The mitigation is a deployment setting: `MALLOC_MMAP_THRESHOLD_=65536`.** The soak's growth rule is unchanged.

## The one-hour curve saturates

The fitted slope of the one-hour run is an average over a ramp that flattens. The same raw samples (`one-hour-default/*.json`), fitted per window:

| Window (s) | Cycles | Loaded rss_anon slope, MiB/h | Unloaded rss_anon slope, MiB/h |
| --- | --- | --- | --- |
| 144 to 900 (post-warmup start) | 173 | 277 | 252 |
| 900 to 1,800 | 196 | 204 | 176 |
| 1,800 to 2,700 | 205 | -30 | -8 |
| 2,700 to 3,600 | 204 | 91 | 77 |
| 1,800 to 3,600 (second half) | 409 | 15 | 21 |

The medians of the unloaded series in consecutive bins of 67 cycles (about 5 minutes) are 1,121, 1,190, 1,197, 1,223, 1,238, 1,258, 1,264, 1,277, 1,263, 1,273, 1,268, 1,277 and 1,260 MiB: flat from about 30 minutes on. The run fails the screening rule because the window-wide fit is 65 MiB (limit 32 to 39 MiB) and the second half still fits 21 MiB, above a quarter of the limit (Analysis.cs). A cumulative leak of 65 MiB/h would have kept its slope in the second half.

## Controlled runs (this change)

All runs: 9 minutes (`--duration 9m`; the tool's limit of one foreground command is 10 minutes), `--threads 4`, 2 callers, CPU, source `7a915d9` plus this change (`git_dirty: true`), on the shared workstation (i7-5820K, 12 threads; another agent's gate evaluations ran at the same time, load average 6 to 14, recorded in each report). With 4 threads instead of the runtime default, the baseline (a), not the one-hour run, is the reference for (b) to (f). Nine minutes sits in the steep part of the ramp above, so the runs are compared with each other and by shape, not by the fitted slope against the one-hour run. Slopes are least squares over the post-warmup window (the report's "Slope MB/h"); "thirds" are slopes over each third of the run.

| Run | Setting | Cycles | Loaded rss_anon median, MiB (slope MiB/h, verdict) | Unloaded rss_anon median, MiB (slope MiB/h, verdict) | Shape |
| --- | --- | --- | --- | --- | --- |
| (a) `a-baseline/` | default glibc, 100 requests per cycle | 120 | 1,850 (951, plateau) | 1,478 (929, plateau) | Ramp, then flattening; thirds of the loaded series 1,043, 1,112, -390 MiB/h. Noisy: loaded 1,708 to 2,225 MiB from cycle to cycle. |
| (b) `b-malloc-trim/` | `--malloc-trim` | 52 | 1,574 (134, plateau) | 1,015 (440, retention) | **Trimmed** series (after `malloc_trim(0)`): 76 MiB at cycle 0, 150 MiB by cycle 15, then a band of 136 to 169 MiB to cycle 51 (slope 54 MiB/h, plateau). `malloc_trim` returns about 85% of the unloaded memory. |
| (c) `c-malloc-arena-max-2/` | `MALLOC_ARENA_MAX=2` | 55 | 1,612 (44, plateau) | 1,040 (87, plateau) | Lower than (a) by about 250 and 440 MiB; retention between loads remains. |
| (d) `d-mmap-threshold-64k/` | `MALLOC_MMAP_THRESHOLD_=65536` | 56 | 1,115 (129, plateau) | **76 (-2, plateau)** | Unloaded flat at 74 to 82 MiB for every post-warmup cycle: each unload returns everything the load and its requests allocated. Loaded about 740 MiB below (a). |
| (e) `e-requests-only/` | `--keep-loaded` (new): one load, no unloads, 100 requests per cycle | 93 | 1,699 (656, plateau) | (same sample, model loaded) | Monotone steps, flat between them: 1,570, 1,624, ..., 1,657 for 25 cycles, one step of 42 MiB at cycle 56, then 1,699 to 1,700 for the last 37 cycles (last third 0 MiB/h). |
| (f) `f-load-unload-only/` | `--requests-per-cycle 1 --deadline-every 0` | 213 and 228 | 1,346 and 1,353 (98 and 171, plateau) | 935 and 953 (124 and 52, plateau) | Ramp that decays: unloaded thirds 1,376, 739, 256 MiB/h and 1,146, 232, 15 MiB/h. |

Run (f) has two reports: the command was started twice, 47 s apart, and the two processes overlapped for 8 minutes (a harness retry; not intended). They agree with each other. The managed heap plateaued in every run (12 to 13 MiB unloaded), and every request in every run gave the first cycle's output.

## Diagnosis

Each candidate cause, and the evidence against or for it:

- **Native memory of ONNX Runtime sessions not freed on dispose (OrtValue, SessionOptions, arena).** Ruled out by (d). With the mmap threshold fixed at 64 KiB, the unloaded `rss_anon` is flat at 76 MiB over 56 cycles; a per-load leak of any size above the noise (about 4 MiB) would show there. The default run (a) holds about 1.4 GiB more between loads only because glibc keeps freed memory resident. Code review agrees: `OnnxModelSession` releases the session, the session options and the CUDA provider options when the last reference goes (`Release`), and each `RunOptions` is disposed per call.
- **Per-request buffers in our GLiNER2 code.** Ruled out. Every input `OrtValue` and the `IDisposableReadOnlyCollection<OrtValue>` of outputs are in `using` declarations (`Gliner2EntityExtractor.RunChunk`, `ReadOutputs`); the input arrays are managed, sized per call and not pinned beyond the call; nothing static or cached grows. The managed heap is flat. Requests alone (e) do not ramp: they move the resident size in rare steps (the ORT CPU arena or the glibc heap reaching a new peak for a new concurrent shape, such as two 16-text deadline batches at once) and stay flat between them.
- **The tokenizer.** Ruled out. `Starlings.Tokenizers` is a managed assembly (no native runtime assets, nothing disposable); the managed heap is flat at 12 to 13 MiB unloaded and 18 MiB loaded in the one-hour run.
- **glibc retention and fragmentation that creeps.** Confirmed, and it saturates. The memory that persists across unloads is returned by `malloc_trim` (b) and does not accumulate when large blocks are mmapped (d). What remains after trimming grows for about 15 cycles and then holds a band (b). The load/unload ramp (f) decays to near zero within 9 minutes, and the one-hour run flattens after about 30 minutes. glibc's dynamic mmap threshold explains the mechanism: after a large block is freed, glibc raises the threshold, later allocations of that size come from the arena heaps instead of mmap, and freed heap pages are not returned to the kernel; with 4 to 12 ORT threads plus the .NET threads there are several arenas, each with its own high-water mark. Fixing the threshold (d) disables that adaptation. Capping the arenas (c) reduces the footprint but leaves retention.

## Decision

Option (ii): a documented limit and a deployment recommendation, not a library change.

- No library mitigation is adopted. The persistent memory is not in a session (it outlives every session), so ORT session settings (`EnableCpuMemArena`, `EnableMemoryPattern`) cannot remove it, and disabling the ORT CPU arena moves its allocations onto the glibc heaps. Calling `mallopt` or `malloc_trim` from a library would change process-wide allocator behaviour for the host application.
- Deployments on Linux with glibc that load and unload models in one long-lived process, or that need a bounded resident footprint, should set `MALLOC_MMAP_THRESHOLD_=65536` in the process environment (it must be set before the process starts). In (d) it held the unloaded footprint at 76 MiB and lowered the loaded footprint by about 740 MiB. Request latency in these shared-machine runs was not measurably worse (p50 130 ms in (d) against 150 to 155 ms in (b) and (c) at the same machine load); load time was higher (median 2.6 s against 2.1 to 2.2 s), so this needs the G5 benchmark before it becomes a default recommendation for latency-sensitive use. `MALLOC_ARENA_MAX=2` is a weaker alternative.
- Under default glibc settings, the resident footprint of a process that cycles GLiNER2 rises by about 65 to 90 MiB over the first half hour and then holds. This is a documented limit, not a leak.

## One-hour qualification rerun (maintainer; quiet machine)

The rerun with the deployment recommendation, which is the configuration this decision qualifies:

```sh
MALLOC_MMAP_THRESHOLD_=65536 STARLING_MODELS=<cache> dotnet run --project ai-dotnet/tools/Starlings.Soak.Tool -c Release -- \
  gliner2 --duration 1h --out <dir> --note "G4 one-hour qualification soak, CPU, quiet machine, MALLOC_MMAP_THRESHOLD_=65536"
```

The default-allocator run may be repeated for comparison (same command without the variable). Under the unchanged growth rule it can fail again on the saturating ramp; its windowed slopes, as in the table above, are the evidence to read alongside the verdict. The tool records `MALLOC_*` variables in each report's environment manifest.

## Reproduce the diagnostic runs

```sh
dotnet build ai-dotnet/tools/Starlings.Soak.Tool -c Release   # then run each with STARLING_MODELS=<cache>
T=ai-dotnet/tools/Starlings.Soak.Tool/bin/Release/net10.0/Starlings.Soak.Tool
$T gliner2 --duration 9m --threads 4 --out a                                   # (a)
$T gliner2 --duration 9m --threads 4 --malloc-trim --out b                     # (b)
MALLOC_ARENA_MAX=2 $T gliner2 --duration 9m --threads 4 --out c                # (c)
MALLOC_MMAP_THRESHOLD_=65536 $T gliner2 --duration 9m --threads 4 --out d      # (d)
$T gliner2 --duration 9m --threads 4 --keep-loaded --out e                     # (e)
$T gliner2 --duration 9m --threads 4 --requests-per-cycle 1 --deadline-every 0 --out f   # (f)
```
