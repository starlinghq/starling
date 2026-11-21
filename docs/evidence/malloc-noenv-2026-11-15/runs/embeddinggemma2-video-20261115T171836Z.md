# G4 soak: embeddinggemma-2/video-fp32-cpu

**Verdict:** INCOMPLETE (checkpoint at 607 s of 3600 s; the run did not finish): PASS: every gating settled series plateaus after warmup; no wrong output or error (note: loaded private (address space, report-only) grows 115 MB)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on, trim_native_heap_after_load=on, trim_native_heap_after_run=on |
| Duration requested / elapsed | 3600 s / 607 s |
| Load/unload cycles | 11 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 33; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 7585, 12523 |
| Unload ms (median, max) | 266.0, 326.1 |
| Request p50 ms (median over cycles) | 13727.0 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 11 | 0 | 14.96 | 45.92 | 52.60 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 9 | 201.7 | 201.8 | 201.8 | 201.7 | 201.9 | 0.3 | 0.0 | 0.0 | 0.0 | 32.0 | plateau |
| unloaded | working_set | yes | 9 | 444.8 | 446.5 | 450.1 | 440.4 | 465.3 | 112.9 | 8.5 | 8.7 | -13.9 | 32.0 | plateau |
| unloaded | private | no | 9 | 2290.8 | 2357.3 | 2374.5 | 2290.8 | 2412.9 | 653.1 | 49.4 | 73.9 | -40.8 | 47.5 | plateau |
| unloaded | rss_anon | yes | 9 | 365.1 | 365.5 | 370.1 | 360.3 | 384.6 | 96.1 | 7.3 | 7.8 | -14.3 | 32.0 | plateau |
| loaded | managed_heap | yes | 9 | 359.9 | 360.9 | 360.8 | 359.9 | 362.4 | 17.6 | 1.3 | 0.7 | 0.8 | 32.0 | plateau |
| loaded | working_set | yes | 9 | 2194.4 | 2223.8 | 2211.3 | 2194.0 | 2225.5 | 393.6 | 29.7 | 14.0 | 15.6 | 44.2 | plateau |
| loaded | private | no | 9 | 2915.8 | 3080.5 | 3023.5 | 2915.8 | 3080.5 | 1521.2 | 115.0 | 87.4 | 37.5 | 60.5 | growth |
| loaded | rss_anon | yes | 9 | 2114.3 | 2142.8 | 2130.7 | 2113.9 | 2144.6 | 379.6 | 28.7 | 13.4 | 15.2 | 42.6 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | Intel(R) Core(TM) i7-5820K CPU @ 3.30GHz |
| logical_cpus | 12 |
| ram_total_mb | 96442 |
| os | Omarchy |
| kernel | 7.2.8-9-omarchy |
| runtime | .NET 10.0.14 |
| server_gc | false |
| ort_version | 1.30.0 |
| genai_version | 0.17.1 |
| genai_native_package | Microsoft.ML.OnnxRuntimeGenAI/0.17.1 |
| gpu_identity | 0, NVIDIA GeForce RTX 3090, 615.78.08, 24576 MiB; 1, NVIDIA GeForce RTX 4060 Ti, 615.78.08, 16380 MiB |
| cpu_governor | schedutil |
| allocator_and_runtime_env | {"DOTNET_MULTILEVEL_LOOKUP":"0"} |
| git_commit | e1f11dbd1da66809ef8e77e56ad44dc33fa0b3fd |
| git_dirty | false |
| command_line | embeddinggemma2-video --out /home/mark/.hermes/cache/scratch/soak-noenv/runs --duration 1h --note issue #178 no-env: no MALLOC_ env, THP madvise, local workstation |
| machine_note | issue #178 no-env: no MALLOC_ env, THP madvise, local workstation |
| load before (loadavg, MemAvailable MB) | [8.97,9.8,6.04], 59829 |
| load after (loadavg, MemAvailable MB) | [19.49,19.9,13.79], 46533 |
