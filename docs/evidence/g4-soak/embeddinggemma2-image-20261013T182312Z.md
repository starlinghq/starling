# G4 soak: embeddinggemma-2/image-fp32-cpu

> **Short development run (480 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error (note: unloaded working_set retention 210 MB between loads, reused by the next load; unloaded rss_anon retention 210 MB between loads, reused by the next load)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 480 s / 486 s |
| Load/unload cycles | 12 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 48; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3675, 7841 |
| Unload ms (median, max) | 290.2, 422.0 |
| Request p50 ms (median over cycles) | 13530.7 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 12 | 0 | 18.58 | 47.75 | 68.92 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 10 | 161.1 | 180.3 | 170.4 | 161.1 | 180.3 | 18.2 | 1.4 | 0.3 | 22.2 | 32.0 | plateau |
| unloaded | working_set | yes | 10 | 2077.3 | 2398.0 | 2243.9 | 2077.3 | 2398.0 | 2779.9 | 209.7 | 151.8 | 172.9 | 44.9 | retention |
| unloaded | private | no | 10 | 2168.3 | 2556.6 | 2368.6 | 2168.3 | 2556.6 | 3872.9 | 292.1 | 206.4 | 175.3 | 47.4 | retention |
| unloaded | rss_anon | yes | 10 | 2000.0 | 2320.3 | 2166.2 | 2000.0 | 2320.3 | 2777.6 | 209.5 | 151.8 | 172.7 | 43.3 | retention |
| loaded | managed_heap | yes | 10 | 239.8 | 259.0 | 249.2 | 239.8 | 259.0 | 18.2 | 1.4 | 0.3 | 22.2 | 32.0 | plateau |
| loaded | working_set | yes | 10 | 4929.2 | 5162.2 | 5138.6 | 4656.5 | 5343.0 | 4508.4 | 340.1 | -5.6 | -174.7 | 102.8 | plateau |
| loaded | private | no | 10 | 7096.7 | 7808.7 | 7336.7 | 6531.3 | 7808.7 | 11047.8 | 833.3 | -23.4 | 160.5 | 146.7 | plateau |
| loaded | rss_anon | yes | 10 | 4851.9 | 5084.4 | 5060.9 | 4578.7 | 5265.5 | 4505.8 | 339.9 | -5.5 | -174.9 | 101.2 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | Intel(R) Core(TM) i7-5820K CPU @ 3.30GHz |
| logical_cpus | 12 |
| ram_total_mb | 96447 |
| os | Omarchy |
| kernel | 7.2.5-3-omarchy |
| runtime | .NET 10.0.12 |
| server_gc | false |
| ort_version | 1.30.0 |
| genai_version | 0.17.1 |
| genai_native_package | Microsoft.ML.OnnxRuntimeGenAI/0.17.1 |
| gpu_identity | 0, NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB; 1, NVIDIA GeForce RTX 4060 Ti, 610.57.04, 16380 MiB |
| cpu_governor | schedutil |
| allocator_and_runtime_env | {} |
| git_commit | fb276585f2cd5fc6a848d902280c0002a49c2fc6 |
| git_dirty | true |
| command_line | embeddinggemma2-image --duration 8m --out docs/evidence/g4-soak --note short development run on a shared workstation (other agents\u0027 builds, tests and model runs concurrently; see loadavg); CPU only; default glibc allocator settings |
| machine_note | short development run on a shared workstation (other agents\u0027 builds, tests and model runs concurrently; see loadavg); CPU only; default glibc allocator settings |
| load before (loadavg, MemAvailable MB) | [22.82,22.26,20.61], 14840 |
| load after (loadavg, MemAvailable MB) | [20.18,20.12,20.08], 18536 |
