# G4 soak: embeddinggemma-2/onnx-fp32-cpu

> **Short development run (300 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 300 s / 302 s |
| Load/unload cycles | 46 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 2484; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2480, 3445 |
| Unload ms (median, max) | 73.8, 119.7 |
| Request p50 ms (median over cycles) | 101.9 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 276 | 0 | 284.68 | 559.35 | 758.63 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 33 | 288.4 | 255.8 | 255.8 | 223.1 | 288.4 | -76.9 | -5.0 | 29.0 | -6.3 | 32.0 | plateau |
| unloaded | working_set | yes | 33 | 1401.1 | 1444.3 | 1420.3 | 1357.7 | 1459.5 | -81.9 | -5.4 | 30.6 | 9.1 | 32.0 | plateau |
| unloaded | private | no | 33 | 1611.8 | 1648.7 | 1648.7 | 1563.5 | 1712.0 | 809.4 | 52.9 | 62.1 | -5.7 | 33.0 | plateau |
| unloaded | rss_anon | yes | 33 | 1323.7 | 1365.9 | 1342.3 | 1279.3 | 1381.6 | -97.0 | -6.3 | 30.2 | 8.9 | 32.0 | plateau |
| loaded | managed_heap | yes | 33 | 367.1 | 334.5 | 334.5 | 301.8 | 367.1 | -77.0 | -5.0 | 29.0 | -6.3 | 32.0 | plateau |
| loaded | working_set | yes | 33 | 2238.8 | 2230.1 | 2228.7 | 2193.5 | 2264.7 | -259.1 | -16.9 | 7.2 | -9.7 | 44.6 | plateau |
| loaded | private | no | 33 | 2719.8 | 2696.7 | 2710.1 | 2643.6 | 2761.8 | 575.8 | 37.6 | 37.7 | -17.5 | 54.2 | plateau |
| loaded | rss_anon | yes | 33 | 2161.3 | 2151.6 | 2150.8 | 2115.8 | 2186.3 | -274.3 | -17.9 | 6.8 | -9.9 | 43.0 | plateau |

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
| command_line | embeddinggemma2 --duration 5m --out docs/evidence/g4-soak --note short development run on a shared workstation (other agents\u0027 builds, tests and model runs concurrently; see loadavg); CPU only; default glibc allocator settings |
| machine_note | short development run on a shared workstation (other agents\u0027 builds, tests and model runs concurrently; see loadavg); CPU only; default glibc allocator settings |
| load before (loadavg, MemAvailable MB) | [11.76,16.72,19.55], 31487 |
| load after (loadavg, MemAvailable MB) | [17.9,17.92,19.22], 24114 |
