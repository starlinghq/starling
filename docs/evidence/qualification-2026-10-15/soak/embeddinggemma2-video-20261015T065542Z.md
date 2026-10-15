# G4 soak: embeddinggemma-2/video-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 3600 s / 3609 s |
| Load/unload cycles | 292 |
| Requests per cycle, concurrency | 4, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 876; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1993, 2866 |
| Unload ms (median, max) | 20.6, 43.4 |
| Request p50 ms (median over cycles) | 5353.4 |

## Timed-out requests and cancellation latency

Every 4th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 292 | 0 | 2.77 | 18.74 | 33.56 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 237 | 196.4 | 198.3 | 197.2 | 196.2 | 229.0 | -5.0 | -4.0 | 0.0 | -3.6 | 32.0 | plateau |
| unloaded | working_set | yes | 237 | 2139.5 | 2206.7 | 2240.7 | 2065.1 | 2350.0 | -26.7 | -21.3 | 24.4 | -5.1 | 44.8 | plateau |
| unloaded | private | no | 237 | 2234.0 | 2303.0 | 2341.8 | 2221.0 | 2450.1 | -34.8 | -27.8 | 14.3 | -1.0 | 46.8 | plateau |
| unloaded | rss_anon | yes | 237 | 2058.6 | 2125.1 | 2159.1 | 1983.9 | 2268.8 | -27.3 | -21.8 | 24.0 | -5.1 | 43.2 | plateau |
| loaded | managed_heap | yes | 237 | 275.1 | 277.1 | 276.0 | 274.9 | 307.7 | -5.0 | -4.0 | 0.0 | -3.6 | 32.0 | plateau |
| loaded | working_set | yes | 237 | 3902.5 | 3830.0 | 3891.2 | 3639.1 | 4346.7 | -30.1 | -24.1 | -44.7 | -58.5 | 77.8 | plateau |
| loaded | private | no | 237 | 5174.1 | 5263.0 | 5265.2 | 5145.2 | 5338.2 | -6.9 | -5.5 | 13.0 | -15.6 | 105.3 | plateau |
| loaded | rss_anon | yes | 237 | 3821.6 | 3748.4 | 3810.0 | 3557.6 | 4265.2 | -30.7 | -24.6 | -45.0 | -58.5 | 76.2 | plateau |

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
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"1"} |
| git_commit | 683457f9e10374204778cec00f47314a755612ba |
| git_dirty | false |
| command_line | embeddinggemma2-video --duration 1h --out /home/mark/.cache/starling-qualification/2026-10-15/soak/embeddinggemma2-video-cpu --note G4 one-hour qualification soak, cpu, quiet machine |
| machine_note | G4 one-hour qualification soak, cpu, quiet machine |
| load before (loadavg, MemAvailable MB) | [6.98,7.15,7.42], 43475 |
| load after (loadavg, MemAvailable MB) | [7.22,7.37,7.45], 40717 |
