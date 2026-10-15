# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error (note: unloaded working_set retention 672 MB between loads, reused by the next load; unloaded rss_anon retention 671 MB between loads, reused by the next load)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 3600 s / 3616 s |
| Load/unload cycles | 198 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 792; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2172, 3166 |
| Unload ms (median, max) | 42.6, 79.8 |
| Request p50 ms (median over cycles) | 7603.1 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 198 | 0 | 5.25 | 30.22 | 50.92 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 160 | 180.6 | 190.0 | 189.4 | 178.9 | 191.0 | -2.9 | -2.3 | -3.2 | 0.9 | 32.0 | plateau |
| unloaded | working_set | yes | 160 | 1711.9 | 2329.2 | 1827.7 | 1603.3 | 2698.4 | 839.8 | 672.0 | 500.4 | 647.7 | 36.6 | retention |
| unloaded | private | no | 160 | 1867.4 | 2605.5 | 2097.7 | 1830.5 | 3032.3 | 1141.2 | 913.2 | 580.6 | 654.4 | 42.0 | retention |
| unloaded | rss_anon | yes | 160 | 1631.2 | 2247.8 | 1746.6 | 1521.9 | 2617.3 | 838.9 | 671.3 | 499.9 | 647.6 | 34.9 | retention |
| loaded | managed_heap | yes | 160 | 259.3 | 268.8 | 267.9 | 257.1 | 269.7 | -2.9 | -2.3 | -3.3 | 0.9 | 32.0 | plateau |
| loaded | working_set | yes | 160 | 4911.3 | 5230.7 | 5339.0 | 4645.8 | 6132.6 | 54.5 | 43.6 | 238.7 | 167.3 | 106.8 | plateau |
| loaded | private | no | 160 | 6781.1 | 8206.6 | 7781.7 | 6675.8 | 8235.6 | 204.0 | 163.2 | -115.2 | 274.4 | 155.6 | plateau |
| loaded | rss_anon | yes | 160 | 4830.6 | 5149.2 | 5257.8 | 4564.4 | 6051.4 | 53.6 | 42.9 | 238.1 | 167.3 | 105.2 | plateau |

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
| command_line | embeddinggemma2-image --duration 1h --out /home/mark/.cache/starling-qualification/2026-10-15/soak/embeddinggemma2-image-cpu --note G4 one-hour qualification soak, cpu, quiet machine |
| machine_note | G4 one-hour qualification soak, cpu, quiet machine |
| load before (loadavg, MemAvailable MB) | [4.32,5.17,5.49], 43682 |
| load after (loadavg, MemAvailable MB) | [6.98,7.15,7.42], 41294 |
