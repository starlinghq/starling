# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 3600 s / 3601 s |
| Load/unload cycles | 840 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 45360; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1298, 2122 |
| Unload ms (median, max) | 11.7, 35.6 |
| Request p50 ms (median over cycles) | 53.8 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 5040 | 0 | 227.98 | 334.01 | 1134.97 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 656 | 288.5 | 256.2 | 256.9 | 223.3 | 321.3 | -13.9 | -11.1 | -4.9 | -10.7 | 32.0 | plateau |
| unloaded | working_set | yes | 656 | 1573.8 | 1633.1 | 1606.0 | 1527.7 | 1657.4 | 81.9 | 65.5 | -10.7 | 11.3 | 32.1 | plateau |
| unloaded | private | no | 656 | 1790.9 | 1808.6 | 1797.8 | 1665.5 | 1890.2 | 100.0 | 80.0 | -13.3 | 4.9 | 36.0 | plateau |
| unloaded | rss_anon | yes | 656 | 1493.0 | 1551.9 | 1525.1 | 1446.8 | 1576.3 | 81.3 | 65.0 | -11.0 | 11.3 | 32.0 | plateau |
| loaded | managed_heap | yes | 656 | 367.2 | 334.9 | 335.6 | 302.0 | 400.0 | -13.2 | -10.6 | -4.9 | -9.7 | 32.0 | plateau |
| loaded | working_set | yes | 656 | 2360.1 | 2419.4 | 2399.5 | 2340.9 | 2684.6 | 60.3 | 48.2 | -15.1 | 13.5 | 48.0 | plateau |
| loaded | private | no | 656 | 2830.9 | 2840.6 | 2845.1 | 2735.2 | 3349.3 | 53.0 | 42.4 | -23.7 | 5.5 | 56.9 | plateau |
| loaded | rss_anon | yes | 656 | 2279.4 | 2338.3 | 2318.5 | 2260.1 | 2603.7 | 59.7 | 47.7 | -15.4 | 13.5 | 46.4 | plateau |

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
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"1","DOTNET_ROOT_X64":"/home/mark/.dotnet"} |
| git_commit | 69f00f7e9a915bef3fbd939bbb254ff47e36848c |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/mark/.claude/jobs/944c911d/tmp/soak-1h/embeddinggemma2 --duration 1h --note G4 one-hour qualification soak, CPU, quiet machine |
| machine_note | G4 one-hour qualification soak, CPU, quiet machine |
| load before (loadavg, MemAvailable MB) | [4.78,6.1,6.39], 37974 |
| load after (loadavg, MemAvailable MB) | [10.22,9.19,8.08], 29198 |
