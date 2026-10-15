# G4 soak: embeddinggemma-2/onnx-fp32-cuda

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Duration requested / elapsed | 3600 s / 3603 s |
| Load/unload cycles | 1303 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 70362; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1389, 2071 |
| Unload ms (median, max) | 83.1, 98.8 |
| Request p50 ms (median over cycles) | 16.2 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 7059 | 759 | 0.57 | 1.75 | 21.23 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 1042 | 223.3 | 256.4 | 255.9 | 223.3 | 325.5 | 12.1 | 9.7 | 13.5 | -18.7 | 32.0 | plateau |
| unloaded | working_set | yes | 1042 | 1651.1 | 1649.5 | 1643.4 | 1608.4 | 1860.3 | 17.2 | 13.8 | 15.7 | -11.8 | 32.9 | plateau |
| unloaded | private | no | 1042 | 1750.1 | 1749.1 | 1743.8 | 1681.4 | 1981.3 | 24.4 | 19.5 | 18.2 | -20.3 | 34.9 | plateau |
| unloaded | rss_anon | yes | 1042 | 1368.7 | 1366.6 | 1360.8 | 1325.9 | 1577.5 | 16.6 | 13.3 | 15.3 | -11.8 | 32.0 | plateau |
| unloaded | gpu | yes | 1042 | 258.0 | 258.0 | 258.0 | 258.0 | 258.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 1042 | 302.0 | 335.1 | 334.7 | 302.0 | 404.2 | 12.1 | 9.7 | 13.9 | -19.4 | 32.0 | plateau |
| loaded | working_set | yes | 1042 | 1857.5 | 1883.9 | 1883.9 | 1827.2 | 1939.9 | 14.9 | 12.0 | 11.7 | -11.3 | 37.7 | plateau |
| loaded | private | no | 1042 | 1972.6 | 1999.6 | 1996.7 | 1923.4 | 2088.0 | 22.2 | 17.8 | 14.2 | -20.1 | 39.9 | plateau |
| loaded | rss_anon | yes | 1042 | 1575.1 | 1601.0 | 1601.2 | 1544.6 | 1657.1 | 14.3 | 11.5 | 11.4 | -11.4 | 32.0 | plateau |
| loaded | gpu | yes | 1042 | 3370.0 | 3372.0 | 3372.0 | 3370.0 | 3644.0 | -1.0 | -0.8 | -10.3 | 7.3 | 67.4 | plateau |

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
| genai_native_package | Microsoft.ML.OnnxRuntimeGenAI.Cuda/0.17.1 |
| gpu_identity | 0, NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB; 1, NVIDIA GeForce RTX 4060 Ti, 610.57.04, 16380 MiB |
| cpu_governor | schedutil |
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"1"} |
| git_commit | n/a |
| git_dirty | n/a |
| command_line | embeddinggemma2 --cuda --duration 1h --out /home/mark/.cache/starling-qualification/2026-10-15/soak/embeddinggemma2-cuda --note G4 one-hour qualification soak, cuda, quiet machine |
| machine_note | G4 one-hour qualification soak, cuda, quiet machine |
| load before (loadavg, MemAvailable MB) | [1.84,2.05,2.12], 42709 |
| load after (loadavg, MemAvailable MB) | [1.69,1.98,2.27], 41119 |
