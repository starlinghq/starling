# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** INCOMPLETE (checkpoint at 3491 s of 3600 s; the run did not finish): PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 3600 s / 3491 s |
| Load/unload cycles | 133 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 532; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2745, 4435 |
| Unload ms (median, max) | 38.2, 135.8 |
| Request p50 ms (median over cycles) | 11408.0 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 133 | 0 | 5.78 | 30.19 | 54.58 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 108 | 180.3 | 180.4 | 180.3 | 178.8 | 191.0 | 2.0 | 1.5 | -0.3 | 2.6 | 32.0 | plateau |
| unloaded | working_set | yes | 108 | 2396.7 | 2446.1 | 2495.7 | 2328.0 | 2672.9 | -38.3 | -29.6 | 1.3 | -83.3 | 49.9 | plateau |
| unloaded | private | no | 108 | 2560.0 | 2795.6 | 2716.7 | 2520.0 | 2959.1 | 279.8 | 216.4 | 162.8 | 68.2 | 54.3 | retention |
| unloaded | rss_anon | yes | 108 | 2316.5 | 2364.7 | 2414.4 | 2246.6 | 2591.8 | -39.6 | -30.6 | 0.5 | -83.6 | 48.3 | plateau |
| loaded | managed_heap | yes | 108 | 259.0 | 259.2 | 259.1 | 257.1 | 269.7 | 1.9 | 1.5 | -0.3 | 2.6 | 32.0 | plateau |
| loaded | working_set | yes | 108 | 4813.2 | 4954.2 | 5389.3 | 4785.1 | 5959.4 | -132.5 | -102.5 | 100.4 | -135.9 | 107.8 | plateau |
| loaded | private | no | 108 | 6860.1 | 8016.7 | 7737.4 | 6794.1 | 8101.1 | 19.8 | 15.3 | 396.7 | 109.7 | 154.7 | plateau |
| loaded | rss_anon | yes | 108 | 4733.0 | 4872.8 | 5308.0 | 4703.7 | 5878.4 | -133.8 | -103.5 | 99.6 | -136.1 | 106.2 | plateau |

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
| command_line | embeddinggemma2-image --out /home/mark/.claude/jobs/944c911d/tmp/soak-1h/embeddinggemma2-image --duration 1h --note G4 one-hour qualification soak, CPU, quiet machine |
| machine_note | G4 one-hour qualification soak, CPU, quiet machine |
| load before (loadavg, MemAvailable MB) | [10.22,9.19,8.08], 30647 |
| load after (loadavg, MemAvailable MB) | [12.7,11.28,11.31], 34058 |
