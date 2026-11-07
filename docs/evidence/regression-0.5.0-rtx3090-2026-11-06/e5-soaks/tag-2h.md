# G4 soak: multilingual-e5-small/onnx-fp32-cuda

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error (note: unloaded private (address space, report-only) grows 94 MB; loaded private (address space, report-only) grows 96 MB)

## Run

| Item | Value |
| --- | --- |
| Model lock | intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3 |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=on, trim_native_heap_on_release=on, dedicated_run_threads=off, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 7200 s / 7201 s |
| Load/unload cycles | 3498 |
| Requests per cycle, concurrency | 200, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 629640; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1159, 2111 |
| Unload ms (median, max) | 25.8, 106.3 |
| Request p50 ms (median over cycles) | 2.8 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 9758 | 60202 | 9.10 | 18.43 | 39.71 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 2786 | 40.5 | 41.8 | 48.9 | 40.5 | 68.8 | 1.8 | 2.9 | 3.5 | -4.4 | 32.0 | plateau |
| unloaded | working_set | yes | 2786 | 1039.2 | 1121.6 | 1113.6 | 1009.6 | 1159.0 | 76.1 | 121.7 | 64.2 | -5.7 | 32.0 | plateau |
| unloaded | private | no | 2786 | 1271.2 | 1306.7 | 1307.6 | 1225.5 | 1406.0 | 59.0 | 94.4 | 50.1 | 10.4 | 32.0 | growth |
| unloaded | rss_anon | yes | 2786 | 684.5 | 766.5 | 758.4 | 654.7 | 803.8 | 75.7 | 121.2 | 63.7 | -5.7 | 32.0 | plateau |
| unloaded | gpu | yes | 2786 | 270.0 | 270.0 | 270.0 | 270.0 | 270.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 2786 | 52.0 | 53.3 | 60.4 | 49.2 | 80.3 | 1.9 | 3.0 | 3.6 | -4.4 | 32.0 | plateau |
| loaded | working_set | yes | 2786 | 1161.5 | 1228.4 | 1223.4 | 1115.7 | 1286.4 | 76.5 | 122.3 | 60.6 | -8.4 | 32.0 | plateau |
| loaded | private | no | 2786 | 1355.5 | 1390.9 | 1390.7 | 1309.8 | 1468.4 | 59.8 | 95.6 | 47.8 | 8.7 | 32.0 | growth |
| loaded | rss_anon | yes | 2786 | 804.9 | 871.2 | 866.2 | 758.8 | 929.3 | 76.2 | 121.8 | 60.0 | -8.4 | 32.0 | plateau |
| loaded | gpu | yes | 2786 | 1336.0 | 1398.0 | 1398.0 | 950.0 | 1464.0 | 13.3 | 21.3 | 38.6 | 32.2 | 32.0 | plateau |

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
| genai_native_package | Microsoft.ML.OnnxRuntimeGenAI.Cuda/0.17.1 |
| gpu_identity | 0, NVIDIA GeForce RTX 3090, 615.78.08, 24576 MiB; 1, NVIDIA GeForce RTX 4060 Ti, 615.78.08, 16380 MiB |
| cpu_governor | schedutil |
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"1","DOTNET_NOLOGO":"1"} |
| git_commit | n/a |
| git_dirty | n/a |
| command_line | e5 --cuda --duration 2h --out <local>/out-e5-tag-2h-clean/e5 --note local RTX 3090, tag ai-v0.5.0, two hours, idle GPU |
| machine_note | local RTX 3090, tag ai-v0.5.0, two hours, idle GPU |
| load before (loadavg, MemAvailable MB) | [1.42,1.98,2.13], 62155 |
| load after (loadavg, MemAvailable MB) | [1.66,2.05,2.66], 52416 |
