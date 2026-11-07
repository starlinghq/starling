# G4 soak: multilingual-e5-small/onnx-fp32-cuda

**Verdict:** FAIL: loaded working_set grows 35 MB (22 MB/h); loaded rss_anon grows 37 MB (23 MB/h)

## Run

| Item | Value |
| --- | --- |
| Model lock | intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3 |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=on, trim_native_heap_on_release=on, dedicated_run_threads=off, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 7200 s / 7201 s |
| Load/unload cycles | 3532 |
| Requests per cycle, concurrency | 200, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 635760; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1161, 2191 |
| Unload ms (median, max) | 26.5, 102.7 |
| Request p50 ms (median over cycles) | 2.8 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 9838 | 60802 | 9.28 | 18.36 | 62.78 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 2831 | 48.6 | 57.5 | 48.6 | 40.5 | 73.8 | 3.3 | 5.3 | 2.3 | 5.6 | 32.0 | plateau |
| unloaded | working_set | yes | 2831 | 1070.8 | 1133.5 | 1105.1 | 1025.3 | 1148.4 | 17.6 | 28.1 | 41.7 | 9.8 | 32.0 | plateau |
| unloaded | private | no | 2831 | 1244.3 | 1333.3 | 1296.4 | 1224.2 | 1383.2 | 16.1 | 25.7 | 37.1 | 39.2 | 32.0 | plateau |
| unloaded | rss_anon | yes | 2831 | 717.2 | 781.4 | 751.2 | 671.8 | 796.3 | 18.6 | 29.8 | 41.2 | 12.9 | 32.0 | plateau |
| unloaded | gpu | yes | 2831 | 270.0 | 270.0 | 270.0 | 270.0 | 270.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 2831 | 60.1 | 68.9 | 60.1 | 51.8 | 85.3 | 3.3 | 5.3 | 2.3 | 5.6 | 32.0 | plateau |
| loaded | working_set | yes | 2831 | 1192.9 | 1240.4 | 1218.3 | 1150.5 | 1279.4 | 22.1 | 35.4 | 44.0 | 19.7 | 32.0 | growth |
| loaded | private | no | 2831 | 1333.8 | 1407.4 | 1374.7 | 1302.3 | 1456.4 | 22.9 | 36.7 | 31.1 | 39.8 | 32.0 | growth |
| loaded | rss_anon | yes | 2831 | 837.3 | 886.3 | 862.7 | 794.9 | 925.4 | 23.2 | 37.1 | 43.5 | 22.8 | 32.0 | growth |
| loaded | gpu | yes | 2831 | 1400.0 | 1398.0 | 1398.0 | 950.0 | 1464.0 | -3.9 | -6.2 | 7.6 | -19.9 | 32.0 | plateau |

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
| command_line | e5 --cuda --duration 2h --out <local>/out-e5-tag-2h/e5 --note local RTX 3090, tag ai-v0.5.0, two hours |
| machine_note | local RTX 3090, tag ai-v0.5.0, two hours |
| load before (loadavg, MemAvailable MB) | [4.19,3.62,2.68], 72754 |
| load after (loadavg, MemAvailable MB) | [3.07,3.78,3.34], 64970 |
