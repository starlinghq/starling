# G4 soak: multilingual-e5-small/onnx-fp32-cuda

**Verdict:** FAIL: unloaded working_set grows 116 MB (145 MB/h); unloaded rss_anon grows 115 MB (144 MB/h); loaded working_set grows 115 MB (144 MB/h); loaded rss_anon grows 115 MB (143 MB/h)

## Run

| Item | Value |
| --- | --- |
| Model lock | intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3 |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=on, trim_native_heap_on_release=on, dedicated_run_threads=off, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 3600 s / 3602 s |
| Load/unload cycles | 1854 |
| Requests per cycle, concurrency | 200, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 333720; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1114, 1557 |
| Unload ms (median, max) | 26.0, 38.7 |
| Request p50 ms (median over cycles) | 2.7 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 5100 | 31980 | 9.60 | 18.29 | 40.28 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 1484 | 40.3 | 48.6 | 44.1 | 40.3 | 56.8 | 3.3 | 2.6 | 0.4 | 2.2 | 32.0 | plateau |
| unloaded | working_set | yes | 1484 | 987.3 | 1087.3 | 1070.0 | 971.5 | 1122.9 | 144.7 | 115.8 | 88.3 | 20.6 | 32.0 | growth |
| unloaded | private | no | 1484 | 1128.5 | 1218.8 | 1216.6 | 1093.5 | 1341.2 | 170.0 | 136.0 | 110.8 | 37.6 | 32.0 | growth |
| unloaded | rss_anon | yes | 1484 | 631.6 | 731.2 | 713.9 | 615.8 | 766.8 | 144.1 | 115.3 | 87.9 | 20.5 | 32.0 | growth |
| unloaded | gpu | yes | 1484 | 270.0 | 270.0 | 270.0 | 270.0 | 270.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 1484 | 51.8 | 60.0 | 55.5 | 51.8 | 68.2 | 3.3 | 2.6 | 0.4 | 2.2 | 32.0 | plateau |
| loaded | working_set | yes | 1484 | 1102.4 | 1199.7 | 1177.8 | 1092.1 | 1228.4 | 143.8 | 115.1 | 87.6 | 18.1 | 32.0 | growth |
| loaded | private | no | 1484 | 1244.1 | 1353.8 | 1321.3 | 1209.7 | 1383.4 | 165.4 | 132.3 | 107.8 | 33.9 | 32.0 | growth |
| loaded | rss_anon | yes | 1484 | 744.7 | 841.6 | 819.6 | 734.4 | 870.3 | 143.2 | 114.5 | 87.3 | 18.1 | 32.0 | growth |
| loaded | gpu | yes | 1484 | 1464.0 | 1398.0 | 1398.0 | 950.0 | 1464.0 | 14.4 | 11.5 | -10.0 | 1.0 | 32.0 | plateau |

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
| command_line | e5 --cuda --duration 1h --out <local>/out-main-extra/e5 --note local RTX 3090 regression, main at 3a021dc0b7f8, CUDA libraries from the pinned eg2 requirements |
| machine_note | local RTX 3090 regression, main at 3a021dc0b7f8, CUDA libraries from the pinned eg2 requirements |
| load before (loadavg, MemAvailable MB) | [1.3,1.59,1.88], 74340 |
| load after (loadavg, MemAvailable MB) | [2.06,1.79,1.78], 72816 |
