# G4 soak: multilingual-e5-small/onnx-fp32-cuda

**Verdict:** FAIL: unloaded working_set grows 175 MB (109 MB/h); unloaded rss_anon grows 175 MB (109 MB/h); loaded working_set grows 170 MB (106 MB/h); loaded rss_anon grows 170 MB (106 MB/h)

## Run

| Item | Value |
| --- | --- |
| Model lock | intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3 |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=on, trim_native_heap_on_release=on, dedicated_run_threads=off, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 7200 s / 7201 s |
| Load/unload cycles | 3704 |
| Requests per cycle, concurrency | 200, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 666720; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1115, 1595 |
| Unload ms (median, max) | 25.7, 40.5 |
| Request p50 ms (median over cycles) | 2.7 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 9953 | 64127 | 9.21 | 18.26 | 39.57 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 2964 | 40.5 | 57.5 | 44.4 | 40.5 | 65.7 | 4.2 | 6.7 | 6.1 | 6.6 | 32.0 | plateau |
| unloaded | working_set | yes | 2964 | 974.1 | 1127.2 | 1097.5 | 959.9 | 1171.8 | 109.5 | 175.2 | 137.3 | 54.0 | 32.0 | growth |
| unloaded | private | no | 2964 | 1112.0 | 1239.4 | 1225.4 | 1070.0 | 1361.5 | 103.1 | 164.9 | 145.9 | 38.8 | 32.0 | growth |
| unloaded | rss_anon | yes | 2964 | 618.3 | 770.8 | 741.1 | 604.1 | 815.3 | 109.1 | 174.6 | 136.8 | 53.9 | 32.0 | growth |
| unloaded | gpu | yes | 2964 | 270.0 | 270.0 | 270.0 | 270.0 | 270.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 2964 | 52.0 | 69.0 | 55.8 | 50.0 | 77.2 | 4.2 | 6.7 | 6.1 | 6.6 | 32.0 | plateau |
| loaded | working_set | yes | 2964 | 1089.2 | 1242.6 | 1209.1 | 1067.8 | 1290.2 | 106.3 | 170.1 | 135.6 | 48.1 | 32.0 | growth |
| loaded | private | no | 2964 | 1222.1 | 1352.5 | 1331.1 | 1179.2 | 1467.7 | 101.5 | 162.4 | 143.2 | 32.5 | 32.0 | growth |
| loaded | rss_anon | yes | 2964 | 731.4 | 884.2 | 850.7 | 709.9 | 931.8 | 106.0 | 169.6 | 135.1 | 48.1 | 32.0 | growth |
| loaded | gpu | yes | 2964 | 1368.0 | 1400.0 | 1398.0 | 950.0 | 1464.0 | -3.5 | -5.6 | 3.2 | -31.4 | 32.0 | plateau |

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
| command_line | e5 --cuda --duration 2h --out <local>/out-e5-main-2h/e5 --note local RTX 3090, main at 3a021dc0b7f8, two hours, CUDA libraries from the pinned eg2 requirements |
| machine_note | local RTX 3090, main at 3a021dc0b7f8, two hours, CUDA libraries from the pinned eg2 requirements |
| load before (loadavg, MemAvailable MB) | [1.25,1.7,1.8], 76290 |
| load after (loadavg, MemAvailable MB) | [1.49,1.64,1.71], 73140 |
