# G4 soak: multilingual-e5-small/onnx-fp32-cuda

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3 |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=on, trim_native_heap_on_release=on, dedicated_run_threads=off, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 7200 s / 7201 s |
| Load/unload cycles | 3524 |
| Requests per cycle, concurrency | 200, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 634320; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1209, 1778 |
| Unload ms (median, max) | 20.2, 34.2 |
| Request p50 ms (median over cycles) | 2.8 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 9923 | 60557 | 9.47 | 18.66 | 41.37 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 2823 | 48.6 | 50.0 | 44.4 | 40.5 | 65.5 | 2.3 | 3.7 | 2.1 | 2.0 | 32.0 | plateau |
| unloaded | working_set | yes | 2823 | 833.7 | 829.6 | 826.5 | 805.2 | 858.2 | 2.8 | 4.5 | 8.4 | 21.6 | 32.0 | plateau |
| unloaded | private | no | 2823 | 1002.3 | 1007.3 | 987.4 | 952.9 | 1042.3 | 7.2 | 11.5 | 10.7 | 22.5 | 32.0 | plateau |
| unloaded | rss_anon | yes | 2823 | 478.9 | 474.4 | 471.2 | 450.2 | 502.9 | 2.5 | 4.0 | 8.0 | 21.5 | 32.0 | plateau |
| unloaded | gpu | yes | 2823 | 270.0 | 270.0 | 270.0 | 270.0 | 270.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 2823 | 60.1 | 61.5 | 55.9 | 49.5 | 77.0 | 2.3 | 3.7 | 2.1 | 2.0 | 32.0 | plateau |
| loaded | working_set | yes | 2823 | 873.9 | 877.1 | 867.2 | 833.7 | 901.1 | 4.5 | 7.2 | 6.8 | 23.4 | 32.0 | plateau |
| loaded | private | no | 2823 | 1047.6 | 1059.2 | 1037.6 | 984.7 | 1090.9 | 9.0 | 14.4 | 8.5 | 24.1 | 32.0 | plateau |
| loaded | rss_anon | yes | 2823 | 517.2 | 519.8 | 510.1 | 476.5 | 543.8 | 4.2 | 6.7 | 6.4 | 23.3 | 32.0 | plateau |
| loaded | gpu | yes | 2823 | 1462.0 | 1368.0 | 1398.0 | 950.0 | 1464.0 | 13.6 | 21.7 | -41.5 | 28.9 | 32.0 | plateau |

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
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"1","DOTNET_NOLOGO":"1","MALLOC_MMAP_THRESHOLD_":"65536"} |
| git_commit | n/a |
| git_dirty | n/a |
| command_line | e5 --cuda --duration 2h --out <local>/out-e5-main-2h-mmap/e5 --note local RTX 3090, main, two hours, MALLOC_MMAP_THRESHOLD_=65536 diagnostic |
| machine_note | local RTX 3090, main, two hours, MALLOC_MMAP_THRESHOLD_=65536 diagnostic |
| load before (loadavg, MemAvailable MB) | [2.75,3.68,3.31], 66730 |
| load after (loadavg, MemAvailable MB) | [2.17,2.17,2.2], 59207 |
