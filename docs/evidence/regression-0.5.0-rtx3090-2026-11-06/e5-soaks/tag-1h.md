# G4 soak: multilingual-e5-small/onnx-fp32-cuda

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3 |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=on, trim_native_heap_on_release=on, dedicated_run_threads=off, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 3600 s / 3601 s |
| Load/unload cycles | 1844 |
| Requests per cycle, concurrency | 200, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 331920; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1118, 1628 |
| Unload ms (median, max) | 27.1, 41.2 |
| Request p50 ms (median over cycles) | 2.8 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 5044 | 31836 | 9.21 | 18.25 | 41.53 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 1475 | 40.3 | 48.6 | 42.0 | 40.3 | 58.3 | 2.0 | 1.6 | 1.5 | 2.0 | 32.0 | plateau |
| unloaded | working_set | yes | 1475 | 916.4 | 1031.9 | 1024.5 | 908.7 | 1051.2 | 144.9 | 115.9 | 124.7 | -2.0 | 32.0 | plateau |
| unloaded | private | no | 1475 | 1076.2 | 1206.8 | 1191.2 | 1045.0 | 1277.9 | 183.4 | 146.7 | 142.3 | 32.9 | 32.0 | retention |
| unloaded | rss_anon | yes | 1475 | 560.9 | 676.0 | 668.6 | 553.2 | 695.3 | 144.3 | 115.4 | 124.3 | -2.0 | 32.0 | plateau |
| unloaded | gpu | yes | 1475 | 270.0 | 270.0 | 270.0 | 270.0 | 270.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 1475 | 51.8 | 60.0 | 53.4 | 49.6 | 69.8 | 2.0 | 1.6 | 1.5 | 2.0 | 32.0 | plateau |
| loaded | working_set | yes | 1475 | 1033.2 | 1144.0 | 1137.5 | 1020.7 | 1178.0 | 155.6 | 124.5 | 126.4 | 0.7 | 32.0 | plateau |
| loaded | private | no | 1475 | 1186.5 | 1279.4 | 1290.9 | 1153.0 | 1341.1 | 148.5 | 118.8 | 139.1 | 1.3 | 32.0 | plateau |
| loaded | rss_anon | yes | 1475 | 675.7 | 786.1 | 779.6 | 663.2 | 820.0 | 155.0 | 124.0 | 126.0 | 0.6 | 32.0 | plateau |
| loaded | gpu | yes | 1475 | 1400.0 | 1462.0 | 1398.0 | 950.0 | 1464.0 | 33.6 | 26.9 | 33.5 | 33.5 | 32.0 | plateau |

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
| command_line | e5 --cuda --duration 1h --out <local>/out-e5-tag-1h/e5 --note local RTX 3090, tag ai-v0.5.0, CUDA libraries from the pinned eg2 requirements |
| machine_note | local RTX 3090, tag ai-v0.5.0, CUDA libraries from the pinned eg2 requirements |
| load before (loadavg, MemAvailable MB) | [1.39,1.65,1.73], 75889 |
| load after (loadavg, MemAvailable MB) | [1.96,1.86,1.85], 74446 |
