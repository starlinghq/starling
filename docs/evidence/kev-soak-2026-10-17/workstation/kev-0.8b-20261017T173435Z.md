# G4 soak: kev-0.8b/onnx-fp32-cuda

**Verdict:** FAIL: loaded rss_anon grows 51 MB (64 MB/h)

## Run

| Item | Value |
| --- | --- |
| Model lock | jaredpalmer/kev-0.8b@9a45d25eb2ab761841196625383fa1dff0e56c1e |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=on, trim_native_heap_on_release=on, dedicated_run_threads=off, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 3600 s / 3605 s |
| Load/unload cycles | 477 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 25758; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1183, 2138 |
| Unload ms (median, max) | 43.2, 93.3 |
| Request p50 ms (median over cycles) | 175.2 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 2862 | 0 | 7.65 | 108.42 | 271.98 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 403 | 96.6 | 96.4 | 96.3 | 96.1 | 98.4 | 1.0 | 0.8 | 0.3 | 0.3 | 32.0 | plateau |
| unloaded | working_set | yes | 403 | 1079.2 | 1146.0 | 1087.8 | 1051.4 | 1146.7 | 32.4 | 25.9 | -14.8 | 48.6 | 32.0 | plateau |
| unloaded | private | no | 403 | 1264.4 | 1394.0 | 1296.2 | 1264.4 | 1394.0 | 86.9 | 69.5 | 11.3 | 83.9 | 32.0 | growth |
| unloaded | rss_anon | yes | 403 | 749.1 | 832.2 | 770.1 | 737.9 | 832.8 | 49.6 | 39.7 | 5.5 | 48.6 | 32.0 | plateau |
| unloaded | gpu | yes | 403 | 268.0 | 268.0 | 268.0 | 268.0 | 268.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 403 | 163.5 | 163.3 | 163.2 | 162.5 | 165.3 | 0.7 | 0.5 | 0.1 | 0.2 | 32.0 | plateau |
| loaded | working_set | yes | 403 | 1246.9 | 1290.9 | 1260.1 | 1203.3 | 1322.7 | 48.4 | 38.7 | 3.4 | 41.2 | 32.0 | plateau |
| loaded | private | no | 403 | 1427.3 | 1557.5 | 1483.2 | 1402.0 | 1584.9 | 90.9 | 72.7 | 18.7 | 79.5 | 32.0 | growth |
| loaded | rss_anon | yes | 403 | 908.8 | 969.0 | 936.5 | 881.2 | 1000.9 | 63.8 | 51.0 | 22.6 | 40.9 | 32.0 | growth |
| loaded | gpu | yes | 403 | 4398.0 | 3886.0 | 4398.0 | 3374.0 | 4432.0 | 72.1 | 57.6 | 214.0 | -147.3 | 88.0 | plateau |

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
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"1","DOTNET_ROOT_X64":"/home/mark/.dotnet"} |
| git_commit | d0438f32a167ae18ce952f02377952d3b3eda8cc |
| git_dirty | false |
| command_line | kev-0.8b --duration 1h --cuda --out /tmp/kev-soak/soak/1h/kev-0.8b-cuda --note workstation i7-5820K; RTX 3090 (PCI 0); local Kev development soak |
| machine_note | workstation i7-5820K; RTX 3090 (PCI 0); local Kev development soak |
| load before (loadavg, MemAvailable MB) | [22.09,24.66,18.73], 30789 |
| load after (loadavg, MemAvailable MB) | [10.47,10.44,14.49], 30487 |
