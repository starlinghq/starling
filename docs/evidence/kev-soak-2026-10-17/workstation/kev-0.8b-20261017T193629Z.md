# G4 soak: kev-0.8b/onnx-fp32-cuda

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | jaredpalmer/kev-0.8b@9a45d25eb2ab761841196625383fa1dff0e56c1e |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=on, trim_native_heap_on_release=on, dedicated_run_threads=off, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 3600 s / 3605 s |
| Load/unload cycles | 691 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 37314; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 929, 2281 |
| Unload ms (median, max) | 33.9, 68.7 |
| Request p50 ms (median over cycles) | 133.6 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 4146 | 0 | 3.66 | 51.26 | 173.90 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 559 | 96.1 | 96.5 | 96.4 | 96.1 | 98.5 | 0.9 | 0.8 | 0.6 | 0.1 | 32.0 | plateau |
| unloaded | working_set | yes | 559 | 968.8 | 975.8 | 971.4 | 945.3 | 982.9 | -3.2 | -2.6 | -7.5 | -4.3 | 32.0 | plateau |
| unloaded | private | no | 559 | 1083.8 | 1083.8 | 1080.6 | 1052.6 | 1105.4 | -2.5 | -2.0 | -6.4 | -2.6 | 32.0 | plateau |
| unloaded | rss_anon | yes | 559 | 638.6 | 645.3 | 640.9 | 614.8 | 652.4 | -3.6 | -2.9 | -7.8 | -4.3 | 32.0 | plateau |
| unloaded | gpu | yes | 559 | 268.0 | 268.0 | 268.0 | 268.0 | 268.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 559 | 163.0 | 163.4 | 163.3 | 161.8 | 165.4 | 0.6 | 0.5 | 0.4 | -0.0 | 32.0 | plateau |
| loaded | working_set | yes | 559 | 1151.1 | 1129.3 | 1150.3 | 1101.8 | 1170.1 | 5.8 | 4.7 | 3.9 | 6.0 | 32.0 | plateau |
| loaded | private | no | 559 | 1309.2 | 1303.8 | 1303.3 | 1239.1 | 1342.9 | 7.6 | 6.1 | 5.7 | 3.8 | 32.0 | plateau |
| loaded | rss_anon | yes | 559 | 812.8 | 790.8 | 811.8 | 769.4 | 831.8 | 5.5 | 4.4 | 3.8 | 5.9 | 32.0 | plateau |
| loaded | gpu | yes | 559 | 4406.0 | 3638.0 | 4398.0 | 3374.0 | 4430.0 | 85.4 | 68.3 | 113.0 | 205.6 | 88.0 | plateau |

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
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"1","DOTNET_ROOT_X64":"/home/mark/.dotnet","MALLOC_MMAP_THRESHOLD_":"65536"} |
| git_commit | d0438f32a167ae18ce952f02377952d3b3eda8cc |
| git_dirty | false |
| command_line | kev-0.8b --duration 1h --cuda --out /tmp/kev-soak/soak/1h-mmap/kev-0.8b-cuda --note workstation i7-5820K; RTX 3090 (PCI 0); local Kev development soak |
| machine_note | workstation i7-5820K; RTX 3090 (PCI 0); local Kev development soak |
| load before (loadavg, MemAvailable MB) | [14.67,13.28,12.48], 33460 |
| load after (loadavg, MemAvailable MB) | [4.6,7.35,8.33], 32623 |
