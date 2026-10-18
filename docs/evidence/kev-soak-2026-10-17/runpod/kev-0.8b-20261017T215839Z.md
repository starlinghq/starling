# G4 soak: kev-0.8b/onnx-fp32-cuda

> **Short development run (600 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | jaredpalmer/kev-0.8b@9a45d25eb2ab761841196625383fa1dff0e56c1e |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=on, trim_native_heap_on_release=on, dedicated_run_threads=off, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 600 s / 605 s |
| Load/unload cycles | 122 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 6588; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1020, 5602 |
| Unload ms (median, max) | 71.6, 798.7 |
| Request p50 ms (median over cycles) | 104.4 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 732 | 0 | 5.27 | 50.83 | 332.81 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 100 | 96.1 | 96.1 | 96.2 | 96.1 | 96.7 | 2.8 | 0.4 | 0.1 | 0.3 | 32.0 | plateau |
| unloaded | working_set | yes | 100 | 813.5 | 808.7 | 815.5 | 797.8 | 819.5 | 27.3 | 3.6 | -10.7 | -1.1 | 32.0 | plateau |
| unloaded | private | no | 100 | 1035.9 | 1042.0 | 1046.8 | 1028.2 | 1079.6 | 42.0 | 5.6 | -7.8 | 0.4 | 32.0 | plateau |
| unloaded | rss_anon | yes | 100 | 523.0 | 516.8 | 524.1 | 506.1 | 528.2 | 19.8 | 2.6 | -11.7 | -1.2 | 32.0 | plateau |
| loaded | managed_heap | yes | 100 | 163.0 | 163.0 | 163.0 | 159.4 | 163.6 | -1.1 | -0.1 | 0.3 | -0.0 | 32.0 | plateau |
| loaded | working_set | yes | 100 | 993.0 | 996.7 | 995.3 | 959.6 | 1006.5 | 32.6 | 4.4 | -0.4 | -4.1 | 32.0 | plateau |
| loaded | private | no | 100 | 1498.0 | 1504.8 | 1483.9 | 1415.9 | 1513.7 | 50.9 | 6.8 | -4.8 | -3.6 | 32.0 | plateau |
| loaded | rss_anon | yes | 100 | 694.5 | 696.9 | 695.6 | 660.4 | 706.8 | 24.8 | 3.3 | -1.4 | -4.2 | 32.0 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | AMD EPYC 7H12 64-Core Processor |
| logical_cpus | 32 |
| ram_total_mb | 1031827 |
| os | Ubuntu 24.04.5 LTS |
| kernel | 6.8.0-110-generic |
| runtime | .NET 10.0.0 |
| server_gc | false |
| ort_version | 1.30.0 |
| genai_version | 0.17.1 |
| genai_native_package | Microsoft.ML.OnnxRuntimeGenAI.Cuda/0.17.1 |
| gpu_identity | 0, NVIDIA GeForce RTX 3090, 580.159.03, 24576 MiB |
| cpu_governor | schedutil |
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"1","DOTNET_NOLOGO":"1","MALLOC_MMAP_THRESHOLD_":"65536"} |
| git_commit | 855ec53b740183926be5ee539c49481c0e3f7d66 |
| git_dirty | false |
| command_line | kev-0.8b --cuda --duration 10m --out /root/out/soak/kev-0.8b --note RunPod EU-CZ-1 RTX 3090 agent pod; MALLOC_MMAP_THRESHOLD_=65536 |
| machine_note | RunPod EU-CZ-1 RTX 3090 agent pod; MALLOC_MMAP_THRESHOLD_=65536 |
| load before (loadavg, MemAvailable MB) | [75.5,62.31,59.09], 790723 |
| load after (loadavg, MemAvailable MB) | [76.7,79.53,70.46], 789315 |
