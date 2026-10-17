# G4 soak: kev-0.8b/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | jaredpalmer/kev-0.8b@9a45d25eb2ab761841196625383fa1dff0e56c1e |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=on, trim_native_heap_on_release=on, dedicated_run_threads=off, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 3600 s / 3606 s |
| Load/unload cycles | 133 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 7182; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1781, 3099 |
| Unload ms (median, max) | 45.6, 183.6 |
| Request p50 ms (median over cycles) | 899.7 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 798 | 0 | 3.55 | 10.14 | 41.45 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 109 | 96.3 | 96.2 | 96.2 | 96.1 | 97.2 | 0.7 | 0.6 | 0.2 | -0.2 | 32.0 | plateau |
| unloaded | working_set | yes | 109 | 276.4 | 259.9 | 277.3 | 250.8 | 284.7 | -10.6 | -8.4 | 0.6 | -4.3 | 32.0 | plateau |
| unloaded | private | no | 109 | 361.2 | 345.6 | 364.1 | 336.6 | 387.9 | -9.9 | -7.9 | 1.9 | -11.0 | 32.0 | plateau |
| unloaded | rss_anon | yes | 109 | 195.5 | 178.1 | 196.0 | 169.0 | 202.9 | -11.6 | -9.3 | 0.1 | -4.5 | 32.0 | plateau |
| loaded | managed_heap | yes | 109 | 163.2 | 163.1 | 163.1 | 163.0 | 164.1 | 0.4 | 0.3 | 0.2 | -0.5 | 32.0 | plateau |
| loaded | working_set | yes | 109 | 2309.7 | 2323.3 | 2313.6 | 2303.9 | 2349.8 | 10.8 | 8.7 | 7.2 | -0.2 | 46.3 | plateau |
| loaded | private | no | 109 | 3490.5 | 3485.8 | 3484.7 | 3481.1 | 3499.6 | 1.9 | 1.5 | -0.8 | 0.6 | 69.7 | plateau |
| loaded | rss_anon | yes | 109 | 2213.4 | 2226.1 | 2216.6 | 2207.2 | 2252.6 | 9.9 | 7.9 | 6.8 | -0.3 | 44.3 | plateau |

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
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"1","DOTNET_ROOT_X64":"/home/mark/.dotnet","MALLOC_MMAP_THRESHOLD_":"65536"} |
| git_commit | d0438f32a167ae18ce952f02377952d3b3eda8cc |
| git_dirty | false |
| command_line | kev-0.8b --duration 1h --out /tmp/kev-soak/soak/1h-mmap/kev-0.8b-cpu --note workstation i7-5820K; RTX 3090 (PCI 0); local Kev development soak |
| machine_note | workstation i7-5820K; RTX 3090 (PCI 0); local Kev development soak |
| load before (loadavg, MemAvailable MB) | [6.82,7.68,8.42], 29497 |
| load after (loadavg, MemAvailable MB) | [9.23,10.01,9.92], 31728 |
