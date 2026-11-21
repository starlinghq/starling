# G4 soak: kev-0.8b/onnx-fp32-cpu

**Verdict:** FAIL: unloaded working_set grows 70 MB (88 MB/h); unloaded rss_anon grows 68 MB (86 MB/h); loaded working_set grows 75 MB (94 MB/h); loaded rss_anon grows 73 MB (92 MB/h)

## Run

| Item | Value |
| --- | --- |
| Model lock | jaredpalmer/kev-0.8b@9a45d25eb2ab761841196625383fa1dff0e56c1e |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=on, trim_native_heap_on_release=on, dedicated_run_threads=off, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 3600 s / 3616 s |
| Load/unload cycles | 84 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 4536; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3600, 6472 |
| Unload ms (median, max) | 286.4, 727.9 |
| Request p50 ms (median over cycles) | 1273.1 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 504 | 0 | 4.17 | 24.62 | 169.76 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 76 | 96.1 | 96.2 | 96.3 | 96.1 | 97.2 | 0.9 | 0.7 | 0.2 | 0.5 | 32.0 | plateau |
| unloaded | working_set | yes | 76 | 336.1 | 407.5 | 389.1 | 336.1 | 436.6 | 87.8 | 70.0 | 34.8 | 23.4 | 32.0 | growth |
| unloaded | private | no | 76 | 462.7 | 606.2 | 542.6 | 462.6 | 623.1 | 118.1 | 94.1 | 44.5 | 58.9 | 32.0 | growth |
| unloaded | rss_anon | yes | 76 | 254.6 | 324.1 | 306.1 | 254.6 | 353.2 | 85.9 | 68.4 | 33.7 | 22.8 | 32.0 | growth |
| loaded | managed_heap | yes | 76 | 163.0 | 163.1 | 163.3 | 162.6 | 164.1 | 0.5 | 0.4 | 0.2 | 0.1 | 32.0 | plateau |
| loaded | working_set | yes | 76 | 2373.3 | 2473.7 | 2414.1 | 2357.2 | 2480.5 | 94.1 | 75.0 | 36.9 | 50.9 | 48.3 | growth |
| loaded | private | no | 76 | 3538.5 | 3680.9 | 3610.8 | 3538.5 | 3682.9 | 136.3 | 108.6 | 45.1 | 67.7 | 72.2 | growth |
| loaded | rss_anon | yes | 76 | 2271.1 | 2369.5 | 2310.3 | 2254.8 | 2376.3 | 92.1 | 73.3 | 35.8 | 50.3 | 46.2 | growth |

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
| genai_native_package | Microsoft.ML.OnnxRuntimeGenAI/0.17.1 |
| gpu_identity | 0, NVIDIA GeForce RTX 3090, 615.78.08, 24576 MiB; 1, NVIDIA GeForce RTX 4060 Ti, 615.78.08, 16380 MiB |
| cpu_governor | schedutil |
| allocator_and_runtime_env | {"DOTNET_MULTILEVEL_LOOKUP":"0"} |
| git_commit | e1f11dbd1da66809ef8e77e56ad44dc33fa0b3fd |
| git_dirty | false |
| command_line | kev-0.8b --out /home/mark/.hermes/cache/scratch/soak-noenv/runs --duration 1h --note issue #178 no-env: no MALLOC_ env, THP madvise, local workstation |
| machine_note | issue #178 no-env: no MALLOC_ env, THP madvise, local workstation |
| load before (loadavg, MemAvailable MB) | [10.4,10.07,6.17], 59944 |
| load after (loadavg, MemAvailable MB) | [13.33,13.52,13.83], 50226 |
