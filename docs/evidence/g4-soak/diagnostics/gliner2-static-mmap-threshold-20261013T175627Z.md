# G4 soak: gliner2-base-v1/onnx-fp32-cpu

> **Short development run (300 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** FAIL: loaded working_set grows 165 MB (2423 MB/h); loaded rss_anon grows 163 MB (2402 MB/h)

## Run

| Item | Value |
| --- | --- |
| Model lock | fastino/gliner2-base-v1@f9634218e53580c56edf0de97ca1a7d3f1c2354e |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 300 s / 309 s |
| Load/unload cycles | 21 |
| Requests per cycle, concurrency | 100, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 1890; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 4690, 5924 |
| Unload ms (median, max) | 195.9, 300.4 |
| Request p50 ms (median over cycles) | 232.0 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 210 | 0 | 3.29 | 27.32 | 47.05 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 19 | 12.2 | 12.2 | 12.2 | 12.2 | 12.2 | 0.2 | 0.0 | 0.0 | 0.0 | 32.0 | plateau |
| unloaded | working_set | yes | 19 | 149.4 | 151.5 | 152.0 | 146.3 | 156.1 | 38.1 | 2.6 | 4.0 | -1.5 | 32.0 | plateau |
| unloaded | private | no | 19 | 232.5 | 234.8 | 235.5 | 226.5 | 258.6 | 18.9 | 1.3 | 4.4 | -10.9 | 32.0 | plateau |
| unloaded | rss_anon | yes | 19 | 72.3 | 72.9 | 73.8 | 69.1 | 77.8 | 17.4 | 1.2 | 2.8 | -1.9 | 32.0 | plateau |
| loaded | managed_heap | yes | 19 | 17.9 | 17.9 | 17.9 | 17.9 | 17.9 | 0.2 | 0.0 | 0.0 | 0.0 | 32.0 | plateau |
| loaded | working_set | yes | 19 | 996.6 | 1108.8 | 1106.2 | 996.6 | 1243.3 | 2423.2 | 164.5 | 49.3 | 51.2 | 32.0 | growth |
| loaded | private | no | 19 | 1118.2 | 1353.5 | 1352.8 | 1118.2 | 1615.5 | 3873.9 | 263.0 | 86.3 | 89.3 | 32.0 | growth |
| loaded | rss_anon | yes | 19 | 919.5 | 1030.2 | 1027.7 | 919.5 | 1164.7 | 2402.0 | 163.1 | 48.1 | 50.8 | 32.0 | growth |

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
| allocator_and_runtime_env | n/a |
| git_commit | fb276585f2cd5fc6a848d902280c0002a49c2fc6 |
| git_dirty | true |
| command_line | gliner2 --duration 5m --out /tmp/claude-soak-diag/mmap64k |
| machine_note | n/a |
| load before (loadavg, MemAvailable MB) | [19.32,20.48,20.88], 22080 |
| load after (loadavg, MemAvailable MB) | [17.81,20.26,20.86], 25810 |
