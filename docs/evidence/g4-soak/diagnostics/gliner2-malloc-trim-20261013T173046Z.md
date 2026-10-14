# G4 soak: gliner2-base-v1/onnx-fp32-cpu

> **Short development run (300 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error (note: unloaded working_set retention 123 MB between loads, reused by the next load; unloaded rss_anon retention 121 MB between loads, reused by the next load; trimmed working_set retention 110 MB between loads, reused by the next load; trimmed rss_anon retention 109 MB between loads, reused by the next load)

## Run

| Item | Value |
| --- | --- |
| Model lock | fastino/gliner2-base-v1@f9634218e53580c56edf0de97ca1a7d3f1c2354e |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 300 s / 307 s |
| Load/unload cycles | 22 |
| Requests per cycle, concurrency | 100, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 1980; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2718, 4882 |
| Unload ms (median, max) | 27.6, 162.2 |
| Request p50 ms (median over cycles) | 243.2 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 220 | 0 | 4.13 | 25.12 | 61.02 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 19 | 12.2 | 12.2 | 12.2 | 12.2 | 12.2 | 0.2 | 0.0 | 0.0 | 0.0 | 32.0 | plateau |
| unloaded | working_set | yes | 19 | 1047.7 | 1128.1 | 1076.2 | 998.3 | 1169.6 | 1846.0 | 122.6 | 58.6 | 47.1 | 32.0 | retention |
| unloaded | private | no | 19 | 1187.4 | 1261.0 | 1236.1 | 1146.4 | 1330.4 | 1853.8 | 123.1 | 60.5 | 34.4 | 32.0 | retention |
| unloaded | rss_anon | yes | 19 | 970.7 | 1049.8 | 998.2 | 921.2 | 1091.3 | 1828.7 | 121.4 | 57.6 | 46.7 | 32.0 | retention |
| loaded | managed_heap | yes | 19 | 17.9 | 18.0 | 17.9 | 17.9 | 18.0 | 0.2 | 0.0 | 0.0 | 0.0 | 32.0 | plateau |
| loaded | working_set | yes | 19 | 1582.8 | 1765.6 | 1597.2 | 1469.2 | 1777.2 | 2982.3 | 198.0 | -6.0 | 116.6 | 32.0 | plateau |
| loaded | private | no | 19 | 1819.4 | 2187.9 | 1867.1 | 1694.0 | 2215.4 | 4276.9 | 284.0 | -57.0 | 326.2 | 37.3 | plateau |
| loaded | rss_anon | yes | 19 | 1505.9 | 1687.3 | 1519.3 | 1391.6 | 1698.9 | 2964.9 | 196.9 | -7.0 | 116.2 | 32.0 | plateau |
| trimmed | managed_heap | yes | 19 | 12.4 | 12.4 | 12.4 | 12.4 | 12.4 | 0.3 | 0.0 | 0.0 | 0.0 | 32.0 | plateau |
| trimmed | working_set | yes | 19 | 163.1 | 274.3 | 218.4 | 163.1 | 274.3 | 1661.4 | 110.3 | 65.6 | 68.0 | 32.0 | retention |
| trimmed | private | no | 19 | 1169.3 | 1260.9 | 1208.9 | 1146.2 | 1312.3 | 1838.1 | 122.1 | 46.0 | 50.1 | 32.0 | retention |
| trimmed | rss_anon | yes | 19 | 86.1 | 196.0 | 140.4 | 86.1 | 196.0 | 1644.0 | 109.2 | 64.6 | 67.6 | 32.0 | retention |

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
| command_line | gliner2 --duration 5m --out /tmp/claude-soak-diag --malloc-trim --note diagnostic: malloc_trim after each unload |
| machine_note | diagnostic: malloc_trim after each unload |
| load before (loadavg, MemAvailable MB) | [30.06,29.85,24.07], 16090 |
| load after (loadavg, MemAvailable MB) | [18.47,23.86,23.21], 27433 |
