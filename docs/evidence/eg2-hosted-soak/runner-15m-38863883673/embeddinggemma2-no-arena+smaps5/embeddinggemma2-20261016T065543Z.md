# G4 soak: embeddinggemma-2/onnx-fp32-cpu

> **Short development run (900 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 900 s / 902 s |
| Load/unload cycles | 226 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 12204; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1305, 1678 |
| Unload ms (median, max) | 106.2, 152.3 |
| Request p50 ms (median over cycles) | 46.9 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 1356 | 0 | 304.74 | 384.19 | 450.44 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 182 | 255.8 | 255.9 | 288.5 | 223.4 | 289.5 | 95.2 | 19.1 | 14.5 | -6.2 | 32.0 | plateau |
| unloaded | working_set | yes | 182 | 408.5 | 463.2 | 441.1 | 408.5 | 483.7 | 301.7 | 60.4 | 14.8 | -1.1 | 32.0 | plateau |
| unloaded | private | no | 182 | 1547.5 | 1680.8 | 1652.8 | 1518.2 | 1696.8 | 641.0 | 128.3 | 58.1 | -3.8 | 33.1 | plateau |
| unloaded | rss_anon | yes | 182 | 334.7 | 388.3 | 366.6 | 334.7 | 408.8 | 296.9 | 59.4 | 14.1 | -1.4 | 32.0 | plateau |
| loaded | managed_heap | yes | 182 | 334.5 | 334.7 | 367.2 | 302.1 | 368.2 | 95.2 | 19.1 | 14.5 | -6.2 | 32.0 | plateau |
| loaded | working_set | yes | 182 | 1821.3 | 1880.1 | 1852.0 | 1810.3 | 1901.7 | 304.6 | 61.0 | 17.7 | -4.1 | 37.0 | plateau |
| loaded | private | no | 182 | 2126.3 | 2216.9 | 2188.9 | 2100.9 | 2236.5 | 453.5 | 90.8 | 41.2 | -4.2 | 43.8 | plateau |
| loaded | rss_anon | yes | 182 | 1747.5 | 1805.2 | 1777.6 | 1736.6 | 1826.8 | 299.9 | 60.0 | 17.0 | -4.4 | 35.6 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | AMD EPYC 9V45 96-Core Processor |
| logical_cpus | 4 |
| ram_total_mb | 15990 |
| os | Ubuntu 24.04.5 LTS |
| kernel | 6.17.0-1022-azure |
| runtime | .NET 10.0.13 |
| server_gc | false |
| ort_version | 1.30.0 |
| genai_version | 0.17.1 |
| genai_native_package | Microsoft.ML.OnnxRuntimeGenAI/0.17.1 |
| gpu_identity | n/a |
| cpu_governor | n/a |
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"true","DOTNET_MULTILEVEL_LOOKUP":"0","DOTNET_NOLOGO":"true","DOTNET_ROOT_X64":"/usr/share/dotnet","DOTNET_SKIP_FIRST_TIME_EXPERIENCE":"1"} |
| git_commit | 8acc5069fce1a0c89865ce3b766fd3bfa71f1521 |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 15m --note GitHub-hosted ubuntu-latest, scheduled qualification run 38863883673, variant no-arena\u002Bsmaps5 --no-cpu-arena --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38863883673, variant no-arena\u002Bsmaps5 |
| load before (loadavg, MemAvailable MB) | [1.97,0.65,0.23], 14219 |
| load after (loadavg, MemAvailable MB) | [2.14,2.13,1.5], 13546 |
