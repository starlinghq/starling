# G4 soak: embeddinggemma-2/onnx-fp32-cpu

> **Short development run (900 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 900 s / 904 s |
| Load/unload cycles | 195 |
| Keep loaded (diagnostic) | yes: one load, no unloads; the `unloaded` series is a second sample with the model loaded |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 10530; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 0, 2940 |
| Unload ms (median, max) | 0.0, 0.0 |
| Request p50 ms (median over cycles) | 91.4 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 1170 | 0 | 508.49 | 707.80 | 995.69 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 158 | 177.0 | 178.1 | 177.6 | 144.5 | 210.7 | 14.5 | 2.9 | 13.5 | -10.8 | 32.0 | plateau |
| unloaded | working_set | yes | 158 | 2189.5 | 2200.1 | 2198.8 | 2189.5 | 2200.8 | 26.7 | 5.4 | 5.1 | -1.2 | 44.0 | plateau |
| unloaded | private | no | 158 | 2958.1 | 2967.0 | 2994.9 | 2928.0 | 3000.5 | 35.5 | 7.1 | 21.2 | -12.4 | 59.9 | plateau |
| unloaded | rss_anon | yes | 158 | 2116.6 | 2126.4 | 2125.5 | 2116.6 | 2127.4 | 22.5 | 4.5 | 4.6 | -1.5 | 42.5 | plateau |
| loaded | managed_heap | yes | 158 | 177.0 | 178.1 | 177.6 | 144.5 | 210.7 | 14.5 | 2.9 | 13.5 | -10.8 | 32.0 | plateau |
| loaded | working_set | yes | 158 | 2189.5 | 2200.1 | 2198.8 | 2189.5 | 2200.8 | 26.7 | 5.4 | 5.1 | -1.2 | 44.0 | plateau |
| loaded | private | no | 158 | 2958.1 | 2967.0 | 2994.9 | 2928.0 | 3000.5 | 35.5 | 7.1 | 21.2 | -12.4 | 59.9 | plateau |
| loaded | rss_anon | yes | 158 | 2116.6 | 2126.4 | 2125.5 | 2116.6 | 2127.4 | 22.5 | 4.5 | 4.6 | -1.5 | 42.5 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | AMD EPYC 7763 64-Core Processor |
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
| git_commit | 9a4d2ce14573a1e1000bedb5de865a622db469de |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 15m --note GitHub-hosted ubuntu-latest, scheduled qualification run 38861019342, variant keep-loaded\u002Bsmaps5 --keep-loaded --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38861019342, variant keep-loaded\u002Bsmaps5 |
| load before (loadavg, MemAvailable MB) | [1.85,0.58,0.2], 14136 |
| load after (loadavg, MemAvailable MB) | [6.47,6.32,4.31], 14796 |
