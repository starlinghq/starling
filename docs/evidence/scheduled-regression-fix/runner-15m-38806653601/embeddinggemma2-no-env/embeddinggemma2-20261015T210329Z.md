# G4 soak: embeddinggemma-2/onnx-fp32-cpu

> **Short development run (900 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 900 s / 902 s |
| Load/unload cycles | 160 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 8640; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1128, 1914 |
| Unload ms (median, max) | 20.1, 49.0 |
| Request p50 ms (median over cycles) | 90.3 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 960 | 0 | 499.15 | 683.23 | 977.13 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 129 | 255.8 | 256.4 | 256.3 | 223.1 | 321.6 | -49.1 | -9.8 | 6.0 | -27.4 | 32.0 | plateau |
| unloaded | working_set | yes | 129 | 488.7 | 618.7 | 499.3 | 445.5 | 621.2 | 653.9 | 130.8 | -0.3 | 125.5 | 32.0 | plateau |
| unloaded | private | no | 129 | 1758.0 | 1861.7 | 1770.6 | 1687.7 | 1894.8 | 643.5 | 128.8 | 22.4 | 77.5 | 35.4 | retention |
| unloaded | rss_anon | yes | 129 | 415.4 | 544.3 | 425.4 | 371.5 | 546.8 | 648.2 | 129.7 | -1.2 | 125.3 | 32.0 | plateau |
| loaded | managed_heap | yes | 129 | 334.5 | 335.1 | 335.0 | 301.8 | 400.3 | -49.1 | -9.8 | 6.0 | -27.4 | 32.0 | plateau |
| loaded | working_set | yes | 129 | 2224.0 | 2591.7 | 2221.8 | 2144.7 | 2637.0 | 652.9 | 130.6 | -19.6 | 119.7 | 44.4 | plateau |
| loaded | private | no | 129 | 2838.1 | 3437.9 | 2852.6 | 2767.9 | 3454.2 | 770.2 | 154.1 | 9.7 | 96.5 | 57.1 | plateau |
| loaded | rss_anon | yes | 129 | 2150.7 | 2517.3 | 2147.4 | 2070.7 | 2562.5 | 647.2 | 129.5 | -20.5 | 119.4 | 42.9 | plateau |

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
| git_commit | ff58b212568b4d484c6f849b75b0094a2e597636 |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 15m --note GitHub-hosted ubuntu-latest, scheduled qualification run 38806653601, variant no-env |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38806653601, variant no-env |
| load before (loadavg, MemAvailable MB) | [1.82,0.65,0.24], 14316 |
| load after (loadavg, MemAvailable MB) | [6.41,5.38,3.54], 14230 |
