# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=off, trim_native_heap_on_release=on |
| Duration requested / elapsed | 3600 s / 3606 s |
| Load/unload cycles | 586 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 31644; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1725, 2169 |
| Unload ms (median, max) | 100.0, 141.6 |
| Request p50 ms (median over cycles) | 82.3 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 3516 | 0 | 505.63 | 687.22 | 1020.74 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 469 | 223.8 | 258.1 | 256.4 | 223.2 | 321.2 | 0.4 | 0.3 | 6.6 | -9.4 | 32.0 | plateau |
| unloaded | working_set | yes | 469 | 453.9 | 474.1 | 470.9 | 447.3 | 483.0 | 17.3 | 13.8 | 11.8 | 9.6 | 32.0 | plateau |
| unloaded | private | no | 469 | 1637.2 | 1655.3 | 1658.2 | 1604.1 | 1728.2 | 12.6 | 10.1 | 16.5 | -1.0 | 33.2 | plateau |
| unloaded | rss_anon | yes | 469 | 379.2 | 398.8 | 395.7 | 372.5 | 407.7 | 16.6 | 13.3 | 11.3 | 9.6 | 32.0 | plateau |
| loaded | managed_heap | yes | 469 | 302.5 | 336.8 | 335.4 | 302.0 | 400.0 | 0.8 | 0.6 | 6.6 | -9.1 | 32.0 | plateau |
| loaded | working_set | yes | 469 | 1869.4 | 1891.4 | 1884.4 | 1853.6 | 1906.7 | 15.5 | 12.4 | 11.3 | 8.6 | 37.7 | plateau |
| loaded | private | no | 469 | 2205.3 | 2230.4 | 2224.5 | 2160.3 | 2324.9 | 10.2 | 8.2 | 19.4 | -3.8 | 44.5 | plateau |
| loaded | rss_anon | yes | 469 | 1794.7 | 1816.1 | 1809.2 | 1778.9 | 1831.4 | 14.8 | 11.8 | 10.8 | 8.6 | 36.2 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | AMD EPYC 9V74 80-Core Processor |
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
| git_commit | 6df29602d9f84e48e7ce5a1a6c8b8f96c4fbf50c |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38874143326 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38874143326 |
| load before (loadavg, MemAvailable MB) | [2.54,0.69,0.25], 14222 |
| load after (loadavg, MemAvailable MB) | [5.48,5.24,5.05], 14106 |
