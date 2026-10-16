# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on |
| Duration requested / elapsed | 3600 s / 3601 s |
| Load/unload cycles | 700 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 37800; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 1866, 2303 |
| Unload ms (median, max) | 60.0, 75.1 |
| Request p50 ms (median over cycles) | 62.9 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 4200 | 0 | 388.27 | 464.94 | 544.73 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 560 | 255.8 | 260.2 | 258.1 | 223.2 | 292.8 | 4.1 | 3.2 | 3.6 | -12.0 | 32.0 | plateau |
| unloaded | working_set | yes | 560 | 394.2 | 415.4 | 407.3 | 388.4 | 420.5 | 3.2 | 2.6 | 4.2 | 8.5 | 32.0 | plateau |
| unloaded | private | no | 560 | 1595.9 | 1613.4 | 1632.5 | 1553.2 | 1658.8 | 5.7 | 4.6 | 4.7 | -2.2 | 32.7 | plateau |
| unloaded | rss_anon | yes | 560 | 319.2 | 339.9 | 332.0 | 313.1 | 344.9 | 2.5 | 2.0 | 3.7 | 8.5 | 32.0 | plateau |
| loaded | managed_heap | yes | 560 | 334.6 | 338.9 | 336.8 | 301.9 | 371.5 | 4.1 | 3.2 | 3.6 | -12.0 | 32.0 | plateau |
| loaded | working_set | yes | 560 | 1796.9 | 1828.8 | 1819.1 | 1792.9 | 1841.5 | 5.7 | 4.6 | 4.4 | 8.1 | 36.4 | plateau |
| loaded | private | no | 560 | 2140.0 | 2153.5 | 2171.4 | 2097.1 | 2198.9 | 8.6 | 6.9 | 5.5 | -4.8 | 43.4 | plateau |
| loaded | rss_anon | yes | 560 | 1721.9 | 1753.2 | 1743.7 | 1717.9 | 1765.9 | 5.0 | 4.0 | 3.9 | 8.1 | 34.9 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | INTEL(R) XEON(R) PLATINUM 8573C |
| logical_cpus | 4 |
| ram_total_mb | 15989 |
| os | Ubuntu 24.04.5 LTS |
| kernel | 6.17.0-1022-azure |
| runtime | .NET 10.0.13 |
| server_gc | false |
| ort_version | 1.30.0 |
| genai_version | 0.17.1 |
| genai_native_package | Microsoft.ML.OnnxRuntimeGenAI/0.17.1 |
| gpu_identity | n/a |
| cpu_governor | performance |
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"true","DOTNET_MULTILEVEL_LOOKUP":"0","DOTNET_NOLOGO":"true","DOTNET_ROOT_X64":"/usr/share/dotnet","DOTNET_SKIP_FIRST_TIME_EXPERIENCE":"1"} |
| git_commit | 47f1a52e1249d5ba7b2a50c1a91c12130c91e55f |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38913307558, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38913307558, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [2.12,0.71,0.25], 13947 |
| load after (loadavg, MemAvailable MB) | [2.57,2.4,2.33], 13948 |
