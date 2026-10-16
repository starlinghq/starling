# G4 soak: embeddinggemma-2/onnx-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on |
| Duration requested / elapsed | 3600 s / 3604 s |
| Load/unload cycles | 488 |
| Requests per cycle, concurrency | 60, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 26352; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2196, 3062 |
| Unload ms (median, max) | 112.2, 126.5 |
| Request p50 ms (median over cycles) | 95.9 |

## Timed-out requests and cancellation latency

Every 10th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 2928 | 0 | 617.43 | 837.20 | 1180.14 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 390 | 223.7 | 256.1 | 256.1 | 223.2 | 322.2 | 6.6 | 5.2 | -2.2 | 2.0 | 32.0 | plateau |
| unloaded | working_set | yes | 390 | 394.2 | 406.5 | 403.3 | 386.9 | 436.4 | -13.4 | -10.7 | -9.7 | -0.7 | 32.0 | plateau |
| unloaded | private | no | 390 | 1575.2 | 1621.9 | 1621.2 | 1572.5 | 1723.4 | -6.6 | -5.3 | -11.5 | 0.9 | 32.4 | plateau |
| unloaded | rss_anon | yes | 390 | 319.5 | 331.2 | 328.2 | 311.6 | 361.4 | -14.1 | -11.3 | -10.2 | -0.7 | 32.0 | plateau |
| loaded | managed_heap | yes | 390 | 302.4 | 334.9 | 334.8 | 302.0 | 400.9 | 5.3 | 4.2 | -3.4 | 2.1 | 32.0 | plateau |
| loaded | working_set | yes | 390 | 1809.4 | 1810.2 | 1814.7 | 1786.4 | 1852.0 | -9.9 | -7.9 | -8.0 | -3.1 | 36.3 | plateau |
| loaded | private | no | 390 | 2127.4 | 2174.0 | 2178.1 | 2126.6 | 2295.4 | -9.4 | -7.5 | -13.9 | -0.1 | 43.6 | plateau |
| loaded | rss_anon | yes | 390 | 1734.7 | 1734.9 | 1739.6 | 1711.5 | 1777.0 | -10.6 | -8.5 | -8.5 | -3.1 | 34.8 | plateau |

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
| git_commit | 47f1a52e1249d5ba7b2a50c1a91c12130c91e55f |
| git_dirty | false |
| command_line | embeddinggemma2 --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38924489084, variant smaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38924489084, variant smaps5 |
| load before (loadavg, MemAvailable MB) | [2.17,0.6,0.21], 14201 |
| load after (loadavg, MemAvailable MB) | [4.7,4.93,5.01], 14325 |
