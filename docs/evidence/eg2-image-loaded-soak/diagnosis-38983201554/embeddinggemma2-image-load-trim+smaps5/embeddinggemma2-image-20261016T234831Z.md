# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on |
| Duration requested / elapsed | 3600 s / 3622 s |
| Load/unload cycles | 113 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 452; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2905, 3451 |
| Unload ms (median, max) | 90.3, 97.3 |
| Request p50 ms (median over cycles) | 13876.5 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 113 | 0 | 12.86 | 125.98 | 160.79 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 94 | 181.3 | 180.2 | 181.6 | 180.0 | 183.2 | 0.5 | 0.4 | 0.4 | -0.0 | 32.0 | plateau |
| unloaded | working_set | yes | 94 | 398.3 | 442.4 | 438.1 | 381.5 | 500.4 | 25.5 | 20.5 | 20.0 | 1.1 | 32.0 | plateau |
| unloaded | private | no | 94 | 2474.4 | 2531.1 | 2507.9 | 2432.4 | 2616.9 | 26.0 | 20.9 | 31.3 | 17.5 | 50.2 | plateau |
| unloaded | rss_anon | yes | 94 | 323.6 | 366.0 | 361.8 | 305.1 | 423.9 | 23.7 | 19.1 | 18.8 | 1.1 | 32.0 | plateau |
| loaded | managed_heap | yes | 94 | 260.0 | 258.9 | 260.3 | 258.8 | 261.9 | 0.5 | 0.4 | 0.5 | -0.1 | 32.0 | plateau |
| loaded | working_set | yes | 94 | 2322.9 | 2365.1 | 2365.9 | 2309.0 | 2471.4 | 31.1 | 25.0 | 34.2 | 4.6 | 47.3 | plateau |
| loaded | private | no | 94 | 2986.4 | 3043.2 | 3037.9 | 2960.4 | 3153.0 | 23.1 | 18.5 | 28.2 | 15.3 | 60.8 | plateau |
| loaded | rss_anon | yes | 94 | 2248.0 | 2288.6 | 2290.1 | 2233.2 | 2395.3 | 29.2 | 23.4 | 32.7 | 4.5 | 45.8 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz |
| logical_cpus | 4 |
| ram_total_mb | 15993 |
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
| git_commit | a5a0f374012118a3ba0a764018448dac111797a9 |
| git_dirty | false |
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38983201554, variant load-trim\u002Bsmaps5 --trim-after-load --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38983201554, variant load-trim\u002Bsmaps5 |
| load before (loadavg, MemAvailable MB) | [3.8,1.16,0.41], 14079 |
| load after (loadavg, MemAvailable MB) | [3.14,3.06,3.01], 14109 |
