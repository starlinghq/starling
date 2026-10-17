# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on, dedicated_run_threads=on |
| Duration requested / elapsed | 3600 s / 3607 s |
| Load/unload cycles | 86 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 344; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3058, 4054 |
| Unload ms (median, max) | 183.5, 197.5 |
| Request p50 ms (median over cycles) | 18263.4 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 86 | 0 | 14.48 | 67.43 | 183.86 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 72 | 181.0 | 182.6 | 181.6 | 179.1 | 182.6 | 0.9 | 0.7 | 0.4 | 0.6 | 32.0 | plateau |
| unloaded | working_set | yes | 72 | 412.1 | 469.1 | 456.3 | 412.1 | 476.1 | 16.2 | 13.0 | 1.8 | 2.5 | 32.0 | plateau |
| unloaded | private | no | 72 | 2528.7 | 2576.3 | 2541.5 | 2494.2 | 2581.2 | 9.7 | 7.8 | -15.6 | 6.0 | 50.8 | plateau |
| unloaded | rss_anon | yes | 72 | 337.8 | 392.7 | 380.0 | 337.8 | 401.2 | 14.1 | 11.3 | 0.5 | 1.9 | 32.0 | plateau |
| loaded | managed_heap | yes | 72 | 259.8 | 260.9 | 260.0 | 258.8 | 261.4 | 0.9 | 0.7 | 0.5 | 0.6 | 32.0 | plateau |
| loaded | working_set | yes | 72 | 2748.2 | 2798.6 | 2790.4 | 2734.2 | 2897.5 | -13.6 | -10.9 | -50.5 | 32.7 | 55.8 | plateau |
| loaded | private | no | 72 | 3103.7 | 3152.4 | 3147.9 | 3086.4 | 3269.7 | -10.1 | -8.1 | -36.1 | 21.2 | 63.0 | plateau |
| loaded | rss_anon | yes | 72 | 2673.9 | 2722.3 | 2714.6 | 2657.9 | 2821.2 | -15.6 | -12.5 | -51.7 | 32.0 | 54.3 | plateau |

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
| git_commit | a5a0f374012118a3ba0a764018448dac111797a9 |
| git_dirty | false |
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38983201554, variant default\u002Bsmaps5 --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38983201554, variant default\u002Bsmaps5 |
| load before (loadavg, MemAvailable MB) | [2.61,0.97,0.36], 14182 |
| load after (loadavg, MemAvailable MB) | [6.3,6.17,6.21], 14062 |
