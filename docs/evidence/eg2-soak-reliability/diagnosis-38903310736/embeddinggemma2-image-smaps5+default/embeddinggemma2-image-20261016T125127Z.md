# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=off, trim_native_heap_on_release=on; multimodal text: cpu_memory_arena=off, trim_native_heap_on_release=on |
| Duration requested / elapsed | 3600 s / 3618 s |
| Load/unload cycles | 87 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 348; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3019, 3998 |
| Unload ms (median, max) | 175.9, 188.5 |
| Request p50 ms (median over cycles) | 18120.7 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 87 | 0 | 8.18 | 46.82 | 165.06 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 72 | 181.3 | 181.2 | 181.1 | 180.0 | 182.6 | 0.4 | 0.3 | 0.1 | 0.4 | 32.0 | plateau |
| unloaded | working_set | yes | 72 | 418.7 | 535.2 | 490.0 | 418.7 | 597.3 | 120.9 | 96.2 | 78.3 | -27.2 | 32.0 | plateau |
| unloaded | private | no | 72 | 2999.4 | 3143.0 | 3095.3 | 2786.4 | 3382.8 | 367.8 | 292.4 | 264.6 | -188.1 | 61.9 | plateau |
| unloaded | rss_anon | yes | 72 | 344.5 | 459.0 | 414.2 | 344.5 | 521.5 | 118.7 | 94.4 | 76.9 | -27.8 | 32.0 | plateau |
| loaded | managed_heap | yes | 72 | 260.0 | 259.9 | 259.9 | 258.8 | 261.4 | 0.4 | 0.3 | 0.1 | 0.4 | 32.0 | plateau |
| loaded | working_set | yes | 72 | 2774.6 | 2757.3 | 2756.1 | 2601.0 | 2905.6 | 52.4 | 41.7 | 86.2 | -165.4 | 55.1 | plateau |
| loaded | private | no | 72 | 3629.5 | 3735.1 | 3711.1 | 3378.5 | 4005.9 | 361.7 | 287.6 | 272.5 | -200.4 | 74.2 | plateau |
| loaded | rss_anon | yes | 72 | 2700.4 | 2681.2 | 2680.8 | 2525.6 | 2830.1 | 50.1 | 39.8 | 84.8 | -166.1 | 53.6 | plateau |

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
| git_commit | f43a648a795ed6cda366007a61fd7b4158a93082 |
| git_dirty | false |
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38903310736, variant smaps5\u002Bdefault --smaps-every 5 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38903310736, variant smaps5\u002Bdefault |
| load before (loadavg, MemAvailable MB) | [1.54,0.54,0.19], 14206 |
| load after (loadavg, MemAvailable MB) | [5.88,6.16,6.29], 14103 |
