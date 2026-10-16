# G4 soak: embeddinggemma-2/image-fp32-cpu

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error (note: unloaded private (address space, report-only) grows 417 MB; loaded private (address space, report-only) grows 215 MB)

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cpu (CPUExecutionProvider) |
| Duration requested / elapsed | 3600 s / 3619 s |
| Load/unload cycles | 124 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 496; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 2451, 3092 |
| Unload ms (median, max) | 263.2, 302.7 |
| Request p50 ms (median over cycles) | 12586.6 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 124 | 0 | 8.91 | 46.54 | 78.18 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 102 | 181.3 | 191.2 | 191.0 | 180.3 | 193.0 | 3.9 | 3.1 | 4.3 | -2.1 | 32.0 | plateau |
| unloaded | working_set | yes | 102 | 512.6 | 770.7 | 702.5 | 512.6 | 819.7 | 333.2 | 266.6 | 165.3 | 2.6 | 32.0 | plateau |
| unloaded | private | no | 102 | 2574.3 | 3159.7 | 2970.8 | 2574.3 | 3159.7 | 520.5 | 416.5 | 384.1 | 136.5 | 59.4 | growth |
| unloaded | rss_anon | yes | 102 | 438.3 | 694.7 | 626.8 | 438.3 | 743.8 | 331.4 | 265.2 | 164.1 | 2.2 | 32.0 | plateau |
| loaded | managed_heap | yes | 102 | 260.0 | 270.0 | 269.7 | 259.0 | 271.7 | 3.9 | 3.1 | 4.3 | -2.2 | 32.0 | plateau |
| loaded | working_set | yes | 102 | 4877.0 | 5141.9 | 5146.7 | 4530.7 | 5799.5 | 186.1 | 148.9 | 201.6 | -69.5 | 102.9 | plateau |
| loaded | private | no | 102 | 7522.5 | 8103.7 | 7915.2 | 6759.0 | 8365.2 | 268.6 | 214.9 | 435.4 | 452.7 | 158.3 | growth |
| loaded | rss_anon | yes | 102 | 4802.7 | 5065.9 | 5071.0 | 4455.4 | 5723.8 | 184.3 | 147.4 | 200.3 | -69.9 | 101.4 | plateau |

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
| git_commit | ba816c33548410a5920014b5fb010799fd6a4f3a |
| git_dirty | false |
| command_line | embeddinggemma2-image --out /home/runner/work/starling/starling/artifacts/qualification/soak --duration 1h --note GitHub-hosted ubuntu-latest, scheduled qualification run 38822625245 |
| machine_note | GitHub-hosted ubuntu-latest, scheduled qualification run 38822625245 |
| load before (loadavg, MemAvailable MB) | [3.29,0.94,0.33], 14307 |
| load after (loadavg, MemAvailable MB) | [7.18,6.72,6.39], 13823 |
