# G4 soak: embeddinggemma-2/image-fp32-cuda

> **Short development run (600 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=on, trim_native_heap_on_release=on, dedicated_run_threads=off, trim_native_heap_after_load=off, trim_native_heap_after_run=off; multimodal text: cpu_memory_arena=on, trim_native_heap_on_release=on, dedicated_run_threads=off, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 600 s / 602 s |
| Load/unload cycles | 145 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 580; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 3212, 4287 |
| Unload ms (median, max) | 134.1, 152.5 |
| Request p50 ms (median over cycles) | 260.5 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 145 | 0 | 63.33 | 100.14 | 108.58 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 117 | 191.2 | 191.2 | 191.3 | 191.2 | 191.8 | 2.2 | 0.3 | 0.1 | 0.0 | 32.0 | plateau |
| unloaded | working_set | yes | 117 | 1144.5 | 1152.1 | 1152.1 | 1144.0 | 1158.2 | 44.4 | 5.9 | 8.8 | 0.0 | 32.0 | plateau |
| unloaded | private | no | 117 | 2090.9 | 1964.2 | 1972.3 | 1937.0 | 2090.9 | -52.6 | -7.0 | 0.1 | 2.2 | 39.4 | plateau |
| unloaded | rss_anon | yes | 117 | 785.3 | 792.4 | 792.5 | 785.3 | 799.1 | 36.9 | 4.9 | 7.7 | -0.1 | 32.0 | plateau |
| unloaded | gpu | yes | 117 | 282.0 | 282.0 | 282.0 | 282.0 | 282.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 117 | 306.9 | 309.8 | 308.5 | 306.5 | 310.5 | 5.6 | 0.7 | 0.6 | 0.1 | 32.0 | plateau |
| loaded | working_set | yes | 117 | 2137.9 | 2094.8 | 2105.4 | 2060.4 | 2168.9 | -32.2 | -4.3 | 1.2 | -18.6 | 42.1 | plateau |
| loaded | private | no | 117 | 2370.7 | 2331.0 | 2341.6 | 2293.4 | 2413.2 | -7.6 | -1.0 | 2.4 | -19.8 | 46.8 | plateau |
| loaded | rss_anon | yes | 117 | 1777.8 | 1733.1 | 1743.7 | 1698.9 | 1808.0 | -40.6 | -5.4 | -0.0 | -18.8 | 34.9 | plateau |
| loaded | gpu | yes | 117 | 6772.0 | 7532.0 | 6768.0 | 5612.0 | 7676.0 | -87.6 | -11.7 | 13.1 | -199.6 | 135.4 | plateau |

Growth rule: fitted growth over the post-warmup window above the limit, and continuing in both halves of the window (a one-time step, such as an allocator high-water mark, is a plateau). `retention`: the unloaded series grows but the loaded series plateaus, so the memory kept between loads is reused by the next load (allocator retention, not a leak; reported, does not fail). Screening heuristic; the raw samples are in the JSON.

## Environment

| Item | Value |
| --- | --- |
| cpu_model | Intel(R) Core(TM) i7-5820K CPU @ 3.30GHz |
| logical_cpus | 12 |
| ram_total_mb | 96442 |
| os | Omarchy |
| kernel | 7.2.8-9-omarchy |
| runtime | .NET 10.0.14 |
| server_gc | false |
| ort_version | 1.30.0 |
| genai_version | 0.17.1 |
| genai_native_package | Microsoft.ML.OnnxRuntimeGenAI.Cuda/0.17.1 |
| gpu_identity | 0, NVIDIA GeForce RTX 3090, 615.78.08, 24576 MiB; 1, NVIDIA GeForce RTX 4060 Ti, 615.78.08, 16380 MiB |
| cpu_governor | schedutil |
| allocator_and_runtime_env | {"DOTNET_CLI_TELEMETRY_OPTOUT":"1","DOTNET_NOLOGO":"1"} |
| git_commit | n/a |
| git_dirty | n/a |
| command_line | embeddinggemma2-image --cuda --duration 10m --out <local>/out-main/soak/embeddinggemma2-image --note scheduled CUDA qualification on RunPod, NVIDIA GeForce RTX 3090, commit 3a021dc0b7f8, CUDA libraries from the pinned eg2 requirements-cuda wheels |
| machine_note | scheduled CUDA qualification on RunPod, NVIDIA GeForce RTX 3090, commit 3a021dc0b7f8, CUDA libraries from the pinned eg2 requirements-cuda wheels |
| load before (loadavg, MemAvailable MB) | [1.86,1.74,2.59], 74574 |
| load after (loadavg, MemAvailable MB) | [2.44,1.94,2.22], 73150 |
