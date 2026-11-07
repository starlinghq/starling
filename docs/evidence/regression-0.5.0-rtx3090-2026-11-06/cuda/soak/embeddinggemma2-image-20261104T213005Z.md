# G4 soak: embeddinggemma-2/image-fp32-cuda

> **Short development run (600 s), not the one-hour qualification run of plan 10.2.** It exercises the harness and screens for gross growth; it does not qualify gate G4.

**Verdict:** PASS: every gating settled series plateaus after warmup; no wrong output or error

## Run

| Item | Value |
| --- | --- |
| Model lock | google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b |
| Provider (session) | cuda (CUDAExecutionProvider, CPUExecutionProvider) |
| Session memory settings | vision: cpu_memory_arena=on, trim_native_heap_on_release=on, dedicated_run_threads=off, trim_native_heap_after_load=off, trim_native_heap_after_run=off; multimodal text: cpu_memory_arena=on, trim_native_heap_on_release=on, dedicated_run_threads=off, trim_native_heap_after_load=off, trim_native_heap_after_run=off |
| Duration requested / elapsed | 600 s / 604 s |
| Load/unload cycles | 73 |
| Requests per cycle, concurrency | 5, 2 callers |
| Intra-op threads | runtime default |
| Warmup excluded | first 2 cycles and the first 0.20 of the run |
| Requests checked against the first-cycle reference | 292; wrong outputs 0; unexpected errors 0 |
| Load ms (median, max) | 7083, 9640 |
| Unload ms (median, max) | 216.8, 300.1 |
| Request p50 ms (median over cycles) | 296.8 |

## Timed-out requests and cancellation latency

Every 5th request is a long request with a deadline of 5 to 150 ms. Latency is the time from the deadline firing to the call throwing `OperationCanceledException`; the next requests on the same instance are checked against the reference (cancellation then reuse).

| Cancelled | Completed before the deadline | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| 73 | 0 | 69.98 | 105.36 | 108.52 |

## Memory after warmup (settled samples, MiB)

`unloaded`: after disposing the model and a full GC, once per cycle. `loaded`: after the cycle's requests and a full GC, model still loaded. `rss_anon` is anonymous resident memory (managed heap plus native allocations), the main leak signal; `private` is .NET's private bytes, which on Linux is the `VmData` address space (malloc arenas reserve address space they never touch), so there it is reported but does not gate (column Gates); GPU only where nvidia-smi reports this process.

| Series | Metric | Gates | Points | First | Last | Median | Min | Max | Slope MB/h | Fitted growth | 1st half | 2nd half | Limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unloaded | managed_heap | yes | 59 | 191.2 | 191.5 | 191.3 | 191.2 | 191.6 | 1.3 | 0.2 | 0.1 | 0.1 | 32.0 | plateau |
| unloaded | working_set | yes | 59 | 1140.6 | 1159.4 | 1157.0 | 1140.6 | 1161.4 | 145.7 | 19.3 | 20.4 | 3.7 | 32.0 | plateau |
| unloaded | private | no | 59 | 2001.4 | 2044.8 | 2027.9 | 1995.5 | 2063.8 | 275.4 | 36.4 | 20.9 | 2.6 | 40.6 | plateau |
| unloaded | rss_anon | yes | 59 | 782.9 | 800.1 | 798.1 | 782.9 | 802.1 | 136.2 | 18.0 | 19.2 | 3.2 | 32.0 | plateau |
| unloaded | gpu | yes | 59 | 282.0 | 282.0 | 282.0 | 282.0 | 282.0 | 0.0 | 0.0 | 0.0 | 0.0 | 32.0 | plateau |
| loaded | managed_heap | yes | 59 | 308.3 | 316.4 | 308.4 | 306.6 | 317.1 | 3.0 | 0.4 | -1.8 | 2.7 | 32.0 | plateau |
| loaded | working_set | yes | 59 | 2154.7 | 2139.2 | 2124.0 | 2050.0 | 2193.0 | 39.5 | 5.2 | -50.1 | 60.9 | 42.5 | plateau |
| loaded | private | no | 59 | 2334.4 | 2316.3 | 2303.9 | 2220.2 | 2385.5 | 180.2 | 23.9 | -39.2 | 58.5 | 46.1 | plateau |
| loaded | rss_anon | yes | 59 | 1795.2 | 1777.8 | 1763.0 | 1689.6 | 1831.7 | 29.2 | 3.9 | -51.5 | 60.3 | 35.3 | plateau |
| loaded | gpu | yes | 59 | 6764.0 | 7534.0 | 6766.0 | 5614.0 | 7668.0 | 2130.2 | 282.0 | -372.0 | 442.5 | 135.3 | plateau |

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
| command_line | embeddinggemma2-image --cuda --duration 10m --out <local>/out-0.5.0/soak/embeddinggemma2-image --note scheduled CUDA qualification on RunPod, NVIDIA GeForce RTX 3090, commit ce5225854fa7, CUDA libraries from the pinned eg2 requirements-cuda wheels |
| machine_note | scheduled CUDA qualification on RunPod, NVIDIA GeForce RTX 3090, commit ce5225854fa7, CUDA libraries from the pinned eg2 requirements-cuda wheels |
| load before (loadavg, MemAvailable MB) | [38.23,23.37,21.96], 63884 |
| load after (loadavg, MemAvailable MB) | [42.27,35.26,28.54], 56424 |
