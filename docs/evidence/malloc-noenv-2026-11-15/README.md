# One-hour soaks without `MALLOC_MMAP_THRESHOLD_`, 2026-11-15 (issue #178)

Per-profile verdicts on the Linux deployment requirement `MALLOC_MMAP_THRESHOLD_=65536`, from one-hour G4 qualification soaks run without the variable (default glibc allocator), plus the in-process `mallopt(M_MMAP_THRESHOLD, 65536)` opt-in shipped for hosts that cannot set the environment.

## Environment

- Host: the maintainer's workstation (the self-hosted `qualification` runner): Omarchy, kernel 7.2.8-9-omarchy, Intel Core i7-5820K (6 cores, 12 threads, AVX2/FMA), 94 GiB RAM, glibc 2.44, .NET 10.0.14, ONNX Runtime 1.30.0.
- Transparent huge pages: `madvise` (checked, already correct; the runner rule wants `madvise` and no change was needed).
- Contention: the machine was busier than a qualification run normally is — a resident Corvid `app-cuda` process (6.8 GB RSS), a ComfyUI server, GitHub runner jobs and parallel agent builds ran alongside. Request latencies are inflated accordingly (gliner2 p50 60-200 ms against 65-110 ms in past runs); memory series are unaffected and are what the growth rule reads.
- Runs are CPU (`CPUExecutionProvider`); `cuda_visible_devices` was unset, and no CUDA soak ran here (GPU 0 hosts the Strata server).
- Two earlier attempts on this date were killed at 250 s and 3652 s of 3600 s by `systemd-oomd` (exit 137): oomd monitors `app.slice` at a 50% memory-pressure limit for 20 s, and terminal-launched processes live there. The completed reruns launched into `session.slice`, which oomd does not monitor. A third kill (15:36) took the video rerun while it ran next to the gliner2 soak; the video run was requeued serialized, alone, after gliner2 finished. The kill list is recorded here because it is the one caveat on this evidence: the memory series of the killed runs are valid, but they are checkpoints, not full hours.

## Method

The soak tool is `ai-dotnet/tools/Starlings.Soak.Tool`, invocation (no `MALLOC_*` variable anywhere in the environment):

```sh
dotnet run --project ai-dotnet/tools/Starlings.Soak.Tool -- <workload> \
  --out <dir> --duration 1h --note "issue #178 no-env: no MALLOC_ env, THP madvise, local workstation"
```

Verdict rule unchanged (G4): every gating settled series must plateau after warmup (growth within the rule's limit over the hour); any wrong output or error fails. Raw reports under `runs/` (schema `starlings-soak-raw/1`).

## Results

Four profiles, four one-hour default-allocator soaks. The gliner2 and embeddinggemma2-video runs were first killed early (250 s, 607 s checkpoints — both passing at the checkpoint) and rerun to the full hour; kev-0.8b and kev-4b completed on the first pass.

| Profile (workload) | Provider | Duration | Verdict without `MALLOC_MMAP_THRESHOLD_` | Key series (rss_anon MB, settled) |
| --- | --- | --- | --- | --- |
| `gliner2-base-v1/onnx-fp32-cpu` | CPU | 3600 s (rerun; first pass killed at 250 s, also passing) | **PASS** — requirement DROPPED | loaded ~1.55-1.80 GB flat; unloaded ~80 → 165-185 MB rising over the first half hour, then flat (retention is reused by the next load, not growth) |
| `embeddinggemma-2/video-fp32-cpu` | CPU | 3600 s (serialized rerun; first pass killed at 607 s, also passing) | **PASS** — confirms the CPU video profile needs nothing | loaded ~2.05-2.14 GB flat; unloaded ~220 → 365-385 MB rising early, then flat |
| `kev-0.8b/onnx-fp32-cpu` | CPU | 3616 s | **FAIL** — requirement STAYS | unloaded 97 → ~330 MB (≈86 MB/h); loaded 2.14 → 2.37 GB (≈92 MB/h); outputs correct throughout |
| `kev-4b/onnx-fp32-cpu` | CPU | 3653 s | **PASS** — consistent with the 2026-10-17 CUDA result | loaded ~14.0-14.3 GB flat; unloaded ~106 → ~400 MB early, then flat within the rule |

Notes:

- kev-0.8b failing on CPU as well as the documented 64 MB/h CUDA failure (2026-10-17) makes the Kev requirement provider-independent: long-running Kev-0.8B processes need the threshold set, however the model is served.
- The unloaded series of gliner2, video and kev-4b all step up in the first 10-30 minutes and then hold; the soak rule reads the whole-hour fit and the settled tail, and both stay inside it. kev-0.8b's unloaded series climbs in steps for the whole hour and its loaded series climbs linearly — the same shape as its CUDA failure.
- Verdicts are per profile. This rerun covered the CPU profiles only; no CUDA soak ran on this date, so the CUDA requirements (GLiNER2 CUDA, `video-fp32-cuda`) are unchanged by this evidence.

## Verdicts and doc changes

| Profile | Requirement before | Requirement after |
| --- | --- | --- |
| `gliner2-base-v1/onnx-fp32-cpu` | required the variable | **dropped** (G4 passes without it) |
| `gliner2-base-v1/onnx-fp32-cuda` | required | **stays** (no CUDA soak ran; all CUDA evidence is with the variable) |
| `embeddinggemma-2/video-fp32-cpu` | not required | confirmed not required (full hour) |
| `embeddinggemma-2/video-fp32-cuda` | required | **stays** (unchanged; this rerun was CPU) |
| `kev-0.8b/*` | required | **stays**, now evidenced on CPU too |
| `kev-4b/*` | not required (at the default allocator) | confirmed |

## The opt-in for hosts that cannot set the environment

Corvid re-executes itself today to place the variable in its own environment before `Main`. For hosts like that, `Starlings.Runtime.Onnx` now offers an opt-in that applies the same threshold in-process: `OnnxSessionOptions.ApplyGlibcMmapThreshold = true` (or `GlibcMallocThreshold.Apply()` directly) calls glibc's `mallopt(M_MMAP_THRESHOLD, 65536)` once per process, with a loud diagnostic on standard error and the outcome in `OnnxSessionDiagnostics.GlibcMmapThresholdDiagnostic`. It is off by default — the package never calls `mallopt` unless the host asks — because the call changes the allocator of the whole host process and glibc caches the setting after the first call. The environment-variable form remains the primary recommendation. Documented in one place, the [`Starlings.Runtime.Onnx` README, "Setting `MALLOC_MMAP_THRESHOLD_`"](../../../ai-dotnet/src/Starlings.Runtime.Onnx/README.md#setting-malloc_mmap_threshold_), and linked from the package READMEs that state a per-profile requirement.

## Files

- `runs/*.json`, `runs/*.md`: the four raw soak reports (killed-run checkpoints included; verdict text marks the incompleteness).

## Remaining work (maintainer)

- The G4 gate on a quiet hosted runner is a maintainer step (this evidence is from the contended workstation); the release-record change (`docs/releases/ai-0.3.0.md`, Upgrading) follows it.
- A default-allocator CUDA soak for GLiNER2 CUDA and `video-fp32-cuda` would either confirm or drop their requirements; not run here (GPU 0 is the Strata server's).
