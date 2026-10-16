# Scheduled qualification regression (run 38793842060): diagnosis and fix

The scheduled qualification run [38793842060](https://github.com/starlinghq/starling/actions/runs/38793842060) (workflow_dispatch, GitHub-hosted `ubuntu-latest`, source `95ae3c9`) failed three jobs that block Stable for e5 CPU and EmbeddingGemma 2 text CPU:

| Job | Result | Cause (this directory) |
| --- | --- | --- |
| `soak multilingual-e5-small/onnx-fp32-cpu` | FAIL, loaded and unloaded `rss_anon` about 50 MiB/h | Transparent huge pages (THP) `always` on the hosted image: khugepaged fills in partly used 2 MiB ranges of the malloc heaps. Not a leak. |
| `soak embeddinggemma-2/onnx-fp32-cpu` | FAIL, about 155 to 165 MiB/h | The same, on top of about 1.4 GiB of freed memory that glibc keeps resident between loads. |
| `release gates embeddinggemma-2/onnx-fp32-cpu` | G2 refused (evaluator exit 3) | The out-of-Git reference `embeddings.jsonl`, regenerated on the runner's AMD EPYC 7763, differs from the reference CPU's file within 8e-7 (517,641 of 535,248 values), so its SHA-256 (`53cf2f2b…`) is not the one `reference.json` records (`c2ffe5e4…`); the runner regenerates the same `53cf2f2b…` every time. The ISA pins of #56 could not make PyTorch's CPU kernels bit-identical across vendors. |

The full qualification run on this branch, [38822625245](https://github.com/starlinghq/starling/actions/runs/38822625245), is green: every job, including the e5 and EmbeddingGemma 2 text soaks and release gates (`full-qualification-38822625245/`).

## Fix

1. **Library: `Starlings.Runtime.Onnx` returns freed native memory when a session is released.** `OnnxModelSession` calls glibc's `malloc_trim(0)` after the native session, its options and provider options are released (after `Dispose` and the last in-flight run). On by default on Linux with glibc; a no-op elsewhere (no `libc.so.6`: musl, Windows, macOS). Opt out per session with `OnnxSessionOptions.TrimNativeHeapOnRelease = false` or for the process with the `AppContext` switch `Starlings.Runtime.Onnx.DisableNativeHeapTrim`; `OnnxSessionDiagnostics.TrimNativeHeapOnRelease` reports the effective setting. `malloc_trim` only releases memory that is already free; the library changes no allocator setting (`mallopt`, `MALLOC_*`) and no kernel setting. It never runs on the request path. Documented in the package README, "Memory on Linux". The soak tool's `--no-heap-trim` sets the switch (diagnostics).
2. **Qualification soak environment: THP `madvise`.** The hosted image sets `/sys/kernel/mm/transparent_hugepage/enabled` to `always`; the Ubuntu kernel default is `madvise`. The soak step now sets `madvise` (sudo on the runner) and records the setting in the report. This is an environment precondition of the measurement, like the quiet machine of M6 or GLiNER2's documented `MALLOC_MMAP_THRESHOLD_`; it is **not** a change to the growth rule (`Analysis.cs` is unchanged) and not something the library asks users to set. **Maintainer decision:** accept it as the soak's environment, or keep `always` (the variant part `thp-always`) and read a failure with the non-huge-page fit below.
3. **Release gates: EmbeddingGemma 2 text G2 reads the committed reference.** After #60 committed the accepted `embeddings.jsonl` (digest `c2ffe5e4…`) under `protected/`, `tools/release-gates.sh` no longer regenerates it and no longer passes `--reference-embeddings` (which the evaluator refuses for a set that commits its own file). G2 reads the protected file, verified against Git by the evaluator, and the script also checks it against the digest `reference.json` records and fails the job on a mismatch. The job is no longer `continue-on-error`.

Also in this change: `qualification.yml` workflow_dispatch inputs for allocator diagnostics on the hosted runner (`soak_only`, `soak_workloads`, `soak_variants`; defaults leave the qualification run unchanged), and soak tool diagnostics: `AnonHugePages` of `/proc/self/smaps_rollup` in every settled sample, `--smaps-every <n>` mapping dumps, and the THP, khugepaged and glibc settings in the environment manifest.

GLiNER2's documented `MALLOC_MMAP_THRESHOLD_=65536` stays in its README and in the soak job. Its one-hour soak passed without it once with the library trim (`runner-1h-38809646407/gliner2-no-env/`, 263 MiB after unloading against 1,114 MiB without the trim); dropping the requirement needs more runs and is a follow-up.

## Evidence

Every run is a GitHub-hosted `ubuntu-latest` job (4 vCPUs, 16 GB, Ubuntu 24.04, glibc 2.39, kernel 6.17 azure, THP `always` unless stated) on this branch, `soak_only=true`, runtime-default threads, 2 callers. CPU models vary per job (AMD EPYC 7763, 9V45, 9V74; Intel Xeon 8370C, 8573C, 6973P-C) and are recorded in each report. Raw results are gzipped next to each report (`*.json.gz`; `zcat` them, and `analysis/*.py` read the unzipped JSON).

| Directory | Run | What |
| --- | --- | --- |
| `runner-15m-38806653601/` | [38806653601](https://github.com/starlinghq/starling/actions/runs/38806653601) | 15-minute screen: library trim (default), `no-heap-trim` (the old behaviour), and with it `MALLOC_MMAP_THRESHOLD_=65536`, `MALLOC_ARENA_MAX=2`, the tool's `--malloc-trim`; e5, EmbeddingGemma 2 text, GLiNER2 (`no-env`: without its documented variable) |
| `runner-1h-38809646407/` | [38809646407](https://github.com/starlinghq/starling/actions/runs/38809646407) | One hour: trim (twice: `default` and `no-env` are the same configuration for e5 and EmbeddingGemma 2) against no trim |
| `runner-40m-smaps-38817823391/` | [38817823391](https://github.com/starlinghq/starling/actions/runs/38817823391) | 40 minutes with `/proc/self/smaps` dumps every 25 cycles (`*.smaps.jsonl.gz`) |
| `runner-1h-thp-38822628830/` | [38822628830](https://github.com/starlinghq/starling/actions/runs/38822628830) | One hour, THP `madvise` (trim and no trim) against `always` (`thp-always`), with smaps dumps |
| `full-qualification-38822625245/` | [38822625245](https://github.com/starlinghq/starling/actions/runs/38822625245) | The full qualification workflow, default inputs: soak reports and release-gate summaries |
| `local/` | workstation | e5, 9 minutes, `taskset -c 0-3` and `DOTNET_PROCESSOR_COUNT=4`: trim against no trim |
| `g5-ab/` | workstation | Warm latency A/B of the C# harnesses, trim on and off |

### 1. glibc retention between loads: the library trim removes it

Settled `rss_anon` after unloading (median over the post-warmup window, MiB), one-hour runs:

| Profile | No trim | Library trim | Unload ms (median), no trim → trim | Same CPU? |
| --- | --- | --- | --- | --- |
| EmbeddingGemma 2 text | 1,480 | 402 and 504 | 11.4 → 21.0 and 19.4 | yes (EPYC 7763) |
| GLiNER2, without `MALLOC_MMAP_THRESHOLD_` | 1,114 | 263 | 6.1 → 11.6 | no (Xeon 8573C, 6973P-C) |
| e5 | 277 | 264 and 278 | 8.9 → 9.1 and 8.8 | no |

On the workstation (glibc 2.44) the trim also lowers e5: 416 → 188 MiB after unloading (`local/`). On the runner (glibc 2.39) e5 retains little between loads to begin with. In the smaps dumps without the trim, EmbeddingGemma 2's glibc main heap holds 914 MiB after unloading for the whole run; with the trim it holds 10 MiB (`runner-40m-smaps-38817823391/`). The cost is in unload (a few milliseconds more per released session; EmbeddingGemma 2 releases two to three) and in the next load, which faults the returned pages in again (EmbeddingGemma 2 load median 1,092 → 1,138 to 1,212 ms on the same CPU). Request latency is unaffected: the trim never runs while a session is in use.

### 2. The residual growth on the runner is THP fill-in, not allocation

The soak's `rss_anon` split into `AnonHugePages` (from `smaps_rollup`) and the rest, least-squares fit over the tool's post-warmup window (`analysis/nonthp.py`), one-hour runs with THP `always`:

| Run | Verdict | Unloaded `rss_anon` growth, MiB | of which huge pages | rest | Loaded `rss_anon` growth | of which huge pages | rest |
| --- | --- | --- | --- | --- | --- | --- | --- |
| e5, trim (`runner-1h…/e5-no-env`) | **FAIL** | 57 | 54 | 3 | 69 | 63 | 6 |
| e5, trim (`runner-1h…/e5`) | pass | 28 | 5 | 23 | 22 | -1 | 22 |
| e5, no trim | pass | -15 | -24 | 9 | -8 | -16 | 8 |
| EmbeddingGemma 2, trim (`embeddinggemma2-no-env`) | pass | 45 | 95 | -51 | 34 | 83 | -49 |
| GLiNER2, no trim, no variable | **FAIL** | 67 | 353 | -287 | 70 | 357 | -288 |

In the failing e5 run the memory that is not in huge pages is flat (3 and 6 MiB over the window); all the growth is huge pages. Without the trim, khugepaged converts the retained heap to huge pages over time (huge pages up, the rest down by the same amount). The smaps dumps put the huge pages in the 64 MiB-aligned glibc arenas, the main heap and other anonymous mappings (`analysis/smaps_attr.py`). With THP `always` and the default `khugepaged/max_ptes_none` of 511, khugepaged collapses a 2 MiB range that has a single resident page and fills in the other 511 pages, including pages `malloc_trim` just returned. The resident size then rises with no allocation behind it, and how far depends on the scan timing and the CPU the job lands on. Which hosted runs fail is therefore close to chance: the old behaviour without trim passed one hour on an EPYC 9V45, and the same configuration with trim passed once and failed once.

With THP `madvise` (`runner-1h-thp-38822628830/`) the processes have no huge pages, and every series passes:

| Profile | Variant | Unloaded median / growth, MiB | Loaded median / growth, MiB | Verdict |
| --- | --- | --- | --- | --- |
| e5 | trim, `madvise` | 182 / 19 | 753 / 19 | pass |
| e5 | no trim, `madvise` | 270 / 14 | 790 / 15 | pass |
| e5 | trim, `always` | 198 / 30 | 895 / 37 | pass |
| EmbeddingGemma 2 text | trim, `madvise` | 521 / -13 | 2,204 / 36 | pass |
| EmbeddingGemma 2 text | no trim, `madvise` | 1,510 / 126 | 2,300 / 96 | pass (second half 11 MiB/h) |
| EmbeddingGemma 2 text | trim, `always` | 493 / 22 | 2,204 / 57 | pass |

A library cannot change THP for its host (`prctl(PR_SET_THP_DISABLE)` or `madvise` on glibc's heaps is process policy, like `mallopt`), so this is an environment fact. The package README documents it.

The original failures predate the `AnonHugePages` sample, so their split cannot be computed. The same profiles on the same image, with the cause seen in every run that recorded it, are the evidence.

### 3. Release gates

`full-qualification-38822625245/release-gates/embeddinggemma-2--onnx-fp32-cpu.md`: G1 holdout, G2 release (on the protected `embeddings.jsonl`, digest equal to `reference.json`) and G3 retrieval passed; e5 and GLiNER2 passed too. The regenerated runner file of the failing run (`53cf2f2b…`) against the committed one: max abs difference 8.0e-7.

### 4. Latency and fidelity

- Warm latency (C# harness, workstation i7-5820K, 6 threads, shared machine at load average 6 to 8): trim on against off (`g5-ab/`, `analysis/warmcmp.py`). EmbeddingGemma 2 text: every workload within 1.5% (0.985 to 1.015). e5: 0.91 to 1.10 per workload, scattered both ways, plus one long512-b32 outlier at 1.24 with n=5 on the loaded machine. The trim never runs on the warm path (it runs only when a session is released). "Off" is the same build with the switch set in the harness's `runtimeconfig.json`. The full matched G5 protocol (`run-e5-matched.sh`, `run-g5-matched.sh` with `1 6 3`) does not fit this environment's 10-minute command limit; the scheduled job's G5 numbers are unaffected for the same reason.
- Tests: every project in `AI.slnx` passed on the workstation with the models present, including e5 and EmbeddingGemma 2 text fidelity (image and video parity skipped: no local image or video set). One e5 run hit a placement-profiling file race in `TextParityTests` (an ONNX Runtime profile written to the working directory) and passed on rerun; the race is unrelated to this change. The hosted `model tests (linux x64 cpu)` job of the full run passed. Outputs are unchanged: the trim runs after the native session is released, never while a session is in use.

## Reproduce

```sh
# hosted runner diagnostics (read-only token; the defaults are the qualification run)
gh workflow run qualification.yml --ref <branch> -f soak_only=true -f soak_duration=1h \
  -f "soak_workloads=e5 embeddinggemma2" -f 'soak_variants=["smaps","no-heap-trim+smaps","thp-always+smaps"]'
# locally, 4 CPUs
DOTNET_PROCESSOR_COUNT=4 taskset -c 0-3 ai-dotnet/tools/Starlings.Soak.Tool/bin/Release/net10.0/Starlings.Soak.Tool \
  e5 --duration 9m --out <dir> [--no-heap-trim] [--smaps-every 25]
python3 docs/evidence/scheduled-regression-fix/analysis/nonthp.py <raw.json>...
```
