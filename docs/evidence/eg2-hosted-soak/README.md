# EmbeddingGemma 2 one-hour CPU soaks on the hosted runner: diagnosis and fix

The weekly qualification soaks on GitHub-hosted `ubuntu-latest` (4 vCPUs, glibc 2.39, transparent huge pages set to `madvise` by the job) failed for two EmbeddingGemma 2 profiles, although both passed locally and on RunPod:

| Profile | Before this change | Runs |
| --- | --- | --- |
| `embeddinggemma-2/video-fp32-cpu` (Experimental) | **FAIL** in every run, loaded `rss_anon` about 280 MB/h | 38837950614, 38841680114, 38848321624 (`before/embeddinggemma2-video-…md`) |
| `embeddinggemma-2/onnx-fp32-cpu` (text, Stable) | **FAIL** twice (about 130 MB/h), then a pass | 38837950614, 38841680114 (`before/embeddinggemma2-…md`), 38848321624 |

After this change, the full qualification workflow on this branch (`skip_cuda=true`) passed the text and video soaks in three runs out of three:

| Run | Text soak | Video soak | Image soak (context) |
| --- | --- | --- | --- |
| [38874143326](https://github.com/starlinghq/starling/actions/runs/38874143326) | pass: unloaded 13 MiB, loaded 12 MiB fitted growth (EPYC 9V74) | pass: unloaded 41 (second half 4), loaded 32 (7763) | **fail**: unloaded 105, loaded 146 (Xeon 8573C) |
| [38882433616](https://github.com/starlinghq/starling/actions/runs/38882433616) | pass: unloaded 39 (`retention`), loaded **35.7 against a limit of 36.0** (9V45) | pass: unloaded 50 (`retention`), loaded 60 against 55.6, second half 3 (7763) | pass (7763) |
| [38892944927](https://github.com/starlinghq/starling/actions/runs/38892944927) | pass: unloaded 13, loaded 17 (7763) | pass: unloaded -22, loaded -57 (7763) | pass (9V45) |

Reports and raw data: `full-qualification-<run>/`. Every soak report now has a "Session memory settings" row (`cpu_memory_arena=off, trim_native_heap_on_release=on` for every EmbeddingGemma 2 session in these runs).

**The text result is green but not by a wide margin.** In run 38882433616 the loaded series grew 35.7 MiB against a limit of 36.0 MiB. The remaining growth has a known cause (section 3) that a library cannot remove. If it recurs, `MALLOC_ARENA_MAX=2` removed it in the 15-minute screen. That setting is not in the text contract, so adding it is a maintainer decision.

## 1. Job environment (checked first)

- **Contract environment.** Neither the text contract nor the video profile requires a `MALLOC_*` setting on CPU. Only `video-fp32-cuda` requires `MALLOC_MMAP_THRESHOLD_=65536`. The CPU soak jobs ran with no `MALLOC_*` variable (`allocator_and_runtime_env` in every report), which matches.
- **Transparent huge pages.** The write succeeded and read back as `always [madvise] never` in the failing runs (job log of 38848321624 and `transparent_hugepage` in every report). The job now fails if the value read back after the write is not the one it wrote, instead of measuring under `always`.
- **Heap trim (#66).** It was active: `native_heap_trim_disabled: false` in the options. The reports now also record each session's effective setting from `OnnxSessionDiagnostics`.

The job environment matched the contract. The cause was in the process.

## 2. Diagnosis on the hosted runner

Two 15-minute screens with `/proc/self/smaps` and glibc `mallinfo2` dumps every 5 cycles, at both the loaded and unloaded points (new in the soak tool), for text and video. Directories: `runner-15m-38861019342/` and `runner-15m-38863883673/`; scripts in `analysis/`.

| Variant | Text | Video |
| --- | --- | --- |
| default (arena on) | pass (15 min is short) | pass, loaded fitted 132 MiB |
| `MALLOC_MMAP_THRESHOLD_=65536` | pass | **fail**: loaded 589 MiB/h |
| `MALLOC_ARENA_MAX=2` | pass | pass |
| `--keep-loaded` (one session, no unloads) | pass, flat | **fail**: loaded and unloaded **2,234 MiB/h** |
| ONNX Runtime CPU arena off | pass | pass |
| arena off, `--keep-loaded` | pass, flat | pass |
| arena shrinkage after every run (`memory.enable_memory_arena_shrinkage`) | pass | **fail**: 949 MiB/h |
| arena shrinkage, `--keep-loaded` | pass | **fail**: 1,003 MiB/h |

What grows (`analysis/smaps_summary.py`, `analysis/maps_diff.py`, `analysis/arenas.py`):

1. **ONNX Runtime's CPU arena while a session lives.** With one video session kept loaded, a single 2,241 MiB anonymous mapping grew from 1,342 to 1,843 MiB resident in 15 minutes. glibc's `mallinfo2` stayed flat over that time (in use 1,241 MiB, mmapped 2,844 MiB), so nothing new was allocated. This mapping is one of the mmapped chunks counted in `hblkhd`: a region of ONNX Runtime's BFC arena. The arena never returns a region while the session lives, and the region's resident part grows as runs place buffers at different offsets in it. The video graphs' buffers fill a region of 2.2 GB. Text's are small enough that the text keep-loaded run stayed flat.
2. **glibc per-thread arenas after unload.** After unloading and `malloc_trim(0)`, `mallinfo2` reported about 10 MiB in use. The 64 MiB-aligned thread arenas still held 50 to 160 MiB resident, and their number grew during the run (1 to 7 for video, 4 to 7 for text). Each arena kept 16 to 49 MiB. This memory is the arenas' top chunks. glibc trims a non-main arena's top only when a `free` leaves it above the trim threshold, and the dynamic threshold rises to twice the largest mmapped chunk freed so far (up to 64 MiB). `malloc_trim` never trims it. `analysis/trimtest2.c` and `trimtest3.c` show both cases: 20 MiB freed below a live chunk is returned by `malloc_trim` from another thread, and 20 MiB freed into the top is not, whether the thread is alive or has exited (glibc 2.44; the runner's 2.39 behaves the same in the dumps). With the arena on, ONNX Runtime extends its arena by doubling regions (1, 2, 4, ... MiB). The regions of up to 32 MiB are allocated on the calling threads, land in those threads' arenas, and are freed at unload into their tops.

`MALLOC_MMAP_THRESHOLD_=65536` (the GLiNER2 setting) does not help video: the arena regions are mmapped anyway, and it is the first mechanism that grows. `MALLOC_ARENA_MAX=2` bounds the second mechanism but not the first; it passed only because 15 minutes is short.

## 3. Fix: EmbeddingGemma 2 CPU sessions run without ONNX Runtime's CPU arena

- `Starlings.Runtime.Onnx`: new `OnnxSessionOptions.CpuMemoryArena` (`bool?`). `null` keeps ONNX Runtime's default (on) for sessions created directly. `OnnxSessionDiagnostics.CpuMemoryArena` reports the effective value.
- `Starlings.Models.EmbeddingGemma2`: the text encoder and the multimodal embedder (vision and multimodal text graphs) create CPU sessions with the arena off unless the caller sets `CpuMemoryArena`. CUDA sessions are unchanged.
- No `mallopt` and no `MALLOC_*` setting. The heap trim on release (#66) is unchanged.

With the arena off, ONNX Runtime allocates and frees each buffer per run. Buffers above glibc's mmap threshold are unmapped when freed, and the per-thread arenas keep smaller tops (about 20 MiB each instead of 40 to 49 MiB), because the 16 and 32 MiB arena regions no longer pass through them. The loaded footprint drops: video from about 3.9 to 2.7 GB, text from about 2.1 to 1.8 GB, images from about 5.0 to 2.7 GB.

**What remains.** Thread-arena tops below the trim threshold still accumulate as the .NET thread pool rotates the threads that call ONNX Runtime. This is the growth left in run 38882433616. A library cannot return that memory without changing process-wide allocator policy (`mallopt`, `MALLOC_ARENA_MAX`), which this change does not do.

Alternatives tested and rejected: arena shrinkage after every run (worse for video, above), and `MALLOC_*` settings (process policy, and not in the contracts).

## 4. Fidelity

- Embeddings are bit-identical with the arena on and off (`CpuArenaTests`). This was checked on the workstation (i7-5820K, AVX2) and on six hosted runners in run [38892924012](https://github.com/starlinghq/starling/actions/runs/38892924012), a diagnostic branch not merged: EPYC 7763 three times, EPYC 9V45 twice and Xeon 8573C once (`release-gates-video/arena-ab-38892924012.txt`).
- Every project in `AI.slnx` that touches this code passed on the workstation with the models present: `Starlings.Runtime.Onnx.Tests` (37), `Starlings.Models.EmbeddingGemma2.Tests` (146, 19 image and video parity tests skipped for lack of the local fixture sets), the Extensions.AI (44) and ML.NET (27) adapters, and `Starlings.Soak.Tool.Tests` (12). The hosted `model tests (linux x64 cpu)` job passed in all three runs.
- Release gates in the three runs: text, images, e5, GLiNER2 and Qwen passed every gate. **Video G2 failed in runs 38874143326 and 38882433616.** `dotnet-ids.token_embeddings` reached 0.00182808 against a bound of 0.001, on EPYC 9V45 and 9V74 (AVX-512). Video G2 passed in run 38892944927 and on main (0.000381, run 38848321624), all on EPYC 7763 (AVX2). Before this branch, the video gate had only run on EPYC 7763. The same value on two different AVX-512 CPUs and bit-identical outputs with the arena on and off on AVX-512 runners show that the difference comes from the CPU, not from this change. This is an open issue for the video profile, which is not Stable: its G2 depends on the runner's instruction set (`release-gates-video/`).

## 5. Latency (development numbers, uncontrolled shared workstation)

These are i7-5820K numbers at 6 threads with a load average of 3 to 6 from other processes, so they are not G5 evidence. Raw results are in `g5-dev/`.

- **Interleaved A/B** (`ab-text-on/off`; three alternating runs of each, the same build, `--cpu-arena on|off`). Arena off against on: short-b1 34.6 → 35.0 ms (+1.2%), short-b8 306 → 311 ms (+1.6%), words100-b1 157.8 → 161.0 ms (+2%).
- **Matched protocol, one round** (`run-g5-matched.sh eg2-text cpu … 1 6 3`, both sides). C# over Python warm-median ratios: arena on 0.94 to 1.01, arena off 0.98 to 1.07 (short-b1 1.069). Both are within the 1.10 budget. The single-round short-b1 gap is mostly run-to-run noise; the interleaved A/B above measures the arena's cost.
- **Video**, run per workload because one command is limited to 10 minutes (`video-fix`, `video-base`, `video-py`). Ratios: video-1080p-b1 0.995 (arena on 0.985), video-b1 1.048 (arena on 1.027; Python n=12 with a wide interval). Preprocessing share 0.001 to 0.002. Cold start: C# 9.9 s, Python 14.3 s. video-32-frames-b1, video-b4 and video-text-interleaved were not run, because each exceeds the 10-minute limit.

## Also in this change

- The soak tool records `mallinfo2` with every smaps dump and dumps at the loaded point as well (`--smaps-every`). It also prints and records the sessions' memory settings, and has the diagnostics flags `--cpu-arena` and `--no-cpu-arena`.
- The benchmark harnesses accept `--cpu-arena on|off` for A/B runs.
- `qualification.yml` diagnostics variant parts: `smaps5`, `keep-loaded`, `cpu-arena` and `no-arena`. The THP precondition is now checked after the write. The growth rule (`Analysis.cs`) is unchanged.

## Reproduce

```sh
gh workflow run qualification.yml --ref <branch> -f soak_only=true -f soak_duration=15m \
  -f "soak_workloads=embeddinggemma2 embeddinggemma2-video" \
  -f 'soak_variants=["smaps5","cpu-arena+smaps5","keep-loaded+smaps5","keep-loaded+cpu-arena+smaps5"]'
python3 -I docs/evidence/eg2-hosted-soak/analysis/smaps_summary.py <x.smaps.jsonl>
python3 -I docs/evidence/eg2-hosted-soak/analysis/arenas.py <x.smaps.jsonl> unloaded
gcc -O0 -fno-builtin -pthread -o /tmp/t docs/evidence/eg2-hosted-soak/analysis/trimtest3.c && /tmp/t 20
```

The `runner-15m-38863883673/` reports name the arena flag by its development spelling. `no-arena` there is today's `--no-cpu-arena`, and `arena-shrink` was a temporary option that has been removed.
