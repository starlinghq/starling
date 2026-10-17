# EmbeddingGemma 2 image soak: what the loaded series holds, two native heap trims, and the managed residue

After #86 (dedicated run threads), the one-hour hosted soak of `embeddinggemma-2/image-fp32-cpu` (Stable) still had a noisy **loaded** series. Run [38967975211](https://github.com/starlinghq/starling/actions/runs/38967975211) (Xeon 8573C) failed: loaded `rss_anon` growth 66.1 MiB against a limit of 53.9, halves 51.1 and 42.1. Passing runs had wide halves too (38975054175: 38.3/53.6, halves 10.7 and 67.4; 38933567784: 52.6/55.1, halves 78.5 and -4.0).

**Status: not resolved.** Two library changes remove the native heap components of the noise, verified in the dumps. A managed component (GC committed memory, large object heap fragmentation) remains. With the two changes the image soak passed 4 times out of 4, but one pass was 37.3 against a limit of 41.3, both halves positive. In the same run one video soak failed. Details in sections 4 and 5.

## 1. What the loaded series measures

`Starlings.Soak.Tool` (`Program.cs`, `SoakRunner.Run`) runs one load/unload cycle after another for the hour. Each cycle:

1. loads the model: `EmbeddingGemma2MultimodalEmbedder.Create`, which creates two **new** ONNX Runtime sessions (vision and multimodal text) on the soak's main thread;
2. serves 5 requests from 2 parallel callers (thread-pool tasks); every 5th request is a deadline request (a 6-image batch cancelled after 5 to 150 ms);
3. takes the **loaded** sample: two forced compacting full GCs, then `RssAnon` from `/proc/self/status`, with the model still loaded;
4. disposes the embedder (the runtime calls `malloc_trim(0)` when each session is released, #66);
5. takes the **unloaded** sample the same way.

So the loaded series is not one session kept alive: every point comes from a fresh pair of sessions, after that cycle's requests. One session kept alive for the whole hour is the `--keep-loaded` diagnostic. `loaded - unloaded` is that cycle's load plus whatever its runs left behind. The soak feeds synthetic images (`EmbeddingGemma2Image.FromRgb24`), so SkiaSharp is not involved. Per request, the managed work is the resampler and the patch tensors.

## 2. Diagnosis: run 38983201554 (`soak_only`, image only, four variants)

New in the soak tool for this diagnosis: each `--smaps-every` dump now also records glibc's `malloc_info` (per arena: memory from the system, free chunks by size), and the GC's committed bytes, heap size and fragmentation (`GCMemoryInfo`). There is also a diagnostic `--trim-after-load` (workflow variant part `load-trim`): `malloc_trim(0)` right after each load. Dumps: `diagnosis-38983201554/`. Tables: `analysis/dumps.py <dump> loaded` and `analysis/decompose.py loaded <dumps>`.

| Variant | Runner | Loaded growth / limit [halves] | Loaded median |
| --- | --- | --- | --- |
| `smaps5` | EPYC 7763 | 77.5 / 53.3 [38.3, -14.8] | 2665 |
| `default+smaps5` (same configuration) | EPYC 7763 | -12.5 / 54.3 [-51.7, 32.0] | 2715 |
| `load-trim+smaps5` | Xeon 8370C | 23.4 / 45.8 [32.7, 4.5] | 2290 |
| `smaps5+load-trim` | EPYC 7763 | 29.0 / 47.4 [35.1, -1.8] | 2368 |

The two identical configurations on the same CPU model gave -12.5 and 77.5: that spread is the noise this change goes after.

A loaded sample (`smaps5`, cycle 10, MiB): 2,649 anonymous resident in total. The main glibc heap holds 1,582 of it, with 405 MiB free inside it. The two run-thread arenas hold 171 and 63 MiB, all of it free. Other anonymous mappings hold 881: 512 in mmapped malloc chunks (model weights) and the GC's regions (committed 318). Three components move from load to load:

1. **Main heap: about 400 MiB freed during session creation stays resident.** Creating the two sessions on the main thread allocates and frees graph transformation and initializer copies. The bytes in use are the same every cycle (main arena `system - free` = about 1,190 MiB), but resident memory is 1,533 to 1,650 MiB depending on the load. With `load-trim` the main heap is **exactly 1,192 MiB in every dump** on the Xeon run.
2. **Run-thread arenas: free chunks after the runs, rising in steps.** The two dedicated run threads' arenas hold 60 to 190 MiB each, spread over 2 to 3 sub-heaps, and almost all of it is free chunks in the unsorted and large bins, not the top chunk (`malloc_info` of arena 6 at cycle 60: 186 MiB from the system, 186 MiB free, of which 125 MiB is unsorted and 60 MiB is in 7 to 22 MiB chunks). How much a run leaves depends on how its buffers fragment the heap, so the high-water mark rises in steps at random times. In `smaps5` the second run thread went from 63 to 157 MiB at cycle 75, and the arena class gained 98 MiB over the window with both halves positive (`decompose.py`: arena 98.2 [59.6, 42.8]). That is the failure shape of 38967975211. `malloc_trim` releases free chunks inside every arena (it cannot release a non-main arena's top), so it covers this memory.
3. **Managed heap: GC committed memory swings by 60 to 90 MiB.** After the two full GCs, the large object heap holds 260 to 340 MiB and its fragmentation is 7 to 80 MiB (`gc_kb.generations[3]`). Committed memory follows: 280 to 370 MiB. The LOH is not compacted. The EmbeddingGemma 2 path allocates large per-request arrays (`EmbeddingGemma2ImageProcessor` pixel values, the resampler's output, `soft[b] = ...ToArray()` per image), and `ArrayPool<T>.Shared` drops idle buffers on gen-2 collections. Both move the LOH's live size and fragmentation from cycle to cycle.

The mmapped chunks (512 MiB) and the remaining anonymous mappings are flat (`mmap - gc_com` within 2 MiB in every run).

## 3. Changes

`Starlings.Runtime.Onnx`: two new session options, `bool?`. `null`, the default, means **off** for sessions created directly and lets a model library choose. Both call `NativeHeap.Trim()`, the same `malloc_trim(0)` as the release trim (#66): it returns only free memory, changes no allocator setting, is a no-op off Linux glibc, and is turned off for the whole process by the `AppContext` switch `Starlings.Runtime.Onnx.DisableNativeHeapTrim`. `OnnxSessionDiagnostics` reports both.

- `TrimNativeHeapAfterLoad`: one trim at the end of `OnnxModelSession.Create`.
- `TrimNativeHeapAfterRun`: one trim on the thread that ran (the dedicated run thread when one is used), after each run returns. The trim is in the run, so its cost (milliseconds) is part of the request. The next run faults the pages in again.

`Starlings.Models.EmbeddingGemma2`: `EmbeddingGemma2MultimodalEmbedder.Create` turns both on for its two CPU sessions (vision and multimodal text) unless the caller set them. That covers the image and video paths. The text encoder (`EmbeddingGemma2Encoder`, text-only soak with large margins) and CUDA sessions are unchanged. No `mallopt`, no `MALLOC_*` setting, no change to contracts, growth rules, limits or budgets.

Embeddings are bit-identical with and without the trims (`CpuArenaTests.ImageEmbeddingsAreBitIdenticalWithAndWithoutTheNativeHeapTrims`, local). Latency (hosted, same CPU model): image request p50 on EPYC 7763 was 18.0 to 18.3 s without the trims and 18.0 to 18.1 s with both; load median 2.93 to 3.06 s without, 3.09 to 3.12 s with both.

Soak tool: `--load-trim`/`--no-load-trim` and `--run-trim`/`--no-run-trim` override the library choice (workflow variant parts `lib-load-trim`, `no-load-trim`, `run-trim`, `no-run-trim`).

## 4. Verification run 38988962141 (`soak_only`, image and video, four variants)

Code at d579809 (both trims). Format: growth / limit [halves], MiB, loaded `rss_anon`; unloaded where it matters. Dumps: `ab-38988962141/`.

| Soak | Runner | Verdict | Loaded | Loaded median | Request p50 |
| --- | --- | --- | --- | --- | --- |
| image | EPYC 9V74 | pass | -18.0 / 41.4 [-38.8, -32.0] | 2068 | 15.6 s |
| image `smaps5` | EPYC 7763 | pass | 26.7 / 40.7 [36.8, -42.6] | 2037 | 18.0 s |
| image `default+smaps5` | EPYC 7763 | pass (unloaded `retention` 41.6) | 37.3 / 41.3 [17.8, 19.2] | 2064 | 18.1 s |
| image `no-run-trim+smaps5` (load trim only) | EPYC 7763 | **FAIL** (loaded `working_set` 60 MB) | 57.7 / 46.3 [29.2, 11.2] | 2316 | 18.1 s |
| video | EPYC 9V45 | pass | -17.6 / 42.8 [-16.9, -66.2] | 2141 | 5.6 s |
| video `smaps5` | Xeon 8573C | pass | -1.3 / 42.4 [-31.3, 10.8] | 2118 | 8.8 s |
| video `default+smaps5` | EPYC 7763 | **FAIL** | 69.4 / 43.7 [53.6, 35.6]; unloaded 60.5 / 32 [54.8, 29.9] | 2186 | 13.0 s |
| video `no-run-trim+smaps5` | EPYC 9V74 | pass | -52.6 / 49.2 [-31.9, -46.1] | 2460 | 11.5 s |

Components with both trims (`decompose.py`, loaded): the main heap is flat at 1,192 MiB in every dump of every trimmed soak. The run-thread arenas drop from 120 to 350 MiB resident to 10 to 90 MiB. What is left is one top chunk of up to about 65 MiB, which `malloc_trim` cannot release (image `default+smaps5`: 12 to 89). The GC's committed memory is unchanged and now dominates:

| Soak (both trims) | Loaded anon | heap | arenas | GC committed |
| --- | --- | --- | --- | --- |
| image `default+smaps5` | 47.2 [13.0, 35.1] | 0.0 | 38.4 [33.6, -1.2] | 8.9 [-23.1, 37.3] |
| image `smaps5` | 12.4 [8.5, -10.7] | 0.0 | -17.0 | 29.4 [1.4, 5.7] |
| video `default+smaps5` | 49.3 [42.5, 34.8] | 0.0 | 8.0 | 40.2 [28.8, 32.7] |

In the failed video soak, LOH fragmentation rises from 50 to 131 MiB over the hour and GC committed from 340 to 424 MiB. Its unloaded series grows by the same amount, so this is managed memory that stays across loads.

## 5. Left open

- **Managed LOH fragmentation in the image and video paths.** This is the remaining source. It is not new: the pre-#86 video dumps (`../eg2-soak-reliability/verification-38924489084/embeddinggemma2-video-smaps5`) show the same drift in the class that holds the GC regions (loaded +21.4 [14.3, 33.4], unloaded +26.1 [20.1, 33.7]). What changed is the limit. The limit is `max(32, 2% of the loaded median)`, and the trims lowered the median by about 600 MiB, so the image limit fell from about 54 to 41 MiB and video's from about 55 to 43. The managed swing stayed the same size, so it now uses most of the margin. Two possible fixes for a maintainer to choose between: per-embedder reuse of the large per-request buffers (pixel values, resampler output, soft-token copies, video soft tokens), or accepting the managed swing. Both change the core model path, so neither was attempted here (two repair attempts made: section 3).
- **Run-thread top chunks.** Up to about 65 MiB per run-thread arena stays resident as the top chunk. `malloc_trim` does not release a non-main arena's top.
- Not run: the three full qualification runs (`skip_cuda=true`) that the goal asks for. They cannot meet it while video `default+smaps5` fails and image passes by 4 MiB.

## Reproduce

```sh
gh workflow run qualification.yml --ref <branch> -f soak_only=true -f skip_cuda=true -f soak_duration=1h \
  -f "soak_workloads=embeddinggemma2-image embeddinggemma2-video" \
  -f 'soak_variants=["default","smaps5","default+smaps5","no-run-trim+smaps5"]'
python3 -I docs/evidence/eg2-image-loaded-soak/analysis/dumps.py <x.smaps.jsonl.gz> loaded
python3 -I docs/evidence/eg2-image-loaded-soak/analysis/decompose.py loaded <x.smaps.jsonl.gz>...
```
