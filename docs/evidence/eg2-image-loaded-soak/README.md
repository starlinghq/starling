# EmbeddingGemma 2 image and video soaks: what the loaded series holds, native heap trims, reused request buffers

After #86 (dedicated run threads), the one-hour hosted soak of `embeddinggemma-2/image-fp32-cpu` (Stable) still had a noisy **loaded** series. Run [38967975211](https://github.com/starlinghq/starling/actions/runs/38967975211) (Xeon 8573C) failed: loaded `rss_anon` growth 66.1 MiB against a limit of 53.9, halves 51.1 and 42.1. Passing runs had wide halves too (38975054175: 38.3/53.6, halves 10.7 and 67.4; 38933567784: 52.6/55.1, halves 78.5 and -4.0).

**Status: resolved.** The noise had three sources, verified in the dumps, and each has a library-level fix. The main glibc heap gets a trim after session creation, the run-thread arenas get a trim after each run, and the managed large object heap gets per-embedder reuse of the large request buffers. In three consecutive full qualification runs, all 12 image and video soaks passed (at most 71% of the limit), and so did every other soak (section 6).

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

## 5. Managed side: reused request buffers (per-embedder pool)

After the trims, the GC's committed memory and large-object-heap (LOH) fragmentation were the remaining source. The LOH is not compacted. Every image request allocated and freed arrays above the 85 KB threshold:

- the resized pixels: the resampler's result, its horizontal pass, the copy made when no resize is needed, and the weight tables of strong downscales;
- the patch values and positions;
- one `ToArray` copy of each image's soft tokens;
- for videos, one soft-token copy per frame, the `videoSoft` concatenation and a `List<float>` copy of it per row.

The vision and multimodal graph buffers came from `ArrayPool<T>.Shared`, which drops idle arrays on gen-2 collections, so the soak's full GCs moved that memory too.

`Starlings.Models.EmbeddingGemma2`: new internal `EmbeddingGemma2BufferPool`, owned by each `EmbeddingGemma2MultimodalEmbedder` and dropped on `Dispose`.

- **No shared buffers.** A request rents each buffer for its own use and returns it when done, so concurrent callers never share one. The pool holds only buffers that no request is using. A lock guards the free lists.
- **Bounded size.** A rent takes the smallest free buffer of its element type that fits. If none fits, it allocates a new one at a quarter-octave size class (at most 25% over the request) and drops the largest free buffer that was too small. At most `max(16, 4 x processors)` free buffers are kept per element type. What the pool holds is therefore bounded by what concurrent requests used at the same time.
- **Fewer buffers per request.** The soft tokens of all images, or all frames, of a request now go into one pooled buffer by offset instead of one array each. A video's soft tokens are a slice of that buffer, so the per-video and per-row copies are gone. A fallback assembles them, as before, if a frame's soft-token count ever differs from the video's.
- **Public API unchanged.** `EmbeddingGemma2ImageProcessor.Preprocess` still returns fresh arrays to its callers. The embedder uses an internal pooled overload.

Measurements (local, one caller, the 8 inputs of `MeasurementInputs`: images that resize up, strongly down, horizontally only or not at all, an image with text, two images in one input, two videos; `RequestMeasurementTests`, public API only, run on 9cb3516 and on 0357079):

| | Before (9cb3516) | After (0357079) |
| --- | --- | --- |
| Output digest (SHA-256 of all 8 embeddings) | `ae521bb9...0602` | `ae521bb9...0602` (identical) |
| Managed allocation per pass of 8 requests (after the first) | 92.2 MiB | 2.8 MiB |
| Peak RSS during requests (peak reset after load) | 3,099 MiB | 2,871 MiB |
| RSS after the requests and a full GC | 2,222 MiB | 1,978 MiB |
| Request latency | same or lower on every input (busy shared workstation, not a benchmark) | |

`BufferPoolTests`: the pool's reuse, exclusive ownership under 8 parallel callers, size bound, size classes and `Dispose`. Also image and video embeddings bit-identical with and without the pool, single (twice, so the second pass reuses buffers), batched, and 8 concurrent requests on 4 callers. The whole `Starlings.Models.EmbeddingGemma2.Tests` project passes (176 tests), including the image and video parity checks against the reference.

## 6. Verification: three full qualification runs

Full qualification workflow, `skip_cuda=true`, `soak_variants=["default","smaps5"]` (every soak twice), code at 0357079. Later commits change documentation and evidence only. Format: loaded `rss_anon` growth / limit [halves], MiB. Dumps: `verification-<run>/`. Before the fix, the failing run 38967975211 had image loaded growth of 66.1 / 53.9 [51.1, 42.1].

| Run | Image | Image `smaps5` | Video | Video `smaps5` |
| --- | --- | --- | --- | --- |
| [38996280821](https://github.com/starlinghq/starling/actions/runs/38996280821) | 6.5 / 40.2 [8.5, -5.6] EPYC 9V45 | 28.5 / 40.4 [9.3, 10.6] EPYC 7763 | 14.1 / 42.2 [13.7, 11.9] Xeon 8573C | 3.5 / 42.3 [-2.5, 0.4] EPYC 9V45 |
| [39001130485](https://github.com/starlinghq/starling/actions/runs/39001130485) | 1.9 / 40.1 [5.3, 14.0] EPYC 7763 | 20.7 / 40.4 [12.0, 29.1] EPYC 7763 | 24.4 / 42.4 [17.7, 9.5] EPYC 7763 | 12.2 / 42.0 [21.9, 15.7] EPYC 9V45 |
| [39006453578](https://github.com/starlinghq/starling/actions/runs/39006453578) | -0.6 / 40.5 [0.9, 2.9] EPYC 7763 | 10.1 / 40.4 [10.8, -6.0] EPYC 7763 | 19.4 / 42.3 [11.6, 10.8] EPYC 7763 | 5.0 / 42.8 [3.4, 2.9] EPYC 7763 |

All 12 image and video soaks pass, and so do the other 24 (e5, GLiNER2, text, Qwen 3.5, twice each per run); no series reports growth. One Qwen 3.5 unloaded series reports `retention`, which does not gate; Qwen 3.5 does not use this code. The largest image growth is 28.5 against 40.4 (71% of the limit), and the largest video growth is 24.4 against 42.4 (58%). Runs 38996280821 and 39001130485 passed every job. Run 39006453578 failed one job, the release gate of `embeddinggemma-2/video-fp32-cpu`. That is the known AVX-512 G2 deviation of the Experimental video profile on an EPYC 9V45 runner (`dotnet-ids.token_embeddings` row-relative 1.83e-3 against 1e-3; the EmbeddingGemma 2 README describes it). It is not related to this change: the embeddings are bit-identical to before (section 5).

Components in the `smaps5` dumps (`decompose.py loaded`):

- The main heap stays at 1,192 MiB.
- GC committed memory moves within 290 to 331 MiB (images) and 344 to 386 MiB (videos), against 280 to 442 before. LOH fragmentation is 2 to 3 MiB, against 7 to 131 before.
- The run-thread arenas hold 10 to 87 MiB.

What remains are single-cycle spikes of 40 to 90 MiB in the per-cycle loaded series, for example image `smaps5` in 38996280821: 2,081 to 2,110 MiB in 7 of 88 cycles, otherwise 2,000 to 2,048. Their size matches one run thread's top chunk, which `malloc_trim` cannot release. They do not persist: the next cycle is back at the base.

Request latency (median request p50 over cycles, same CPU model): image on EPYC 7763 was 17.97 to 18.13 s, against 18.0 to 18.3 s before any of these changes. Video on EPYC 7763 was 13.0 to 13.2 s, against 13.0 s with the trims alone. Load median on EPYC 7763 was 3.08 to 3.15 s, against 2.93 to 3.12 s.

## 7. Left open

- **Run-thread top chunks.** Each run-thread arena can keep up to about 64 MiB resident as its top chunk. `malloc_trim` never releases a non-main arena's top, and `free` releases it only above glibc's dynamic trim threshold. This is the source of the single-cycle spikes above. It is bounded and does not accumulate. Removing it would need an allocator setting, which is out of scope.
- **Margin.** The limit is `max(32, 2% of the loaded median)`. The trims lowered the median from about 2.7 GB to about 2.0 GB, so the image limit is now about 40 MiB instead of 54, and video's is about 42 instead of 55. In the three runs the worst growth is 71% (image) and 58% (video) of the new limit. The limits and the growth rule are unchanged.

## Reproduce

```sh
gh workflow run qualification.yml --ref <branch> -f soak_only=true -f skip_cuda=true -f soak_duration=1h \
  -f "soak_workloads=embeddinggemma2-image embeddinggemma2-video" \
  -f 'soak_variants=["default","smaps5","default+smaps5","no-run-trim+smaps5"]'
gh workflow run qualification.yml --ref <branch> -f skip_cuda=true -f soak_duration=1h -f 'soak_variants=["default","smaps5"]'
python3 -I docs/evidence/eg2-image-loaded-soak/analysis/dumps.py <x.smaps.jsonl.gz> loaded
python3 -I docs/evidence/eg2-image-loaded-soak/analysis/decompose.py loaded <x.smaps.jsonl.gz>...
```
