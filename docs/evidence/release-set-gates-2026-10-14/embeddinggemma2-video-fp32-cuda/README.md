# embeddinggemma-2 `video-fp32-cuda` on the P-030 video release set

Run on 2026-10-14 at `472e187` on the RTX 3090, GPU 0 only. Decoding used the same pinned FFmpeg 8.1.3 LGPL build as the CPU run ([`../embeddinggemma2-video-fp32-cpu/`](../embeddinggemma2-video-fp32-cpu/README.md)), and the graphs ran on `CUDAExecutionProvider` with `use_tf32: 0`.

| Gate | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- |
| G1 and G2 (rerun) | **passed** | see "Rerun after the adapter decodes as torchcodec does" below | [`g1-g2-video-release-torchcodec-seek/report.md`](g1-g2-video-release-torchcodec-seek/report.md) |
| G1 (first run) | **failed** | The same mismatch as on CPU: decoded frames, resized pixels and pixel values differ on the 3/12 video inputs that sample `commons-video-02` (Ogg Theora), and on the still image taken from it. Ids, frame indices and positions are exact. | [`g1-g2-video-release/report.md`](g1-g2-video-release/report.md) |
| G2 (first run) | **failed** | embedding max abs 8.88e-3 (1e-5), min cosine 0.99793, worst `mm-image-and-video`. Rows without `-02` measure 2.0e-7 to 2.8e-6 per row; rows with it measure 2.6e-3 to 8.9e-3. Soft-token and token-embedding checks not run (4-thread reference tensors). | same |

The failure comes from decoding, which runs on the CPU, so the provider does not change it. Wall clock: .NET dump 61.5 s.

## Rerun after the adapter decodes as torchcodec does (2026-10-14): G1 and G2 passed

**What torchcodec does with `commons-video-02`.** torchcodec 0.17.0 `get_frames_at` (`SingleStreamDecoder`, exact mode, read from the source at commit `d9dc54238e173000bc3969eb3959777026ff9ae8`) decodes the sorted indices on the demuxer that made the packet scan. It seeks back to 0 after the scan with `avformat_seek_file(ctx, 0, INT64_MIN, 0, 0, 0)`. For each index that does not follow the previous one it sets a cursor at the scanned pts. It reaches the cursor either by decoding on or by `avformat_seek_file` to the scanned keyframe at or before it, followed by `avcodec_flush_buffers`. A seek is avoided when the target shares the last frame's keyframe, or when the keyframe lies within `has_b_frames + thread_count - 1` frames of it (FFmpeg frame threading). It returns the first frame whose pts is at or after the cursor. `commons-video-02` has a keyframe every 12 frames, and the sampled indices are 0, 24, 48 and so on (1 fps at 24 fps). From frame 0 to 24 the gap exceeds the in-flight frames (`num_ffmpeg_threads=0`, 12 CPUs), so torchcodec seeks. The Theora frame decoded after that seek and flush differs from the sequential frame 24, by mean 6.3 levels per channel. Later seeks (48 and on) return the sequential frames.

The output is deterministic: five runs of the fixture's call gave the same frames, and so did `num_ffmpeg_threads=4`. It depends on the decoder's frame threading, though. With `num_ffmpeg_threads=1`, frame 24 equals the sequential frame and frames 48 and on differ. So the reference value is upstream's seek-and-flush behaviour on this FFmpeg build and thread configuration, not random. The old adapter decoded sequentially, which explains the first run's failure.

**Change.** The FFmpeg adapter (`EmbeddingGemma2VideoDecoder`, both FFmpeg majors) now makes torchcodec's calls in the same order: the same scan table, the seek back to 0 on the same demuxer, other streams discarded, the same seek-avoidance rule read from the open codec context (`has_b_frames`, `thread_count`, `active_thread_type`), and the same seek, flush and pts filter. Its default `DecoderThreads = 0` matches upstream's `num_ffmpeg_threads=0`. The 22 development videos and the 9 edge-case videos (rotation, HDR PQ and HLG, interlaced, VFR) stay exact on FFmpeg 8.1.3 and on FFmpeg 9.0.1 (every frame and every sampled frame). A new test checks the 4 release videos against the protected per-row frame digests, and it checks that the sequential decode of `-02` still equals `video-frames.json`.

**Reference tensors.** The first run's soft-token and token-embedding checks did not run because the reference tensors were regenerated at 4 threads. Regenerated at the recorded 6 threads (`OMP_NUM_THREADS=6`, the `eg2-video` environment, FFmpeg 8.1.3 LGPL), `tensors.json`, `video-embeddings.jsonl` and `video-frames.json` are byte-identical to the protected files (`9d2ec884…`, `a9c69aaa…`, `b5eacc89…`), and the evaluator checked every `.npz` against `tensors.json`. The run's optional stability section was then killed for lack of memory on the shared machine; the stability figures come from the protected `video-reference.json` and gate nothing.

| Gate | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- |
| G1 | **passed** | ids 11/11; frame indices, decoded frames, resized pixels, pixel values and positions exact 12/12; still image exact; decoder frames vs torchcodec exact 12/12 (FFmpeg n8.1.3-17-gc389840894) | [`g1-g2-video-release-torchcodec-seek/report.md`](g1-g2-video-release-torchcodec-seek/report.md) |
| G2 | **passed** | embedding max abs 2.77e-6 (1e-5), worst `commons-video-04`; min cosine 0.999999999818 (0.9999995); soft tokens 2.50e-4 row-relative (1e-3); token embeddings 7.97e-4 (1e-3); batches vs single 0 (1e-5); provider options as bound | same |

Run at `a8717f6` (branch `eg2-video-torchcodec-seek`, rebased on main after #38) on CUDA (RTX 3090, GPU 0 only, `use_tf32: 0`). Bounds come from the profile's protected `tolerances.yaml`; nothing under `protected/` changed. The report's date is UTC (2026-10-15). Wall clock for the .NET dump: 59 s (shared machine, not a benchmark); reference regeneration about 9 minutes to the tensors.

