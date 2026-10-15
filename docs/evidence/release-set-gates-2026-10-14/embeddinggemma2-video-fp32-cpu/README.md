# embeddinggemma-2 `video-fp32-cpu` on the P-030 video release set

Run on 2026-10-14 at `cc59fee`. The set is `embeddinggemma-2/release/video-fp32-release`: 4 Wikimedia Commons videos and 11 rows. Decoding used the C# FFmpeg 8 adapter on the pinned FFmpeg `n8.1.3-17-gc389840894` LGPL shared build. That build came from `fetch_eg2_ffmpeg.sh`, and the archive SHA-256 `bf3db18a…` matches the script. The evaluator verified the protected evidence against Git. Nothing under `protected/` was changed.

| Gate | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- |
| G1 and G2 (rerun) | **passed** | see "Rerun after the adapter decodes as torchcodec does" below | [`g1-g2-video-release-torchcodec-seek/report.md`](g1-g2-video-release-torchcodec-seek/report.md) |
| G1 (first run) | **failed** | rendered ids exact 11/11; frame indices exact 12/12; positions exact 12/12. **Decoded frames, resized pixels and pixel values differ on 3/12 video inputs**, all of them `commons-video-02` (Ogg Theora), and on the still image `mm-image-and-video` takes from the same file. `decoder_ffmpeg.decoded_rgb_vs_torchcodec` fails on the same inputs. | [`g1-g2-video-release/report.md`](g1-g2-video-release/report.md) |
| G2 (first run) | **failed** | embedding max abs 8.88e-3 (1e-5), min cosine 0.99793 (0.9999995), worst `mm-image-and-video`. Every row without `commons-video-02` is within the bound (measured per row from the stage results: 1.9e-7 to 1.2e-6). The rows with `-02` are `commons-video-02` 5.6e-3, `mm-two-videos` 2.6e-3 and `mm-image-and-video` 8.9e-3. Batches vs single 0. Soft-token and token-embedding checks not run: the regenerated reference tensors do not match `tensors.json` (as for images: 4 threads instead of the reference's 6). | same |

## Finding: Ogg Theora frames differ from torchcodec's seek-based decode

The release set's own `SOURCE.md` records this risk. On `commons-video-02.ogv`, torchcodec returns a different frame 24 for `get_frames_at` (which seeks, as upstream does) than for sequential decoding. A decoder that reads Theora sequentially can therefore differ from the fixture on that frame. The C# FFmpeg adapter differs from the fixture on exactly the inputs that sample `commons-video-02`. The other Theora file (`-00`) and both WebM files are exact. This is an implementation-versus-reference mismatch on accepted release data, so it is reported as a failure and not repaired here. Minimal reproducer, from `SOURCE.md`: with the pinned FFmpeg 8 on `LD_LIBRARY_PATH`, open `commons-video-02.ogv` with torchcodec `VideoDecoder(..., seek_mode="exact")` and compare `get_frames_at([24])` with `get_frames_in_range(0, 266)[24]`. Then compare with the C# adapter's frames, which are hashed per frame in `tokenizer.results.json` (`video<k>_frame_sha256`). The maintainer's options are those `SOURCE.md` lists: replace `-02`, drop Ogg from the release set, or make the adapter seek as torchcodec does.

The regenerated reference matches the protected `video-frames.json` byte for byte (`b5eacc89…`), so the reference side reproduces exactly, and the difference is on the C# side.

## Commands

```sh
REL=model-pipeline/protected/embeddinggemma-2/release/video-fp32-release
FF=<FFmpeg n8.1.3 LGPL shared>/lib   # fetch_eg2_ffmpeg.sh, archive SHA-256 bf3db18a4a4bdc066f6b74ee5ec3d7b7be7d6738dcfd7162eae3e09cb857c6b2
# media dir: the 4 videos (hard links from the fetch_eg2_release_media.py download, SHA-256 equal to videos.json) plus the protected videos.json
LD_LIBRARY_PATH=$FF EG2_TORCHCODEC_FFMPEG_MAJOR=8 OMP_NUM_THREADS=4 <eg2-video venv>/bin/python model-pipeline/reference/embeddinggemma2/run_embeddinggemma2_video.py \
  --model-dir <snapshot> --revision 914f7f89142e33e77833254d9c9b90c3cef7303b --videos <media dir> --cases $REL/video-cases.json --out <ref> --tensors <ref>/tensors
dotnet ai-dotnet/tools/Starlings.Fidelity.Tool/bin/Release/net10.0/Starlings.Fidelity.Tool.dll embeddinggemma2-videos --out <run>/stages \
  --videos <media dir> --ffmpeg $FF --fixtures $REL --threads 4
model-pipeline evaluate model-pipeline/registry/embeddinggemma-2/video-fp32-cpu/manifest.yaml --fixtures $REL --results <run>/stages --out <report> --require-gates G1,G2
```

Wall clock (shared machine, not benchmarks): reference regeneration 2,045 s; .NET dump 1,147 s.

## Rerun after the adapter decodes as torchcodec does (2026-10-14): G1 and G2 passed

**What torchcodec does with `commons-video-02`.** torchcodec 0.17.0 `get_frames_at` (`SingleStreamDecoder`, exact mode, read from the source at commit `d9dc54238e173000bc3969eb3959777026ff9ae8`) decodes the sorted indices on the demuxer that made the packet scan. It seeks back to 0 after the scan with `avformat_seek_file(ctx, 0, INT64_MIN, 0, 0, 0)`. For each index that does not follow the previous one it sets a cursor at the scanned pts. It reaches the cursor either by decoding on or by `avformat_seek_file` to the scanned keyframe at or before it, followed by `avcodec_flush_buffers`. A seek is avoided when the target shares the last frame's keyframe, or when the keyframe lies within `has_b_frames + thread_count - 1` frames of it (FFmpeg frame threading). It returns the first frame whose pts is at or after the cursor. `commons-video-02` has a keyframe every 12 frames, and the sampled indices are 0, 24, 48 and so on (1 fps at 24 fps). From frame 0 to 24 the gap exceeds the in-flight frames (`num_ffmpeg_threads=0`, 12 CPUs), so torchcodec seeks. The Theora frame decoded after that seek and flush differs from the sequential frame 24, by mean 6.3 levels per channel. Later seeks (48 and on) return the sequential frames.

The output is deterministic: five runs of the fixture's call gave the same frames, and so did `num_ffmpeg_threads=4`. It depends on the decoder's frame threading, though. With `num_ffmpeg_threads=1`, frame 24 equals the sequential frame and frames 48 and on differ. So the reference value is upstream's seek-and-flush behaviour on this FFmpeg build and thread configuration, not random. The old adapter decoded sequentially, which explains the first run's failure.

**Change.** The FFmpeg adapter (`EmbeddingGemma2VideoDecoder`, both FFmpeg majors) now makes torchcodec's calls in the same order: the same scan table, the seek back to 0 on the same demuxer, other streams discarded, the same seek-avoidance rule read from the open codec context (`has_b_frames`, `thread_count`, `active_thread_type`), and the same seek, flush and pts filter. Its default `DecoderThreads = 0` matches upstream's `num_ffmpeg_threads=0`. The 22 development videos and the 9 edge-case videos (rotation, HDR PQ and HLG, interlaced, VFR) stay exact on FFmpeg 8.1.3 and on FFmpeg 9.0.1 (every frame and every sampled frame). A new test checks the 4 release videos against the protected per-row frame digests, and it checks that the sequential decode of `-02` still equals `video-frames.json`.

**Reference tensors.** The first run's soft-token and token-embedding checks did not run because the reference tensors were regenerated at 4 threads. Regenerated at the recorded 6 threads (`OMP_NUM_THREADS=6`, the `eg2-video` environment, FFmpeg 8.1.3 LGPL), `tensors.json`, `video-embeddings.jsonl` and `video-frames.json` are byte-identical to the protected files (`9d2ec884…`, `a9c69aaa…`, `b5eacc89…`), and the evaluator checked every `.npz` against `tensors.json`. The run's optional stability section was then killed for lack of memory on the shared machine; the stability figures come from the protected `video-reference.json` and gate nothing.

| Gate | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- |
| G1 | **passed** | ids 11/11; frame indices, decoded frames, resized pixels, pixel values and positions exact 12/12; still image exact; decoder frames vs torchcodec exact 12/12 (FFmpeg n8.1.3-17-gc389840894) | [`g1-g2-video-release-torchcodec-seek/report.md`](g1-g2-video-release-torchcodec-seek/report.md) |
| G2 | **passed** | embedding max abs 1.29e-6 (1e-5), worst `mm-two-videos`; min cosine 0.999999999959 (0.9999995); soft tokens 2.37e-4 row-relative (1e-3); token embeddings 3.81e-4 (1e-3); batches vs single 0 (1e-5); provider options as bound | same |

Run at `a8717f6` (branch `eg2-video-torchcodec-seek`, rebased on main after #38) on CPU. Bounds come from the profile's protected `tolerances.yaml`; nothing under `protected/` changed. The report's date is UTC (2026-10-15). Wall clock for the .NET dump: 1,446 s (shared machine, not a benchmark); reference regeneration about 9 minutes to the tensors.

