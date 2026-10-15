# embeddinggemma-2 `video-fp32-cpu` on the P-030 video release set

Run on 2026-10-14 at `cc59fee`. The set is `embeddinggemma-2/release/video-fp32-release`: 4 Wikimedia Commons videos and 11 rows. Decoding used the C# FFmpeg 8 adapter on the pinned FFmpeg `n8.1.3-17-gc389840894` LGPL shared build. That build came from `fetch_eg2_ffmpeg.sh`, and the archive SHA-256 `bf3db18a…` matches the script. The evaluator verified the protected evidence against Git. Nothing under `protected/` was changed.

| Gate | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- |
| G1 | **failed** | rendered ids exact 11/11; frame indices exact 12/12; positions exact 12/12. **Decoded frames, resized pixels and pixel values differ on 3/12 video inputs**, all of them `commons-video-02` (Ogg Theora), and on the still image `mm-image-and-video` takes from the same file. `decoder_ffmpeg.decoded_rgb_vs_torchcodec` fails on the same inputs. | [`g1-g2-video-release/report.md`](g1-g2-video-release/report.md) |
| G2 | **failed** | embedding max abs 8.88e-3 (1e-5), min cosine 0.99793 (0.9999995), worst `mm-image-and-video`. Every row without `commons-video-02` is within the bound (measured per row from the stage results: 1.9e-7 to 1.2e-6). The rows with `-02` are `commons-video-02` 5.6e-3, `mm-two-videos` 2.6e-3 and `mm-image-and-video` 8.9e-3. Batches vs single 0. Soft-token and token-embedding checks not run: the regenerated reference tensors do not match `tensors.json` (as for images: 4 threads instead of the reference's 6). | same |

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
