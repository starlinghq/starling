# Proposed: EmbeddingGemma 2 video release fixture (P-030), Wikimedia Commons videos

Status: **proposed**. This fixture is not accepted and is not under `model-pipeline/protected/`. Intended use: the held-out G1/G2 video release set for `embeddinggemma-2/video-fp32-cpu` and `video-fp32-cuda`. If accepted, the files move byte-identical to a protected release directory in a `protected-change` pull request. No .NET run has used these files.

## Media

The candidates were the 6 videos of the Stable proposal's `embeddinggemma-2/media/media.json` (PR #20, SHA-256 `7501f65cfb6d2586305f9efe76f512babbaf999d8fb9301405d87db80b3b7198`). `select_media.py` drew them from `Category:Videos by NASA` by seeded sample, and no agent chose any video. **No media bytes are committed.** `videos.json` pins each file by its Commons URL, revision, SHA-1 and SHA-256. The script below regenerates the files and the manifests, with the license check described in `../image-fp32-cpu-release/PROPOSED.md`:

```sh
python -I model-pipeline/reference/embeddinggemma2/fetch_eg2_release_media.py --media <media.json> --out <media dir>
```

`videos.json` here is that script's output minus the two excluded videos below, which are listed under `excluded` with the reason.

License check (Commons `extmetadata` and description-page templates, 2026-10-14, `licenses.json`): **all 6 passed; none was rejected for its license.** Licenses as Commons states them (data, not a legal review; the maintainer confirms each):

| Id | File (description page) | Revision | License (page templates) | Author / credit (Commons `Artist`) | Bytes | SHA-256 |
| --- | --- | --- | --- | --- | ---: | --- |
| `commons-video-00` | [Aquarius flat 2048x1024.ogv](https://commons.wikimedia.org/wiki/File:Aquarius_flat_2048x1024.ogv) | 2014-01-28T06:24:44Z | Public domain (PD-USGov-NASA) | The mission is a collaboration between NASA and the Space Agency of Argentina | 10,573,931 | `87d36a705ffb2f1957f103fbe4a4982fe0dc14bcc5564e5f7c427e3d90af7e34` |
| `commons-video-01` | [Columbia Glacier Retreat.ogv](https://commons.wikimedia.org/wiki/File:Columbia_Glacier_Retreat.ogv) | 2012-06-15T09:30:47Z | Public domain (PD-USGov-NASA) | NASA/Goddard Space Flight Center | 3,741,798 | `d86a2819d465cd7f390b14a948d3af515f6d6db88f5ed1b74c63547dfa57af49` |
| `commons-video-02` | [Cyclone Fay rainfall.ogv](https://commons.wikimedia.org/wiki/File:Cyclone_Fay_rainfall.ogv) | 2012-09-08T07:45:19Z | Public domain (PD-USGov-NASA) | Hal Pierce | 2,125,307 | `0b1ce456249d9a355f5ec2915e3f53f174cc45cf6a897321dc2780c7e9470e5b` |
| `commons-video-03` | [Evolution of globular clusters.ogv](https://commons.wikimedia.org/wiki/File:Evolution_of_globular_clusters.ogv) | 2012-12-20T12:31:54Z | CC BY 4.0 (Cc-by-4.0) | ESO/NASA/ESA, L. Calçada, F. Ferraro (University of Bologna) | 2,663,269 | `a99a040063dd757e4602175af417580d76f6cc4f55642b8505adcb978ffc9c9d` |
| `commons-video-04` | [GPM scans hurricane Hermine.webm](https://commons.wikimedia.org/wiki/File:GPM_scans_hurricane_Hermine.webm) | 2016-09-21T08:20:43Z | Public domain (PD-USGov-NASA) | NASA's Scientific Visualization Studio | 6,595,291 | `dc4c6dc8c56f291b6ad16fff9abc6a0ef4ddd81fe63ef12f36d5437427959202` |
| `commons-video-05` | [LDCM spacecraft rotating.webm](https://commons.wikimedia.org/wiki/File:LDCM_spacecraft_rotating.webm) | 2014-06-21T01:16:36Z | Public domain (PD-USGov-NASA) | NASA/Goddard Space Flight Center Conceptual Image Lab | 17,279,454 | `52278014faee688d8234fdf70f7c716a51fe827ad4f0de8e454f4b955db4ee0a` |

**Two videos excluded (decoder, not license).** `commons-video-01` (Columbia Glacier Retreat) and `commons-video-03` (Evolution of globular clusters, the only CC BY video) cannot be opened by the upstream decoding path. torchcodec 0.17.0 `VideoDecoder(path, seek_mode="exact")` on FFmpeg 8.1.3 is the call transformers 5.19.0 `fetch_videos` makes, and it fails with "Could not seek file to pts=0: Operation not permitted ... the 'ogg' format does not support seeking". `seek_mode="approximate"` opens both files, but upstream does not use it, so there is no reference to compare against. The set is therefore 4 videos, all public domain (`PD-USGov-NASA`): two Ogg Theora (`-00`, 2048x1024, 12 s; `-02`, 680x488, 11 s) and two WebM VP8 (`-04`, 1920x1080, 45 s; `-05`, 1280x720 at 59.94 fps, 17 s).

Decoder: torchcodec 0.17.0 on the pinned **FFmpeg 8.1.3 LGPL shared build** (BtbN autobuild `n8.1.3-17-gc389840894`, from `fetch_eg2_ffmpeg.sh`, SHA-256-checked). It was loaded through `LD_LIBRARY_PATH` with `EG2_TORCHCODEC_FFMPEG_MAJOR=8`, so the system FFmpeg 9 was refused. This is the same decoder path as the development fixtures. `video-environment.json` records the library versions.

## Rows (`video-cases.json`)

There is one path row per video, plus seven rows from `video-cases.json`. These rows add no natural-language text:

- `vid-query-prompt` (`-00`) and `vid-document-prompt` (`-05`): the model's own query and document prompts.
- `mm-two-videos`: `<|video|> <|video|>` with `-02` and `-05`.
- `mm-image-and-video`: `<|image|> <|video|>`, with frame 100 of `-02` as the image.
- `vid-array-with-metadata`, `vid-array-no-metadata` and `vid-decoder-object` on `-05`: the decoded-array and decoder-object input forms.

The batch check uses `-00`, `-02`, `-04` and `-05`; the eager check uses `-00`, `-02` and `-04`. There are no text queries, so the query-by-video table is empty.

**Finding (Ogg Theora seek).** On `commons-video-02`, torchcodec returns a different frame 24 for `get_frames_at` (seek, which upstream uses) than for sequential decoding (`get_frames_in_range`). The other 10 sampled frames are identical. That is why the decoded-array rows are on the WebM `-05` and not on `-02`; with `-02`, the runner's restated-preprocessing check stopped. `-00` (Theora) and `-05` (VP8) show no difference. Reproducer (with the pinned FFmpeg 8 on `LD_LIBRARY_PATH`): open `commons-video-02.ogv` twice with `VideoDecoder(..., seek_mode="exact")` and compare `get_frames_at([24])` with `get_frames_in_range(0, 266)[24]`. Consequence for the .NET port: a decoder that reads Theora sequentially, rather than by seeking as torchcodec does, can differ from this fixture on that frame. Before acceptance, the maintainer may prefer to replace `-02` or drop Ogg from the release set.

## Files

| File | SHA-256 | Content |
| --- | --- | --- |
| `licenses.json` | `83efc78e2df0a893b4efbfe841a592efcb2b88e6c5a98701b56aa871dcdf8625` | Commons extmetadata and page license templates of all 22 media files, checked 2026-10-14 |
| `tensors.json` | `9d2ec884fbb34a0213e14b8fd12ee661703af5f29da62367d15e7b34d36082fe` | digests of the per-row stage tensors (not committed) |
| `video-cases.json` | `584a98b67258d1ed6bc0e634b8a50c70c382f65ca955992f08627f8858d5b0ae` | the extra rows and check lists (`--cases`) |
| `video-embeddings.jsonl` | `a9c69aaa2bf91df5497ba2572bb43cef93c1b20415a22186eb39756a8e49d1ab` | per row: rendered ids, per-video metadata, sampled indices, frame and stage digests, fp32 embedding |
| `video-environment.json` | `71234cca1327c5dd0cf6268dac0bb611d47f21ee970b2678873a1b9ae29f7417` | runner commit, packages, FFmpeg libraries, asset and media digests |
| `video-frames.json` | `b5eacc892bd565e7f24802bf024913ad46f4a46d6dc717d23d46ccfe0875b968` | per video: torchcodec metadata, sampled indices, SHA-256 of every decoded frame |
| `video-reference.json` | `7cbc33345d67cbc3dbd2152eb921cadce44c76d3a43c887db90f2c9989656e8c` | reference stability (batch, eager, vision path, input forms), quirks |
| `videos.json` | `71f059885ccde317b7c27731b596243fd5c4597531f6c116e64f7a6ae4caac0c` | media manifest of the four videos used, plus the two excluded with the reason |

`tensors.json` lists the digests of the per-row `.npz` stage tensors (decoded and resized frames, positions, soft tokens, token embeddings), which are large and not committed.

## Reference

`reference/embeddinggemma2/run_embeddinggemma2_video.py --cases video-cases.json` ran at runner commit `b98c9ce` on PyTorch 2.14.1 CPU fp32 (6 threads), sentence-transformers 6.1.0, transformers 5.19.0 and torchcodec 0.17.0. It used `google/embeddinggemma-2` at `914f7f89142e33e77833254d9c9b90c3cef7303b`, whose assets matched the protected development digests. One model process ran at a time, with no GPU, in about 15 minutes.

```sh
LD_LIBRARY_PATH=<FFmpeg 8.1.3 LGPL>/lib <eg2-video venv>/bin/python model-pipeline/reference/embeddinggemma2/run_embeddinggemma2_video.py \
  --model-dir <snapshot> --revision 914f7f89142e33e77833254d9c9b90c3cef7303b --videos <media dir without -01, -03> \
  --cases model-pipeline/proposed/stable/embeddinggemma-2/video-fp32-cpu-release/video-cases.json --out <out> --tensors <tensors>
```

Findings (development values in parentheses):

- 11 rows, no NaN or Inf values. 11 to 32 sampled frames per video, with 120 to 130 soft tokens per frame. The longest row has 3,906 tokens.
- Restated sampling and preprocessing equal the processor on every row.
- Batch of 4 vs single: 8.9e-8 (8.9e-8).
- Restated vision path on the soft tokens: 9.5e-7 padded (1.9e-6), 1.6e-4 unpadded (1.5e-4), 2.3e-4 one frame at a time (1.6e-4).
- Eager vs sdpa attention: **3.6e-6 (4.8e-7)**. This is larger than in development because the sequences are longer (up to 3,906 tokens); it is a reference-side measurement, not a tolerance.
- Path vs decoded array with metadata: 0. Path vs decoder object: 0.
- Path vs decoded array without metadata (upstream keeps every frame, then the 32-frame cap): cosine 0.970 (0.965). This is the known upstream quirk.
- `infer_modality` treats `.webm` and `.m4v` paths as text (the known quirk). The runner passes `{"video": path}` for those.

## Decisions for the reviewer

1. Confirm the 4 licenses used (all public domain) and accept the two decoder exclusions.
2. Decide on the Ogg Theora seek finding: keep `commons-video-02` (seek-based decoding is the contract) or replace it.
3. Accept the set as the P-030 video release set, with the seven extra rows.
