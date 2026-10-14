# Proposed: P-030 image and video media, EmbeddingGemma 2

Status: **proposed**. Not accepted and not under `model-pipeline/protected/`. Intended use: the held-out image and video release sets of `embeddinggemma-2/image-fp32-*` and `video-fp32-*` (G1/G2), replacing the development media (`make_eg2_images.py`, `make_eg2_videos.py`) as release evidence. The media bytes are never committed; `media.json` is the record.

## What it is

`media.json` lists 16 images and 6 videos from Wikimedia Commons, each with title, description page, file revision timestamp, Commons SHA-1, size, dimensions, MIME type, duration (videos), license and author or credit as Commons reports them. It was written by `model-pipeline/reference/stable/select_media.py select` on 2026-10-14:

- Images: `Category:Featured pictures on Wikimedia Commons`, the first 100 members from each sort-key prefix `A E I M Q U` (600 members), JPEG or PNG, at most 20 MB: 137 eligible under the license allowlist, 16 drawn with `random.Random("20261014:images")`. Sizes range from 1,867x1,400 to 10,977x7,200; one PNG.
- Videos: `Category:Videos by NASA`, WebM or Ogg, at most 40 MB and 60 s: 492 eligible of 1,042 members, 6 drawn with `random.Random("20261014:videos")`. Four Ogg (`.ogv`), two WebM; 11 to 45 s; 680x488 to 2,048x1,024.

License allowlist: public domain (for example NASA and US Navy works, and photographs whose copyright has expired), CC0 and CC BY (2.0, 4.0). CC BY-SA, GFDL, NC, ND and unknown licenses were excluded, so fixtures derived from the media carry attribution obligations only. In this draw: 12 images public domain, 1 CC0, 2 CC BY 2.0 (Flickr), 1 CC BY 4.0; 5 videos public domain, 1 CC BY 4.0. Each license is Commons' own statement in the file's `extmetadata`; **the maintainer must confirm each one on its description page before acceptance** (Commons metadata is data, not a legal review).

Commons is a live site: category membership changes and a file can get a new revision. The selection is therefore reproducible from `media.json`, not by re-querying. `fetch` downloads exactly the recorded revisions and refuses any file whose SHA-1 or size differs.

```sh
python -I model-pipeline/reference/stable/select_media.py fetch --media model-pipeline/proposed/stable/embeddinggemma-2/media/media.json --out <dir>
```

`fetch` writes the files (109 MB) and `images.json` / `videos.json` with id, file, source, license and SHA-256 per file, the manifest shape the EmbeddingGemma 2 runners read. It waits 5 s between downloads (upload.wikimedia.org rate-limits bulk clients).

## Reference fixtures: not generated, one runner change needed

`reference/embeddinggemma2/run_embeddinggemma2_image.py` and `run_embeddinggemma2_video.py` add fixed interleaved and query rows (`INTERLEAVED`, `QUERIES`) that name development media ids, so they cannot run on another media set unchanged. The change is an option to take those rows from the manifest (or skip them), as `--batch-ids` does for the text runners in this proposal. After it:

```sh
LD_LIBRARY_PATH=$LGPL/lib <eg2-video venv>/bin/python run_embeddinggemma2_image.py --model-dir <eg2 snapshot> \
  --revision 914f7f89142e33e77833254d9c9b90c3cef7303b --images <dir> --out <image release out> --tensors <tensor dir>
LD_LIBRARY_PATH=$LGPL/lib <eg2-video venv>/bin/python run_embeddinggemma2_video.py --model-dir <eg2 snapshot> \
  --revision 914f7f89142e33e77833254d9c9b90c3cef7303b --videos <dir> --out <video release out> --tensors <tensor dir>
```

Cost on the reference workstation: about 16 s per image on CPU for the .NET path (development benchmark), similar for PyTorch; the video path runs up to 32 frames per video. Minutes, not hours.

## Decisions for the reviewer

1. Confirm the 22 licenses and the attribution text (draft notice in `docs/plan/stable-proposal.md`).
2. Accept the media as the P-030 image and video release media, and approve the runner change above.
3. Whether to keep the bytes outside Git (proposed: yes; the release record pins them by SHA-256, as for the development media).
