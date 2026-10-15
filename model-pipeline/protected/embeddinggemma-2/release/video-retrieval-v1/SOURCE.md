# EmbeddingGemma 2 G3 text-to-video retrieval set (`video-retrieval-v1`)

The G3 task-quality set for `embeddinggemma-2/video-fp32-cpu` and `video-fp32-cuda`. It has 50 text queries against 50 NASA videos from Wikimedia Commons. Each query is its video's own Commons description, and the video is the query's only relevant document (relevance 1). Metrics are nDCG@10 and MRR@10 from `model-pipeline evaluate --g3 retrieval`. The budgets in the two profiles' `task-quality.yaml` were set from the upstream reference's score on this exact set, before any .NET run on it.

**No media bytes are committed.** `corpus.jsonl` names each video by file name and SHA-256, so the dataset digest pins the bytes. The videos are fetched by script.

## Selection

No agent wrote or chose any query or video. `model-pipeline/reference/embeddinggemma2/select_eg2_retrieval.py select --kind video --exclude release/media/media.json` ran with seed 20261015, queried 2026-10-15T19:23Z. Its steps:

1. Read every member of `Category:Videos by NASA` (1,042).
2. Keep only WebM files of at most 40 MB and 60 s. Ogg is excluded because two of the six Ogg/WebM P-030 release videos could not be opened by upstream's torchcodec `seek_mode="exact"`, and an Ogg Theora seek quirk differs from sequential decoding (`release/video-fp32-release/SOURCE.md`).
3. Keep only the license allowlist: public domain, CC0 or CC BY.
4. Drop the P-030 release titles.
5. Require an English `ImageDescription` of at least 20 characters after cleaning. Cleaning unescapes HTML entities, removes tags and collapses whitespace; a description longer than 300 characters is cut at the last space before that limit.
6. Drop descriptions shared by two files. This left 121 eligible files.
7. Draw 50 with `random.Random("20261015:video-retrieval")`.

`video-media.json` is the recorded selection, with each description verbatim. Commons is a live site, so everything is rebuilt from this record and never by re-querying.

`fetch_eg2_release_media.py --media video-media.json` checked every license again through the Commons API (`licenses.json`, 2026-10-15), along with each revision's SHA-1 and size. All 50 files were kept. `select_eg2_retrieval.py build` then wrote `queries.jsonl`, `corpus.jsonl`, `qrels.jsonl` and `dataset.json`.

```sh
S=model-pipeline/reference/embeddinggemma2
python -I $S/fetch_eg2_release_media.py --media video-media.json --out <media dir>
python -I $S/select_eg2_retrieval.py build --media video-media.json --fetched <media dir> --out <dir> \
  --name embeddinggemma2-commons-video-retrieval-v1                                     # byte-identical to this directory
```

The videos are WebM: 37 VP8, 10 VP9 and 3 AV1 (ffprobe), 1.9 to 60.0 s long, from 321x240 to 3840x3840. The upstream reference and the .NET FFmpeg 8 adapter both decode them through the pinned FFmpeg `n8.1.3-17-gc389840894` LGPL shared build (`fetch_eg2_ffmpeg.sh`).

## Licenses

**Videos:** 47 public domain (mostly `PD-USGov-NASA`), 2 CC BY 3.0 and 1 CC BY 2.0. Each license is as Commons states it for the file (`licenses.json`); this is data, not a legal review. Authors and credits per file are in `dataset.json` (`files`) and `licenses.json`. The videos are not redistributed.

**Queries:** the descriptions are text from the files' Commons description pages, under Wikimedia's text license, CC BY-SA 4.0. Many are copied from NASA captions, which are in the public domain. `queries.jsonl`, and the descriptions in `video-media.json` and `dataset.json`, are therefore committed under **CC BY-SA 4.0**, with attribution to each file's description page (`source` in `dataset.json`), as for the MIRACL passages. Evaluating with them and publishing scores is unrestricted.

## Format and prompts

The format is the evaluator's retrieval format with a media corpus (`model-pipeline/README.md`, "Task quality (G3)"), and `dataset.json` declares `"modality": "video"`. Queries are embedded with the model's `query` prompt. Videos are embedded alone without a prompt, sampled as upstream samples them (1 fps, at most 32 frames).

## Reference score and budget

Dataset digest: `fe9b83c107c717d55d78e464f0c47ff8b53f05d380423b79c28a8b893c2fac08` over the data files. With `dataset.json` included it is `c1573b0f0715d310875399c48510c2275fc92adb78b1bb618d785bf8af349ed9`, and the budgets bind that form.

The reference is `reference/embeddinggemma2/reference_eg2_media_retrieval.py` at `9a16ec8` (#53). It ran `google/embeddinggemma-2` at `914f7f89142e33e77833254d9c9b90c3cef7303b` with sentence-transformers 6.1.0, transformers 5.19.0, PyTorch 2.14.1+cpu and torchcodec 0.17.0, on the pinned FFmpeg n8.1.3 LGPL build (`EG2_TORCHCODEC_FFMPEG_MAJOR=8`). It used fp32 on CPU with 14 threads, on a RunPod EPYC 7H12 pod on 2026-10-15. Each input went through its own forward call, and the scores came from the evaluator's own functions. The run took 2,575 s. The embeddings stay outside Git, with SHA-256 `87e57a9599780ed933cf92b3248509ffe3a3617dc232d934b285af5e526bafe0`.

| nDCG@10 | MRR@10 | recall@1 | recall@5 | recall@10 |
| ---: | ---: | ---: | ---: | ---: |
| 0.8126 | 0.7714 | 0.68 | 0.90 | 0.94 |

Three queries rank their video below 10. Two queries are near-ties: the gap between their video and the best other video is 9.5e-6 and 2.9e-5. The next smallest gap is 1.2e-3, and 8 queries have gaps below 1e-2.

**Budget** for `video-fp32-cpu` and `video-fp32-cuda`: `ndcg@10 ≥ 0.797`, `mrr@10 ≥ 0.751`. That is the reference minus 0.015 and minus 0.020, rounded down to three decimals. The rule is the same as for the image set, which admits about two adjacent swaps at the top of the ranking. With 50 queries, one swap of ranks 1 and 2 costs (1 − 1/log2 3)/50 = 0.0074 nDCG@10 and 0.5/50 = 0.010 MRR@10. The text sets' 0.006 would admit no flip at all on 50 queries, and this set has two near-ties that a fidelity-equivalent fp32 path can flip.
