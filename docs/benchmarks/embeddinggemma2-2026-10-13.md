# EmbeddingGemma 2 text and image, CPU: uncontrolled development run, 2026-10-13

> **Uncontrolled development numbers, not G5 evidence.** One brief run on a shared workstation while other agents' builds, tests, soak runs (including an EmbeddingGemma 2 image soak) and a GPU job ran (load average 15 to 22 on 12 logical CPUs). The same short text took 65 ms in an earlier smoke run and 299 ms here, so the timings below are dominated by contention. They show that the harness works and what it records; they must not be compared with a budget or quoted as performance. The G5 budgets are maintainer decisions (production-readiness plan, M2), and G5 needs a controlled machine (M6). Protocol: [PROTOCOL.md](PROTOCOL.md).

`google/embeddinggemma-2@914f7f89`, text graph `embeddinggemma2-text-fp32.onnx` (release `embeddinggemma-2-onnx-r1`, SHA-256 `02cc7ecc…2d11`), image graphs of `embeddinggemma-2-vision-onnx-r1` (vision and multimodal text; digests in the run records), profiles `onnx-fp32-cpu` and `image-fp32-cpu`. `Starlings.Benchmarks.EmbeddingGemma2Harness` at `92d5fd7` (`git_dirty: false`; binaries built from that tree before a message-only amend, so informational versions show `42d93dd`). Workload file `ai-dotnet/benchmarks/workloads/embeddinggemma2-v1.json`. Image set regenerated with `make_eg2_images.py` (every file's SHA-256 checked against `images.json`). ONNX Runtime 1.30.0, `Starlings.Tokenizers` 0.3.0-preview.1, SkiaSharp 4.153.1, .NET 10.0.12, Intel i7-5820K, `intra_op_num_threads=6`, `inter_op_num_threads=1`.

- Raw results: [`embeddinggemma2-2026-10-13.raw.jsonl`](embeddinggemma2-2026-10-13.raw.jsonl). Summary: [`.summary.md`](embeddinggemma2-2026-10-13.summary.md), [`.summary.json`](embeddinggemma2-2026-10-13.summary.json).
- Commands: text `warm --paths text --samples-scale 0.25` and `cold --paths text --processes 3`; image `warm --paths image --preliminary workloads --samples-scale 0.2` (preliminary sweep limited to the 8 image rows the workloads use, to keep the run brief).

## Quality attached to the timings

Text preliminary sweep: all 138 protected rows alone, max abs 3.7e-7 (bound 5e-6, `float.embedding_max_abs`), min cosine 0.99999999999, finite; and in padded batches of 8. Image preliminary sweep: 8 rows, max abs 1.3e-6 (bound 1e-5). Every series passed after its timed loop (worst 1.3e-6, on the batch of 4 images). `preprocess-only` reproduced the fixture's pixel-values SHA-256 exactly. The 100- and 400-word inputs have no upstream reference; they were compared with the same text embedded alone at load (identical; the summary's "Quality max abs" of 0 for those rows is that comparison).

## Results (median, ms)

| Path | Workload | Input | n | Median | Stages |
| --- | --- | --- | ---: | ---: | --- |
| text | short-b1 | 10–45 tokens | 25 | 299 | tokenize 0.08, encode 298 |
| text | short-b8 | 8 texts | 10 | 2,036 | tokenize 0.28, encode 2,035 |
| text | short-b16 | 16 texts | 5 | 4,011 | tokenize 3.9, encode 4,006 |
| text | words100-b1 | first 100 words | 8 | 603 | tokenize 0.32, encode 602 |
| text | words400-b1 | first 400 words | 4 | 2,664 | tokenize 1.3, encode 2,663 |
| text | long-b1 | 1,572–1,750 tokens | 2 | 8,985 | tokenize 3.5, encode 8,982 |
| text | long-b4 | 4 long texts | 1 | 27,436 | tokenize 9.3, encode 27,426 |
| image | photo-b1 | photos, 451×300 to 600×400 | 4 | 16,059 | decode 9.0, embed 16,050 |
| image | tiny-b1 | 1×1 and 8×6 | 4 | 15,980 | decode 0.34, embed 15,980 |
| image | huge-b1 | 4000×3000 PNG | 2 | 16,518 | decode 220, embed 16,298 |
| image | photo-b4 | 4 photos, one call | 1 | 59,460 | decode 67, embed 59,393 |
| image | two-images-interleaved | text + 2 images | 2 | 12,524 | decode 13, embed 12,511 |
| image | preprocess-only | 3 images (incl. 4000×3000) | 10 | 58 | decode 5.4, preprocess 21 |

Text cold start (n=3): spawn to first embedding 5,686 ms; tokenizer load 1,229 ms, session create 3,617 ms, first request 411 ms. Warm load: tokenizer 890 ms, text session 5,145 ms, multimodal embedder (tokenizer and both image graphs) 7,141 ms.

What the run shows despite the noise: tokenization and image decoding are small next to the ONNX Runtime runs; image cost is almost independent of the source size (every image is resized to about 256–266 soft tokens), and the image processor itself (21 ms median) is a small part of `embed`. Text batches gave no throughput gain over single requests on this CPU (8 texts about 6.8 times one); worth checking on a quiet machine.

## Gaps

- Video is not benchmarked: the video fixtures need FFmpeg to decode and no decoded-frame fixtures exist.
- The image `embed` stage is one public call (preprocessing, vision graph, multimodal graph); only preprocessing is timed separately.
- No Python same-backend comparison (feasible for text, as for e5, once `bench_e5_matched.py` is factored into a shared runner; see PROTOCOL.md). No CUDA run. No allocation or memory figures.
