# embeddinggemma-2 `image-fp32-cpu` on the P-030 image release set

Run on 2026-10-14 at `e0e29b9`, the #27 branch commit. Its code is identical to `7a915d9`, the squash merge on main. Set `embeddinggemma-2/release/image-fp32-release`: 16 Wikimedia Commons images and 19 rows (one per image, the two prompt rows, and `mm-two-images`). The evaluator verified the protected evidence against Git, and the bounds come from `protected/embeddinggemma-2/image-fp32-cpu/tolerances.yaml`. Nothing under `protected/` was changed.

| Gate | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- |
| G1 | **passed** | rendered ids exact 19/19; SkiaSharp decoded RGB vs PIL, resized pixels, pixel values and positions exact on 20/20 images | [`g1-g2-image-release/report.md`](g1-g2-image-release/report.md) |
| G2 | **not-run** | embedding max abs 8.55e-7 (1e-5), min cosine 0.999999999981 (0.9999995), both batch layouts vs single 0 (1e-5). `float.soft_tokens_row_relative` and `float.token_embeddings_row_relative` not run: no reference tensors whose digests match `tensors.json` (below). | same |

## Findings

1. **The package's default pixel limit refuses a release-set image.** `commons-image-14` (`Umnak oli 2014123 lrg.jpg`, 10977x7200, 79 MP) is part of the accepted set, and the reference embeds it. With the defaults, the C# dump stops on it: `EmbeddingGemma2ImageTooLargeException: The image is 10977x7200 (79034400 pixels); the limit is 50000000 decoded pixels` ([`dotnet-default-limit.log`](dotnet-default-limit.log)). `MaxImagePixels` and the decoder's `MaxPixels` are documented memory bounds that upstream does not have. The run above used `--max-image-pixels 80000000` (#27), which the stage environment records as `max_image_pixels`. The maintainer has three options: raise the default; leave it and mark images above 50 MP as outside the default contract (a `protected-change` that makes the row excluded by contract); or replace the image.
2. **The reference tensors could not be reproduced bit for bit.** `.npz` stage tensors are not in Git. I regenerated them with the runner and cases the set names (`run_embeddinggemma2_image.py --cases image-cases.json`, venv from `requirements-video.txt`, 4 threads, because the workstation was running soak measurements). The integer arrays and the PyTorch vision encoder outputs match `tensors.json` (pixel values, encoder and pooled outputs: 21/21). The float arrays that the soft-token and token-embedding checks read differ (soft tokens 20/20, token embeddings 19/19, embedding 19/19), as does `image-embeddings.jsonl` (`007bf29e…` against the protected `13bf5780…`). The reference ran with 6 threads. The evaluator refuses tensors whose digests differ, so those two checks did not run and G2 cannot pass. A 6-thread regeneration on a quiet machine is the next step.

## Commands

```sh
REL=model-pipeline/protected/embeddinggemma-2/release/image-fp32-release
python -I model-pipeline/reference/embeddinggemma2/fetch_eg2_release_media.py --media model-pipeline/protected/embeddinggemma-2/release/media/media.json --out ~/.cache/starling-release-media
# images.json produced by the fetch is byte-identical to the protected one (888b56cb…); all 16 licenses re-checked, none rejected
OMP_NUM_THREADS=4 <eg2-video venv>/bin/python model-pipeline/reference/embeddinggemma2/run_embeddinggemma2_image.py --model-dir <snapshot 914f7f89> \
  --revision 914f7f89142e33e77833254d9c9b90c3cef7303b --images ~/.cache/starling-release-media --cases $REL/image-cases.json --out <ref> --tensors <ref>/tensors
dotnet ai-dotnet/tools/Starlings.Fidelity.Tool/bin/Release/net10.0/Starlings.Fidelity.Tool.dll embeddinggemma2-images --out <run>/stages \
  --images ~/.cache/starling-release-media --fixtures $REL --threads 4 --max-image-pixels 80000000
model-pipeline evaluate model-pipeline/registry/embeddinggemma-2/image-fp32-cpu/manifest.yaml --fixtures $REL --results <run>/stages --out <report> --require-gates G1,G2
```

Wall clock (shared machine, not benchmarks): reference regeneration 1,133 s; .NET dump 328 s; default-limit dump until the refusal 105 s.
