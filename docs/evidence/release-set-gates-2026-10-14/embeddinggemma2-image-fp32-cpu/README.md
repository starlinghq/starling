# embeddinggemma-2 `image-fp32-cpu` on the P-030 image release set

The .NET stage results were produced on 2026-10-14 at `e0e29b9`, the #27 branch commit; its code equals `7a915d9` on main. The evaluator ran at `472e187`. The set is `embeddinggemma-2/release/image-fp32-release`: 16 Wikimedia Commons images and 19 rows (one per image, the two prompt rows, and `mm-two-images`). The evaluator verified the protected evidence against Git, and the bounds come from `protected/embeddinggemma-2/image-fp32-cpu/tolerances.yaml`. Nothing under `protected/` was changed.

| Gate | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- |
| G1 | **passed** | rendered ids exact 19/19; SkiaSharp decoded RGB vs PIL, resized pixels, pixel values and positions exact on 20/20 images | [`g1-g2-image-release/report.md`](g1-g2-image-release/report.md) |
| G2 | **passed**, first divergence none | embedding max abs 8.55e-7 (1e-5); min cosine 0.999999999981 (0.9999995); soft tokens row-relative 8.68e-5 (1e-3); token embeddings row-relative 1.79e-5 (1e-3); both batch layouts vs single 0 (1e-5). Reference tensors are checked against `tensors.json`, every array equal. | same |

## Findings

1. **The package's default pixel limit refuses a release-set image.** `commons-image-14` (`Umnak oli 2014123 lrg.jpg`, 10977x7200, 79 MP) is in the accepted set, and the reference embeds it. With the defaults, the C# dump stops on that image: `EmbeddingGemma2ImageTooLargeException: The image is 10977x7200 (79034400 pixels); the limit is 50000000 decoded pixels` ([`dotnet-default-limit.log`](dotnet-default-limit.log)). `MaxImagePixels` and the decoder's `MaxPixels` are documented memory bounds that upstream does not have. The passing run used `--max-image-pixels 80000000` (#27), and the stage environment records the value (`max_image_pixels`). **So G2 passes with the limit raised for this one image, not at the package default.** Decision (coordinator, 2026-10-14): the default stays at 50 MP, because it is a documented memory bound and not a fidelity rule. The row stays in the set and is evaluated with `--max-image-pixels` raised explicitly (80,000,000 here). The manifests record the capability `image_over_default_pixel_limit_with_option: experimental`, since the row passes G1 and G2 with the option.
2. **The reference tensors reproduce only at the reference's thread count.** The `.npz` stage tensors are kept outside Git. A regeneration with 4 threads (`run_embeddinggemma2_image.py --cases image-cases.json`, venv from `requirements-video.txt`) matched the integer and vision-encoder arrays. The soft tokens, token embeddings and `image-embeddings.jsonl` differed (`007bf29e…`). With these tensors the evaluator ran the soft-token and token-embedding checks as not-run, and G2 was not-run. A 6-thread regeneration after the soak measurements finished, matching the reference's `torch_threads: 6`, reproduced every array of `tensors.json` and `image-embeddings.jsonl` byte for byte (`13bf5780…`). The report above uses those tensors.

## Commands

```sh
REL=model-pipeline/protected/embeddinggemma-2/release/image-fp32-release
python -I model-pipeline/reference/embeddinggemma2/fetch_eg2_release_media.py --media model-pipeline/protected/embeddinggemma-2/release/media/media.json --out ~/.cache/starling-release-media
# images.json from the fetch is byte-identical to the protected one (888b56cb…); all 16 licenses re-checked, none rejected
OMP_NUM_THREADS=6 <eg2-video venv>/bin/python model-pipeline/reference/embeddinggemma2/run_embeddinggemma2_image.py --model-dir <snapshot 914f7f89> \
  --revision 914f7f89142e33e77833254d9c9b90c3cef7303b --images ~/.cache/starling-release-media --cases $REL/image-cases.json --out <ref> --tensors <ref>/tensors
dotnet ai-dotnet/tools/Starlings.Fidelity.Tool/bin/Release/net10.0/Starlings.Fidelity.Tool.dll embeddinggemma2-images --out <run>/stages \
  --images ~/.cache/starling-release-media --fixtures $REL --threads 4 --max-image-pixels 80000000
model-pipeline evaluate model-pipeline/registry/embeddinggemma-2/image-fp32-cpu/manifest.yaml --fixtures $REL --results <run>/stages \
  --reference-tensors <ref>/tensors --out <report> --require-gates G1,G2
```

Wall clock (shared machine, not benchmarks): reference regeneration 1,133 s at 4 threads and 620 s at 6 threads; .NET dump 328 s (4 threads); the default-limit dump ran 105 s before the refusal.
