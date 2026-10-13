# Release drill with ML.NET, public-hosting and Native AOT steps (R9)

[`rc-drill.log`](rc-drill.log) is the full output of this command:

```sh
tools/release-drill-ai.sh --seed-model-from <e5 snapshot> --seed-gliner2-from <GLiNER2 snapshot> \
    --seed-embeddinggemma2-from <EmbeddingGemma 2 snapshot> --seed-qwen35-from <Qwen fp32x snapshot> --aot
```

It ran at commit `756ba5a` on 2026-10-13. That was a work-in-progress commit later squashed into this PR, with the same `ai-dotnet/` and `tools/`. Machine: Linux x64, CPU only, shared, with `DOTNET_PROCESSOR_COUNT=4`. The result was **DRILL PASS**.

- **Packing.** 14 packages, all nuspec checks unchanged. The new `IsAotCompatible` adds no package dependency.
- **ML.NET, new steps 8c and 9e.** A new checker (`tools/release-drill/mlnet-consumer`) consumes the packed `Starlings.Models.E5.MLNet` and `Starlings.Models.EmbeddingGemma2.MLNet`. Offline, it checks:
  - column types `Vector<Single, 384>` and `Vector<Single, 768>`;
  - row-wise and look-ahead batched transforms equal to the typed API exactly;
  - token-count columns;
  - expected ranking;
  - a Matryoshka 256 column (EmbeddingGemma 2);
  - Save rejected.

  Logs: [`consumer-mlnet-e5.log`](consumer-mlnet-e5.log) and [`consumer-mlnet-eg2.log`](consumer-mlnet-eg2.log). Before this change, `E5.MLNet` ran only inside the e5 checker and `EmbeddingGemma2.MLNet` never ran.
- **Public hosting, new step 11.** Disabled by default. The log shows `skipped: hosting private (maintainer action M3 pending ...)`. Even with `--public-hosting`, each model is skipped while its library still records its hosting as private or a placeholder.
- **Native AOT, new step 12.** `tools/aot-check-ai.sh` ran on the drill feed. Results are in [`../aot/README.md`](../aot/README.md).
- **Not run.** The EmbeddingGemma 2 image and video searches, because the seed directory had no `images/` or `videos/` and no FFmpeg directory was given. All CUDA runs.
