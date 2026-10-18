# Choosing an embedding model

Both models run locally through `IEmbeddingGenerator<string, Embedding<float>>`. Start with the input you need to represent and the download you can afford; measure retrieval quality on your own documents before choosing. Switching models requires re-embedding your documents: their vector spaces are not interchangeable.

This guide covers the two published embedding families. The experimental BERT embedders in `Starlings.Models.Bert` (all-MiniLM-L6-v2, bge-small-en-v1.5, bge-base-en-v1.5, nomic-embed-text-v1.5) are source only, not on nuget.org, have only proposed evidence, and are not compared here ([BERT API](../../ai-dotnet/src/Starlings.Models.Bert/README.md)).

| Property | multilingual-e5-small | EmbeddingGemma 2 |
| --- | --- | --- |
| Output dimensions | [384](../../ai-dotnet/src/Starlings.Models.E5/README.md) | [768, with Matryoshka sizes 512, 256 or 128](../../ai-dotnet/src/Starlings.Models.EmbeddingGemma2/README.md); truncate and normalize again. These are the [model card's supported sizes](../m0/embeddinggemma2-spike.md#2-card-and-configuration-facts). |
| Maximum input | [512 tokens](../../model-pipeline/registry/multilingual-e5-small/onnx-fp32-cpu/manifest.yaml), including prefix and special tokens | [8,192 tokens](../../ai-dotnet/src/Starlings.Models.EmbeddingGemma2/README.md), including prompt, special tokens and any image/video tokens |
| Download for text | About **487 MB**: graph plus tokenizer ([asset lock](../../ai-dotnet/src/Starlings.Models.E5/E5ModelAssets.cs)) | About **1.12 GB**: text graph, tokenizer and prompt configuration ([asset lock](../../ai-dotnet/src/Starlings.Models.EmbeddingGemma2/EmbeddingGemma2ModelAssets.cs)) |
| Download with images/video | Not supported | About **2.87 GB total**, including the text files and the vision and multimodal graphs ([image asset lock](../../ai-dotnet/src/Starlings.Models.EmbeddingGemma2/EmbeddingGemma2ModelAssets.cs)); existing text downloads are reused |
| Retrieval inputs | `query: ` before questions; `passage: ` before documents ([adapter usage](../../ai-dotnet/src/Starlings.Models.E5.Extensions.AI/README.md)) | `EmbeddingGemma2Prompt.Query` for questions; `EmbeddingGemma2Prompt.Document` for documents ([adapter usage](../../ai-dotnet/src/Starlings.Models.EmbeddingGemma2.Extensions.AI/README.md)) |
| Modalities in Starling | Text | Text, images and video; audio is unsupported ([model package](../../ai-dotnet/src/Starlings.Models.EmbeddingGemma2/README.md)) |
| Model weights license | MIT, as recorded in the [package's model-card attribution](../../ai-dotnet/src/Starlings.Models.E5/README.md#stable-contract) and [pinned upstream card](https://huggingface.co/intfloat/multilingual-e5-small/blob/614241f622f53c4eeff9890bdc4f31cfecc418b3/README.md) | Apache-2.0, as recorded in the [card review](../m0/embeddinggemma2-spike.md#2-card-and-configuration-facts) and [pinned upstream card](https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/README.md) |

Download totals are rounded decimal sizes calculated from the linked locks, not NuGet package sizes or runtime RAM requirements. The benchmark reports do not measure downloads. Matryoshka reduces stored vector size; it does not shrink the downloaded model. Configure prefixes or prompts explicitly: neither generator adds them by default. If you configure an E5 input type, do not also prepend its prefix yourself. Split long documents into chunks that fit the chosen model.

## Speed: what the repository actually measures

The [e5 matched-backend report](../benchmarks/e5-2026-10-09.md#headline-warm-latency-c-library-against-python) records a **4.83 ms CPU** and **2.09 ms CUDA** C# median for `short-b1`, text input, concurrency **1**. Its [environment](../benchmarks/e5-2026-10-09.md#environment) was an Intel Core i7-5820K with **6 intra-op threads**, and an RTX 4060 Ti for CUDA. These are warm, exploratory workstation measurements, not a latency promise.

The [EmbeddingGemma text/image benchmark](../benchmarks/embeddinggemma2-2026-10-13.md) explicitly says its contended development timings must not be quoted as performance. It therefore supplies **no defensible speed comparison with e5**. Use the linked workload and protocol to measure your input lengths, batching and hardware; do not divide numbers from these separate runs into a speedup claim.

For platform support and release maturity, check the [compatibility matrix](../catalog/compatibility-matrix.md) and the [README release status](../../README.md#models). The capabilities on `main` can be ahead of the published packages. The model licenses above are distinct from the Starling code license; the EmbeddingGemma [licensing notes](../../ai-dotnet/src/Starlings.Models.EmbeddingGemma2/README.md#license-and-usage-policy) also record its card's usage-policy statement.

Choose **e5** for text-only retrieval when the smaller download is the priority.<br>
Choose **EmbeddingGemma 2** for longer text inputs, selectable vector sizes, or image/video retrieval.<br>
For either, decide on measured retrieval quality and latency for your own workload.
