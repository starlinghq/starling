namespace Starlings.Models.EmbeddingGemma2;

/// <summary>Support status of one capability (plan section 9). A missing entry never means supported.</summary>
public enum EmbeddingGemma2CapabilityStatus
{
    /// <summary>Qualified for the advertised maturity.</summary>
    Supported,

    /// <summary>Not available; requests for it are rejected or the API does not offer it.</summary>
    Unsupported,

    /// <summary>Implemented and tested against the reference fixtures, not yet qualified.</summary>
    Experimental,

    /// <summary>May work, but no sufficient evidence exists.</summary>
    Untested,
}

/// <summary>
/// Capability descriptor for this package version (profiles <c>embeddinggemma-2/onnx-fp32-cpu</c> and
/// <c>embeddinggemma-2/onnx-fp32-cuda</c>). The model is admitted at maturity Experimental, text embeddings only
/// (<c>model-pipeline/admissions/embeddinggemma-2.md</c>).
/// </summary>
public static class EmbeddingGemma2Capabilities
{
    /// <summary>Status per capability. Evidence for each entry is listed in <c>ai-dotnet/README.md</c>.</summary>
    public static IReadOnlyDictionary<string, EmbeddingGemma2CapabilityStatus> All { get; } = new Dictionary<string, EmbeddingGemma2CapabilityStatus>(StringComparer.Ordinal)
    {
        // Text embedding from token ids: 138 protected rows (46 texts x no prompt, query, document; up to 1,750 tokens).
        ["embedding.pretokenized.cpu"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["embedding.pretokenized.cuda"] = EmbeddingGemma2CapabilityStatus.Experimental,

        // Text through EmbeddingGemma2Tokenizer (owned profile gemma-bpe-embeddinggemma2, qualified with tokenizer
        // 0.1.0-preview.4, now Starlings.Tokenizers 0.3.0-preview.1): 138 fixture texts x prompt settings, ids exact, embeddings within the binding bound (CPU).
        ["embedding.text"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["tokenizer.owned"] = EmbeddingGemma2CapabilityStatus.Experimental,

        // Explicit named prompts from config_sentence_transformers.json (never inserted silently; default none). The
        // fixtures cover none, query and document; the other 18 names map to prompt texts without fixture rows.
        ["prompts.query_document"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["prompts.task"] = EmbeddingGemma2CapabilityStatus.Untested,
        ["prompt_insertion.implicit"] = EmbeddingGemma2CapabilityStatus.Unsupported,

        // Truncate the leading dimensions, then L2-normalize again (equal to sentence-transformers truncate_dim).
        ["matryoshka.512"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["matryoshka.256"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["matryoshka.128"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["matryoshka.other_sizes"] = EmbeddingGemma2CapabilityStatus.Unsupported,

        // Input length, special tokens and prompt included. Fixtures reach 1,750 tokens, batch-independence checks
        // 2,049; 2,050 to 8,192 tokens has no reference evidence. Longer input is rejected, never truncated.
        ["input.up_to_2049_tokens"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["input.2050_to_8192_tokens"] = EmbeddingGemma2CapabilityStatus.Untested,
        ["input.over_8192_tokens"] = EmbeddingGemma2CapabilityStatus.Unsupported,
        ["truncation"] = EmbeddingGemma2CapabilityStatus.Unsupported,

        ["batch"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["cancellation"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["concurrent_calls"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["asset_acquisition"] = EmbeddingGemma2CapabilityStatus.Experimental,
        // The graph is hosted as a GitHub release asset of starlinghq/starling; until the release exists the address is a
        // placeholder (EmbeddingGemma2ModelAssets.ModelSourceIsPlaceholder) and the graph must be produced locally.
        ["asset_acquisition.graph_download"] = EmbeddingGemma2CapabilityStatus.Untested,

        // Images (contract of 2026-10-11, EmbeddingGemma2MultimodalEmbedder; profiles image-fp32-cpu and image-fp32-cuda).
        // From decoded RGB: 42 protected rows (28 images alone, 6 interleaved or prompted, 8 text queries) within the
        // binding bounds; preprocessing bit-exact on the 28 supported fixture images; rendered ids exact on all 43 rows.
        ["image"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["image.cpu"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["image.cuda"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["image.interleaved_text"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["image.multiple_per_input"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["image.prompts"] = EmbeddingGemma2CapabilityStatus.Experimental, // rendered as upstream; the card recommends none
        ["image.budget_280"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["image.budget_other"] = EmbeddingGemma2CapabilityStatus.Untested, // 70, 140, 560, 1,120: accepted, no evidence
        ["image.batch"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["image.matryoshka"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["image.over_8192_tokens"] = EmbeddingGemma2CapabilityStatus.Unsupported, // rejected before inference
        ["image.over_50_megapixels"] = EmbeddingGemma2CapabilityStatus.Unsupported, // default limit, configurable
        ["image.alpha_compositing"] = EmbeddingGemma2CapabilityStatus.Unsupported, // alpha is dropped, as upstream
        ["image.16_bit_per_channel"] = EmbeddingGemma2CapabilityStatus.Unsupported, // rejected by the decoder; upstream renders white
        // Decoding is in Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp: PNG and JPEG decoded RGB equal to PIL
        // on the fixture images; GIF, BMP and WebP decoded without evidence; EXIF orientation ignored unless opted in.
        ["image.decode.png_jpeg"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["image.decode.gif_bmp_webp"] = EmbeddingGemma2CapabilityStatus.Untested,
        ["image.decode.other_formats"] = EmbeddingGemma2CapabilityStatus.Unsupported,
        ["image.exif_orientation_opt_in"] = EmbeddingGemma2CapabilityStatus.Experimental, // deviation from upstream
        ["image.graph_download"] = EmbeddingGemma2CapabilityStatus.Untested, // release embeddinggemma-2-vision-onnx-r1 not published
        ["extensions_ai.image_embedding_generator"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["mlnet.image_column"] = EmbeddingGemma2CapabilityStatus.Unsupported,

        // Videos (contract of 2026-10-11, EmbeddingGemma2MultimodalEmbedder; profiles video-fp32-cpu and video-fp32-cuda).
        // Checkpoint settings only: 1 frame per second, at most 32 frames, 140 soft tokens per frame, no timestamps. From
        // decoded RGB frames: 35 protected rows (22 videos alone, 6 interleaved or prompted, 2 input forms, 5 text
        // queries) within the binding bounds; sampling exact on 20,849 upstream cases; per-frame preprocessing bit-exact
        // on all 32 video inputs; rendered ids exact on all 36 rows.
        ["video"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["video.cpu"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["video.cuda"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["video.decoded_frames_with_timing"] = EmbeddingGemma2CapabilityStatus.Experimental, // the package samples as upstream
        ["video.pre_sampled_frames"] = EmbeddingGemma2CapabilityStatus.Experimental, // caller's choice, at most 32
        ["video.frames_without_timing"] = EmbeddingGemma2CapabilityStatus.Unsupported, // rejected by default (typed exception)
        ["video.frames_without_timing.upstream_opt_in"] = EmbeddingGemma2CapabilityStatus.Experimental, // differs from the file (cosine 0.965)
        ["video.interleaved_text"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["video.interleaved_image"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["video.multiple_per_input"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["video.prompts"] = EmbeddingGemma2CapabilityStatus.Experimental, // rendered as upstream; the card recommends none
        ["video.batch"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["video.matryoshka"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["video.sub_2s_one_frame"] = EmbeddingGemma2CapabilityStatus.Experimental, // upstream quirk kept for parity
        ["video.timestamps"] = EmbeddingGemma2CapabilityStatus.Unsupported, // add_timestamps is off in the checkpoint
        ["video.other_fps_max_frames_or_budget"] = EmbeddingGemma2CapabilityStatus.Unsupported, // not configurable
        ["video.variable_frame_rate"] = EmbeddingGemma2CapabilityStatus.Untested, // index-based sampling, as upstream
        ["video.interlaced"] = EmbeddingGemma2CapabilityStatus.Untested,
        ["video.hdr"] = EmbeddingGemma2CapabilityStatus.Untested, // 10-bit SDR is covered (FFmpeg adapter)
        ["video.audio_track"] = EmbeddingGemma2CapabilityStatus.Unsupported, // ignored, as upstream
        ["video.over_8192_tokens"] = EmbeddingGemma2CapabilityStatus.Unsupported, // rejected before inference
        ["video.frames_over_50_megapixels"] = EmbeddingGemma2CapabilityStatus.Unsupported, // default limit, configurable
        // Decoding is in Starlings.Models.EmbeddingGemma2.Video.FFmpeg on the user's own FFmpeg 8.x (none shipped):
        // metadata, sampled indices and the RGB of every frame equal torchcodec's on the 22 fixture videos (H.264 8/10-bit,
        // HEVC, VP9, AV1; rotation 90; odd sizes) with FFmpeg 8.1 on x86-64.
        ["video.decode.ffmpeg8"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["video.decode.ffmpeg9"] = EmbeddingGemma2CapabilityStatus.Unsupported, // other ABI; rejected with a typed error
        ["video.decode.rotation_90"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["video.decode.rotation_other"] = EmbeddingGemma2CapabilityStatus.Untested, // -90, 180: torchcodec's transforms, no fixture
        ["video.decode.streams"] = EmbeddingGemma2CapabilityStatus.Experimental, // seekable streams and memory, equal to files
        ["video.decode.arm64_or_non_avx2"] = EmbeddingGemma2CapabilityStatus.Untested, // swscale SIMD paths may differ by a level
        ["video.graph_download"] = EmbeddingGemma2CapabilityStatus.Untested, // same graphs as images; release not published
        ["extensions_ai.video_embedding_generator"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["mlnet.video_column"] = EmbeddingGemma2CapabilityStatus.Unsupported,

        // Out of the contract.
        ["audio"] = EmbeddingGemma2CapabilityStatus.Unsupported,
        ["media_placeholder_tokens"] = EmbeddingGemma2CapabilityStatus.Unsupported,
        ["pair_input"] = EmbeddingGemma2CapabilityStatus.Unsupported,
        ["streaming"] = EmbeddingGemma2CapabilityStatus.Unsupported,
        ["tf32"] = EmbeddingGemma2CapabilityStatus.Unsupported,
        ["bf16"] = EmbeddingGemma2CapabilityStatus.Unsupported,
        ["fp16"] = EmbeddingGemma2CapabilityStatus.Unsupported,
        ["quantized"] = EmbeddingGemma2CapabilityStatus.Unsupported,
        ["tensorrt"] = EmbeddingGemma2CapabilityStatus.Unsupported,
        ["training"] = EmbeddingGemma2CapabilityStatus.Unsupported,

        // Microsoft.Extensions.AI IEmbeddingGenerator, in Starlings.Models.EmbeddingGemma2.Extensions.AI.
        // Tested on the CPU profile with the owned tokenizer.
        ["extensions_ai.embedding_generator"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["extensions_ai.embedding_generator.cuda"] = EmbeddingGemma2CapabilityStatus.Untested,
        ["extensions_ai.dimensions_matryoshka"] = EmbeddingGemma2CapabilityStatus.Experimental,

        // ML.NET estimator/transformer, in Starlings.Models.EmbeddingGemma2.MLNet. Persistence: Save throws.
        ["mlnet.embedding_transform"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["mlnet.embedding_transform.cuda"] = EmbeddingGemma2CapabilityStatus.Untested,
        ["mlnet.batching"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["mlnet.persistence"] = EmbeddingGemma2CapabilityStatus.Unsupported,
        ["mlnet.cancellation"] = EmbeddingGemma2CapabilityStatus.Unsupported,
        ["mlnet.training"] = EmbeddingGemma2CapabilityStatus.Unsupported,

        ["aot"] = EmbeddingGemma2CapabilityStatus.Untested,
        ["windows"] = EmbeddingGemma2CapabilityStatus.Untested,
        ["macos"] = EmbeddingGemma2CapabilityStatus.Untested,
        ["arm64"] = EmbeddingGemma2CapabilityStatus.Untested,
    };
}
