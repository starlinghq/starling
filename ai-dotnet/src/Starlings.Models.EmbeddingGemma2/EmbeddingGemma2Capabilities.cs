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
/// Capability descriptor for this package version (text profiles <c>embeddinggemma-2/onnx-fp32-cpu</c> and
/// <c>embeddinggemma-2/onnx-fp32-cuda</c>; image and video profiles <c>image-fp32-*</c> and <c>video-fp32-*</c>).
/// </summary>
/// <remarks>
/// <see cref="All"/> is package-wide. The text keys without a provider suffix give the status on the Stable-contract
/// profile <see cref="StableProfile"/> (text on Linux x64 CPU); keys with <c>.cuda</c> give the experimental CUDA
/// profile's; image and video keys belong to the experimental image and video profiles. <see cref="Profiles"/> repeats
/// the Stable-contract profile's statuses keyed exactly as its registry manifest's <c>capabilities</c>, and a test keeps
/// the two equal key for key.
/// </remarks>
public static class EmbeddingGemma2Capabilities
{
    /// <summary>The profile of this package's Stable contract (the README's "Stable contract" section).</summary>
    public const string StableProfile = "embeddinggemma-2/onnx-fp32-cpu";

    /// <summary>
    /// Status per capability. Evidence for each entry is listed in <c>ai-dotnet/README.md</c> and, for
    /// <see cref="EmbeddingGemma2CapabilityStatus.Experimental"/>, in the profile's gate evidence (<c>docs/catalog/README.md</c>).
    /// </summary>
    public static IReadOnlyDictionary<string, EmbeddingGemma2CapabilityStatus> All { get; } = new Dictionary<string, EmbeddingGemma2CapabilityStatus>(StringComparer.Ordinal)
    {
        // Text on Linux x64 CPU, Stable since 2026-10-15 (the README's Stable contract): every gate passed, and the weekly
        // qualification reruns G1 to G3 on the release sets (G2 on the protected reference embeddings, #60) and a one-hour
        // soak, green. Evidence: G1 on the P-030 tokenizer holdout (232 rows, ids exact) and G2 on the text
        // release set (696 rows: no prompt, query and document; embeddings within 5.1e-7 against 5e-6, padded batches
        // within 2.0e-7 of single calls), the accepted development fixtures (138 rows up to 1,750 tokens, batch
        // independence to 2,049), G3 on MIRACL (nDCG@10 0.8325), G4 one-hour soak (concurrent callers, timed-out
        // requests, plateau), G6 anonymous graph download and offline run, G7 adapter tests and drill consumers.
        ["embedding.pretokenized.cpu"] = EmbeddingGemma2CapabilityStatus.Supported,
        ["embedding.pretokenized.cuda"] = EmbeddingGemma2CapabilityStatus.Experimental,

        // Text through EmbeddingGemma2Tokenizer (owned profile gemma-bpe-embeddinggemma2, Starlings.Tokenizers 0.3.0-preview.1).
        ["embedding.text"] = EmbeddingGemma2CapabilityStatus.Supported,
        ["tokenizer.owned"] = EmbeddingGemma2CapabilityStatus.Supported,

        // Explicit named prompts from config_sentence_transformers.json (never inserted silently; default none). The
        // accepted fixtures and the release set cover none, query and document; all 20 names on four public texts are in
        // the proposed task-prompt rows (ids exact, within the protected text bound on CPU and CUDA;
        // docs/evidence/capability-closure-2026-10-14/), proposed rows and so outside the Stable contract.
        ["prompts.query_document"] = EmbeddingGemma2CapabilityStatus.Supported,
        ["prompts.task"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["prompt_insertion.implicit"] = EmbeddingGemma2CapabilityStatus.Unsupported,

        // Truncate the leading dimensions, then L2-normalize again (equal to sentence-transformers truncate_dim); checked
        // against the reference sliced and renormalized, within the protected bound, and through both adapters (G7).
        ["matryoshka.512"] = EmbeddingGemma2CapabilityStatus.Supported,
        ["matryoshka.256"] = EmbeddingGemma2CapabilityStatus.Supported,
        ["matryoshka.128"] = EmbeddingGemma2CapabilityStatus.Supported,
        ["matryoshka.other_sizes"] = EmbeddingGemma2CapabilityStatus.Unsupported,

        // Input length, special tokens and prompt included. Accepted fixtures reach 1,750 tokens, batch-independence
        // checks 2,049; 2,050 to 8,192 tokens is covered only by the proposed long-input rows (8 text rows on CPU and
        // CUDA within the protected bound), so it is outside the Stable contract. Longer input is rejected, never truncated.
        ["input.up_to_2049_tokens"] = EmbeddingGemma2CapabilityStatus.Supported,
        ["input.2050_to_8192_tokens"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["input.over_8192_tokens"] = EmbeddingGemma2CapabilityStatus.Unsupported,
        ["truncation"] = EmbeddingGemma2CapabilityStatus.Unsupported,

        ["batch"] = EmbeddingGemma2CapabilityStatus.Supported,
        ["cancellation"] = EmbeddingGemma2CapabilityStatus.Supported,
        ["concurrent_calls"] = EmbeddingGemma2CapabilityStatus.Supported,
        ["asset_acquisition"] = EmbeddingGemma2CapabilityStatus.Supported,
        // The text graph is hosted in the public Hugging Face repository EmbeddingGemma2ModelAssets.HostingRepository. The
        // release drills downloaded it with no token into an empty cache, verified it and ran the search sample offline
        // (G6, docs/evidence/g6-distribution-2026-10-14/).
        ["asset_acquisition.graph_download"] = EmbeddingGemma2CapabilityStatus.Supported,

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
        ["image.budget_other"] = EmbeddingGemma2CapabilityStatus.Experimental, // 70, 140, 560, 1,120: proposed rows, within the protected bounds
        ["image.batch"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["image.matryoshka"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["image.over_8192_tokens"] = EmbeddingGemma2CapabilityStatus.Unsupported, // rejected before inference
        ["image.over_50_megapixels"] = EmbeddingGemma2CapabilityStatus.Unsupported, // default limit, configurable
        ["image.alpha_compositing"] = EmbeddingGemma2CapabilityStatus.Unsupported, // alpha is dropped, as upstream
        ["image.16_bit_per_channel"] = EmbeddingGemma2CapabilityStatus.Unsupported, // rejected by the decoder; upstream renders white
        // Decoding is in Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp: PNG and JPEG decoded RGB equal to PIL
        // on the fixture images, GIF, BMP and WebP on 12 files written from them; a GIF with transparent pixels is
        // rejected (PIL keeps their palette colour); EXIF orientation ignored unless opted in.
        ["image.decode.png_jpeg"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["image.decode.gif_bmp_webp"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["image.decode.gif_transparency"] = EmbeddingGemma2CapabilityStatus.Unsupported, // EmbeddingGemma2ImageRejection.GifTransparency
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
        // Edge-case videos (proposed video-edge): every frame, metadata and sampled indices equal torchcodec's.
        ["video.variable_frame_rate"] = EmbeddingGemma2CapabilityStatus.Experimental, // index-based sampling, as upstream
        ["video.interlaced"] = EmbeddingGemma2CapabilityStatus.Experimental, // no deinterlacing, as upstream
        ["video.hdr"] = EmbeddingGemma2CapabilityStatus.Experimental, // PQ and HLG signalling; no tone mapping, as upstream
        ["video.audio_track"] = EmbeddingGemma2CapabilityStatus.Unsupported, // ignored, as upstream
        ["video.over_8192_tokens"] = EmbeddingGemma2CapabilityStatus.Unsupported, // rejected before inference
        ["video.frames_over_50_megapixels"] = EmbeddingGemma2CapabilityStatus.Unsupported, // default limit, configurable
        // Decoding is in Starlings.Models.EmbeddingGemma2.Video.FFmpeg on the user's own FFmpeg 8.x (none shipped):
        // metadata, sampled indices and the RGB of every frame equal torchcodec's on the 22 fixture videos (H.264 8/10-bit,
        // HEVC, VP9, AV1; rotation 90; odd sizes) with FFmpeg 8.1 on x86-64.
        ["video.decode.ffmpeg8"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["video.decode.ffmpeg9"] = EmbeddingGemma2CapabilityStatus.Unsupported, // other ABI; rejected with a typed error
        ["video.decode.rotation_90"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["video.decode.rotation_other"] = EmbeddingGemma2CapabilityStatus.Experimental, // -90, 180: equal to torchcodec on the edge-case videos
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

        // Microsoft.Extensions.AI IEmbeddingGenerator, in Starlings.Models.EmbeddingGemma2.Extensions.AI. Supported on
        // the CPU text profile (G7), experimental over CUDA sessions (EmbeddingGemma2.Tests.Cuda).
        ["extensions_ai.embedding_generator"] = EmbeddingGemma2CapabilityStatus.Supported,
        ["extensions_ai.embedding_generator.cuda"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["extensions_ai.dimensions_matryoshka"] = EmbeddingGemma2CapabilityStatus.Supported,

        // ML.NET estimator/transformer, in Starlings.Models.EmbeddingGemma2.MLNet (G7 on the CPU text profile).
        // Persistence: Save throws.
        ["mlnet.embedding_transform"] = EmbeddingGemma2CapabilityStatus.Supported,
        ["mlnet.embedding_transform.cuda"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["mlnet.batching"] = EmbeddingGemma2CapabilityStatus.Supported,
        ["mlnet.persistence"] = EmbeddingGemma2CapabilityStatus.Unsupported,
        ["mlnet.cancellation"] = EmbeddingGemma2CapabilityStatus.Unsupported,
        ["mlnet.training"] = EmbeddingGemma2CapabilityStatus.Unsupported,

        // Native AOT on Linux x64 CPU from the packed packages: no trim/AOT warnings, equal to JIT bit for bit (typed text
        // API and Extensions.AI generator; docs/evidence/aot/). The ML.NET adapter does not run under Native AOT.
        ["aot"] = EmbeddingGemma2CapabilityStatus.Supported,
        ["mlnet.aot"] = EmbeddingGemma2CapabilityStatus.Unsupported,
        // Windows x64: build, weight-free tests and the clean-install drill without the model in CI (ci.yml,
        // ai-dotnet-windows); no model run, so outside the Stable contract. macOS and ARM64: no evidence and no CI job.
        ["windows"] = EmbeddingGemma2CapabilityStatus.Experimental,
        ["macos"] = EmbeddingGemma2CapabilityStatus.Unsupported,
        ["arm64"] = EmbeddingGemma2CapabilityStatus.Unsupported,
    };

    /// <summary>
    /// Status per capability of each profile this package version publishes a contract for, keyed exactly as the
    /// profile's registry manifest (<c>model-pipeline/registry/&lt;profile&gt;/manifest.yaml</c>, <c>capabilities</c>).
    /// </summary>
    public static IReadOnlyDictionary<string, IReadOnlyDictionary<string, EmbeddingGemma2CapabilityStatus>> Profiles { get; } =
        new Dictionary<string, IReadOnlyDictionary<string, EmbeddingGemma2CapabilityStatus>>(StringComparer.Ordinal)
        {
            [StableProfile] = new Dictionary<string, EmbeddingGemma2CapabilityStatus>(StringComparer.Ordinal)
            {
                ["batch"] = EmbeddingGemma2CapabilityStatus.Supported,
                ["streaming"] = EmbeddingGemma2CapabilityStatus.Unsupported,
                ["cancellation"] = EmbeddingGemma2CapabilityStatus.Supported,
                ["persistence"] = EmbeddingGemma2CapabilityStatus.Unsupported,
                ["training"] = EmbeddingGemma2CapabilityStatus.Unsupported,
                ["aot"] = EmbeddingGemma2CapabilityStatus.Supported,
                ["extensions_ai"] = EmbeddingGemma2CapabilityStatus.Supported,
                ["mlnet"] = EmbeddingGemma2CapabilityStatus.Supported,
                ["embedding_pretokenized"] = EmbeddingGemma2CapabilityStatus.Supported,
                ["embedding_text"] = EmbeddingGemma2CapabilityStatus.Supported,
                ["tokenizer_owned"] = EmbeddingGemma2CapabilityStatus.Supported,
                ["prompts_query_document"] = EmbeddingGemma2CapabilityStatus.Supported,
                ["prompts_task"] = EmbeddingGemma2CapabilityStatus.Experimental,
                ["matryoshka"] = EmbeddingGemma2CapabilityStatus.Supported,
                ["input_up_to_2049_tokens"] = EmbeddingGemma2CapabilityStatus.Supported,
                ["input_2050_to_8192_tokens"] = EmbeddingGemma2CapabilityStatus.Experimental,
                ["input_over_8192_tokens"] = EmbeddingGemma2CapabilityStatus.Unsupported,
                ["concurrent_calls"] = EmbeddingGemma2CapabilityStatus.Supported,
                // The text profile; images and videos are the image-fp32-* and video-fp32-* profiles.
                ["image"] = EmbeddingGemma2CapabilityStatus.Unsupported,
                ["audio"] = EmbeddingGemma2CapabilityStatus.Unsupported,
                ["video"] = EmbeddingGemma2CapabilityStatus.Unsupported,
            },
        };
}
