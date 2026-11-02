using Status = Starlings.Models.CapabilityStatus;

namespace Starlings.Models.EmbeddingGemma2;

/// <summary>Support status of one capability (plan section 9). A missing entry never means supported.</summary>
[Obsolete("Use Starlings.Models.CapabilityStatus; removed in the next minor version.")]
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
/// <see cref="Statuses"/> is package-wide. Keys without a provider suffix give the status on the Stable CPU profiles: text
/// keys on <see cref="StableProfile"/>, <c>image.*</c> keys on <see cref="StableImageProfile"/> and <c>video.*</c> keys on
/// <see cref="StableVideoProfile"/> (Linux x64 CPU). Keys with <c>.cuda</c> give the Stable CUDA profiles'
/// (<see cref="StableCudaProfile"/>, <see cref="StableCudaImageProfile"/> and <see cref="StableCudaVideoProfile"/>, Linux
/// x64, RTX 3090). <see cref="ProfileCapabilities"/> holds the capability record of every
/// profile the package serves, generated from the registry manifests; a test keeps the package-wide table consistent
/// with the Stable profiles' records.
/// </remarks>
public static class EmbeddingGemma2Capabilities
{
    /// <summary>The text profile of this package's Stable contract (the README's "Stable contract" section).</summary>
    public const string StableProfile = "embeddinggemma-2/onnx-fp32-cpu";

    /// <summary>The image profile of this package's Stable contract (the README's "Stable contract: images" section).</summary>
    public const string StableImageProfile = "embeddinggemma-2/image-fp32-cpu";

    /// <summary>The video profile of this package's Stable contract (the README's "Stable contract: videos" section).</summary>
    public const string StableVideoProfile = "embeddinggemma-2/video-fp32-cpu";

    /// <summary>The CUDA text profile of this package's Stable contract (the README's "Stable contract: CUDA" section).</summary>
    public const string StableCudaProfile = "embeddinggemma-2/onnx-fp32-cuda";

    /// <summary>The CUDA image profile of this package's Stable contract (the README's "Stable contract: CUDA" section).</summary>
    public const string StableCudaImageProfile = "embeddinggemma-2/image-fp32-cuda";

    /// <summary>The CUDA video profile of this package's Stable contract (the README's "Stable contract: CUDA" section).</summary>
    public const string StableCudaVideoProfile = "embeddinggemma-2/video-fp32-cuda";

    /// <summary>
    /// Status per capability. Evidence for each entry is listed in <c>ai-dotnet/README.md</c> and, for
    /// <see cref="Status.Experimental"/>, in the profile's gate evidence (<c>docs/catalog/README.md</c>).
    /// </summary>
    public static IReadOnlyDictionary<string, Status> Statuses { get; } = new Dictionary<string, Status>(StringComparer.Ordinal)
    {
        // Text on Linux x64 CPU, Stable since 2026-10-15 (the README's Stable contract): every gate passed, and the weekly
        // qualification reruns G1 to G3 on the release sets (G2 on the protected reference embeddings, #60) and a one-hour
        // soak, green. Evidence: G1 on the P-030 tokenizer holdout (232 rows, ids exact) and G2 on the text
        // release set (696 rows: no prompt, query and document; embeddings within 5.1e-7 against 5e-6, padded batches
        // within 2.0e-7 of single calls), the accepted development fixtures (138 rows up to 1,750 tokens, batch
        // independence to 2,049), G3 on MIRACL (nDCG@10 0.8325), G4 one-hour soak (concurrent callers, timed-out
        // requests, plateau), G6 anonymous graph download and offline run, G7 adapter tests and drill consumers.
        ["embedding.pretokenized.cpu"] = Status.Supported,
        // CUDA (use_tf32 0) on an RTX 3090, text and images Stable since 2026-10-16 (the README's "Stable contract: CUDA"):
        // G1 to G3 on the release sets, G4 one-hour soak and G5 (binding round), G6, G7; the per-release GPU regression
        // reruns the CUDA tests, G1 to G3 and a soak smoke test (every check green in run 38857860186).
        ["embedding.pretokenized.cuda"] = Status.Supported,

        // Text through EmbeddingGemma2Tokenizer (owned profile gemma-bpe-embeddinggemma2, Starlings.Tokenizers 0.3.4).
        ["embedding.text"] = Status.Supported,
        ["tokenizer.owned"] = Status.Supported,

        // Explicit named prompts from config_sentence_transformers.json (never inserted silently; default none). The
        // accepted fixtures and the release set cover none, query and document; all 20 names on four public texts are in
        // the proposed task-prompt rows (ids exact, within the protected text bound on CPU and CUDA;
        // docs/evidence/capability-closure-2026-10-14/), proposed rows and so outside the Stable contract.
        ["prompts.query_document"] = Status.Supported,
        ["prompts.task"] = Status.Experimental,
        ["prompt_insertion.implicit"] = Status.Unsupported,

        // Truncate the leading dimensions, then L2-normalize again (equal to sentence-transformers truncate_dim); checked
        // against the reference sliced and renormalized, within the protected bound, and through both adapters (G7).
        ["matryoshka.512"] = Status.Supported,
        ["matryoshka.256"] = Status.Supported,
        ["matryoshka.128"] = Status.Supported,
        ["matryoshka.other_sizes"] = Status.Unsupported,

        // Input length, special tokens and prompt included. Accepted fixtures reach 1,750 tokens, batch-independence
        // checks 2,049; 2,050 to 8,192 tokens is covered only by the proposed long-input rows (8 text rows on CPU and
        // CUDA within the protected bound), so it is outside the Stable contract. Longer input is rejected, never truncated.
        ["input.up_to_2049_tokens"] = Status.Supported,
        ["input.2050_to_8192_tokens"] = Status.Experimental,
        ["input.over_8192_tokens"] = Status.Unsupported,
        ["truncation"] = Status.Unsupported,
        // Text truncation (#193): longer text is rejected, never truncated (ModelProfileCapabilities.Truncation is
        // TruncationPolicy.Rejects).
        ["text.truncation"] = Status.Unsupported,

        ["batch"] = Status.Supported,
        // Opt-in length-sorted chunking, EmbeddingGemma2EncoderOptions.SortByLength (text path): off by default, and the
        // defaults keep the qualified outputs bit for bit. With it, outputs stay within the padded-batch-vs-single
        // bound in the CPU tests; no qualification run yet.
        ["batch.sort_by_length"] = Status.Experimental,
        ["cancellation"] = Status.Supported,
        ["concurrent_calls"] = Status.Supported,
        ["asset_acquisition"] = Status.Supported,
        // The text graph is hosted in the public Hugging Face repository EmbeddingGemma2ModelAssets.HostingRepository. The
        // release drills downloaded it with no token into an empty cache, verified it and ran the search sample offline
        // (G6, docs/evidence/g6-distribution-2026-10-14/).
        ["asset_acquisition.graph_download"] = Status.Supported,

        // Images (EmbeddingGemma2MultimodalEmbedder). Keys without .cuda: the Stable profile image-fp32-cpu (the README's
        // "Stable contract: images", since 2026-10-16); .cuda: the Stable image-fp32-cuda ("Stable contract: CUDA"). From decoded RGB: 42
        // protected rows (28 images alone, 6 interleaved or prompted, 8 text queries) within the binding bounds;
        // preprocessing bit-exact on the 28 supported fixture images; rendered ids exact on all 43 rows. The P-030 image
        // release set (19 rows) passes G1 and G2, image retrieval passes G3, and G4 to G7 passed.
        ["image"] = Status.Supported,
        ["image.cpu"] = Status.Supported,
        ["image.cuda"] = Status.Supported,
        ["image.interleaved_text"] = Status.Supported,
        ["image.multiple_per_input"] = Status.Supported,
        ["image.prompts"] = Status.Supported, // rendered as upstream; the card recommends none
        ["image.budget_280"] = Status.Supported,
        ["image.budget_other"] = Status.Experimental, // 70, 140, 560, 1,120: proposed rows, within the protected bounds
        ["image.batch"] = Status.Supported,
        // Opt-in vision batch limits from a free-memory probe at load (EmbeddingGemma2MultimodalOptions.AutoBatchLimits):
        // off by default, so the qualified defaults are unchanged. The memory estimate is derived from the CUDA
        // observations behind the defaults, not measured per device; no qualification run.
        ["image.auto_batch_limits"] = Status.Experimental,
        ["image.matryoshka"] = Status.Supported,
        ["image.over_8192_tokens"] = Status.Unsupported, // rejected before inference
        ["image.over_50_megapixels"] = Status.Unsupported, // default limit, configurable
        ["image.alpha_compositing"] = Status.Unsupported, // alpha is dropped, as upstream
        ["image.16_bit_per_channel"] = Status.Unsupported, // rejected by the decoder; upstream renders white
        // Decoding is in Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp: PNG and JPEG decoded RGB equal to PIL
        // on the fixture images, GIF, BMP and WebP on 12 files written from them; a GIF with transparent pixels is
        // rejected (PIL keeps their palette colour); EXIF orientation ignored unless opted in.
        ["image.decode.png_jpeg"] = Status.Supported,
        ["image.decode.gif_bmp_webp"] = Status.Experimental,
        ["image.decode.gif_transparency"] = Status.Unsupported, // EmbeddingGemma2ImageRejection.GifTransparency
        ["image.decode.other_formats"] = Status.Unsupported,
        ["image.exif_orientation_opt_in"] = Status.Experimental, // deviation from upstream
        ["image.graph_download"] = Status.Supported, // public ImageHostingRepository: anonymous download, offline run (G6)
        ["extensions_ai.image_embedding_generator"] = Status.Supported,
        // Over CUDA: CudaAdapterTests needs the video set too, which the per-release GPU pod lacks, so it is skipped there.
        ["extensions_ai.image_embedding_generator.cuda"] = Status.Experimental,
        ["mlnet.image_column"] = Status.Unsupported,

        // Videos (EmbeddingGemma2MultimodalEmbedder). Keys without .cuda: the Stable profile video-fp32-cpu (the README's
        // "Stable contract: videos", since 2026-10-17); .cuda: the Stable video-fp32-cuda ("Stable contract: CUDA", RTX
        // 3090). Every gate G0 to G7 passed; the P-030 video release set (11 rows) passes G1 (FFmpeg 8.1 and 9.0 frames
        // exact) and G2 on AVX2 and AVX-512 hosts and on the RTX 3090, video retrieval passes G3. Checkpoint settings only: 1 frame per second, at most 32 frames, 140 soft tokens per frame, no timestamps. From
        // decoded RGB frames: 35 protected rows (22 videos alone, 6 interleaved or prompted, 2 input forms, 5 text
        // queries) within the binding bounds; sampling exact on 20,849 upstream cases; per-frame preprocessing bit-exact
        // on all 32 video inputs; rendered ids exact on all 36 rows.
        ["video"] = Status.Supported,
        ["video.cpu"] = Status.Supported,
        ["video.cuda"] = Status.Supported,
        ["video.decoded_frames_with_timing"] = Status.Supported, // the package samples as upstream
        ["video.pre_sampled_frames"] = Status.Experimental, // caller's choice, at most 32; equal to the file only for upstream's frames
        ["video.frames_without_timing"] = Status.Unsupported, // rejected by default (typed exception)
        ["video.frames_without_timing.upstream_opt_in"] = Status.Experimental, // differs from the file (cosine 0.965)
        ["video.interleaved_text"] = Status.Supported,
        ["video.interleaved_image"] = Status.Supported,
        ["video.multiple_per_input"] = Status.Supported,
        ["video.prompts"] = Status.Supported, // rendered as upstream; the card recommends none
        ["video.batch"] = Status.Supported,
        ["video.matryoshka"] = Status.Supported,
        ["video.sub_2s_one_frame"] = Status.Supported, // upstream quirk kept for parity
        ["video.timestamps"] = Status.Unsupported, // add_timestamps is off in the checkpoint
        ["video.other_fps_max_frames_or_budget"] = Status.Unsupported, // not configurable
        // Edge-case videos (proposed video-edge): every frame, metadata and sampled indices equal torchcodec's.
        ["video.variable_frame_rate"] = Status.Experimental, // index-based sampling, as upstream
        ["video.interlaced"] = Status.Experimental, // no deinterlacing, as upstream
        ["video.hdr"] = Status.Experimental, // PQ and HLG signalling; no tone mapping, as upstream
        ["video.audio_track"] = Status.Unsupported, // ignored, as upstream
        ["video.over_8192_tokens"] = Status.Unsupported, // rejected before inference
        ["video.frames_over_50_megapixels"] = Status.Unsupported, // default limit, configurable
        // Decoding is in Starlings.Models.EmbeddingGemma2.Video.FFmpeg on the user's own FFmpeg 8.x (none shipped):
        // metadata, sampled indices and the RGB of every frame equal torchcodec's on the 22 fixture videos (H.264 8/10-bit,
        // HEVC, VP9, AV1; rotation 90; odd sizes) with FFmpeg 8.1 on x86-64, and the release set's sampled frames (G1,
        // AVX2 and AVX-512 hosts).
        ["video.decode.ffmpeg8"] = Status.Supported,
        ["video.decode.ffmpeg9"] = Status.Unsupported, // the FFmpeg 8 adapter on FFmpeg 9: other ABI, typed error
        // Starlings.Models.EmbeddingGemma2.Video.FFmpeg9 on the user's FFmpeg 9.0: every frame of the 22 fixture videos and
        // the release set's sampled frames equal torchcodec's (protected decoder_ffmpeg9 bound, #64; G1 on the release set).
        ["video.decode.ffmpeg9_adapter"] = Status.Supported,
        ["video.decode.rotation_90"] = Status.Supported,
        ["video.decode.rotation_other"] = Status.Experimental, // -90, 180: equal to torchcodec on the edge-case videos
        ["video.decode.streams"] = Status.Experimental, // seekable streams and memory, equal to files (adapter tests only)
        ["video.decode.arm64_or_non_avx2"] = Status.Untested, // ARM64, x86-64 without AVX2: swscale SIMD paths may differ by a level
        ["video.graph_download"] = Status.Supported, // the same public graphs as images (G6)
        ["extensions_ai.video_embedding_generator"] = Status.Supported,
        // Over CUDA: CudaAdapterTests needs the video set, which the per-release GPU pod lacks, so it is skipped there.
        ["extensions_ai.video_embedding_generator.cuda"] = Status.Experimental,
        ["mlnet.video_column"] = Status.Unsupported,

        // Out of the contract.
        ["audio"] = Status.Unsupported,
        ["media_placeholder_tokens"] = Status.Unsupported,
        ["pair_input"] = Status.Unsupported,
        ["streaming"] = Status.Unsupported,
        ["tf32"] = Status.Unsupported,
        ["bf16"] = Status.Unsupported,
        ["fp16"] = Status.Unsupported,
        ["quantized"] = Status.Unsupported,
        ["tensorrt"] = Status.Unsupported,
        ["training"] = Status.Unsupported,

        // Token counting without embedding (CountTokens, Validate) and TextChunker over the owned tokenizer; text whose
        // length alone proves it above the limit is rejected without tokenizing it
        // (Starlings.Models.Abstractions): weight-free tests on the pinned tokenizer.json. Pooled chunk vectors
        // (TextChunker.EmbedPooled) are a convention with no parity claim and no fidelity gate.
        ["token_counting"] = Status.Experimental,
        ["chunking"] = Status.Experimental,

        // Microsoft.Extensions.AI IEmbeddingGenerator, in Starlings.Models.EmbeddingGemma2.Extensions.AI. Supported on
        // the CPU text profile (G7) and over CUDA sessions (EmbeddingGemma2.Tests.Cuda CudaAdapterTests, per release).
        ["extensions_ai.embedding_generator"] = Status.Supported,
        ["extensions_ai.embedding_generator.cuda"] = Status.Supported,
        ["extensions_ai.dimensions_matryoshka"] = Status.Supported,
        // Cross-request micro-batching (opt-in EmbeddingGemma2EmbeddingGeneratorOptions.MicroBatching, text generator): concurrent single requests share padded batches on one
        // dispatcher thread. Weight-free tests (coalescing, ordering, per-caller cancellation, backpressure, grouping,
        // disposal, failure isolation) and one weight-dependent test against direct calls within the padded-batch
        // tolerance; no benchmark adoption decision and no soak yet. Outputs depend on the concurrent traffic (within
        // tolerance, not bit for bit reproducible). Not part of any Stable profile's contract.
        ["extensions_ai.micro_batching"] = Status.Experimental,
        // EmbedStreamAsync (IAsyncEnumerable of text in, embeddings out in order, grouped as the encoder's batch plan) and
        // EmbeddingGemma2EmbeddingGeneratorOptions (and the multimodal generator's options).ExecutionScheduler (null keeps inline execution; a TaskScheduler moves GenerateAsync and the stream's
        // encoder runs onto it): weight-free tests and equality with GenerateAsync. The profile-level "streaming" key
        // above stays unsupported: these are adapter features, not part of any Stable profile's contract.
        ["extensions_ai.stream"] = Status.Experimental,
        ["extensions_ai.execution_scheduler"] = Status.Experimental,

        // ML.NET estimator/transformer, in Starlings.Models.EmbeddingGemma2.MLNet (G7 on the CPU text profile).
        // Persistence: Save throws.
        ["mlnet.embedding_transform"] = Status.Supported,
        ["mlnet.embedding_transform.cuda"] = Status.Supported,
        ["mlnet.batching"] = Status.Supported,
        ["mlnet.persistence"] = Status.Unsupported,
        ["mlnet.cancellation"] = Status.Unsupported,
        ["mlnet.training"] = Status.Unsupported,

        // Native AOT on Linux x64 CPU from the packed packages: no trim/AOT warnings, equal to JIT bit for bit (typed text
        // API and Extensions.AI generator; docs/evidence/aot/). The ML.NET adapter does not run under Native AOT.
        ["aot"] = Status.Supported,
        // Native AOT with the CUDA natives (tools/aot-check-ai.sh --cuda): one local run, no per-release regression.
        ["aot.cuda"] = Status.Experimental,
        ["mlnet.aot"] = Status.Unsupported,
        // Windows x64: build, weight-free tests and the clean-install drill without the model in CI (ci.yml,
        // ai-dotnet-windows); no model run, so outside the Stable contract. macOS and ARM64: no evidence and no CI job.
        ["windows"] = Status.Experimental,
        ["macos"] = Status.Unsupported,
        ["arm64"] = Status.Unsupported,
    };

    /// <summary>
    /// <see cref="Statuses"/> in this package's former status enum. Equal key for key; the enum values have the same
    /// names and order.
    /// </summary>
    [Obsolete("Use EmbeddingGemma2Capabilities.Statuses; removed in the next minor version.")]
    public static IReadOnlyDictionary<string, EmbeddingGemma2CapabilityStatus> All { get; } = Legacy(Statuses);

    /// <summary>
    /// The capability record of every registry profile this package serves (<see cref="ModelProfileCapabilities"/>),
    /// generated at build time from the profiles' registry manifests by <c>model-pipeline catalog</c>, keyed by profile
    /// id. Each record's <see cref="ModelProfileCapabilities.Capabilities"/> is the manifest's <c>capabilities</c> map
    /// key for key; the package-wide <see cref="Statuses"/> adds finer keys.
    /// </summary>
    public static IReadOnlyDictionary<string, ModelProfileCapabilities> ProfileCapabilities => GeneratedProfileCapabilities.Profiles;

    /// <summary>
    /// Status per capability of each Stable profile, keyed exactly as the profile's registry manifest
    /// (<c>model-pipeline/registry/&lt;profile&gt;/manifest.yaml</c>, <c>capabilities</c>): the Stable entries of
    /// <see cref="ProfileCapabilities"/> in this package's former status enum.
    /// </summary>
    [Obsolete("Use EmbeddingGemma2Capabilities.ProfileCapabilities[profile].Capabilities; removed in the next minor version.")]
    public static IReadOnlyDictionary<string, IReadOnlyDictionary<string, EmbeddingGemma2CapabilityStatus>> Profiles { get; } =
        ProfileCapabilities.Values
            .Where(p => p.Maturity == ProfileMaturity.Stable)
            .ToDictionary(p => p.ProfileId, p => (IReadOnlyDictionary<string, EmbeddingGemma2CapabilityStatus>)Legacy(p.Capabilities), StringComparer.Ordinal);

    [Obsolete("Converts to the former status enum for the obsolete members only.")]
    private static Dictionary<string, EmbeddingGemma2CapabilityStatus> Legacy(IReadOnlyDictionary<string, Status> statuses) =>
        statuses.ToDictionary(e => e.Key, e => (EmbeddingGemma2CapabilityStatus)(int)e.Value, StringComparer.Ordinal);
}
