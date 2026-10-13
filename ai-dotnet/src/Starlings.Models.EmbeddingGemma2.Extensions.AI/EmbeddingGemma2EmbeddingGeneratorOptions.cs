namespace Starlings.Models.EmbeddingGemma2.Extensions.AI;

/// <summary>Immutable configuration of an <see cref="EmbeddingGemma2EmbeddingGenerator"/>.</summary>
public sealed record EmbeddingGemma2EmbeddingGeneratorOptions
{
    /// <summary>
    /// The model id the pinned asset lock names: <c>google/embeddinggemma-2@914f7f89...</c> (repository and full commit
    /// SHA).
    /// </summary>
    public static string PinnedModelId { get; } = $"{EmbeddingGemma2ModelAssets.Repository}@{EmbeddingGemma2ModelAssets.Revision}";

    /// <summary>
    /// Prompt prepended to every input unless a request overrides it through
    /// <see cref="EmbeddingGemma2EmbeddingGenerator.PromptPropertyName"/>. Default <see cref="EmbeddingGemma2Prompt.None"/>:
    /// texts are embedded exactly as given.
    /// </summary>
    public EmbeddingGemma2Prompt Prompt { get; init; } = EmbeddingGemma2Prompt.None;

    /// <summary>
    /// Output size when a request does not set <c>EmbeddingGenerationOptions.Dimensions</c>: 768, or a Matryoshka size
    /// 512, 256 or 128. Reported as <c>EmbeddingGeneratorMetadata.DefaultModelDimensions</c>. Default 768.
    /// </summary>
    public int Dimensions { get; init; } = EmbeddingGemma2Encoder.NativeDimension;

    /// <summary>
    /// Reported as <c>EmbeddingGeneratorMetadata.DefaultModelId</c> and <c>Embedding.ModelId</c>, and the only value
    /// accepted in <c>EmbeddingGenerationOptions.ModelId</c>. Default <see cref="PinnedModelId"/>.
    /// <see cref="EmbeddingGemma2EmbeddingGenerator.Create(EmbeddingGemma2ModelFiles, EmbeddingGemma2EmbeddingGeneratorOptions?)"/> enforces the pinned graph digest; when wrapping an existing
    /// <see cref="EmbeddingGemma2TextEmbedder"/>, the caller is responsible for it.
    /// </summary>
    public string ModelId { get; init; } = PinnedModelId;

    /// <summary>
    /// Encoder options used by <see cref="EmbeddingGemma2EmbeddingGenerator.Create(EmbeddingGemma2ModelFiles, EmbeddingGemma2EmbeddingGeneratorOptions?)"/> (provider, threads, batch limits).
    /// The graph's expected SHA-256 is always set to <see cref="EmbeddingGemma2ModelAssets.ModelSha256"/>, and the
    /// encoder always produces 768 dimensions so every request can choose its Matryoshka size. Ignored by the
    /// constructor that wraps an existing <see cref="EmbeddingGemma2TextEmbedder"/>.
    /// </summary>
    public EmbeddingGemma2EncoderOptions Encoder { get; init; } = new();

    internal void Validate()
    {
        ArgumentNullException.ThrowIfNull(Prompt);
        if (!EmbeddingGemma2Encoder.IsSupportedDimension(Dimensions))
        {
            throw new ArgumentOutOfRangeException(nameof(Dimensions), Dimensions, "Supported dimensions are 768, 512, 256 and 128 (Matryoshka truncation).");
        }

        ArgumentException.ThrowIfNullOrWhiteSpace(ModelId);
        ArgumentNullException.ThrowIfNull(Encoder);
    }
}
