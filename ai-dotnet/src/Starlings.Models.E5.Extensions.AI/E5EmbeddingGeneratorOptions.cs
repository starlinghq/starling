namespace Starlings.Models.E5.Extensions.AI;

/// <summary>Immutable configuration of an <see cref="E5EmbeddingGenerator"/>.</summary>
public sealed record E5EmbeddingGeneratorOptions
{
    /// <summary>
    /// The model id the pinned asset lock names: <c>intfloat/multilingual-e5-small@614241f6...</c> (repository and
    /// full commit SHA).
    /// </summary>
    public static string PinnedModelId { get; } = $"{E5ModelAssets.Repository}@{E5ModelAssets.Revision}";

    /// <summary>
    /// Prefix applied to every input unless a request overrides it through
    /// <see cref="E5EmbeddingGenerator.InputTypePropertyName"/>. Default <see cref="E5InputType.None"/>: no prefix.
    /// </summary>
    public E5InputType InputType { get; init; } = E5InputType.None;

    /// <summary>
    /// Reported as <c>EmbeddingGeneratorMetadata.DefaultModelId</c> and <c>Embedding.ModelId</c>, and the only
    /// value accepted in <c>EmbeddingGenerationOptions.ModelId</c>. Default <see cref="PinnedModelId"/>.
    /// <see cref="E5EmbeddingGenerator.Create"/> enforces the pinned model and tokenizer digests, so the default is
    /// accurate there; when wrapping an existing <see cref="E5TextEmbedder"/>, the caller is responsible for it.
    /// </summary>
    public string ModelId { get; init; } = PinnedModelId;

    /// <summary>
    /// Encoder options used by <see cref="E5EmbeddingGenerator.Create"/> (provider, threads, <c>MaxBatchSize</c>).
    /// The model's expected SHA-256 is always set to <see cref="E5ModelAssets.ModelSha256"/>. Ignored by the
    /// constructor that wraps an existing <see cref="E5TextEmbedder"/>.
    /// </summary>
    public E5EncoderOptions Encoder { get; init; } = new();

    internal void Validate()
    {
        if (!Enum.IsDefined(InputType))
        {
            throw new ArgumentOutOfRangeException(nameof(InputType), InputType, "Unknown e5 input type.");
        }

        ArgumentException.ThrowIfNullOrWhiteSpace(ModelId);
        ArgumentNullException.ThrowIfNull(Encoder);
    }
}
