using Starlings.Runtime.Onnx.Batching;

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

    /// <summary>
    /// Cross-request micro-batching (Experimental, capability <c>extensions_ai.micro_batching</c>), or null (default):
    /// off, every call runs on the calling thread exactly as without this option. When set, a call with exactly one
    /// value is tokenized and validated on the calling thread, then queued; one dispatcher thread embeds the queued values
    /// of concurrent callers in padded batches (<see cref="MicroBatcher{TItem, TResult}"/>), and the call's task
    /// completes when its batch has run. Calls with several values still run as one explicit batch on the calling
    /// thread. A queued value's embedding then depends on the other values in its batch: equal to a single call within
    /// the profile's padded-batch tolerance, not bit for bit reproducible.
    /// </summary>
    public MicroBatchingOptions? MicroBatching { get; init; }

    internal void Validate()
    {
        if (!Enum.IsDefined(InputType))
        {
            throw new ArgumentOutOfRangeException(nameof(InputType), InputType, "Unknown e5 input type.");
        }

        ArgumentException.ThrowIfNullOrWhiteSpace(ModelId);
        ArgumentNullException.ThrowIfNull(Encoder);
        MicroBatching?.Validate();
    }
}
