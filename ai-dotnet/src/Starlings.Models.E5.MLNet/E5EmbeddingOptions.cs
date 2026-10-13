namespace Starlings.Models.E5.MLNet;

/// <summary>
/// Which e5 prefix the transformer prepends to every text before tokenizing. e5 was trained with <c>"query: "</c>
/// and <c>"passage: "</c> prefixes (model card); omitting them degrades retrieval quality. An existing prefix is never
/// detected or removed: with <see cref="Query"/> the text <c>"query: x"</c> becomes <c>"query: query: x"</c>.
/// </summary>
public enum E5TextPrefix
{
    /// <summary>Default. Texts are tokenized exactly as given; the data adds a prefix if wanted.</summary>
    None,

    /// <summary>Prepends <see cref="E5TextEmbedder.QueryPrefix"/> (<c>"query: "</c>), for search queries and symmetric tasks.</summary>
    Query,

    /// <summary>Prepends <see cref="E5TextEmbedder.PassagePrefix"/> (<c>"passage: "</c>), for documents being searched.</summary>
    Passage,
}

/// <summary>Immutable configuration of the e5 ML.NET transform, beyond the column names.</summary>
public sealed record E5EmbeddingOptions
{
    /// <summary>
    /// Name of an optional <see cref="int"/> output column holding each row's token count (special tokens and any
    /// prefix included, after truncation to 512). Default <see langword="null"/>: no token-count column.
    /// </summary>
    public string? TokenCountColumnName { get; init; }

    /// <summary>Prefix prepended to every text. Default <see cref="E5TextPrefix.None"/>.</summary>
    public E5TextPrefix Prefix { get; init; } = E5TextPrefix.None;

    /// <summary>
    /// Rows embedded per inference call. Default 1: each row is embedded on its own, exactly as
    /// <see cref="E5TextEmbedder.Embed"/> does. Values above 1 enable look-ahead batching on sequential, unshuffled
    /// cursors only (see the package README, "Batching"): a second cursor over the input reads the next
    /// <see cref="BatchSize"/> texts ahead, the batch is embedded with one
    /// <see cref="E5TextEmbedder.EmbedBatch(IReadOnlyList{string}, Span{int}, CancellationToken)"/> call, and row ids
    /// are compared so a mismatch fails instead of misaligning vectors. Each batch is right-padded to its own longest
    /// row, so values may differ from row-wise ones within the protected padded-batch tolerance.
    /// </summary>
    public int BatchSize { get; init; } = 1;

    /// <summary>
    /// Encoder options used when the transform loads the model itself from <see cref="E5ModelFiles"/> (provider,
    /// threads, <c>MaxBatchSize</c>). The model's expected SHA-256 is always set to
    /// <see cref="E5ModelAssets.ModelSha256"/>. Ignored when wrapping an existing <see cref="E5TextEmbedder"/>.
    /// </summary>
    public E5EncoderOptions Encoder { get; init; } = new();

    internal string PrefixText => Prefix switch
    {
        E5TextPrefix.Query => E5TextEmbedder.QueryPrefix,
        E5TextPrefix.Passage => E5TextEmbedder.PassagePrefix,
        _ => string.Empty,
    };

    internal void Validate()
    {
        if (!Enum.IsDefined(Prefix))
        {
            throw new ArgumentOutOfRangeException(nameof(Prefix), Prefix, "Unknown e5 prefix.");
        }

        ArgumentOutOfRangeException.ThrowIfLessThan(BatchSize, 1, nameof(BatchSize));
        if (TokenCountColumnName is not null)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(TokenCountColumnName, nameof(TokenCountColumnName));
        }

        ArgumentNullException.ThrowIfNull(Encoder);
    }
}
