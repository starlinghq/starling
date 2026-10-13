namespace Starlings.Models.EmbeddingGemma2.MLNet;

/// <summary>Immutable configuration of the EmbeddingGemma 2 ML.NET transform, beyond the column names.</summary>
public sealed record EmbeddingGemma2EmbeddingOptions
{
    /// <summary>
    /// Name of an optional <see cref="int"/> output column holding each row's token count (special tokens and prompt
    /// included). Default <see langword="null"/>: no token-count column.
    /// </summary>
    public string? TokenCountColumnName { get; init; }

    /// <summary>
    /// Prompt prepended to every text before tokenization. Default <see cref="EmbeddingGemma2Prompt.None"/>: texts are
    /// embedded exactly as given. A prompt already in the data is never detected or removed.
    /// </summary>
    public EmbeddingGemma2Prompt Prompt { get; init; } = EmbeddingGemma2Prompt.None;

    /// <summary>
    /// Rows embedded per inference call. Default 1: each row is embedded on its own, exactly as
    /// <see cref="EmbeddingGemma2TextEmbedder.Embed(string, EmbeddingGemma2Prompt, CancellationToken)"/> does. Values above
    /// 1 enable look-ahead batching on sequential, unshuffled cursors only: a second cursor over the input reads the next
    /// <see cref="BatchSize"/> texts ahead, the batch is embedded with one call, and row ids are compared so a mismatch
    /// fails instead of misaligning vectors. Each batch is right-padded, so values may differ from row-wise ones within
    /// the protected padded-batch tolerance.
    /// </summary>
    public int BatchSize { get; init; } = 1;

    /// <summary>
    /// Encoder options used when the transform loads the graph itself from <see cref="EmbeddingGemma2ModelFiles"/>
    /// (provider, threads, batch limits, <see cref="EmbeddingGemma2EncoderOptions.Dimension"/>, which sets the size of
    /// the embedding column: 768, 512, 256 or 128). The graph's expected SHA-256 is always set to
    /// <see cref="EmbeddingGemma2ModelAssets.ModelSha256"/>. Ignored when wrapping an existing
    /// <see cref="EmbeddingGemma2TextEmbedder"/>, whose encoder's dimension is used.
    /// </summary>
    public EmbeddingGemma2EncoderOptions Encoder { get; init; } = new();

    internal void Validate()
    {
        ArgumentNullException.ThrowIfNull(Prompt);
        ArgumentOutOfRangeException.ThrowIfLessThan(BatchSize, 1, nameof(BatchSize));
        if (TokenCountColumnName is not null)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(TokenCountColumnName, nameof(TokenCountColumnName));
        }

        ArgumentNullException.ThrowIfNull(Encoder);
        if (!EmbeddingGemma2Encoder.IsSupportedDimension(Encoder.Dimension))
        {
            throw new ArgumentOutOfRangeException(nameof(Encoder), Encoder.Dimension, "Supported dimensions are 768, 512, 256 and 128 (Matryoshka truncation).");
        }
    }
}
