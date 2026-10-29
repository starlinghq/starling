using Microsoft.ML;
using Starlings.Models.MLNet;

namespace Starlings.Models.E5.MLNet;

/// <summary>
/// ML.NET estimator for <see cref="E5EmbeddingTransformer"/>. <see cref="Fit"/> trains nothing: it validates the input
/// schema and returns a transformer over the pretrained model (from an existing embedder, or loaded from verified
/// local files, in which case each call to <see cref="Fit"/> loads a new session owned by the returned transformer).
/// </summary>
public sealed class E5EmbeddingEstimator : IEstimator<E5EmbeddingTransformer>
{
    /// <summary>Embedding dimension of multilingual-e5-small (<see cref="E5Encoder.NativeDimension"/>), declared before the model is loaded.</summary>
    public const int E5SmallDimension = E5Encoder.NativeDimension;

    private readonly E5TextEmbedder? _embedder;
    private readonly E5ModelFiles? _files;
    private readonly TextEmbeddingColumnNames _names;
    private readonly E5EmbeddingOptions _options;

    internal E5EmbeddingEstimator(E5TextEmbedder embedder, TextEmbeddingColumnNames names, E5EmbeddingOptions options)
    {
        ArgumentNullException.ThrowIfNull(embedder);
        options.Validate();
        _embedder = embedder;
        _names = names;
        _options = options;
    }

    internal E5EmbeddingEstimator(E5ModelFiles files, TextEmbeddingColumnNames names, E5EmbeddingOptions options)
    {
        ArgumentNullException.ThrowIfNull(files);
        options.Validate();
        _files = files;
        _names = names;
        _options = options;
    }

    private int Dimension => _embedder?.Encoder.Dimension ?? E5SmallDimension;

    /// <summary>
    /// Validates the input schema and returns the transformer. Nothing is trained and no row is read.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">The text column is missing or is not a scalar text column.</exception>
    public E5EmbeddingTransformer Fit(IDataView input)
    {
        ArgumentNullException.ThrowIfNull(input);
        TextEmbeddingBindings.Create(input.Schema, _names, Dimension);
        if (_embedder is not null)
        {
            return new E5EmbeddingTransformer(_embedder, _names, _options, ownedEncoder: null);
        }

        var transformer = E5EmbeddingTransformer.Create(_files!, _names, _options);
        if (transformer.Dimension != E5SmallDimension)
        {
            transformer.Dispose();
            throw new NotSupportedException($"The model produces {transformer.Dimension} dimensions; the estimator declared {E5SmallDimension} (multilingual-e5-small).");
        }

        return transformer;
    }

    /// <summary>The input shape plus the embedding column (vector of Single, known size) and the optional token-count column.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The text column is missing or is not a scalar text column.</exception>
    public SchemaShape GetOutputSchema(SchemaShape inputSchema) => TextEmbeddingBindings.CreateShape(inputSchema, _names, Dimension);
}

/// <summary>Creates the e5 embedding estimator from <c>mlContext.Transforms</c>.</summary>
public static class E5EmbeddingCatalogExtensions
{
    /// <summary>
    /// Embeds <paramref name="inputColumnName"/> (text) into <paramref name="outputColumnName"/>
    /// (<c>Vector&lt;Single, 384&gt;</c>) with an existing embedder, which the pipeline neither owns nor disposes.
    /// </summary>
    /// <param name="catalog">The transforms catalog.</param>
    /// <param name="embedder">Tokenizer and encoder.</param>
    /// <param name="outputColumnName">Name of the embedding column.</param>
    /// <param name="inputColumnName">Name of the text column; defaults to <paramref name="outputColumnName"/>.</param>
    /// <param name="options">Prefix (default none), batch size (default 1), token-count column (default none).</param>
    public static E5EmbeddingEstimator ApplyE5Embedding(
        this TransformsCatalog catalog, E5TextEmbedder embedder, string outputColumnName, string? inputColumnName = null, E5EmbeddingOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        return new E5EmbeddingEstimator(embedder, TextEmbeddingColumnNames.Create(outputColumnName, inputColumnName, options?.TokenCountColumnName), options ?? new E5EmbeddingOptions());
    }

    /// <summary>
    /// Embeds <paramref name="inputColumnName"/> (text) into <paramref name="outputColumnName"/>
    /// (<c>Vector&lt;Single, 384&gt;</c>), loading the pinned model from verified local files at <c>Fit</c>
    /// (model and tokenizer SHA-256 enforced, never downloads). Each fitted transformer owns its session; dispose it.
    /// </summary>
    /// <param name="catalog">The transforms catalog.</param>
    /// <param name="files">From <see cref="E5ModelAssets.OpenLocal"/> or <see cref="E5ModelAssets.OpenDirectory"/>.</param>
    /// <param name="outputColumnName">Name of the embedding column.</param>
    /// <param name="inputColumnName">Name of the text column; defaults to <paramref name="outputColumnName"/>.</param>
    /// <param name="options">Prefix, batch size, token-count column, encoder options.</param>
    public static E5EmbeddingEstimator ApplyE5Embedding(
        this TransformsCatalog catalog, E5ModelFiles files, string outputColumnName, string? inputColumnName = null, E5EmbeddingOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        return new E5EmbeddingEstimator(files, TextEmbeddingColumnNames.Create(outputColumnName, inputColumnName, options?.TokenCountColumnName), options ?? new E5EmbeddingOptions());
    }
}
