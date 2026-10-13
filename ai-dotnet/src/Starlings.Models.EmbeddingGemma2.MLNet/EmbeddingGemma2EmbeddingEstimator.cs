using Microsoft.ML;

namespace Starlings.Models.EmbeddingGemma2.MLNet;

/// <summary>
/// ML.NET estimator for <see cref="EmbeddingGemma2EmbeddingTransformer"/>. <see cref="Fit"/> trains nothing: it
/// validates the input schema and returns a transformer over the pretrained model (from an existing embedder, or loaded
/// from verified local files, in which case each call to <see cref="Fit"/> loads a new session owned by the returned
/// transformer).
/// </summary>
public sealed class EmbeddingGemma2EmbeddingEstimator : IEstimator<EmbeddingGemma2EmbeddingTransformer>
{
    private readonly EmbeddingGemma2TextEmbedder? _embedder;
    private readonly EmbeddingGemma2ModelFiles? _files;
    private readonly IEmbeddingGemma2Tokenizer? _tokenizer;
    private readonly EmbeddingGemma2ColumnNames _names;
    private readonly EmbeddingGemma2EmbeddingOptions _options;

    internal EmbeddingGemma2EmbeddingEstimator(EmbeddingGemma2TextEmbedder embedder, EmbeddingGemma2ColumnNames names, EmbeddingGemma2EmbeddingOptions options)
    {
        ArgumentNullException.ThrowIfNull(embedder);
        options.Validate();
        _embedder = embedder;
        _names = names;
        _options = options;
    }

    internal EmbeddingGemma2EmbeddingEstimator(EmbeddingGemma2ModelFiles files, IEmbeddingGemma2Tokenizer tokenizer, EmbeddingGemma2ColumnNames names, EmbeddingGemma2EmbeddingOptions options)
    {
        ArgumentNullException.ThrowIfNull(files);
        ArgumentNullException.ThrowIfNull(tokenizer);
        options.Validate();
        _files = files;
        _tokenizer = tokenizer;
        _names = names;
        _options = options;
    }

    /// <summary>Size of the embedding column, declared before the model is loaded.</summary>
    public int Dimension => _embedder?.Encoder.Dimension ?? _options.Encoder.Dimension;

    /// <summary>Validates the input schema and returns the transformer. Nothing is trained and no row is read.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The text column is missing or is not a scalar text column.</exception>
    public EmbeddingGemma2EmbeddingTransformer Fit(IDataView input)
    {
        ArgumentNullException.ThrowIfNull(input);
        EmbeddingGemma2Bindings.Create(input.Schema, _names, Dimension);
        return _embedder is not null
            ? new EmbeddingGemma2EmbeddingTransformer(_embedder, _names, _options, ownedEncoder: null)
            : EmbeddingGemma2EmbeddingTransformer.Create(_files!, _tokenizer!, _names, _options);
    }

    /// <summary>The input shape plus the embedding column (vector of Single, known size) and the optional token-count column.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The text column is missing or is not a scalar text column.</exception>
    public SchemaShape GetOutputSchema(SchemaShape inputSchema) => EmbeddingGemma2Bindings.CreateShape(inputSchema, _names, Dimension);
}

/// <summary>Creates the EmbeddingGemma 2 embedding estimator from <c>mlContext.Transforms</c>.</summary>
public static class EmbeddingGemma2EmbeddingCatalogExtensions
{
    /// <summary>
    /// Embeds <paramref name="inputColumnName"/> (text) into <paramref name="outputColumnName"/>
    /// (<c>Vector&lt;Single, Dimension&gt;</c>) with an existing embedder, which the pipeline neither owns nor disposes.
    /// </summary>
    /// <param name="catalog">The transforms catalog.</param>
    /// <param name="embedder">Tokenizer and encoder.</param>
    /// <param name="outputColumnName">Name of the embedding column.</param>
    /// <param name="inputColumnName">Name of the text column; defaults to <paramref name="outputColumnName"/>.</param>
    /// <param name="options">Prompt (default none), batch size (default 1), token-count column (default none).</param>
    public static EmbeddingGemma2EmbeddingEstimator ApplyEmbeddingGemma2Embedding(
        this TransformsCatalog catalog, EmbeddingGemma2TextEmbedder embedder, string outputColumnName, string? inputColumnName = null, EmbeddingGemma2EmbeddingOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        return new EmbeddingGemma2EmbeddingEstimator(embedder, EmbeddingGemma2ColumnNames.Create(outputColumnName, inputColumnName, options?.TokenCountColumnName), options ?? new EmbeddingGemma2EmbeddingOptions());
    }

    /// <summary>
    /// Embeds <paramref name="inputColumnName"/> (text) into <paramref name="outputColumnName"/>, loading the pinned graph
    /// and the owned tokenizer (<see cref="EmbeddingGemma2Tokenizer"/>) from verified local files (SHA-256 enforced,
    /// never downloads). Each fitted transformer owns its session; dispose it.
    /// </summary>
    /// <param name="catalog">The transforms catalog.</param>
    /// <param name="files">From <see cref="EmbeddingGemma2ModelAssets.OpenLocal"/> or <see cref="EmbeddingGemma2ModelAssets.OpenDirectory"/>.</param>
    /// <param name="outputColumnName">Name of the embedding column.</param>
    /// <param name="inputColumnName">Name of the text column; defaults to <paramref name="outputColumnName"/>.</param>
    /// <param name="options">Prompt, batch size, token-count column, encoder options (including the dimension).</param>
    public static EmbeddingGemma2EmbeddingEstimator ApplyEmbeddingGemma2Embedding(
        this TransformsCatalog catalog, EmbeddingGemma2ModelFiles files, string outputColumnName, string? inputColumnName = null, EmbeddingGemma2EmbeddingOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        ArgumentNullException.ThrowIfNull(files);
        var tokenizer = EmbeddingGemma2Tokenizer.FromFile(files.TokenizerPath, EmbeddingGemma2ModelAssets.TokenizerSha256);
        return new EmbeddingGemma2EmbeddingEstimator(files, tokenizer, EmbeddingGemma2ColumnNames.Create(outputColumnName, inputColumnName, options?.TokenCountColumnName), options ?? new EmbeddingGemma2EmbeddingOptions());
    }

    /// <summary>
    /// Like the overload without a tokenizer, with another <see cref="IEmbeddingGemma2Tokenizer"/> for text (it must
    /// reproduce the pinned <c>tokenizer.json</c>). Each fitted transformer owns its session; dispose it.
    /// </summary>
    /// <param name="catalog">The transforms catalog.</param>
    /// <param name="files">From <see cref="EmbeddingGemma2ModelAssets.OpenLocal"/> or <see cref="EmbeddingGemma2ModelAssets.OpenDirectory"/>.</param>
    /// <param name="tokenizer">A tokenizer that reproduces the pinned <c>tokenizer.json</c>.</param>
    /// <param name="outputColumnName">Name of the embedding column.</param>
    /// <param name="inputColumnName">Name of the text column; defaults to <paramref name="outputColumnName"/>.</param>
    /// <param name="options">Prompt, batch size, token-count column, encoder options (including the dimension).</param>
    public static EmbeddingGemma2EmbeddingEstimator ApplyEmbeddingGemma2Embedding(
        this TransformsCatalog catalog, EmbeddingGemma2ModelFiles files, IEmbeddingGemma2Tokenizer tokenizer, string outputColumnName, string? inputColumnName = null, EmbeddingGemma2EmbeddingOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(catalog);
        return new EmbeddingGemma2EmbeddingEstimator(files, tokenizer, EmbeddingGemma2ColumnNames.Create(outputColumnName, inputColumnName, options?.TokenCountColumnName), options ?? new EmbeddingGemma2EmbeddingOptions());
    }
}
