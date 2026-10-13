using Microsoft.ML;
using Microsoft.ML.Data;

namespace Starlings.Models.EmbeddingGemma2.MLNet;

/// <summary>
/// ML.NET transformer: appends an EmbeddingGemma 2 embedding column (<c>Vector&lt;Single, Dimension&gt;</c>,
/// L2-normalized; 768 or a Matryoshka size) computed from a text column, and optionally a token-count column
/// (<see cref="int"/>). Text only.
/// </summary>
/// <remarks>
/// <para>
/// <b>Lazy, one inference per row.</b> <see cref="Transform"/> returns a view and computes nothing. A cursor embeds a
/// row the first time one of its output getters is called on it; further getter calls on that row reuse the result.
/// Rows whose output getters are never called are never embedded.
/// </para>
/// <para>
/// <b>Prompts.</b> <see cref="EmbeddingGemma2EmbeddingOptions.Prompt"/> (default none) is prepended to every text; the
/// transform never adds one implicitly.
/// </para>
/// <para>
/// <b>Batching.</b> With <see cref="EmbeddingGemma2EmbeddingOptions.BatchSize"/> 1 (default) each row is embedded on its
/// own and equals <see cref="EmbeddingGemma2TextEmbedder.Embed(string, EmbeddingGemma2Prompt, CancellationToken)"/>
/// exactly. Larger values batch on sequential unshuffled cursors through a look-ahead cursor; shuffled cursors and
/// row-to-row mappers (prediction engines) stay row-wise.
/// </para>
/// <para>
/// <b>Errors.</b> A row above 8,192 tokens throws <see cref="EmbeddingGemma2InputTooLongException"/> from the getter;
/// nothing is truncated.
/// </para>
/// <para>
/// <b>Threading.</b> The transformer and its views are immutable and may be cursored from several threads at once; a
/// single cursor is used by one thread at a time (the ML.NET contract).
/// </para>
/// <para>
/// <b>Cancellation.</b> ML.NET cursors and getters take no <see cref="CancellationToken"/>, so a getter call that is
/// embedding cannot be interrupted. For cancellable work call <see cref="EmbeddingGemma2TextEmbedder"/> directly.
/// </para>
/// <para>
/// <b>Lifetime.</b> A transformer created from <see cref="EmbeddingGemma2ModelFiles"/> owns its encoder and ONNX Runtime
/// session and disposes them in <see cref="Dispose"/>; one wrapping an existing <see cref="EmbeddingGemma2TextEmbedder"/>
/// owns nothing. After an owning transformer is disposed, getters of its views throw <see cref="ObjectDisposedException"/>.
/// </para>
/// <para>
/// <b>Persistence is unsupported.</b> Saving throws <see cref="NotSupportedException"/>: the model contract is the pinned
/// asset lock (<see cref="EmbeddingGemma2ModelAssets.Lock"/>), resolved at run time, plus the column names and options.
/// </para>
/// </remarks>
public sealed class EmbeddingGemma2EmbeddingTransformer : ITransformer, IDisposable
{
    private readonly EmbeddingGemma2Encoder? _ownedEncoder;
    private readonly EmbeddingGemma2ColumnNames _names;
    private readonly EmbeddingGemma2Inference _inference;
    private int _disposed;

    /// <summary>Wraps an existing embedder. Neither it nor its encoder is owned or disposed by this instance.</summary>
    /// <param name="embedder">Tokenizer and encoder.</param>
    /// <param name="outputColumnName">Name of the embedding column.</param>
    /// <param name="inputColumnName">Name of the text column; defaults to <paramref name="outputColumnName"/> (which then hides it).</param>
    /// <param name="options">Prompt, batching, token-count column; defaults to none, 1, none.</param>
    public EmbeddingGemma2EmbeddingTransformer(EmbeddingGemma2TextEmbedder embedder, string outputColumnName, string? inputColumnName = null, EmbeddingGemma2EmbeddingOptions? options = null)
        : this(embedder, EmbeddingGemma2ColumnNames.Create(outputColumnName, inputColumnName, options?.TokenCountColumnName), options ?? new EmbeddingGemma2EmbeddingOptions(), ownedEncoder: null)
    {
    }

    internal EmbeddingGemma2EmbeddingTransformer(EmbeddingGemma2TextEmbedder embedder, EmbeddingGemma2ColumnNames names, EmbeddingGemma2EmbeddingOptions options, EmbeddingGemma2Encoder? ownedEncoder, Action<int>? inferenceObserver = null)
    {
        ArgumentNullException.ThrowIfNull(embedder);
        options.Validate();
        _names = names;
        _ownedEncoder = ownedEncoder;
        Options = options;
        _inference = new EmbeddingGemma2Inference(embedder, options.Prompt, inferenceObserver);
    }

    /// <summary>Name of the text column read.</summary>
    public string InputColumnName => _names.Input;

    /// <summary>Name of the embedding column.</summary>
    public string OutputColumnName => _names.Output;

    /// <summary>Options in effect.</summary>
    public EmbeddingGemma2EmbeddingOptions Options { get; }

    /// <summary>Embedding dimension (the encoder's: 768, 512, 256 or 128).</summary>
    public int Dimension => _inference.Embedder.Encoder.Dimension;

    /// <summary>The underlying embedder.</summary>
    public EmbeddingGemma2TextEmbedder Embedder => _inference.Embedder;

    /// <summary>Always <see langword="true"/>: the transform maps each row on its own (prediction engines work).</summary>
    public bool IsRowToRowMapper => true;

    /// <summary>
    /// Loads the pinned graph from verified local files (never downloads), with <paramref name="tokenizer"/> for text,
    /// and owns the resulting encoder and session. The graph's SHA-256 is enforced.
    /// </summary>
    public static EmbeddingGemma2EmbeddingTransformer Create(EmbeddingGemma2ModelFiles files, IEmbeddingGemma2Tokenizer tokenizer, string outputColumnName, string? inputColumnName = null, EmbeddingGemma2EmbeddingOptions? options = null)
    {
        var names = EmbeddingGemma2ColumnNames.Create(outputColumnName, inputColumnName, options?.TokenCountColumnName);
        return Create(files, tokenizer, names, options ?? new EmbeddingGemma2EmbeddingOptions());
    }

    internal static EmbeddingGemma2EmbeddingTransformer Create(EmbeddingGemma2ModelFiles files, IEmbeddingGemma2Tokenizer tokenizer, EmbeddingGemma2ColumnNames names, EmbeddingGemma2EmbeddingOptions options)
    {
        ArgumentNullException.ThrowIfNull(files);
        ArgumentNullException.ThrowIfNull(tokenizer);
        options.Validate();
        var encoderOptions = options.Encoder with
        {
            Session = options.Encoder.Session with { ExpectedModelSha256 = EmbeddingGemma2ModelAssets.ModelSha256 },
        };
        var encoder = EmbeddingGemma2Encoder.Create(files.ModelPath, encoderOptions);
        try
        {
            return new EmbeddingGemma2EmbeddingTransformer(new EmbeddingGemma2TextEmbedder(tokenizer, encoder), names, options, encoder);
        }
        catch
        {
            encoder.Dispose();
            throw;
        }
    }

    /// <summary>The input schema plus the embedding column and the optional token-count column.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The text column is missing or is not a scalar text column.</exception>
    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => Bind(inputSchema).OutputSchema;

    /// <summary>Returns a lazily transformed view; see the class remarks for when inference runs.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The text column is missing or is not a scalar text column.</exception>
    public IDataView Transform(IDataView input)
    {
        ArgumentNullException.ThrowIfNull(input);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return new EmbeddingGemma2EmbeddingDataView(input, Bind(input.Schema), _inference, Options.BatchSize);
    }

    /// <summary>A row-wise mapper (no look-ahead batching), used by prediction engines.</summary>
    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return new EmbeddingGemma2RowToRowMapper(Bind(inputSchema), _inference);
    }

    /// <summary>Unsupported: always throws. See the class remarks, "Persistence is unsupported".</summary>
    /// <exception cref="NotSupportedException">Always.</exception>
    void ICanSaveModel.Save(ModelSaveContext ctx) => throw new NotSupportedException(PersistenceUnsupportedMessage);

    /// <summary>Disposes the encoder and session if this instance owns them. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _ownedEncoder?.Dispose();
        }
    }

    internal const string PersistenceUnsupportedMessage =
        "Saving the EmbeddingGemma 2 embedding transformer is not supported (capability mlnet.persistence: unsupported). " +
        "The model is resolved at run time from its pinned asset lock (EmbeddingGemma2ModelAssets.Lock: google/embeddinggemma-2 " +
        "at a full commit SHA, SHA-256 per file), not embedded in the ML.NET model file. Save the pipeline without this " +
        "transformer and, after loading it, append mlContext.Transforms.ApplyEmbeddingGemma2Embedding(...) with the same " +
        "column names and options.";

    private EmbeddingGemma2Bindings Bind(DataViewSchema inputSchema)
    {
        ArgumentNullException.ThrowIfNull(inputSchema);
        return EmbeddingGemma2Bindings.Create(inputSchema, _names, Dimension);
    }
}
