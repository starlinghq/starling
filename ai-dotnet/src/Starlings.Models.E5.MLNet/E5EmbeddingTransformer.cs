using Microsoft.ML;
using Microsoft.ML.Data;
using Starlings.Models.MLNet;

namespace Starlings.Models.E5.MLNet;

/// <summary>
/// ML.NET transformer: appends an e5 embedding column (<c>Vector&lt;Single, Dimension&gt;</c>, L2-normalized, 384 for
/// multilingual-e5-small) computed from a text column, and optionally a token-count column (<see cref="int"/>).
/// </summary>
/// <remarks>
/// <para>
/// <b>Lazy, one inference per row.</b> <see cref="Transform"/> returns a view and computes nothing. A cursor embeds a
/// row the first time one of its output getters is called on it; every further getter call on that row (the
/// embedding and the token count, any number of times) reuses the result. Rows whose output getters are never called
/// are never embedded; cursors that do not request an output column never read the text column.
/// </para>
/// <para>
/// <b>Batching.</b> With <see cref="E5EmbeddingOptions.BatchSize"/> 1 (default) each row is embedded on its own and
/// equals <see cref="E5TextEmbedder.Embed"/> exactly. Larger values batch on sequential unshuffled cursors through a
/// look-ahead cursor (see <see cref="E5EmbeddingOptions.BatchSize"/>); shuffled cursors and row-to-row mappers
/// (prediction engines) stay row-wise.
/// </para>
/// <para>
/// <b>Threading.</b> The transformer and its views are immutable and may be cursored from several threads at once; a
/// single cursor is used by one thread at a time (the ML.NET contract). Row-wise, <c>GetRowCursorSet</c> wraps each
/// input cursor, so parallel cursors embed concurrently on the shared encoder, which is safe for concurrent calls.
/// </para>
/// <para>
/// <b>Cancellation.</b> ML.NET cursors and getters take no <see cref="CancellationToken"/>, so a getter call that is
/// embedding cannot be interrupted; it returns when the row (or batch) is done. To stop, stop calling
/// <c>MoveNext</c> and dispose the cursor. For cancellable work call <see cref="E5TextEmbedder.EmbedBatch(IReadOnlyList{string}, CancellationToken)"/> directly.
/// </para>
/// <para>
/// <b>Lifetime.</b> A transformer created from <see cref="E5ModelFiles"/> owns its encoder and ONNX Runtime session
/// and disposes them in <see cref="Dispose"/>; one wrapping an existing <see cref="E5TextEmbedder"/> owns nothing.
/// After an owning transformer is disposed, getters of its views throw <see cref="ObjectDisposedException"/>.
/// </para>
/// <para>
/// <b>Persistence is unsupported.</b> Saving (<c>MLContext.Model.Save</c>, directly or inside a chain) throws
/// <see cref="NotSupportedException"/>: the model contract is the pinned asset lock (<see cref="E5ModelAssets.Lock"/>),
/// resolved at run time, plus the column names and options. Save the rest of the pipeline without this transformer
/// and recreate it at load time from <see cref="E5ModelAssets.OpenLocal"/> with the same names and options.
/// </para>
/// </remarks>
public sealed class E5EmbeddingTransformer : ITransformer, IDisposable
{
    private readonly E5Encoder? _ownedEncoder;
    private readonly TextEmbeddingColumnNames _names;
    private readonly E5TextEmbedder _embedder;
    private readonly TextEmbeddingInference _inference;
    private int _disposed;

    /// <summary>
    /// Wraps an existing embedder. Neither it nor its encoder is owned or disposed by this instance.
    /// </summary>
    /// <param name="embedder">Tokenizer and encoder.</param>
    /// <param name="outputColumnName">Name of the embedding column.</param>
    /// <param name="inputColumnName">Name of the text column; defaults to <paramref name="outputColumnName"/> (which then hides it).</param>
    /// <param name="options">Prefix, batching, token-count column; defaults to none, 1, none.</param>
    public E5EmbeddingTransformer(E5TextEmbedder embedder, string outputColumnName, string? inputColumnName = null, E5EmbeddingOptions? options = null)
        : this(embedder, TextEmbeddingColumnNames.Create(outputColumnName, inputColumnName, options?.TokenCountColumnName), options ?? new E5EmbeddingOptions(), ownedEncoder: null)
    {
    }

    internal E5EmbeddingTransformer(E5TextEmbedder embedder, TextEmbeddingColumnNames names, E5EmbeddingOptions options, E5Encoder? ownedEncoder, Action<int>? inferenceObserver = null)
    {
        ArgumentNullException.ThrowIfNull(embedder);
        options.Validate();
        _names = names;
        _ownedEncoder = ownedEncoder;
        Options = options;
        _embedder = embedder;
        _inference = new TextEmbeddingInference(new PrefixedTextEmbedder(embedder, options.PrefixText), inferenceObserver);
    }

    /// <summary>Name of the text column read.</summary>
    public string InputColumnName => _names.Input;

    /// <summary>Name of the embedding column.</summary>
    public string OutputColumnName => _names.Output;

    /// <summary>Options in effect.</summary>
    public E5EmbeddingOptions Options { get; }

    /// <summary>Embedding dimension (384 for multilingual-e5-small).</summary>
    public int Dimension => _embedder.Encoder.Dimension;

    /// <summary>The underlying embedder.</summary>
    public E5TextEmbedder Embedder => _embedder;

    /// <summary>Always <see langword="true"/>: the transform maps each row on its own (prediction engines work).</summary>
    public bool IsRowToRowMapper => true;

    /// <summary>
    /// Loads the pinned model and tokenizer from verified local files (<see cref="E5ModelAssets.OpenLocal"/> or
    /// <see cref="E5ModelAssets.OpenDirectory"/>; never downloads) and owns the resulting encoder and session. The
    /// model's and the tokenizer's SHA-256 digests are enforced.
    /// </summary>
    public static E5EmbeddingTransformer Create(E5ModelFiles files, string outputColumnName, string? inputColumnName = null, E5EmbeddingOptions? options = null)
    {
        var names = TextEmbeddingColumnNames.Create(outputColumnName, inputColumnName, options?.TokenCountColumnName);
        return Create(files, names, options ?? new E5EmbeddingOptions());
    }

    internal static E5EmbeddingTransformer Create(E5ModelFiles files, TextEmbeddingColumnNames names, E5EmbeddingOptions options)
    {
        ArgumentNullException.ThrowIfNull(files);
        options.Validate();
        var tokenizer = E5Tokenizer.FromFile(files.TokenizerPath, E5ModelAssets.TokenizerSha256, options.Encoder.MaxSequenceLength);
        var encoderOptions = options.Encoder with
        {
            Session = options.Encoder.Session with
            {
                ExpectedModelSha256 = E5ModelAssets.ModelSha256,
                KnownModelSha256 = options.Encoder.Session.KnownModelSha256 ?? files.ModelSha256,
            },
        };
        var encoder = E5Encoder.Create(files.ModelPath, encoderOptions);
        try
        {
            return new E5EmbeddingTransformer(new E5TextEmbedder(tokenizer, encoder), names, options, encoder);
        }
        catch
        {
            encoder.Dispose();
            throw;
        }
    }

    /// <summary>
    /// The input schema plus the embedding column and the optional token-count column.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">The text column is missing or is not a scalar text column.</exception>
    public DataViewSchema GetOutputSchema(DataViewSchema inputSchema) => Bind(inputSchema).OutputSchema;

    /// <summary>Returns a lazily transformed view; see the class remarks for when inference runs.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The text column is missing or is not a scalar text column.</exception>
    public IDataView Transform(IDataView input)
    {
        ArgumentNullException.ThrowIfNull(input);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return new TextEmbeddingDataView(input, Bind(input.Schema), _inference, Options.BatchSize);
    }

    /// <summary>A row-wise mapper (no look-ahead batching), used by prediction engines.</summary>
    public IRowToRowMapper GetRowToRowMapper(DataViewSchema inputSchema)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return new TextEmbeddingRowToRowMapper(Bind(inputSchema), _inference);
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
        "Saving the e5 embedding transformer is not supported (capability mlnet.persistence: unsupported). The model is " +
        "resolved at run time from its pinned asset lock (E5ModelAssets.Lock: intfloat/multilingual-e5-small at a full " +
        "commit SHA, SHA-256 per file), not embedded in the ML.NET model file. Save the pipeline without this transformer " +
        "and, after loading it, append mlContext.Transforms.ApplyE5Embedding(E5ModelAssets.OpenLocal(), ...) with the same " +
        "column names and options.";

    private TextEmbeddingBindings Bind(DataViewSchema inputSchema)
    {
        ArgumentNullException.ThrowIfNull(inputSchema);
        return TextEmbeddingBindings.Create(inputSchema, _names, Dimension);
    }
}
