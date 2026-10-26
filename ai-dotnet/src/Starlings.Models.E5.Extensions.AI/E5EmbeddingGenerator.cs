using System.Runtime.CompilerServices;
using Microsoft.Extensions.AI;
using Starlings.Models.ExtensionsAI;
using Starlings.Runtime.Onnx.Batching;

namespace Starlings.Models.E5.Extensions.AI;

/// <summary>
/// <c>intfloat/multilingual-e5-small</c> as a Microsoft.Extensions.AI
/// <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/> of <see cref="string"/> to <see cref="Embedding{T}"/> of
/// <see cref="float"/>, over <see cref="E5TextEmbedder"/>: same tokenizer, same encoder, same numbers.
/// </summary>
/// <remarks>
/// <para>
/// <b>Execution: inline unless a scheduler is set (plan 5.1).</b> By default
/// (<see cref="E5EmbeddingGeneratorOptions.ExecutionScheduler"/> null) <see cref="GenerateAsync"/> tokenizes and runs
/// ONNX Runtime on the calling thread and returns an already-completed task; it does not offload to the thread pool.
/// With a scheduler set (Experimental, capability <c>extensions_ai.execution_scheduler</c>), the call returns at once
/// and the same work runs as one task on that scheduler (<see cref="TaskScheduler.Default"/> for the thread pool, or the
/// caller's own, for example one thread per device). Failures are never thrown synchronously: they are in the returned
/// task (faulted, or cancelled when the token was cancelled).
/// </para>
/// <para>
/// <b>Streaming (Experimental, capability <c>extensions_ai.stream</c>).</b>
/// <see cref="EmbedStreamAsync"/> takes an <see cref="IAsyncEnumerable{T}"/> of texts and yields one embedding per text
/// in input order, without materializing the source. Values are tokenized as they arrive and grouped as the encoder
/// groups an explicit batch (<see cref="E5EncoderOptions.MaxBatchSize"/> rows, and <see cref="E5EncoderOptions.MaxBatchTokens"/>
/// padded cells when set), so with <see cref="E5EncoderOptions.SortByLength"/> off the embeddings equal those of one
/// <see cref="GenerateAsync"/> call over the same values bit for bit. A group runs when it is full or the source ends.
/// </para>
/// <para>
/// <b>Batching.</b> One call embeds all values in order with one <see cref="E5TextEmbedder.EmbedBatch(IReadOnlyList{string}, Span{int}, CancellationToken)"/>
/// call: the encoder runs consecutive ONNX Runtime calls of at most <see cref="E5EncoderOptions.MaxBatchSize"/> rows,
/// each right-padded to its own longest row. Results are in input order; <see cref="GeneratedEmbeddings{T}.Usage"/>
/// reports the token count (special tokens and any prefix included, after truncation to 512).
/// </para>
/// <para>
/// <b>Micro-batching (Experimental, opt-in).</b> With <see cref="E5EmbeddingGeneratorOptions.MicroBatching"/> set,
/// a call with exactly one value is tokenized and validated on the calling thread (an invalid value fails only its own
/// call), queued, and embedded together with other callers' queued values on one dispatcher thread; the returned task
/// completes when that batch has run. The embedding then depends on the concurrent traffic within the padded-batch
/// tolerance. Cancelling a queued call removes its value; cancelling a dispatched one cancels its task at once without
/// stopping the batch the other callers wait for. <see cref="Dispose"/> faults queued calls with
/// <see cref="ObjectDisposedException"/> and waits for a running batch. Calls with several values run as described
/// above.
/// </para>
/// <para>
/// <b>Cancellation.</b> The token is checked before each value is tokenized, before each chunk of
/// <c>MaxBatchSize</c> rows, and inside a run: on CPU between graph nodes (ONNX Runtime's terminate flag; latency
/// bounded by the longest single node), on CUDA when the current run completes (latency bounded by one chunk); then
/// the task is cancelled. Nothing is returned
/// for a cancelled call, and the generator stays usable.
/// </para>
/// <para>
/// <b>Options.</b> <see cref="EmbeddingGenerationOptions.Dimensions"/> must be null or the model dimension (384):
/// e5 has no Matryoshka truncation. <see cref="EmbeddingGenerationOptions.ModelId"/> must be null or
/// <see cref="E5EmbeddingGeneratorOptions.ModelId"/>. <see cref="EmbeddingGenerationOptions.RawRepresentationFactory"/>
/// must be null (there is no provider-specific option object). <see cref="EmbeddingGenerationOptions.AdditionalProperties"/>
/// may contain only <see cref="InputTypePropertyName"/>. Anything else fails with <see cref="NotSupportedException"/>
/// rather than being ignored.
/// </para>
/// <para>
/// <b>Threading and lifetime.</b> Concurrent calls are safe. An instance from <see cref="Create"/> owns its encoder
/// and ONNX Runtime session and disposes them; an instance wrapping an existing <see cref="E5TextEmbedder"/> owns
/// nothing. <see cref="Dispose"/> is idempotent; later calls fail with <see cref="ObjectDisposedException"/>. A run
/// already executing when the owned session is disposed finishes first (<c>OnnxModelSession</c> reference count).
/// </para>
/// </remarks>
public sealed class E5EmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    /// <summary>Reported as <see cref="EmbeddingGeneratorMetadata.ProviderName"/>.</summary>
    public const string ProviderName = "starlings.onnxruntime";

    /// <summary>
    /// Key in <see cref="EmbeddingGenerationOptions.AdditionalProperties"/> that overrides
    /// <see cref="E5EmbeddingGeneratorOptions.InputType"/> for one request. The value is an <see cref="E5InputType"/>
    /// or its name as a string (<c>"none"</c>, <c>"query"</c>, <c>"passage"</c>, case-insensitive).
    /// </summary>
    public const string InputTypePropertyName = "e5_input_type";

    private readonly E5TextEmbedder _embedder;
    private readonly bool _ownsEmbedder;
    private readonly E5EmbeddingGeneratorOptions _options;
    private readonly EmbeddingGeneratorMetadata _metadata;
    private readonly MicroBatcher<EncoderInput, float[]>? _batcher;
    private int _disposed;

    /// <summary>Wraps an existing embedder. Neither it nor its encoder is owned or disposed by this instance.</summary>
    public E5EmbeddingGenerator(E5TextEmbedder embedder, E5EmbeddingGeneratorOptions? options = null)
        : this(embedder, options ?? new E5EmbeddingGeneratorOptions(), ownsEmbedder: false)
    {
    }

    private E5EmbeddingGenerator(E5TextEmbedder embedder, E5EmbeddingGeneratorOptions options, bool ownsEmbedder)
    {
        ArgumentNullException.ThrowIfNull(embedder);
        options.Validate();
        _embedder = embedder;
        _options = options;
        _ownsEmbedder = ownsEmbedder;
        _metadata = new EmbeddingGeneratorMetadata(ProviderName, providerUri: null, options.ModelId, embedder.Encoder.Dimension);
        if (options.MicroBatching is { } microBatching)
        {
            var encoder = embedder.Encoder;
            _batcher = new MicroBatcher<EncoderInput, float[]>(
                (inputs, token) => encoder.EmbedBatch(inputs, token), microBatching, static input => input.Length);
        }
    }

    /// <summary>Options in effect.</summary>
    public E5EmbeddingGeneratorOptions Options => _options;

    /// <summary>Provider name, model id and dimension (384), as <see cref="GetService"/> returns it.</summary>
    public EmbeddingGeneratorMetadata Metadata => _metadata;

    /// <summary>The underlying embedder.</summary>
    public E5TextEmbedder Embedder => _embedder;

    /// <summary>
    /// Loads the pinned model and tokenizer from verified local files (<see cref="E5ModelAssets.OpenLocal"/> or
    /// <see cref="E5ModelAssets.OpenDirectory"/>; never downloads) through
    /// <see cref="E5TextEmbedder.Create(E5ModelFiles, E5EncoderOptions?)"/> and owns the resulting embedder, encoder
    /// and session. The model's SHA-256 (<see cref="E5ModelAssets.ModelSha256"/>) and the tokenizer's
    /// (<see cref="E5ModelAssets.TokenizerSha256"/>) are enforced.
    /// </summary>
    public static E5EmbeddingGenerator Create(E5ModelFiles files, E5EmbeddingGeneratorOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(files);
        options ??= new E5EmbeddingGeneratorOptions();
        options.Validate();
        var embedder = E5TextEmbedder.Create(files, options.Encoder);
        try
        {
            return new E5EmbeddingGenerator(embedder, options, ownsEmbedder: true);
        }
        catch
        {
            embedder.Dispose();
            throw;
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// Runs inline unless a scheduler is set: with <see cref="E5EmbeddingGeneratorOptions.ExecutionScheduler"/> null it
    /// runs synchronously on the calling thread and returns a completed task, except a one-value call with
    /// micro-batching on; with a scheduler, the work (tokenizing, and the run or the queueing) is one task on it. The
    /// values are enumerated on the thread that does the work. See the class remarks for batching, micro-batching,
    /// cancellation and options.
    /// </remarks>
    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        return EmbeddingGeneratorCore.Run(() =>
        {
            var texts = Prepare(values, options, cancellationToken);
            return _batcher is not null && texts.Count == 1
                ? GenerateQueuedAsync(_batcher, texts[0], cancellationToken)
                : Task.FromResult(Generate(texts, cancellationToken));
        }, _options.ExecutionScheduler, cancellationToken);
    }

    /// <summary>
    /// Embeds a stream of texts and yields one embedding per text, in input order, without materializing the source.
    /// Experimental (capability <c>extensions_ai.stream</c>).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Request options are resolved once, before the first value, with the rules of <see cref="GenerateAsync"/>. Each
    /// value is tokenized (with the input type's prefix) as it arrives, on the thread that enumerates. Values are grouped
    /// as the encoder groups an explicit batch in input order (<see cref="E5EncoderOptions.MaxBatchSize"/> rows and, when
    /// set, <see cref="E5EncoderOptions.MaxBatchTokens"/> padded cells); each group runs as one encoder call when the next
    /// value does not fit or the source ends, and its embeddings are then yielded. With
    /// <see cref="E5EncoderOptions.SortByLength"/> off the embeddings equal one <see cref="GenerateAsync"/> call over the
    /// same values bit for bit; with it on, a group is sorted only within itself (within the padded-batch tolerance).
    /// </para>
    /// <para>
    /// Runs inline unless a scheduler is set: each group's encoder run is on the enumerating thread, or on
    /// <see cref="E5EmbeddingGeneratorOptions.ExecutionScheduler"/> when set. Micro-batching does not apply. There is no
    /// per-embedding usage. A null value fails the enumeration with <see cref="ArgumentException"/> naming its position;
    /// embeddings of earlier groups have been yielded by then. Cancellation is checked before each value and inside a
    /// run, as for <see cref="GenerateAsync"/>.
    /// </para>
    /// </remarks>
    /// <param name="values">The texts.</param>
    /// <param name="options">Request options (input type, dimensions, model id), as for <see cref="GenerateAsync"/>.</param>
    /// <param name="cancellationToken">Cancels the enumeration.</param>
    public IAsyncEnumerable<Embedding<float>> EmbedStreamAsync(
        IAsyncEnumerable<string> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(values);
        return EmbedStreamCoreAsync(values, options, cancellationToken);
    }

    /// <inheritdoc />
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null)
        {
            return null;
        }

        if (serviceType == typeof(EmbeddingGeneratorMetadata))
        {
            return _metadata;
        }

        if (serviceType.IsInstanceOfType(this))
        {
            return this;
        }

        if (serviceType == typeof(E5TextEmbedder))
        {
            return _embedder;
        }

        return serviceType == typeof(E5Encoder) ? _embedder.Encoder : null;
    }

    /// <summary>Disposes the encoder and session if this instance owns them. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            // Queued calls fault; a running batch finishes before the encoder goes.
            _batcher?.Dispose();
            if (_ownsEmbedder)
            {
                _embedder.Dispose();
            }
        }
    }

    /// <summary>
    /// Statistics of the micro-batcher (batches, items, padding, queue and run time), or null when micro-batching is off.
    /// Experimental.
    /// </summary>
    public MicroBatcherStatistics? MicroBatchingStatistics => _batcher?.GetStatistics();

    private List<string> Prepare(IEnumerable<string> values, EmbeddingGenerationOptions? options, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(values);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        cancellationToken.ThrowIfCancellationRequested();
        string prefix = ResolveInputType(options) switch
        {
            E5InputType.Query => E5TextEmbedder.QueryPrefix,
            E5InputType.Passage => E5TextEmbedder.PassagePrefix,
            _ => string.Empty,
        };

        return EmbeddingGeneratorCore.Texts(values, prefix);
    }

    private async IAsyncEnumerable<Embedding<float>> EmbedStreamCoreAsync(
        IAsyncEnumerable<string> values, EmbeddingGenerationOptions? options, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        cancellationToken.ThrowIfCancellationRequested();
        string prefix = ResolveInputType(options) switch
        {
            E5InputType.Query => E5TextEmbedder.QueryPrefix,
            E5InputType.Passage => E5TextEmbedder.PassagePrefix,
            _ => string.Empty,
        };
        var encoder = _embedder.Encoder;
        var tokenizer = _embedder.Tokenizer;
        await foreach (var embedding in EmbeddingGeneratorCore.Stream(
            values,
            (text, _) =>
            {
                var input = tokenizer.Tokenize(prefix.Length == 0 ? text : prefix + text);
                encoder.Validate(input);
                return input;
            },
            static input => input.Length,
            encoder.Options.MaxBatchSize,
            encoder.Options.MaxBatchTokens,
            (inputs, token) => encoder.EmbedBatch(inputs, token),
            static vector => vector,
            _options.ModelId,
            _options.ExecutionScheduler,
            cancellationToken).ConfigureAwait(false))
        {
            yield return embedding;
        }
    }

    private async Task<GeneratedEmbeddings<Embedding<float>>> GenerateQueuedAsync(MicroBatcher<EncoderInput, float[]> batcher, string text, CancellationToken cancellationToken)
    {
        // Validated here, so an invalid value fails this call alone and never the batch it would have joined.
        var input = _embedder.Tokenizer.Tokenize(text);
        _embedder.Encoder.Validate(input);
        float[] vector = await batcher.EnqueueAsync(input, cancellationToken).ConfigureAwait(false);
        return EmbeddingGeneratorCore.Single(vector, input.Length, _options.ModelId);
    }

    private GeneratedEmbeddings<Embedding<float>> Generate(List<string> texts, CancellationToken cancellationToken)
    {
        var tokenCounts = new int[texts.Count];
        float[][] vectors = _embedder.EmbedBatch(texts, tokenCounts, cancellationToken);
        return EmbeddingGeneratorCore.Result(vectors, tokenCounts, _options.ModelId, static vector => vector);
    }

    private E5InputType ResolveInputType(EmbeddingGenerationOptions? options) =>
        ResolveInputType(options, _options.InputType, _options.ModelId, _embedder.Encoder.Dimension);

    /// <summary>
    /// The request-option rules of <see cref="GenerateAsync"/> (shared with the browser generator): returns the input
    /// type for this request or throws for an unsupported option.
    /// </summary>
    internal static E5InputType ResolveInputType(EmbeddingGenerationOptions? options, E5InputType defaultInputType, string generatorModelId, int dimension)
    {
        if (options is null)
        {
            return defaultInputType;
        }

        if (options.Dimensions is { } dimensions && dimensions != dimension)
        {
            throw new NotSupportedException($"EmbeddingGenerationOptions.Dimensions = {dimensions} is not supported: {generatorModelId} produces {dimension} dimensions and has no truncation (Matryoshka) support.");
        }

        return EmbeddingGeneratorCore.CheckOptions(options, generatorModelId, InputTypePropertyName, out object? value)
            ? ParseInputType(value)
            : defaultInputType;
    }

    private static E5InputType ParseInputType(object? value) => value switch
    {
        E5InputType t when Enum.IsDefined(t) => t,
        // Names only: Enum.TryParse would also accept numbers such as "1".
        string s when s.Length > 0 && char.IsAsciiLetter(s[0]) && Enum.TryParse(s, ignoreCase: true, out E5InputType t) && Enum.IsDefined(t) => t,
        _ => throw new ArgumentException($"AdditionalProperties['{InputTypePropertyName}'] = '{value}' is not an e5 input type; expected {nameof(E5InputType.None)}, {nameof(E5InputType.Query)} or {nameof(E5InputType.Passage)}."),
    };
}
