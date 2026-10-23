using Microsoft.Extensions.AI;
using Starlings.Runtime.Onnx.Batching;

namespace Starlings.Models.E5.Extensions.AI;

/// <summary>
/// <c>intfloat/multilingual-e5-small</c> as a Microsoft.Extensions.AI
/// <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/> of <see cref="string"/> to <see cref="Embedding{T}"/> of
/// <see cref="float"/>, over <see cref="E5TextEmbedder"/>: same tokenizer, same encoder, same numbers.
/// </summary>
/// <remarks>
/// <para>
/// <b>Synchronous execution (plan 5.1).</b> <see cref="GenerateAsync"/> tokenizes and runs ONNX Runtime on the
/// calling thread and returns an already-completed task. It does not offload to the thread pool; a caller that must
/// not block wraps the call in <c>Task.Run</c> itself. Failures are never thrown synchronously: they are in the
/// returned task (faulted, or cancelled when the token was cancelled).
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
    private readonly E5Encoder? _ownedEncoder;
    private readonly E5EmbeddingGeneratorOptions _options;
    private readonly EmbeddingGeneratorMetadata _metadata;
    private readonly MicroBatcher<EncoderInput, float[]>? _batcher;
    private int _disposed;

    /// <summary>Wraps an existing embedder. Neither it nor its encoder is owned or disposed by this instance.</summary>
    public E5EmbeddingGenerator(E5TextEmbedder embedder, E5EmbeddingGeneratorOptions? options = null)
        : this(embedder, options ?? new E5EmbeddingGeneratorOptions(), ownedEncoder: null)
    {
    }

    private E5EmbeddingGenerator(E5TextEmbedder embedder, E5EmbeddingGeneratorOptions options, E5Encoder? ownedEncoder)
    {
        ArgumentNullException.ThrowIfNull(embedder);
        options.Validate();
        _embedder = embedder;
        _options = options;
        _ownedEncoder = ownedEncoder;
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
    /// <see cref="E5ModelAssets.OpenDirectory"/>; never downloads) and owns the resulting encoder and session. The
    /// model's SHA-256 (<see cref="E5ModelAssets.ModelSha256"/>) and the tokenizer's
    /// (<see cref="E5ModelAssets.TokenizerSha256"/>) are enforced.
    /// </summary>
    public static E5EmbeddingGenerator Create(E5ModelFiles files, E5EmbeddingGeneratorOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(files);
        options ??= new E5EmbeddingGeneratorOptions();
        options.Validate();
        var tokenizer = E5Tokenizer.FromFile(files.TokenizerPath, E5ModelAssets.TokenizerSha256, options.Encoder.MaxSequenceLength);
        var encoderOptions = options.Encoder with
        {
            Session = options.Encoder.Session with { ExpectedModelSha256 = E5ModelAssets.ModelSha256 },
        };
        var encoder = E5Encoder.Create(files.ModelPath, encoderOptions);
        try
        {
            return new E5EmbeddingGenerator(new E5TextEmbedder(tokenizer, encoder), options, encoder);
        }
        catch
        {
            encoder.Dispose();
            throw;
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// Runs synchronously and returns a completed task, except a one-value call with micro-batching on; see the class
    /// remarks for batching, micro-batching, cancellation and options.
    /// </remarks>
    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var texts = Prepare(values, options, cancellationToken);
            return _batcher is not null && texts.Count == 1
                ? GenerateQueuedAsync(_batcher, texts[0], cancellationToken)
                : Task.FromResult(Generate(texts, cancellationToken));
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            return Task.FromCanceled<GeneratedEmbeddings<Embedding<float>>>(cancellationToken);
        }
#pragma warning disable CA1031 // Every failure is reported through the returned task, never thrown synchronously.
        catch (Exception ex)
#pragma warning restore CA1031
        {
            return Task.FromException<GeneratedEmbeddings<Embedding<float>>>(ex);
        }
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
            _ownedEncoder?.Dispose();
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

        var texts = new List<string>(values is ICollection<string> c ? c.Count : 0);
        foreach (string? value in values)
        {
            if (value is null)
            {
                throw new ArgumentException($"Value {texts.Count} is null.", nameof(values));
            }

            texts.Add(prefix.Length == 0 ? value : prefix + value);
        }

        return texts;
    }

    private async Task<GeneratedEmbeddings<Embedding<float>>> GenerateQueuedAsync(MicroBatcher<EncoderInput, float[]> batcher, string text, CancellationToken cancellationToken)
    {
        // Validated here, so an invalid value fails this call alone and never the batch it would have joined.
        var input = _embedder.Tokenizer.Tokenize(text);
        _embedder.Encoder.Validate(input);
        float[] vector = await batcher.EnqueueAsync(input, cancellationToken).ConfigureAwait(false);
        var result = new GeneratedEmbeddings<Embedding<float>>(1)
        {
            new Embedding<float>(vector) { ModelId = _options.ModelId, CreatedAt = DateTimeOffset.UtcNow },
        };
        result.Usage = new UsageDetails { InputTokenCount = input.Length, TotalTokenCount = input.Length };
        return result;
    }

    private GeneratedEmbeddings<Embedding<float>> Generate(List<string> texts, CancellationToken cancellationToken)
    {
        var tokenCounts = new int[texts.Count];
        float[][] vectors = _embedder.EmbedBatch(texts, tokenCounts, cancellationToken);
        var createdAt = DateTimeOffset.UtcNow;
        var result = new GeneratedEmbeddings<Embedding<float>>(vectors.Length);
        foreach (float[] vector in vectors)
        {
            result.Add(new Embedding<float>(vector) { ModelId = _options.ModelId, CreatedAt = createdAt });
        }

        long tokens = 0;
        foreach (int count in tokenCounts)
        {
            tokens += count;
        }

        result.Usage = new UsageDetails { InputTokenCount = tokens, TotalTokenCount = tokens };
        return result;
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

        if (options.ModelId is { } modelId && !string.Equals(modelId, generatorModelId, StringComparison.Ordinal))
        {
            throw new NotSupportedException($"EmbeddingGenerationOptions.ModelId = '{modelId}' is not supported: this generator serves only '{generatorModelId}'.");
        }

        if (options.RawRepresentationFactory is not null)
        {
            throw new NotSupportedException("EmbeddingGenerationOptions.RawRepresentationFactory is not supported: this generator has no provider-specific options object.");
        }

        var inputType = defaultInputType;
        if (options.AdditionalProperties is { Count: > 0 } properties)
        {
            foreach (var (key, value) in properties)
            {
                if (!string.Equals(key, InputTypePropertyName, StringComparison.Ordinal))
                {
                    throw new NotSupportedException($"EmbeddingGenerationOptions.AdditionalProperties['{key}'] is not supported; the only supported key is '{InputTypePropertyName}'.");
                }

                inputType = ParseInputType(value);
            }
        }

        return inputType;
    }

    private static E5InputType ParseInputType(object? value) => value switch
    {
        E5InputType t when Enum.IsDefined(t) => t,
        // Names only: Enum.TryParse would also accept numbers such as "1".
        string s when s.Length > 0 && char.IsAsciiLetter(s[0]) && Enum.TryParse(s, ignoreCase: true, out E5InputType t) && Enum.IsDefined(t) => t,
        _ => throw new ArgumentException($"AdditionalProperties['{InputTypePropertyName}'] = '{value}' is not an e5 input type; expected {nameof(E5InputType.None)}, {nameof(E5InputType.Query)} or {nameof(E5InputType.Passage)}."),
    };
}
