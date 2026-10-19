using Microsoft.Extensions.AI;
using Starlings.Runtime.Onnx.Batching;

namespace Starlings.Models.EmbeddingGemma2.Extensions.AI;

/// <summary>
/// <c>google/embeddinggemma-2</c> text embeddings as a Microsoft.Extensions.AI
/// <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/> of <see cref="string"/> to <see cref="Embedding{T}"/> of
/// <see cref="float"/>, over <see cref="EmbeddingGemma2TextEmbedder"/>: same tokenizer, same encoder, same numbers.
/// Text only.
/// </summary>
/// <remarks>
/// <para>
/// <b>Synchronous execution (plan 5.1).</b> <see cref="GenerateAsync"/> tokenizes and runs ONNX Runtime on the
/// calling thread and returns an already-completed task. A caller that must not block wraps the call in
/// <c>Task.Run</c> itself. Failures are never thrown synchronously: they are in the returned task (faulted, or
/// cancelled when the token was cancelled).
/// </para>
/// <para>
/// <b>Prompts.</b> Explicit only: <see cref="EmbeddingGemma2EmbeddingGeneratorOptions.Prompt"/> (default
/// <see cref="EmbeddingGemma2Prompt.None"/>) or, per request, <see cref="PromptPropertyName"/> in
/// <see cref="EmbeddingGenerationOptions.AdditionalProperties"/>. A prompt is never detected or de-duplicated.
/// </para>
/// <para>
/// <b>Dimensions.</b> <see cref="EmbeddingGenerationOptions.Dimensions"/> selects a Matryoshka size: 768, 512, 256 or
/// 128 (keep the leading dimensions, L2-normalize again). Any other value, or a size larger than the wrapped encoder
/// produces, fails with <see cref="NotSupportedException"/>. Null means
/// <see cref="EmbeddingGemma2EmbeddingGeneratorOptions.Dimensions"/>.
/// </para>
/// <para>
/// <b>Batching and usage.</b> One call embeds all values in order with one
/// <see cref="EmbeddingGemma2TextEmbedder.EmbedBatch(IReadOnlyList{string}, EmbeddingGemma2Prompt, Span{int}, CancellationToken)"/>
/// call (chunked by the encoder's row and padded-cell limits). <see cref="GeneratedEmbeddings{T}.Usage"/> reports the
/// token count, special tokens and prompt included. An input above 8,192 tokens fails the call with
/// <see cref="EmbeddingGemma2InputTooLongException"/>; nothing is truncated.
/// </para>
/// <para>
/// <b>Micro-batching (Experimental, opt-in).</b> With <see cref="EmbeddingGemma2EmbeddingGeneratorOptions.MicroBatching"/>
/// set, a call with exactly one value is tokenized and validated on the calling thread (an input above 8,192 tokens
/// fails only its own call), queued, and embedded together with other callers' queued values on one dispatcher thread;
/// the returned task completes when that batch has run. The embedding then depends on the concurrent traffic within the
/// padded-batch tolerance. Cancelling a queued call removes its value; cancelling a dispatched one cancels its task at
/// once without stopping the batch the other callers wait for. <see cref="Dispose"/> faults queued calls with
/// <see cref="ObjectDisposedException"/> and waits for a running batch. Calls with several values run as described
/// above.
/// </para>
/// <para>
/// <b>Cancellation.</b> The token is checked before each value is tokenized, before each chunk, and inside a run (on CPU
/// between graph nodes, on CUDA when the current run completes); then the task is cancelled. Nothing is returned for a cancelled call,
/// and the generator stays usable.
/// </para>
/// <para>
/// <b>Options.</b> <see cref="EmbeddingGenerationOptions.ModelId"/> must be null or
/// <see cref="EmbeddingGemma2EmbeddingGeneratorOptions.ModelId"/>; <see cref="EmbeddingGenerationOptions.RawRepresentationFactory"/>
/// must be null; <see cref="EmbeddingGenerationOptions.AdditionalProperties"/> may contain only
/// <see cref="PromptPropertyName"/>. Anything else fails with <see cref="NotSupportedException"/> rather than being ignored.
/// </para>
/// <para>
/// <b>Threading and lifetime.</b> Concurrent calls are safe. An instance from <see cref="Create(EmbeddingGemma2ModelFiles, EmbeddingGemma2EmbeddingGeneratorOptions?)"/> owns its encoder and
/// ONNX Runtime session and disposes them; an instance wrapping an existing <see cref="EmbeddingGemma2TextEmbedder"/>
/// owns nothing. <see cref="Dispose"/> is idempotent; later calls fail with <see cref="ObjectDisposedException"/>.
/// </para>
/// </remarks>
public sealed class EmbeddingGemma2EmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    /// <summary>Reported as <see cref="EmbeddingGeneratorMetadata.ProviderName"/>.</summary>
    public const string ProviderName = "starlings.onnxruntime";

    /// <summary>
    /// Key in <see cref="EmbeddingGenerationOptions.AdditionalProperties"/> that overrides
    /// <see cref="EmbeddingGemma2EmbeddingGeneratorOptions.Prompt"/> for one request. The value is an
    /// <see cref="EmbeddingGemma2Prompt"/> or a prompt name as a string: <c>"none"</c> or one of the 20 names of
    /// <see cref="EmbeddingGemma2Prompt.Named"/> (exact, case-sensitive).
    /// </summary>
    public const string PromptPropertyName = "embeddinggemma2_prompt";

    private readonly EmbeddingGemma2TextEmbedder _embedder;
    private readonly EmbeddingGemma2Encoder? _ownedEncoder;
    private readonly EmbeddingGemma2EmbeddingGeneratorOptions _options;
    private readonly EmbeddingGeneratorMetadata _metadata;
    private readonly MicroBatcher<EmbeddingGemma2Input, float[]>? _batcher;
    private int _disposed;

    /// <summary>Wraps an existing embedder. Neither it nor its encoder is owned or disposed by this instance.</summary>
    /// <exception cref="ArgumentException"><see cref="EmbeddingGemma2EmbeddingGeneratorOptions.Dimensions"/> is larger than the encoder's dimension.</exception>
    public EmbeddingGemma2EmbeddingGenerator(EmbeddingGemma2TextEmbedder embedder, EmbeddingGemma2EmbeddingGeneratorOptions? options = null)
        : this(embedder, options ?? new EmbeddingGemma2EmbeddingGeneratorOptions(), ownedEncoder: null)
    {
    }

    private EmbeddingGemma2EmbeddingGenerator(EmbeddingGemma2TextEmbedder embedder, EmbeddingGemma2EmbeddingGeneratorOptions options, EmbeddingGemma2Encoder? ownedEncoder)
    {
        ArgumentNullException.ThrowIfNull(embedder);
        options.Validate();
        if (options.Dimensions > embedder.Encoder.Dimension)
        {
            throw new ArgumentException($"Dimensions = {options.Dimensions} exceeds the encoder's {embedder.Encoder.Dimension} dimensions; truncation cannot widen an embedding.", nameof(options));
        }

        _embedder = embedder;
        _options = options;
        _ownedEncoder = ownedEncoder;
        _metadata = new EmbeddingGeneratorMetadata(ProviderName, providerUri: null, options.ModelId, options.Dimensions);
        if (options.MicroBatching is { } microBatching)
        {
            var encoder = embedder.Encoder;
            _batcher = new MicroBatcher<EmbeddingGemma2Input, float[]>(
                (inputs, token) => encoder.EmbedBatch(inputs, token), microBatching, static input => input.Length);
        }
    }

    /// <summary>Options in effect.</summary>
    public EmbeddingGemma2EmbeddingGeneratorOptions Options => _options;

    /// <summary>Provider name, model id and default dimensions, as <see cref="GetService"/> returns it.</summary>
    public EmbeddingGeneratorMetadata Metadata => _metadata;

    /// <summary>The underlying embedder.</summary>
    public EmbeddingGemma2TextEmbedder Embedder => _embedder;

    /// <summary>
    /// Loads the pinned graph and the owned tokenizer (<see cref="EmbeddingGemma2Tokenizer"/>) from verified local files
    /// (<see cref="EmbeddingGemma2ModelAssets.OpenLocal"/> or <see cref="EmbeddingGemma2ModelAssets.OpenDirectory"/>;
    /// never downloads) and owns the resulting encoder and session. The graph's and the tokenizer's SHA-256 are enforced
    /// and the encoder produces 768 dimensions.
    /// </summary>
    public static EmbeddingGemma2EmbeddingGenerator Create(EmbeddingGemma2ModelFiles files, EmbeddingGemma2EmbeddingGeneratorOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(files);
        return Create(files, EmbeddingGemma2Tokenizer.FromFile(files.TokenizerPath, EmbeddingGemma2ModelAssets.TokenizerSha256), options);
    }

    /// <summary>
    /// Like <see cref="Create(EmbeddingGemma2ModelFiles, EmbeddingGemma2EmbeddingGeneratorOptions?)"/>, with another
    /// <see cref="IEmbeddingGemma2Tokenizer"/> for text (it must reproduce the pinned <c>tokenizer.json</c>).
    /// </summary>
    /// <param name="files">Verified files.</param>
    /// <param name="tokenizer">The tokenizer.</param>
    /// <param name="options">Generator options.</param>
    public static EmbeddingGemma2EmbeddingGenerator Create(EmbeddingGemma2ModelFiles files, IEmbeddingGemma2Tokenizer tokenizer, EmbeddingGemma2EmbeddingGeneratorOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(files);
        ArgumentNullException.ThrowIfNull(tokenizer);
        options ??= new EmbeddingGemma2EmbeddingGeneratorOptions();
        options.Validate();
        var encoderOptions = options.Encoder with
        {
            Dimension = EmbeddingGemma2Encoder.NativeDimension,
            Session = options.Encoder.Session with { ExpectedModelSha256 = EmbeddingGemma2ModelAssets.ModelSha256 },
        };
        var encoder = EmbeddingGemma2Encoder.Create(files.ModelPath, encoderOptions);
        try
        {
            return new EmbeddingGemma2EmbeddingGenerator(new EmbeddingGemma2TextEmbedder(tokenizer, encoder), options, encoder);
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
    /// remarks for prompts, dimensions, batching, micro-batching and cancellation.
    /// </remarks>
    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var (texts, prompt, dimensions) = Prepare(values, options, cancellationToken);
            return _batcher is not null && texts.Count == 1
                ? GenerateQueuedAsync(_batcher, texts[0], prompt, dimensions, cancellationToken)
                : Task.FromResult(Generate(texts, prompt, dimensions, cancellationToken));
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

        if (serviceType == typeof(EmbeddingGemma2TextEmbedder))
        {
            return _embedder;
        }

        return serviceType == typeof(EmbeddingGemma2Encoder) ? _embedder.Encoder : null;
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

    private (List<string> Texts, EmbeddingGemma2Prompt Prompt, int Dimensions) Prepare(IEnumerable<string> values, EmbeddingGenerationOptions? options, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(values);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        cancellationToken.ThrowIfCancellationRequested();
        var (prompt, dimensions) = Resolve(options);

        var texts = new List<string>(values is ICollection<string> c ? c.Count : 0);
        foreach (string? value in values)
        {
            texts.Add(value ?? throw new ArgumentException($"Value {texts.Count} is null.", nameof(values)));
        }

        return (texts, prompt, dimensions);
    }

    private async Task<GeneratedEmbeddings<Embedding<float>>> GenerateQueuedAsync(
        MicroBatcher<EmbeddingGemma2Input, float[]> batcher, string text, EmbeddingGemma2Prompt prompt, int dimensions, CancellationToken cancellationToken)
    {
        // Validated here, so an invalid value fails this call alone and never the batch it would have joined.
        var input = _embedder.Tokenizer.Tokenize(prompt.Apply(text));
        _embedder.Encoder.Validate(input);
        float[] vector = await batcher.EnqueueAsync(input, cancellationToken).ConfigureAwait(false);
        float[] output = vector.Length == dimensions ? vector : EmbeddingGemma2Encoder.Truncate(vector, dimensions);
        var result = new GeneratedEmbeddings<Embedding<float>>(1)
        {
            new Embedding<float>(output) { ModelId = _options.ModelId, CreatedAt = DateTimeOffset.UtcNow },
        };
        result.Usage = new UsageDetails { InputTokenCount = input.Length, TotalTokenCount = input.Length };
        return result;
    }

    private GeneratedEmbeddings<Embedding<float>> Generate(List<string> texts, EmbeddingGemma2Prompt prompt, int dimensions, CancellationToken cancellationToken)
    {
        var tokenCounts = new int[texts.Count];
        float[][] vectors = _embedder.EmbedBatch(texts, prompt, tokenCounts, cancellationToken);
        var createdAt = DateTimeOffset.UtcNow;
        var result = new GeneratedEmbeddings<Embedding<float>>(vectors.Length);
        foreach (float[] vector in vectors)
        {
            float[] output = vector.Length == dimensions ? vector : EmbeddingGemma2Encoder.Truncate(vector, dimensions);
            result.Add(new Embedding<float>(output) { ModelId = _options.ModelId, CreatedAt = createdAt });
        }

        long tokens = 0;
        foreach (int count in tokenCounts)
        {
            tokens += count;
        }

        result.Usage = new UsageDetails { InputTokenCount = tokens, TotalTokenCount = tokens };
        return result;
    }

    private (EmbeddingGemma2Prompt Prompt, int Dimensions) Resolve(EmbeddingGenerationOptions? options)
    {
        if (options is null)
        {
            return (_options.Prompt, _options.Dimensions);
        }

        int dimensions = _options.Dimensions;
        if (options.Dimensions is { } requested)
        {
            if (!EmbeddingGemma2Encoder.IsSupportedDimension(requested))
            {
                throw new NotSupportedException($"EmbeddingGenerationOptions.Dimensions = {requested} is not supported: EmbeddingGemma 2 supports the Matryoshka sizes 768, 512, 256 and 128.");
            }

            if (requested > _embedder.Encoder.Dimension)
            {
                throw new NotSupportedException($"EmbeddingGenerationOptions.Dimensions = {requested} is not supported: the encoder produces {_embedder.Encoder.Dimension} dimensions and truncation cannot widen an embedding.");
            }

            dimensions = requested;
        }

        if (options.ModelId is { } modelId && !string.Equals(modelId, _options.ModelId, StringComparison.Ordinal))
        {
            throw new NotSupportedException($"EmbeddingGenerationOptions.ModelId = '{modelId}' is not supported: this generator serves only '{_options.ModelId}'.");
        }

        if (options.RawRepresentationFactory is not null)
        {
            throw new NotSupportedException("EmbeddingGenerationOptions.RawRepresentationFactory is not supported: this generator has no provider-specific options object.");
        }

        var prompt = _options.Prompt;
        if (options.AdditionalProperties is { Count: > 0 } properties)
        {
            foreach (var (key, value) in properties)
            {
                if (!string.Equals(key, PromptPropertyName, StringComparison.Ordinal))
                {
                    throw new NotSupportedException($"EmbeddingGenerationOptions.AdditionalProperties['{key}'] is not supported; the only supported key is '{PromptPropertyName}'.");
                }

                prompt = value switch
                {
                    EmbeddingGemma2Prompt p => p,
                    string name when EmbeddingGemma2Prompt.TryFromName(name, out var p) => p,
                    _ => throw new ArgumentException($"AdditionalProperties['{PromptPropertyName}'] = '{value}' is not an EmbeddingGemma 2 prompt; expected an EmbeddingGemma2Prompt, 'none' or one of: {string.Join(", ", EmbeddingGemma2Prompt.Named.Keys)} (case-sensitive)."),
                };
            }
        }

        return (prompt, dimensions);
    }
}
