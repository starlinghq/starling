using Microsoft.Extensions.AI;
using Starlings.Models.E5.Extensions.AI;

namespace Starlings.Models.E5.Browser;

/// <summary>
/// <c>intfloat/multilingual-e5-small</c> in the browser as a Microsoft.Extensions.AI
/// <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/> of <see cref="string"/> to <see cref="Embedding{T}"/> of
/// <see cref="float"/>, over <see cref="E5BrowserEmbedder"/>. Request options follow
/// <see cref="E5EmbeddingGenerator"/> exactly (input type prefix through
/// <see cref="E5EmbeddingGenerator.InputTypePropertyName"/>, <c>Dimensions</c> 384 or null, <c>ModelId</c> the
/// generator's or null, anything else rejected with <see cref="NotSupportedException"/>).
/// </summary>
/// <remarks>
/// Unlike the native generator this one is asynchronous: onnxruntime-web runs return promises. One call embeds all
/// values with one <see cref="E5BrowserEmbedder.EmbedBatchAsync(IReadOnlyList{string}, int[], CancellationToken)"/>
/// call (chunks of <see cref="E5EncoderOptions.MaxBatchSize"/> rows). There is no micro-batching. Failures are in the
/// returned task, never thrown synchronously.
/// </remarks>
public sealed class E5BrowserEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    /// <summary>Reported as <see cref="EmbeddingGeneratorMetadata.ProviderName"/>.</summary>
    public const string ProviderName = "starlings.onnxruntime-web";

    private readonly E5BrowserEmbedder _embedder;
    private readonly bool _ownsEmbedder;
    private readonly E5InputType _inputType;
    private readonly string _modelId;
    private readonly EmbeddingGeneratorMetadata _metadata;
    private int _disposed;

    /// <summary>
    /// Wraps an embedder.
    /// </summary>
    /// <param name="embedder">The embedder.</param>
    /// <param name="inputType">Prefix applied to every value unless a request overrides it. Default none.</param>
    /// <param name="modelId">Reported model id; default <see cref="E5EmbeddingGeneratorOptions.PinnedModelId"/>.</param>
    /// <param name="ownsEmbedder">When <see langword="true"/>, <see cref="Dispose"/> also disposes the embedder.</param>
    public E5BrowserEmbeddingGenerator(E5BrowserEmbedder embedder, E5InputType inputType = E5InputType.None, string? modelId = null, bool ownsEmbedder = false)
    {
        ArgumentNullException.ThrowIfNull(embedder);
        if (!Enum.IsDefined(inputType))
        {
            throw new ArgumentOutOfRangeException(nameof(inputType), inputType, "Unknown e5 input type.");
        }

        _embedder = embedder;
        _ownsEmbedder = ownsEmbedder;
        _inputType = inputType;
        _modelId = modelId ?? E5EmbeddingGeneratorOptions.PinnedModelId;
        ArgumentException.ThrowIfNullOrWhiteSpace(_modelId, nameof(modelId));
        _metadata = new EmbeddingGeneratorMetadata(ProviderName, providerUri: null, _modelId, embedder.Encoder.Dimension);
    }

    /// <summary>The underlying embedder.</summary>
    public E5BrowserEmbedder Embedder => _embedder;

    /// <summary>Provider name, model id and dimension (384).</summary>
    public EmbeddingGeneratorMetadata Metadata => _metadata;

    /// <inheritdoc />
    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        try
        {
            ArgumentNullException.ThrowIfNull(values);
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            cancellationToken.ThrowIfCancellationRequested();
            var inputType = E5EmbeddingGenerator.ResolveInputType(options, _inputType, _modelId, _embedder.Encoder.Dimension);
            string prefix = inputType switch
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

            return GenerateCoreAsync(texts, cancellationToken);
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

        if (serviceType == typeof(E5BrowserEmbedder))
        {
            return _embedder;
        }

        return serviceType == typeof(E5BrowserEncoder) ? _embedder.Encoder : null;
    }

    /// <summary>Disposes the embedder if this instance owns it. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0 && _ownsEmbedder)
        {
            _embedder.Dispose();
        }
    }

    private async Task<GeneratedEmbeddings<Embedding<float>>> GenerateCoreAsync(List<string> texts, CancellationToken cancellationToken)
    {
        var tokenCounts = new int[texts.Count];
        float[][] vectors = await _embedder.EmbedBatchAsync(texts, tokenCounts, cancellationToken).ConfigureAwait(false);
        var createdAt = DateTimeOffset.UtcNow;
        var result = new GeneratedEmbeddings<Embedding<float>>(vectors.Length);
        foreach (float[] vector in vectors)
        {
            result.Add(new Embedding<float>(vector) { ModelId = _modelId, CreatedAt = createdAt });
        }

        long tokens = 0;
        foreach (int count in tokenCounts)
        {
            tokens += count;
        }

        result.Usage = new UsageDetails { InputTokenCount = tokens, TotalTokenCount = tokens };
        return result;
    }
}
