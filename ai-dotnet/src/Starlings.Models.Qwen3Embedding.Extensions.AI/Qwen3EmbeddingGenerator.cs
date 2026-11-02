using Microsoft.Extensions.AI;
using Starlings.Models.ExtensionsAI;

namespace Starlings.Models.Qwen3Embedding.Extensions.AI;

/// <summary>
/// <c>Qwen/Qwen3-Embedding-0.6B</c> as a Microsoft.Extensions.AI <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/>
/// of <see cref="string"/> to <see cref="Embedding{T}"/> of <see cref="float"/>, over <see cref="Qwen3TextEmbedder"/>:
/// same tokenizer, same graph, same numbers. Experimental (<see cref="Qwen3EmbeddingCapabilities.ExtensionsAI"/>).
/// </summary>
/// <remarks>
/// <para>
/// <b>Prompt.</b> Each request embeds its values after one named prompt: <see cref="DefaultPrompt"/> of the instance
/// (<c>none</c> unless set), or the value of <see cref="PromptPropertyName"/> in
/// <see cref="EmbeddingGenerationOptions.AdditionalProperties"/> for that request (<c>none</c>, <c>document</c> or
/// <c>query</c>). Embed queries with <c>query</c> and documents with <c>none</c>.
/// </para>
/// <para>
/// <b>Execution.</b> <see cref="GenerateAsync"/> tokenizes and runs ONNX Runtime on the calling thread and returns a
/// completed task; failures are in the task, never thrown synchronously. <see cref="GeneratedEmbeddings{T}.Usage"/>
/// reports the token count (prompt and the appended token included, after truncation).
/// </para>
/// <para>
/// <b>Options.</b> <see cref="EmbeddingGenerationOptions.Dimensions"/> must be null or 1024 (Matryoshka dimensions are
/// not offered). <see cref="EmbeddingGenerationOptions.ModelId"/> must be null or the generator's model id.
/// <see cref="EmbeddingGenerationOptions.RawRepresentationFactory"/> must be null, and
/// <see cref="EmbeddingGenerationOptions.AdditionalProperties"/> may hold only <see cref="PromptPropertyName"/>. Anything
/// else fails with <see cref="NotSupportedException"/>.
/// </para>
/// <para>
/// <b>Lifetime.</b> Concurrent calls are safe. An instance from <see cref="Create"/> owns its embedder; one wrapping an
/// existing embedder does not. <see cref="Dispose"/> is idempotent.
/// </para>
/// </remarks>
public sealed class Qwen3EmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    /// <summary>Reported as <see cref="EmbeddingGeneratorMetadata.ProviderName"/>.</summary>
    public const string ProviderName = "starlings.onnxruntime";

    /// <summary>
    /// Key in <see cref="EmbeddingGenerationOptions.AdditionalProperties"/> that sets the named prompt for one request
    /// (a string: <c>none</c>, <c>document</c> or <c>query</c>).
    /// </summary>
    public const string PromptPropertyName = "qwen3_prompt";

    private readonly Qwen3TextEmbedder _embedder;
    private readonly bool _ownsEmbedder;
    private readonly EmbeddingGeneratorMetadata _metadata;
    private int _disposed;

    /// <summary>Wraps an existing embedder, which this instance does not own or dispose.</summary>
    /// <param name="embedder">The embedder.</param>
    /// <param name="defaultPrompt">The named prompt of requests that do not set one.</param>
    public Qwen3EmbeddingGenerator(Qwen3TextEmbedder embedder, string defaultPrompt = "none")
        : this(embedder, defaultPrompt, ownsEmbedder: false)
    {
    }

    private Qwen3EmbeddingGenerator(Qwen3TextEmbedder embedder, string defaultPrompt, bool ownsEmbedder)
    {
        ArgumentNullException.ThrowIfNull(embedder);
        ArgumentNullException.ThrowIfNull(defaultPrompt);
        if (!embedder.Profile.Prompts.ContainsKey(defaultPrompt))
        {
            throw new ArgumentException($"Prompt '{defaultPrompt}' is not declared by {embedder.Profile.Id}.", nameof(defaultPrompt));
        }

        _embedder = embedder;
        _ownsEmbedder = ownsEmbedder;
        DefaultPrompt = defaultPrompt;
        _metadata = new EmbeddingGeneratorMetadata(ProviderName, providerUri: null, embedder.Profile.Repository, embedder.Dimension);
    }

    /// <summary>The named prompt of requests that do not set <see cref="PromptPropertyName"/>.</summary>
    public string DefaultPrompt { get; }

    /// <summary>Provider name, model id (the upstream repository) and dimension (1024).</summary>
    public EmbeddingGeneratorMetadata Metadata => _metadata;

    /// <summary>The underlying embedder.</summary>
    public Qwen3TextEmbedder Embedder => _embedder;

    /// <summary>
    /// Verifies a released directory, loads the embedder (<see cref="Qwen3TextEmbedder.Create"/>) and owns it. Never
    /// downloads.
    /// </summary>
    /// <param name="directory">The released directory of <see cref="Qwen3EmbeddingProfile.Embedding06B"/>.</param>
    /// <param name="options">Threads, batching and verification.</param>
    /// <param name="defaultPrompt">The named prompt of requests that do not set one.</param>
    public static Qwen3EmbeddingGenerator Create(string directory, Qwen3EmbeddingOptions? options = null, string defaultPrompt = "none")
    {
        var embedder = Qwen3TextEmbedder.Create(directory, options);
        try
        {
            return new Qwen3EmbeddingGenerator(embedder, defaultPrompt, ownsEmbedder: true);
        }
        catch
        {
            embedder.Dispose();
            throw;
        }
    }

    /// <inheritdoc />
    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        return EmbeddingGeneratorCore.Run(() =>
        {
            ArgumentNullException.ThrowIfNull(values);
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            cancellationToken.ThrowIfCancellationRequested();
            string prompt = ResolvePrompt(options);
            var texts = EmbeddingGeneratorCore.Texts(values);
            var inputs = new Qwen3Input[texts.Count];
            var counts = new int[texts.Count];
            for (int i = 0; i < inputs.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                inputs[i] = _embedder.Tokenize(texts[i], prompt);
                counts[i] = inputs[i].Length;
            }

            float[][] vectors = _embedder.EmbedTokens(inputs, cancellationToken);
            return Task.FromResult(EmbeddingGeneratorCore.Result(vectors, counts, _metadata.DefaultModelId!, static v => v));
        }, cancellationToken);
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

        return serviceType == typeof(Qwen3TextEmbedder) ? _embedder : null;
    }

    /// <summary>Disposes the embedder if this instance owns it. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0 && _ownsEmbedder)
        {
            _embedder.Dispose();
        }
    }

    private string ResolvePrompt(EmbeddingGenerationOptions? options)
    {
        if (options is null)
        {
            return DefaultPrompt;
        }

        if (options.Dimensions is { } dimensions && dimensions != _embedder.Dimension)
        {
            throw new NotSupportedException($"EmbeddingGenerationOptions.Dimensions = {dimensions} is not supported: only {_embedder.Dimension} (Matryoshka dimensions are not offered).");
        }

        if (!EmbeddingGeneratorCore.CheckOptions(options, _metadata.DefaultModelId!, PromptPropertyName, out object? value))
        {
            return DefaultPrompt;
        }

        return value is string prompt && _embedder.Profile.Prompts.ContainsKey(prompt)
            ? prompt
            : throw new NotSupportedException($"AdditionalProperties['{PromptPropertyName}'] must be one of {string.Join(", ", _embedder.Profile.Prompts.Keys)}.");
    }
}
