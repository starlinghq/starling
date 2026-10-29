using Microsoft.Extensions.AI;

namespace Starlings.Models.EmbeddingGemma2.Extensions.AI;

/// <summary>
/// <c>google/embeddinggemma-2</c> image, video and interleaved text/image/video embeddings (experimental) as
/// Microsoft.Extensions.AI embedding generators, over <see cref="EmbeddingGemma2MultimodalEmbedder"/>: same preprocessing,
/// same graphs, same numbers. Two input shapes, one instance:
/// <list type="bullet">
/// <item><see cref="IEmbeddingGenerator{TInput, TEmbedding}"/> of <see cref="DataContent"/>: one image (media type
/// <c>image/*</c>) or one video (<c>video/*</c>) per value, embedded alone, as upstream embeds it.</item>
/// <item><see cref="IEmbeddingGenerator{TInput, TEmbedding}"/> of <see cref="IEnumerable{T}"/> of <see cref="AIContent"/>:
/// one input per value, its <see cref="TextContent"/>, image and video <see cref="DataContent"/> parts in order (a medium
/// becomes its expanded placeholders at its position; text parts are concatenated with no separator).</item>
/// </list>
/// </summary>
/// <remarks>
/// <para>
/// <b>Why these shapes.</b> Microsoft.Extensions.AI represents media as <see cref="DataContent"/> and leaves the input type
/// of <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/> to the implementation; a separate interface instantiation
/// keeps <c>IEmbeddingGenerator&lt;string, Embedding&lt;float&gt;&gt;</c> (text, <see cref="EmbeddingGemma2EmbeddingGenerator"/>)
/// unchanged and lets consumers ask for exactly the modality they need. Text-only values of the content shape run on the
/// released text graph and equal <see cref="EmbeddingGemma2EmbeddingGenerator"/>.
/// </para>
/// <para>
/// <b>Decoding.</b> <see cref="DataContent"/> holds encoded bytes; <see cref="EmbeddingGemma2MultimodalEmbeddingGeneratorOptions.ImageDecoder"/>
/// turns them into RGB (for example <c>EmbeddingGemma2ImageDecoder.Decode</c> of
/// <c>Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp</c>, which reproduces PIL), and
/// <see cref="EmbeddingGemma2MultimodalEmbeddingGeneratorOptions.VideoDecoder"/> into an <see cref="EmbeddingGemma2Video"/>
/// (for example <c>EmbeddingGemma2VideoDecoder.Decode</c> of <c>Starlings.Models.EmbeddingGemma2.Video.FFmpeg</c>,
/// which reproduces torchcodec on the user's own FFmpeg 8.x). Without the matching decoder, image or video values fail
/// with <see cref="NotSupportedException"/>. This package has no decoder or native dependency of its own.
/// </para>
/// <para>
/// <b>Rejected content</b> (<see cref="NotSupportedException"/>, never ignored): <see cref="DataContent"/> whose media
/// type is neither <c>image/*</c> nor <c>video/*</c> (audio, documents), <see cref="UriContent"/> (nothing is downloaded),
/// and any other <see cref="AIContent"/> type. Text with media placeholders (<c>&lt;|image|&gt;</c>, <c>&lt;|video|&gt;</c>
/// and the other media tokens) fails with <see cref="ArgumentException"/>.
/// </para>
/// <para>
/// <b>Prompts, dimensions, options, usage, cancellation, threading</b>: as <see cref="EmbeddingGemma2EmbeddingGenerator"/>
/// (default prompt <see cref="EmbeddingGemma2Prompt.None"/>, which the model card recommends for images;
/// <see cref="EmbeddingGemma2EmbeddingGenerator.PromptPropertyName"/> per request; Matryoshka sizes 768, 512, 256 and
/// 128; usage counts image tokens). Execution is inline unless a scheduler is set: both <c>GenerateAsync</c> methods run
/// synchronously on the calling thread and return a completed task, or, with
/// <see cref="EmbeddingGemma2MultimodalEmbeddingGeneratorOptions.ExecutionScheduler"/> set (Experimental), run as one task
/// on that scheduler.
/// </para>
/// </remarks>
public sealed class EmbeddingGemma2MultimodalEmbeddingGenerator :
    IEmbeddingGenerator<DataContent, Embedding<float>>,
    IEmbeddingGenerator<IEnumerable<AIContent>, Embedding<float>>
{
    private readonly EmbeddingGemma2MultimodalEmbedder _embedder;
    private readonly bool _ownsEmbedder;
    private readonly EmbeddingGemma2MultimodalEmbeddingGeneratorOptions _options;
    private readonly EmbeddingGeneratorMetadata _metadata;
    private int _disposed;

    /// <summary>Wraps an existing embedder (not owned, not disposed by this instance).</summary>
    public EmbeddingGemma2MultimodalEmbeddingGenerator(EmbeddingGemma2MultimodalEmbedder embedder, EmbeddingGemma2MultimodalEmbeddingGeneratorOptions? options = null)
        : this(embedder, options ?? new EmbeddingGemma2MultimodalEmbeddingGeneratorOptions(), ownsEmbedder: false)
    {
    }

    private EmbeddingGemma2MultimodalEmbeddingGenerator(EmbeddingGemma2MultimodalEmbedder embedder, EmbeddingGemma2MultimodalEmbeddingGeneratorOptions options, bool ownsEmbedder)
    {
        ArgumentNullException.ThrowIfNull(embedder);
        options.Validate();
        if (options.Dimensions > embedder.Dimension)
        {
            throw new ArgumentException($"Dimensions = {options.Dimensions} exceeds the embedder's {embedder.Dimension} dimensions; truncation cannot widen an embedding.", nameof(options));
        }

        _embedder = embedder;
        _options = options;
        _ownsEmbedder = ownsEmbedder;
        _metadata = new EmbeddingGeneratorMetadata(EmbeddingGemma2EmbeddingGenerator.ProviderName, providerUri: null, options.ModelId, options.Dimensions);
    }

    /// <summary>Options in effect.</summary>
    public EmbeddingGemma2MultimodalEmbeddingGeneratorOptions Options => _options;

    /// <summary>Provider name, model id and default dimensions.</summary>
    public EmbeddingGeneratorMetadata Metadata => _metadata;

    /// <summary>The underlying embedder.</summary>
    public EmbeddingGemma2MultimodalEmbedder Embedder => _embedder;

    /// <summary>
    /// Loads the tokenizer and the image graphs from verified local files (never downloads; SHA-256 enforced) and owns the
    /// resulting embedder. The embedder produces 768 dimensions so each request can choose its Matryoshka size.
    /// </summary>
    public static EmbeddingGemma2MultimodalEmbeddingGenerator Create(EmbeddingGemma2ImageModelFiles files, EmbeddingGemma2MultimodalEmbeddingGeneratorOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(files);
        options ??= new EmbeddingGemma2MultimodalEmbeddingGeneratorOptions();
        options.Validate();
        var embedder = EmbeddingGemma2MultimodalEmbedder.Create(files, options.Embedder with
        {
            Dimension = EmbeddingGemma2Encoder.NativeDimension,
            VerifyModelSha256 = true,
        });
        try
        {
            return new EmbeddingGemma2MultimodalEmbeddingGenerator(embedder, options, ownsEmbedder: true);
        }
        catch
        {
            embedder.Dispose();
            throw;
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// Each value is one image or video embedded alone. Runs inline unless a scheduler is set: synchronously on the calling
    /// thread with a completed task, or as one task on <see cref="EmbeddingGemma2MultimodalEmbeddingGeneratorOptions.ExecutionScheduler"/>.
    /// </remarks>
    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<DataContent> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default) =>
        Run(() =>
        {
            ArgumentNullException.ThrowIfNull(values);
            var inputs = new List<EmbeddingGemma2MultimodalInput>();
            foreach (var value in values)
            {
                inputs.Add(Decode(value ?? throw new ArgumentException($"Value {inputs.Count} is null.", nameof(values)), inputs.Count) switch
                {
                    EmbeddingGemma2Video video => EmbeddingGemma2MultimodalInput.FromVideo(video),
                    var media => EmbeddingGemma2MultimodalInput.FromImage((EmbeddingGemma2Image)media),
                });
            }

            return inputs;
        }, options, cancellationToken);

    /// <inheritdoc />
    /// <remarks>
    /// Each value is one input of text and media parts in order. Runs inline unless a scheduler is set: synchronously on
    /// the calling thread with a completed task, or as one task on <see cref="EmbeddingGemma2MultimodalEmbeddingGeneratorOptions.ExecutionScheduler"/>.
    /// </remarks>
    public Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
        IEnumerable<IEnumerable<AIContent>> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default) =>
        Run(() =>
        {
            ArgumentNullException.ThrowIfNull(values);
            var inputs = new List<EmbeddingGemma2MultimodalInput>();
            foreach (var value in values)
            {
                int index = inputs.Count;
                var parts = new List<object>();
                foreach (var content in value ?? throw new ArgumentException($"Value {index} is null.", nameof(values)))
                {
                    parts.Add(content switch
                    {
                        TextContent text => text.Text ?? string.Empty,
                        DataContent data => Decode(data, index),
                        null => throw new ArgumentException($"Value {index} has a null content part.", nameof(values)),
                        _ => throw new NotSupportedException($"Value {index}: {content.GetType().Name} is not supported; EmbeddingGemma 2 embeds TextContent and image or video DataContent parts."),
                    });
                }

                inputs.Add(EmbeddingGemma2MultimodalInput.Create(parts));
            }

            return inputs;
        }, options, cancellationToken);

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

        return serviceType == typeof(EmbeddingGemma2MultimodalEmbedder) ? _embedder : null;
    }

    /// <summary>Disposes the embedder if this instance owns it. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0 && _ownsEmbedder)
        {
            _embedder.Dispose();
        }
    }

    /// <summary>An image (<c>image/*</c>) or a video (<c>video/*</c>) through the configured decoder; anything else is rejected.</summary>
    private object Decode(DataContent data, int index)
    {
        if (data.MediaType.StartsWith("video/", StringComparison.OrdinalIgnoreCase))
        {
            var videoDecoder = _options.VideoDecoder
                ?? throw new NotSupportedException($"Value {index}: a video needs EmbeddingGemma2MultimodalEmbeddingGeneratorOptions.VideoDecoder (for example EmbeddingGemma2VideoDecoder.Decode from Starlings.Models.EmbeddingGemma2.Video.FFmpeg).");
            return videoDecoder(data.Data, data.MediaType) ?? throw new InvalidOperationException($"Value {index}: the video decoder returned null.");
        }

        if (!data.MediaType.StartsWith("image/", StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException($"Value {index}: DataContent with media type '{data.MediaType}' is not supported; EmbeddingGemma 2 embeds images (image/*), videos (video/*) and text. Audio and documents are unsupported.");
        }

        var decoder = _options.ImageDecoder
            ?? throw new NotSupportedException($"Value {index}: an image needs EmbeddingGemma2MultimodalEmbeddingGeneratorOptions.ImageDecoder (for example EmbeddingGemma2ImageDecoder.Decode from Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp).");
        return decoder(data.Data, data.MediaType) ?? throw new InvalidOperationException($"Value {index}: the image decoder returned null.");
    }

    private Task<GeneratedEmbeddings<Embedding<float>>> Run(Func<List<EmbeddingGemma2MultimodalInput>> build, EmbeddingGenerationOptions? options, CancellationToken cancellationToken) =>
        _options.ExecutionScheduler is { } scheduler
            ? Task.Factory.StartNew(() => RunInline(build, options, cancellationToken), cancellationToken, TaskCreationOptions.DenyChildAttach, scheduler).Unwrap()
            : RunInline(build, options, cancellationToken);

    private Task<GeneratedEmbeddings<Embedding<float>>> RunInline(Func<List<EmbeddingGemma2MultimodalInput>> build, EmbeddingGenerationOptions? options, CancellationToken cancellationToken)
    {
        try
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            cancellationToken.ThrowIfCancellationRequested();
            var (prompt, dimensions) = Resolve(options);
            var inputs = build();
            var tokenCounts = new int[inputs.Count];
            float[][] vectors = _embedder.EmbedBatch(inputs, prompt, tokenCounts, cancellationToken);
            var createdAt = DateTimeOffset.UtcNow;
            var result = new GeneratedEmbeddings<Embedding<float>>(vectors.Length);
            foreach (float[] vector in vectors)
            {
                float[] output = vector.Length == dimensions ? vector : EmbeddingGemma2Encoder.Truncate(vector, dimensions);
                result.Add(new Embedding<float>(output) { ModelId = _options.ModelId, CreatedAt = createdAt });
            }

            long tokens = tokenCounts.Sum(c => (long)c);
            result.Usage = new UsageDetails { InputTokenCount = tokens, TotalTokenCount = tokens };
            return Task.FromResult(result);
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

    private (EmbeddingGemma2Prompt Prompt, int Dimensions) Resolve(EmbeddingGenerationOptions? options)
    {
        if (options is null)
        {
            return (_options.Prompt, _options.Dimensions);
        }

        int dimensions = _options.Dimensions;
        if (options.Dimensions is { } requested)
        {
            if (!EmbeddingGemma2Encoder.IsSupportedDimension(requested) || requested > _embedder.Dimension)
            {
                throw new NotSupportedException($"EmbeddingGenerationOptions.Dimensions = {requested} is not supported: EmbeddingGemma 2 supports the Matryoshka sizes 768, 512, 256 and 128, at most the embedder's {_embedder.Dimension}.");
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
                if (!string.Equals(key, EmbeddingGemma2EmbeddingGenerator.PromptPropertyName, StringComparison.Ordinal))
                {
                    throw new NotSupportedException($"EmbeddingGenerationOptions.AdditionalProperties['{key}'] is not supported; the only supported key is '{EmbeddingGemma2EmbeddingGenerator.PromptPropertyName}'.");
                }

                prompt = value switch
                {
                    EmbeddingGemma2Prompt p => p,
                    string name when EmbeddingGemma2Prompt.TryFromName(name, out var p) => p,
                    _ => throw new ArgumentException($"AdditionalProperties['{EmbeddingGemma2EmbeddingGenerator.PromptPropertyName}'] = '{value}' is not an EmbeddingGemma 2 prompt."),
                };
            }
        }

        return (prompt, dimensions);
    }
}

/// <summary>Immutable configuration of an <see cref="EmbeddingGemma2MultimodalEmbeddingGenerator"/>.</summary>
public sealed record EmbeddingGemma2MultimodalEmbeddingGeneratorOptions
{
    /// <summary>
    /// Turns an image <see cref="DataContent"/>'s bytes and media type into RGB. Null (default): image values are
    /// rejected. With <c>Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp</c>:
    /// <c>(bytes, _) =&gt; EmbeddingGemma2ImageDecoder.Decode(bytes.Span)</c>.
    /// </summary>
    public Func<ReadOnlyMemory<byte>, string, EmbeddingGemma2Image>? ImageDecoder { get; init; }

    /// <summary>
    /// Turns a video <see cref="DataContent"/>'s bytes and media type (<c>video/*</c>) into an <see cref="EmbeddingGemma2Video"/>.
    /// Null (default): video values are rejected with <see cref="NotSupportedException"/>. With
    /// <c>Starlings.Models.EmbeddingGemma2.Video.FFmpeg</c> (the user's own FFmpeg 8.x):
    /// <c>(bytes, _) =&gt; decoder.Decode(bytes).Video</c>, which samples as upstream (1 frame per second, at most 32). A
    /// decoder that returns frames without timing is rejected unless <see cref="EmbeddingGemma2MultimodalOptions.AllowVideosWithoutTiming"/>
    /// is set on <see cref="Embedder"/>.
    /// </summary>
    public Func<ReadOnlyMemory<byte>, string, EmbeddingGemma2Video>? VideoDecoder { get; init; }

    /// <summary>Prompt rendered first unless a request overrides it. Default <see cref="EmbeddingGemma2Prompt.None"/> (the card: no prefix for images).</summary>
    public EmbeddingGemma2Prompt Prompt { get; init; } = EmbeddingGemma2Prompt.None;

    /// <summary>Output size when a request does not set one: 768, 512, 256 or 128. Default 768.</summary>
    public int Dimensions { get; init; } = EmbeddingGemma2Encoder.NativeDimension;

    /// <summary>Reported model id and the only accepted <c>EmbeddingGenerationOptions.ModelId</c>. Default the pinned id.</summary>
    public string ModelId { get; init; } = EmbeddingGemma2EmbeddingGeneratorOptions.PinnedModelId;

    /// <summary>
    /// Embedder options used by <see cref="EmbeddingGemma2MultimodalEmbeddingGenerator.Create"/> (provider, threads, vision
    /// batch, limits). The dimension is always 768 and the graphs' SHA-256 always checked.
    /// </summary>
    public EmbeddingGemma2MultimodalOptions Embedder { get; init; } = new();

    /// <summary>
    /// Where both <c>GenerateAsync</c> methods execute (decoding, preprocessing and the graph runs; Experimental,
    /// capability <c>extensions_ai.execution_scheduler</c>), or null (default): inline on the calling thread, a completed
    /// task, as without this option. <see cref="TaskScheduler.Default"/> offloads to the thread pool. The scheduler
    /// changes only where the work runs, not the numbers.
    /// </summary>
    public TaskScheduler? ExecutionScheduler { get; init; }

    internal void Validate()
    {
        ArgumentNullException.ThrowIfNull(Prompt);
        if (!EmbeddingGemma2Encoder.IsSupportedDimension(Dimensions))
        {
            throw new ArgumentOutOfRangeException(nameof(Dimensions), Dimensions, "Supported dimensions are 768, 512, 256 and 128 (Matryoshka truncation).");
        }

        ArgumentException.ThrowIfNullOrWhiteSpace(ModelId);
        ArgumentNullException.ThrowIfNull(Embedder);
    }
}
