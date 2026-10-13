using System.Collections.Concurrent;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Caching.Distributed;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace Starlings.Models.E5.Extensions.AI.Tests;

/// <summary>
/// The adapter's contract on a weight-free graph with the e5 signature (hidden state <c>[id, 0, 1, 0.5 * mask]</c>,
/// dimension 4) and a stand-in tokenizer that maps each character to its code point between <c>&lt;s&gt;</c> and
/// <c>&lt;/s&gt;</c>. Runs in CI; the real model is covered by <see cref="RealModelTests"/>.
/// </summary>
public sealed class GeneratorContractTests : IDisposable
{
    private readonly E5Encoder _encoder = new(
        OnnxModelSession.Create(TestModels.FakeE5, new OnnxSessionOptions { IntraOpNumThreads = 1, ComputeModelSha256 = false }),
        new E5EncoderOptions { MaxBatchSize = 2 },
        ownsSession: true);

    private readonly RecordingTokenizer _tokenizer = new();

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose() => _encoder.Dispose();

    private E5TextEmbedder Embedder() => new(_tokenizer, _encoder);

    private E5EmbeddingGenerator Generator(E5EmbeddingGeneratorOptions? options = null) => new(Embedder(), options);

    /// <summary>normalize([mean of 0, code points, 2; 0; 1; 0.5]).</summary>
    private static float[] Expected(string text)
    {
        double sum = 2 + text.Sum(ch => (double)ch);
        double[] v = [sum / (text.Length + 2), 0, 1, 0.5];
        double norm = Math.Sqrt(v.Sum(x => x * x));
        return v.Select(x => (float)(x / norm)).ToArray();
    }

    [Fact]
    public void MetadataNamesProviderPinnedModelAndDimension()
    {
        using var generator = Generator();
        IEmbeddingGenerator<string, Embedding<float>> contract = generator;
        var metadata = contract.GetService<EmbeddingGeneratorMetadata>();
        Assert.NotNull(metadata);
        Assert.Same(generator.Metadata, metadata);
        Assert.Equal(E5EmbeddingGenerator.ProviderName, metadata.ProviderName);
        Assert.Equal("intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3", metadata.DefaultModelId);
        Assert.Equal(E5EmbeddingGeneratorOptions.PinnedModelId, metadata.DefaultModelId);
        Assert.Equal(4, metadata.DefaultModelDimensions);
        Assert.Null(metadata.ProviderUri);

        Assert.Same(generator, contract.GetService<E5EmbeddingGenerator>());
        Assert.Same(generator, contract.GetService<IEmbeddingGenerator<string, Embedding<float>>>());
        Assert.Same(generator.Embedder, contract.GetService<E5TextEmbedder>());
        Assert.Same(_encoder, contract.GetService<E5Encoder>());
        Assert.Null(contract.GetService<EmbeddingGeneratorMetadata>("some-key"));
        Assert.Null(contract.GetService<string>());
    }

    [Fact]
    public async Task BatchIsChunkedOrderedAndEqualToTheEmbedder()
    {
        using var generator = Generator();
        string[] texts = ["a", "longer text", "", "zz", "mid", "the last one"];
        var result = await generator.GenerateAsync(texts, cancellationToken: Ct);

        // MaxBatchSize = 2: three ONNX Runtime calls, each padded to its own longest row; order is preserved.
        Assert.Equal(texts.Length, result.Count);
        var direct = Embedder().EmbedBatch(texts, Ct);
        for (int i = 0; i < texts.Length; i++)
        {
            Assert.Equal(direct[i], result[i].Vector.ToArray());
            Assert.Equal(Expected(texts[i]), result[i].Vector.ToArray());
            Assert.Equal(4, result[i].Dimensions);
            Assert.Equal(E5EmbeddingGeneratorOptions.PinnedModelId, result[i].ModelId);
            Assert.NotNull(result[i].CreatedAt);
        }

        Assert.NotNull(result.Usage);
        long tokens = texts.Sum(t => t.Length + 2);
        Assert.Equal(tokens, result.Usage.InputTokenCount);
        Assert.Equal(tokens, result.Usage.TotalTokenCount);
        Assert.Null(result.Usage.OutputTokenCount);
    }

    [Fact]
    public async Task EmptyInputGivesEmptyResult()
    {
        using var generator = Generator();
        var result = await generator.GenerateAsync([], cancellationToken: Ct);
        Assert.Empty(result);
        Assert.Equal(0, result.Usage!.InputTokenCount);
    }

    [Fact]
    public async Task NoPrefixIsAddedByDefault()
    {
        using var generator = Generator();
        await generator.GenerateAsync(["hello", "query: hello"], cancellationToken: Ct);
        Assert.Equal(["hello", "query: hello"], _tokenizer.Seen);
    }

    [Fact]
    public async Task InputTypeOptionAddsThePrefixExplicitly()
    {
        using var query = Generator(new E5EmbeddingGeneratorOptions { InputType = E5InputType.Query });
        var result = await query.GenerateAsync(["hello"], cancellationToken: Ct);
        Assert.Equal(Expected("query: hello"), result[0].Vector.ToArray());
        Assert.Equal("query: hello".Length + 2, result.Usage!.InputTokenCount);

        // Per request, as an enum or a case-insensitive name; never detected or de-duplicated.
        await query.GenerateAsync(["doc"], Props(E5EmbeddingGenerator.InputTypePropertyName, E5InputType.Passage), Ct);
        await query.GenerateAsync(["doc"], Props(E5EmbeddingGenerator.InputTypePropertyName, "none"), Ct);
        await query.GenerateAsync(["query: q"], Props(E5EmbeddingGenerator.InputTypePropertyName, "Query"), Ct);
        Assert.Equal(["query: hello", "passage: doc", "doc", "query: query: q"], _tokenizer.Seen);
    }

    [Theory]
    [InlineData("search_query")]
    [InlineData("1")]
    [InlineData("query, passage")]
    [InlineData("")]
    public async Task InvalidInputTypeIsRejected(string value)
    {
        using var generator = Generator();
        await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateAsync(["x"], Props(E5EmbeddingGenerator.InputTypePropertyName, value), Ct));
        await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateAsync(["x"], Props(E5EmbeddingGenerator.InputTypePropertyName, (E5InputType)7), Ct));
        Assert.Throws<ArgumentOutOfRangeException>(() => Generator(new E5EmbeddingGeneratorOptions { InputType = (E5InputType)7 }));
        Assert.Empty(_tokenizer.Seen);
    }

    [Fact]
    public async Task UnsupportedOptionsAreRejectedNotIgnored()
    {
        using var generator = Generator();
        var e1 = await Assert.ThrowsAsync<NotSupportedException>(() => generator.GenerateAsync(["x"], new EmbeddingGenerationOptions { Dimensions = 2 }, Ct));
        Assert.Contains("Dimensions = 2", e1.Message, StringComparison.Ordinal);
        await Assert.ThrowsAsync<NotSupportedException>(() => generator.GenerateAsync(["x"], new EmbeddingGenerationOptions { ModelId = "intfloat/multilingual-e5-large" }, Ct));
        await Assert.ThrowsAsync<NotSupportedException>(() => generator.GenerateAsync(["x"], new EmbeddingGenerationOptions { RawRepresentationFactory = _ => new object() }, Ct));
        var e2 = await Assert.ThrowsAsync<NotSupportedException>(() => generator.GenerateAsync(["x"], Props("user", "u1"), Ct));
        Assert.Contains("'user'", e2.Message, StringComparison.Ordinal);
        Assert.Empty(_tokenizer.Seen);

        // Values that restate the model's own are accepted.
        var ok = await generator.GenerateAsync(["x"], new EmbeddingGenerationOptions { Dimensions = 4, ModelId = E5EmbeddingGeneratorOptions.PinnedModelId }, Ct);
        Assert.Single(ok);
    }

    [Fact]
    public async Task NullsAreReportedThroughTheTask()
    {
        using var generator = Generator();
        await Assert.ThrowsAsync<ArgumentNullException>(() => generator.GenerateAsync(null!, cancellationToken: Ct));
        await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateAsync(["a", null!], cancellationToken: Ct));
    }

    [Fact]
    public async Task RunsSynchronouslyOnTheCallingThread()
    {
        using var generator = Generator();
        int caller = Environment.CurrentManagedThreadId;
        var task = generator.GenerateAsync(["a", "b", "c"], cancellationToken: Ct);
        Assert.True(task.IsCompletedSuccessfully, "GenerateAsync returns a completed task");
        Assert.All(_tokenizer.Threads, t => Assert.Equal(caller, t));
        Assert.Equal(3, (await task).Count);
    }

    [Fact]
    public void CancelledTokenPreventsAnyWork()
    {
        using var generator = Generator();
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        var task = generator.GenerateAsync(["a"], cancellationToken: cts.Token);
        Assert.True(task.IsCanceled);
        Assert.Empty(_tokenizer.Seen);
    }

    [Fact]
    public async Task CancellationDuringTokenizationStopsBeforeTheNextValue()
    {
        using var generator = Generator();
        using var cts = new CancellationTokenSource();
        _tokenizer.OnTokenize = text =>
        {
            if (text == "second")
            {
                cts.Cancel();
            }
        };

        var task = generator.GenerateAsync(["first", "second", "third"], cancellationToken: cts.Token);
        Assert.True(task.IsCanceled);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => task);
        Assert.Equal(["first", "second"], _tokenizer.Seen);

        // The generator stays usable.
        _tokenizer.OnTokenize = null;
        var after = await generator.GenerateAsync(["first"], cancellationToken: Ct);
        Assert.Equal(Expected("first"), after[0].Vector.ToArray());
    }

    [Fact]
    public void CancellationAfterTokenizationStopsBeforeTheFirstChunk()
    {
        using var generator = Generator();
        using var cts = new CancellationTokenSource();
        _tokenizer.OnTokenize = text =>
        {
            if (text == "last")
            {
                cts.Cancel();
            }
        };

        var task = generator.GenerateAsync(["a", "last"], cancellationToken: cts.Token);
        Assert.True(task.IsCanceled);
    }

    [Fact]
    public async Task DisposeIsIdempotentAndDoesNotDisposeABorrowedEncoder()
    {
        var generator = Generator();
        generator.Dispose();
        generator.Dispose();
        await Assert.ThrowsAsync<ObjectDisposedException>(() => generator.GenerateAsync(["a"], cancellationToken: Ct));

        // The wrapped embedder and its encoder belong to the caller.
        Assert.Equal(Expected("a"), Embedder().Embed("a", Ct));
    }

    [Fact]
    public async Task ConcurrentCallsGiveTheSameResults()
    {
        using var generator = Generator();
        string[] texts = ["one", "two", "three", "four", "five"];
        var expected = texts.Select(Expected).ToArray();
        var results = await Task.WhenAll(Enumerable.Range(0, 8).Select(_ => Task.Run(() => generator.GenerateAsync(texts, cancellationToken: Ct), Ct)));
        Assert.All(results, r => Assert.Equal(expected, r.Select(e => e.Vector.ToArray()).ToArray()));
    }

    [Fact]
    public async Task WorksThroughTheGenericContractAndExtensions()
    {
        using var generator = Generator();
        IEmbeddingGenerator<string, Embedding<float>> contract = generator;
        var vector = await contract.GenerateVectorAsync("abc", cancellationToken: Ct);
        Assert.Equal(Expected("abc"), vector.ToArray());
        var single = await contract.GenerateAsync("abc", cancellationToken: Ct);
        Assert.Equal(Expected("abc"), single.Vector.ToArray());
        var zipped = await contract.GenerateAndZipAsync(["x", "yy"], cancellationToken: Ct);
        Assert.Equal(["x", "yy"], zipped.Select(z => z.Value));
        Assert.Equal(Expected("yy"), zipped[1].Embedding.Vector.ToArray());
    }

    [Fact]
    public async Task WorksInsideAMiddlewarePipeline()
    {
        using var generator = Generator();
        var logs = new ListLoggerProvider();
        using var loggerFactory = LoggerFactory.Create(b => b.AddProvider(logs).SetMinimumLevel(LogLevel.Trace));
        var cache = new DictionaryCache();
        using var pipeline = generator.AsBuilder()
            .UseLogging(loggerFactory)
            .UseDistributedCache(cache)
            .Build();

        var first = await pipeline.GenerateAsync(["a", "bb"], cancellationToken: Ct);
        var second = await pipeline.GenerateAsync(["a", "bb"], cancellationToken: Ct);

        Assert.Equal(Expected("a"), first[0].Vector.ToArray());
        Assert.Equal(Expected("bb"), second[1].Vector.ToArray());
        Assert.Equal(["a", "bb"], _tokenizer.Seen);       // the second call was served from the cache
        Assert.NotEmpty(cache.Entries);
        Assert.Contains(logs.Messages, m => m.Contains("GenerateAsync", StringComparison.Ordinal));

        // Metadata and the underlying instances stay reachable through the pipeline.
        Assert.Equal(4, pipeline.GetService<EmbeddingGeneratorMetadata>()!.DefaultModelDimensions);
        Assert.Same(generator, pipeline.GetService<E5EmbeddingGenerator>());

        // Unsupported options surface through the pipeline as well.
        await Assert.ThrowsAsync<NotSupportedException>(() => pipeline.GenerateAsync(["new"], new EmbeddingGenerationOptions { Dimensions = 8 }, Ct));

        // Cancellation surfaces through the pipeline.
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => pipeline.GenerateAsync(["uncached"], cancellationToken: cts.Token));
    }

    [Fact]
    public async Task DependencyInjectionSharesOneInstanceAndTheContainerDisposesIt()
    {
        var services = new ServiceCollection();
        services.AddE5EmbeddingGenerator(_ => Generator());
        E5EmbeddingGenerator generator;
        await using (var provider = services.BuildServiceProvider())
        {
            generator = provider.GetRequiredService<E5EmbeddingGenerator>();
            var contract = provider.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();
            Assert.Same(generator, contract);
            Assert.Equal(Expected("di"), (await contract.GenerateVectorAsync("di", cancellationToken: Ct)).ToArray());
        }

        await Assert.ThrowsAsync<ObjectDisposedException>(() => generator.GenerateAsync(["a"], cancellationToken: Ct));
    }

    [Fact]
    public async Task DependencyInjectionComposesWithTheExtensionsAiBuilder()
    {
        var services = new ServiceCollection();
        services.AddLogging();
        services.AddE5EmbeddingGenerator(_ => Generator());
        services.AddEmbeddingGenerator(sp => sp.GetRequiredService<E5EmbeddingGenerator>()).UseLogging();
        await using var provider = services.BuildServiceProvider();

        var contract = provider.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();
        Assert.IsNotType<E5EmbeddingGenerator>(contract);
        Assert.Same(provider.GetRequiredService<E5EmbeddingGenerator>(), contract.GetService<E5EmbeddingGenerator>());
        Assert.Equal(Expected("di"), (await contract.GenerateVectorAsync("di", cancellationToken: Ct)).ToArray());
    }

    private static EmbeddingGenerationOptions Props(string key, object? value) =>
        new() { AdditionalProperties = new AdditionalPropertiesDictionary { [key] = value } };

    private sealed class RecordingTokenizer : IEncoderTokenizer
    {
        public ConcurrentQueue<string> Seen { get; } = new();

        public ConcurrentQueue<int> Threads { get; } = new();

        public Action<string>? OnTokenize { get; set; }

        public EncoderInput Tokenize(string text)
        {
            Seen.Enqueue(text);
            Threads.Enqueue(Environment.CurrentManagedThreadId);
            OnTokenize?.Invoke(text);
            var ids = new long[text.Length + 2];
            ids[0] = 0;
            for (int i = 0; i < text.Length; i++)
            {
                ids[i + 1] = text[i];
            }

            ids[^1] = 2;
            return new EncoderInput(ids);
        }
    }

    private sealed class DictionaryCache : IDistributedCache
    {
        public ConcurrentDictionary<string, byte[]> Entries { get; } = new();

        public byte[]? Get(string key) => Entries.TryGetValue(key, out var v) ? v : null;

        public Task<byte[]?> GetAsync(string key, CancellationToken token = default) => Task.FromResult(Get(key));

        public void Refresh(string key)
        {
        }

        public Task RefreshAsync(string key, CancellationToken token = default) => Task.CompletedTask;

        public void Remove(string key) => Entries.TryRemove(key, out _);

        public Task RemoveAsync(string key, CancellationToken token = default)
        {
            Remove(key);
            return Task.CompletedTask;
        }

        public void Set(string key, byte[] value, DistributedCacheEntryOptions options) => Entries[key] = value;

        public Task SetAsync(string key, byte[] value, DistributedCacheEntryOptions options, CancellationToken token = default)
        {
            Set(key, value, options);
            return Task.CompletedTask;
        }
    }

    private sealed class ListLoggerProvider : ILoggerProvider
    {
        public ConcurrentQueue<string> Messages { get; } = new();

        public ILogger CreateLogger(string categoryName) => new ListLogger(Messages);

        public void Dispose()
        {
        }

        private sealed class ListLogger(ConcurrentQueue<string> messages) : ILogger
        {
            public IDisposable? BeginScope<TState>(TState state)
                where TState : notnull => null;

            public bool IsEnabled(LogLevel logLevel) => true;

            public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter) =>
                messages.Enqueue(formatter(state, exception));
        }
    }
}
