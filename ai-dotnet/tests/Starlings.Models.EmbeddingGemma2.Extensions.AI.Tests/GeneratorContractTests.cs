using System.Collections.Concurrent;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Caching.Distributed;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace Starlings.Models.EmbeddingGemma2.Extensions.AI.Tests;

/// <summary>
/// The adapter's contract on a weight-free graph with the EmbeddingGemma 2 signature (token embeddings
/// <c>id * (1 + d / 768) + mask * cos(d)</c>) and a stand-in tokenizer that maps each character to its code point plus
/// 10 between <c>&lt;bos&gt;</c> and <c>&lt;eos&gt;</c>. Runs in CI; the real model is covered by <see cref="RealModelTests"/>.
/// </summary>
public sealed class GeneratorContractTests : IDisposable
{
    private readonly EmbeddingGemma2Encoder _encoder = new(
        OnnxModelSession.Create(TestModels.FakeEmbeddingGemma2, new OnnxSessionOptions { IntraOpNumThreads = 1, ComputeModelSha256 = false }),
        new EmbeddingGemma2EncoderOptions { MaxBatchSize = 2 },
        ownsSession: true);

    private readonly RecordingTokenizer _tokenizer = new();

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose() => _encoder.Dispose();

    private EmbeddingGemma2TextEmbedder Embedder() => new(_tokenizer, _encoder);

    private EmbeddingGemma2EmbeddingGenerator Generator(EmbeddingGemma2EmbeddingGeneratorOptions? options = null) => new(Embedder(), options);

    private static long[] Ids(string text) => new long[] { 2 }.Concat(text.Select(c => (long)c + 10)).Append(1).ToArray();

    /// <summary>The closed-form embedding of <paramref name="text"/> as the stand-in tokenizer encodes it.</summary>
    private static float[] Expected(string text, int dimension = 768)
    {
        var ids = Ids(text);
        var sum = new double[768];
        foreach (long id in ids)
        {
            for (int d = 0; d < 768; d++)
            {
                sum[d] += (float)((float)id * (float)(1.0 + (d / 768.0))) + (float)Math.Cos(d);
            }
        }

        double norm = 0;
        for (int d = 0; d < dimension; d++)
        {
            sum[d] /= ids.Length;
            norm += sum[d] * sum[d];
        }

        norm = Math.Sqrt(norm);
        return sum.Take(dimension).Select(x => (float)(x / norm)).ToArray();
    }

    private static void Near(float[] expected, ReadOnlyMemory<float> actual)
    {
        var a = actual.ToArray();
        Assert.Equal(expected.Length, a.Length);
        for (int i = 0; i < a.Length; i++)
        {
            Assert.True(Math.Abs(expected[i] - a[i]) <= 2e-6, $"[{i}] {a[i]} vs {expected[i]}");
        }
    }

    [Fact]
    public void MetadataNamesProviderPinnedModelAndDimensions()
    {
        using var generator = Generator();
        IEmbeddingGenerator<string, Embedding<float>> contract = generator;
        var metadata = contract.GetService<EmbeddingGeneratorMetadata>();
        Assert.NotNull(metadata);
        Assert.Same(generator.Metadata, metadata);
        Assert.Equal(EmbeddingGemma2EmbeddingGenerator.ProviderName, metadata.ProviderName);
        Assert.Equal("google/embeddinggemma-2@914f7f89142e33e77833254d9c9b90c3cef7303b", metadata.DefaultModelId);
        Assert.Equal(768, metadata.DefaultModelDimensions);
        Assert.Null(metadata.ProviderUri);
        Assert.Same(generator, contract.GetService<EmbeddingGemma2EmbeddingGenerator>());
        Assert.Same(generator.Embedder, contract.GetService<EmbeddingGemma2TextEmbedder>());
        Assert.Same(_encoder, contract.GetService<EmbeddingGemma2Encoder>());
        Assert.Null(contract.GetService<EmbeddingGeneratorMetadata>("some-key"));
        Assert.Null(contract.GetService<string>());

        using var small = Generator(new EmbeddingGemma2EmbeddingGeneratorOptions { Dimensions = 256 });
        Assert.Equal(256, small.Metadata.DefaultModelDimensions);
    }

    [Fact]
    public async Task BatchIsOrderedAndEqualToTheEmbedderWithUsage()
    {
        using var generator = Generator();
        string[] texts = ["a", "longer text", "", "zz", "mid", "the last one"];
        var result = await generator.GenerateAsync(texts, cancellationToken: Ct);
        Assert.Equal(texts.Length, result.Count);
        var direct = Embedder().EmbedBatch(texts, Ct);
        for (int i = 0; i < texts.Length; i++)
        {
            Assert.Equal(direct[i], result[i].Vector.ToArray());
            Near(Expected(texts[i]), result[i].Vector);
            Assert.Equal(768, result[i].Dimensions);
            Assert.Equal(EmbeddingGemma2EmbeddingGeneratorOptions.PinnedModelId, result[i].ModelId);
            Assert.NotNull(result[i].CreatedAt);
        }

        long tokens = texts.Sum(t => t.Length + 2);
        Assert.Equal(tokens, result.Usage!.InputTokenCount);
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
    public async Task NoPromptIsAddedByDefault()
    {
        using var generator = Generator();
        await generator.GenerateAsync(["hello", "task: search result | query: hello"], cancellationToken: Ct);
        Assert.Equal(["hello", "task: search result | query: hello"], _tokenizer.Seen);
    }

    [Fact]
    public async Task PromptOptionAddsThePromptExplicitly()
    {
        using var query = Generator(new EmbeddingGemma2EmbeddingGeneratorOptions { Prompt = EmbeddingGemma2Prompt.Query });
        var result = await query.GenerateAsync(["hello"], cancellationToken: Ct);
        Near(Expected("task: search result | query: hello"), result[0].Vector);
        Assert.Equal("task: search result | query: hello".Length + 2, result.Usage!.InputTokenCount);

        // Per request, as a prompt object or an exact name; never detected or de-duplicated.
        await query.GenerateAsync(["doc"], Props(EmbeddingGemma2EmbeddingGenerator.PromptPropertyName, EmbeddingGemma2Prompt.Document), Ct);
        await query.GenerateAsync(["doc"], Props(EmbeddingGemma2EmbeddingGenerator.PromptPropertyName, "none"), Ct);
        await query.GenerateAsync(["c"], Props(EmbeddingGemma2EmbeddingGenerator.PromptPropertyName, "Clustering"), Ct);
        await query.GenerateAsync(["task: search result | query: q"], cancellationToken: Ct);
        Assert.Equal(
            ["task: search result | query: hello", "title: none | text: doc", "doc", "task: clustering | query: c", "task: search result | query: task: search result | query: q"],
            _tokenizer.Seen);
    }

    [Theory]
    [InlineData("Query")]
    [InlineData("QUERY")]
    [InlineData("search_query")]
    [InlineData("1")]
    [InlineData("")]
    public async Task InvalidPromptNamesAreRejected(string value)
    {
        using var generator = Generator();
        await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateAsync(["x"], Props(EmbeddingGemma2EmbeddingGenerator.PromptPropertyName, value), Ct));
        await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateAsync(["x"], Props(EmbeddingGemma2EmbeddingGenerator.PromptPropertyName, 7), Ct));
        await Assert.ThrowsAsync<ArgumentException>(() => generator.GenerateAsync(["x"], Props(EmbeddingGemma2EmbeddingGenerator.PromptPropertyName, null), Ct));
        Assert.Throws<ArgumentNullException>(() => Generator(new EmbeddingGemma2EmbeddingGeneratorOptions { Prompt = null! }));
        Assert.Empty(_tokenizer.Seen);
    }

    [Theory]
    [InlineData(768)]
    [InlineData(512)]
    [InlineData(256)]
    [InlineData(128)]
    public async Task DimensionsSelectAMatryoshkaSize(int dimensions)
    {
        using var generator = Generator();
        var result = await generator.GenerateAsync(["matryoshka"], new EmbeddingGenerationOptions { Dimensions = dimensions }, Ct);
        Assert.Equal(dimensions, result[0].Dimensions);
        Near(Expected("matryoshka", dimensions), result[0].Vector);

        using var byDefault = Generator(new EmbeddingGemma2EmbeddingGeneratorOptions { Dimensions = dimensions });
        var viaDefault = await byDefault.GenerateAsync(["matryoshka"], cancellationToken: Ct);
        Assert.Equal(result[0].Vector.ToArray(), viaDefault[0].Vector.ToArray());
    }

    [Theory]
    [InlineData(1)]
    [InlineData(64)]
    [InlineData(384)]
    [InlineData(767)]
    [InlineData(1024)]
    public async Task OtherDimensionsAreRejectedNotIgnored(int dimensions)
    {
        using var generator = Generator();
        var ex = await Assert.ThrowsAsync<NotSupportedException>(() => generator.GenerateAsync(["x"], new EmbeddingGenerationOptions { Dimensions = dimensions }, Ct));
        Assert.Contains($"Dimensions = {dimensions}", ex.Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentOutOfRangeException>(() => Generator(new EmbeddingGemma2EmbeddingGeneratorOptions { Dimensions = dimensions }));
        Assert.Empty(_tokenizer.Seen);
    }

    [Fact]
    public async Task ASmallerEncoderCannotBeWidened()
    {
        using var encoder256 = new EmbeddingGemma2Encoder(_encoder.Session, new EmbeddingGemma2EncoderOptions { Dimension = 256 });
        var embedder = new EmbeddingGemma2TextEmbedder(_tokenizer, encoder256);
        Assert.Throws<ArgumentException>(() => new EmbeddingGemma2EmbeddingGenerator(embedder));
        using var generator = new EmbeddingGemma2EmbeddingGenerator(embedder, new EmbeddingGemma2EmbeddingGeneratorOptions { Dimensions = 256 });
        await Assert.ThrowsAsync<NotSupportedException>(() => generator.GenerateAsync(["x"], new EmbeddingGenerationOptions { Dimensions = 512 }, Ct));
        var r = await generator.GenerateAsync(["x"], new EmbeddingGenerationOptions { Dimensions = 128 }, Ct);
        Near(Expected("x", 128), r[0].Vector);
    }

    [Fact]
    public async Task UnsupportedOptionsAreRejectedNotIgnored()
    {
        using var generator = Generator();
        await Assert.ThrowsAsync<NotSupportedException>(() => generator.GenerateAsync(["x"], new EmbeddingGenerationOptions { ModelId = "google/embeddinggemma-300m" }, Ct));
        await Assert.ThrowsAsync<NotSupportedException>(() => generator.GenerateAsync(["x"], new EmbeddingGenerationOptions { RawRepresentationFactory = _ => new object() }, Ct));
        var e = await Assert.ThrowsAsync<NotSupportedException>(() => generator.GenerateAsync(["x"], Props("user", "u1"), Ct));
        Assert.Contains("'user'", e.Message, StringComparison.Ordinal);
        Assert.Empty(_tokenizer.Seen);
        var ok = await generator.GenerateAsync(["x"], new EmbeddingGenerationOptions { Dimensions = 768, ModelId = EmbeddingGemma2EmbeddingGeneratorOptions.PinnedModelId }, Ct);
        Assert.Single(ok);
    }

    [Fact]
    public async Task TooLongInputFailsTheTaskWithTheTypedException()
    {
        using var generator = Generator();
        var ex = await Assert.ThrowsAsync<EmbeddingGemma2InputTooLongException>(() => generator.GenerateAsync(["ok", new string('a', 8191)], cancellationToken: Ct));
        Assert.Equal(1, ex.InputIndex);
        Assert.Equal(8193, ex.TokenCount);
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
        Assert.True(task.IsCompletedSuccessfully);
        Assert.All(_tokenizer.Threads, t => Assert.Equal(caller, t));
        Assert.Equal(3, (await task).Count);
    }

    [Fact]
    public void CancelledTokenPreventsAnyWork()
    {
        using var generator = Generator();
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.True(generator.GenerateAsync(["a"], cancellationToken: cts.Token).IsCanceled);
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
        _tokenizer.OnTokenize = null;
        var after = await generator.GenerateAsync(["first"], cancellationToken: Ct);
        Near(Expected("first"), after[0].Vector);
    }

    [Fact]
    public async Task DisposeIsIdempotentAndDoesNotDisposeABorrowedEncoder()
    {
        var generator = Generator();
        generator.Dispose();
        generator.Dispose();
        await Assert.ThrowsAsync<ObjectDisposedException>(() => generator.GenerateAsync(["a"], cancellationToken: Ct));
        Assert.False(_encoder.Session.IsDisposed);
    }

    [Fact]
    public async Task ConcurrentCallsGiveTheSameResults()
    {
        using var generator = Generator();
        string[] texts = ["one", "two", "three", "four", "five"];
        var expected = (await generator.GenerateAsync(texts, cancellationToken: Ct)).Select(e => e.Vector.ToArray()).ToArray();
        var results = await Task.WhenAll(Enumerable.Range(0, 8).Select(_ => Task.Run(() => generator.GenerateAsync(texts, cancellationToken: Ct), Ct)));
        Assert.All(results, r => Assert.Equal(expected, r.Select(e => e.Vector.ToArray()).ToArray()));
    }

    [Fact]
    public async Task WorksThroughTheGenericContractAndInsideAMiddlewarePipeline()
    {
        using var generator = Generator();
        IEmbeddingGenerator<string, Embedding<float>> contract = generator;
        Near(Expected("abc"), await contract.GenerateVectorAsync("abc", cancellationToken: Ct));

        var logs = new ConcurrentQueue<string>();
        using var loggerFactory = LoggerFactory.Create(b => b.AddProvider(new ListLoggerProvider(logs)).SetMinimumLevel(LogLevel.Trace));
        var cache = new DictionaryCache();
        using var pipeline = generator.AsBuilder().UseLogging(loggerFactory).UseDistributedCache(cache).Build();
        var first = await pipeline.GenerateAsync(["a", "bb"], cancellationToken: Ct);
        var second = await pipeline.GenerateAsync(["a", "bb"], cancellationToken: Ct);
        Near(Expected("a"), first[0].Vector);
        Near(Expected("bb"), second[1].Vector);
        Assert.Equal(["abc", "a", "bb"], _tokenizer.Seen);
        Assert.NotEmpty(cache.Entries);
        Assert.Contains(logs, m => m.Contains("GenerateAsync", StringComparison.Ordinal));
        Assert.Same(generator, pipeline.GetService<EmbeddingGemma2EmbeddingGenerator>());
        var small = await pipeline.GenerateAsync(["new"], new EmbeddingGenerationOptions { Dimensions = 128 }, Ct);
        Assert.Equal(128, small[0].Dimensions);
        await Assert.ThrowsAsync<NotSupportedException>(() => pipeline.GenerateAsync(["new2"], new EmbeddingGenerationOptions { Dimensions = 100 }, Ct));
    }

    [Fact]
    public async Task DependencyInjectionSharesOneInstanceAndTheContainerDisposesIt()
    {
        var services = new ServiceCollection();
        services.AddEmbeddingGemma2EmbeddingGenerator(_ => Generator());
        EmbeddingGemma2EmbeddingGenerator generator;
        await using (var provider = services.BuildServiceProvider())
        {
            generator = provider.GetRequiredService<EmbeddingGemma2EmbeddingGenerator>();
            var contract = provider.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();
            Assert.Same(generator, contract);
            Near(Expected("di"), await contract.GenerateVectorAsync("di", cancellationToken: Ct));
        }

        await Assert.ThrowsAsync<ObjectDisposedException>(() => generator.GenerateAsync(["a"], cancellationToken: Ct));
    }

    private static EmbeddingGenerationOptions Props(string key, object? value) =>
        new() { AdditionalProperties = new AdditionalPropertiesDictionary { [key] = value } };

    private sealed class RecordingTokenizer : IEmbeddingGemma2Tokenizer
    {
        public ConcurrentQueue<string> Seen { get; } = new();

        public ConcurrentQueue<int> Threads { get; } = new();

        public Action<string>? OnTokenize { get; set; }

        public EmbeddingGemma2Input Tokenize(string text)
        {
            Seen.Enqueue(text);
            Threads.Enqueue(Environment.CurrentManagedThreadId);
            OnTokenize?.Invoke(text);
            return new EmbeddingGemma2Input(Ids(text));
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

    private sealed class ListLoggerProvider(ConcurrentQueue<string> messages) : ILoggerProvider
    {
        public ILogger CreateLogger(string categoryName) => new ListLogger(messages);

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
