using Starlings.Runtime.Onnx;
using Starlings.Runtime.Onnx.Batching;
using Starlings.TestSupport;

namespace Starlings.Models.E5.Extensions.AI.Tests;

/// <summary>
/// Opt-in micro-batching (<see cref="E5EmbeddingGeneratorOptions.MicroBatching"/>, Experimental) on the weight-free e5
/// graph and a code-point tokenizer. A long <see cref="MicroBatchingOptions.MaxWait"/> makes batch composition
/// deterministic: a batch is dispatched only when it holds <see cref="MicroBatchingOptions.MaxRows"/> values.
/// </summary>
public sealed class MicroBatchingTests : IDisposable
{
    private static readonly MicroBatchingOptions FillTwo = new() { MaxWait = TimeSpan.FromHours(1), MaxRows = 2 };

    private readonly E5Encoder _encoder = new(
        OnnxModelSession.Create(TestModels.FakeE5, new OnnxSessionOptions { IntraOpNumThreads = 1, ComputeModelSha256 = false }),
        new E5EncoderOptions { MaxBatchSize = 2 },
        ownsSession: true);

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose() => _encoder.Dispose();

    private E5TextEmbedder Embedder() => new(new CodePointTokenizer(), _encoder);

    private E5EmbeddingGenerator Generator(MicroBatchingOptions? microBatching) => new(Embedder(), new E5EmbeddingGeneratorOptions { MicroBatching = microBatching });

    private static Task<T> Within<T>(Task<T> task) => task.WaitAsync(TimeSpan.FromSeconds(30), Ct);

    [Fact]
    public async Task OffByDefaultAndCallsStaySynchronous()
    {
        Assert.Null(new E5EmbeddingGeneratorOptions().MicroBatching);
        using var generator = Generator(null);
        var task = generator.GenerateAsync(["a"], cancellationToken: Ct);
        Assert.True(task.IsCompletedSuccessfully);
        Assert.Null(generator.MicroBatchingStatistics);
        Assert.Equal(Embedder().Embed("a", Ct), (await task)[0].Vector.ToArray());
    }

    [Fact]
    public void InvalidOptionsAreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Generator(new MicroBatchingOptions { MaxRows = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => Generator(new MicroBatchingOptions { QueueCapacity = 0 }));
    }

    [Fact]
    public async Task ConcurrentSingleValueCallsShareABatchAndEqualDirectCalls()
    {
        using var generator = Generator(new MicroBatchingOptions { MaxWait = TimeSpan.FromHours(1), MaxRows = 4 });
        using var direct = Generator(null);
        string[] texts = ["a", "a much longer text", "zz", "mid"];
        var calls = texts.Select(t => generator.GenerateAsync([t], cancellationToken: Ct)).ToArray();
        var results = await Within(Task.WhenAll(calls));

        for (int i = 0; i < texts.Length; i++)
        {
            var expected = (await direct.GenerateAsync([texts[i]], cancellationToken: Ct))[0];
            Assert.Single(results[i]);
            Assert.Equal(expected.Vector.ToArray(), results[i][0].Vector.ToArray());
            Assert.Equal(E5EmbeddingGeneratorOptions.PinnedModelId, results[i][0].ModelId);
            Assert.Equal(texts[i].Length + 2, results[i].Usage!.InputTokenCount);
        }

        var stats = generator.MicroBatchingStatistics!.Value;
        Assert.Equal(1, stats.Batches);
        Assert.Equal(4, stats.Items);
        Assert.Equal(texts.Sum(t => t.Length + 2), stats.Tokens);
        Assert.Equal(4 * (texts.Max(t => t.Length) + 2), stats.TokenCells);
    }

    [Fact]
    public async Task PrefixesApplyPerCallInsideASharedBatch()
    {
        using var generator = Generator(FillTwo);
        var query = generator.GenerateAsync(["x"], new() { AdditionalProperties = new() { [E5EmbeddingGenerator.InputTypePropertyName] = "query" } }, Ct);
        var passage = generator.GenerateAsync(["x"], new() { AdditionalProperties = new() { [E5EmbeddingGenerator.InputTypePropertyName] = "passage" } }, Ct);
        Assert.Equal(Embedder().Embed("query: x", Ct), (await Within(query))[0].Vector.ToArray());
        Assert.Equal(Embedder().Embed("passage: x", Ct), (await Within(passage))[0].Vector.ToArray());
        Assert.Equal(1, generator.MicroBatchingStatistics!.Value.Batches);
    }

    [Fact]
    public async Task MultiValueCallsRunDirectly()
    {
        using var generator = Generator(FillTwo);
        var task = generator.GenerateAsync(["a", "b", "c"], cancellationToken: Ct);
        Assert.True(task.IsCompletedSuccessfully);
        Assert.Equal(3, (await task).Count);
        Assert.Equal(0, generator.MicroBatchingStatistics!.Value.Items);
    }

    [Fact]
    public async Task AnInvalidValueFailsAloneBeforeItIsQueued()
    {
        using var generator = Generator(FillTwo);
        // 600 code points: 602 tokens, above the encoder's 512 (the stand-in tokenizer does not truncate).
        var invalid = generator.GenerateAsync([new string('x', 600)], cancellationToken: Ct);
        await Assert.ThrowsAsync<ArgumentException>(() => Within(invalid));
        var valid = new[] { generator.GenerateAsync(["a"], cancellationToken: Ct), generator.GenerateAsync(["b"], cancellationToken: Ct) };
        await Within(Task.WhenAll(valid));
        var stats = generator.MicroBatchingStatistics!.Value;
        Assert.Equal((1L, 2L), (stats.Batches, stats.Items));
    }

    [Fact]
    public async Task CancellingAQueuedCallLeavesTheOthers()
    {
        using var generator = Generator(FillTwo);
        using var cts = new CancellationTokenSource();
        var cancelled = generator.GenerateAsync(["a"], cancellationToken: cts.Token);
        await cts.CancelAsync();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Within(cancelled));
        Assert.True(cancelled.IsCanceled);

        var b = generator.GenerateAsync(["b"], cancellationToken: Ct);
        var c = generator.GenerateAsync(["c"], cancellationToken: Ct);
        await Within(Task.WhenAll(b, c));
        Assert.Equal(Embedder().Embed("b", Ct), (await b)[0].Vector.ToArray());
        Assert.Equal(2, generator.MicroBatchingStatistics!.Value.Items);
    }

    [Fact]
    public async Task DisposeFaultsQueuedCallsAndLaterCalls()
    {
        var generator = Generator(FillTwo);
        var queued = generator.GenerateAsync(["a"], cancellationToken: Ct);
        Assert.False(queued.IsCompleted);
        generator.Dispose();
        await Assert.ThrowsAsync<ObjectDisposedException>(() => Within(queued));
        await Assert.ThrowsAsync<ObjectDisposedException>(() => generator.GenerateAsync(["b"], cancellationToken: Ct));
    }

    [Fact]
    public void CapabilityIsExperimental() =>
        Assert.Equal(CapabilityStatus.Experimental, E5Capabilities.All["extensions_ai.micro_batching"]);

    private sealed class CodePointTokenizer : IEncoderTokenizer
    {
        public EncoderInput Tokenize(string text)
        {
            var ids = new long[text.Length + 2];
            for (int i = 0; i < text.Length; i++)
            {
                ids[i + 1] = text[i];
            }

            ids[^1] = 2;
            return new EncoderInput(ids);
        }
    }
}
