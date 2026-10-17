using Microsoft.Extensions.AI;
using Starlings.Runtime.Onnx;
using Starlings.Runtime.Onnx.Batching;
using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Extensions.AI.Tests;

/// <summary>
/// Opt-in micro-batching (<see cref="EmbeddingGemma2EmbeddingGeneratorOptions.MicroBatching"/>, Experimental) on the
/// weight-free EmbeddingGemma 2 graph and a code-point tokenizer. A long <see cref="MicroBatchingOptions.MaxWait"/> makes
/// batch composition deterministic: a batch is dispatched only when it holds <see cref="MicroBatchingOptions.MaxRows"/>
/// values.
/// </summary>
public sealed class MicroBatchingTests : IDisposable
{
    private static readonly MicroBatchingOptions FillTwo = new() { MaxWait = TimeSpan.FromHours(1), MaxRows = 2 };

    private readonly EmbeddingGemma2Encoder _encoder = new(
        OnnxModelSession.Create(TestModels.FakeEmbeddingGemma2, new OnnxSessionOptions { IntraOpNumThreads = 1, ComputeModelSha256 = false }),
        new EmbeddingGemma2EncoderOptions { MaxBatchSize = 2, MaxSequenceLength = 64 },
        ownsSession: true);

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose() => _encoder.Dispose();

    private EmbeddingGemma2TextEmbedder Embedder() => new(new CodePointTokenizer(), _encoder);

    private EmbeddingGemma2EmbeddingGenerator Generator(MicroBatchingOptions? microBatching) =>
        new(Embedder(), new EmbeddingGemma2EmbeddingGeneratorOptions { MicroBatching = microBatching });

    private static Task<T> Within<T>(Task<T> task) => task.WaitAsync(TimeSpan.FromSeconds(30), Ct);

    [Fact]
    public async Task OffByDefaultAndCallsStaySynchronous()
    {
        Assert.Null(new EmbeddingGemma2EmbeddingGeneratorOptions().MicroBatching);
        using var generator = Generator(null);
        var task = generator.GenerateAsync(["a"], cancellationToken: Ct);
        Assert.True(task.IsCompletedSuccessfully);
        Assert.Null(generator.MicroBatchingStatistics);
        Assert.Equal(Embedder().Embed("a", Ct), (await task)[0].Vector.ToArray());
    }

    [Fact]
    public void InvalidOptionsAreRejected() =>
        Assert.Throws<ArgumentOutOfRangeException>(() => Generator(new MicroBatchingOptions { MaxTokens = 0 }));

    [Fact]
    public async Task CallsWithDifferentPromptsAndDimensionsShareABatchAndEqualDirectCalls()
    {
        using var generator = Generator(new MicroBatchingOptions { MaxWait = TimeSpan.FromHours(1), MaxRows = 3 });
        using var direct = Generator(null);
        var query = new EmbeddingGenerationOptions { AdditionalProperties = new() { [EmbeddingGemma2EmbeddingGenerator.PromptPropertyName] = "query" } };
        var small = new EmbeddingGenerationOptions { Dimensions = 128 };
        (string Text, EmbeddingGenerationOptions? Options)[] calls = [("a", null), ("a longer text", query), ("mid", small)];
        var tasks = calls.Select(c => generator.GenerateAsync([c.Text], c.Options, Ct)).ToArray();
        var results = await Within(Task.WhenAll(tasks));

        for (int i = 0; i < calls.Length; i++)
        {
            var expected = await direct.GenerateAsync([calls[i].Text], calls[i].Options, Ct);
            Assert.Equal(expected[0].Vector.ToArray(), results[i][0].Vector.ToArray());
            Assert.Equal(expected.Usage!.InputTokenCount, results[i].Usage!.InputTokenCount);
        }

        Assert.Equal(128, results[2][0].Dimensions);
        var stats = generator.MicroBatchingStatistics!.Value;
        Assert.Equal((1L, 3L), (stats.Batches, stats.Items));
    }

    [Fact]
    public async Task AnInputTooLongFailsAloneBeforeItIsQueued()
    {
        using var generator = Generator(FillTwo);
        var tooLong = generator.GenerateAsync([new string('x', 100)], cancellationToken: Ct);
        var error = await Assert.ThrowsAsync<EmbeddingGemma2InputTooLongException>(() => Within(tooLong));
        Assert.Equal(0, error.InputIndex);
        await Within(Task.WhenAll(generator.GenerateAsync(["a"], cancellationToken: Ct), generator.GenerateAsync(["b"], cancellationToken: Ct)));
        var stats = generator.MicroBatchingStatistics!.Value;
        Assert.Equal((1L, 2L), (stats.Batches, stats.Items));
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
        Assert.Equal(Embedder().Embed("c", Ct), (await c)[0].Vector.ToArray());
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
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Experimental, EmbeddingGemma2Capabilities.All["extensions_ai.micro_batching"]);

    private sealed class CodePointTokenizer : IEmbeddingGemma2Tokenizer
    {
        public EmbeddingGemma2Input Tokenize(string text) =>
            new(new long[] { 2 }.Concat(text.Select(c => (long)c + 10)).Append(1).ToArray());
    }
}
