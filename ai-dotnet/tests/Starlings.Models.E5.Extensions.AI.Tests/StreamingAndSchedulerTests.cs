using Microsoft.Extensions.AI;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.E5.Extensions.AI.Tests;

/// <summary>
/// <see cref="E5EmbeddingGenerator.EmbedStreamAsync"/> and <see cref="E5EmbeddingGeneratorOptions.ExecutionScheduler"/>
/// on the weight-free e5 graph with a code-point tokenizer. Runs in CI.
/// </summary>
public sealed class StreamingAndSchedulerTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static readonly string[] Texts = ["a", "longer text", "", "zz", "mid", "the last one", "x"];

    private static E5Encoder Encoder(int? maxBatchTokens = null) => new(
        OnnxModelSession.Create(TestModels.FakeE5, new OnnxSessionOptions { IntraOpNumThreads = 1, ComputeModelSha256 = false }),
        new E5EncoderOptions { MaxBatchSize = 2, MaxBatchTokens = maxBatchTokens },
        ownsSession: true);

    private static async IAsyncEnumerable<string> Source(IEnumerable<string> texts, Action<int>? onYield = null)
    {
        int i = 0;
        foreach (string text in texts)
        {
            await Task.Yield();
            onYield?.Invoke(i++);
            yield return text;
        }
    }

    private static async Task<List<Embedding<float>>> Collect(IAsyncEnumerable<Embedding<float>> stream)
    {
        var list = new List<Embedding<float>>();
        await foreach (var embedding in stream)
        {
            list.Add(embedding);
        }

        return list;
    }

    [Theory]
    [InlineData(null)]
    [InlineData(30)]
    public async Task TheStreamEqualsOneGenerateCallBitForBit(int? maxBatchTokens)
    {
        using var encoder = Encoder(maxBatchTokens);
        var tokenizer = new CodePointTokenizer();
        using var generator = new E5EmbeddingGenerator(new E5TextEmbedder(tokenizer, encoder), new E5EmbeddingGeneratorOptions { InputType = E5InputType.Query });

        var batch = await generator.GenerateAsync(Texts, cancellationToken: Ct);
        var streamed = await Collect(generator.EmbedStreamAsync(Source(Texts), cancellationToken: Ct));

        Assert.Equal(Texts.Length, streamed.Count);
        for (int i = 0; i < Texts.Length; i++)
        {
            Assert.Equal(batch[i].Vector.ToArray(), streamed[i].Vector.ToArray());
            Assert.Equal(E5EmbeddingGeneratorOptions.PinnedModelId, streamed[i].ModelId);
        }

        Assert.Contains("query: longer text", tokenizer.Seen);
    }

    [Fact]
    public async Task TheStreamYieldsBeforeTheSourceEnds()
    {
        using var encoder = Encoder();
        using var generator = new E5EmbeddingGenerator(new E5TextEmbedder(new CodePointTokenizer(), encoder));
        int pulled = 0;
        int seen = 0;
        await foreach (var embedding in generator.EmbedStreamAsync(Source(Enumerable.Repeat("abc", 1_000_000), i => pulled = i + 1), cancellationToken: Ct))
        {
            Assert.Equal(4, embedding.Dimensions);
            if (++seen == 3)
            {
                break;
            }
        }

        // Two rows per group: the third embedding is yielded once the second group runs, after at most five values.
        Assert.InRange(pulled, 3, 5);
    }

    [Fact]
    public async Task ANullValueFailsTheStreamNamingItsPosition()
    {
        using var encoder = Encoder();
        using var generator = new E5EmbeddingGenerator(new E5TextEmbedder(new CodePointTokenizer(), encoder));
        var received = new List<Embedding<float>>();
        var ex = await Assert.ThrowsAsync<ArgumentException>(async () =>
        {
            await foreach (var embedding in generator.EmbedStreamAsync(Source(["a", "b", "c", null!]), cancellationToken: Ct))
            {
                received.Add(embedding);
            }
        });
        Assert.Contains("Value 3", ex.Message, StringComparison.Ordinal);
        Assert.Equal(2, received.Count);
        Assert.Throws<ArgumentNullException>(() => generator.EmbedStreamAsync(null!, cancellationToken: Ct));
    }

    [Fact]
    public async Task UnsupportedOptionsFailTheStreamBeforeAnyValueIsRead()
    {
        using var encoder = Encoder();
        using var generator = new E5EmbeddingGenerator(new E5TextEmbedder(new CodePointTokenizer(), encoder));
        int pulled = 0;
        await Assert.ThrowsAsync<NotSupportedException>(async () =>
            await Collect(generator.EmbedStreamAsync(Source(Texts, _ => pulled++), new EmbeddingGenerationOptions { Dimensions = 8 }, Ct)));
        Assert.Equal(0, pulled);
    }

    [Fact]
    public async Task ACancelledStreamStops()
    {
        using var encoder = Encoder();
        using var generator = new E5EmbeddingGenerator(new E5TextEmbedder(new CodePointTokenizer(), encoder));
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(Ct);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var _ in generator.EmbedStreamAsync(Source(Enumerable.Repeat("abc", 1_000_000)), cancellationToken: cts.Token))
            {
                await cts.CancelAsync();
            }
        });
    }

    [Fact]
    public async Task WithoutASchedulerGenerateRunsInline()
    {
        using var encoder = Encoder();
        var tokenizer = new CodePointTokenizer();
        using var generator = new E5EmbeddingGenerator(new E5TextEmbedder(tokenizer, encoder));
        var task = generator.GenerateAsync(["abc"], cancellationToken: Ct);
        Assert.True(task.IsCompletedSuccessfully);
        Assert.Equal(new[] { Environment.CurrentManagedThreadId }, tokenizer.Threads);
        await task;
    }

    [Fact]
    public async Task WithASchedulerGenerateRunsOnItWithTheSameNumbers()
    {
        using var encoder = Encoder();
        var tokenizer = new CodePointTokenizer();
        var scheduler = new CountingTaskScheduler();
        var embedder = new E5TextEmbedder(tokenizer, encoder);
        using var inline = new E5EmbeddingGenerator(embedder);
        using var scheduled = new E5EmbeddingGenerator(embedder, new E5EmbeddingGeneratorOptions { ExecutionScheduler = scheduler });

        var expected = await inline.GenerateAsync(Texts, cancellationToken: Ct);
        tokenizer.Threads.Clear();
        var result = await scheduled.GenerateAsync(Texts, cancellationToken: Ct);

        Assert.Equal(1, scheduler.Count);
        Assert.All(tokenizer.Threads, id => Assert.Contains(id, scheduler.Threads));
        Assert.Equal(expected.Select(e => e.Vector.ToArray()), result.Select(e => e.Vector.ToArray()));
        Assert.Equal(expected.Usage!.InputTokenCount, result.Usage!.InputTokenCount);

        // Failures and cancellation are in the task, as inline.
        await Assert.ThrowsAsync<ArgumentException>(() => scheduled.GenerateAsync(["a", null!], cancellationToken: Ct));
        var cancelled = scheduled.GenerateAsync(["a"], cancellationToken: new CancellationToken(canceled: true));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => cancelled);
        Assert.True(cancelled.IsCanceled);
    }

    [Fact]
    public async Task WithASchedulerTheStreamRunsEachGroupOnIt()
    {
        using var encoder = Encoder();
        var scheduler = new CountingTaskScheduler();
        var embedder = new E5TextEmbedder(new CodePointTokenizer(), encoder);
        using var generator = new E5EmbeddingGenerator(embedder, new E5EmbeddingGeneratorOptions { ExecutionScheduler = scheduler });
        var streamed = await Collect(generator.EmbedStreamAsync(Source(Texts), cancellationToken: Ct));

        Assert.Equal(4, scheduler.Count); // 7 values, 2 rows per group
        Assert.Equal(embedder.EmbedBatch(Texts, Ct), streamed.Select(e => e.Vector.ToArray()));
    }

    [Fact]
    public void TheNewCapabilitiesAreExperimental()
    {
        Assert.Equal(Starlings.Models.CapabilityStatus.Experimental, E5Capabilities.Statuses["extensions_ai.stream"]);
        Assert.Equal(Starlings.Models.CapabilityStatus.Experimental, E5Capabilities.Statuses["extensions_ai.execution_scheduler"]);
        Assert.Equal(Starlings.Models.CapabilityStatus.Unsupported, E5Capabilities.Statuses["streaming"]);
    }

    private sealed class CodePointTokenizer : IEncoderTokenizer
    {
        public System.Collections.Concurrent.ConcurrentQueue<string> Seen { get; } = new();

        public System.Collections.Concurrent.ConcurrentQueue<int> Threads { get; } = new();

        public EncoderInput Tokenize(string text)
        {
            Seen.Enqueue(text);
            Threads.Enqueue(Environment.CurrentManagedThreadId);
            return new EncoderInput(new long[] { 0 }.Concat(text.Select(c => (long)c)).Append(2).ToArray());
        }
    }
}
