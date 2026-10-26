using System.Collections.Concurrent;
using Microsoft.Extensions.AI;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;
using Status = Starlings.Models.CapabilityStatus;

namespace Starlings.Models.EmbeddingGemma2.Extensions.AI.Tests;

/// <summary>
/// <see cref="EmbeddingGemma2EmbeddingGenerator.EmbedStreamAsync"/> and the <c>ExecutionScheduler</c> option of the text
/// and multimodal generators, on the weight-free graphs with a code-point tokenizer. Runs in CI.
/// </summary>
public sealed class StreamingAndSchedulerTests : IDisposable
{
    private static readonly string[] Texts = ["a", "longer text", "", "zz", "mid"];

    private readonly EmbeddingGemma2Encoder _encoder = new(
        OnnxModelSession.Create(TestModels.FakeEmbeddingGemma2, new OnnxSessionOptions { IntraOpNumThreads = 1, ComputeModelSha256 = false }),
        new EmbeddingGemma2EncoderOptions { MaxBatchSize = 2, MaxSequenceLength = 64 },
        ownsSession: true);

    private readonly CodePointTokenizer _tokenizer = new();

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose() => _encoder.Dispose();

    private static async IAsyncEnumerable<string> Source(IEnumerable<string> texts)
    {
        foreach (string text in texts)
        {
            await Task.Yield();
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

    [Fact]
    public async Task TheStreamEqualsOneGenerateCallWithPromptAndDimensions()
    {
        using var generator = new EmbeddingGemma2EmbeddingGenerator(new EmbeddingGemma2TextEmbedder(_tokenizer, _encoder));
        var options = new EmbeddingGenerationOptions { Dimensions = 256, AdditionalProperties = new() { [EmbeddingGemma2EmbeddingGenerator.PromptPropertyName] = "query" } };
        var batch = await generator.GenerateAsync(Texts, options, Ct);
        var streamed = await Collect(generator.EmbedStreamAsync(Source(Texts), options, Ct));

        Assert.Equal(Texts.Length, streamed.Count);
        for (int i = 0; i < Texts.Length; i++)
        {
            Assert.Equal(256, streamed[i].Dimensions);
            Assert.Equal(batch[i].Vector.ToArray(), streamed[i].Vector.ToArray());
        }
    }

    [Fact]
    public async Task AnInputAboveTheLimitFailsTheStreamWithItsPosition()
    {
        using var generator = new EmbeddingGemma2EmbeddingGenerator(new EmbeddingGemma2TextEmbedder(_tokenizer, _encoder));
        var received = new List<Embedding<float>>();
        var ex = await Assert.ThrowsAsync<EmbeddingGemma2InputTooLongException>(async () =>
        {
            await foreach (var embedding in generator.EmbedStreamAsync(Source(["a", "b", "c", new string('x', 70)]), cancellationToken: Ct))
            {
                received.Add(embedding);
            }
        });
        Assert.Equal(3, ex.InputIndex);
        Assert.Equal(72, ex.TokenCount);
        Assert.Equal(64, ex.MaxTokens);
        Assert.Equal(2, received.Count);
    }

    [Fact]
    public async Task WithASchedulerTextGenerationRunsOnIt()
    {
        var scheduler = new CountingTaskScheduler();
        var embedder = new EmbeddingGemma2TextEmbedder(_tokenizer, _encoder);
        using var inline = new EmbeddingGemma2EmbeddingGenerator(embedder);
        using var scheduled = new EmbeddingGemma2EmbeddingGenerator(embedder, new EmbeddingGemma2EmbeddingGeneratorOptions { ExecutionScheduler = scheduler });

        var task = inline.GenerateAsync(Texts, cancellationToken: Ct);
        Assert.True(task.IsCompletedSuccessfully);
        var expected = await task;
        _tokenizer.Threads.Clear();

        var result = await scheduled.GenerateAsync(Texts, cancellationToken: Ct);
        Assert.Equal(1, scheduler.Count);
        Assert.All(_tokenizer.Threads, id => Assert.Contains(id, scheduler.Threads));
        Assert.Equal(expected.Select(e => e.Vector.ToArray()), result.Select(e => e.Vector.ToArray()));

        var streamed = await Collect(scheduled.EmbedStreamAsync(Source(Texts), cancellationToken: Ct));
        Assert.Equal(1 + 3, scheduler.Count); // 5 values, 2 rows per group
        Assert.Equal(expected.Select(e => e.Vector.ToArray()), streamed.Select(e => e.Vector.ToArray()));
    }

    [Fact]
    public async Task WithASchedulerMultimodalGenerationRunsOnIt()
    {
        using var embedder = FakeMultimodal.Create();
        var scheduler = new CountingTaskScheduler();
        var decoderThreads = new ConcurrentBag<int>();
        EmbeddingGemma2Image Decode(ReadOnlyMemory<byte> bytes, string mediaType)
        {
            decoderThreads.Add(Environment.CurrentManagedThreadId);
            return FakeMultimodal.Image(61, 40, 1);
        }

        using var generator = new EmbeddingGemma2MultimodalEmbeddingGenerator(embedder, new EmbeddingGemma2MultimodalEmbeddingGeneratorOptions { ImageDecoder = Decode, ExecutionScheduler = scheduler });
        var result = await generator.GenerateAsync(new[] { new DataContent(new byte[] { 0x89, 0x50, 0x4E, 0x47, 1 }, "image/png") }, cancellationToken: Ct);

        Assert.Equal(1, scheduler.Count);
        Assert.Contains(Assert.Single(decoderThreads), scheduler.Threads);
        Assert.Equal(embedder.Embed(FakeMultimodal.Image(61, 40, 1), Ct), result[0].Vector.ToArray());
    }

    [Fact]
    public void TheNewCapabilitiesAreExperimental()
    {
        Assert.Equal(Status.Experimental, EmbeddingGemma2Capabilities.Statuses["extensions_ai.stream"]);
        Assert.Equal(Status.Experimental, EmbeddingGemma2Capabilities.Statuses["extensions_ai.execution_scheduler"]);
        Assert.Equal(Status.Unsupported, EmbeddingGemma2Capabilities.Statuses["streaming"]);
    }

    private sealed class CodePointTokenizer : IEmbeddingGemma2Tokenizer
    {
        public ConcurrentQueue<int> Threads { get; } = new();

        public EmbeddingGemma2Input Tokenize(string text)
        {
            Threads.Enqueue(Environment.CurrentManagedThreadId);
            return new EmbeddingGemma2Input(new long[] { 2 }.Concat(text.Select(c => (long)c + 10)).Append(1).ToArray());
        }
    }
}
