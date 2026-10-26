using Microsoft.Extensions.AI;
using Starlings.Models.E5.Browser;
using Starlings.Models.E5.Extensions.AI;

namespace Starlings.Models.E5.Browser.Tests;

public sealed class BrowserGeneratorTests
{
    /// <summary>One token per character (id = char code + 3) between &lt;s&gt; and &lt;/s&gt;; records the texts it saw.</summary>
    private sealed class CharTokenizer : IEncoderTokenizer
    {
        public List<string> Seen { get; } = [];

        public EncoderInput Tokenize(string text)
        {
            Seen.Add(text);
            return new EncoderInput((long[])[0, .. text.Select(c => (long)c + 3), 2]);
        }
    }

    private static (E5BrowserEmbeddingGenerator Generator, CharTokenizer Tokenizer, FakeGraphRunner Runner) Create(E5InputType inputType = E5InputType.None)
    {
        var tokenizer = new CharTokenizer();
        var runner = new FakeGraphRunner(dimension: 384);
        var embedder = new E5BrowserEmbedder(tokenizer, new E5BrowserEncoder(runner, null));
        return (new E5BrowserEmbeddingGenerator(embedder, inputType), tokenizer, runner);
    }

    [Fact]
    public async Task EmbedsValuesInOrderWithUsageAndMetadata()
    {
        var (generator, tokenizer, runner) = Create();
        var result = await generator.GenerateAsync(["ab", "c"], cancellationToken: TestContext.Current.CancellationToken);

        Assert.Equal(2, result.Count);
        Assert.Equal(runner.Expected(tokenizer.Tokenize("ab")), result[0].Vector.ToArray());
        Assert.Equal(runner.Expected(tokenizer.Tokenize("c")), result[1].Vector.ToArray());
        Assert.Equal(7, result.Usage!.InputTokenCount);
        Assert.Equal(E5EmbeddingGeneratorOptions.PinnedModelId, result[0].ModelId);
        Assert.Equal(E5BrowserEmbeddingGenerator.ProviderName, generator.Metadata.ProviderName);
        Assert.Equal(384, generator.Metadata.DefaultModelDimensions);
        Assert.Single(runner.Calls);
    }

    [Fact]
    public async Task AppliesTheDefaultInputTypeAndThePerRequestOverride()
    {
        var (generator, tokenizer, _) = Create(E5InputType.Query);
        await generator.GenerateAsync(["x"], cancellationToken: TestContext.Current.CancellationToken);
        await generator.GenerateAsync(
            ["y"],
            new EmbeddingGenerationOptions { AdditionalProperties = new() { [E5EmbeddingGenerator.InputTypePropertyName] = "passage" } },
            TestContext.Current.CancellationToken);
        Assert.Equal(["query: x", "passage: y"], tokenizer.Seen);
    }

    public static TheoryData<EmbeddingGenerationOptions> Unsupported => new()
    {
        new EmbeddingGenerationOptions { Dimensions = 128 },
        new EmbeddingGenerationOptions { ModelId = "other" },
        new EmbeddingGenerationOptions { AdditionalProperties = new() { ["temperature"] = 1 } },
        new EmbeddingGenerationOptions { RawRepresentationFactory = _ => new object() },
    };

    [Theory]
    [MemberData(nameof(Unsupported))]
    public async Task RejectsUnsupportedOptionsThroughTheTask(EmbeddingGenerationOptions options)
    {
        var (generator, _, runner) = Create();
        var task = generator.GenerateAsync(["x"], options, TestContext.Current.CancellationToken);
        await Assert.ThrowsAsync<NotSupportedException>(() => task);
        Assert.Empty(runner.Calls);
    }

    [Fact]
    public async Task AcceptsTheModelDimensionAndId()
    {
        var (generator, _, _) = Create();
        var options = new EmbeddingGenerationOptions { Dimensions = 384, ModelId = E5EmbeddingGeneratorOptions.PinnedModelId };
        Assert.Single(await generator.GenerateAsync(["x"], options, TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task FailuresAndCancellationComeThroughTheTask()
    {
        var (generator, _, _) = Create();
        Task nulls = generator.GenerateAsync([null!], cancellationToken: TestContext.Current.CancellationToken);
        await Assert.ThrowsAsync<ArgumentException>(() => nulls);

        using var cts = new CancellationTokenSource();
        await cts.CancelAsync();
        Task cancelled = generator.GenerateAsync(["x"], cancellationToken: cts.Token);
        Assert.True(cancelled.IsCanceled);
    }

    [Fact]
    public async Task GetServiceAndDispose()
    {
        var (generator, _, runner) = Create();
        Assert.Same(generator.Embedder, generator.GetService(typeof(E5BrowserEmbedder)));
        Assert.Same(generator.Embedder.Encoder, generator.GetService(typeof(E5BrowserEncoder)));
        Assert.NotNull(generator.GetService(typeof(EmbeddingGeneratorMetadata)));
        Assert.Null(generator.GetService(typeof(EmbeddingGeneratorMetadata), "key"));
        generator.Dispose();
        Assert.Equal(0, runner.DisposeCount); // Not owned.
        await Assert.ThrowsAsync<ObjectDisposedException>(() => generator.GenerateAsync(["x"], cancellationToken: TestContext.Current.CancellationToken));
    }
}
