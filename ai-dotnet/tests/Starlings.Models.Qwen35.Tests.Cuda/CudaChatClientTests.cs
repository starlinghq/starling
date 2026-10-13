using Starlings.Generation.OnnxGenAI;
using Starlings.Models.Qwen35.Extensions.AI;
using Starlings.TestSupport;

namespace Starlings.Models.Qwen35.Tests.Cuda;

/// <summary>The fp32x graph on CUDA behind the IChatClient adapter, with 128-token prefill chunks (shared GPU).</summary>
public sealed class Fp32xCudaChatClientFixture : IDisposable
{
    public Fp32xCudaChatClientFixture()
    {
        Reason = CudaGate.Check() ?? string.Empty;
        if (Reason.Length > 0)
        {
            return;
        }

        var (files, reason) = Qwen35TestAssets.FindModel(Qwen35Variant.Fp32x);
        Reason = reason;
        if (files is not null)
        {
            Client = Qwen35ChatClient.Create(
                files,
                new Qwen35ModelOptions { Provider = GenAIExecutionProvider.Cuda },
                new Qwen35ChatClientOptions { Defaults = new Qwen35ChatOptions { PrefillChunkTokens = 128 } });
        }
    }

    public Qwen35ChatClient? Client { get; }

    public string Reason { get; }

    public void Dispose() => Client?.Dispose();
}

/// <summary>P-017 on CUDA: all 21 text prompts at the fixture's 128 tokens, cancellation, concurrent requests (serialized).</summary>
[Collection("cuda")]
public sealed class Fp32xCudaChatClientTests(Fp32xCudaChatClientFixture fixture) : IClassFixture<Fp32xCudaChatClientFixture>
{
    private Qwen35ChatClient Client
    {
        get
        {
            Assert.SkipWhen(fixture.Client is null, fixture.Reason);
            return fixture.Client!;
        }
    }

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static GenerationCase Case(string id) => Qwen35TestAssets.LoadCases().Single(c => c.Id == id);

    public static TheoryData<string> TextCaseIds() => [.. ChatClientModelChecks.TextCases().Select(c => c.Id)];

    [Theory]
    [MemberData(nameof(TextCaseIds))]
    public Task StreamingEqualsResponseAndTheReference(string id) =>
        ChatClientModelChecks.StreamingEqualsResponseAndReference(Client, Client.Model, Case(id), Case(id).MaxNewTokens, Ct);

    [Fact]
    public async Task CancellationBetweenStepsThenReuse()
    {
        var latency = await ChatClientModelChecks.CancelBetweenStepsThenReuse(Client, Client.Model, Case("en-list"), Case("en-short"), 3, Ct);
        TestContext.Current.TestOutputHelper?.WriteLine($"cuda cancel between steps: {latency.TotalMilliseconds:F1} ms from Cancel() to OperationCanceledException");
        Assert.True(latency < TimeSpan.FromSeconds(5), $"{latency}");
    }

    [Fact]
    public async Task CancellationDuringChunkedPrefillThenReuse()
    {
        using var chunked = new Qwen35ChatClient(Client.Model, new Qwen35ChatClientOptions { Defaults = new Qwen35ChatOptions { PrefillChunkTokens = 16 } });
        var (latency, total) = await ChatClientModelChecks.CancelDuringPrefillThenReuse(chunked, Client.Model, Case("long-license"), Case("en-short"), TimeSpan.FromMilliseconds(300), Ct);
        TestContext.Current.TestOutputHelper?.WriteLine(
            $"cuda cancel during prefill (16-token chunks, 1,666-token prompt, cancelled 300 ms in): {latency.TotalMilliseconds:F1} ms from Cancel() to OperationCanceledException, call {total.TotalMilliseconds:F0} ms");
        Assert.True(latency < TimeSpan.FromSeconds(5), $"{latency}");
    }

    [Fact]
    public async Task ConcurrentResponsesEqualSequential()
    {
        Assert.True(Client.Model.Info.SerializesNativeCalls);
        await ChatClientModelChecks.ConcurrentResponsesEqualSequential(
            Client, Client.Model, [Case("zh"), Case("code-py"), Case("ja"), Case("en-math")], 32, rounds: 4, Ct);
    }
}
