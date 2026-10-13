using System.Globalization;
using Starlings.TestSupport;

namespace Starlings.Models.Qwen35.Extensions.AI.Tests;

/// <summary>Loads the fp32x bundle on CPU once (skips without it: not in Git, 3.1 GB).</summary>
public sealed class Qwen35ChatClientFixture : IDisposable
{
    public Qwen35ChatClientFixture()
    {
        var (files, reason) = Qwen35TestAssets.FindModel(Qwen35Variant.Fp32x);
        Reason = reason;
        if (files is not null)
        {
            Client = Qwen35ChatClient.Create(files);
        }
    }

    public Qwen35ChatClient? Client { get; }

    public string Reason { get; }

    public void Dispose() => Client?.Dispose();
}

/// <summary>
/// The IChatClient on the fp32x model on CPU against the transformers fp32 reference (prompts-v1). Completion length
/// per prompt: <c>$STARLING_QWEN35_CHAT_TOKENS</c>, default 32 (the fixture has 128; 32 keeps the run near
/// three minutes on a 6-core CPU).
/// </summary>
public sealed class RealModelTests(Qwen35ChatClientFixture fixture) : IClassFixture<Qwen35ChatClientFixture>
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

    private static int Tokens => Environment.GetEnvironmentVariable("STARLING_QWEN35_CHAT_TOKENS") is { Length: > 0 } v
        ? int.Parse(v, CultureInfo.InvariantCulture)
        : 32;

    private static GenerationCase Case(string id) => Qwen35TestAssets.LoadCases().Single(c => c.Id == id);

    public static TheoryData<string> TextCaseIds() => [.. ChatClientModelChecks.TextCases().Select(c => c.Id)];

    [Theory]
    [MemberData(nameof(TextCaseIds))]
    public Task StreamingEqualsResponseAndTheReference(string id) =>
        ChatClientModelChecks.StreamingEqualsResponseAndReference(Client, Client.Model, Case(id), Math.Min(Tokens, Case(id).MaxNewTokens), Ct);

    [Fact]
    public async Task CancellationBetweenStepsThenReuse()
    {
        var latency = await ChatClientModelChecks.CancelBetweenStepsThenReuse(Client, Client.Model, Case("en-list"), Case("en-short"), 3, Ct);
        TestContext.Current.TestOutputHelper?.WriteLine($"cpu cancel between steps: {latency.TotalMilliseconds:F1} ms from Cancel() to OperationCanceledException");
        Assert.True(latency < TimeSpan.FromSeconds(5), $"{latency}");
    }

    [Fact]
    public async Task CancellationDuringChunkedPrefillThenReuse()
    {
        using var chunked = new Qwen35ChatClient(Client.Model, new Qwen35ChatClientOptions { Defaults = new Qwen35ChatOptions { PrefillChunkTokens = 64 } });
        var (latency, total) = await ChatClientModelChecks.CancelDuringPrefillThenReuse(chunked, Client.Model, Case("long-license"), Case("en-short"), TimeSpan.FromSeconds(1), Ct);
        TestContext.Current.TestOutputHelper?.WriteLine(
            $"cpu cancel during prefill (64-token chunks, 1,666-token prompt, cancelled 1 s in): {latency.TotalMilliseconds:F1} ms from Cancel() to OperationCanceledException, call {total.TotalMilliseconds:F0} ms");
        Assert.True(latency < TimeSpan.FromSeconds(5), $"{latency}");
    }

    [Fact]
    public async Task ConcurrentResponsesEqualSequential()
    {
        Assert.False(Client.Model.Info.SerializesNativeCalls);
        await ChatClientModelChecks.ConcurrentResponsesEqualSequential(Client, Client.Model, [Case("zh"), Case("code-py"), Case("ja"), Case("en-math")], 12, rounds: 2, Ct);
    }
}
