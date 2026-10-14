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

/// <summary>The fp16x graph on CUDA behind the IChatClient adapter, with 128-token prefill chunks.</summary>
public sealed class Fp16xCudaChatClientFixture : IDisposable
{
    public Fp16xCudaChatClientFixture()
    {
        Reason = CudaGate.Check() ?? string.Empty;
        if (Reason.Length > 0)
        {
            return;
        }

        var (files, reason) = Qwen35TestAssets.FindModel(Qwen35Variant.Fp16x);
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

/// <summary>
/// Capability <c>extensions_ai</c> of <c>ortgenai-fp16x-cuda</c>: the IChatClient adapter over the fp16x model. fp16 may
/// leave the fp32 reference at near ties (the model tests hold the typed API to the profile's near-tie rule), so here the
/// adapter is held to the typed model on the same fp16x session: for every text prompt the response and the streamed
/// updates equal <see cref="Qwen35ChatModel.Generate(IReadOnlyList{Qwen35Message}, Qwen35ChatOptions?, CancellationToken)"/>
/// exactly, with usage and finish reason; cancellation and serialized concurrent requests as on fp32x.
/// </summary>
[Collection("cuda")]
public sealed class Fp16xCudaChatClientTests(Fp16xCudaChatClientFixture fixture) : IClassFixture<Fp16xCudaChatClientFixture>
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
    public async Task ResponseAndStreamingEqualTheTypedModel(string id)
    {
        var c = Case(id);
        var client = Client;
        var messages = ChatClientModelChecks.ToChatMessages(c);
        var options = new Microsoft.Extensions.AI.ChatOptions { MaxOutputTokens = c.MaxNewTokens };
        if (c.EnableThinking)
        {
            options.AdditionalProperties = new Microsoft.Extensions.AI.AdditionalPropertiesDictionary { [Qwen35ChatClient.EnableThinkingPropertyName] = true };
        }

        var typed = client.Model.Generate(Qwen35Prompt.FromTokenIds(c.PromptIds), new Qwen35ChatOptions { MaxNewTokens = c.MaxNewTokens, PrefillChunkTokens = 128 }, Ct);
        var response = await client.GetResponseAsync(messages, options, Ct);
        var updates = new List<Microsoft.Extensions.AI.ChatResponseUpdate>();
        await foreach (var u in client.GetStreamingResponseAsync(messages, options, Ct))
        {
            updates.Add(u);
        }

        Assert.Equal(typed.Text, response.Text);
        Assert.Equal(response.Text, string.Concat(updates.Select(u => u.Text)));
        Assert.Equal(c.PromptIds.Length, response.Usage!.InputTokenCount);
        Assert.Equal(typed.TokenIds.Length, response.Usage.OutputTokenCount);
        Assert.Equal(response.FinishReason, updates[^1].FinishReason);
    }

    [Fact]
    public async Task CancellationBetweenStepsThenReuse()
    {
        var client = Client;
        var shortCase = Case("en-short");
        using var cts = new CancellationTokenSource();
        int seen = 0;
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var _ in client.GetStreamingResponseAsync(ChatClientModelChecks.ToChatMessages(Case("en-list")), new Microsoft.Extensions.AI.ChatOptions { MaxOutputTokens = 256 }, cts.Token))
            {
                if (++seen == 3)
                {
                    await cts.CancelAsync();
                }
            }
        });
        var typed = client.Model.Generate(Qwen35Prompt.FromTokenIds(shortCase.PromptIds), new Qwen35ChatOptions { MaxNewTokens = 32, PrefillChunkTokens = 128 }, Ct);
        var after = await client.GetResponseAsync(ChatClientModelChecks.ToChatMessages(shortCase), new Microsoft.Extensions.AI.ChatOptions { MaxOutputTokens = 32 }, Ct);
        Assert.Equal(typed.Text, after.Text);
    }

    [Fact]
    public async Task ConcurrentResponsesEqualSequential()
    {
        var client = Client;
        Assert.True(client.Model.Info.SerializesNativeCalls);
        var cases = new[] { Case("zh"), Case("code-py"), Case("ja"), Case("en-math") };
        var options = new Microsoft.Extensions.AI.ChatOptions { MaxOutputTokens = 32 };
        var sequential = new List<string>();
        foreach (var c in cases)
        {
            sequential.Add((await client.GetResponseAsync(ChatClientModelChecks.ToChatMessages(c), options, Ct)).Text);
        }

        for (int round = 0; round < 4; round++)
        {
            var concurrent = await Task.WhenAll(cases.Select(c => client.GetResponseAsync(ChatClientModelChecks.ToChatMessages(c), options, Ct)));
            Assert.Equal(sequential, concurrent.Select(r => r.Text));
        }
    }
}
