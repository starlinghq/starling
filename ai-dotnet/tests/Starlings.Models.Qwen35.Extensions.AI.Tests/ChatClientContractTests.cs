using System.Collections.Concurrent;
using System.Reflection;
using System.Text.Json;
using Starlings.Generation.OnnxGenAI;
using Starlings.TestSupport;
using Starlings.Tokenizers;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Caching.Distributed;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace Starlings.Models.Qwen35.Extensions.AI.Tests;

/// <summary>
/// The IChatClient contract over the real template, tokenizer, decode loop and detokenizer with a scripted native
/// sequence (no model weights; runs in CI with the pinned tokenizer.json).
/// </summary>
public sealed class ChatClientContractTests
{
    // Multi-byte characters split across byte-level tokens exercise the UTF-8 hold-back.
    private const string Reply = "Hello! The capital of France is Paris.\n\nIt is known for 你好 and 🙂🧪 emoji.";

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static Tokenizer Tokenizer
    {
        get
        {
            var (tokenizer, reason) = Qwen35TestAssets.Tokenizer;
            Assert.SkipWhen(tokenizer is null, reason);
            return tokenizer!;
        }
    }

    private static int[] ReplyIds(Tokenizer tokenizer, string text = Reply) => tokenizer.Encode(text, addSpecialTokens: false).Ids.ToArray();

    private static ScriptedBackend Backend(string text = Reply, int stop = Qwen35ChatModel.ImEndTokenId)
    {
        var tokenizer = Tokenizer;
        int[] ids = [.. ReplyIds(tokenizer, text), stop];
        return new ScriptedBackend(tokenizer, _ => ids);
    }

    private static ChatMessage[] Conversation => [new(ChatRole.System, "Be brief."), new(ChatRole.User, "What is the capital of France?")];

    private static Qwen35Message[] TypedConversation => [Qwen35Message.System("Be brief."), Qwen35Message.User("What is the capital of France?")];

    [Fact]
    public async Task ResponseCarriesTextFinishReasonAndUsageWithoutTheStopToken()
    {
        var backend = Backend();
        using var client = new Qwen35ChatClient(backend.Model);
        ChatResponse response = await client.GetResponseAsync(Conversation, cancellationToken: Ct);

        var expectedPrompt = backend.Model.CreatePrompt(TypedConversation);
        Assert.Equal(expectedPrompt.TokenIds, backend.Single().Prompt);
        Assert.Equal(Reply, response.Text);
        Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
        int completion = ReplyIds(backend.Tokenizer).Length;
        Assert.Equal(expectedPrompt.TokenIds.Length, response.Usage!.InputTokenCount);
        Assert.Equal(completion, response.Usage.OutputTokenCount);
        Assert.Equal(expectedPrompt.TokenIds.Length + completion, response.Usage.TotalTokenCount);
        var message = Assert.Single(response.Messages);
        Assert.Equal(ChatRole.Assistant, message.Role);
        Assert.IsType<TextContent>(Assert.Single(message.Contents));
        Assert.Equal(client.ModelId, response.ModelId);
        Assert.Equal("Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17", response.ModelId);
        Assert.NotNull(response.ResponseId);
        Assert.Equal(response.ResponseId, message.MessageId);
        Assert.NotNull(response.CreatedAt);
        var raw = Assert.IsType<GenerationResult>(response.RawRepresentation);
        Assert.Equal(Qwen35ChatModel.ImEndTokenId, raw.StopTokenId);
        Assert.DoesNotContain(Qwen35ChatModel.ImEndTokenId, raw.TokenIds);
    }

    [Fact]
    public async Task StreamingReconstructsTheResponseWithFinishReasonAndUsageOnTheLastUpdate()
    {
        var backend = Backend();
        using var client = new Qwen35ChatClient(backend.Model);
        ChatResponse response = await client.GetResponseAsync(Conversation, cancellationToken: Ct);
        var updates = await ToListAsync(client.GetStreamingResponseAsync(Conversation, cancellationToken: Ct));

        Assert.True(updates.Count > 5);
        Assert.Equal(response.Text, string.Concat(updates.Select(u => u.Text)));
        Assert.All(updates.SkipLast(1), u =>
        {
            Assert.NotEmpty(u.Text);
            Assert.Null(u.FinishReason);
            Assert.DoesNotContain(u.Contents, c => c is UsageContent);
            Assert.False(char.IsHighSurrogate(u.Text[^1]), "an update ends inside a surrogate pair");
            Assert.DoesNotContain('�', u.Text);
        });
        var last = updates[^1];
        Assert.Equal(ChatFinishReason.Stop, last.FinishReason);
        var usage = Assert.Single(last.Contents.OfType<UsageContent>()).Details;
        Assert.Equal(response.Usage!.InputTokenCount, usage.InputTokenCount);
        Assert.Equal(response.Usage.OutputTokenCount, usage.OutputTokenCount);
        Assert.Equal(response.Usage.TotalTokenCount, usage.TotalTokenCount);
        Assert.All(updates, u =>
        {
            Assert.Equal(ChatRole.Assistant, u.Role);
            Assert.Equal(updates[0].ResponseId, u.ResponseId);
            Assert.Equal(updates[0].ResponseId, u.MessageId);
            Assert.Equal(updates[0].CreatedAt, u.CreatedAt);
            Assert.Equal(client.ModelId, u.ModelId);
            Assert.IsType<GenerationUpdate>(u.RawRepresentation);
        });

        // The standard aggregation gives the same response.
        ChatResponse combined = updates.ToChatResponse();
        Assert.Equal(response.Text, combined.Text);
        Assert.Equal(response.FinishReason, combined.FinishReason);
        Assert.Equal(response.Usage.InputTokenCount, combined.Usage!.InputTokenCount);
        Assert.Equal(response.Usage.OutputTokenCount, combined.Usage.OutputTokenCount);
        Assert.Single(combined.Messages);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    public async Task MaxOutputTokensEndsWithLengthInBothModes(int max)
    {
        var backend = Backend();
        using var client = new Qwen35ChatClient(backend.Model);
        var options = new ChatOptions { MaxOutputTokens = max };
        ChatResponse response = await client.GetResponseAsync(Conversation, options, Ct);
        var updates = await ToListAsync(client.GetStreamingResponseAsync(Conversation, options, Ct));

        Assert.Equal(ChatFinishReason.Length, response.FinishReason);
        Assert.Equal(max, response.Usage!.OutputTokenCount);
        Assert.Equal(backend.Tokenizer.Decode(ReplyIds(backend.Tokenizer).AsSpan(0, max), false), response.Text);
        Assert.Equal(response.Text, string.Concat(updates.Select(u => u.Text)));
        Assert.Equal(ChatFinishReason.Length, updates[^1].FinishReason);
        Assert.Equal(max, updates[^1].Contents.OfType<UsageContent>().Single().Details.OutputTokenCount);
        Assert.All(backend.Sequences, s => Assert.Equal(max, s.Steps));
    }

    [Fact]
    public async Task StopSequenceMapsToStopAndIsNeverStreamed()
    {
        var backend = Backend();
        using var client = new Qwen35ChatClient(backend.Model);
        var options = new ChatOptions { StopSequences = ["\n\n"] };
        ChatResponse response = await client.GetResponseAsync(Conversation, options, Ct);
        var updates = await ToListAsync(client.GetStreamingResponseAsync(Conversation, options, Ct));

        Assert.Equal("Hello! The capital of France is Paris.", response.Text);
        Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
        Assert.Equal(GenerationFinishReason.StopSequence, ((GenerationResult)response.RawRepresentation!).FinishReason);
        Assert.Equal(response.Text, string.Concat(updates.Select(u => u.Text)));
        Assert.DoesNotContain(updates, u => u.Text.Contains('\n', StringComparison.Ordinal));
        Assert.Equal(ChatFinishReason.Stop, updates[^1].FinishReason);
        Assert.Equal(response.Usage!.OutputTokenCount, updates[^1].Contents.OfType<UsageContent>().Single().Details.OutputTokenCount);
    }

    [Fact]
    public async Task EndOfTextStopsLikeImEnd()
    {
        var backend = Backend("Short.", Qwen35ChatModel.EndOfTextTokenId);
        using var client = new Qwen35ChatClient(backend.Model);
        ChatResponse response = await client.GetResponseAsync(Conversation, cancellationToken: Ct);
        Assert.Equal("Short.", response.Text);
        Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
        Assert.Equal(ReplyIds(backend.Tokenizer, "Short.").Length, response.Usage!.OutputTokenCount);
    }

    [Fact]
    public async Task AnEmptyCompletionHasNoTextButAFinalUpdate()
    {
        var tokenizer = Tokenizer;
        var backend = new ScriptedBackend(tokenizer, _ => [Qwen35ChatModel.ImEndTokenId]);
        using var client = new Qwen35ChatClient(backend.Model);
        ChatResponse response = await client.GetResponseAsync(Conversation, cancellationToken: Ct);
        var update = Assert.Single(await ToListAsync(client.GetStreamingResponseAsync(Conversation, cancellationToken: Ct)));

        Assert.Equal(string.Empty, response.Text);
        Assert.Empty(response.Messages[0].Contents);
        Assert.Equal(0, response.Usage!.OutputTokenCount);
        Assert.Equal(ChatFinishReason.Stop, update.FinishReason);
        Assert.Equal(0, update.Contents.OfType<UsageContent>().Single().Details.OutputTokenCount);
    }

    [Fact]
    public async Task RolesContentsAndReasoningRenderLikeTheTypedApi()
    {
        var backend = Backend();
        using var client = new Qwen35ChatClient(backend.Model);
        ChatMessage[] messages =
        [
            new(ChatRole.System, [new TextContent("Be "), new TextContent("brief.")]),
            new(ChatRole.User, "Hi"),
            new(ChatRole.Assistant, [new TextReasoningContent("The user greets."), new TextContent("Hello!")]),
            new(ChatRole.User, "Capital of France?"),
        ];
        await client.GetResponseAsync(messages, cancellationToken: Ct);
        var expected = backend.Model.CreatePrompt(
        [
            Qwen35Message.System("Be brief."),
            Qwen35Message.User("Hi"),
            Qwen35Message.Assistant("Hello!", "The user greets."),
            Qwen35Message.User("Capital of France?"),
        ]);
        Assert.Equal(expected.TokenIds, backend.Single().Prompt);
    }

    [Fact]
    public async Task InstructionsBecomeTheSystemMessage()
    {
        var backend = Backend();
        using var client = new Qwen35ChatClient(backend.Model);
        await client.GetResponseAsync([new ChatMessage(ChatRole.User, "What is the capital of France?")], new ChatOptions { Instructions = "Be brief." }, Ct);
        Assert.Equal(backend.Model.CreatePrompt(TypedConversation).TokenIds, backend.Single().Prompt);
    }

    [Fact]
    public async Task EnableThinkingIsAnExplicitAdditionalProperty()
    {
        var backend = Backend();
        using var client = new Qwen35ChatClient(backend.Model);
        var thinking = backend.Model.CreatePrompt(TypedConversation, enableThinking: true).TokenIds;
        var direct = backend.Model.CreatePrompt(TypedConversation).TokenIds;
        Assert.NotEqual(thinking, direct);

        await client.GetResponseAsync(Conversation, Props(Qwen35ChatClient.EnableThinkingPropertyName, true), Ct);
        await client.GetResponseAsync(Conversation, Props(Qwen35ChatClient.EnableThinkingPropertyName, JsonDocument.Parse("true").RootElement), Ct);
        await client.GetResponseAsync(Conversation, Props(Qwen35ChatClient.EnableThinkingPropertyName, false), Ct);
        await ToListAsync(client.GetStreamingResponseAsync(Conversation, Props(Qwen35ChatClient.EnableThinkingPropertyName, true), Ct));
        var prompts = backend.Sequences.Select(s => s.Prompt).ToList();
        Assert.Equal(thinking, prompts[0]);
        Assert.Equal(thinking, prompts[1]);
        Assert.Equal(direct, prompts[2]);
        Assert.Equal(thinking, prompts[3]);

        // The client default applies when the request does not set it.
        using var thinkingClient = new Qwen35ChatClient(backend.Model, new Qwen35ChatClientOptions { Defaults = new Qwen35ChatOptions { EnableThinking = true } });
        await thinkingClient.GetResponseAsync(Conversation, cancellationToken: Ct);
        Assert.Equal(thinking, backend.Sequences.Last().Prompt);

        await Assert.ThrowsAsync<ArgumentException>(() => client.GetResponseAsync(Conversation, Props(Qwen35ChatClient.EnableThinkingPropertyName, "yes"), Ct));
    }

    public static TheoryData<string> UnsupportedRequests() =>
    [
        "empty tool message", "text in tool message", "function call in user message", "function result in user message",
        "image data", "image uri", "author name", "reasoning in user message", "hosted tool", "duplicate tool names",
        "tool mode required", "tool mode require specific", "single tool call only", "json format", "json schema format", "conversation id", "frequency penalty",
        "presence penalty", "reasoning options", "background responses", "continuation token", "raw representation factory",
        "other model id", "unknown additional property", "sampling without seed", "top-k without seed",
        "greedy temperature with top-p", "instructions and a system message",
    ];

    [Theory]
    [MemberData(nameof(UnsupportedRequests))]
    public async Task UnsupportedRequestsAreRejectedBeforeAnyGeneration(string name)
    {
        var backend = Backend();
        using var client = new Qwen35ChatClient(backend.Model);
        var (messages, options) = UnsupportedRequest(name);

        var ex = await Assert.ThrowsAsync<NotSupportedException>(() => client.GetResponseAsync(messages, options, Ct));
        Assert.False(string.IsNullOrWhiteSpace(ex.Message));
        await Assert.ThrowsAsync<NotSupportedException>(() => ToListAsync(client.GetStreamingResponseAsync(messages, options, Ct)));
        Assert.Empty(backend.Sequences);
    }

    [Fact]
    public async Task InvalidValuesAreRejected()
    {
        var backend = Backend();
        using var client = new Qwen35ChatClient(backend.Model);
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => client.GetResponseAsync(Conversation, new ChatOptions { MaxOutputTokens = 0 }, Ct));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => client.GetResponseAsync(Conversation, new ChatOptions { Temperature = -1 }, Ct));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => client.GetResponseAsync(Conversation, new ChatOptions { Temperature = 0.5f, Seed = -1 }, Ct));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => client.GetResponseAsync(Conversation, new ChatOptions { Temperature = 0.5f, Seed = (long)int.MaxValue + 1 }, Ct));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => client.GetResponseAsync(Conversation, new ChatOptions { TopP = 1.5f, Seed = 1 }, Ct));
        await Assert.ThrowsAsync<ArgumentException>(() => client.GetResponseAsync([new ChatMessage(ChatRole.User, "a"), null!], cancellationToken: Ct));
        await Assert.ThrowsAsync<ArgumentNullException>(() => client.GetResponseAsync((IEnumerable<ChatMessage>)null!, cancellationToken: Ct));

        // The template's own rules surface unchanged.
        await Assert.ThrowsAsync<Qwen35TemplateException>(() => client.GetResponseAsync([new ChatMessage(ChatRole.System, "only a system message")], cancellationToken: Ct));
        await Assert.ThrowsAsync<Qwen35TemplateException>(() => ToListAsync(client.GetStreamingResponseAsync([new ChatMessage(ChatRole.User, "a"), new ChatMessage(ChatRole.System, "late")], cancellationToken: Ct)));
    }

    /// <summary>
    /// Long context beyond the evidence is unsupported: prompt plus new tokens above
    /// <see cref="Qwen35ChatModel.MaxContextTokens"/> (8,192) is rejected before any native call, never truncated,
    /// although the model's own context length is 262,144. Exactly 8,192 is accepted.
    /// </summary>
    [Fact]
    public async Task SequencesOverTheAcceptedContextAreRejectedBeforeInference()
    {
        var backend = Backend();
        var model = backend.Model;
        var prompt = Qwen35Prompt.FromTokenIds(Enumerable.Repeat(1_000, 8_000).ToArray());
        var ex = Assert.Throws<GenerationContextLengthException>(() => model.Generate(prompt, new Qwen35ChatOptions { MaxNewTokens = 193 }, Ct));
        Assert.Equal((8_000, 193, Qwen35ChatModel.MaxContextTokens), (ex.PromptTokens, ex.MaxNewTokens, ex.ContextLength));
        Assert.Throws<GenerationContextLengthException>(() => model.GenerateStreaming(prompt, new Qwen35ChatOptions { MaxNewTokens = 193 }, Ct));
        Assert.Throws<GenerationContextLengthException>(() => model.Generate(prompt, cancellationToken: Ct)); // default 512 new tokens
        Assert.Empty(backend.Sequences);

        // 8,000 + 192 = 8,192 is accepted.
        var ok = model.Generate(prompt, new Qwen35ChatOptions { MaxNewTokens = 192 }, Ct);
        Assert.Equal(8_000, Assert.Single(backend.Sequences).Prompt.Length);
        Assert.NotEmpty(ok.TokenIds);

        using var client = new Qwen35ChatClient(model);
        string longText = string.Concat(Enumerable.Repeat("long context ", 9_000));
        await Assert.ThrowsAsync<GenerationContextLengthException>(() => client.GetResponseAsync([new ChatMessage(ChatRole.User, longText)], cancellationToken: Ct));
        Assert.Equal(8_192, Qwen35ChatModel.MaxContextTokens);
    }

    [Fact]
    public async Task AcceptedNoOpOptionsChangeNothing()
    {
        var backend = Backend();
        using var client = new Qwen35ChatClient(backend.Model);
        var plain = await client.GetResponseAsync(Conversation, cancellationToken: Ct);
        var noOps = await client.GetResponseAsync(Conversation, new ChatOptions
        {
            ModelId = client.ModelId,
            ResponseFormat = ChatResponseFormat.Text,
            Tools = [],
            ToolMode = ChatToolMode.Auto,
            AllowMultipleToolCalls = false,
            FrequencyPenalty = 0,
            PresencePenalty = 0,
            AllowBackgroundResponses = false,
            Seed = 5,
            AdditionalProperties = [],
        }, Ct);
        Assert.Equal(plain.Text, noOps.Text);
        Assert.All(backend.Sequences, s => Assert.Null(s.Options.Sampling));
    }

    [Fact]
    public async Task SamplingNeedsASeedAndMapsOntoTheSampler()
    {
        var backend = Backend();
        using var client = new Qwen35ChatClient(backend.Model);
        await client.GetResponseAsync(Conversation, new ChatOptions { Temperature = 0.7f, TopK = 20, TopP = 0.8f, Seed = 42 }, Ct);
        await client.GetResponseAsync(Conversation, new ChatOptions { TopK = 1, Seed = 3 }, Ct);
        await client.GetResponseAsync(Conversation, new ChatOptions { Temperature = 0 }, Ct);
        await client.GetResponseAsync(Conversation, new ChatOptions { Seed = 9 }, Ct);
        var sampling = backend.Sequences.Select(s => s.Options.Sampling).ToList();

        Assert.Equal(42, sampling[0]!.Seed);
        Assert.Equal(0.7, sampling[0]!.Temperature, 1e-6);
        Assert.Equal(20, sampling[0]!.TopK);
        Assert.Equal(0.8, sampling[0]!.TopP, 1e-6);
        Assert.Equal(new SamplingOptions { Seed = 3, TopK = 1 }, sampling[1]);
        Assert.Null(sampling[2]);
        Assert.Null(sampling[3]);

        // A seeded default samples; a request seed re-seeds it, Temperature = 0 turns it off.
        var seeded = new SamplingOptions { Seed = 7, Temperature = 0.5, TopK = 10 };
        using var sampler = new Qwen35ChatClient(backend.Model, new Qwen35ChatClientOptions { Defaults = new Qwen35ChatOptions { Sampling = seeded } });
        await sampler.GetResponseAsync(Conversation, cancellationToken: Ct);
        await sampler.GetResponseAsync(Conversation, new ChatOptions { Seed = 8 }, Ct);
        await sampler.GetResponseAsync(Conversation, new ChatOptions { TopP = 0.5f }, Ct);
        await sampler.GetResponseAsync(Conversation, new ChatOptions { Temperature = 0 }, Ct);
        sampling = [.. backend.Sequences.Skip(4).Select(s => s.Options.Sampling)];
        Assert.Equal(seeded, sampling[0]);
        Assert.Equal(seeded with { Seed = 8 }, sampling[1]);
        Assert.Equal(seeded with { TopP = 0.5 }, sampling[2]);
        Assert.Null(sampling[3]);
    }

    [Fact]
    public void EveryChatOptionsPropertyHasADeliberateMapping()
    {
        // A Microsoft.Extensions.AI upgrade that adds a request option must be mapped or rejected on purpose, not ignored.
        string[] handled =
        [
            "AdditionalProperties", "AllowBackgroundResponses", "AllowMultipleToolCalls", "ContinuationToken", "ConversationId",
            "FrequencyPenalty", "Instructions", "MaxOutputTokens", "ModelId", "PresencePenalty", "RawRepresentationFactory",
            "Reasoning", "ResponseFormat", "Seed", "StopSequences", "Temperature", "ToolMode", "Tools", "TopK", "TopP",
        ];
        var actual = typeof(ChatOptions).GetProperties(BindingFlags.Public | BindingFlags.Instance).Select(p => p.Name).Order(StringComparer.Ordinal);
        Assert.Equal(handled.Order(StringComparer.Ordinal), actual);
    }

    [Fact]
    public async Task RequestsRunOnADedicatedThreadAndNeverBlockTheCaller()
    {
        var backend = Backend();
        using var release = new ManualResetEventSlim();
        backend.OnStep = (_, step) =>
        {
            if (step == 1)
            {
                release.Wait(TimeSpan.FromSeconds(30));
            }
        };
        using var client = new Qwen35ChatClient(backend.Model);

        int caller = Environment.CurrentManagedThreadId;
        Task<ChatResponse> pending = client.GetResponseAsync(Conversation, cancellationToken: Ct);
        Assert.False(pending.IsCompleted);         // returned while the first decode step is still blocked
        await using var stream = client.GetStreamingResponseAsync(Conversation, cancellationToken: Ct).GetAsyncEnumerator(Ct);
        var next = stream.MoveNextAsync();
        Assert.False(next.IsCompleted);
        release.Set();
        await pending;
        Assert.True(await next);
        while (await stream.MoveNextAsync())
        {
        }

        Assert.All(backend.Sequences, s =>
        {
            // Creation, every chunk and step, and disposal on one thread, which is not the caller's or a pool thread.
            Assert.Single(s.Threads.Distinct());
            Assert.NotEqual(caller, s.Threads.First());
            Assert.Equal("Qwen35ChatClient request", s.ThreadName);
            Assert.True(s.Disposed);
        });
    }

    [Fact]
    public async Task ACancelledTokenStartsNoWork()
    {
        var backend = Backend();
        using var client = new Qwen35ChatClient(backend.Model);
        using var cts = new CancellationTokenSource();
        await cts.CancelAsync();
        var task = client.GetResponseAsync(Conversation, cancellationToken: cts.Token);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => task);
        Assert.True(task.IsCanceled);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => ToListAsync(client.GetStreamingResponseAsync(Conversation, cancellationToken: cts.Token)));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            // The enumerator's token (WithCancellation) is honored too.
#pragma warning disable xUnit1051
            await foreach (var u in client.GetStreamingResponseAsync(Conversation).WithCancellation(cts.Token))
#pragma warning restore xUnit1051
            {
            }
        });
        Assert.Empty(backend.Sequences);
    }

    [Fact]
    public async Task CancellationBetweenPrefillChunksStopsBeforeTheNextChunk()
    {
        var backend = Backend();
        using var cts = new CancellationTokenSource();
        backend.OnAppend = (_, chunk) =>
        {
            if (chunk == 2)
            {
                cts.Cancel();
            }
        };
        using var client = new Qwen35ChatClient(backend.Model, new Qwen35ChatClientOptions { Defaults = new Qwen35ChatOptions { PrefillChunkTokens = 4 } });
        var task = client.GetResponseAsync(Conversation, cancellationToken: cts.Token);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => task);
        Assert.True(task.IsCanceled);
        var s = backend.Single();
        Assert.Equal([4, 4], s.Chunks);
        Assert.Equal(0, s.Steps);
        Assert.True(s.Disposed);

        // The same through the stream, then the model serves the next request in full.
        using var cts2 = new CancellationTokenSource();
        backend.OnAppend = (_, chunk) =>
        {
            if (chunk == 3)
            {
                cts2.Cancel();
            }
        };
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => ToListAsync(client.GetStreamingResponseAsync(Conversation, cancellationToken: cts2.Token)));
        Assert.Equal([4, 4, 4], backend.Sequences.Last().Chunks);
        backend.OnAppend = null;
        Assert.Equal(Reply, (await client.GetResponseAsync(Conversation, cancellationToken: Ct)).Text);
    }

    [Fact]
    public async Task CancellationBetweenStepsEndsTheStreamAndReleasesTheSequence()
    {
        var backend = Backend();
        using var client = new Qwen35ChatClient(backend.Model, new Qwen35ChatClientOptions { StreamingBufferCapacity = 1 });
        using var cts = new CancellationTokenSource();
        int seen = 0;
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var u in client.GetStreamingResponseAsync(Conversation, cancellationToken: cts.Token))
            {
                if (++seen == 3)
                {
                    await cts.CancelAsync();
                }
            }
        });
        var s = backend.Single();
        Assert.True(s.Disposed);
        Assert.InRange(seen, 3, 3 + 2);                         // at most the buffer and the step in flight
        Assert.True(s.Steps < ReplyIds(backend.Tokenizer).Length);
        Assert.Equal(Reply, (await client.GetResponseAsync(Conversation, cancellationToken: Ct)).Text);
    }

    [Fact]
    public async Task StoppingTheEnumerationEarlyStopsTheWorkerBeforeDisposalReturns()
    {
        var backend = Backend();
        using var client = new Qwen35ChatClient(backend.Model, new Qwen35ChatClientOptions { StreamingBufferCapacity = 1 });
        await foreach (var u in client.GetStreamingResponseAsync(Conversation, cancellationToken: Ct))
        {
            break;
        }

        var s = backend.Single();
        Assert.True(s.Disposed);
        Assert.True(s.Steps <= 3, $"{s.Steps} steps after one update with a buffer of 1");
    }

    [Fact]
    public async Task ConcurrentRequestsEqualSequentialOnes()
    {
        var tokenizer = Tokenizer;
        // Each prompt gets its own scripted reply, so crossed requests would be visible.
        string[] questions = ["Capital of France?", "Capital of Japan?", "2 + 2?", "Say hi.", "Name a color.", "Name a fruit."];
        var backend = new ScriptedBackend(tokenizer, prompt => [.. ReplyIds(tokenizer, "Answer " + prompt.Aggregate(17L, (h, x) => unchecked((h * 31) + x)) + " 🙂."), Qwen35ChatModel.ImEndTokenId]);
        using var client = new Qwen35ChatClient(backend.Model);
        var sequential = new List<string>();
        foreach (string q in questions)
        {
            sequential.Add((await client.GetResponseAsync(q, cancellationToken: Ct)).Text);
        }

        Assert.Equal(questions.Length, sequential.Distinct().Count());
        var concurrent = await Task.WhenAll(questions.Select(q => client.GetResponseAsync(q, cancellationToken: Ct)));
        var streamed = await Task.WhenAll(questions.Select(async q => string.Concat((await ToListAsync(client.GetStreamingResponseAsync(q, cancellationToken: Ct))).Select(u => u.Text))));
        Assert.Equal(sequential, concurrent.Select(r => r.Text));
        Assert.Equal(sequential, streamed);
    }

    [Fact]
    public async Task MetadataAndServices()
    {
        var backend = Backend();
        using var client = new Qwen35ChatClient(backend.Model);
        IChatClient contract = client;
        var metadata = contract.GetService<ChatClientMetadata>()!;
        Assert.Equal(Qwen35ChatClient.ProviderName, metadata.ProviderName);
        Assert.Equal("starlings.onnxruntime-genai", metadata.ProviderName);
        Assert.Equal("Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17", metadata.DefaultModelId);
        Assert.Null(metadata.ProviderUri);
        Assert.Same(client, contract.GetService<Qwen35ChatClient>());
        Assert.Same(client, contract.GetService<IChatClient>());
        Assert.Same(backend.Model, contract.GetService<Qwen35ChatModel>());
        Assert.Same(backend.Model.Info, contract.GetService<Qwen35ModelInfo>());
        Assert.Null(contract.GetService<ChatClientMetadata>("key"));
        Assert.Null(contract.GetService<string>());
        await Task.CompletedTask;
    }

    [Fact]
    public async Task DisposalIsIdempotentAndAWrappedModelIsNotOwned()
    {
        var backend = Backend();
        var client = new Qwen35ChatClient(backend.Model);
        client.Dispose();
        client.Dispose();
        await Assert.ThrowsAsync<ObjectDisposedException>(() => client.GetResponseAsync(Conversation, cancellationToken: Ct));
        await Assert.ThrowsAsync<ObjectDisposedException>(() => ToListAsync(client.GetStreamingResponseAsync(Conversation, cancellationToken: Ct)));

        using var second = new Qwen35ChatClient(backend.Model);
        Assert.Equal(Reply, (await second.GetResponseAsync(Conversation, cancellationToken: Ct)).Text);
        Assert.Throws<ArgumentOutOfRangeException>(() => new Qwen35ChatClient(backend.Model, new Qwen35ChatClientOptions { StreamingBufferCapacity = 0 }));
        Assert.Throws<ArgumentNullException>(() => new Qwen35ChatClient(null!));
    }

    [Fact]
    public async Task WorksInsideAMiddlewarePipeline()
    {
        var backend = Backend();
        using var client = new Qwen35ChatClient(backend.Model);
        var logs = new ListLoggerProvider();
        using var loggerFactory = LoggerFactory.Create(b => b.AddProvider(logs).SetMinimumLevel(LogLevel.Trace));
        var cache = new DictionaryCache();
        using var pipeline = client.AsBuilder()
            .UseFunctionInvocation(loggerFactory)
            .UseLogging(loggerFactory)
            .UseDistributedCache(cache)
            .Build();

        var first = await pipeline.GetResponseAsync(Conversation, cancellationToken: Ct);
        var second = await pipeline.GetResponseAsync(Conversation, cancellationToken: Ct);
        Assert.Equal(Reply, first.Text);
        Assert.Equal(Reply, second.Text);
        Assert.Single(backend.Sequences);                 // the second call was served from the cache
        Assert.NotEmpty(cache.Entries);
        Assert.Equal(first.Usage!.OutputTokenCount, second.Usage!.OutputTokenCount);
        Assert.Equal(ChatFinishReason.Stop, second.FinishReason);
        Assert.Contains(logs.Messages, m => m.Contains("GetResponseAsync", StringComparison.Ordinal));

        var streamed = await ToListAsync(pipeline.GetStreamingResponseAsync([new ChatMessage(ChatRole.User, "Uncached?")], cancellationToken: Ct));
        Assert.Equal(Reply, string.Concat(streamed.Select(u => u.Text)));
        Assert.Equal(ChatFinishReason.Stop, streamed[^1].FinishReason);

        // Metadata and the underlying instances stay reachable through the pipeline.
        Assert.Equal(client.ModelId, pipeline.GetService<ChatClientMetadata>()!.DefaultModelId);
        Assert.Same(client, pipeline.GetService<Qwen35ChatClient>());

        // Function invocation middleware forwards tools; RequireAny is rejected instead of being ignored (the tool loop
        // itself is in ToolCallingTests).
        var tool = AIFunctionFactory.Create(() => "sunny", "get_weather");
        await Assert.ThrowsAsync<NotSupportedException>(() => pipeline.GetResponseAsync("Weather?", new ChatOptions { Tools = [tool], ToolMode = ChatToolMode.RequireAny }, Ct));
        await Assert.ThrowsAsync<NotSupportedException>(() => ToListAsync(pipeline.GetStreamingResponseAsync("Weather?", new ChatOptions { Tools = [tool], ToolMode = ChatToolMode.RequireAny }, Ct)));

        // Cancellation surfaces through the pipeline.
        using var cts = new CancellationTokenSource();
        await cts.CancelAsync();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => pipeline.GetResponseAsync("uncached", cancellationToken: cts.Token));
    }

    [Fact]
    public async Task DependencyInjectionSharesOneInstanceAndTheContainerDisposesIt()
    {
        var backend = Backend();
        var services = new ServiceCollection();
        services.AddQwen35ChatClient(_ => new Qwen35ChatClient(backend.Model));
        Qwen35ChatClient client;
        await using (var provider = services.BuildServiceProvider())
        {
            client = provider.GetRequiredService<Qwen35ChatClient>();
            Assert.Same(client, provider.GetRequiredService<IChatClient>());
            Assert.Equal(Reply, (await provider.GetRequiredService<IChatClient>().GetResponseAsync("di", cancellationToken: Ct)).Text);
        }

        await Assert.ThrowsAsync<ObjectDisposedException>(() => client.GetResponseAsync("after", cancellationToken: Ct));
    }

    [Fact]
    public async Task DependencyInjectionComposesWithTheExtensionsAiBuilder()
    {
        var backend = Backend();
        var services = new ServiceCollection();
        services.AddLogging();
        services.AddQwen35ChatClient(_ => new Qwen35ChatClient(backend.Model));
        services.AddChatClient(sp => sp.GetRequiredService<Qwen35ChatClient>()).UseLogging();
        await using var provider = services.BuildServiceProvider();

        var contract = provider.GetRequiredService<IChatClient>();
        Assert.IsNotType<Qwen35ChatClient>(contract);
        Assert.Same(provider.GetRequiredService<Qwen35ChatClient>(), contract.GetService<Qwen35ChatClient>());
        Assert.Equal(Reply, (await contract.GetResponseAsync("di", cancellationToken: Ct)).Text);
    }

    [Fact]
    public void CapabilitiesReportTheAdapter()
    {
        var all = Qwen35Capabilities.All;
        Assert.Equal(Qwen35CapabilityStatus.Experimental, all["extensions_ai.chat_client"]);
        Assert.Equal(Qwen35CapabilityStatus.Experimental, all["extensions_ai.streaming"]);
        Assert.Equal(Qwen35CapabilityStatus.Experimental, all["extensions_ai.cancellation"]);
        Assert.Equal(Qwen35CapabilityStatus.Experimental, all["extensions_ai.tools"]);
        Assert.Equal(Qwen35CapabilityStatus.Experimental, all["extensions_ai.function_invocation"]);
        Assert.Equal(Qwen35CapabilityStatus.Unsupported, all["extensions_ai.tool_mode.required"]);
        Assert.Equal(Qwen35CapabilityStatus.Unsupported, all["extensions_ai.structured_output"]);
        Assert.Equal(Qwen35CapabilityStatus.Unsupported, all["extensions_ai.multimodal"]);
        Assert.Equal(Qwen35CapabilityStatus.Unsupported, all["extensions_ai.unseeded_sampling"]);
        Assert.False(all.ContainsKey("extensions_ai"));
    }

    private static (ChatMessage[] Messages, ChatOptions? Options) UnsupportedRequest(string name)
    {
        ChatMessage[] user = [new(ChatRole.User, "Hi")];
        return name switch
        {
            "empty tool message" => ([.. user, new ChatMessage(ChatRole.Tool, [])], null),
            "text in tool message" => ([.. user, new ChatMessage(ChatRole.Tool, "42")], null),
            "function call in user message" => ([.. user, new ChatMessage(ChatRole.User, [new FunctionCallContent("c1", "f")])], null),
            "function result in user message" => ([.. user, new ChatMessage(ChatRole.User, [new FunctionResultContent("c1", "r")])], null),
            "hosted tool" => (user, new ChatOptions { Tools = [new HostedWebSearchTool()] }),
            "duplicate tool names" => (user, new ChatOptions { Tools = [AIFunctionFactory.Create(() => 1, "f"), AIFunctionFactory.Create(() => 2, "f")] }),
            "tool mode require specific" => (user, new ChatOptions { Tools = [AIFunctionFactory.Create(() => 1, "f")], ToolMode = ChatToolMode.RequireSpecific("f") }),
            "single tool call only" => (user, new ChatOptions { Tools = [AIFunctionFactory.Create(() => 1, "f")], AllowMultipleToolCalls = false }),
            "image data" => ([new ChatMessage(ChatRole.User, [new TextContent("What is this?"), new DataContent(new byte[] { 1, 2, 3 }, "image/png")])], null),
            "image uri" => ([new ChatMessage(ChatRole.User, [new UriContent(new Uri("https://example.com/a.png"), "image/png")])], null),
            "author name" => ([new ChatMessage(ChatRole.User, "Hi") { AuthorName = "mark" }], null),
            "reasoning in user message" => ([new ChatMessage(ChatRole.User, [new TextReasoningContent("hmm")])], null),
            "tool mode required" => (user, new ChatOptions { ToolMode = ChatToolMode.RequireAny }),
            "json format" => (user, new ChatOptions { ResponseFormat = ChatResponseFormat.Json }),
            "json schema format" => (user, new ChatOptions { ResponseFormat = ChatResponseFormat.ForJsonSchema(JsonDocument.Parse("""{"type":"object"}""").RootElement) }),
            "conversation id" => (user, new ChatOptions { ConversationId = "c" }),
            "frequency penalty" => (user, new ChatOptions { FrequencyPenalty = 0.5f }),
            "presence penalty" => (user, new ChatOptions { PresencePenalty = 0.5f }),
            "reasoning options" => (user, new ChatOptions { Reasoning = new ReasoningOptions() }),
            "background responses" => (user, new ChatOptions { AllowBackgroundResponses = true }),
            "continuation token" => (user, new ChatOptions { ContinuationToken = ResponseContinuationToken.FromBytes(new byte[] { 1 }) }),
            "raw representation factory" => (user, new ChatOptions { RawRepresentationFactory = _ => new object() }),
            "other model id" => (user, new ChatOptions { ModelId = "Qwen/Qwen3.5-0.8B" }),
            "unknown additional property" => (user, Props("enable_thinkin", true)),
            "sampling without seed" => (user, new ChatOptions { Temperature = 0.7f }),
            "top-k without seed" => (user, new ChatOptions { TopK = 5 }),
            "greedy temperature with top-p" => (user, new ChatOptions { Temperature = 0, TopP = 0.9f, Seed = 1 }),
            "instructions and a system message" => ([new ChatMessage(ChatRole.System, "s"), .. user], new ChatOptions { Instructions = "i" }),
            _ => throw new ArgumentOutOfRangeException(nameof(name), name, null),
        };
    }

    private static ChatOptions Props(string key, object? value) => new() { AdditionalProperties = new AdditionalPropertiesDictionary { [key] = value } };

    private static async Task<List<ChatResponseUpdate>> ToListAsync(IAsyncEnumerable<ChatResponseUpdate> updates)
    {
        var list = new List<ChatResponseUpdate>();
        await foreach (var u in updates)
        {
            list.Add(u);
        }

        return list;
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
