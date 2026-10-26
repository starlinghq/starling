using System.Text;
using System.Text.Json;
using Starlings.Generation.OnnxGenAI;
using Starlings.TestSupport;
using Starlings.Tokenizers;
using Microsoft.Extensions.AI;

namespace Starlings.Models.Qwen35.Extensions.AI.Tests;

/// <summary>
/// The opt-in reasoning split (<see cref="Qwen35ChatClientOptions.SplitReasoning"/>) and per-token log probabilities
/// (<see cref="Qwen35ChatClient.LogprobsPropertyName"/>) through the IChatClient, with a scripted native sequence (the
/// real template, tokenizer, decode loop and detokenizer; only the model's token choice is scripted).
/// </summary>
public sealed class ReasoningAndLogprobsTests
{
    private const string Reasoning = "The user asks for the capital of France. That is Paris, 你好 🙂.\n";
    private const string Answer = "\n\nThe capital of France is Paris.";
    private const string Thought = Reasoning + "</think>" + Answer;
    private const string Call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";

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

    private static ScriptedBackend Backend(string completion, bool stop = true)
    {
        var tokenizer = Tokenizer;
        int[] ids = [.. tokenizer.Encode(completion, addSpecialTokens: false).Ids];
        return new ScriptedBackend(tokenizer, _ => stop ? [.. ids, Qwen35ChatModel.ImEndTokenId] : ids);
    }

    private static ChatMessage[] Question => [new(ChatRole.User, "What is the capital of France?")];

    private static ChatOptions Thinking(ChatOptions? options = null)
    {
        options ??= new ChatOptions();
        (options.AdditionalProperties ??= [])[Qwen35ChatClient.EnableThinkingPropertyName] = true;
        return options;
    }

    private static Qwen35ChatClientOptions Split => new() { SplitReasoning = true };

    private static async Task<List<ChatResponseUpdate>> ToListAsync(IAsyncEnumerable<ChatResponseUpdate> updates)
    {
        var list = new List<ChatResponseUpdate>();
        await foreach (var update in updates)
        {
            list.Add(update);
        }

        return list;
    }

    private static string Concat<T>(IEnumerable<ChatResponseUpdate> updates, Func<T, string> text)
        where T : AIContent =>
        string.Concat(updates.SelectMany(u => u.Contents).OfType<T>().Select(text));

    [Fact]
    public async Task ByDefaultTheThinkBlockStaysOneTextContent()
    {
        using var client = new Qwen35ChatClient(Backend(Thought).Model);
        var response = await client.GetResponseAsync(Question, Thinking(), Ct);
        var content = Assert.Single(Assert.Single(response.Messages).Contents);
        Assert.Equal(Thought, Assert.IsType<TextContent>(content).Text);
    }

    [Fact]
    public async Task SplitReturnsReasoningThenAnswerWithoutTheTag()
    {
        using var client = new Qwen35ChatClient(Backend(Thought).Model, Split);
        var response = await client.GetResponseAsync(Question, Thinking(), Ct);
        var contents = Assert.Single(response.Messages).Contents;
        Assert.Equal(2, contents.Count);
        Assert.Equal(Reasoning, Assert.IsType<TextReasoningContent>(contents[0]).Text);
        Assert.Equal(Answer, Assert.IsType<TextContent>(contents[1]).Text);
        Assert.Equal(Answer, response.Text);
        Assert.Equal(ChatFinishReason.Stop, response.FinishReason);
    }

    [Fact]
    public async Task StreamedSplitHoldsBackOnlyAPossibleCloseTagAndReassembles()
    {
        using var client = new Qwen35ChatClient(Backend(Thought).Model, Split);
        var updates = await ToListAsync(client.GetStreamingResponseAsync(Question, Thinking(), Ct));

        Assert.Equal(Reasoning, Concat<TextReasoningContent>(updates, r => r.Text));
        Assert.Equal(Answer, Concat<TextContent>(updates, t => t.Text));
        Assert.All(updates.SelectMany(u => u.Contents).OfType<TextContent>(), t => Assert.DoesNotContain("</", t.Text, StringComparison.Ordinal));
        Assert.All(updates.SelectMany(u => u.Contents).OfType<TextReasoningContent>(), r => Assert.DoesNotContain("</", r.Text, StringComparison.Ordinal));

        // Reasoning streams before the tag resolves: several reasoning updates precede the first answer update.
        int firstAnswer = updates.FindIndex(u => u.Contents.OfType<TextContent>().Any());
        Assert.True(updates.Take(firstAnswer).Count(u => u.Contents.OfType<TextReasoningContent>().Any()) > 1);
        Assert.Equal(ChatFinishReason.Stop, updates[^1].FinishReason);

        var response = updates.ToChatResponse();
        Assert.Equal(Answer, response.Text);
    }

    [Fact]
    public async Task ACompletionThatEndsBeforeTheTagIsReasoningOnly()
    {
        using var client = new Qwen35ChatClient(Backend("Still thinking about it </thi", stop: false).Model, Split);
        var options = Thinking(new ChatOptions { MaxOutputTokens = 6 });
        var response = await client.GetResponseAsync(Question, options, Ct);
        Assert.Equal(ChatFinishReason.Length, response.FinishReason);
        var content = Assert.Single(Assert.Single(response.Messages).Contents);
        string reasoning = Assert.IsType<TextReasoningContent>(content).Text;

        var updates = await ToListAsync(client.GetStreamingResponseAsync(Question, options, Ct));
        Assert.Equal(reasoning, Concat<TextReasoningContent>(updates, r => r.Text));
        Assert.Empty(updates.SelectMany(u => u.Contents).OfType<TextContent>());
    }

    [Fact]
    public async Task WithoutThinkingTheSplitChangesNothing()
    {
        using var client = new Qwen35ChatClient(Backend("Paris.").Model, Split);
        var response = await client.GetResponseAsync(Question, cancellationToken: Ct);
        Assert.Equal("Paris.", Assert.IsType<TextContent>(Assert.Single(Assert.Single(response.Messages).Contents)).Text);
    }

    [Fact]
    public async Task SplitWithToolsReturnsReasoningAndTheCall()
    {
        var backend = Backend("The user wants the weather.\n</think>\n\n" + Call);
        using var client = new Qwen35ChatClient(backend.Model, Split);
        var function = AIFunctionFactory.Create((string city) => city, "get_weather", "Current weather for a city.");
        var response = await client.GetResponseAsync(Question, Thinking(new ChatOptions { Tools = [function] }), Ct);
        Assert.Equal(ChatFinishReason.ToolCalls, response.FinishReason);
        var contents = Assert.Single(response.Messages).Contents;
        Assert.Equal("The user wants the weather.\n", Assert.IsType<TextReasoningContent>(contents[0]).Text);
        var call = Assert.IsType<FunctionCallContent>(Assert.Single(contents.Skip(1)));
        Assert.Equal("get_weather", call.Name);

        var updates = await ToListAsync(client.GetStreamingResponseAsync(Question, Thinking(new ChatOptions { Tools = [function] }), Ct));
        Assert.Equal("The user wants the weather.\n", Concat<TextReasoningContent>(updates, r => r.Text));
        Assert.Single(updates.SelectMany(u => u.Contents).OfType<FunctionCallContent>());
        Assert.Empty(updates.SelectMany(u => u.Contents).OfType<TextContent>());
        Assert.Equal(ChatFinishReason.ToolCalls, updates[^1].FinishReason);
    }

    [Fact]
    public void AnySplitIntoDeltasGivesTheWholeCompletionSplit()
    {
        foreach (string completion in new[] { Thought, "a</think>b", "</think>", "no tag at all", "<</think></think>x", "ends with </thin" })
        {
            var expected = Qwen35ReasoningSplitter.Split(completion);
            for (int i = 0; i <= completion.Length; i++)
            {
                for (int j = i; j <= completion.Length; j++)
                {
                    var splitter = new Qwen35ReasoningSplitter();
                    var reasoning = new StringBuilder();
                    var answer = new StringBuilder();
                    foreach (string delta in new[] { completion[..i], completion[i..j], completion[j..] })
                    {
                        var (r, a) = splitter.Append(delta);
                        reasoning.Append(r);
                        answer.Append(a);
                    }

                    reasoning.Append(splitter.Complete());
                    Assert.Equal(expected.Reasoning, reasoning.ToString());
                    Assert.Equal(expected.Answer, answer.ToString());
                }
            }
        }
    }

    [Fact]
    public async Task LogprobsAreOptInAndCoverEveryCompletionToken()
    {
        var backend = Backend("Hello! The capital of France is Paris, 你好.");
        using var client = new Qwen35ChatClient(backend.Model);

        var plain = await client.GetResponseAsync(Question, cancellationToken: Ct);
        Assert.Null(plain.AdditionalProperties);

        var options = new ChatOptions { AdditionalProperties = new() { [Qwen35ChatClient.LogprobsPropertyName] = true, [Qwen35ChatClient.TopLogprobsPropertyName] = 2 } };
        var response = await client.GetResponseAsync(Question, options, Ct);
        var result = Assert.IsType<GenerationResult>(response.RawRepresentation);
        var logprobs = Assert.IsAssignableFrom<IReadOnlyList<TokenLogprob>>(response.AdditionalProperties![Qwen35ChatClient.LogprobsPropertyName]);
        Assert.Equal(result.TokenIds, logprobs.Select(l => l.TokenId));
        Assert.Equal(response.Usage!.OutputTokenCount, logprobs.Count);
        Assert.All(logprobs, l => Assert.Equal(2, l.TopLogprobs.Length));

        // Streaming: each update carries the entries of its tokens; concatenated they equal the response's list.
        var updates = await ToListAsync(client.GetStreamingResponseAsync(Question, options, Ct));
        var streamed = updates
            .Where(u => u.AdditionalProperties is not null)
            .SelectMany(u => (IReadOnlyList<TokenLogprob>)u.AdditionalProperties![Qwen35ChatClient.LogprobsPropertyName]!)
            .ToList();
        Assert.Equal(logprobs, streamed);
        Assert.All(updates, u => Assert.Equal(
            ((GenerationUpdate)u.RawRepresentation!).TokenIds,
            u.AdditionalProperties is null ? [] : ((IReadOnlyList<TokenLogprob>)u.AdditionalProperties[Qwen35ChatClient.LogprobsPropertyName]!).Select(l => l.TokenId)));
    }

    [Fact]
    public async Task LogprobsOfHeldBackTextMoveToTheUpdateThatCarriesIt()
    {
        var backend = Backend("Let me check.\n</think>\n\n" + Call);
        using var client = new Qwen35ChatClient(backend.Model, Split);
        var function = AIFunctionFactory.Create((string city) => city, "get_weather", "Current weather for a city.");
        var options = Thinking(new ChatOptions { Tools = [function], AdditionalProperties = new() { [Qwen35ChatClient.LogprobsPropertyName] = JsonDocument.Parse("true").RootElement } });
        var updates = await ToListAsync(client.GetStreamingResponseAsync(Question, options, Ct));
        var result = ((GenerationUpdate)updates[^1].RawRepresentation!).Result!;
        var streamed = updates
            .Where(u => u.AdditionalProperties is not null)
            .SelectMany(u => (IReadOnlyList<TokenLogprob>)u.AdditionalProperties![Qwen35ChatClient.LogprobsPropertyName]!)
            .Select(l => l.TokenId);
        Assert.Equal(result.TokenIds, streamed);
        Assert.Equal(result.TokenIds.Length, result.Logprobs.Length);
    }

    [Fact]
    public async Task InvalidLogprobRequestsAreRejected()
    {
        using var client = new Qwen35ChatClient(Backend("Hi.").Model);
        ChatOptions With(string key, object? value) => new() { AdditionalProperties = new() { [key] = value } };

        await Assert.ThrowsAsync<ArgumentException>(() => client.GetResponseAsync(Question, With(Qwen35ChatClient.TopLogprobsPropertyName, 3), Ct));
        await Assert.ThrowsAsync<ArgumentException>(() => client.GetResponseAsync(Question, With(Qwen35ChatClient.LogprobsPropertyName, "yes"), Ct));
        var tooMany = new ChatOptions { AdditionalProperties = new() { [Qwen35ChatClient.LogprobsPropertyName] = true, [Qwen35ChatClient.TopLogprobsPropertyName] = 21 } };
        await Assert.ThrowsAsync<ArgumentException>(() => client.GetResponseAsync(Question, tooMany, Ct));
        await Assert.ThrowsAsync<NotSupportedException>(() => client.GetResponseAsync(Question, With("logit_bias", 1), Ct));

        // Turning them off per request overrides a default that turns them on.
        using var defaults = new Qwen35ChatClient(Backend("Hi.").Model, new Qwen35ChatClientOptions { Defaults = new Qwen35ChatOptions { ReturnLogprobs = true } });
        Assert.NotNull((await defaults.GetResponseAsync(Question, cancellationToken: Ct)).AdditionalProperties);
        Assert.Null((await defaults.GetResponseAsync(Question, With(Qwen35ChatClient.LogprobsPropertyName, false), Ct)).AdditionalProperties);
    }
}
