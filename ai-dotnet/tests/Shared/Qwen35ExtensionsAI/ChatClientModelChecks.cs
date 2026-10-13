using System.Diagnostics;
using Starlings.Models.Qwen35;
using Starlings.Models.Qwen35.Extensions.AI;
using Microsoft.Extensions.AI;
using Xunit;

namespace Starlings.TestSupport;

/// <summary>IChatClient checks on the real model, shared by the CPU and CUDA test projects (P-017).</summary>
internal static class ChatClientModelChecks
{
    public static ChatMessage[] ToChatMessages(GenerationCase c) =>
    [
        .. c.Messages!.Select(m => new ChatMessage(
            m.Role switch
            {
                Qwen35Role.System => ChatRole.System,
                Qwen35Role.User => ChatRole.User,
                _ => ChatRole.Assistant,
            },
            m.Content)),
    ];

    /// <summary>The text prompts of the fixture (the tools prompt cannot be expressed: tools are unsupported).</summary>
    public static IReadOnlyList<GenerationCase> TextCases() => [.. Qwen35TestAssets.LoadCases().Where(c => !c.HasTools)];

    /// <summary>
    /// Through IChatClient from messages: the non-streamed text equals the reference completion (first
    /// <paramref name="maxOutputTokens"/> tokens), the streamed updates concatenate to it, and finish reason and usage match
    /// in both modes (input tokens are the reference prompt ids; output tokens exclude the stop token).
    /// </summary>
    public static async Task StreamingEqualsResponseAndReference(IChatClient client, Qwen35ChatModel model, GenerationCase c, int maxOutputTokens, CancellationToken ct)
    {
        var messages = ToChatMessages(c);
        var options = new ChatOptions { MaxOutputTokens = maxOutputTokens };
        if (c.EnableThinking)
        {
            options.AdditionalProperties = new AdditionalPropertiesDictionary { [Qwen35ChatClient.EnableThinkingPropertyName] = true };
        }

        ChatResponse response = await client.GetResponseAsync(messages, options, ct);
        var updates = new List<ChatResponseUpdate>();
        await foreach (var u in client.GetStreamingResponseAsync(messages, options, ct))
        {
            updates.Add(u);
        }

        int[] expectedIds = [.. c.ContentIds.Take(maxOutputTokens)];
        bool stops = c.ContentIds.Length <= maxOutputTokens && c.FinishReason == "eos";
        Assert.Equal(model.Tokenizer.Decode(expectedIds, false), response.Text);
        Assert.Equal(stops ? ChatFinishReason.Stop : ChatFinishReason.Length, response.FinishReason);
        Assert.Equal(c.PromptIds.Length, response.Usage!.InputTokenCount);
        Assert.Equal(expectedIds.Length, response.Usage.OutputTokenCount);

        Assert.Equal(response.Text, string.Concat(updates.Select(u => u.Text)));
        Assert.All(updates.SkipLast(1), u => Assert.NotEmpty(u.Text));
        Assert.Equal(response.FinishReason, updates[^1].FinishReason);
        var usage = Assert.Single(updates[^1].Contents.OfType<UsageContent>()).Details;
        Assert.Equal(response.Usage.InputTokenCount, usage.InputTokenCount);
        Assert.Equal(response.Usage.OutputTokenCount, usage.OutputTokenCount);
        Assert.Equal(response.Usage.TotalTokenCount, usage.TotalTokenCount);
    }

    /// <summary>
    /// Cancels a streaming request after <paramref name="afterUpdates"/> updates and returns the time from
    /// <c>Cancel()</c> to the <see cref="OperationCanceledException"/>; then the same client answers correctly.
    /// </summary>
    public static async Task<TimeSpan> CancelBetweenStepsThenReuse(IChatClient client, Qwen35ChatModel model, GenerationCase longCase, GenerationCase shortCase, int afterUpdates, CancellationToken ct)
    {
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        var clock = new Stopwatch();
        int seen = 0;
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var u in client.GetStreamingResponseAsync(ToChatMessages(longCase), new ChatOptions { MaxOutputTokens = 64 }, cts.Token))
            {
                if (++seen == afterUpdates)
                {
                    clock.Start();
                    await cts.CancelAsync();
                }
            }
        });
        clock.Stop();
        Assert.InRange(seen, afterUpdates, afterUpdates + Qwen35ChatClientOptions.DefaultStreamingBufferCapacity + 1);
        await StreamingEqualsResponseAndReference(client, model, shortCase, 16, ct);
        return clock.Elapsed;
    }

    /// <summary>
    /// Cancels a request <paramref name="after"/> into the prefill of a long prompt (chunked by the client's defaults)
    /// and returns the time from <c>Cancel()</c> to the <see cref="OperationCanceledException"/> and the whole call's
    /// duration; then the same client answers correctly.
    /// </summary>
    public static async Task<(TimeSpan Latency, TimeSpan Total)> CancelDuringPrefillThenReuse(IChatClient client, Qwen35ChatModel model, GenerationCase longCase, GenerationCase shortCase, TimeSpan after, CancellationToken ct)
    {
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        var total = Stopwatch.StartNew();
        var task = client.GetResponseAsync(ToChatMessages(longCase), new ChatOptions { MaxOutputTokens = 8 }, cts.Token);
        await Task.Delay(after, ct);
        Assert.False(task.IsCompleted, "the prefill finished before the cancellation");
        var clock = Stopwatch.StartNew();
        await cts.CancelAsync();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => task);
        clock.Stop();
        total.Stop();
        Assert.True(task.IsCanceled);
        await StreamingEqualsResponseAndReference(client, model, shortCase, 16, ct);
        return (clock.Elapsed, total.Elapsed);
    }

    /// <summary>Concurrent GetResponseAsync calls each equal the sequential (reference) result, over several rounds.</summary>
    public static async Task ConcurrentResponsesEqualSequential(IChatClient client, Qwen35ChatModel model, IReadOnlyList<GenerationCase> cases, int maxOutputTokens, int rounds, CancellationToken ct)
    {
        var options = new ChatOptions { MaxOutputTokens = maxOutputTokens };
        var sequential = new List<string>();
        foreach (var c in cases)
        {
            sequential.Add((await client.GetResponseAsync(ToChatMessages(c), options, ct)).Text);
        }

        Assert.Equal(cases.Select(c => model.Tokenizer.Decode([.. c.ContentIds.Take(maxOutputTokens)], false)), sequential);
        for (int round = 0; round < rounds; round++)
        {
            var concurrent = await Task.WhenAll(cases.Select(c => client.GetResponseAsync(ToChatMessages(c), options, ct)));
            Assert.Equal(sequential, concurrent.Select(r => r.Text));
        }
    }
}
