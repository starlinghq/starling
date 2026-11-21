using System.Runtime.CompilerServices;
using System.Text.Json;
using Microsoft.Extensions.AI;

namespace Starlings.Guardrails;

/// <summary>What the middleware does with a flagged text.</summary>
public enum GuardAction
{
    /// <summary>Let it through and report the finding in the response's <see cref="GuardrailsChatClient.ReportKey"/> property.</summary>
    Annotate,

    /// <summary>Throw <see cref="GuardrailsViolationException"/>. A flagged input never reaches the inner client.</summary>
    Block,
}

/// <summary>How a streaming response is checked.</summary>
public enum StreamingOutputCheck
{
    /// <summary>Hold every update, check the whole response, then release the updates (or block). Nothing unchecked reaches the caller; the first token arrives only at the end.</summary>
    Buffer,

    /// <summary>Pass updates through as they arrive and check the whole response at the end. Blocking then throws after the content was already streamed.</summary>
    AfterCompletion,
}

/// <summary>Options of <see cref="GuardrailsChatClient"/>.</summary>
public sealed record GuardrailsOptions
{
    /// <summary>Check new user messages (those after the last assistant message) before the call. Default true.</summary>
    public bool CheckUserMessages { get; init; } = true;

    /// <summary>Check new tool results (<see cref="FunctionResultContent"/>) before the call, the usual route of indirect injection. Default true.</summary>
    public bool CheckToolResults { get; init; } = true;

    /// <summary>Check the response. Default true.</summary>
    public bool CheckResponse { get; init; } = true;

    /// <summary>What to do with a flagged text. Default <see cref="GuardAction.Block"/>.</summary>
    public GuardAction OnFlagged { get; init; } = GuardAction.Block;

    /// <summary>
    /// Block a text the guard did not fully check: not scanned at all (for example over the model's length limit), or
    /// scanned with <see cref="GuardResult.Gaps"/> (a cascade's model tier skipped it, or the model declined a question).
    /// Default false: it passes and the report records it as not fully scanned.
    /// </summary>
    public bool BlockWhenNotScanned { get; init; }

    /// <summary>How streaming responses are checked. Default <see cref="StreamingOutputCheck.Buffer"/>.</summary>
    public StreamingOutputCheck StreamingOutput { get; init; } = StreamingOutputCheck.Buffer;

    /// <summary>
    /// Also check the last this-many user messages joined (earlier turns included), as one more input text with source
    /// <see cref="GuardSource.Conversation"/> and message index <see cref="GuardrailsChatClient.ConversationIndex"/>, to
    /// catch an attack split across messages. Runs only when the window holds two or more user messages; 1 or less turns
    /// it off. Default 4. Costs one more guard check per call (negligible for patterns, one more model request for a
    /// model guard). With <see cref="GuardAction.Annotate"/>, a flagged earlier message keeps flagging the joined text
    /// while it is in the window.
    /// </summary>
    public int ConversationWindow { get; init; } = 4;
}

/// <summary>One checked text and its result.</summary>
/// <param name="Stage">Input or output.</param>
/// <param name="Source">User message, tool result, or response.</param>
/// <param name="MessageIndex">Index of the message in the request; -1 for the response, -2 for the joined conversation window.</param>
/// <param name="Result">The guard's result.</param>
public sealed record GuardFinding(GuardStage Stage, GuardSource Source, int MessageIndex, GuardResult Result);

/// <summary>Every check the middleware ran for one call.</summary>
/// <param name="Findings">The checks, inputs first, then the response.</param>
public sealed record GuardrailsReport(IReadOnlyList<GuardFinding> Findings)
{
    /// <summary>True when any check flagged its text.</summary>
    public bool Flagged => Findings.Any(f => f.Result.Flagged);

    /// <summary>True when any text was not scanned, or scanned with <see cref="GuardResult.Gaps"/>.</summary>
    public bool HasUnscanned => Findings.Any(f => !f.Result.Complete);

    /// <summary>The highest score over all checks.</summary>
    public double Score => Findings.Count == 0 ? 0 : Findings.Max(f => f.Result.Score);
}

/// <summary>A guard blocked a call. <see cref="Report"/> holds every check so far; <see cref="Finding"/> the blocking one.</summary>
public sealed class GuardrailsViolationException : Exception
{
    /// <summary>Creates the exception.</summary>
    public GuardrailsViolationException()
        : this("A guardrail blocked the call.")
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public GuardrailsViolationException(string message)
        : base(message)
    {
        Report = new([]);
    }

    /// <summary>Creates the exception with a message and the underlying cause.</summary>
    public GuardrailsViolationException(string message, Exception innerException)
        : base(message, innerException)
    {
        Report = new([]);
    }

    /// <summary>Creates the exception for a blocking finding.</summary>
    public GuardrailsViolationException(GuardFinding finding, GuardrailsReport report)
        : base(Describe(finding))
    {
        ArgumentNullException.ThrowIfNull(report);
        Finding = finding;
        Report = report;
    }

    /// <summary>The finding that blocked the call; null for the message-only constructors.</summary>
    public GuardFinding? Finding { get; }

    /// <summary>Every check run before the block.</summary>
    public GuardrailsReport Report { get; }

    private static string Describe(GuardFinding finding)
    {
        ArgumentNullException.ThrowIfNull(finding);
        var where = finding.MessageIndex switch
        {
            GuardrailsChatClient.ConversationIndex => "the recent user messages joined",
            < 0 => "the response",
            _ => $"message {finding.MessageIndex} ({finding.Source})",
        };
        var result = finding.Result;
        if (result.Status == GuardStatus.NotScanned)
        {
            return $"A guardrail blocked {where}: not scanned ({result.Reason}).";
        }

        return result.Flagged
            ? $"A guardrail blocked {where}: {string.Join(", ", result.FlaggedCategories.Select(c => $"{c.Category} {c.Score:F2}"))}."
            : $"A guardrail blocked {where}: not fully scanned ({string.Join("; ", result.Gaps)}).";
    }
}

/// <summary>
/// Middleware that checks the new inputs of each call before passing it on, and the response after. Results land in
/// <see cref="ChatResponse.AdditionalProperties"/> under <see cref="ReportKey"/> as a <see cref="GuardrailsReport"/>
/// (for streaming, on one extra final update), or a <see cref="GuardrailsViolationException"/> is thrown. Experimental.
/// </summary>
public sealed class GuardrailsChatClient : DelegatingChatClient
{
    /// <summary>The <see cref="AdditionalPropertiesDictionary"/> key of the <see cref="GuardrailsReport"/>.</summary>
    public const string ReportKey = "starlings.guardrails";

    /// <summary>The <see cref="GuardFinding.MessageIndex"/> of the joined conversation window (<see cref="GuardrailsOptions.ConversationWindow"/>).</summary>
    public const int ConversationIndex = -2;

    private readonly IContentGuard _guard;
    private readonly GuardrailsOptions _options;

    /// <summary>Wraps a client.</summary>
    public GuardrailsChatClient(IChatClient innerClient, IContentGuard guard, GuardrailsOptions? options = null)
        : base(innerClient)
    {
        ArgumentNullException.ThrowIfNull(guard);
        _guard = guard;
        _options = options ?? new GuardrailsOptions();
    }

    /// <summary>The report of a response, or null when it has none.</summary>
    public static GuardrailsReport? GetReport(ChatResponse response)
    {
        ArgumentNullException.ThrowIfNull(response);
        return response.AdditionalProperties?.TryGetValue(ReportKey, out var value) == true ? value as GuardrailsReport : null;
    }

    /// <inheritdoc />
    public override async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var list = messages as IReadOnlyList<ChatMessage> ?? [.. messages];
        var findings = new List<GuardFinding>();
        await CheckInputsAsync(list, findings, cancellationToken).ConfigureAwait(false);
        var response = await base.GetResponseAsync(list, options, cancellationToken).ConfigureAwait(false);
        if (_options.CheckResponse)
        {
            await CheckAsync(GuardRequest.Response(response.Text), -1, findings, cancellationToken).ConfigureAwait(false);
        }

        (response.AdditionalProperties ??= [])[ReportKey] = new GuardrailsReport(findings);
        return response;
    }

    /// <inheritdoc />
    public override async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages, ChatOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var list = messages as IReadOnlyList<ChatMessage> ?? [.. messages];
        var findings = new List<GuardFinding>();
        await CheckInputsAsync(list, findings, cancellationToken).ConfigureAwait(false);

        // The transcript is kept only when the response is checked; otherwise only the last update (for the ids).
        var updates = new List<ChatResponseUpdate>();
        ChatResponseUpdate? last = null;
        bool buffer = _options.CheckResponse && _options.StreamingOutput == StreamingOutputCheck.Buffer;
        await foreach (var update in base.GetStreamingResponseAsync(list, options, cancellationToken).ConfigureAwait(false))
        {
            last = update;
            if (_options.CheckResponse)
            {
                updates.Add(update);
            }

            if (!buffer)
            {
                yield return update;
            }
        }

        if (_options.CheckResponse)
        {
            await CheckAsync(GuardRequest.Response(updates.ToChatResponse().Text), -1, findings, cancellationToken).ConfigureAwait(false);
        }

        if (buffer)
        {
            foreach (var update in updates)
            {
                yield return update;
            }
        }

        // No MessageId: ToChatResponse() then keeps the report on the response, not on the last message.
        yield return new ChatResponseUpdate
        {
            Role = ChatRole.Assistant,
            ResponseId = last?.ResponseId,
            ConversationId = last?.ConversationId,
            AdditionalProperties = new() { [ReportKey] = new GuardrailsReport(findings) },
        };
    }

    private async Task CheckInputsAsync(IReadOnlyList<ChatMessage> messages, List<GuardFinding> findings, CancellationToken cancellationToken)
    {
        int start = 0;
        for (int i = messages.Count - 1; i >= 0; i--)
        {
            if (messages[i].Role == ChatRole.Assistant)
            {
                start = i + 1;
                break;
            }
        }

        for (int i = start; i < messages.Count; i++)
        {
            var message = messages[i];
            if (_options.CheckToolResults)
            {
                foreach (var result in message.Contents.OfType<FunctionResultContent>())
                {
                    if (!TryGetModelText(result.Result, out var text))
                    {
                        Record(new GuardFinding(GuardStage.Input, GuardSource.ToolResult, i, GuardResult.NotScanned("guardrails", text)), findings);
                    }
                    else if (text.Length > 0)
                    {
                        await CheckAsync(new GuardRequest(text, GuardStage.Input, GuardSource.ToolResult), i, findings, cancellationToken).ConfigureAwait(false);
                    }
                }
            }

            if (_options.CheckUserMessages && message.Role == ChatRole.User && message.Text is { Length: > 0 } userText)
            {
                await CheckAsync(GuardRequest.UserInput(userText), i, findings, cancellationToken).ConfigureAwait(false);
            }
        }

        if (_options.CheckUserMessages && _options.ConversationWindow > 1 && start < messages.Count)
        {
            var window = messages.Where(m => m.Role == ChatRole.User && m.Text.Length > 0).TakeLast(_options.ConversationWindow).Select(m => m.Text).ToArray();
            if (window.Length > 1)
            {
                var joined = new GuardRequest(string.Join(' ', window), GuardStage.Input, GuardSource.Conversation);
                await CheckAsync(joined, ConversationIndex, findings, cancellationToken).ConfigureAwait(false);
            }
        }
    }

    /// <summary>
    /// The text the model sees for a tool result: a string as is; anything else as the JSON that chat clients send
    /// (structured results are serialized, so their <c>ToString()</c> is not what reaches the model). False, with the
    /// reason in <paramref name="text"/>, when the result cannot be serialized.
    /// </summary>
    private static bool TryGetModelText(object? result, out string text)
    {
        switch (result)
        {
            case null:
                text = "";
                return true;
            case string s:
                text = s;
                return true;
            case JsonElement { ValueKind: JsonValueKind.String } element:
                text = element.GetString() ?? "";
                return true;
        }

        try
        {
            text = JsonSerializer.Serialize(result, AIJsonUtilities.DefaultOptions.GetTypeInfo(result.GetType()));
            return true;
        }
        catch (Exception e) when (e is NotSupportedException or JsonException or InvalidOperationException)
        {
            text = $"Tool result of type {result.GetType().Name} could not be serialized to JSON for checking: {e.Message}";
            return false;
        }
    }

    private async Task CheckAsync(GuardRequest request, int index, List<GuardFinding> findings, CancellationToken cancellationToken)
    {
        var result = await _guard.CheckAsync(request, cancellationToken).ConfigureAwait(false);
        Record(new GuardFinding(request.Stage, request.Source, index, result), findings);
    }

    private void Record(GuardFinding finding, List<GuardFinding> findings)
    {
        var result = finding.Result;
        findings.Add(finding);
        bool block = (result.Flagged && _options.OnFlagged == GuardAction.Block)
            || (!result.Complete && _options.BlockWhenNotScanned);
        if (block)
        {
            throw new GuardrailsViolationException(finding, new GuardrailsReport([.. findings]));
        }
    }
}

/// <summary>Adds guardrails to a chat client.</summary>
public static class GuardrailsChatClientExtensions
{
    /// <summary>
    /// Wraps <paramref name="client"/> in a <see cref="GuardrailsChatClient"/>. In a pipeline:
    /// <c>builder.Use(inner =&gt; inner.WithGuardrails(guard))</c>.
    /// </summary>
    public static IChatClient WithGuardrails(this IChatClient client, IContentGuard guard, GuardrailsOptions? options = null) =>
        new GuardrailsChatClient(client, guard, options);
}
