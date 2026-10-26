using System.Text;
using System.Text.Json;
using Starlings.Generation.OnnxGenAI;
using Microsoft.Extensions.AI;

namespace Starlings.Models.Qwen35.Extensions.AI;

/// <summary>
/// Maps Microsoft.Extensions.AI messages and options onto <see cref="Qwen35Message"/> and <see cref="Qwen35ChatOptions"/>:
/// what <see cref="Qwen35ChatClient"/> renders for a request, exposed so a conversation can be rendered or counted
/// (<see cref="Qwen35ChatModel.CreatePrompt(IReadOnlyList{Qwen35Message}, Qwen35ChatOptions)"/>) without a request.
/// </summary>
/// <remarks>Anything the model cannot honor fails with <see cref="NotSupportedException"/>; it is never silently ignored.</remarks>
public static class Qwen35ChatRequestMapper
{
    /// <summary>
    /// Maps a conversation as the class remarks of <see cref="Qwen35ChatClient"/> describe: <paramref name="instructions"/>
    /// (<see cref="ChatOptions.Instructions"/>) becomes a leading system message when given, system, user and assistant
    /// messages map one to one (text concatenated; assistant reasoning and function calls carried over), and each
    /// <see cref="FunctionResultContent"/> of a tool message becomes one tool message.
    /// </summary>
    /// <exception cref="NotSupportedException">A role, content type, author name or instructions placement the template cannot render.</exception>
    public static List<Qwen35Message> MapMessages(IEnumerable<ChatMessage> messages, string? instructions)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var result = new List<Qwen35Message>(messages is ICollection<ChatMessage> c ? c.Count + 1 : 4);
        if (instructions is not null)
        {
            result.Add(Qwen35Message.System(instructions));
        }

        int index = 0;
        foreach (ChatMessage? message in messages)
        {
            if (message is null)
            {
                throw new ArgumentException($"Message {index} is null.", nameof(messages));
            }

            Qwen35Role role = MapRole(message.Role, index);
            if (role == Qwen35Role.Tool)
            {
                MapToolMessage(message, index, result);
                index++;
                continue;
            }

            if (role == Qwen35Role.System && instructions is not null)
            {
                throw new NotSupportedException(
                    "ChatOptions.Instructions is supported only when the conversation has no system message: the chat template accepts one system message, at the beginning. Put the instructions into that system message instead.");
            }

            if (message.AuthorName is not null)
            {
                throw new NotSupportedException($"Message {index}: ChatMessage.AuthorName is not supported; the chat template has no participant names.");
            }

            StringBuilder? text = null;
            StringBuilder? reasoning = null;
            List<Qwen35ToolCall>? calls = null;
            foreach (AIContent content in message.Contents)
            {
                switch (content)
                {
                    case TextContent t:
                        (text ??= new StringBuilder()).Append(t.Text);
                        break;
                    case TextReasoningContent r when role == Qwen35Role.Assistant:
                        (reasoning ??= new StringBuilder()).Append(r.Text);
                        break;
                    case TextReasoningContent:
                        throw new NotSupportedException($"Message {index}: reasoning content is supported only in assistant messages.");
                    case FunctionCallContent call when role == Qwen35Role.Assistant:
                        (calls ??= []).Add(MapCall(call, index));
                        break;
                    case FunctionCallContent:
                        throw new NotSupportedException($"Message {index}: FunctionCallContent is supported only in assistant messages.");
                    case FunctionResultContent:
                        throw new NotSupportedException($"Message {index}: FunctionResultContent is supported only in tool messages (ChatRole.Tool).");
                    case DataContent or UriContent or HostedFileContent:
                        throw new NotSupportedException($"Message {index}: {content.GetType().Name} is not supported: this model accepts text only (Qwen35Capabilities: vision unsupported).");
                    default:
                        throw new NotSupportedException($"Message {index}: content of type {content.GetType().Name} is not supported; only TextContent (and TextReasoningContent and FunctionCallContent in assistant messages) can be rendered.");
                }
            }

            result.Add(new Qwen35Message(role, text?.ToString() ?? string.Empty, reasoning?.ToString()) { ToolCalls = calls });
            index++;
        }

        return result;
    }

    /// <summary>
    /// Maps request options onto <paramref name="defaults"/> as the class remarks of <see cref="Qwen35ChatClient"/>
    /// describe: null returns <paramref name="defaults"/> unchanged; otherwise every set <see cref="ChatOptions"/>
    /// property is applied or rejected, never ignored. <paramref name="modelId"/> is the only
    /// <see cref="ChatOptions.ModelId"/> accepted. Settings without a <see cref="ChatOptions"/> counterpart
    /// (<see cref="Qwen35ChatOptions.EscapeSpecialTokens"/>, <see cref="Qwen35ChatOptions.StopAtRoleMarker"/>,
    /// <see cref="Qwen35ChatOptions.PrefillChunkTokens"/>) are kept from <paramref name="defaults"/>.
    /// </summary>
    /// <exception cref="NotSupportedException">An option the model cannot honor is set.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A value is out of range (<c>MaxOutputTokens</c>, <c>Temperature</c>, <c>Seed</c>).</exception>
    public static Qwen35ChatOptions MapOptions(ChatOptions? options, Qwen35ChatOptions defaults, string modelId) =>
        MapOptions(options, defaults, modelId, Unguided);

    /// <summary>
    /// Maps the options. <paramref name="guidance"/> says whether the loaded model can serve guided decoding; it decides
    /// whether <see cref="ChatToolMode.RequireAny"/>, a required function and a JSON <see cref="ChatOptions.ResponseFormat"/>
    /// map to a grammar or fail with <see cref="NotSupportedException"/>.
    /// </summary>
    internal static Qwen35ChatOptions MapOptions(ChatOptions? options, Qwen35ChatOptions defaults, string modelId, GuidanceSupport guidance)
    {
        ArgumentNullException.ThrowIfNull(defaults);
        ArgumentNullException.ThrowIfNull(modelId);
        if (options is null)
        {
            return defaults;
        }

        if (options.ModelId is { } requested && !string.Equals(requested, modelId, StringComparison.Ordinal))
        {
            throw new NotSupportedException($"ChatOptions.ModelId = '{requested}' is not supported: this client serves only '{modelId}'.");
        }

        List<Qwen35Tool> tools = options.ToolMode is NoneChatToolMode ? [] : MapTools(options.Tools);
        GenerationGuidance? constraint = null;
        string? constrainedBy = null;
        if (options.ToolMode is RequiredChatToolMode required)
        {
            constrainedBy = $"ChatOptions.ToolMode = {(required.RequiredFunctionName is { } n ? $"RequireSpecific('{n}')" : "RequireAny")}";
            guidance.ThrowIfUnavailable(constrainedBy, "the chat template has no way to require a tool call, so it needs a grammar-constrained decode");
            if (tools.Count == 0)
            {
                throw new NotSupportedException($"{constrainedBy} needs at least one function in ChatOptions.Tools.");
            }

            List<Qwen35Tool> allowed = tools;
            if (required.RequiredFunctionName is { } name)
            {
                allowed = tools.FindAll(t => string.Equals(t.Name, name, StringComparison.Ordinal));
                if (allowed.Count == 0)
                {
                    throw new NotSupportedException($"{constrainedBy}: ChatOptions.Tools has no function named '{name}'.");
                }
            }

            try
            {
                constraint = Qwen35ToolGrammar.Build(allowed, new Qwen35ToolGrammarOptions
                {
                    AllowMultipleCalls = options.AllowMultipleToolCalls != false,
                    FastForward = guidance.FastForward,
                });
            }
            catch (ArgumentException ex)
            {
                throw new NotSupportedException($"{constrainedBy}: {ex.Message}", ex);
            }
        }
        else if (tools.Count > 0 && options.AllowMultipleToolCalls == false)
        {
            throw new NotSupportedException(
                "ChatOptions.AllowMultipleToolCalls = false is supported only with ChatToolMode.RequireAny or RequireSpecific: with ChatToolMode.Auto the chat template lets the model call several tools in one turn and decoding is not constrained.");
        }

        if (options.ResponseFormat is ChatResponseFormatJson json)
        {
            const string What = "ChatOptions.ResponseFormat = Json";
            guidance.ThrowIfUnavailable(What, "structured (JSON) output needs a schema-constrained decode");
            if (constraint is not null)
            {
                throw new NotSupportedException($"{What} cannot be combined with {constrainedBy}.");
            }

            if (tools.Count > 0)
            {
                throw new NotSupportedException(
                    $"{What} cannot be combined with offered tools: output constrained to JSON cannot hold a tool call. Use ChatToolMode.None or no tools.");
            }

            // Without a schema, the OpenAI-style JSON mode: one JSON object.
            constraint = GenerationGuidance.JsonSchema(json.Schema is { } schema ? schema.GetRawText() : """{"type": "object"}""");
            constrainedBy = What;
        }
        else if (options.ResponseFormat is not (null or ChatResponseFormatText))
        {
            throw new NotSupportedException($"ChatOptions.ResponseFormat {options.ResponseFormat.GetType().Name} is not supported; text and JSON are.");
        }

        if (options.ConversationId is not null)
        {
            throw new NotSupportedException("ChatOptions.ConversationId is not supported: the client keeps no conversation state; send the whole conversation with each request.");
        }

        if (options.FrequencyPenalty is { } frequency && frequency != 0)
        {
            throw new NotSupportedException("ChatOptions.FrequencyPenalty is not supported by this model's decoding.");
        }

        if (options.PresencePenalty is { } presence && presence != 0)
        {
            throw new NotSupportedException("ChatOptions.PresencePenalty is not supported by this model's decoding.");
        }

        if (options.Reasoning is not null)
        {
            throw new NotSupportedException(
                $"ChatOptions.Reasoning is not supported: the model has no reasoning effort levels. Use AdditionalProperties['{Qwen35ChatClient.EnableThinkingPropertyName}'] = true or false to switch the template's thinking mode, and Qwen35ChatClientOptions.SplitReasoning to return the reasoning as TextReasoningContent.");
        }

        if (options.AllowBackgroundResponses == true || options.ContinuationToken is not null)
        {
            throw new NotSupportedException("Background responses (ChatOptions.AllowBackgroundResponses, ContinuationToken) are not supported.");
        }

        if (options.RawRepresentationFactory is not null)
        {
            throw new NotSupportedException("ChatOptions.RawRepresentationFactory is not supported: this client has no provider-specific options object.");
        }

        var mapped = defaults with { Tools = tools };
        if (constraint is not null)
        {
            mapped = mapped with { Guidance = constraint };
        }
        if (options.MaxOutputTokens is { } max)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(max, "ChatOptions.MaxOutputTokens");
            mapped = mapped with { MaxNewTokens = max };
        }

        if (options.StopSequences is { } stops)
        {
            mapped = mapped with { StopSequences = [.. stops] };
        }

        mapped = mapped with { Sampling = MapSampling(options, defaults.Sampling) };

        if (options.AdditionalProperties is { Count: > 0 } properties)
        {
            int? topLogprobs = null;
            foreach (var (key, value) in properties)
            {
                switch (key)
                {
                    case Qwen35ChatClient.EnableThinkingPropertyName:
                        mapped = mapped with { EnableThinking = ParseBoolean(key, value) };
                        break;
                    case Qwen35ChatClient.LogprobsPropertyName:
                        mapped = mapped with { ReturnLogprobs = ParseBoolean(key, value) };
                        break;
                    case Qwen35ChatClient.TopLogprobsPropertyName:
                        topLogprobs = ParseTopLogprobs(value);
                        break;
                    default:
                        throw new NotSupportedException(
                            $"ChatOptions.AdditionalProperties['{key}'] is not supported; the supported keys are '{Qwen35ChatClient.EnableThinkingPropertyName}', '{Qwen35ChatClient.LogprobsPropertyName}' and '{Qwen35ChatClient.TopLogprobsPropertyName}'.");
                }
            }

            if (topLogprobs is int top)
            {
                if (top > 0 && !mapped.ReturnLogprobs)
                {
                    throw new ArgumentException(
                        $"AdditionalProperties['{Qwen35ChatClient.TopLogprobsPropertyName}'] needs AdditionalProperties['{Qwen35ChatClient.LogprobsPropertyName}'] = true.");
                }

                mapped = mapped with { TopLogprobs = top };
            }
            else if (!mapped.ReturnLogprobs)
            {
                // A request that turns log probabilities off also drops the defaults' top-k, which would otherwise be
                // rejected without them.
                mapped = mapped with { TopLogprobs = 0 };
            }
        }

        if (constraint is not null)
        {
            if (mapped.EnableThinking)
            {
                throw new NotSupportedException(
                    $"{constrainedBy} is not supported with '{Qwen35ChatClient.EnableThinkingPropertyName}' = true: the constraint applies from the first completion token, which would be reasoning text.");
            }

            if (mapped.ReturnLogprobs)
            {
                throw new NotSupportedException(
                    $"{constrainedBy} cannot be combined with AdditionalProperties['{Qwen35ChatClient.LogprobsPropertyName}'] = true: log probabilities are not offered for guided decoding.");
            }
        }

        return mapped;
    }

    private static readonly GuidanceSupport Unguided = new(GenAIModel.GuidanceFiles, "the model directory", FastForward: false);

    /// <summary>Whether the loaded model can serve guided decoding, and how the mapper should build constraints.</summary>
    /// <param name="MissingFiles">GenAI's guidance files missing from the model directory (empty: available).</param>
    /// <param name="ModelDirectory">The model directory, for the error message.</param>
    /// <param name="FastForward">Build tool grammars with fast-forward tokens (<see cref="Qwen35ChatClientOptions.GuidanceFastForward"/>).</param>
    internal sealed record GuidanceSupport(IReadOnlyList<string> MissingFiles, string ModelDirectory, bool FastForward)
    {
        public void ThrowIfUnavailable(string what, string why)
        {
            if (MissingFiles.Count > 0)
            {
                throw new NotSupportedException(
                    $"{what} is not supported by this model directory: {why}, and ONNX Runtime GenAI builds the constraint's token table from " +
                    $"{string.Join(" and ", GenAIModel.GuidanceFiles)} in {ModelDirectory}, where {string.Join(" and ", MissingFiles)} " +
                    (MissingFiles.Count == 1 ? "is" : "are") + " missing (Qwen35ModelInfo.GuidanceMissingFiles).");
            }
        }
    }

    private static int ParseTopLogprobs(object? value)
    {
        int? top = value switch
        {
            int i => i,
            long l when l is >= int.MinValue and <= int.MaxValue => (int)l,
            JsonElement { ValueKind: JsonValueKind.Number } e when e.TryGetInt32(out int i) => i,
            _ => null,
        };
        if (top is not int n || n < 0 || n > GenerationOptions.MaxTopLogprobs)
        {
            throw new ArgumentException(
                $"AdditionalProperties['{Qwen35ChatClient.TopLogprobsPropertyName}'] = '{value}' is not an integer in [0, {GenerationOptions.MaxTopLogprobs}].");
        }

        return n;
    }

    private static Qwen35Role MapRole(ChatRole role, int index)
    {
        if (role == ChatRole.System)
        {
            return Qwen35Role.System;
        }

        if (role == ChatRole.User)
        {
            return Qwen35Role.User;
        }

        if (role == ChatRole.Assistant)
        {
            return Qwen35Role.Assistant;
        }

        if (role == ChatRole.Tool)
        {
            return Qwen35Role.Tool;
        }

        throw new NotSupportedException($"Message {index}: role '{role}' is not supported; the chat template knows system, user, assistant and tool.");
    }

    // ChatOptions.Tools: each AIFunctionDeclaration (AIFunction included) becomes
    // {"type": "function", "function": {"name", "description" (when not empty), "parameters": its JSON schema}}.
    private static List<Qwen35Tool> MapTools(IList<AITool>? tools)
    {
        if (tools is not { Count: > 0 })
        {
            return [];
        }

        var mapped = new List<Qwen35Tool>(tools.Count);
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (AITool? tool in tools)
        {
            if (tool is not AIFunctionDeclaration function)
            {
                throw new NotSupportedException(
                    $"ChatOptions.Tools: {tool?.GetType().Name ?? "null"} is not supported; only function tools (AIFunctionDeclaration, AIFunction) can be rendered by the chat template.");
            }

            if (!names.Add(function.Name))
            {
                throw new NotSupportedException($"ChatOptions.Tools: the function name '{function.Name}' appears twice.");
            }

            try
            {
                mapped.Add(Qwen35Tool.Function(function.Name, string.IsNullOrEmpty(function.Description) ? null : function.Description, function.JsonSchema));
            }
            catch (ArgumentException ex)
            {
                throw new NotSupportedException($"ChatOptions.Tools: {ex.Message}", ex);
            }
        }

        return mapped;
    }

    private static Qwen35ToolCall MapCall(FunctionCallContent call, int index)
    {
        if (string.IsNullOrEmpty(call.Name))
        {
            throw new NotSupportedException($"Message {index}: a FunctionCallContent has no name.");
        }

        var arguments = new List<KeyValuePair<string, JsonElement>>();
        if (call.Arguments is { } args)
        {
            foreach (var (key, value) in args)
            {
                arguments.Add(new(key, ToJson(value)));
            }
        }

        return Qwen35ToolCall.Create(call.Name, arguments);
    }

    // A tool message: one template tool message per FunctionResultContent, in order.
    private static void MapToolMessage(ChatMessage message, int index, List<Qwen35Message> result)
    {
        if (message.AuthorName is not null)
        {
            throw new NotSupportedException($"Message {index}: ChatMessage.AuthorName is not supported; the chat template has no participant names.");
        }

        int count = 0;
        foreach (AIContent content in message.Contents)
        {
            if (content is not FunctionResultContent r)
            {
                throw new NotSupportedException($"Message {index}: a tool message may contain only FunctionResultContent, not {content.GetType().Name}.");
            }

            result.Add(Qwen35Message.Tool(ResultText(r.Result)));
            count++;
        }

        if (count == 0)
        {
            throw new NotSupportedException($"Message {index}: a tool message needs at least one FunctionResultContent.");
        }
    }

    // A function result as the tool message text: a string as is, anything else as compact JSON.
    internal static string ResultText(object? result)
    {
        if (result is string s)
        {
            return s;
        }

        JsonElement e = ToJson(result);
        if (e.ValueKind == JsonValueKind.String)
        {
            return e.GetString()!;
        }

        using var stream = new MemoryStream();
        using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping }))
        {
            e.WriteTo(writer);
        }

        return Encoding.UTF8.GetString(stream.ToArray());
    }

    private static JsonElement ToJson(object? value) => value switch
    {
        JsonElement e => e,
        _ => JsonSerializer.SerializeToElement(value, AIJsonUtilities.DefaultOptions.GetTypeInfo(typeof(object))),
    };

    // Greedy unless sampling is requested. Sampling needs an explicit seed (P-016: every sampled run is reproducible).
    // A request that sets Temperature or TopP without TopK, over no sampling default, gets SamplingOptions' default
    // top-k of 50 (the GenAI default); ChatOptions has no way to switch top-k off, and this is stated in the docs.
    private static SamplingOptions? MapSampling(ChatOptions options, SamplingOptions? defaults)
    {
        bool samplingRequested = options.TopP is not null || options.TopK is not null || options.Temperature is not null;
        if (options.Temperature is { } t && !(t >= 0 && float.IsFinite(t)))
        {
            throw new ArgumentOutOfRangeException(nameof(options), t, "ChatOptions.Temperature must be a finite number, 0 (greedy) or greater.");
        }

        if (options.Temperature == 0)
        {
            if (options.TopP is not null || options.TopK is not null)
            {
                throw new NotSupportedException("ChatOptions.Temperature = 0 selects greedy decoding; TopP and TopK cannot be combined with it.");
            }

            return null;
        }

        int? seed = options.Seed is { } s
            ? s is >= 0 and <= int.MaxValue ? (int)s : throw new ArgumentOutOfRangeException(nameof(options), s, $"ChatOptions.Seed must be in [0, {int.MaxValue}].")
            : null;
        if (!samplingRequested)
        {
            // No sampling parameter: the default decoding. A seed alone re-seeds default sampling and has no effect on
            // greedy decoding, which is deterministic anyway.
            return defaults is not null && seed is int reseed ? defaults with { Seed = reseed } : defaults;
        }

        seed ??= defaults?.Seed;
        if (seed is null)
        {
            throw new NotSupportedException(
                "Sampling (ChatOptions.Temperature > 0, TopP or TopK) requires ChatOptions.Seed: unseeded sampling is unsupported, so that every sampled response is reproducible. Set Seed, or omit Temperature, TopP and TopK (or set Temperature = 0) for greedy decoding.");
        }

        var baseline = defaults ?? new SamplingOptions { Seed = seed.Value };
        return baseline with
        {
            Seed = seed.Value,
            Temperature = options.Temperature ?? baseline.Temperature,
            TopP = options.TopP ?? baseline.TopP,
            TopK = options.TopK ?? baseline.TopK,
        };
    }

    private static bool ParseBoolean(string key, object? value) => value switch
    {
        bool b => b,
        JsonElement { ValueKind: JsonValueKind.True } => true,
        JsonElement { ValueKind: JsonValueKind.False } => false,
        _ => throw new ArgumentException($"AdditionalProperties['{key}'] = '{value}' is not a Boolean; expected true or false."),
    };
}
