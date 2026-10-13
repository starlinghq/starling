using System.Text;
using System.Text.Json;
using Starlings.Generation.OnnxGenAI;
using Microsoft.Extensions.AI;

namespace Starlings.Models.Qwen35.Extensions.AI;

/// <summary>Maps Microsoft.Extensions.AI messages and options onto <see cref="Qwen35Message"/> and <see cref="Qwen35ChatOptions"/>.</summary>
/// <remarks>Anything the model cannot honor fails with <see cref="NotSupportedException"/>; it is never silently ignored.</remarks>
internal static class Qwen35ChatRequestMapper
{
    private const string ToolsUnsupported = "tool calling is unsupported for Qwen3.5-0.8B in this package version (Qwen35Capabilities: tool_calling, template.tools, template.tool_messages)";

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
                    case FunctionCallContent or FunctionResultContent:
                        throw new NotSupportedException($"Message {index}: {content.GetType().Name} is not supported: {ToolsUnsupported}.");
                    case DataContent or UriContent or HostedFileContent:
                        throw new NotSupportedException($"Message {index}: {content.GetType().Name} is not supported: this model accepts text only (Qwen35Capabilities: vision unsupported).");
                    default:
                        throw new NotSupportedException($"Message {index}: content of type {content.GetType().Name} is not supported; only TextContent (and TextReasoningContent in assistant messages) can be rendered.");
                }
            }

            result.Add(new Qwen35Message(role, text?.ToString() ?? string.Empty, reasoning?.ToString()));
            index++;
        }

        return result;
    }

    public static Qwen35ChatOptions MapOptions(ChatOptions? options, Qwen35ChatOptions defaults, string modelId)
    {
        if (options is null)
        {
            return defaults;
        }

        if (options.ModelId is { } requested && !string.Equals(requested, modelId, StringComparison.Ordinal))
        {
            throw new NotSupportedException($"ChatOptions.ModelId = '{requested}' is not supported: this client serves only '{modelId}'.");
        }

        if (options.Tools is { Count: > 0 })
        {
            throw new NotSupportedException($"ChatOptions.Tools is not supported: {ToolsUnsupported}.");
        }

        if (options.ToolMode is RequiredChatToolMode)
        {
            throw new NotSupportedException($"ChatOptions.ToolMode = Required is not supported: {ToolsUnsupported}.");
        }

        // ToolMode None or Auto and AllowMultipleToolCalls have no effect without tools, so they are accepted.
        if (options.ResponseFormat is not (null or ChatResponseFormatText))
        {
            throw new NotSupportedException("ChatOptions.ResponseFormat other than text is not supported: structured (JSON) output is unsupported (Qwen35Capabilities: structured_output).");
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
                $"ChatOptions.Reasoning is not supported: the model has no reasoning effort levels and its reasoning is not split from the answer. Use AdditionalProperties['{Qwen35ChatClient.EnableThinkingPropertyName}'] = true or false to switch the template's thinking mode.");
        }

        if (options.AllowBackgroundResponses == true || options.ContinuationToken is not null)
        {
            throw new NotSupportedException("Background responses (ChatOptions.AllowBackgroundResponses, ContinuationToken) are not supported.");
        }

        if (options.RawRepresentationFactory is not null)
        {
            throw new NotSupportedException("ChatOptions.RawRepresentationFactory is not supported: this client has no provider-specific options object.");
        }

        var mapped = defaults;
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
            foreach (var (key, value) in properties)
            {
                if (!string.Equals(key, Qwen35ChatClient.EnableThinkingPropertyName, StringComparison.Ordinal))
                {
                    throw new NotSupportedException(
                        $"ChatOptions.AdditionalProperties['{key}'] is not supported; the only supported key is '{Qwen35ChatClient.EnableThinkingPropertyName}'.");
                }

                mapped = mapped with { EnableThinking = ParseBoolean(value) };
            }
        }

        return mapped;
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
            throw new NotSupportedException($"Message {index}: role 'tool' is not supported: {ToolsUnsupported}.");
        }

        throw new NotSupportedException($"Message {index}: role '{role}' is not supported; the chat template knows system, user and assistant.");
    }

    // Greedy unless sampling is requested. Sampling needs an explicit seed (P-016: every sampled run is reproducible).
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

    private static bool ParseBoolean(object? value) => value switch
    {
        bool b => b,
        JsonElement { ValueKind: JsonValueKind.True } => true,
        JsonElement { ValueKind: JsonValueKind.False } => false,
        _ => throw new ArgumentException(
            $"AdditionalProperties['{Qwen35ChatClient.EnableThinkingPropertyName}'] = '{value}' is not a Boolean; expected true or false."),
    };
}
