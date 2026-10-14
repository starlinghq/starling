using System.Text.Json;
using Starlings.Generation.OnnxGenAI;
using Starlings.TestSupport;
using Microsoft.Extensions.AI;

namespace Starlings.Models.Qwen35.Extensions.AI.Tests;

/// <summary>
/// P-029 through the IChatClient: every case of the template corpus expressed as Microsoft.Extensions.AI messages and
/// options. Supported cases, tools (as <see cref="AIFunctionDeclaration"/>), <see cref="FunctionCallContent"/> and
/// <see cref="FunctionResultContent"/> included, must reach the model as the upstream prompt ids (transformers 5.18.0
/// rendering through the transformers tokenizer); every unsupported template feature (images, video, add_vision_id,
/// add_generation_prompt=False, unknown roles) must fail with <see cref="NotSupportedException"/> before any generator
/// is created. Scripted native sequence; needs the pinned tokenizer.json.
/// </summary>
public sealed class TemplateCorpusChatClientTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public static TheoryData<string> CaseIds() => [.. Qwen35TemplateCorpus.Load().Select(c => c.Id)];

    [Theory]
    [MemberData(nameof(CaseIds))]
    public async Task ChatClientSendsTheUpstreamPromptOrFailsExplicitly(string id)
    {
        var (tokenizer, reason) = Qwen35TestAssets.Tokenizer;
        Assert.SkipWhen(tokenizer is null, reason);
        var c = Qwen35TemplateCorpus.Get(id);
        var backend = new ScriptedBackend(tokenizer!, _ => [Qwen35ChatModel.ImEndTokenId]);
        using var client = new Qwen35ChatClient(backend.Model);
        var (messages, options) = ToChatRequest(c);

        // Text parts are concatenated by the adapter, as the template concatenates them.
        bool rendersHere = c.Owned == "render" || c.UnsupportedFeature == "content_parts";
        if (rendersHere && c.PromptIds is { } promptIds && promptIds.Length + GenerationOptions.DefaultMaxNewTokens > Qwen35ChatModel.MaxContextTokens)
        {
            // Beyond the accepted context (long_context_over_8192_tokens: unsupported): rejected before any native call,
            // with the rendered prompt's length; with the limit lifted (test seam) the prompt is the upstream one.
            var ex = await Assert.ThrowsAsync<GenerationContextLengthException>(() => client.GetResponseAsync(messages, options, Ct));
            Assert.Equal(promptIds.Length, ex.PromptTokens);
            Assert.Empty(backend.Sequences);
            backend.Model.EnforcedContextTokens = int.MaxValue;
            await client.GetResponseAsync(messages, options, Ct);
            Assert.Equal(c.PromptIds, backend.Single().Prompt);
        }
        else if (rendersHere)
        {
            await client.GetResponseAsync(messages, options, Ct);
            Assert.Equal(c.PromptIds, backend.Single().Prompt);
        }
        else if (c.Owned == "template_error")
        {
            await Assert.ThrowsAsync<Qwen35TemplateException>(() => client.GetResponseAsync(messages, options, Ct));
            Assert.Empty(backend.Sequences);
        }
        else
        {
            Assert.Equal("unsupported", c.Owned);
            await Assert.ThrowsAsync<NotSupportedException>(() => client.GetResponseAsync(messages, options, Ct));
            Assert.Empty(backend.Sequences);
        }
    }

    /// <summary>The case as a chat request: every corpus feature has a Microsoft.Extensions.AI form.</summary>
    private static (List<ChatMessage> Messages, ChatOptions Options) ToChatRequest(TemplateCorpusCase c)
    {
        var options = new ChatOptions();
        if (c.Tools.ValueKind == JsonValueKind.Array)
        {
            options.Tools = [.. c.Tools.EnumerateArray().Select(ToDeclaration)];
        }

        foreach (var kw in c.Kwargs.EnumerateObject())
        {
            (options.AdditionalProperties ??= [])[kw.Name] = kw.Value.Clone();
        }

        if (!c.AddGenerationPrompt)
        {
            (options.AdditionalProperties ??= [])["add_generation_prompt"] = false;
        }

        var messages = new List<ChatMessage>();
        foreach (var m in c.Messages.EnumerateArray())
        {
            string role = m.GetProperty("role").GetString()!;
            var contents = new List<AIContent>();
            if (m.TryGetProperty("reasoning_content", out var rc) && rc.ValueKind == JsonValueKind.String)
            {
                contents.Add(new TextReasoningContent(rc.GetString()));
            }

            var content = m.GetProperty("content");
            if (role == "tool")
            {
                contents.Add(new FunctionResultContent("call-0", content.GetString()));
            }
            else if (content.ValueKind == JsonValueKind.Null)
            {
                // No text: an assistant turn with tool calls only.
            }
            else if (content.ValueKind == JsonValueKind.String)
            {
                contents.Add(new TextContent(content.GetString()));
            }
            else if (content.ValueKind == JsonValueKind.Array)
            {
                foreach (var part in content.EnumerateArray())
                {
                    contents.Add(part.GetProperty("type").GetString() switch
                    {
                        "text" => new TextContent(part.GetProperty("text").GetString()),
                        "image" => new DataContent(new byte[] { 0x89, 0x50, 0x4E, 0x47 }, "image/png"),
                        "video" => new DataContent(new byte[] { 0, 0, 0, 0x18 }, "video/mp4"),
                        var other => throw new InvalidDataException($"{c.Id}: content part {other}"),
                    });
                }
            }

            if (m.TryGetProperty("tool_calls", out var calls))
            {
                int i = 0;
                foreach (var call in calls.EnumerateArray())
                {
                    var f = call.GetProperty("function");
                    var args = f.TryGetProperty("arguments", out var a) ? a.EnumerateObject().ToDictionary(p => p.Name, p => (object?)p.Value.Clone()) : null;
                    contents.Add(new FunctionCallContent($"call-{i++}", f.GetProperty("name").GetString()!, args));
                }
            }

            messages.Add(new ChatMessage(new ChatRole(role), contents));
        }

        return (messages, options);
    }

    // The corpus tools have the form the adapter writes for a declaration: type, then function name, description
    // (absent when empty) and parameters.
    private static AIFunctionDeclaration ToDeclaration(JsonElement tool)
    {
        var f = tool.GetProperty("function");
        return AIFunctionFactory.CreateDeclaration(
            f.GetProperty("name").GetString()!,
            f.TryGetProperty("description", out var d) ? d.GetString() : null,
            f.GetProperty("parameters").Clone());
    }
}
