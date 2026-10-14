using System.Text.Json;
using Starlings.Models.Qwen35;

namespace Starlings.TestSupport;

/// <summary>
/// The P-029 chat-template corpus: upstream renderings (transformers 5.18.0 apply_chat_template with the pinned
/// chat_template.jinja) written by model-pipeline/reference/generation/make_template_corpus.py. Read from the protected
/// copy once the maintainer accepts it, else from the proposal (model-pipeline/proposed/qwen3.5-0.8b/template/).
/// </summary>
internal static class Qwen35TemplateCorpus
{
    public static string CorpusPath
    {
        get
        {
            string root = Qwen35TestAssets.RepositoryRoot;
            string accepted = Path.Combine(root, "model-pipeline", "protected", "qwen3.5-0.8b", "template", "template-corpus.jsonl");
            return File.Exists(accepted)
                ? accepted
                : Path.Combine(root, "model-pipeline", "proposed", "qwen3.5-0.8b", "template", "template-corpus.jsonl");
        }
    }

    private static readonly Lazy<IReadOnlyList<TemplateCorpusCase>> Cases = new(() =>
        Qwen35TestAssets.ReadJsonLines(CorpusPath).Select(TemplateCorpusCase.Parse).ToList());

    public static IReadOnlyList<TemplateCorpusCase> Load() => Cases.Value;

    public static TemplateCorpusCase Get(string id) => Load().Single(c => c.Id == id);
}

/// <summary>One corpus case. <see cref="Owned"/>: render, template_error or unsupported (see the generator's docstring).</summary>
internal sealed record TemplateCorpusCase(
    string Id,
    string Category,
    string Owned,
    string? UnsupportedFeature,
    JsonElement Messages,
    JsonElement Tools,
    JsonElement Kwargs,
    bool AddGenerationPrompt,
    string? Rendered,
    string? Error,
    string? ErrorSource,
    int[]? PromptIds)
{
    public static TemplateCorpusCase Parse(JsonElement r) => new(
        r.GetProperty("id").GetString()!,
        r.GetProperty("category").GetString()!,
        r.GetProperty("owned").GetString()!,
        r.GetProperty("unsupported_feature").GetString(),
        r.GetProperty("messages"),
        r.GetProperty("tools"),
        r.GetProperty("chat_template_kwargs"),
        r.GetProperty("add_generation_prompt").GetBoolean(),
        r.GetProperty("rendered").GetString(),
        r.GetProperty("error").GetString(),
        r.GetProperty("error_source").GetString(),
        r.GetProperty("prompt_ids").ValueKind == JsonValueKind.Array ? r.GetProperty("prompt_ids").EnumerateArray().Select(x => x.GetInt32()).ToArray() : null);

    public bool HasTools => Tools.ValueKind == JsonValueKind.Array && Tools.GetArrayLength() > 0;

    /// <summary>The template's enable_thinking: only a JSON true counts (the template tests <c>is true</c>).</summary>
    public bool EnableThinking => Kwargs.TryGetProperty("enable_thinking", out var e) && e.ValueKind == JsonValueKind.True;

    /// <summary>
    /// The case as the typed API's input, or null with the reason when <see cref="Qwen35Message"/> and
    /// <see cref="Qwen35ChatTemplate.Render"/> have no way to express it (the explicit rejection of tools, tool messages,
    /// tool calls, non-string content and template arguments other than enable_thinking).
    /// </summary>
    public (Qwen35Message[]? Messages, string Reason) ToTypedMessages()
    {
        if (HasTools)
        {
            return (null, "tool definitions: no tools parameter");
        }

        if (!AddGenerationPrompt)
        {
            return (null, "add_generation_prompt=False: Render always appends the generation prompt");
        }

        foreach (var kw in Kwargs.EnumerateObject())
        {
            if (kw.Name != "enable_thinking")
            {
                return (null, $"template argument {kw.Name}: Render takes only enable_thinking");
            }
        }

        var list = new List<Qwen35Message>();
        foreach (var m in Messages.EnumerateArray())
        {
            if (m.TryGetProperty("tool_calls", out _))
            {
                return (null, "assistant tool calls: Qwen35Message has no tool calls");
            }

            Qwen35Role? role = m.GetProperty("role").GetString() switch
            {
                "system" => Qwen35Role.System,
                "user" => Qwen35Role.User,
                "assistant" => Qwen35Role.Assistant,
                _ => null,
            };
            if (role is null)
            {
                return (null, $"role {m.GetProperty("role").GetString()}: Qwen35Role has system, user and assistant only");
            }

            var content = m.GetProperty("content");
            if (content.ValueKind != JsonValueKind.String)
            {
                return (null, "non-string content (parts, images, video): Qwen35Message content is a string");
            }

            string? reasoning = m.TryGetProperty("reasoning_content", out var rc) && rc.ValueKind == JsonValueKind.String ? rc.GetString() : null;
            list.Add(new Qwen35Message(role.Value, content.GetString()!, reasoning));
        }

        return ([.. list], string.Empty);
    }
}
