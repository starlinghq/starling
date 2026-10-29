using System.Text.Json;
using Starlings.Models.QwenChat;
using Starlings.Tokenizers;

namespace Starlings.Models.QwenChat.Tests;

/// <summary>
/// The QwenChat template corpora: upstream renderings (transformers 5.18.0 apply_chat_template with each pinned
/// template) written by model-pipeline/reference/qwen-chat/make_template_corpus.py. Read from the protected copy once
/// the maintainer accepts it, else from the proposal (model-pipeline/proposed/&lt;model&gt;/template/).
/// </summary>
internal static class TemplateCorpus
{
    public const string Qwen3 = "qwen3-0.6b";
    public const string Qwen25 = "qwen2.5-0.5b-instruct";

    private static readonly Dictionary<string, Lazy<IReadOnlyList<TemplateCorpusCase>>> Cases = new()
    {
        [Qwen3] = new(() => Read(Qwen3)),
        [Qwen25] = new(() => Read(Qwen25)),
    };

    private static readonly Dictionary<string, Lazy<(Tokenizer? Tokenizer, string Reason)>> Tokenizers = new()
    {
        [Qwen3] = new(() => OpenTokenizer(QwenChatProfile.Qwen3)),
        [Qwen25] = new(() => OpenTokenizer(QwenChatProfile.Qwen25)),
    };

    public static string RepositoryRoot { get; } = FindRepositoryRoot();

    public static QwenChatProfile Profile(string model) => model == Qwen3 ? QwenChatProfile.Qwen3 : QwenChatProfile.Qwen25;

    public static string CorpusPath(string model)
    {
        string accepted = Path.Combine(RepositoryRoot, "model-pipeline", "protected", model, "template", "template-corpus.jsonl");
        return File.Exists(accepted)
            ? accepted
            : Path.Combine(RepositoryRoot, "model-pipeline", "proposed", model, "template", "template-corpus.jsonl");
    }

    public static IReadOnlyList<TemplateCorpusCase> Load(string model) => Cases[model].Value;

    public static TemplateCorpusCase Get(string model, string id) => Load(model).Single(c => c.Id == id);

    /// <summary>The profile's pinned tokenizer.json from <c>$STARLING_MODELS</c> or <c>&lt;repo&gt;/.models</c> (tools/fetch-tokenizer-assets.sh), or null with the reason.</summary>
    public static (Tokenizer? Tokenizer, string Reason) Tokenizer(string model) => Tokenizers[model].Value;

    private static List<TemplateCorpusCase> Read(string model) =>
        [.. File.ReadAllText(CorpusPath(model)).Split('\n')
            .Where(l => l.Length > 0)
            .Select(l => TemplateCorpusCase.Parse(JsonDocument.Parse(l).RootElement.Clone()))];

    private static (Tokenizer?, string) OpenTokenizer(QwenChatProfile profile)
    {
        string root = Environment.GetEnvironmentVariable("STARLING_MODELS") is { Length: > 0 } models
            ? models
            : Path.Combine(RepositoryRoot, ".models");
        string path = Path.Combine(root, profile.Repository, profile.Revision, "tokenizer.json");
        return File.Exists(path)
            ? (profile.OpenTokenizer(path), string.Empty)
            : (null, $"{path} not found; run tools/fetch-tokenizer-assets.sh.");
    }

    private static string FindRepositoryRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "model-pipeline", "protected")) &&
                Directory.Exists(Path.Combine(dir.FullName, "ai-dotnet")))
            {
                return dir.FullName;
            }
        }

        throw new DirectoryNotFoundException("Repository root not found above " + AppContext.BaseDirectory);
    }
}

/// <summary>One corpus case. <see cref="Owned"/>: render or unsupported (see the generator's docstring).</summary>
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
        r.GetProperty("prompt_ids").ValueKind == JsonValueKind.Array ? [.. r.GetProperty("prompt_ids").EnumerateArray().Select(x => x.GetInt32())] : null);

    /// <summary>The case's tool definitions for the typed API (JSON as given), or null.</summary>
    public QwenChatTool[]? TypedTools => Tools.ValueKind == JsonValueKind.Array ? [.. Tools.EnumerateArray().Select(QwenChatTool.FromJson)] : null;

    /// <summary>
    /// The case as the typed API's input, or null with the reason when <see cref="QwenChatMessage"/> and
    /// <see cref="QwenChatTemplate"/> have no way to express it.
    /// </summary>
    public (QwenChatMessage[]? Messages, string Reason) ToTypedMessages(QwenChatProfile profile)
    {
        if (!AddGenerationPrompt)
        {
            return (null, "add_generation_prompt=False: Render always appends the generation prompt");
        }

        foreach (var kw in Kwargs.EnumerateObject())
        {
            if (!(profile == QwenChatProfile.Qwen3 && kw.Name == "enable_thinking" && kw.Value.ValueKind == JsonValueKind.False))
            {
                return (null, $"template argument {kw.Name}={kw.Value.GetRawText()}: the profile fixes enable_thinking=false");
            }
        }

        var list = new List<QwenChatMessage>();
        foreach (var m in Messages.EnumerateArray())
        {
            QwenChatRole? role = m.GetProperty("role").GetString() switch
            {
                "system" => QwenChatRole.System,
                "user" => QwenChatRole.User,
                "assistant" => QwenChatRole.Assistant,
                "tool" => QwenChatRole.Tool,
                _ => null,
            };
            if (role is null)
            {
                return (null, $"role {m.GetProperty("role").GetString()}: QwenChatRole has system, user, assistant and tool only");
            }

            var content = m.GetProperty("content");
            if (content.ValueKind != JsonValueKind.String)
            {
                return (null, "non-string content: QwenChatMessage content is a string");
            }

            List<QwenChatToolCall>? calls = null;
            if (m.TryGetProperty("tool_calls", out var toolCalls))
            {
                calls = [];
                foreach (var call in toolCalls.EnumerateArray())
                {
                    var f = call.TryGetProperty("function", out var inner) ? inner : call;
                    calls.Add(new QwenChatToolCall(f.GetProperty("name").GetString()!, f.TryGetProperty("arguments", out var args) ? args.Clone() : default));
                }
            }

            string? reasoning = m.TryGetProperty("reasoning_content", out var rc) && rc.ValueKind == JsonValueKind.String ? rc.GetString() : null;
            list.Add(new QwenChatMessage(role.Value, content.GetString()!, reasoning) { ToolCalls = calls });
        }

        return ([.. list], string.Empty);
    }
}
