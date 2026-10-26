using System.Text.Json;

namespace Starlings.Generation;

/// <summary>Role of a chat turn.</summary>
public enum ChatTurnRole
{
    /// <summary>System instructions.</summary>
    System,

    /// <summary>A user turn.</summary>
    User,

    /// <summary>An assistant turn, optionally with tool calls (<see cref="ChatTurn.ToolCalls"/>).</summary>
    Assistant,

    /// <summary>The result of one tool call, as text.</summary>
    Tool,
}

/// <summary>
/// One text turn of a conversation, independent of a model's template. A template (<see cref="IChatTemplate"/>) checks
/// the turns it renders and rejects what its format cannot carry.
/// </summary>
/// <param name="Role">The role.</param>
/// <param name="Content">Plain text content (may be empty). For a <see cref="ChatTurnRole.Tool"/> turn: the tool's result as text.</param>
/// <param name="ReasoningContent">Assistant only: the reasoning of that turn, or null when not given.</param>
public sealed record ChatTurn(ChatTurnRole Role, string Content, string? ReasoningContent = null)
{
    /// <summary>Assistant only: the tool calls of that turn, after <see cref="Content"/>. Null or empty: no tool calls.</summary>
    public IReadOnlyList<ToolCall>? ToolCalls { get; init; }

    /// <summary>A system turn.</summary>
    public static ChatTurn System(string content) => new(ChatTurnRole.System, content);

    /// <summary>A user turn.</summary>
    public static ChatTurn User(string content) => new(ChatTurnRole.User, content);

    /// <summary>An assistant turn.</summary>
    public static ChatTurn Assistant(string content, string? reasoningContent = null) => new(ChatTurnRole.Assistant, content, reasoningContent);

    /// <summary>An assistant turn with tool calls (and optional text before them).</summary>
    public static ChatTurn Assistant(string content, IReadOnlyList<ToolCall> toolCalls, string? reasoningContent = null) =>
        new(ChatTurnRole.Assistant, content, reasoningContent) { ToolCalls = toolCalls };

    /// <summary>A tool result turn.</summary>
    public static ChatTurn Tool(string content) => new(ChatTurnRole.Tool, content);
}

/// <summary>A tool call of an assistant turn: the function name and its arguments.</summary>
/// <param name="Name">The function name.</param>
/// <param name="Arguments">
/// The arguments as a JSON object (property order is kept), or an undefined element (<c>default</c>) for a call without
/// arguments. How a value is written depends on the template.
/// </param>
public sealed record ToolCall(string Name, JsonElement Arguments)
{
    /// <summary>A call whose arguments are given as name and JSON value pairs, in order.</summary>
    public static ToolCall Create(string name, IEnumerable<KeyValuePair<string, JsonElement>> arguments)
    {
        ArgumentNullException.ThrowIfNull(arguments);
        using var stream = new MemoryStream();
        using (var writer = new Utf8JsonWriter(stream))
        {
            writer.WriteStartObject();
            foreach (var (key, value) in arguments)
            {
                writer.WritePropertyName(key);
                value.WriteTo(writer);
            }

            writer.WriteEndObject();
        }

        using var document = JsonDocument.Parse(stream.ToArray());
        return new ToolCall(name, document.RootElement.Clone());
    }
}

/// <summary>
/// A tool (function) definition, held as the JSON the chat templates render (<c>tool | tojson</c>), keys in their given
/// order. A template checks the name against its own call format when it renders the definition.
/// </summary>
public sealed class ToolDefinition
{
    private ToolDefinition(JsonElement json, string name, JsonElement parameters)
    {
        Json = json;
        Name = name;
        Parameters = parameters;
    }

    /// <summary>The function name.</summary>
    public string Name { get; }

    /// <summary>The JSON a template renders into its tools block.</summary>
    public JsonElement Json { get; }

    /// <summary>The function's parameters JSON schema, or an undefined element when the definition has none.</summary>
    public JsonElement Parameters { get; }

    /// <summary>
    /// A function definition in the form transformers and the OpenAI API use:
    /// <c>{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}</c>, with
    /// <c>description</c> left out when null and <c>parameters</c> left out when not given.
    /// </summary>
    /// <param name="name">The function name (non-empty).</param>
    /// <param name="description">What the function does, or null.</param>
    /// <param name="parameters">The JSON schema of the arguments, or null.</param>
    public static ToolDefinition Function(string name, string? description = null, JsonElement? parameters = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        using var stream = new MemoryStream();
        using (var writer = new Utf8JsonWriter(stream))
        {
            writer.WriteStartObject();
            writer.WriteString("type", "function");
            writer.WritePropertyName("function");
            writer.WriteStartObject();
            writer.WriteString("name", name);
            if (description is not null)
            {
                writer.WriteString("description", description);
            }

            if (parameters is { } p)
            {
                writer.WritePropertyName("parameters");
                p.WriteTo(writer);
            }

            writer.WriteEndObject();
            writer.WriteEndObject();
        }

        using var document = JsonDocument.Parse(stream.ToArray());
        return FromJson(document.RootElement);
    }

    /// <summary>
    /// A definition given as JSON, rendered exactly as given: an object with a <c>function</c> object (or the function
    /// object itself) whose <c>name</c> is a non-empty string.
    /// </summary>
    /// <exception cref="ArgumentException">The JSON is not such an object, or the name is not valid UTF-16.</exception>
    public static ToolDefinition FromJson(JsonElement json)
    {
        if (json.ValueKind != JsonValueKind.Object)
        {
            throw new ArgumentException("A tool definition must be a JSON object.", nameof(json));
        }

        JsonElement function = json.TryGetProperty("function", out var f) && f.ValueKind == JsonValueKind.Object ? f : json;
        if (!function.TryGetProperty("name", out var name) || name.ValueKind != JsonValueKind.String)
        {
            throw new ArgumentException("A tool definition needs a string function name (\"function\": {\"name\": ...}).", nameof(json));
        }

        string n;
        try
        {
            n = name.GetString()!;
        }
        catch (InvalidOperationException ex)
        {
            throw new ArgumentException("The tool name contains a lone UTF-16 surrogate.", nameof(json), ex);
        }

        if (n.Length == 0)
        {
            throw new ArgumentException("The tool name is empty.", nameof(json));
        }

        JsonElement parameters = function.TryGetProperty("parameters", out var p) ? p.Clone() : default;
        return new ToolDefinition(json.Clone(), n, parameters);
    }
}

/// <summary>One piece of an assistant completion: text, or a parsed tool call.</summary>
/// <param name="Text">Text, when this part is text.</param>
/// <param name="ToolCall">The call, when this part is a tool call.</param>
public sealed record CompletionPart(string? Text, ToolCall? ToolCall)
{
    /// <summary>A text part.</summary>
    public static CompletionPart FromText(string text) => new(text, null);

    /// <summary>A tool-call part.</summary>
    public static CompletionPart FromToolCall(ToolCall call) => new(null, call);
}

/// <summary>A completion split into text and tool calls by an <see cref="IToolCallParser"/>.</summary>
/// <param name="Parts">Text and tool-call parts in output order; adjacent text is merged.</param>
public sealed record ParsedCompletion(IReadOnlyList<CompletionPart> Parts)
{
    /// <summary>All text parts, concatenated.</summary>
    public string Text => string.Concat(Parts.Where(p => p.Text is not null).Select(p => p.Text));

    /// <summary>The tool calls, in order.</summary>
    public IReadOnlyList<ToolCall> ToolCalls => [.. Parts.Where(p => p.ToolCall is not null).Select(p => p.ToolCall!)];

    /// <summary>The completion as an assistant turn for the next request: <see cref="Text"/> and <see cref="ToolCalls"/>.</summary>
    public ChatTurn ToAssistantTurn() => new(ChatTurnRole.Assistant, Text) { ToolCalls = ToolCalls is { Count: > 0 } calls ? calls : null };

    /// <summary>
    /// Parses a whole completion with a fresh parser: <see cref="IToolCallParser.Append"/> of the whole text, then
    /// <see cref="IToolCallParser.Complete"/>, with adjacent text parts merged.
    /// </summary>
    /// <param name="parser">A parser that has not been fed yet.</param>
    /// <param name="completion">The completion text.</param>
    public static ParsedCompletion Parse(IToolCallParser parser, string completion)
    {
        ArgumentNullException.ThrowIfNull(parser);
        ArgumentNullException.ThrowIfNull(completion);
        var parts = new List<CompletionPart>(parser.Append(completion));
        parts.AddRange(parser.Complete());
        return new ParsedCompletion(Merge(parts));
    }

    private static List<CompletionPart> Merge(List<CompletionPart> parts)
    {
        var merged = new List<CompletionPart>(parts.Count);
        foreach (var part in parts)
        {
            if (part.Text is { } text && merged.Count > 0 && merged[^1].Text is { } previous)
            {
                merged[^1] = CompletionPart.FromText(previous + text);
            }
            else if (part.Text is not { Length: 0 })
            {
                merged.Add(part);
            }
        }

        return merged;
    }
}
