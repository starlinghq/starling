using System.Text.Json;
using Starlings.Generation;

namespace Starlings.Models.QwenChat;

/// <summary>
/// A tool (function) definition for the tools block of the Qwen3 and Qwen2.5 templates. The templates write each tool as
/// <c>tool | tojson</c>, so the definition is held as JSON and rendered with Python <c>json.dumps</c> formatting, keys in
/// their given order. Experimental.
/// </summary>
public sealed class QwenChatTool
{
    private QwenChatTool(JsonElement json, string name)
    {
        Json = json;
        Name = name;
    }

    /// <summary>The function name.</summary>
    public string Name { get; }

    /// <summary>The JSON rendered into the tools block.</summary>
    public JsonElement Json { get; }

    /// <summary>
    /// A function definition in the form transformers and the OpenAI API use:
    /// <c>{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}</c>, with
    /// <c>description</c> left out when null and <c>parameters</c> left out when not given.
    /// </summary>
    /// <param name="name">The function name (see <see cref="QwenChatToolCall"/> for the characters it may hold).</param>
    /// <param name="description">What the function does, or null.</param>
    /// <param name="parameters">The JSON schema of the arguments (usually <c>{"type": "object", "properties": ...}</c>), or null.</param>
    /// <exception cref="ArgumentException">The name cannot appear in the template's tool-call format.</exception>
    public static QwenChatTool Function(string name, string? description = null, JsonElement? parameters = null)
    {
        QwenChatToolCall.ValidateName(name, nameof(name));
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
    /// object itself) whose <c>name</c> is a string.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// The JSON is not such an object, or the name cannot appear in a tool call (including a name with a lone surrogate).
    /// </exception>
    public static QwenChatTool FromJson(JsonElement json)
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
            n = QwenChatPythonJson.CheckedString(name.GetString()!);
        }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException)
        {
            throw new ArgumentException("The tool name contains a lone UTF-16 surrogate.", nameof(json), ex);
        }

        QwenChatToolCall.ValidateName(n, nameof(json));
        return new QwenChatTool(json.Clone(), n);
    }

    /// <summary>The definition of a backend-neutral <see cref="ToolDefinition"/>, rendered from its JSON (<see cref="FromJson"/>).</summary>
    /// <exception cref="ArgumentException">The name cannot appear in the templates' call format.</exception>
    public static QwenChatTool FromDefinition(ToolDefinition definition)
    {
        ArgumentNullException.ThrowIfNull(definition);
        return FromJson(definition.Json);
    }

    /// <summary>This definition as a backend-neutral <see cref="ToolDefinition"/> (the same JSON).</summary>
    public ToolDefinition ToDefinition() => ToolDefinition.FromJson(Json);
}

/// <summary>
/// A tool call of an assistant turn, in the JSON form both templates use:
/// <c>&lt;tool_call&gt;\n{"name": NAME, "arguments": ARGUMENTS}\n&lt;/tool_call&gt;</c>. Experimental.
/// </summary>
/// <param name="Name">
/// The function name. The templates write it between quotes without escaping, so it must be non-empty and hold no
/// <c>"</c>, <c>\</c> or control character (U+0000 to U+001F); rendering rejects other names.
/// </param>
/// <param name="Arguments">
/// The arguments as a JSON object (property order is kept), rendered as Python <c>json.dumps</c>. Anything else (an
/// undefined element, a JSON string of arguments, an array) is rejected when rendering: upstream raises for missing
/// arguments, and the Qwen3 and Qwen2.5 templates render a string differently.
/// </param>
public sealed record QwenChatToolCall(string Name, JsonElement Arguments)
{
    /// <summary>A call whose arguments are given as name and JSON value pairs, in order.</summary>
    public static QwenChatToolCall Create(string name, IEnumerable<KeyValuePair<string, JsonElement>> arguments)
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
        return new QwenChatToolCall(name, document.RootElement.Clone());
    }

    /// <summary>The call of a backend-neutral <see cref="Starlings.Generation.ToolCall"/>.</summary>
    public static QwenChatToolCall FromToolCall(ToolCall call)
    {
        ArgumentNullException.ThrowIfNull(call);
        return new QwenChatToolCall(call.Name, call.Arguments);
    }

    /// <summary>This call as a backend-neutral <see cref="Starlings.Generation.ToolCall"/>.</summary>
    public ToolCall ToToolCall() => new(Name, Arguments);

    internal static bool IsValidName(string? name)
    {
        if (string.IsNullOrEmpty(name))
        {
            return false;
        }

        foreach (char c in name)
        {
            if (c is '"' or '\\' or < ' ')
            {
                return false;
            }
        }

        return true;
    }

    internal static void ValidateName(string name, string parameterName)
    {
        ArgumentNullException.ThrowIfNull(name, parameterName);
        if (!IsValidName(name))
        {
            throw new ArgumentException(
                $"Tool name '{name}' is empty or contains '\"', '\\' or a control character, which the template writes unescaped into the tool-call JSON.",
                parameterName);
        }
    }
}
