using System.Text.Json;
using Starlings.Generation;

namespace Starlings.Models.Qwen35;

/// <summary>
/// A tool (function) definition for the chat template's tools block. The template writes each tool as
/// <c>tool | tojson</c>, so the definition is held as JSON and rendered with Python <c>json.dumps</c> formatting, keys in
/// their given order.
/// </summary>
public sealed class Qwen35Tool
{
    private Qwen35Tool(JsonElement json, string name, JsonElement parameters)
    {
        Json = json;
        Name = name;
        Parameters = parameters;
    }

    /// <summary>The function name.</summary>
    public string Name { get; }

    /// <summary>The JSON rendered into the tools block.</summary>
    public JsonElement Json { get; }

    /// <summary>The function's parameters JSON schema, or an undefined element when the definition has none.</summary>
    public JsonElement Parameters { get; }

    /// <summary>
    /// A function definition in the form transformers and the OpenAI API use:
    /// <c>{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}</c>, with
    /// <c>description</c> left out when null and <c>parameters</c> left out when not given.
    /// </summary>
    /// <param name="name">The function name (non-empty, no <c>&gt;</c>, <c>&lt;</c> or line break, which the call format cannot carry).</param>
    /// <param name="description">What the function does, or null.</param>
    /// <param name="parameters">The JSON schema of the arguments (usually <c>{"type": "object", "properties": ...}</c>), or null.</param>
    public static Qwen35Tool Function(string name, string? description = null, JsonElement? parameters = null)
    {
        ValidateName(name);
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
    /// A definition given as JSON, rendered exactly as given: an object with a <c>function</c> object (or, as the
    /// template's callers also accept, the function object itself) whose <c>name</c> is a string.
    /// </summary>
    /// <exception cref="ArgumentException">The JSON is not such an object.</exception>
    public static Qwen35Tool FromJson(JsonElement json)
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

        string n = name.GetString()!;
        ValidateName(n);
        JsonElement parameters = function.TryGetProperty("parameters", out var p) ? p.Clone() : default;
        return new Qwen35Tool(json.Clone(), n, parameters);
    }

    /// <summary>The definition of a backend-neutral <see cref="ToolDefinition"/>, rendered from its JSON (<see cref="FromJson"/>).</summary>
    /// <exception cref="ArgumentException">The name cannot appear in the template's call format.</exception>
    public static Qwen35Tool FromDefinition(ToolDefinition definition)
    {
        ArgumentNullException.ThrowIfNull(definition);
        return FromJson(definition.Json);
    }

    /// <summary>This definition as a backend-neutral <see cref="ToolDefinition"/> (the same JSON).</summary>
    public ToolDefinition ToDefinition() => ToolDefinition.FromJson(Json);

    /// <summary>The schema of one parameter (<c>parameters.properties[name]</c>), or an undefined element.</summary>
    internal JsonElement ParameterSchema(string parameter) =>
        Parameters.ValueKind == JsonValueKind.Object && Parameters.TryGetProperty("properties", out var properties) &&
        properties.ValueKind == JsonValueKind.Object && properties.TryGetProperty(parameter, out var schema)
            ? schema
            : default;

    internal static void ValidateName(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        if (name.Length == 0 || name.AsSpan().IndexOfAny("<>\r\n") >= 0)
        {
            throw new ArgumentException($"Tool name '{name}' is empty or contains '<', '>' or a line break, which the tool-call format cannot carry.", nameof(name));
        }
    }
}

/// <summary>A tool call of an assistant turn: the function name and its arguments.</summary>
/// <param name="Name">The function name.</param>
/// <param name="Arguments">
/// The arguments as a JSON object (property order is kept), or an undefined element (<c>default</c>) for a call
/// without arguments. String values render as written; numbers, Booleans and null as Python's <c>str()</c> of the
/// value (<c>True</c>, <c>None</c>, <c>1e-05</c>); objects and arrays as Python <c>json.dumps</c>, as the template does.
/// </param>
public sealed record Qwen35ToolCall(string Name, JsonElement Arguments)
{
    /// <summary>A call whose arguments are given as name and JSON value pairs, in order.</summary>
    public static Qwen35ToolCall Create(string name, IEnumerable<KeyValuePair<string, JsonElement>> arguments)
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
        return new Qwen35ToolCall(name, document.RootElement.Clone());
    }

    /// <summary>The call of a backend-neutral <see cref="Starlings.Generation.ToolCall"/>.</summary>
    public static Qwen35ToolCall FromToolCall(ToolCall call)
    {
        ArgumentNullException.ThrowIfNull(call);
        return new Qwen35ToolCall(call.Name, call.Arguments);
    }

    /// <summary>This call as a backend-neutral <see cref="Starlings.Generation.ToolCall"/>.</summary>
    public ToolCall ToToolCall() => new(Name, Arguments);
}
