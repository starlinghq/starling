using System.Globalization;
using System.Text;
using System.Text.Json;
using Starlings.Generation.OnnxGenAI;

namespace Starlings.Models.Qwen35;

/// <summary>What a <see cref="Qwen35ToolGrammar"/> grammar lets the model write.</summary>
public enum Qwen35ToolGrammarMode
{
    /// <summary>Tool calls only (tool choice "required"): one call, or several with <see cref="Qwen35ToolGrammarOptions.AllowMultipleCalls"/>.</summary>
    Required,

    /// <summary>
    /// Either tool calls as in <see cref="Required"/>, or text that does not start with <c>&lt;</c> (tool choice
    /// "auto"). Built and unit-tested only; no model run has measured it.
    /// </summary>
    Auto,
}

/// <summary>Settings of <see cref="Qwen35ToolGrammar.Build(IReadOnlyList{Qwen35Tool}, Qwen35ToolGrammarOptions?)"/>. Immutable.</summary>
public sealed record Qwen35ToolGrammarOptions
{
    /// <summary>Default <see cref="MaxValueLength"/>.</summary>
    public const int DefaultMaxValueLength = 1024;

    /// <summary>Most digits the grammar admits in the integer part, and in the fraction, of a number value.</summary>
    public const int MaxNumberDigits = 20;

    /// <summary>Tool calls only (default), or tool calls or text.</summary>
    public Qwen35ToolGrammarMode Mode { get; init; } = Qwen35ToolGrammarMode.Required;

    /// <summary>False (default): one call. True: one or more calls, separated by a line feed as the template writes them.</summary>
    public bool AllowMultipleCalls { get; init; }

    /// <summary>
    /// The most characters (at least 1) of a free-text value: a parameter whose schema is not an <c>enum</c>,
    /// <c>integer</c>, <c>number</c> or <c>boolean</c>. Bounds a value the model would otherwise write until the token
    /// limit (corvid spike S5 saw one). Default <see cref="DefaultMaxValueLength"/>.
    /// </summary>
    public int MaxValueLength { get; init; } = DefaultMaxValueLength;

    /// <summary>
    /// Opt-in: fast-forward tokens (<see cref="GenerationGuidance.WithFastForward"/>), so the grammar's fixed text, such
    /// as <c>&lt;parameter=NAME&gt;</c> and the closing tags, is appended without a decode step per token. The bound per
    /// step is the grammar's UTF-8 byte length plus 1 (see the <see cref="Qwen35ToolGrammar"/> remarks). Default false:
    /// one token per step, the configuration corvid spike S5 measured.
    /// </summary>
    public bool FastForward { get; init; }

    internal void Validate() => ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MaxValueLength, nameof(MaxValueLength));
}

/// <summary>
/// Experimental: builds an llguidance Lark grammar that admits only the pinned chat template's tool-call format for the
/// given tools, so the model cannot name a tool or parameter that was not offered, leave out a required parameter, or
/// write a value outside its <c>enum</c>, <c>integer</c>, <c>number</c> or <c>boolean</c> schema:
/// <code>
/// &lt;tool_call&gt;
/// &lt;function=NAME&gt;
/// &lt;parameter=ARG&gt;
/// VALUE
/// &lt;/parameter&gt;
/// &lt;/function&gt;
/// &lt;/tool_call&gt;
/// </code>
/// Pass the result as <see cref="Qwen35ChatOptions.Guidance"/> with the same tools in <see cref="Qwen35ChatOptions.Tools"/>
/// (the prompt may offer more tools than the grammar admits), and with thinking off: the grammar describes a completion
/// that starts with the call.
/// </summary>
/// <remarks>
/// <para>
/// <b>Parameters</b> are written in the schema's <c>properties</c> order; required ones must appear, optional ones may be
/// left out. Values: an <c>enum</c> admits its members (strings as written, other members as their JSON text);
/// <c>integer</c> admits an optional minus sign and 1 to <see cref="Qwen35ToolGrammarOptions.MaxNumberDigits"/> ASCII
/// digits; <c>number</c> the same with an optional fraction of 1 to that many digits; <c>boolean</c> admits <c>true</c>,
/// <c>false</c>, <c>True</c> and <c>False</c> (<see cref="Qwen35ToolCallParser"/> reads each). Anything else (strings,
/// objects, arrays, a list of types) is free text: 0 to <see cref="Qwen35ToolGrammarOptions.MaxValueLength"/>
/// characters, none of them <c>&lt;</c>, so a value never contains a closing tag.
/// </para>
/// <para>
/// <b>Terminals.</b> Each parameter, from <c>&lt;parameter=</c> to the line feed after <c>&lt;/parameter&gt;</c>, is one
/// terminal: llguidance's lexer is greedy, so a value terminal followed by a separate closing-tag terminal would swallow
/// the value's final line feed and then fail to match the closing tag (found in corvid spike S5).
/// </para>
/// <para>
/// <b>Fast-forward bound.</b> Every byte the grammar can force comes from its literal text (tags, names, enum members).
/// Every choice point (a tool, an optional parameter, a value's first character or digit, another call or the end) ends
/// a forced run, and no rule is reached twice without passing one, so a forced run is shorter than the grammar text. A
/// token holds at least one byte, so the grammar's UTF-8 byte length plus 1 bounds the tokens of one fast-forward step.
/// </para>
/// </remarks>
public static class Qwen35ToolGrammar
{
    /// <summary>Builds the grammar as a <see cref="GenerationGuidance"/> of kind <see cref="GenerationGuidanceKind.Lark"/>.</summary>
    /// <param name="tools">The tools the model may call (at least one).</param>
    /// <param name="options">Settings; null for the <see cref="Qwen35ToolGrammarOptions"/> defaults (one required call).</param>
    /// <exception cref="ArgumentException">No tools, or a parameter name the call format cannot carry.</exception>
    public static GenerationGuidance Build(IReadOnlyList<Qwen35Tool> tools, Qwen35ToolGrammarOptions? options = null)
    {
        options ??= new Qwen35ToolGrammarOptions();
        string lark = BuildLark(tools, options);
        var guidance = GenerationGuidance.Lark(lark);
        return options.FastForward ? guidance.WithFastForward(Encoding.UTF8.GetByteCount(lark) + 1) : guidance;
    }

    /// <summary>Builds the Lark grammar text.</summary>
    /// <param name="tools">The tools the model may call (at least one).</param>
    /// <param name="options">Settings; null for the <see cref="Qwen35ToolGrammarOptions"/> defaults (one required call).</param>
    /// <exception cref="ArgumentException">No tools, or a parameter name the call format cannot carry.</exception>
    public static string BuildLark(IReadOnlyList<Qwen35Tool> tools, Qwen35ToolGrammarOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(tools);
        options ??= new Qwen35ToolGrammarOptions();
        options.Validate();
        if (tools.Count == 0)
        {
            throw new ArgumentException("At least one tool is needed.", nameof(tools));
        }

        string freeText = "[^<]{0," + options.MaxValueLength.ToString(CultureInfo.InvariantCulture) + "}";
        var rules = new StringBuilder();
        var functions = new List<string>(tools.Count);
        for (int t = 0; t < tools.Count; t++)
        {
            Qwen35Tool tool = tools[t] ?? throw new ArgumentException($"Tool {t} is null.", nameof(tools));
            string rule = "f" + t.ToString(CultureInfo.InvariantCulture);
            functions.Add(rule);
            var body = new StringBuilder();
            body.Append(Literal($"<function={tool.Name}>\n"));
            if (tool.Parameters.ValueKind == JsonValueKind.Object &&
                tool.Parameters.TryGetProperty("properties", out var properties) && properties.ValueKind == JsonValueKind.Object)
            {
                var required = new HashSet<string>(StringComparer.Ordinal);
                if (tool.Parameters.TryGetProperty("required", out var req) && req.ValueKind == JsonValueKind.Array)
                {
                    foreach (var r in req.EnumerateArray())
                    {
                        if (r.ValueKind == JsonValueKind.String)
                        {
                            required.Add(r.GetString()!);
                        }
                    }
                }

                int p = 0;
                foreach (var property in properties.EnumerateObject())
                {
                    if (property.Name.Length == 0 || property.Name.AsSpan().IndexOfAny("<>\r\n") >= 0)
                    {
                        throw new ArgumentException(
                            $"Tool '{tool.Name}': parameter name '{property.Name}' is empty or contains '<', '>' or a line break, which the tool-call format cannot carry.",
                            nameof(tools));
                    }

                    string terminal = string.Create(CultureInfo.InvariantCulture, $"P{t}_{p++}");
                    rules.Append(terminal).Append(": /")
                        .Append(RegexLiteral($"<parameter={property.Name}>\n"))
                        .Append(ValuePattern(property.Value, freeText))
                        .Append(RegexLiteral("\n</parameter>\n"))
                        .Append("/\n");
                    body.Append(' ').Append(terminal);
                    if (!required.Contains(property.Name))
                    {
                        body.Append('?');
                    }
                }
            }

            body.Append(' ').Append(Literal("</function>\n"));
            rules.Append(rule).Append(": ").Append(body).Append('\n');
        }

        var grammar = new StringBuilder();
        string calls = options.AllowMultipleCalls ? $"call ({Literal("\n")} call)*" : "call";
        grammar.Append(options.Mode == Qwen35ToolGrammarMode.Required
            ? $"start: {calls}\n"
            : $"start: {calls} | TEXT\nTEXT: /[^<](?s:.)*/\n");
        grammar.Append("call: ").Append(Literal("<tool_call>\n")).Append(" (").Append(string.Join(" | ", functions)).Append(") ")
            .Append(Literal("</tool_call>")).Append('\n');
        grammar.Append(rules);
        return grammar.ToString();
    }

    private static string ValuePattern(JsonElement schema, string freeText)
    {
        if (schema.ValueKind != JsonValueKind.Object)
        {
            return freeText;
        }

        if (schema.TryGetProperty("enum", out var values) && values.ValueKind == JsonValueKind.Array)
        {
            var members = new List<string>();
            foreach (var v in values.EnumerateArray())
            {
                members.Add(RegexLiteral(v.ValueKind == JsonValueKind.String ? v.GetString()! : v.GetRawText()));
            }

            if (members.Count > 0)
            {
                return "(" + string.Join("|", members) + ")";
            }
        }

        string digits = "[0-9]{1," + Qwen35ToolGrammarOptions.MaxNumberDigits.ToString(CultureInfo.InvariantCulture) + "}";
        string? type = schema.TryGetProperty("type", out var ty) && ty.ValueKind == JsonValueKind.String ? ty.GetString() : null;
        return type switch
        {
            "integer" => "-?" + digits,
            "number" => "-?" + digits + "(\\." + digits + ")?",
            "boolean" => "(true|false|True|False)",
            _ => freeText,
        };
    }

    /// <summary>A Lark string literal (JSON-style escapes).</summary>
    internal static string Literal(string text)
    {
        var sb = new StringBuilder(text.Length + 2).Append('"');
        foreach (char c in text)
        {
            switch (c)
            {
                case '"': sb.Append("\\\""); break;
                case '\\': sb.Append("\\\\"); break;
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                default: sb.Append(c); break;
            }
        }

        return sb.Append('"').ToString();
    }

    /// <summary>Text matched literally inside a Lark <c>/.../</c> regex (Rust regex syntax, with <c>/</c> escaped).</summary>
    internal static string RegexLiteral(string text)
    {
        var sb = new StringBuilder(text.Length + 8);
        foreach (char c in text)
        {
            switch (c)
            {
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                case '/' or '\\' or '.' or '+' or '*' or '?' or '(' or ')' or '|' or '[' or ']' or '{' or '}' or '^' or '$':
                    sb.Append('\\').Append(c);
                    break;
                default: sb.Append(c); break;
            }
        }

        return sb.ToString();
    }
}
