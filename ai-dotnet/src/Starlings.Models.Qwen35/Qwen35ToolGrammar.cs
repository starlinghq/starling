// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.

using System.Text;
using System.Text.Json;
using Starlings.Generation.OnnxGenAI;

namespace Starlings.Models.Qwen35;

/// <summary>Spike S5 (experimental): what <see cref="Qwen35ToolGrammar"/> lets the model write.</summary>
public enum Qwen35ToolGrammarMode
{
    /// <summary>Exactly one tool call (tool choice "required").</summary>
    Required,

    /// <summary>Either one tool call, or free text that does not start with <c>&lt;</c> (tool choice "auto").</summary>
    Auto,
}

/// <summary>
/// Spike S5 (experimental): builds an llguidance Lark grammar that only admits the pinned chat template's tool-call
/// format for the offered tools, so the model cannot name a tool or parameter that was not offered:
/// <code>
/// &lt;tool_call&gt;
/// &lt;function=NAME&gt;
/// &lt;parameter=ARG&gt;
/// VALUE
/// &lt;/parameter&gt;
/// &lt;/function&gt;
/// &lt;/tool_call&gt;
/// </code>
/// </summary>
/// <remarks>
/// <para>
/// Parameters are written in the schema's <c>properties</c> order; required ones must appear, optional ones may be
/// left out. Values follow the parameter's schema: <c>enum</c> (strings, numbers, booleans), <c>integer</c>,
/// <c>number</c>, <c>boolean</c>; anything else (strings, objects, arrays) is any text without <c>&lt;</c>.
/// </para>
/// <para>
/// Each parameter, from <c>&lt;parameter=</c> to the line feed after <c>&lt;/parameter&gt;</c>, is one terminal:
/// llguidance's lexer is greedy, so a value terminal followed by a separate closing-tag terminal would swallow the
/// value's final line feed and then fail to match the closing tag.
/// </para>
/// </remarks>
public static class Qwen35ToolGrammar
{
    /// <summary>Builds the grammar as <see cref="GenerationGuidance"/> (type <c>lark_grammar</c>).</summary>
    public static GenerationGuidance Build(IReadOnlyList<Qwen35Tool> tools, Qwen35ToolGrammarMode mode = Qwen35ToolGrammarMode.Required) =>
        GenerationGuidance.Lark(BuildLark(tools, mode));

    /// <summary>Builds the Lark grammar text.</summary>
    public static string BuildLark(IReadOnlyList<Qwen35Tool> tools, Qwen35ToolGrammarMode mode = Qwen35ToolGrammarMode.Required)
    {
        ArgumentNullException.ThrowIfNull(tools);
        if (tools.Count == 0)
        {
            throw new ArgumentException("At least one tool is needed.", nameof(tools));
        }

        var rules = new StringBuilder();
        var functions = new List<string>(tools.Count);
        for (int t = 0; t < tools.Count; t++)
        {
            Qwen35Tool tool = tools[t];
            string rule = $"f{t}";
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
                    string terminal = $"P{t}_{p++}";
                    rules.Append(terminal).Append(": /")
                        .Append(RegexLiteral($"<parameter={property.Name}>\n"))
                        .Append(ValuePattern(property.Value))
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
        string call = $"call: {Literal("<tool_call>\n")} ({string.Join(" | ", functions)}) {Literal("</tool_call>")}";
        grammar.Append(mode == Qwen35ToolGrammarMode.Required ? "start: call\n" : "start: call | TEXT\nTEXT: /[^<](?s:.)*/\n");
        grammar.Append(call).Append('\n');
        grammar.Append(rules);
        return grammar.ToString();
    }

    private static string ValuePattern(JsonElement schema)
    {
        if (schema.ValueKind != JsonValueKind.Object)
        {
            return "[^<]*";
        }

        if (schema.TryGetProperty("enum", out var values) && values.ValueKind == JsonValueKind.Array)
        {
            var options = new List<string>();
            foreach (var v in values.EnumerateArray())
            {
                options.Add(RegexLiteral(v.ValueKind == JsonValueKind.String ? v.GetString()! : v.GetRawText()));
            }

            if (options.Count > 0)
            {
                return "(" + string.Join("|", options) + ")";
            }
        }

        string? type = schema.TryGetProperty("type", out var ty) && ty.ValueKind == JsonValueKind.String ? ty.GetString() : null;
        return type switch
        {
            "integer" => "-?[0-9]+",
            "number" => "-?[0-9]+(\\.[0-9]+)?",
            "boolean" => "(true|false)",
            _ => "[^<]*",
        };
    }

    /// <summary>A Lark string literal (JSON-style escapes).</summary>
    private static string Literal(string text)
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

    /// <summary>Text matched literally inside a Lark <c>/.../</c> regex (Rust regex syntax).</summary>
    private static string RegexLiteral(string text)
    {
        var sb = new StringBuilder(text.Length + 8);
        foreach (char c in text)
        {
            switch (c)
            {
                case '\n': sb.Append("\\n"); break;
                case '\r': sb.Append("\\r"); break;
                case '\t': sb.Append("\\t"); break;
                case '/': sb.Append("\\/"); break;
                case '\\' or '.' or '+' or '*' or '?' or '(' or ')' or '|' or '[' or ']' or '{' or '}' or '^' or '$' or '#' or '&' or '-' or '~':
                    sb.Append('\\').Append(c);
                    break;
                default: sb.Append(c); break;
            }
        }

        return sb.ToString();
    }
}
