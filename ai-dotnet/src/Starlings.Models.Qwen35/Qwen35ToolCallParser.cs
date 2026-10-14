// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.
//
// Argument casting (CastArgument, SchemaTypes) ported from transformers v5.18.0 (commit
// a906d3c4b65095f2308b6a6a193e934d03b8eb5d), src/transformers/utils/chat_parsing/response_parser.py (_coerce,
// _schema_types), Copyright 2026 The HuggingFace Team, Apache License 2.0. Modified by StarlingHq: C# over
// System.Text.Json; ASCII digits only.

using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Starlings.Models.Qwen35;

/// <summary>One piece of an assistant completion: text, or a parsed tool call.</summary>
/// <param name="Text">Text, when this part is text.</param>
/// <param name="ToolCall">The call, when this part is a tool call.</param>
public sealed record Qwen35OutputPart(string? Text, Qwen35ToolCall? ToolCall)
{
    /// <summary>A text part.</summary>
    public static Qwen35OutputPart FromText(string text) => new(text, null);

    /// <summary>A tool-call part.</summary>
    public static Qwen35OutputPart FromToolCall(Qwen35ToolCall call) => new(null, call);
}

/// <summary>A completion split into text and tool calls by <see cref="Qwen35ToolCallParser"/>.</summary>
/// <param name="Parts">Text and tool-call parts in output order; adjacent text is merged.</param>
public sealed record Qwen35ParsedOutput(IReadOnlyList<Qwen35OutputPart> Parts)
{
    /// <summary>All text parts, concatenated.</summary>
    public string Text => string.Concat(Parts.Where(p => p.Text is not null).Select(p => p.Text));

    /// <summary>The tool calls, in order.</summary>
    public IReadOnlyList<Qwen35ToolCall> ToolCalls => [.. Parts.Where(p => p.ToolCall is not null).Select(p => p.ToolCall!)];

    /// <summary>The completion as an assistant message for the next request: <see cref="Text"/> and <see cref="ToolCalls"/>.</summary>
    public Qwen35Message ToAssistantMessage() => new(Qwen35Role.Assistant, Text) { ToolCalls = ToolCalls is { Count: > 0 } calls ? calls : null };
}

/// <summary>
/// Parses the tool-call format the pinned chat template instructs the model to use, from a whole completion or from
/// streamed text deltas:
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
/// <b>Blocks.</b> A block is <c>&lt;tool_call&gt;</c> to the next <c>&lt;/tool_call&gt;</c> holding exactly one
/// <c>&lt;function=NAME&gt;...&lt;/function&gt;</c> with zero or more distinct <c>&lt;parameter=ARG&gt;VALUE&lt;/parameter&gt;</c>;
/// whitespace between the tags is allowed. A value is the text between its tags without one leading and one trailing
/// line feed (the template writes exactly those). With thinking enabled, nothing before <c>&lt;/think&gt;</c> is parsed.
/// </para>
/// <para>
/// <b>Arguments.</b> Values are cast with the declared tool's JSON schema as transformers' response parser does
/// (<c>_coerce</c>): for each type of the parameter's <c>type</c>, <c>anyOf</c> and <c>oneOf</c> (and <c>nullable</c>)
/// in order, <c>integer</c>, <c>number</c>, <c>boolean</c> (<c>true</c>/<c>1</c>/<c>false</c>/<c>0</c>, any case),
/// <c>null</c> (<c>null</c> or <c>None</c>), <c>object</c> and <c>array</c> (JSON) are tried; the first that succeeds wins,
/// otherwise the value stays a string. Undeclared tools and parameters keep string values. Digits are ASCII only.
/// </para>
/// <para>
/// <b>Malformed output never throws.</b> A block that does not parse, or one still open when the completion ends,
/// is returned as text, verbatim. Whitespace-only text directly before a tool call, between calls and after the last
/// call is dropped (the template trims content and writes its own separators); all other text is kept verbatim.
/// </para>
/// <para>
/// <b>Streaming.</b> <see cref="Append"/> returns text as soon as it cannot be the start of a tool call (or of
/// <c>&lt;/think&gt;</c> while thinking), and each tool call once its block is complete; raw tool-call markup is never
/// returned as text unless it is malformed. The parts of any split of a completion into deltas, with adjacent text
/// merged, equal <see cref="Parse(string, IReadOnlyList{Qwen35Tool}, bool)"/> of the whole completion.
/// </para>
/// </remarks>
public sealed partial class Qwen35ToolCallParser
{
    private const string CallOpen = "<tool_call>";
    private const string CallClose = "</tool_call>";
    private const string ThinkClose = "</think>";

    private readonly Dictionary<string, Qwen35Tool> _tools = new(StringComparer.Ordinal);
    private readonly StringBuilder _pending = new();
    private string _heldWhitespace = string.Empty;
    private bool _inReasoning;
    private bool _lastWasCall;
    private bool _completed;

    /// <summary>A parser for one completion.</summary>
    /// <param name="tools">The tools of the request (their schemas cast argument values); may be empty.</param>
    /// <param name="enableThinking">The request's <c>enable_thinking</c>: the completion starts with reasoning, which is not parsed.</param>
    public Qwen35ToolCallParser(IReadOnlyList<Qwen35Tool> tools, bool enableThinking)
    {
        ArgumentNullException.ThrowIfNull(tools);
        foreach (var tool in tools)
        {
            // transformers keeps the last definition of a name.
            _tools[tool.Name] = tool;
        }

        _inReasoning = enableThinking;
    }

    /// <summary>Parses a whole completion.</summary>
    public static Qwen35ParsedOutput Parse(string completion, IReadOnlyList<Qwen35Tool> tools, bool enableThinking)
    {
        ArgumentNullException.ThrowIfNull(completion);
        var parser = new Qwen35ToolCallParser(tools, enableThinking);
        var parts = new List<Qwen35OutputPart>(parser.Append(completion));
        parts.AddRange(parser.Complete());
        return new Qwen35ParsedOutput(Merge(parts));
    }

    /// <summary>Parses a whole completion of a request made with <paramref name="options"/> (its tools and thinking switch).</summary>
    public static Qwen35ParsedOutput Parse(string completion, Qwen35ChatOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        return Parse(completion, options.Tools, options.EnableThinking);
    }

    /// <summary>Feeds the next text delta; returns the parts that are now final.</summary>
    public IReadOnlyList<Qwen35OutputPart> Append(string delta)
    {
        ArgumentNullException.ThrowIfNull(delta);
        if (_completed)
        {
            throw new InvalidOperationException("The parser has completed.");
        }

        _pending.Append(delta);
        var parts = new List<Qwen35OutputPart>();
        Process(parts, final: false);
        return Merge(parts);
    }

    /// <summary>Ends the completion; returns the remaining parts (an unfinished block is returned as text).</summary>
    public IReadOnlyList<Qwen35OutputPart> Complete()
    {
        if (_completed)
        {
            return [];
        }

        var parts = new List<Qwen35OutputPart>();
        Process(parts, final: true);
        if (_pending.Length > 0)
        {
            EmitText(parts, _pending.ToString());
            _pending.Clear();
        }

        if (!_lastWasCall && _heldWhitespace.Length > 0)
        {
            parts.Add(Qwen35OutputPart.FromText(_heldWhitespace));
        }

        _heldWhitespace = string.Empty;
        _completed = true;
        return Merge(parts);
    }

    private void Process(List<Qwen35OutputPart> parts, bool final)
    {
        while (_pending.Length > 0)
        {
            string pending = _pending.ToString();
            if (_inReasoning)
            {
                int close = pending.IndexOf(ThinkClose, StringComparison.Ordinal);
                if (close < 0)
                {
                    int keep = final ? 0 : PartialSuffix(pending, ThinkClose);
                    if (pending.Length - keep > 0)
                    {
                        parts.Add(Qwen35OutputPart.FromText(pending[..^keep]));
                    }

                    _pending.Clear().Append(pending[^keep..]);
                    return;
                }

                parts.Add(Qwen35OutputPart.FromText(pending[..(close + ThinkClose.Length)]));
                _pending.Remove(0, close + ThinkClose.Length);
                _inReasoning = false;
                continue;
            }

            int open = pending.IndexOf(CallOpen, StringComparison.Ordinal);
            if (open < 0)
            {
                int keep = final ? 0 : PartialSuffix(pending, CallOpen);
                EmitText(parts, pending[..^keep]);
                _pending.Clear().Append(pending[^keep..]);
                return;
            }

            if (open > 0)
            {
                EmitText(parts, pending[..open]);
                _pending.Remove(0, open);
                continue;
            }

            int end = pending.IndexOf(CallClose, CallOpen.Length, StringComparison.Ordinal);
            if (end < 0)
            {
                // An open block: wait for its end (Complete returns it as text).
                return;
            }

            string block = pending[..(end + CallClose.Length)];
            _pending.Remove(0, block.Length);
            if (TryParseBlock(block[CallOpen.Length..end], out var call))
            {
                _heldWhitespace = string.Empty;
                parts.Add(Qwen35OutputPart.FromToolCall(call));
                _lastWasCall = true;
            }
            else
            {
                EmitText(parts, block);
            }
        }
    }

    // Text after the reasoning: whitespace is held until it is known not to separate tool calls.
    private void EmitText(List<Qwen35OutputPart> parts, string text)
    {
        if (text.Length == 0)
        {
            return;
        }

        int end = text.Length;
        while (end > 0 && Qwen35ChatTemplate.IsPythonSpace(text[end - 1]))
        {
            end--;
        }

        if (end == 0)
        {
            _heldWhitespace += text;
            return;
        }

        parts.Add(Qwen35OutputPart.FromText(_heldWhitespace + text[..end]));
        _heldWhitespace = text[end..];
        _lastWasCall = false;
    }

    // Length of the longest proper suffix of text that is a prefix of marker.
    private static int PartialSuffix(string text, string marker)
    {
        for (int n = Math.Min(text.Length, marker.Length - 1); n > 0; n--)
        {
            if (text.AsSpan(text.Length - n).SequenceEqual(marker.AsSpan(0, n)))
            {
                return n;
            }
        }

        return 0;
    }

    private static List<Qwen35OutputPart> Merge(List<Qwen35OutputPart> parts)
    {
        var merged = new List<Qwen35OutputPart>(parts.Count);
        foreach (var part in parts)
        {
            if (part.Text is { } text && merged.Count > 0 && merged[^1].Text is { } previous)
            {
                merged[^1] = Qwen35OutputPart.FromText(previous + text);
            }
            else if (part.Text is not { Length: 0 })
            {
                merged.Add(part);
            }
        }

        return merged;
    }

    private bool TryParseBlock(string body, out Qwen35ToolCall call)
    {
        call = null!;
        int pos = SkipSpace(body, 0);
        const string FunctionOpen = "<function=";
        if (!body.AsSpan(pos).StartsWith(FunctionOpen, StringComparison.Ordinal))
        {
            return false;
        }

        pos += FunctionOpen.Length;
        int nameEnd = body.IndexOf('>', pos);
        if (nameEnd < 0)
        {
            return false;
        }

        string name = body[pos..nameEnd];
        if (name.Length == 0 || name.AsSpan().IndexOfAny("<\r\n") >= 0)
        {
            return false;
        }

        pos = nameEnd + 1;
        _tools.TryGetValue(name, out var tool);
        var arguments = new List<KeyValuePair<string, JsonElement>>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        while (true)
        {
            pos = SkipSpace(body, pos);
            const string ParameterOpen = "<parameter=";
            const string ParameterClose = "</parameter>";
            const string FunctionClose = "</function>";
            if (body.AsSpan(pos).StartsWith(FunctionClose, StringComparison.Ordinal))
            {
                pos = SkipSpace(body, pos + FunctionClose.Length);
                if (pos != body.Length)
                {
                    return false;
                }

                break;
            }

            if (!body.AsSpan(pos).StartsWith(ParameterOpen, StringComparison.Ordinal))
            {
                return false;
            }

            pos += ParameterOpen.Length;
            int keyEnd = body.IndexOf('>', pos);
            if (keyEnd < 0)
            {
                return false;
            }

            string key = body[pos..keyEnd];
            if (key.Length == 0 || key.AsSpan().IndexOfAny("<\r\n") >= 0 || !seen.Add(key))
            {
                return false;
            }

            int valueEnd = body.IndexOf(ParameterClose, keyEnd + 1, StringComparison.Ordinal);
            if (valueEnd < 0)
            {
                return false;
            }

            string value = body[(keyEnd + 1)..valueEnd];
            if (value.StartsWith('\n'))
            {
                value = value[1..];
            }

            if (value.EndsWith('\n'))
            {
                value = value[..^1];
            }

            if (!IsValidUtf16(value))
            {
                return false;
            }

            JsonElement schema = tool is null ? default : tool.ParameterSchema(key);
            arguments.Add(new(key, CastArgument(value, SchemaTypes(schema))));
            pos = valueEnd + ParameterClose.Length;
        }

        call = Qwen35ToolCall.Create(name, arguments);
        return true;
    }

    private static int SkipSpace(string s, int pos)
    {
        while (pos < s.Length && Qwen35ChatTemplate.IsPythonSpace(s[pos]))
        {
            pos++;
        }

        return pos;
    }

    private static bool IsValidUtf16(string s)
    {
        for (int i = 0; i < s.Length; i++)
        {
            if (char.IsHighSurrogate(s[i]) && i + 1 < s.Length && char.IsLowSurrogate(s[i + 1]))
            {
                i++;
            }
            else if (char.IsSurrogate(s[i]))
            {
                return false;
            }
        }

        return true;
    }

    // transformers _schema_types: "type" (a string or a list), then anyOf and oneOf recursively, then "nullable".
    internal static List<string> SchemaTypes(JsonElement schema)
    {
        var types = new List<string>();
        if (schema.ValueKind != JsonValueKind.Object)
        {
            return types;
        }

        if (schema.TryGetProperty("type", out var declared))
        {
            if (declared.ValueKind == JsonValueKind.String)
            {
                types.Add(declared.GetString()!);
            }
            else if (declared.ValueKind == JsonValueKind.Array)
            {
                types.AddRange(declared.EnumerateArray().Where(t => t.ValueKind == JsonValueKind.String).Select(t => t.GetString()!));
            }
        }

        foreach (string union in (ReadOnlySpan<string>)["anyOf", "oneOf"])
        {
            if (schema.TryGetProperty(union, out var choices) && choices.ValueKind == JsonValueKind.Array)
            {
                foreach (var choice in choices.EnumerateArray())
                {
                    types.AddRange(SchemaTypes(choice));
                }
            }
        }

        if (schema.TryGetProperty("nullable", out var nullable) && IsTruthy(nullable) && !types.Contains("null"))
        {
            types.Add("null");
        }

        return types;
    }

    // transformers _coerce: the first declared type the raw text casts to, else the raw text.
    internal static JsonElement CastArgument(string raw, IReadOnlyList<string> types)
    {
        foreach (string type in types)
        {
            switch (type)
            {
                case "integer" when TryPythonInt(raw, out var integer):
                    return Json(integer.ToString(CultureInfo.InvariantCulture));
                case "number" when TryPythonFloat(raw, out double number) && double.IsFinite(number):
                    // Preserve ints when the source text had no fractional part.
                    return Json(double.IsInteger(number) && !raw.Contains('.', StringComparison.Ordinal)
                        ? new BigInteger(number).ToString(CultureInfo.InvariantCulture)
                        : Qwen35PythonJson.FloatRepr(number));
                case "boolean":
                    {
                        string b = Qwen35ChatTemplate.PyStrip(raw).ToLowerInvariant();
                        if (b is "true" or "1")
                        {
                            return Json("true");
                        }

                        if (b is "false" or "0")
                        {
                            return Json("false");
                        }

                        break;
                    }

                case "null" when Qwen35ChatTemplate.PyStrip(raw) is "null" or "None":
                    return Json("null");
                case "object" or "array":
                    try
                    {
                        using var document = JsonDocument.Parse(raw);
                        if (document.RootElement.ValueKind == (type == "object" ? JsonValueKind.Object : JsonValueKind.Array))
                        {
                            return document.RootElement.Clone();
                        }
                    }
                    catch (JsonException)
                    {
                    }

                    break;
            }
        }

        using var stream = new MemoryStream();
        using (var writer = new Utf8JsonWriter(stream))
        {
            writer.WriteStringValue(raw);
        }

        using var stringDocument = JsonDocument.Parse(stream.ToArray());
        return stringDocument.RootElement.Clone();
    }

    private static JsonElement Json(string text)
    {
        using var document = JsonDocument.Parse(text);
        return document.RootElement.Clone();
    }

    private static bool IsTruthy(JsonElement e) => e.ValueKind switch
    {
        JsonValueKind.True => true,
        JsonValueKind.String => e.GetString()!.Length > 0,
        JsonValueKind.Number => e.GetDouble() != 0,
        JsonValueKind.Object => e.EnumerateObject().Any(),
        JsonValueKind.Array => e.GetArrayLength() > 0,
        _ => false,
    };

    // Python int(str): surrounding whitespace, a sign, ASCII digits with single underscores between them.
    private static bool TryPythonInt(string raw, out BigInteger value)
    {
        value = default;
        string s = Qwen35ChatTemplate.PyStrip(raw);
        if (!PythonIntPattern().IsMatch(s))
        {
            return false;
        }

        value = BigInteger.Parse(s.Replace("_", string.Empty, StringComparison.Ordinal), NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture);
        return true;
    }

    // Python float(str) for finite decimal text (inf and nan are rejected by the caller anyway).
    private static bool TryPythonFloat(string raw, out double value)
    {
        value = default;
        string s = Qwen35ChatTemplate.PyStrip(raw);
        if (!PythonFloatPattern().IsMatch(s))
        {
            return false;
        }

        return double.TryParse(s.Replace("_", string.Empty, StringComparison.Ordinal), NumberStyles.Float, CultureInfo.InvariantCulture, out value);
    }

    [GeneratedRegex(@"\A[+-]?[0-9](?:_?[0-9])*\z", RegexOptions.CultureInvariant)]
    private static partial Regex PythonIntPattern();

    [GeneratedRegex(@"\A[+-]?(?:[0-9](?:_?[0-9])*(?:\.(?:[0-9](?:_?[0-9])*)?)?|\.[0-9](?:_?[0-9])*)(?:[eE][+-]?[0-9](?:_?[0-9])*)?\z", RegexOptions.CultureInvariant)]
    private static partial Regex PythonFloatPattern();
}
