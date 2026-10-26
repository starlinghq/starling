using System.Text;
using System.Text.Json;

namespace Starlings.Models.QwenChat;

/// <summary>One piece of an assistant completion: text, or a parsed tool call.</summary>
/// <param name="Text">Text, when this part is text.</param>
/// <param name="ToolCall">The call, when this part is a tool call.</param>
public sealed record QwenChatOutputPart(string? Text, QwenChatToolCall? ToolCall)
{
    /// <summary>A text part.</summary>
    public static QwenChatOutputPart FromText(string text) => new(text, null);

    /// <summary>A tool-call part.</summary>
    public static QwenChatOutputPart FromToolCall(QwenChatToolCall call) => new(null, call);
}

/// <summary>A completion split into text and tool calls by <see cref="QwenChatToolCallParser"/>.</summary>
/// <param name="Parts">Text and tool-call parts in output order; adjacent text is merged.</param>
public sealed record QwenChatParsedOutput(IReadOnlyList<QwenChatOutputPart> Parts)
{
    /// <summary>All text parts, concatenated.</summary>
    public string Text => string.Concat(Parts.Where(p => p.Text is not null).Select(p => p.Text));

    /// <summary>The tool calls, in order.</summary>
    public IReadOnlyList<QwenChatToolCall> ToolCalls => [.. Parts.Where(p => p.ToolCall is not null).Select(p => p.ToolCall!)];

    /// <summary>The completion as an assistant message for the next request: <see cref="Text"/> and <see cref="ToolCalls"/>.</summary>
    public QwenChatMessage ToAssistantMessage() =>
        new(QwenChatRole.Assistant, Text) { ToolCalls = ToolCalls is { Count: > 0 } calls ? calls : null };
}

/// <summary>
/// Parses the tool-call format the Qwen3 and Qwen2.5 templates instruct the model to use, from a whole completion or
/// from streamed text deltas:
/// <code>
/// &lt;tool_call&gt;
/// {"name": NAME, "arguments": {...}}
/// &lt;/tool_call&gt;
/// </code>
/// Experimental.
/// </summary>
/// <remarks>
/// <para>
/// <b>Blocks.</b> A block is <c>&lt;tool_call&gt;</c> to the next <c>&lt;/tool_call&gt;</c> outside a JSON string, so an
/// argument value may contain the tag (a string left open keeps the block open until the completion ends). Its body, without
/// surrounding JSON whitespace, must be one strict JSON object (RFC 8259: no comments, trailing commas, <c>NaN</c> or
/// single quotes) with exactly the members <c>name</c>, a string the templates can render back (see
/// <see cref="QwenChatToolCall"/>), and <c>arguments</c>, a JSON object. No object in it may repeat a key, and no
/// string may hold a lone surrogate. The arguments keep their JSON types as written; nothing is cast.
/// </para>
/// <para>
/// <b>Malformed output never throws.</b> A block that does not parse, or one still open when the completion ends, is
/// returned as text, verbatim. Whitespace-only text directly before a tool call, between calls and after the last call
/// is dropped (the templates write their own separators); all other text is kept verbatim.
/// </para>
/// <para>
/// <b>Streaming.</b> <see cref="Append"/> returns text as soon as it cannot be the start of a tool call, and each tool
/// call once its block is complete; raw tool-call markup is never returned as text unless it is malformed. The parts of
/// any split of a completion into deltas, with adjacent text merged, equal <see cref="Parse(string)"/> of the whole
/// completion.
/// </para>
/// <para>
/// The Qwen3 profile runs with thinking disabled, so the completion holds no reasoning and nothing is skipped.
/// The Qwen3.5 package's parser reads a different (XML parameter) form and is not used here.
/// </para>
/// </remarks>
public sealed class QwenChatToolCallParser
{
    private const string CallOpen = "<tool_call>";
    private const string CallClose = "</tool_call>";

    private static readonly JsonDocumentOptions StrictJson = new()
    {
        AllowTrailingCommas = false,
        CommentHandling = JsonCommentHandling.Disallow,
    };

    private readonly StringBuilder _pending = new();
    private string _heldWhitespace = string.Empty;
    private bool _lastWasCall;
    private bool _completed;

    /// <summary>Parses a whole completion.</summary>
    public static QwenChatParsedOutput Parse(string completion)
    {
        ArgumentNullException.ThrowIfNull(completion);
        var parser = new QwenChatToolCallParser();
        var parts = new List<QwenChatOutputPart>(parser.Append(completion));
        parts.AddRange(parser.Complete());
        return new QwenChatParsedOutput(Merge(parts));
    }

    /// <summary>Feeds the next text delta; returns the parts that are now final.</summary>
    /// <exception cref="InvalidOperationException">The parser has completed.</exception>
    public IReadOnlyList<QwenChatOutputPart> Append(string delta)
    {
        ArgumentNullException.ThrowIfNull(delta);
        if (_completed)
        {
            throw new InvalidOperationException("The parser has completed.");
        }

        _pending.Append(delta);
        var parts = new List<QwenChatOutputPart>();
        Process(parts, final: false);
        return Merge(parts);
    }

    /// <summary>Ends the completion; returns the remaining parts (an unfinished block is returned as text).</summary>
    public IReadOnlyList<QwenChatOutputPart> Complete()
    {
        if (_completed)
        {
            return [];
        }

        var parts = new List<QwenChatOutputPart>();
        Process(parts, final: true);
        if (_pending.Length > 0)
        {
            EmitText(parts, _pending.ToString());
            _pending.Clear();
        }

        if (!_lastWasCall && _heldWhitespace.Length > 0)
        {
            parts.Add(QwenChatOutputPart.FromText(_heldWhitespace));
        }

        _heldWhitespace = string.Empty;
        _completed = true;
        return Merge(parts);
    }

    private void Process(List<QwenChatOutputPart> parts, bool final)
    {
        while (_pending.Length > 0)
        {
            string pending = _pending.ToString();
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

            int end = FindBlockEnd(pending);
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
                parts.Add(QwenChatOutputPart.FromToolCall(call));
                _lastWasCall = true;
            }
            else
            {
                EmitText(parts, block);
            }
        }
    }

    // The first </tool_call> after <tool_call> outside a JSON string (a quote toggles, a backslash escapes the next
    // character), or -1 while the block is open.
    private static int FindBlockEnd(string pending)
    {
        bool inString = false;
        for (int i = CallOpen.Length; i < pending.Length; i++)
        {
            char c = pending[i];
            if (inString)
            {
                if (c == '\\')
                {
                    i++;
                }
                else if (c == '"')
                {
                    inString = false;
                }
            }
            else if (c == '"')
            {
                inString = true;
            }
            else if (c == '<' && pending.AsSpan(i).StartsWith(CallClose, StringComparison.Ordinal))
            {
                return i;
            }
        }

        return -1;
    }

    // Whitespace is held until it is known not to separate tool calls.
    private void EmitText(List<QwenChatOutputPart> parts, string text)
    {
        if (text.Length == 0)
        {
            return;
        }

        int end = text.Length;
        while (end > 0 && char.IsWhiteSpace(text[end - 1]))
        {
            end--;
        }

        if (end == 0)
        {
            _heldWhitespace += text;
            return;
        }

        parts.Add(QwenChatOutputPart.FromText(_heldWhitespace + text[..end]));
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

    private static List<QwenChatOutputPart> Merge(List<QwenChatOutputPart> parts)
    {
        var merged = new List<QwenChatOutputPart>(parts.Count);
        foreach (var part in parts)
        {
            if (part.Text is { } text && merged.Count > 0 && merged[^1].Text is { } previous)
            {
                merged[^1] = QwenChatOutputPart.FromText(previous + text);
            }
            else if (part.Text is not { Length: 0 })
            {
                merged.Add(part);
            }
        }

        return merged;
    }

    private static bool TryParseBlock(string body, out QwenChatToolCall call)
    {
        call = null!;
        string json = body.Trim(' ', '\t', '\n', '\r');
        if (json.Length == 0 || json[0] != '{')
        {
            return false;
        }

        JsonDocument document;
        try
        {
            document = JsonDocument.Parse(json, StrictJson);
        }
        catch (JsonException)
        {
            return false;
        }

        using (document)
        {
            JsonElement root = document.RootElement;
            bool wellFormed;
            try
            {
                wellFormed = IsWellFormed(root);
            }
            catch (InvalidOperationException)
            {
                // A property name that is not valid UTF-16.
                wellFormed = false;
            }

            if (root.ValueKind != JsonValueKind.Object || !wellFormed)
            {
                return false;
            }

            string? name = null;
            JsonElement arguments = default;
            int count = 0;
            foreach (var property in root.EnumerateObject())
            {
                count++;
                if (property.NameEquals("name") && property.Value.ValueKind == JsonValueKind.String)
                {
                    name = property.Value.GetString();
                }
                else if (property.NameEquals("arguments") && property.Value.ValueKind == JsonValueKind.Object)
                {
                    arguments = property.Value;
                }
                else
                {
                    return false;
                }
            }

            if (count != 2 || !QwenChatToolCall.IsValidName(name) || arguments.ValueKind != JsonValueKind.Object)
            {
                return false;
            }

            call = new QwenChatToolCall(name!, arguments.Clone());
            return true;
        }
    }

    // No repeated keys (Python keeps the last one silently) and no lone surrogates (no UTF-8 form).
    private static bool IsWellFormed(JsonElement value)
    {
        switch (value.ValueKind)
        {
            case JsonValueKind.Object:
                var seen = new HashSet<string>(StringComparer.Ordinal);
                foreach (var property in value.EnumerateObject())
                {
                    if (!IsValidString(property.Name) || !seen.Add(property.Name) || !IsWellFormed(property.Value))
                    {
                        return false;
                    }
                }

                return true;
            case JsonValueKind.Array:
                foreach (var item in value.EnumerateArray())
                {
                    if (!IsWellFormed(item))
                    {
                        return false;
                    }
                }

                return true;
            case JsonValueKind.String:
                try
                {
                    return IsValidString(value.GetString()!);
                }
                catch (InvalidOperationException)
                {
                    return false;
                }

            default:
                return true;
        }
    }

    private static bool IsValidString(string s)
    {
        try
        {
            QwenChatPythonJson.CheckedString(s);
            return true;
        }
        catch (NotSupportedException)
        {
            return false;
        }
    }
}
