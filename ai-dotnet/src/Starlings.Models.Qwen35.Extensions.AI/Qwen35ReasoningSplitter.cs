using System.Text;

namespace Starlings.Models.Qwen35.Extensions.AI;

/// <summary>
/// Splits a completion made with <c>enable_thinking</c> into the reasoning and the answer at the first
/// <c>&lt;/think&gt;</c>, from a whole completion or from streamed text deltas (<see cref="Qwen35ChatClientOptions.SplitReasoning"/>).
/// </summary>
/// <remarks>
/// With thinking enabled the chat template ends the prompt with <c>&lt;think&gt;\n</c>, so the completion is the
/// reasoning, <c>&lt;/think&gt;</c> and the answer. Text before the first <c>&lt;/think&gt;</c> is reasoning and text
/// after it is the answer, both verbatim (whitespace around the tag is kept; the chat template strips it again when the
/// turn is rendered as history); the tag itself is dropped. A completion that ends before the tag (for example at the
/// token limit) is reasoning only. Streaming holds back only a suffix that could still start the tag, as the tool-call
/// parser does, so reasoning streams as it is generated; any split of a completion into deltas gives the same
/// concatenated reasoning and answer as <see cref="Split"/>. One instance per completion; not thread-safe.
/// </remarks>
internal sealed class Qwen35ReasoningSplitter
{
    private const string ThinkClose = "</think>";
    private readonly StringBuilder _held = new();
    private bool _inReasoning = true;

    /// <summary>Splits a whole completion.</summary>
    public static (string Reasoning, string Answer) Split(string completion)
    {
        ArgumentNullException.ThrowIfNull(completion);
        int close = completion.IndexOf(ThinkClose, StringComparison.Ordinal);
        return close < 0 ? (completion, string.Empty) : (completion[..close], completion[(close + ThinkClose.Length)..]);
    }

    /// <summary>Feeds the next text delta; returns the reasoning and answer text that are now final (either may be empty).</summary>
    public (string Reasoning, string Answer) Append(string delta)
    {
        ArgumentNullException.ThrowIfNull(delta);
        if (!_inReasoning)
        {
            return (string.Empty, delta);
        }

        _held.Append(delta);
        string held = _held.ToString();
        int close = held.IndexOf(ThinkClose, StringComparison.Ordinal);
        if (close >= 0)
        {
            _inReasoning = false;
            _held.Clear();
            return (held[..close], held[(close + ThinkClose.Length)..]);
        }

        int keep = PartialSuffix(held, ThinkClose);
        _held.Clear().Append(held[^keep..]);
        return (held[..^keep], string.Empty);
    }

    /// <summary>Ends the completion; returns the held-back text, which is reasoning (the tag never completed).</summary>
    public string Complete()
    {
        string rest = _held.ToString();
        _held.Clear();
        return rest;
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
}
