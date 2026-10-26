namespace Starlings.Generation;

/// <summary>
/// Splits one completion into text and tool calls in a model's tool-call format, from streamed text deltas. One
/// instance per completion; not thread-safe. Malformed tool-call markup is returned as text, never thrown.
/// <see cref="ParsedCompletion.Parse"/> parses a whole completion.
/// </summary>
public interface IToolCallParser
{
    /// <summary>Feeds the next text delta; returns the parts that are now final.</summary>
    /// <exception cref="InvalidOperationException">The parser has completed.</exception>
    IReadOnlyList<CompletionPart> Append(string delta);

    /// <summary>Ends the completion; returns the remaining parts (an unfinished block is returned as text).</summary>
    IReadOnlyList<CompletionPart> Complete();
}
