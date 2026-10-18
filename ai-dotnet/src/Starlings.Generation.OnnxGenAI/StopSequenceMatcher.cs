using System.Text;

namespace Starlings.Generation.OnnxGenAI;

/// <summary>
/// Finds stop sequences in streamed text (ordinal comparison). Text that could still become the start of a stop
/// sequence is held back until it either completes a match or cannot, so no streamed piece ever contains part of a
/// stop sequence. This is the matcher <see cref="TextGenerator"/> applies to
/// <see cref="GenerationOptions.StopSequences"/>: push each detokenized piece (for example from
/// <see cref="IncrementalDetokenizer"/>), emit what it returns, end the request when it reports a match, and emit
/// <see cref="Flush"/> when generation ends without one.
/// </summary>
/// <remarks>
/// A match is reported as soon as any stop sequence is complete in the text pushed so far; among the matches in that
/// text the earliest wins, and at the same start the longest. Not thread-safe; one instance per request.
/// </remarks>
public sealed class StopSequenceMatcher
{
    private readonly string[] _stops;
    private readonly StringBuilder _pending = new();

    /// <summary>Creates a matcher for the stop sequences (copied; an empty list passes all text through).</summary>
    /// <param name="stops">Non-empty stop strings.</param>
    /// <exception cref="ArgumentNullException"><paramref name="stops"/> is null.</exception>
    /// <exception cref="ArgumentException">A stop sequence is null or empty.</exception>
    public StopSequenceMatcher(IReadOnlyList<string> stops)
    {
        ArgumentNullException.ThrowIfNull(stops);
        _stops = [.. stops];
        foreach (string? stop in _stops)
        {
            if (string.IsNullOrEmpty(stop))
            {
                throw new ArgumentException("Stop sequences must be non-empty strings.", nameof(stops));
            }
        }
    }

    /// <summary>Adds text; returns the text safe to emit and the stop sequence matched, if any (text after it is dropped).</summary>
    /// <exception cref="ArgumentNullException"><paramref name="text"/> is null.</exception>
    public (string Emit, string? Matched) Push(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (_stops.Length == 0)
        {
            return (text, null);
        }

        _pending.Append(text);
        string s = _pending.ToString();
        int best = -1;
        string? matched = null;
        foreach (string stop in _stops)
        {
            int index = s.IndexOf(stop, StringComparison.Ordinal);
            if (index >= 0 && (best < 0 || index < best || (index == best && stop.Length > matched!.Length)))
            {
                best = index;
                matched = stop;
            }
        }

        if (matched is not null)
        {
            _pending.Clear();
            return (s[..best], matched);
        }

        int hold = 0;
        foreach (string stop in _stops)
        {
            for (int k = Math.Min(stop.Length - 1, s.Length); k > hold; k--)
            {
                if (s.AsSpan(s.Length - k).SequenceEqual(stop.AsSpan(0, k)))
                {
                    hold = k;
                    break;
                }
            }
        }

        _pending.Clear();
        _pending.Append(s, s.Length - hold, hold);
        return (s[..^hold], null);
    }

    /// <summary>Returns the held-back text (no further text will arrive) and clears it.</summary>
    public string Flush()
    {
        string rest = _pending.ToString();
        _pending.Clear();
        return rest;
    }
}
