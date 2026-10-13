using System.Text;

namespace Starlings.Generation.OnnxGenAI;

/// <summary>
/// Finds stop sequences in streamed text (ordinal comparison). Text that could still become the start of a stop
/// sequence is held back until it either completes a match or cannot, so no streamed piece ever contains part of a
/// stop sequence. Not thread-safe; one instance per request.
/// </summary>
internal sealed class StopSequenceMatcher
{
    private readonly IReadOnlyList<string> _stops;
    private readonly StringBuilder _pending = new();

    public StopSequenceMatcher(IReadOnlyList<string> stops)
    {
        _stops = stops;
    }

    /// <summary>Adds text; returns the text safe to emit and the stop sequence matched, if any (text after it is dropped).</summary>
    public (string Emit, string? Matched) Push(string text)
    {
        if (_stops.Count == 0)
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
