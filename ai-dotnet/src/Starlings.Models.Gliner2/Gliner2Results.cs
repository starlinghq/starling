namespace Starlings.Models.Gliner2;

/// <summary>One extracted entity. Offsets index the caller's original text, never past its end.</summary>
/// <remarks>
/// <para>
/// GLiNER2 computes spans on its normalized text, which ends in an appended <c>"."</c> when the input does not end in
/// <c>.</c>, <c>!</c> or <c>?</c>; upstream offsets can therefore end one past the input (seen at threshold 0.0, for
/// example <c>visited Cupertino.</c> on a 26-character input). This package maps every span back onto the input: the
/// end is clamped to the input's length, trailing whitespace exposed by the clamp is trimmed, and a span that covered
/// only the appended <c>"."</c> is dropped. <see cref="Text"/> is always exactly the input between
/// <see cref="Start"/> and <see cref="End"/>. For spans that do not touch the appended <c>"."</c> (every span on the
/// reference corpus above threshold 0.0) the text and offsets equal GLiNER2's.
/// </para>
/// </remarks>
public sealed record Gliner2Entity
{
    /// <summary>The entity label, as the caller spelled it.</summary>
    public required string Label { get; init; }

    /// <summary>The entity text: <c>input[Start..End]</c>.</summary>
    public required string Text { get; init; }

    /// <summary>Start in UTF-16 code units of the input (a .NET string index).</summary>
    public required int Start { get; init; }

    /// <summary>Exclusive end in UTF-16 code units of the input; at most the input's length.</summary>
    public required int End { get; init; }

    /// <summary>Start in Unicode scalar values of the input (the unit of the reference fixtures).</summary>
    public required int ScalarStart { get; init; }

    /// <summary>Exclusive end in Unicode scalar values of the input.</summary>
    public required int ScalarEnd { get; init; }

    /// <summary>The span score, <c>sigmoid(logit)</c> in float32; many entity scores saturate to exactly 1.</summary>
    public required float Confidence { get; init; }
}

/// <summary>Entities extracted from one text for one label set.</summary>
public sealed class Gliner2ExtractionResult
{
    private readonly Dictionary<string, IReadOnlyList<Gliner2Entity>> _byLabel;

    internal Gliner2ExtractionResult(string text, IReadOnlyList<string> labels, bool gateOpen, int tokenCount, IReadOnlyList<IReadOnlyList<Gliner2Entity>> perLabel)
    {
        Text = text;
        Labels = labels;
        GateOpen = gateOpen;
        TokenCount = tokenCount;
        _byLabel = new Dictionary<string, IReadOnlyList<Gliner2Entity>>(StringComparer.Ordinal);
        var all = new List<Gliner2Entity>();
        for (int i = 0; i < labels.Count; i++)
        {
            _byLabel[labels[i]] = perLabel[i];
            all.AddRange(perLabel[i]);
        }

        Entities = all.AsReadOnly();
    }

    /// <summary>The caller's text.</summary>
    public string Text { get; }

    /// <summary>Labels in schema order (deduplicated, first occurrence kept).</summary>
    public IReadOnlyList<string> Labels { get; }

    /// <summary>
    /// GLiNER2's count gate: false when the count head predicts zero entities, in which case
    /// <see cref="Entities"/> is empty whatever the threshold (upstream returns <c>{}</c>).
    /// </summary>
    public bool GateOpen { get; }

    /// <summary>Tokens the model saw (schema prefix and words).</summary>
    public int TokenCount { get; }

    /// <summary>
    /// All entities: grouped by label in <see cref="Labels"/> order, and within a label in descending confidence
    /// (ties in start-word, then width order), as GLiNER2 orders them. Spans of different labels may overlap; spans of
    /// one label never do.
    /// </summary>
    public IReadOnlyList<Gliner2Entity> Entities { get; }

    /// <summary>The entities of one label, in result order; empty for a label without entities.</summary>
    /// <exception cref="KeyNotFoundException">The label was not requested.</exception>
    public IReadOnlyList<Gliner2Entity> GetEntities(string label)
    {
        ArgumentNullException.ThrowIfNull(label);
        return _byLabel.TryGetValue(label, out var list)
            ? list
            : throw new KeyNotFoundException($"Label '{label}' was not requested; requested: {string.Join(", ", Labels)}.");
    }
}

/// <summary>
/// Graph outputs for one input of <see cref="Gliner2EntityExtractor.RunModel"/>: count logits and the span logits over
/// this input's words (padding removed).
/// </summary>
public sealed class Gliner2ModelOutput
{
    private readonly float[] _countLogits;
    private readonly float[] _spanLogits;

    internal Gliner2ModelOutput(float[] countLogits, float[] spanLogits, int labels, int words, int maxWidth)
    {
        _countLogits = countLogits;
        _spanLogits = spanLogits;
        LabelCount = labels;
        WordCount = words;
        MaxWidth = maxWidth;
    }

    /// <summary>Count-head logits (20 values); the argmax is the predicted entity count, 0 closes the gate.</summary>
    public ReadOnlySpan<float> CountLogits => _countLogits;

    /// <summary>Span logits, row-major <c>[labels, words, max width]</c>; spans running past the last word are not valid.</summary>
    public ReadOnlySpan<float> SpanLogits => _spanLogits;

    /// <summary>Number of labels.</summary>
    public int LabelCount { get; }

    /// <summary>Number of words.</summary>
    public int WordCount { get; }

    /// <summary>Maximum span width in words (8).</summary>
    public int MaxWidth { get; }

    /// <summary>Predicted entity count: the first index of the largest count logit (as <c>argmax</c>).</summary>
    public int PredictedCount
    {
        get
        {
            int best = 0;
            for (int i = 1; i < _countLogits.Length; i++)
            {
                if (_countLogits[i] > _countLogits[best])
                {
                    best = i;
                }
            }

            return best;
        }
    }

    /// <summary>
    /// The logits of valid spans only (start word plus width within the text), in label, start word, width order:
    /// the reference fixtures' <c>span_logits_valid</c>.
    /// </summary>
    public float[] GetValidSpanLogits()
    {
        var valid = new List<float>(LabelCount * WordCount * MaxWidth);
        for (int label = 0; label < LabelCount; label++)
        {
            for (int start = 0; start < WordCount; start++)
            {
                for (int width = 0; width < MaxWidth && start + width < WordCount; width++)
                {
                    valid.Add(_spanLogits[((label * WordCount) + start) * MaxWidth + width]);
                }
            }
        }

        return valid.ToArray();
    }
}
