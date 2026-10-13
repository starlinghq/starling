namespace Starlings.Models.Gliner2;

/// <summary>A span as GLiNER2 reports it: offsets in Unicode scalar values of the normalized text.</summary>
/// <param name="Text">Upstream surface: the normalized text between the offsets, Python-stripped.</param>
/// <param name="Start">Start, Unicode scalar values of the normalized text.</param>
/// <param name="End">Exclusive end, Unicode scalar values of the normalized text (may be one past the input).</param>
/// <param name="Confidence">Score, float32 sigmoid of the logit.</param>
internal readonly record struct Gliner2UpstreamSpan(string Text, int Start, int End, float Confidence);

/// <summary>
/// GLiNER2's entity decoding for span checkpoints (<c>runtime.py:641</c>, <c>:654</c>, <c>:1147</c>,
/// <c>candidate_decoder.py:172</c>, <c>runtime.py:135</c>): count gate, float32 sigmoid, threshold, span to characters,
/// per-label confidence-first greedy suppression of overlapping spans, then the formatting's de-duplication on
/// (lower-cased text, start, end). <see cref="ToResult"/> maps the upstream spans onto the caller's text.
/// </summary>
internal static class Gliner2Decoder
{
    /// <summary>Float32 sigmoid as torch computes it for the scores GLiNER2 thresholds.</summary>
    public static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    /// <summary>
    /// Upstream entities per label (schema order), or null when the count gate is closed (upstream <c>{}</c>).
    /// </summary>
    public static List<Gliner2UpstreamSpan>[]? DecodeUpstream(Gliner2ModelInput input, Gliner2ModelOutput output, float threshold)
    {
        if (output.PredictedCount <= 0)
        {
            return null;
        }

        int words = input.WordCount;
        int width = output.MaxWidth;
        var scalars = input.NormalizedScalars;
        var starts = input.WordStarts;
        var ends = input.WordEnds;
        var logits = output.SpanLogits;
        var result = new List<Gliner2UpstreamSpan>[input.Labels.Count];
        var candidates = new List<Gliner2UpstreamSpan>();
        for (int label = 0; label < input.Labels.Count; label++)
        {
            candidates.Clear();

            // torch.where order: start word, then width.
            for (int start = 0; start < words; start++)
            {
                for (int w = 0; w < width; w++)
                {
                    int end = start + w + 1;
                    if (end > words)
                    {
                        break;
                    }

                    float score = Sigmoid(logits[((label * words) + start) * width + w]);
                    if (!(score >= threshold))
                    {
                        continue;
                    }

                    int cs = starts[start];
                    int ce = ends[end - 1];
                    var (s, e) = Gliner2Text.Strip(scalars, cs, ce);
                    if (e > s)
                    {
                        candidates.Add(new Gliner2UpstreamSpan(Gliner2Text.ToText(scalars, s, e), cs, ce, score));
                    }
                }
            }

            // sorted(reverse=True) on confidence is stable: equal scores keep candidate order.
            var ranked = candidates.OrderByDescending(c => c.Confidence).ToList();
            var kept = new List<Gliner2UpstreamSpan>();
            foreach (var candidate in ranked)
            {
                bool overlaps = false;
                foreach (var k in kept)
                {
                    if (candidate.Start < k.End && k.Start < candidate.End)
                    {
                        overlaps = true;
                        break;
                    }
                }

                if (!overlaps)
                {
                    kept.Add(candidate);
                }
            }

            // format_entity_dict with spans: de-duplicate on (text.lower(), start, end).
            var seen = new HashSet<(string, int, int)>();
            var unique = new List<Gliner2UpstreamSpan>(kept.Count);
            foreach (var span in kept)
            {
                var scalarsOfText = Gliner2Text.ToScalars(span.Text, nameof(span));
                if (seen.Add((Gliner2Text.Lower(scalarsOfText), span.Start, span.End)))
                {
                    unique.Add(span);
                }
            }

            result[label] = unique;
        }

        return result;
    }

    /// <summary>Maps upstream spans onto the caller's text (see <see cref="Gliner2Entity"/> for the rule).</summary>
    public static Gliner2ExtractionResult ToResult(Gliner2ModelInput input, List<Gliner2UpstreamSpan>[]? upstream)
    {
        var labels = input.Labels;
        if (upstream is null)
        {
            var empty = Enumerable.Repeat<IReadOnlyList<Gliner2Entity>>(Array.Empty<Gliner2Entity>(), labels.Count).ToArray();
            return new Gliner2ExtractionResult(input.Text, labels, gateOpen: false, input.TokenCount, empty);
        }

        int length = input.OriginalScalarCount;
        var scalars = input.NormalizedScalars;

        // UTF-16 index of each scalar boundary of the original text.
        var utf16 = new int[length + 1];
        for (int i = 0; i < length; i++)
        {
            utf16[i + 1] = utf16[i] + (scalars[i] > 0xFFFF ? 2 : 1);
        }

        var perLabel = new IReadOnlyList<Gliner2Entity>[labels.Count];
        for (int label = 0; label < labels.Count; label++)
        {
            var mapped = new List<Gliner2Entity>(upstream[label].Count);
            foreach (var span in upstream[label])
            {
                var (s, e) = Gliner2Text.Strip(scalars, span.Start, Math.Min(span.End, length));
                if (e <= s)
                {
                    continue; // only the appended "." (or whitespace before it)
                }

                mapped.Add(new Gliner2Entity
                {
                    Label = labels[label],
                    Text = input.Text[utf16[s]..utf16[e]],
                    Start = utf16[s],
                    End = utf16[e],
                    ScalarStart = s,
                    ScalarEnd = e,
                    Confidence = span.Confidence,
                });
            }

            perLabel[label] = mapped.AsReadOnly();
        }

        return new Gliner2ExtractionResult(input.Text, labels, gateOpen: true, input.TokenCount, perLabel);
    }
}
