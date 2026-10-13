using Starlings.TestSupport;

namespace Starlings.Models.Gliner2.Tests;

/// <summary>
/// GLiNER2's span decoding without a model: the reference logits of every fixture case decode to the reference output,
/// and synthetic logits pin the gate, the threshold, suppression, order and the mapping onto the original text.
/// </summary>
public class DecoderTests(ITestOutputHelper output)
{
    private const int Width = 8;

    /// <summary>A model input from the splitter alone (token ids are irrelevant to decoding).</summary>
    internal static Gliner2ModelInput InputFor(string text, params string[] labels)
    {
        string normalized = Gliner2Text.Normalize(text);
        int[] scalars = Gliner2Text.ToScalars(normalized, nameof(text));
        var words = Gliner2Text.SplitWords(scalars);
        return new Gliner2ModelInput(
            text, normalized, scalars, Gliner2Text.ToScalars(text, nameof(text)).Length, labels, [], words.Select(w => w.Word).ToList(),
            words.Select(w => w.Start).ToArray(), words.Select(w => w.End).ToArray(), [], [], []);
    }

    internal static Gliner2ModelOutput OutputFor(Gliner2ModelInput input, Action<float[], Func<int, int, int, int>> set, int predictedCount = 1)
    {
        int n = input.WordCount;
        var spans = new float[input.Labels.Count * n * Width];
        Array.Fill(spans, -30f);
        set(spans, (label, start, width) => ((label * n) + start) * Width + width);
        var count = new float[20];
        count[predictedCount] = 5f;
        return new Gliner2ModelOutput(count, spans, input.Labels.Count, n, Width);
    }

    private static Gliner2ExtractionResult Decode(Gliner2ModelInput input, Gliner2ModelOutput output, float threshold = 0.5f) =>
        Gliner2Decoder.ToResult(input, Gliner2Decoder.DecodeUpstream(input, output, threshold));

    [Fact]
    public void ReferenceLogitsDecodeToTheReferenceOutput()
    {
        var (path, reason) = Gliner2TestAssets.FindExtractionFixtures();
        Assert.SkipWhen(path is null, reason);
        int cases = 0;
        foreach (var c in Gliner2Fixtures.LoadCases(path!))
        {
            var scalars = Gliner2Text.ToScalars(c.NormalizedText, "text");
            var input = new Gliner2ModelInput(
                c.Text, c.NormalizedText, scalars, Gliner2Text.ToScalars(c.Text, "text").Length, c.Labels, c.SchemaTokens, c.Words,
                c.StartMap, c.EndMap, c.InputIds, c.WordIndex, c.SchemaIndex);
            int n = c.Words.Count;
            var spans = new float[c.Labels.Count * n * Width];
            Array.Fill(spans, float.NegativeInfinity);
            int k = 0;
            for (int label = 0; label < c.Labels.Count; label++)
            {
                for (int start = 0; start < n; start++)
                {
                    for (int w = 0; w < Width && start + w < n; w++)
                    {
                        spans[((label * n) + start) * Width + w] = c.SpanLogitsValid[k++];
                    }
                }
            }

            Assert.Equal(c.SpanLogitsValid.Length, k);
            var modelOutput = new Gliner2ModelOutput(c.CountLogits, spans, c.Labels.Count, n, Width);
            Assert.Equal(c.GateOpen, modelOutput.PredictedCount > 0);
            var actual = Gliner2ParityChecks.UpstreamEntities(input, modelOutput, c.Threshold);
            Assert.True(c.Entities.Select(e => e.Key).SequenceEqual(actual.Select(e => e.Key)), $"{c.Id}: decoded entities differ");
            Assert.Equal(c.Entities.Select(e => (float)e.Confidence), actual.Select(e => (float)e.Confidence));

            var mapped = Gliner2ParityChecks.MapReferenceToOriginal(c);
            var result = Gliner2Decoder.ToResult(input, Gliner2Decoder.DecodeUpstream(input, modelOutput, c.Threshold));
            Assert.Equal(mapped, result.Entities.Select(e => (e.Label, e.Text, e.ScalarStart, e.ScalarEnd)));
            cases++;
        }

        output.WriteLine($"{cases} cases: reference logits decode to the reference entities, offsets, order and confidences exactly");
        Assert.Equal(58, cases);
    }

    [Fact]
    public void ClosedGateGivesNoEntitiesWhateverTheScores()
    {
        var input = InputFor("Tim Cook visited Paris.", "person");
        var o = OutputFor(input, (s, at) => s[at(0, 0, 1)] = 20f, predictedCount: 0);
        var result = Decode(input, o, 0f);
        Assert.False(result.GateOpen);
        Assert.Empty(result.Entities);
        Assert.Empty(result.GetEntities("person"));
        Assert.Throws<KeyNotFoundException>(() => result.GetEntities("location"));
    }

    [Fact]
    public void CountArgmaxTieTakesTheFirstIndex()
    {
        var input = InputFor("Paris.", "location");
        var o = new Gliner2ModelOutput(new float[20], new float[input.WordCount * Width], 1, input.WordCount, Width);
        Assert.Equal(0, o.PredictedCount);
        Assert.False(Decode(input, o).GateOpen);
    }

    [Fact]
    public void ThresholdIsInclusiveInFloat32()
    {
        var input = InputFor("Paris.", "location");
        var o = OutputFor(input, (s, at) => s[at(0, 0, 0)] = 0f); // sigmoid(0) == 0.5 exactly
        Assert.Single(Decode(input, o, 0.5f).Entities);
        Assert.Empty(Decode(input, o, 0.50001f).Entities);
    }

    [Fact]
    public void OverlapsAreSuppressedPerLabelOnly()
    {
        var input = InputFor("Boston Celtics won.", "team", "city");
        var o = OutputFor(input, (s, at) =>
        {
            s[at(0, 0, 1)] = 10f; // team: Boston Celtics
            s[at(0, 0, 0)] = 5f;  // team: Boston (overlaps, lower)
            s[at(1, 0, 0)] = 8f;  // city: Boston (other label: kept)
        });
        var result = Decode(input, o);
        Assert.Equal([("team", "Boston Celtics"), ("city", "Boston")], result.Entities.Select(e => (e.Label, e.Text)));
    }

    [Fact]
    public void EqualScoresKeepStartThenWidthOrder()
    {
        var input = InputFor("Ann met Bob and Cy.", "person");
        var o = OutputFor(input, (s, at) =>
        {
            s[at(0, 4, 0)] = 30f; // Cy
            s[at(0, 0, 0)] = 30f; // Ann
            s[at(0, 2, 0)] = 30f; // Bob
        });
        var result = Decode(input, o);
        Assert.All(result.Entities, e => Assert.Equal(1f, e.Confidence));
        Assert.Equal(["Ann", "Bob", "Cy"], result.Entities.Select(e => e.Text));
    }

    [Fact]
    public void ResultOrderIsDescendingConfidenceWithinALabel()
    {
        var input = InputFor("Ann met Bob and Cy.", "person");
        var o = OutputFor(input, (s, at) =>
        {
            s[at(0, 0, 0)] = 1f;
            s[at(0, 2, 0)] = 3f;
            s[at(0, 4, 0)] = 2f;
        });
        Assert.Equal(["Bob", "Cy", "Ann"], Decode(input, o).Entities.Select(e => e.Text));
    }

    [Fact]
    public void SpanOverTheAppendedFullStopIsClampedToTheInput()
    {
        const string text = "They visited Cupertino";
        var input = InputFor(text, "location");
        Assert.Equal(text + ".", input.NormalizedText);
        var o = OutputFor(input, (s, at) =>
        {
            s[at(0, 2, 1)] = 9f; // "Cupertino." upstream, ending one past the input
            s[at(0, 0, 0)] = 8f; // "They"
        });

        var upstream = Gliner2Decoder.DecodeUpstream(input, o, 0.5f)!;
        Assert.Contains(upstream[0], s => s.Text == "Cupertino." && s.End == text.Length + 1);

        var result = Decode(input, o);
        var cupertino = result.Entities[0];
        Assert.Equal("Cupertino", cupertino.Text);
        Assert.Equal(text.Length, cupertino.End);
        Assert.Equal(text[cupertino.Start..cupertino.End], cupertino.Text);
        Assert.All(result.Entities, e => Assert.InRange(e.End, 1, text.Length));
    }

    [Fact]
    public void SpanOfOnlyTheAppendedFullStopIsDropped()
    {
        var input = InputFor("They visited Cupertino", "location");
        var o = OutputFor(input, (s, at) => s[at(0, 3, 0)] = 9f); // "." (appended)
        Assert.Single(Gliner2Decoder.DecodeUpstream(input, o, 0.5f)![0]);
        var result = Decode(input, o);
        Assert.True(result.GateOpen);
        Assert.Empty(result.Entities);
    }

    [Fact]
    public void ClampTrimsWhitespaceBeforeTheAppendedFullStop()
    {
        var input = InputFor("Boston ", "city");
        Assert.Equal("Boston .", input.NormalizedText);
        var o = OutputFor(input, (s, at) => s[at(0, 0, 1)] = 9f); // "Boston ." upstream
        var e = Assert.Single(Decode(input, o).Entities);
        Assert.Equal(("Boston", 0, 6), (e.Text, e.Start, e.End));
    }

    [Fact]
    public void EmptyTextYieldsNoEntities()
    {
        var input = InputFor(string.Empty, "person");
        Assert.Equal(".", input.NormalizedText);
        var o = OutputFor(input, (s, at) => s[at(0, 0, 0)] = 9f);
        Assert.Empty(Decode(input, o).Entities);
    }

    [Fact]
    public void OffsetsAreUtf16AndScalarIndexesOfTheInput()
    {
        const string text = "𝐀𝐁 met Bob";
        var input = InputFor(text, "person");
        var o = OutputFor(input, (s, at) => s[at(0, 2, 0)] = 9f);
        var e = Assert.Single(Decode(input, o).Entities);
        Assert.Equal("Bob", e.Text);
        Assert.Equal((9, 12), (e.Start, e.End));
        Assert.Equal((7, 10), (e.ScalarStart, e.ScalarEnd));
        Assert.Equal(text[e.Start..e.End], e.Text);
    }

    [Fact]
    public void ValidSpanLogitsSkipSpansPastTheLastWord()
    {
        var input = InputFor("a b", "x");
        Assert.Equal(3, input.WordCount); // a, b, "."
        var o = OutputFor(input, (_, _) => { });
        Assert.Equal(3 + 2 + 1, o.GetValidSpanLogits().Length);
    }
}
