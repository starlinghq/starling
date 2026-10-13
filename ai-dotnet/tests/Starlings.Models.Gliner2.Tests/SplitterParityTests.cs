using System.Text.Json;
using Starlings.TestSupport;

namespace Starlings.Models.Gliner2.Tests;

/// <summary>
/// GLiNER2's normalization, word splitter and lower-casing against recorded reference output: adversarial strings run
/// through GLiNER2's own splitter (Data/splitter-cases.json, always), the accepted tokenizer fixtures (46 cases, on
/// main) and the protected extraction fixtures (58 cases, accepted in PR #19). No tokenizer or model needed.
/// </summary>
public class SplitterParityTests(ITestOutputHelper output)
{
    private static List<Gliner2Word> Split(string text) =>
        Gliner2Text.SplitWords(Gliner2Text.ToScalars(Gliner2Text.Normalize(text), nameof(text)));

    [Fact]
    public void AdversarialStringsMatchGliner2sSplitter()
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Data", "splitter-cases.json")));
        int cases = 0, words = 0;
        foreach (var c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            string id = c.GetProperty("id").GetString()!;
            string text = c.GetProperty("text").GetString()!;
            Assert.Equal(c.GetProperty("normalized_text").GetString(), Gliner2Text.Normalize(text));
            var expected = c.GetProperty("words").EnumerateArray()
                .Select(w => new Gliner2Word(w.GetProperty("word").GetString()!, w.GetProperty("start").GetInt32(), w.GetProperty("end").GetInt32()))
                .ToList();
            var actual = Split(text);
            Assert.True(expected.SequenceEqual(actual), $"{id}: expected [{string.Join("|", expected)}], got [{string.Join("|", actual)}]");
            cases++;
            words += expected.Count;
        }

        output.WriteLine($"adversarial splitter cases {cases}, words {words}: exact");
        Assert.True(cases >= 15);
    }

    [Fact]
    public void AcceptedTokenizerFixturesMatch()
    {
        int cases = 0, words = 0;
        foreach (var row in Gliner2TestAssets.ReadJsonLines(Gliner2TestAssets.TokenizerFixturePath))
        {
            var split = row.GetProperty("whitespace_splitter");
            string text = row.GetProperty("text").GetString()!;
            Assert.Equal(split.GetProperty("normalized_text").GetString(), Gliner2Text.Normalize(text));
            var expected = split.GetProperty("words").EnumerateArray().Select(w =>
            {
                var offsets = w.GetProperty("offsets").GetProperty("unicode-scalars");
                return new Gliner2Word(w.GetProperty("word").GetString()!, offsets[0].GetInt32(), offsets[1].GetInt32());
            }).ToList();
            var actual = Split(text);
            Assert.True(expected.SequenceEqual(actual), $"{row.GetProperty("id").GetString()}: [{string.Join("|", actual)}]");
            cases++;
            words += expected.Count;
        }

        output.WriteLine($"accepted tokenizer fixtures: {cases} cases, {words} words exact");
        Assert.Equal(46, cases);
    }

    [Fact]
    public void ProtectedExtractionFixturesMatch()
    {
        var (path, reason) = Gliner2TestAssets.FindExtractionFixtures();
        Assert.SkipWhen(path is null, reason);
        int cases = 0, words = 0;
        foreach (var c in Gliner2Fixtures.LoadCases(path!))
        {
            Assert.Equal(c.NormalizedText, Gliner2Text.Normalize(c.Text));
            var actual = Split(c.Text);
            if (c.MaxLen is int cap && actual.Count > cap)
            {
                actual = actual.Take(cap).ToList(); // the reference applied GLiNER2's max_len word cap (not offered in C#)
            }

            Assert.Equal(c.Words, actual.Select(w => w.Word));
            Assert.Equal(c.StartMap, actual.Select(w => w.Start));
            Assert.Equal(c.EndMap, actual.Select(w => w.End));
            cases++;
            words += c.Words.Count;
        }

        output.WriteLine($"protected extraction fixtures: {cases} cases, {words} words exact");
        Assert.Equal(58, cases);
    }
}
