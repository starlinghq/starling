using Starlings.TestSupport;

namespace Starlings.Models.Gliner2.Tests;

/// <summary>
/// Discrete model inputs (schema tokens, token ids, word and schema positions) built with the owned tokenizer against
/// the accepted tokenizer fixtures and the protected extraction fixtures. Exact, no tolerance. Needs the pinned
/// tokenizer.json (tools/fetch-tokenizer-assets.sh); no model.
/// </summary>
public class InputExactnessTests(ITestOutputHelper output)
{
    private static readonly Lazy<(Gliner2Preprocessor? Preprocessor, string Reason)> Shared = new(() =>
    {
        var (path, reason) = Gliner2TestAssets.FindTokenizer();
        return path is null ? (null, reason) : (Gliner2Preprocessor.FromFile(path, Gliner2ModelAssets.TokenizerSha256), string.Empty);
    });

    private static Gliner2Preprocessor Preprocessor
    {
        get
        {
            Assert.SkipWhen(Shared.Value.Preprocessor is null, Shared.Value.Reason);
            return Shared.Value.Preprocessor!;
        }
    }

    [Fact]
    public void AcceptedTokenizerFixturesGiveTheReferenceIdsAndPositions()
    {
        var preprocessor = Preprocessor;
        int cases = 0;
        foreach (var row in Gliner2TestAssets.ReadJsonLines(Gliner2TestAssets.TokenizerFixturePath))
        {
            var split = row.GetProperty("whitespace_splitter");
            var schema = split.GetProperty("schema_tokens")[0].EnumerateArray().Select(e => e.GetString()!).ToList();

            // Labels are the tokens after each [E] marker of the recorded schema.
            var labels = Enumerable.Range(0, schema.Count).Where(i => schema[i] == "[E]").Select(i => schema[i + 1]).ToList();
            var input = preprocessor.Build(row.GetProperty("text").GetString()!, Gliner2Preprocessor.NormalizeLabels(labels, "labels"), null);
            string id = row.GetProperty("id").GetString()!;
            Assert.Equal(schema, input.SchemaTokens);
            Assert.True(split.GetProperty("input_ids").EnumerateArray().Select(e => e.GetInt64()).SequenceEqual(input.InputIds.ToArray()), $"{id}: input_ids");
            Assert.True(split.GetProperty("text_word_first_positions").EnumerateArray().Select(e => e.GetInt64()).SequenceEqual(input.WordIndex.ToArray()), $"{id}: word positions");
            Assert.True(split.GetProperty("schema_special_positions")[0].EnumerateArray().Select(e => e.GetInt64()).SequenceEqual(input.SchemaIndex.ToArray()), $"{id}: schema positions");
            cases++;
        }

        output.WriteLine($"accepted tokenizer fixtures: {cases} cases, input ids and positions exact");
        Assert.Equal(46, cases);
    }

    [Fact]
    public void ProtectedExtractionFixturesGiveTheReferenceInputs()
    {
        var preprocessor = Preprocessor;
        var (path, reason) = Gliner2TestAssets.FindExtractionFixtures();
        Assert.SkipWhen(path is null, reason);
        int cases = 0;
        foreach (var c in Gliner2Fixtures.LoadCases(path!))
        {
            if (c.MaxLen is not null)
            {
                continue; // max_len word cap: not offered in C#
            }

            var input = preprocessor.Build(c.Text, Gliner2Preprocessor.NormalizeLabels(c.LabelsRequested, "labels"), c.Descriptions);
            Assert.Equal(c.NormalizedText, input.NormalizedText);
            Assert.Equal(c.Labels, input.Labels);
            Assert.Equal(c.SchemaTokens, input.SchemaTokens);
            Assert.Equal(c.Words, input.Words);
            Assert.True(c.InputIds.AsSpan().SequenceEqual(input.InputIds), $"{c.Id}: input_ids");
            Assert.True(c.WordIndex.AsSpan().SequenceEqual(input.WordIndex), $"{c.Id}: word_index");
            Assert.True(c.SchemaIndex.AsSpan().SequenceEqual(input.SchemaIndex), $"{c.Id}: schema_index");
            Assert.True(c.StartMap.AsSpan().SequenceEqual(input.WordStarts), $"{c.Id}: start_map");
            Assert.True(c.EndMap.AsSpan().SequenceEqual(input.WordEnds), $"{c.Id}: end_map");
            cases++;
        }

        output.WriteLine($"protected extraction fixtures: {cases} cases (word-cap cases excluded), inputs exact, including the two over 512 tokens");
        Assert.Equal(56, cases);
    }

    [Fact]
    public void MemoizedTokenIdsEqualFreshTokenization()
    {
        // A fresh preprocessor tokenizes every token (cold memo); building every case again reads the memo. Both builds
        // must give the reference ids, and a token above the memo's length limit is tokenized every time.
        var (tokenizerPath, reason) = Gliner2TestAssets.FindTokenizer();
        Assert.SkipWhen(tokenizerPath is null, reason);
        var (path, fixtureReason) = Gliner2TestAssets.FindExtractionFixtures();
        Assert.SkipWhen(path is null, fixtureReason);
        var preprocessor = Gliner2Preprocessor.FromFile(tokenizerPath!, Gliner2ModelAssets.TokenizerSha256);
        var cases = Gliner2Fixtures.LoadCases(path!).Where(c => c.MaxLen is null).ToList();
        for (int pass = 0; pass < 2; pass++)
        {
            foreach (var c in cases)
            {
                var input = preprocessor.Build(c.Text, Gliner2Preprocessor.NormalizeLabels(c.LabelsRequested, "labels"), c.Descriptions);
                Assert.True(c.InputIds.AsSpan().SequenceEqual(input.InputIds), $"{c.Id} (pass {pass}): input_ids");
                Assert.True(c.WordIndex.AsSpan().SequenceEqual(input.WordIndex), $"{c.Id} (pass {pass}): word_index");
            }
        }

        string longToken = string.Concat(Enumerable.Repeat("antidisestablishment", 4));
        Assert.True(longToken.Length > Gliner2Preprocessor.MaxCachedTokenLength);
        var first = preprocessor.PieceIds(longToken);
        Assert.NotSame(first, preprocessor.PieceIds(longToken));
        Assert.Equal(first, preprocessor.PieceIds(longToken));
        Assert.Same(preprocessor.PieceIds("person"), preprocessor.PieceIds("person"));
    }

    [Fact]
    public void ConcurrentBuildsShareTheMemoAndGiveTheReferenceInputs()
    {
        // Eight threads build every case at once on one fresh preprocessor (the memo filled concurrently); every input
        // must equal the reference. The memo never evicts: it keeps the first MaxCachedTokens distinct tokens.
        var (tokenizerPath, reason) = Gliner2TestAssets.FindTokenizer();
        Assert.SkipWhen(tokenizerPath is null, reason);
        var (path, fixtureReason) = Gliner2TestAssets.FindExtractionFixtures();
        Assert.SkipWhen(path is null, fixtureReason);
        var preprocessor = Gliner2Preprocessor.FromFile(tokenizerPath!, Gliner2ModelAssets.TokenizerSha256);
        var cases = Gliner2Fixtures.LoadCases(path!).Where(c => c.MaxLen is null).ToList();
        var failures = new System.Collections.Concurrent.ConcurrentBag<string>();
        Parallel.For(0, 8 * cases.Count, new ParallelOptions { MaxDegreeOfParallelism = 8 }, i =>
        {
            var c = cases[i % cases.Count];
            var input = preprocessor.Build(c.Text, Gliner2Preprocessor.NormalizeLabels(c.LabelsRequested, "labels"), c.Descriptions);
            if (!c.InputIds.AsSpan().SequenceEqual(input.InputIds) || !c.WordIndex.AsSpan().SequenceEqual(input.WordIndex))
            {
                failures.Add(c.Id);
            }
        });
        Assert.Empty(failures);
    }

    [Fact]
    public void LabelsAreDeduplicatedInOrderAndValidated()
    {
        Assert.Equal(["person", "Person", "location"], Gliner2Preprocessor.NormalizeLabels(["person", "Person", "person", "location"], "labels"));
        Assert.Throws<ArgumentException>(() => Gliner2Preprocessor.NormalizeLabels([], "labels"));
        Assert.Throws<ArgumentException>(() => Gliner2Preprocessor.NormalizeLabels(["person", " "], "labels"));
        Assert.Throws<ArgumentException>(() => Gliner2Preprocessor.NormalizeLabels(["person", null!], "labels"));
    }
}
