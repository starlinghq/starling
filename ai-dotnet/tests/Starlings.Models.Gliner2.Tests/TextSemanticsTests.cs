using System.Text;
using System.Text.Json;

namespace Starlings.Models.Gliner2.Tests;

/// <summary>
/// Python text semantics GLiNER2 depends on, against tables recorded with the reference interpreter (CPython 3.11.16,
/// Unicode 14.0.0; Data/make_python_text_tables.py) over every Unicode scalar value. Characters unassigned in
/// Unicode 14 are excluded where .NET's newer Unicode data may legitimately differ, and counted. Runs in CI.
/// </summary>
public class TextSemanticsTests(ITestOutputHelper output)
{
    private static readonly Lazy<Tables> Data = new(Tables.Load);

    private static IEnumerable<int> Scalars()
    {
        for (int c = 0; c < 0x110000; c++)
        {
            if (c is < 0xD800 or > 0xDFFF)
            {
                yield return c;
            }
        }
    }

    [Fact]
    public void TablesComeFromTheReferenceInterpreter()
    {
        Assert.StartsWith("3.11.", Data.Value.Python, StringComparison.Ordinal);
        Assert.Equal("14.0.0", Data.Value.Unicode);
    }

    [Fact]
    public void WhitespaceIsPythonsBackslashS()
    {
        // \s for str patterns includes U+001C..U+001F and U+0085, which .NET's char.IsWhiteSpace also has, but not
        // U+200B or U+180E; the predicate must equal Python's on every scalar value, assigned or not.
        var mismatches = Scalars().Where(c => Gliner2Text.IsSpace(c) != Data.Value.Space.Contains(c)).ToList();
        Assert.Empty(mismatches);
        Assert.True(Gliner2Text.IsSpace(0x1C));
        Assert.False(Gliner2Text.IsSpace(0x200B));
    }

    [Fact]
    public void WordCharactersArePythonsBackslashW()
    {
        int skipped = 0;
        var mismatches = new List<int>();
        foreach (int c in Scalars())
        {
            if (Gliner2Text.IsWord(c) == Data.Value.Word.Contains(c))
            {
                continue;
            }

            if (Data.Value.Unassigned.Contains(c))
            {
                skipped++;
            }
            else
            {
                mismatches.Add(c);
            }
        }

        output.WriteLine($"\\w differs from Python 3.11 only on {skipped} scalar values unassigned in Unicode 14 (newer .NET Unicode data)");
        Assert.Empty(mismatches.Select(c => $"U+{c:X4}"));

        // Combining marks are not \w in Python (they are in .NET's \w); astral letters are.
        Assert.False(Gliner2Text.IsWord(0x301));
        Assert.True(Gliner2Text.IsWord(0x1D400));
        Assert.True(Gliner2Text.IsWord('_'));
    }

    [Fact]
    public void IgnoreCaseFoldsMatchPython()
    {
        var nonAscii = Scalars().Where(c => c >= 0x80 && Gliner2Text.IsAsciiLetterIgnoreCase(c)).ToList();
        Assert.Equal(Data.Value.AzIgnoreCase, nonAscii);

        // Only 's' of "https://" and "www." has a non-ASCII IGNORECASE match (U+017F); the scanner relies on that.
        foreach (var (letter, matches) in Data.Value.LiteralIgnoreCase)
        {
            Assert.Equal(letter == "s" ? [0x17F] : [], matches);
        }
    }

    [Fact]
    public void LowerIsPythonsStrLower()
    {
        int skipped = 0;
        var mismatches = new List<string>();
        var expected = Data.Value.Lower;
        foreach (int c in Scalars())
        {
            string want = expected.TryGetValue(c, out var mapped) ? mapped : new Rune(c).ToString();
            string got = Gliner2Text.Lower([c]);
            if (got == want)
            {
                continue;
            }

            if (Data.Value.Unassigned.Contains(c))
            {
                skipped++;
            }
            else
            {
                mismatches.Add($"U+{c:X4}");
            }
        }

        output.WriteLine($"lower() differs from Python 3.11 only on {skipped} scalar values unassigned in Unicode 14");
        Assert.Empty(mismatches);
        Assert.Equal("i̇", Gliner2Text.Lower([0x130]));
        Assert.Equal("i̇stanbul", Gliner2Text.Lower("İstanbul".EnumerateRunes().Select(r => r.Value).ToArray()));
    }

    [Fact]
    public void FinalSigmaContextMatchesPython()
    {
        int skipped = 0;
        var mismatches = new List<string>();
        foreach (int c in Scalars())
        {
            if (c == 0x3A3)
            {
                continue;
            }

            bool ignorable = Gliner2Text.IsCaseIgnorable(c);
            bool casedNotIgnorable = !ignorable && Gliner2Text.IsCased(c);
            bool wantIgnorable = Data.Value.SigmaIgnorable.Contains(c);
            bool wantCased = Data.Value.SigmaCased.Contains(c);
            if (ignorable == wantIgnorable && casedNotIgnorable == wantCased)
            {
                continue;
            }

            if (Data.Value.Unassigned.Contains(c))
            {
                skipped++;
            }
            else
            {
                mismatches.Add($"U+{c:X4}");
            }
        }

        output.WriteLine($"Final_Sigma context differs only on {skipped} scalar values unassigned in Unicode 14");
        Assert.Empty(mismatches);
    }

    [Theory]
    [InlineData("ΟΔΟΣ", "οδος")]
    [InlineData("ΣΑΣ", "σας")]
    [InlineData("Σ", "σ")]
    [InlineData("AΣ", "aς")]
    [InlineData("AΣB", "aσb")]
    [InlineData("A'Σ", "a'ς")]
    [InlineData("AΣ'", "aς'")]
    [InlineData("AΣ'B", "aσ'b")]
    [InlineData("́Σ", "́σ")]
    [InlineData("ªΣ", "ªς")]
    [InlineData("ⅫΣ", "ⅻς")]
    public void FinalSigmaExamples(string word, string expected) =>
        Assert.Equal(expected, Gliner2Text.Lower(word.EnumerateRunes().Select(r => r.Value).ToArray()));

    [Theory]
    [InlineData("", ".")]
    [InlineData(" ", " .")]
    [InlineData("Boston", "Boston.")]
    [InlineData("Boston.", "Boston.")]
    [InlineData("Boston!", "Boston!")]
    [InlineData("Boston?", "Boston?")]
    [InlineData("Boston ", "Boston .")]
    [InlineData("Boston…", "Boston….")]
    public void NormalizationAppendsAFullStopUnlessTheTextEndsInOne(string text, string expected) =>
        Assert.Equal(expected, Gliner2Text.Normalize(text));

    [Fact]
    public void UnpairedSurrogatesAreRejected()
    {
        Assert.Throws<ArgumentException>(() => Gliner2Text.ToScalars("a\uD800b", "text"));
        Assert.Throws<ArgumentException>(() => Gliner2Text.ToScalars("a\uDC00", "text"));
        Assert.Equal([0x61, 0x1D400], Gliner2Text.ToScalars("a\U0001D400", "text"));
    }

    private sealed class Tables
    {
        public required string Python { get; init; }

        public required string Unicode { get; init; }

        public required HashSet<int> Space { get; init; }

        public required HashSet<int> Word { get; init; }

        public required List<int> AzIgnoreCase { get; init; }

        public required Dictionary<string, List<int>> LiteralIgnoreCase { get; init; }

        public required Dictionary<int, string> Lower { get; init; }

        public required HashSet<int> SigmaIgnorable { get; init; }

        public required HashSet<int> SigmaCased { get; init; }

        public required HashSet<int> Unassigned { get; init; }

        public static Tables Load()
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Data", "python311-text-semantics.json")));
            var r = doc.RootElement;
            return new Tables
            {
                Python = r.GetProperty("python").GetString()!,
                Unicode = r.GetProperty("unicode").GetString()!,
                Space = Ranges(r.GetProperty("space")),
                Word = Ranges(r.GetProperty("word")),
                AzIgnoreCase = r.GetProperty("az_ignorecase").EnumerateArray().Select(e => e.GetInt32()).ToList(),
                LiteralIgnoreCase = r.GetProperty("literal_ignorecase").EnumerateObject()
                    .ToDictionary(p => p.Name, p => p.Value.EnumerateArray().Select(e => e.GetInt32()).ToList()),
                Lower = r.GetProperty("lower").EnumerateArray().ToDictionary(
                    e => e[0].GetInt32(),
                    e => string.Concat(e[1].EnumerateArray().Select(x => new Rune(x.GetInt32()).ToString()))),
                SigmaIgnorable = Ranges(r.GetProperty("sigma_ignorable")),
                SigmaCased = Ranges(r.GetProperty("sigma_cased")),
                Unassigned = Ranges(r.GetProperty("unassigned")),
            };
        }

        private static HashSet<int> Ranges(JsonElement ranges)
        {
            var set = new HashSet<int>();
            foreach (var range in ranges.EnumerateArray())
            {
                for (int c = range[0].GetInt32(); c <= range[1].GetInt32(); c++)
                {
                    set.Add(c);
                }
            }

            return set;
        }
    }
}
