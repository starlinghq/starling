using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>
/// <see cref="EmbeddingGemma2Tokenizer"/> on the pinned tokenizer.json against the protected fixtures, and the text
/// path's limits on the weight-free graph. Runs in CI (tools/fetch-tokenizer-assets.sh provides the file); skips
/// without it.
/// </summary>
public class TokenizerTests(ITestOutputHelper output)
{
    [Fact]
    public void FixtureTextsTokenizeToTheProtectedIds()
    {
        var tokenizer = EmbeddingGemma2TestAssets.Tokenizer;
        int cases = 0;
        foreach (var row in EmbeddingGemma2TestAssets.ReadJsonLines(Path.Combine(EmbeddingGemma2TestAssets.FixtureDirectory, "tokenizer.jsonl")))
        {
            if (row.TryGetProperty("pair", out _))
            {
                continue; // pairs are not an embedding input (capability pair_input: unsupported)
            }

            var encoding = row.GetProperty("encoding");
            var input = tokenizer.Tokenize(row.GetProperty("text").GetString()!);
            Assert.Equal(encoding.GetProperty("ids").EnumerateArray().Select(e => e.GetInt64()), input.InputIds.ToArray());
            Assert.Equal(encoding.GetProperty("attention_mask").EnumerateArray().Select(e => e.GetInt64()), input.AttentionMask.ToArray());
            cases++;
        }

        // Prompt rows: the ids the reference computed from prompt_text + text (equal to sentence-transformers' preprocessing).
        int rows = 0;
        foreach (var row in EmbeddingGemma2Fixtures.LoadRows())
        {
            Assert.Equal(row.InputIds, tokenizer.Tokenize(row.PackagePrompt.Apply(row.Text)).InputIds.ToArray());
            rows++;
        }

        output.WriteLine($"{cases} tokenizer.jsonl single cases (ids and masks) and {rows} embeddings.jsonl rows (none/query/document) exact");
        Assert.Equal(46, cases);
        Assert.Equal(138, rows);
    }

    [Fact]
    public void TheFileDigestIsEnforced()
    {
        var (path, reason) = EmbeddingGemma2TestAssets.FindTokenizer();
        Assert.SkipWhen(path is null, reason);
        Assert.Throws<InvalidDataException>(() => EmbeddingGemma2Tokenizer.FromFile(path!, new string('0', 64)));
    }

    [Theory]
    [InlineData("a photo <|image|> of a cat", 258_880L)]
    [InlineData("<|audio|>", 258_881L)]
    [InlineData("clip: <|video|>", 258_884L)]
    [InlineData("<|image>x<image|>", 255_999L)]
    public void MediaPlaceholderLiteralsInTextAreRejected(string text, long id)
    {
        var tokenizer = EmbeddingGemma2TestAssets.Tokenizer;
        Assert.Contains(id, tokenizer.Tokenize(text).InputIds.ToArray());
        using var encoder = EncoderBehaviorTests.FakeEncoder();
        var embedder = new EmbeddingGemma2TextEmbedder(tokenizer, encoder);
        var ex = Assert.Throws<ArgumentException>(() => embedder.Embed(text, TestContext.Current.CancellationToken));
        Assert.Contains("text only", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TextAbove8192TokensIsRejectedNotTruncated()
    {
        var tokenizer = EmbeddingGemma2TestAssets.Tokenizer;
        using var encoder = EncoderBehaviorTests.FakeEncoder();
        var embedder = new EmbeddingGemma2TextEmbedder(tokenizer, encoder);
        var ct = TestContext.Current.CancellationToken;

        // Words of a fixture-like sentence repeated; the largest word count whose text is at most 8,192 tokens.
        string[] words = "The quick brown fox jumps over the lazy dog near the river bank".Split(' ');
        string Text(int count) => string.Join(' ', Enumerable.Range(0, count).Select(i => words[i % words.Length]));
        int lo = 1, hi = 20_000;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) / 2;
            if (tokenizer.Tokenize(Text(mid)).Length <= 8192)
            {
                lo = mid;
            }
            else
            {
                hi = mid - 1;
            }
        }

        int n = lo;
        output.WriteLine($"{n} words = {tokenizer.Tokenize(Text(n)).Length} tokens; {n + 1} words = {tokenizer.Tokenize(Text(n + 1)).Length} tokens");
        Assert.InRange(tokenizer.Tokenize(Text(n)).Length, 8190, 8192);
        Assert.Equal(768, embedder.Embed(Text(n), ct).Length);
        var ex = Assert.Throws<EmbeddingGemma2InputTooLongException>(() => embedder.EmbedBatch(["short", Text(n + 1)], ct));
        Assert.Equal(1, ex.InputIndex);
        Assert.True(ex.TokenCount > 8192);

        // The prompt counts toward the limit.
        Assert.Throws<EmbeddingGemma2InputTooLongException>(() => embedder.Embed(Text(n), EmbeddingGemma2Prompt.Query, ct));
    }
}
