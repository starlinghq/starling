using System.Text;

namespace Starlings.Models.Qwen3Embedding.Tests;

/// <summary>
/// The inputs both profiles build, against the token ids of the PyTorch reference (TestData, written by
/// model-pipeline/reference/qwen3-embedding/make_dotnet_reference.py). Needs only the pinned tokenizers.
/// </summary>
public sealed class TokenizationTests
{
    private static Qwen3Tokenization Embedding()
    {
        var (tokenization, reason) = Qwen3TestAssets.Tokenization(Qwen3EmbeddingProfile.Embedding06B);
        if (tokenization is null)
        {
            Assert.Skip(reason);
        }

        return tokenization;
    }

    private static Qwen3Tokenization Reranker()
    {
        var (tokenization, reason) = Qwen3TestAssets.Tokenization(Qwen3EmbeddingProfile.Reranker06B);
        if (tokenization is null)
        {
            Assert.Skip(reason);
        }

        return tokenization;
    }

    [Fact]
    public void EmbeddingIdsEqualTheReference()
    {
        var tokenization = Embedding();
        foreach (var c in Qwen3TestAssets.Reference("embedding.json").EnumerateArray())
        {
            string prompt = c.GetProperty("prompt").GetString()!;
            string text = c.GetProperty("text").GetString()!;
            long[] expected = Qwen3TestAssets.Longs(c.GetProperty("ids"));
            var input = tokenization.TokenizeText(text, tokenization.PromptText(prompt));
            Assert.Equal(expected, input.InputIds.ToArray());
            Assert.All(input.AttentionMask.ToArray(), m => Assert.Equal(1L, m));
            Assert.Equal(Qwen3EmbeddingProfile.EndOfTextTokenId, input.InputIds.Span[^1]);
            Assert.Equal(expected.Length, tokenization.CountText(text, tokenization.PromptText(prompt)));
        }
    }

    [Fact]
    public void EmbeddingTruncatesFromTheRightAndKeepsTheAppendedToken()
    {
        var tokenization = Embedding();
        string text = new StringBuilder().Insert(0, "word ", 9000).ToString();
        Assert.True(tokenization.CountText(text, string.Empty) > 8192);
        var input = tokenization.TokenizeText(text, Qwen3EmbeddingProfile.QueryPrompt);
        Assert.Equal(8192, input.Length);
        Assert.Equal(Qwen3EmbeddingProfile.EndOfTextTokenId, input.InputIds.Span[^1]);
        var prompt = tokenization.TokenizeText(string.Empty, Qwen3EmbeddingProfile.QueryPrompt);
        Assert.Equal(prompt.InputIds.Span[..^1].ToArray(), input.InputIds.Span[..(prompt.Length - 1)].ToArray());
    }

    [Fact]
    public void UndeclaredPromptIsRejected()
    {
        var tokenization = Embedding();
        Assert.Throws<ArgumentException>(() => tokenization.PromptText("passage"));
        Assert.Throws<InvalidOperationException>(() => tokenization.TokenizePair("i", "q", "d"));
    }

    [Fact]
    public void RerankerRowsEqualTheReference()
    {
        var tokenization = Reranker();
        var reference = Qwen3TestAssets.Reference("reranker.json");
        long[] prefix = Qwen3TestAssets.Longs(reference.GetProperty("prefix_ids"));
        long[] suffix = Qwen3TestAssets.Longs(reference.GetProperty("suffix_ids"));
        Assert.Equal(prefix, tokenization.RerankPrefixIds);
        Assert.Equal(suffix, tokenization.RerankSuffixIds);
        foreach (var c in reference.GetProperty("cases").EnumerateArray())
        {
            long[] content = Qwen3TestAssets.Longs(c.GetProperty("content_ids"));
            string instruction = c.GetProperty("instruction").GetString()!, query = c.GetProperty("query").GetString()!, document = c.GetProperty("document").GetString()!;
            var input = tokenization.TokenizePair(instruction, query, document);
            Assert.Equal([.. prefix, .. content, .. suffix], input.InputIds.ToArray());
            Assert.Equal(input.Length, tokenization.CountPair(instruction, query, document));
        }

        Assert.Throws<InvalidOperationException>(() => tokenization.TokenizeText("x", string.Empty));
    }

    [Fact]
    public void RerankerTruncatesOnlyTheContent()
    {
        var tokenization = Reranker();
        string document = new StringBuilder().Insert(0, "word ", 9000).ToString();
        var input = tokenization.TokenizePair(Qwen3EmbeddingProfile.DefaultRerankInstruction, "query", document);
        Assert.Equal(8192, input.Length);
        Assert.Equal(tokenization.RerankPrefixIds, input.InputIds.Span[..tokenization.RerankPrefixIds.Count].ToArray());
        Assert.Equal(tokenization.RerankSuffixIds, input.InputIds.Span[^tokenization.RerankSuffixIds.Count..].ToArray());
        Assert.True(tokenization.CountPair(Qwen3EmbeddingProfile.DefaultRerankInstruction, "query", document) > 8192);
    }

    [Fact]
    public void RowValidation()
    {
        var tokenization = Embedding();
        long pad = Qwen3EmbeddingProfile.PadTokenId;
        tokenization.Validate(new Qwen3Input(new long[] { 5, 6, pad }, new long[] { 1, 1, 0 }));
        Assert.Throws<ArgumentException>(() => tokenization.Validate(new Qwen3Input(new long[] { 5, 6 }, new long[] { 1 })));
        Assert.Throws<ArgumentException>(() => tokenization.Validate(new Qwen3Input(new long[] { 5, 6 }, new long[] { 0, 1 })));
        Assert.Throws<ArgumentException>(() => tokenization.Validate(new Qwen3Input(new long[] { 5, 7 }, new long[] { 1, 0 })));
        Assert.Throws<ArgumentException>(() => tokenization.Validate(new Qwen3Input(new long[] { 151669 }, new long[] { 1 })));
        Assert.Throws<ArgumentException>(() => tokenization.Validate(new Qwen3Input(Array.Empty<long>(), Array.Empty<long>())));
    }
}
