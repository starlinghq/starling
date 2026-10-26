using System.Diagnostics;
using System.Text;
using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>
/// Token counting without embedding, the length bound that rejects certainly-too-long text without tokenizing it, and
/// chunking, on the pinned <c>tokenizer.json</c> (skips without it; tools/fetch-tokenizer-assets.sh provides it) and on
/// synthetic tokenizer files. Weight-free.
/// </summary>
public class LengthAndCountingTests(ITestOutputHelper output)
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    [Fact]
    public void ThePinnedFileHasALengthBoundOf31()
    {
        var tokenizer = EmbeddingGemma2TestAssets.Tokenizer;
        Assert.Equal(31, tokenizer.MaxCharactersPerToken);
        Assert.Equal(2, tokenizer.SpecialTokenCount);
    }

    [Fact]
    public void AOneMegabyteDocumentIsRejectedWithoutTokenizingIt()
    {
        var tokenizer = EmbeddingGemma2TestAssets.Tokenizer;
        string document = new StringBuilder().Insert(0, "Lorem ipsum dolor sit amet. ", 40_000).ToString();
        Assert.True(document.Length > 1_000_000);

        var watch = Stopwatch.StartNew();
        var check = tokenizer.Validate(document, EmbeddingGemma2Encoder.MaxSupportedSequenceLength);
        var ex = Assert.Throws<EmbeddingGemma2InputTooLongException>(() => tokenizer.Tokenize(document, EmbeddingGemma2Encoder.MaxSupportedSequenceLength, inputIndex: 3));
        watch.Stop();
        output.WriteLine($"{document.Length} code units: lower bound {check.TokenCount} tokens, checked twice in {watch.Elapsed.TotalMilliseconds:F1} ms");

        Assert.True(check.TokenCountIsLowerBound);
        Assert.True(check.ExceedsLimit);
        Assert.Equal(2 + ((document.Length + 30) / 31), check.TokenCount);
        Assert.True(ex.TokenCountIsLowerBound);
        Assert.Equal(3, ex.InputIndex);
        Assert.Equal(check.TokenCount, ex.TokenCount);
        Assert.Equal(8192, ex.MaxTokens);
        Assert.Contains("at least", ex.Message, StringComparison.Ordinal);

        using var encoder = EncoderBehaviorTests.FakeEncoder();
        var embedder = new EmbeddingGemma2TextEmbedder(tokenizer, encoder);
        var batch = Assert.Throws<EmbeddingGemma2InputTooLongException>(() => embedder.EmbedBatch(["short", document], Ct));
        Assert.Equal(1, batch.InputIndex);
        Assert.True(batch.TokenCountIsLowerBound);
        Assert.True(embedder.Validate(document).TokenCountIsLowerBound);
    }

    [Theory]
    [InlineData("The quick brown fox jumps over the lazy dog near the river bank.")]
    [InlineData("                                                                                                    ")]
    [InlineData("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")]
    [InlineData("😀😀😀 混合テキスト ﷽ ‍👩‍👩‍👧")]
    [InlineData("<|tool_response><tool_response|><|tool_call>")]
    public void TheLowerBoundNeverExceedsTheCount(string text)
    {
        var tokenizer = EmbeddingGemma2TestAssets.Tokenizer;
        foreach (string sample in new[] { text, string.Concat(Enumerable.Repeat(text, 50)) })
        {
            int count = tokenizer.CountTokens(sample);
            Assert.True(tokenizer.LowerBound(sample) <= count, $"bound {tokenizer.LowerBound(sample)} > count {count}");
            Assert.Equal(new TokenLimitCheck(count, 8192), tokenizer.Validate(sample, 8192));
        }
    }

    [Fact]
    public void TheLowerBoundHoldsOnTheFixtureTexts()
    {
        var tokenizer = EmbeddingGemma2TestAssets.Tokenizer;
        foreach (var row in EmbeddingGemma2Fixtures.LoadRows())
        {
            string text = row.PackagePrompt.Apply(row.Text);
            Assert.True(tokenizer.LowerBound(text) <= tokenizer.CountTokens(text));
        }
    }

    [Fact]
    public void TextUnderTheBoundIsCheckedExactly()
    {
        var tokenizer = EmbeddingGemma2TestAssets.Tokenizer;
        string text = string.Join(' ', Enumerable.Repeat("token", 9000));
        var check = tokenizer.Validate(text, 8192);
        Assert.False(check.TokenCountIsLowerBound);
        Assert.Equal(tokenizer.Tokenize(text).Length, check.TokenCount);
        Assert.True(check.ExceedsLimit);
        var ex = Assert.Throws<EmbeddingGemma2InputTooLongException>(() => tokenizer.Tokenize(text, 8192));
        Assert.False(ex.TokenCountIsLowerBound);
        Assert.Equal(check.TokenCount, ex.TokenCount);
    }

    [Fact]
    public void CountTokensIncludesThePrompt()
    {
        var tokenizer = EmbeddingGemma2TestAssets.Tokenizer;
        using var encoder = EncoderBehaviorTests.FakeEncoder();
        var embedder = new EmbeddingGemma2TextEmbedder(tokenizer, encoder);
        const string text = "How do I bake bread?";
        Assert.Equal(tokenizer.Tokenize(text).Length, embedder.CountTokens(text));
        Assert.Equal(tokenizer.Tokenize(EmbeddingGemma2Prompt.Query.Apply(text)).Length, embedder.CountTokens(text, EmbeddingGemma2Prompt.Query));
        Assert.Equal([embedder.CountTokens(text), 2], embedder.CountTokens([text, ""]));
        Assert.Equal(embedder.CountTokens(text, EmbeddingGemma2Prompt.Query), embedder.Validate(text, EmbeddingGemma2Prompt.Query).TokenCount);
        Assert.Same(tokenizer, embedder.ChunkingTokenizer);
    }

    [Fact]
    public void ChunksOfALongDocumentFitTheLimitWithThePrompt()
    {
        var tokenizer = EmbeddingGemma2TestAssets.Tokenizer;
        string text = string.Join(' ', Enumerable.Range(0, 3000).Select(i => $"sentence {i} ends here."));
        var prompt = EmbeddingGemma2Prompt.Document;
        var chunker = new TextChunker(tokenizer, new TextChunkerOptions { MaxTokens = 2048, OverlapTokens = 64, Prefix = prompt.Text });
        var chunks = chunker.Chunk(text, Ct);
        Assert.True(chunks.Count > 2, $"{chunks.Count} chunks");
        foreach (var chunk in chunks)
        {
            Assert.InRange(chunk.TokenCount, 1, 2048);
            Assert.Equal(tokenizer.Tokenize(prompt.Apply(chunk.Text)).Length, chunk.TokenCount);
        }
    }

    [Fact]
    public void FilesThatCanDropCharactersHaveNoBound()
    {
        static string File(string preTokenizer = "null", string addedFlags = "false", string model = "BPE", string fallback = "true", string normalizer = "null") =>
            "{\"normalizer\": " + normalizer + ", \"pre_tokenizer\": " + preTokenizer +
            ", \"added_tokens\": [{\"id\": 0, \"content\": \"<s>\", \"lstrip\": " + addedFlags + ", \"rstrip\": false, \"single_word\": false}]" +
            ", \"model\": {\"type\": \"" + model + "\", \"byte_fallback\": " + fallback + ", \"fuse_unk\": true, \"vocab\": {\"abcdefgh\": 1, \"a\": 2}}}";
        static int? Bound(string json) => EmbeddingGemma2Tokenizer.ComputeMaxCharactersPerToken(Encoding.UTF8.GetBytes(json));

        Assert.Equal(8, Bound(File()));
        Assert.Equal(8, Bound(File(preTokenizer: """{"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": false}""")));
        Assert.Equal(8, Bound(File(normalizer: """{"type": "Replace", "pattern": {"String": " "}, "content": "▁"}""")));
        Assert.Null(Bound(File(preTokenizer: """{"type": "Whitespace"}""")));
        Assert.Null(Bound(File(preTokenizer: """{"type": "Split", "pattern": {"String": " "}, "behavior": "Removed", "invert": false}""")));
        Assert.Null(Bound(File(normalizer: """{"type": "Replace", "pattern": {"String": "  "}, "content": " "}""")));
        Assert.Null(Bound(File(normalizer: """{"type": "Strip", "strip_left": true, "strip_right": true}""")));
        Assert.Null(Bound(File(addedFlags: "true")));
        Assert.Null(Bound(File(model: "WordPiece")));
        Assert.Null(Bound(File(fallback: "false")));
        Assert.Null(Bound("not json"));
    }
}
