using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.E5.Tests;

/// <summary>
/// <see cref="TextChunker"/> (Starlings.Models.Abstractions) on stand-in tokenizers, and token counting
/// (<see cref="E5TextEmbedder.CountTokens(string)"/>, <see cref="E5TextEmbedder.Validate"/>) and chunking on the pinned
/// e5 <c>tokenizer.json</c>. The stand-in tests are weight-free; the tokenizer tests skip without the file
/// (tools/fetch-tokenizer-assets.sh).
/// </summary>
public class ChunkingAndCountingTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static string Words(int count) => string.Join(' ', Enumerable.Range(0, count).Select(i => $"w{i}"));

    [Fact]
    public void TextThatFitsIsOneChunkCoveringAllOfIt()
    {
        var chunker = new TextChunker(new WordTokenizer(), new TextChunkerOptions { MaxTokens = 10 });
        var chunk = Assert.Single(chunker.Chunk("  three short words ", Ct));
        Assert.Equal(new TextChunk(0, 20, "  three short words ", 5), chunk);
        Assert.Equal("", Assert.Single(chunker.Chunk("", Ct)).Text);
    }

    [Fact]
    public void LongTextIsSplitIntoOverlappingWindowsOfTokens()
    {
        string text = Words(10);
        var chunker = new TextChunker(new WordTokenizer(), new TextChunkerOptions { MaxTokens = 6, OverlapTokens = 1 });
        var chunks = chunker.Chunk(text, Ct);

        // 4 text tokens per chunk (6 minus the 2 special tokens), each chunk repeating the previous one's last token.
        Assert.Equal(["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"], chunks.Select(c => c.Text));
        Assert.All(chunks, c => Assert.Equal(c.Text, text.Substring(c.Start, c.Length)));
        Assert.All(chunks, c => Assert.Equal(6, c.TokenCount));

        var noOverlap = new TextChunker(new WordTokenizer(), new TextChunkerOptions { MaxTokens = 6 }).Chunk(text, Ct);
        Assert.Equal(["w0 w1 w2 w3", "w4 w5 w6 w7", "w8 w9"], noOverlap.Select(c => c.Text));
    }

    [Fact]
    public void ThePrefixIsCountedAgainstTheLimitButNotPartOfTheChunk()
    {
        var chunker = new TextChunker(new WordTokenizer(), new TextChunkerOptions { MaxTokens = 6, Prefix = "passage: " });
        var chunks = chunker.Chunk(Words(7), Ct);
        Assert.Equal(["w0 w1 w2", "w3 w4 w5", "w6"], chunks.Select(c => c.Text));
        Assert.Equal([6, 6, 4], chunks.Select(c => c.TokenCount));
    }

    [Fact]
    public void AChunkThatNeedsMoreTokensAloneIsShrunkUntilItFits()
    {
        // The stand-in charges one extra token for every text tokenized alone, so a window of 4 spans needs 7 tokens.
        var chunker = new TextChunker(new WordTokenizer(extraPerText: 1), new TextChunkerOptions { MaxTokens = 6 });
        var chunks = chunker.Chunk(Words(7), Ct);
        Assert.Equal(["w0 w1 w2", "w3 w4 w5", "w6"], chunks.Select(c => c.Text));
        Assert.All(chunks, c => Assert.True(c.TokenCount <= 6));
    }

    [Fact]
    public void OptionsThatLeaveNoRoomAreRejected()
    {
        Assert.Throws<ArgumentException>(() => new TextChunker(new WordTokenizer(), new TextChunkerOptions { MaxTokens = 2 }));
        Assert.Throws<ArgumentException>(() => new TextChunker(new WordTokenizer(), new TextChunkerOptions { MaxTokens = 4, Prefix = "a b" }));
        Assert.Throws<ArgumentException>(() => new TextChunker(new WordTokenizer(), new TextChunkerOptions { MaxTokens = 6, OverlapTokens = 4 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TextChunker(new WordTokenizer(), new TextChunkerOptions { MaxTokens = 6, OverlapTokens = -1 }));
    }

    [Fact]
    public void PoolingIsTheNormalizedMeanOfNormalizedVectors()
    {
        float[] pooled = TextChunker.Pool([[3, 0], [0, 0.5f], [0, 0]]);
        float half = (float)Math.Sqrt(0.5);
        Assert.Equal(half, pooled[0], 6);
        Assert.Equal(half, pooled[1], 6);
        Assert.Equal([0f, 0f], TextChunker.Pool([[0, 0]]));
        Assert.Throws<ArgumentException>(() => TextChunker.Pool([]));
        Assert.Throws<ArgumentException>(() => TextChunker.Pool([[1, 0], [1]]));
    }

    [Fact]
    public void EmbedPooledEmbedsEveryChunkWithThePrefixInOneCall()
    {
        var embedder = new RecordingEmbedder();
        var chunker = new TextChunker(new WordTokenizer(), new TextChunkerOptions { MaxTokens = 5, Prefix = "query: " });
        float[] pooled = chunker.EmbedPooled(embedder, Words(4), Ct);
        Assert.Equal(["query: w0 w1", "query: w2 w3"], Assert.Single(embedder.Calls));
        Assert.Equal(TextChunker.Pool([[1, 0], [1, 0]]), pooled);
    }

    [Fact]
    public void CountTokensWithAnotherTokenizerIsTheLengthOfItsOutput()
    {
        using var encoder = new E5Encoder(
            OnnxModelSession.Create(TestModels.FakeE5, new OnnxSessionOptions { IntraOpNumThreads = 1, ComputeModelSha256 = false }),
            new E5EncoderOptions { MaxSequenceLength = 5 },
            ownsSession: true);
        var embedder = new E5TextEmbedder(new CodePointTokenizer(), encoder);
        Assert.Null(embedder.ChunkingTokenizer);
        Assert.Equal(5, embedder.CountTokens("abc"));
        Assert.Equal([2, 3, 6], embedder.CountTokens(["", "a", "abcd"]));
        Assert.Equal(new TokenLimitCheck(6, 5), embedder.Validate("abcd"));
        Assert.True(embedder.Validate("abcd").ExceedsLimit);
        Assert.False(embedder.Validate("abc").ExceedsLimit);
        Assert.Throws<ArgumentException>(() => embedder.CountTokens(["a", null!]));
    }

    [Fact]
    public void TheOwnedTokenizerCountsBeforeTruncation()
    {
        var tokenizer = OwnedTokenizer();
        string text = string.Join(' ', Enumerable.Repeat("The quick brown fox jumps over the lazy dog.", 80));
        int count = tokenizer.CountTokens(text);
        Assert.Equal(512, tokenizer.Tokenize(text).Length);
        Assert.True(count > 512, $"{count} tokens");
        Assert.Equal(2, tokenizer.SpecialTokenCount);
        Assert.Equal(count - 2, tokenizer.GetTokenSpans(text).Count);
        Assert.Equal(tokenizer.Tokenize("short text").Length, tokenizer.CountTokens("short text"));

        using var encoder = new E5Encoder(
            OnnxModelSession.Create(TestModels.FakeE5, new OnnxSessionOptions { IntraOpNumThreads = 1, ComputeModelSha256 = false }),
            ownsSession: true);
        var embedder = new E5TextEmbedder(tokenizer, encoder);
        Assert.Same(tokenizer, embedder.ChunkingTokenizer);
        var check = embedder.Validate(text);
        Assert.Equal(count, check.TokenCount);
        Assert.Equal(512, check.MaxTokens);
        Assert.True(check.ExceedsLimit);
        Assert.False(check.TokenCountIsLowerBound);
    }

    [Fact]
    public void ChunksOfTheOwnedTokenizerFitTheLimitWithThePrefix()
    {
        var tokenizer = OwnedTokenizer();
        string text = string.Join(' ', Enumerable.Range(0, 1500).Select(i => i % 7 == 0 ? $"Größe{i}," : $"word{i}"));
        var chunker = new TextChunker(tokenizer, new TextChunkerOptions { MaxTokens = 512, OverlapTokens = 32, Prefix = E5TextEmbedder.PassagePrefix });
        var chunks = chunker.Chunk(text, Ct);
        Assert.True(chunks.Count > 2, $"{chunks.Count} chunks");
        Assert.Equal(0, chunks[0].Start);
        Assert.Equal(text.Length, chunks[^1].Start + chunks[^1].Length);
        for (int i = 0; i < chunks.Count; i++)
        {
            var chunk = chunks[i];
            Assert.Equal(chunk.Text, text.Substring(chunk.Start, chunk.Length));
            Assert.InRange(chunk.TokenCount, 1, 512);
            Assert.Equal(tokenizer.Tokenize(E5TextEmbedder.PassagePrefix + chunk.Text).Length, chunk.TokenCount);
            if (i > 0)
            {
                Assert.True(chunk.Start < chunks[i - 1].Start + chunks[i - 1].Length, "consecutive chunks overlap");
            }
        }
    }

    private static E5Tokenizer OwnedTokenizer()
    {
        string path = Path.Combine(E5TestAssets.ModelsRoot, E5ModelAssets.Repository, E5ModelAssets.Revision, "tokenizer.json");
        Assert.SkipUnless(File.Exists(path), $"{path} not found; run tools/fetch-tokenizer-assets.sh.");
        return E5Tokenizer.FromFile(path, E5ModelAssets.TokenizerSha256);
    }

    /// <summary>One token per run of non-space characters, two special tokens, plus <c>extraPerText</c> per non-empty text.</summary>
    private sealed class WordTokenizer(int extraPerText = 0) : IChunkingTokenizer
    {
        public int SpecialTokenCount => 2;

        public int CountTokens(string text)
        {
            int words = GetTokenSpans(text).Count;
            return 2 + words + (words > 0 ? extraPerText : 0);
        }

        public IReadOnlyList<TokenSpan> GetTokenSpans(string text)
        {
            var spans = new List<TokenSpan>();
            int i = 0;
            while (i < text.Length)
            {
                while (i < text.Length && text[i] == ' ')
                {
                    i++;
                }

                int start = i;
                while (i < text.Length && text[i] != ' ')
                {
                    i++;
                }

                if (i > start)
                {
                    spans.Add(new TokenSpan(start, i));
                }
            }

            return spans;
        }
    }

    private sealed class CodePointTokenizer : IEncoderTokenizer
    {
        public EncoderInput Tokenize(string text) =>
            new(new long[] { 0 }.Concat(text.Select(c => (long)c)).Append(2).ToArray());
    }

    private sealed class RecordingEmbedder : ITextEmbedder
    {
        public List<string[]> Calls { get; } = [];

        public int Dimension => 2;

        public int MaxSequenceLength => 5;

        public float[][] EmbedBatch(IReadOnlyList<string> texts, Span<int> tokenCounts, CancellationToken cancellationToken = default)
        {
            Calls.Add([.. texts]);
            return texts.Select(_ => new float[] { 2, 0 }).ToArray();
        }

        public int CountTokens(string text) => throw new NotSupportedException();

        public int[] CountTokens(IReadOnlyList<string> texts) => throw new NotSupportedException();

        public TokenLimitCheck Validate(string text) => throw new NotSupportedException();
    }
}
