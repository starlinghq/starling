using System.Collections.Concurrent;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.E5.Tests;

/// <summary>
/// The tokenizer seam. A stand-in tokenizer maps text to ids; this tests only the plumbing, not tokenization
/// (the owned tokenizer is P-024, integration is a follow-up).
/// </summary>
public class TextEmbedderTests
{
    private sealed class RecordingTokenizer(Func<string, long[]> map) : IEncoderTokenizer
    {
        public ConcurrentQueue<string> Seen { get; } = new();

        public EncoderInput Tokenize(string text)
        {
            Seen.Enqueue(text);
            return new EncoderInput(map(text));
        }
    }

    [Fact]
    public void TextIsTokenizedExactlyAsGivenAndNoPrefixIsAdded()
    {
        using var encoder = new E5Encoder(OnnxModelSession.Create(TestModels.FakeE5, new OnnxSessionOptions { ComputeModelSha256 = false }), ownsSession: true);
        var tokenizer = new RecordingTokenizer(text => [0, text.Length, 2]);
        var embedder = new E5TextEmbedder(tokenizer, encoder);

        var single = embedder.Embed("hello", TestContext.Current.CancellationToken);
        var batch = embedder.EmbedBatch([E5TextEmbedder.QueryPrefix + "hello", "hello"], TestContext.Current.CancellationToken);

        Assert.Equal(["hello", "query: hello", "hello"], tokenizer.Seen);
        Assert.Equal(single, batch[1]);
        Assert.Equal(encoder.Embed(new EncoderInput(new long[] { 0, 12, 2 }), TestContext.Current.CancellationToken), batch[0]);
        Assert.Equal("passage: ", E5TextEmbedder.PassagePrefix);
    }

    [Fact]
    public void TokenCountsAreReportedWithIdenticalEmbeddings()
    {
        using var encoder = new E5Encoder(OnnxModelSession.Create(TestModels.FakeE5, new OnnxSessionOptions { ComputeModelSha256 = false }), ownsSession: true);
        var embedder = new E5TextEmbedder(new RecordingTokenizer(text => [0, .. text.Select(c => (long)c), 2]), encoder);
        string[] texts = ["a", "abcd", ""];
        var counts = new int[3];

        var withCounts = embedder.EmbedBatch(texts, counts, TestContext.Current.CancellationToken);

        Assert.Equal([3, 6, 2], counts);
        Assert.Equal(embedder.EmbedBatch(texts, TestContext.Current.CancellationToken), withCounts);
        Assert.Same(encoder, embedder.Encoder);
        Assert.Throws<ArgumentException>(() => embedder.EmbedBatch(texts, new int[2], TestContext.Current.CancellationToken));
    }

    [Fact]
    public void NullTextIsRejected()
    {
        using var encoder = new E5Encoder(OnnxModelSession.Create(TestModels.FakeE5, new OnnxSessionOptions { ComputeModelSha256 = false }), ownsSession: true);
        var embedder = new E5TextEmbedder(new RecordingTokenizer(_ => [0, 2]), encoder);
        Assert.Throws<ArgumentException>(() => embedder.EmbedBatch(["a", null!], TestContext.Current.CancellationToken));
    }
}
