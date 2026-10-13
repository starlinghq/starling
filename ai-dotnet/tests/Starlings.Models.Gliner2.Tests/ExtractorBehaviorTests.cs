using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.Gliner2.Tests;

/// <summary>
/// Extractor behaviour on a weight-free graph with the route A signature (TestModels.FakeGliner2:
/// <c>span_logits[b, m, w, k] = 4 - 8k + 0.001 * word_index[b, w]</c>, gate open): every word is a one-word entity of
/// every label with a confidence that grows with its token position. Needs the pinned tokenizer.json, not the model.
/// </summary>
public class ExtractorBehaviorTests
{
    private static readonly OnnxSessionOptions Fast = new() { IntraOpNumThreads = 1, ComputeModelSha256 = false };

    private static Gliner2EntityExtractor Fake(Gliner2ExtractorOptions? options = null)
    {
        var (tokenizer, reason) = Gliner2TestAssets.FindTokenizer();
        Assert.SkipWhen(tokenizer is null, reason);
        return Gliner2EntityExtractor.Create(OnnxModelSession.Create(TestModels.FakeGliner2, Fast), tokenizer!, options, ownsSession: true);
    }

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    [Fact]
    public void EveryWordIsAnEntityInDescendingPositionOrder()
    {
        using var extractor = Fake();
        var result = extractor.Extract("Ann met Bob", ["person", "place"], cancellationToken: Ct);
        Assert.True(result.GateOpen);
        Assert.Equal(["person", "place"], result.Labels);

        // Words: ann, met, bob, "." (appended, dropped from the public result). Higher token position, higher score.
        Assert.Equal(["Bob", "met", "Ann"], result.GetEntities("person").Select(e => e.Text));
        Assert.Equal(["Bob", "met", "Ann"], result.GetEntities("place").Select(e => e.Text));
        Assert.Equal(6, result.Entities.Count);
        Assert.All(result.Entities, e => Assert.Equal("Ann met Bob"[e.Start..e.End], e.Text));
        Assert.All(result.Entities, e => Assert.InRange(e.Confidence, 0.98f, 0.99f));
    }

    [Fact]
    public void LowThresholdKeepsOnlyNonOverlappingSpans()
    {
        using var extractor = Fake();

        // Two-word spans score about 0.018 > 0.01 but overlap the higher one-word spans and are suppressed. The final
        // "." is part of this input (nothing is appended), so it is an entity too.
        var result = extractor.Extract("Ann met Bob.", ["person"], 0.01f, Ct);
        Assert.Equal([".", "Bob", "met", "Ann"], result.Entities.Select(e => e.Text));
    }

    [Fact]
    public void BatchResultsEqualSingleCalls()
    {
        using var extractor = Fake(new Gliner2ExtractorOptions { MaxBatchSize = 3 });
        string[] texts = ["Ann met Bob.", "A much longer sentence with many more words than the others has.", "Hi!", "", "Zoë went to Malmö."];
        var batch = extractor.ExtractBatch(texts, ["person", "location"], cancellationToken: Ct);
        Assert.Equal(texts.Length, batch.Count);
        for (int i = 0; i < texts.Length; i++)
        {
            var single = extractor.Extract(texts[i], ["person", "location"], cancellationToken: Ct);
            Assert.Equal(single.Entities, batch[i].Entities);
            Assert.Equal(texts[i], batch[i].Text);
        }
    }

    [Fact]
    public void StagedApiEqualsExtract()
    {
        using var extractor = Fake();
        var input = extractor.Prepare("Ann met Bob.", ["person"]);
        var o = extractor.RunModel([input], Ct)[0];
        Assert.Equal(20, o.CountLogits.Length);
        Assert.Equal(1, o.PredictedCount);
        Assert.Equal((1, input.WordCount, 8), (o.LabelCount, o.WordCount, o.MaxWidth));
        Assert.Equal(extractor.Extract("Ann met Bob.", ["person"], cancellationToken: Ct).Entities, extractor.Decode(input, o).Entities);
    }

    [Fact]
    public void TextsOverTheTokenLimitAreRejectedBeforeInference()
    {
        using var extractor = Fake();
        string longText = string.Join(' ', Enumerable.Repeat("word", 600));
        var ex = Assert.Throws<Gliner2InputTooLongException>(() => extractor.ExtractBatch(["fine.", longText], ["person"], cancellationToken: Ct));
        Assert.Equal(1, ex.InputIndex);
        Assert.Equal(512, ex.MaxTokens);
        Assert.True(ex.TokenCount > 512);
        Assert.Contains("Nothing is truncated", ex.Message, StringComparison.Ordinal);
        Assert.IsAssignableFrom<ArgumentException>(ex);

        // The limit counts the schema prefix: a text that fits alone may not fit with many labels.
        using var tight = Fake(new Gliner2ExtractorOptions { MaxSequenceLength = 16 });
        Assert.Throws<Gliner2InputTooLongException>(() => tight.Extract("one two three four five six seven", ["person", "location", "organization"], cancellationToken: Ct));
        var staged = tight.Prepare("one two three four five six seven", ["person"]);
        Assert.Throws<Gliner2InputTooLongException>(() => tight.RunModel([staged], Ct));
    }

    [Fact]
    public void InvalidArgumentsAreRejected()
    {
        using var extractor = Fake();
        Assert.Throws<ArgumentException>(() => extractor.Extract("x", [], cancellationToken: Ct));
        Assert.Throws<ArgumentException>(() => extractor.Extract("x", [""], cancellationToken: Ct));
        Assert.Throws<ArgumentOutOfRangeException>(() => extractor.Extract("x", ["a"], 1.5f, Ct));
        Assert.Throws<ArgumentOutOfRangeException>(() => extractor.Extract("x", ["a"], float.NaN, Ct));
        Assert.Throws<ArgumentException>(() => extractor.Extract("bad \uD800 text", ["a"], cancellationToken: Ct));
        Assert.Throws<ArgumentException>(() => extractor.ExtractBatch(["ok", null!], ["a"], cancellationToken: Ct));
        var one = extractor.Prepare("Ann.", ["a"]);
        var two = extractor.Prepare("Ann.", ["a", "b"]);
        Assert.Throws<ArgumentException>(() => extractor.RunModel([one, two], Ct));
        var o = extractor.RunModel([one], Ct)[0];
        Assert.Throws<ArgumentException>(() => extractor.Decode(two, o));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fake(new Gliner2ExtractorOptions { MaxSequenceLength = 513 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fake(new Gliner2ExtractorOptions { MaxBatchSize = 0 }));
    }

    [Fact]
    public void AGraphWithAnotherSignatureIsRejected()
    {
        var (tokenizer, reason) = Gliner2TestAssets.FindTokenizer();
        Assert.SkipWhen(tokenizer is null, reason);
        using var session = OnnxModelSession.Create(TestModels.FakeE5, Fast);
        Assert.Throws<NotSupportedException>(() => Gliner2EntityExtractor.Create(session, tokenizer!));
        Assert.False(session.IsDisposed);
    }

    [Fact]
    public void AnotherTokenizerFileIsRejected()
    {
        string path = Path.Combine(Path.GetTempPath(), $"df-gliner2-{Guid.NewGuid():N}.json");
        File.WriteAllText(path, "{}");
        try
        {
            using var session = OnnxModelSession.Create(TestModels.FakeGliner2, Fast);
            Assert.Throws<InvalidDataException>(() => Gliner2EntityExtractor.Create(session, path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void CancelledTokenPreventsTheRunAndTheExtractorStaysUsable()
    {
        using var extractor = Fake();
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.ThrowsAny<OperationCanceledException>(() => extractor.Extract("Ann met Bob.", ["person"], cancellationToken: cts.Token));
        Assert.ThrowsAny<OperationCanceledException>(() => extractor.RunModel([extractor.Prepare("Ann.", ["person"])], cts.Token));
        Assert.NotEmpty(extractor.Extract("Ann met Bob.", ["person"], cancellationToken: Ct).Entities);
    }

    [Fact]
    public void ConcurrentCallsGiveTheSequentialResults()
    {
        using var extractor = Fake();
        string[] texts = Enumerable.Range(0, 32).Select(i => $"Person {i} met person {i * 7} in city {i % 5}.").ToArray();
        var sequential = texts.Select(t => extractor.Extract(t, ["person", "city"], cancellationToken: Ct).Entities).ToArray();
        var parallel = new IReadOnlyList<Gliner2Entity>[texts.Length];
        Parallel.For(0, texts.Length, new ParallelOptions { MaxDegreeOfParallelism = 8, CancellationToken = Ct }, i =>
            parallel[i] = extractor.Extract(texts[i], ["person", "city"], cancellationToken: Ct).Entities);
        for (int i = 0; i < texts.Length; i++)
        {
            Assert.Equal(sequential[i], parallel[i]);
        }
    }

    [Fact]
    public void DisposeIsIdempotentAndOwnershipIsRespected()
    {
        var (tokenizer, reason) = Gliner2TestAssets.FindTokenizer();
        Assert.SkipWhen(tokenizer is null, reason);
        var owned = OnnxModelSession.Create(TestModels.FakeGliner2, Fast);
        var extractor = Gliner2EntityExtractor.Create(owned, tokenizer!, ownsSession: true);
        extractor.Dispose();
        extractor.Dispose();
        Assert.True(owned.IsDisposed);
        Assert.Throws<ObjectDisposedException>(() => extractor.Extract("Ann.", ["person"], cancellationToken: Ct));
        Assert.Throws<ObjectDisposedException>(() => extractor.Prepare("Ann.", ["person"]));

        using var borrowed = OnnxModelSession.Create(TestModels.FakeGliner2, Fast);
        Gliner2EntityExtractor.Create(borrowed, tokenizer!).Dispose();
        Assert.False(borrowed.IsDisposed);
    }
}
