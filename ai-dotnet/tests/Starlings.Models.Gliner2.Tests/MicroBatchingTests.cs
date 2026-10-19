using Starlings.Runtime.Onnx;
using Starlings.Runtime.Onnx.Batching;
using Starlings.TestSupport;

namespace Starlings.Models.Gliner2.Tests;

/// <summary>
/// Opt-in micro-batching (<see cref="Gliner2ExtractorOptions.MicroBatching"/>, Experimental) on the weight-free GLiNER2
/// graph with the pinned tokenizer.json (skips without it). A long <see cref="MicroBatchingOptions.MaxWait"/> makes batch
/// composition deterministic: a batch is dispatched only when it holds <see cref="MicroBatchingOptions.MaxRows"/> texts.
/// </summary>
public sealed class MicroBatchingTests
{
    private static readonly OnnxSessionOptions Fast = new() { IntraOpNumThreads = 1, ComputeModelSha256 = false };
    private static readonly MicroBatchingOptions FillTwo = new() { MaxWait = TimeSpan.FromHours(1), MaxRows = 2 };
    private static readonly string[] PersonCity = ["person", "city"];
    private static readonly string[] CityPerson = ["city", "person"];

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static string Tokenizer()
    {
        var (tokenizer, reason) = Gliner2TestAssets.FindTokenizer();
        Assert.SkipWhen(tokenizer is null, reason);
        return tokenizer!;
    }

    private static Gliner2EntityExtractor Fake(MicroBatchingOptions? microBatching, int maxSequenceLength = 512) =>
        Gliner2EntityExtractor.Create(
            OnnxModelSession.Create(TestModels.FakeGliner2, Fast),
            Tokenizer(),
            new Gliner2ExtractorOptions { MicroBatching = microBatching, MaxSequenceLength = maxSequenceLength },
            ownsSession: true);

    private static Task<T> Within<T>(Task<T> task) => task.WaitAsync(TimeSpan.FromSeconds(30), Ct);

    private static void SameEntities(Gliner2ExtractionResult expected, Gliner2ExtractionResult actual) =>
        Assert.Equal(expected.Entities, actual.Entities);

    [Fact]
    public async Task OffByDefaultAndExtractAsyncStaysSynchronous()
    {
        Assert.Null(new Gliner2ExtractorOptions().MicroBatching);
        using var extractor = Fake(null);
        var task = extractor.ExtractAsync("Ann met Bob in Paris.", PersonCity, cancellationToken: Ct);
        Assert.True(task.IsCompletedSuccessfully);
        Assert.Null(extractor.MicroBatchingStatistics);
        SameEntities(extractor.Extract("Ann met Bob in Paris.", PersonCity, cancellationToken: Ct), await task);
        Assert.True((await extractor.ExtractAsync("x", ["", "a"], cancellationToken: Ct).ContinueWith(t => t.IsFaulted, TaskScheduler.Default)));
    }

    [Fact]
    public void InvalidOptionsAreRejected()
    {
        string tokenizer = Tokenizer();
        Assert.Throws<ArgumentOutOfRangeException>(() => Gliner2EntityExtractor.Create(
            OnnxModelSession.Create(TestModels.FakeGliner2, Fast), tokenizer, new Gliner2ExtractorOptions { MicroBatching = new() { MaxRows = 0 } }, ownsSession: true));
    }

    [Fact]
    public async Task OnlyTextsWithTheSameSchemaShareABatchAndEachCallerKeepsItsThreshold()
    {
        using var extractor = Fake(FillTwo);
        var descriptions = new Dictionary<string, string> { ["person"] = "a human being" };
        // Batches are dispatched only when full (two texts with one schema): A+C, B+D, E+F.
        var a = extractor.ExtractAsync("Ann met Bob in Paris.", PersonCity, 0.5f, Ct);
        var b = extractor.ExtractAsync("Cid lives in Rome.", CityPerson, null, Ct);
        var e = extractor.ExtractAsync("Dee left Oslo.", PersonCity, descriptions, null, Ct);
        var c = extractor.ExtractAsync("Eve and a longer text with more words visited Berlin.", PersonCity, 0.99f, Ct);
        var d = extractor.ExtractAsync("Fay saw Lima.", CityPerson, null, Ct);
        var f = extractor.ExtractAsync("Gus is here.", PersonCity, descriptions, 0.9f, Ct);
        await Within(Task.WhenAll(a, b, c, d, e, f));

        SameEntities(extractor.Extract("Ann met Bob in Paris.", PersonCity, 0.5f, Ct), await a);
        SameEntities(extractor.ExtractBatch(["Cid lives in Rome."], CityPerson, null, null, Ct)[0], await b);
        SameEntities(extractor.Extract("Eve and a longer text with more words visited Berlin.", PersonCity, 0.99f, Ct), await c);
        SameEntities(extractor.ExtractBatch(["Fay saw Lima."], CityPerson, null, null, Ct)[0], await d);
        SameEntities(extractor.ExtractBatch(["Dee left Oslo."], PersonCity, descriptions, null, Ct)[0], await e);
        SameEntities(extractor.ExtractBatch(["Gus is here."], PersonCity, descriptions, 0.9f, Ct)[0], await f);
        Assert.NotEmpty((await a).Entities);
        Assert.Empty((await c).Entities); // the fake graph's best score is about 0.982
        Assert.Equal(CityPerson, (await b).Labels);

        var stats = extractor.MicroBatchingStatistics!.Value;
        Assert.Equal((3L, 6L), (stats.Batches, stats.Items));
    }

    [Fact]
    public async Task ATooLongTextFailsAloneBeforeItIsQueued()
    {
        using var extractor = Fake(FillTwo, maxSequenceLength: 24);
        var error = await Assert.ThrowsAsync<Gliner2InputTooLongException>(() => Within(extractor.ExtractAsync(string.Join(' ', Enumerable.Repeat("word", 40)), PersonCity, cancellationToken: Ct)));
        Assert.Equal(0, error.InputIndex);
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => extractor.ExtractAsync("Ann.", PersonCity, 2f, Ct));
        await Within(Task.WhenAll(extractor.ExtractAsync("Ann.", PersonCity, cancellationToken: Ct), extractor.ExtractAsync("Bob.", PersonCity, cancellationToken: Ct)));
        var stats = extractor.MicroBatchingStatistics!.Value;
        Assert.Equal((1L, 2L), (stats.Batches, stats.Items));
    }

    [Fact]
    public async Task CancellingAQueuedCallLeavesTheOthers()
    {
        using var extractor = Fake(FillTwo);
        using var cts = new CancellationTokenSource();
        var cancelled = extractor.ExtractAsync("Ann met Bob.", PersonCity, cancellationToken: cts.Token);
        await cts.CancelAsync();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Within(cancelled));
        Assert.True(cancelled.IsCanceled);

        var b = extractor.ExtractAsync("Cid met Dee.", PersonCity, cancellationToken: Ct);
        var c = extractor.ExtractAsync("Eve met Fay.", PersonCity, cancellationToken: Ct);
        await Within(Task.WhenAll(b, c));
        SameEntities(extractor.Extract("Cid met Dee.", PersonCity, cancellationToken: Ct), await b);
        Assert.Equal(2, extractor.MicroBatchingStatistics!.Value.Items);
    }

    [Fact]
    public async Task DisposeFaultsQueuedCallsAndLaterCalls()
    {
        var extractor = Fake(FillTwo);
        var queued = extractor.ExtractAsync("Ann met Bob.", PersonCity, cancellationToken: Ct);
        Assert.False(queued.IsCompleted);
        extractor.Dispose();
        await Assert.ThrowsAsync<ObjectDisposedException>(() => Within(queued));
        await Assert.ThrowsAsync<ObjectDisposedException>(() => extractor.ExtractAsync("Cid.", PersonCity, cancellationToken: Ct));
    }

    [Fact]
    public void CapabilityIsExperimental() =>
        Assert.Equal(Gliner2CapabilityStatus.Experimental, Gliner2Capabilities.All["micro_batching"]);
}

/// <summary>
/// Micro-batching on the real graph (CPU): the protected fixture cases as concurrent <c>ExtractAsync</c> calls over the
/// fixture's session, coalesced by schema into padded batches; entities must equal single calls and confidences stay
/// within the protected confidence bound. Skips when the graph or fixtures are absent.
/// </summary>
public class MicroBatchingRealModelTests(Gliner2CpuFixture fixture, ITestOutputHelper output) : IClassFixture<Gliner2CpuFixture>
{
    private static readonly Gliner2Tolerances Tolerances = Gliner2Tolerances.Load("onnx-fp32-cpu");

    [Fact]
    public async Task MicroBatchedCallsEqualSingleCalls()
    {
        var direct = fixture.Extractor;
        var ct = TestContext.Current.CancellationToken;
        var (tokenizer, reason) = Gliner2TestAssets.FindTokenizer();
        Assert.SkipWhen(tokenizer is null, reason);
        using var batched = Gliner2EntityExtractor.Create(direct.Session, tokenizer!, new Gliner2ExtractorOptions
        {
            MicroBatching = new() { MaxRows = 8, MaxWait = TimeSpan.FromMilliseconds(20) },
        });
        var cases = fixture.Cases.Where(c => c.OutOfContract is null && c.InputIds.Length <= 512).ToList();
        var calls = cases.Select(c => Task.Run(() => batched.ExtractAsync(c.Text, c.LabelsRequested, c.Descriptions, c.Threshold, ct), ct)).ToArray();
        var results = await Task.WhenAll(calls);

        double confidence = 0;
        int entities = 0;
        for (int i = 0; i < cases.Count; i++)
        {
            var single = direct.ExtractBatch([cases[i].Text], cases[i].LabelsRequested, cases[i].Descriptions, cases[i].Threshold, ct)[0];
            Assert.Equal(single.Entities.Select(e => (e.Label, e.Start, e.End)), results[i].Entities.Select(e => (e.Label, e.Start, e.End)));
            for (int j = 0; j < single.Entities.Count; j++)
            {
                confidence = Math.Max(confidence, Math.Abs(single.Entities[j].Confidence - results[i].Entities[j].Confidence));
                entities++;
            }
        }

        var stats = batched.MicroBatchingStatistics!.Value;
        output.WriteLine($"micro-batched, {cases.Count} concurrent calls in {stats.Batches} batches (padding efficiency {(double)stats.Tokens / stats.TokenCells:P1}); {entities} entities identical, confidence max abs diff vs single {confidence:E2} (bound {Tolerances.ConfidenceMaxAbs:E1})");
        Assert.Equal(cases.Count, stats.Items);
        Assert.True(stats.Batches < cases.Count);
        Assert.True(confidence <= Tolerances.ConfidenceMaxAbs, $"confidence {confidence:E3}");
    }
}
