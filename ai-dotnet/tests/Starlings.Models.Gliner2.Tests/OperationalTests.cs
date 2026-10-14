using Microsoft.ML.OnnxRuntime;
using Starlings.Acquisition;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.Gliner2.Tests;

/// <summary>
/// Gate G4 operational tests (plan 10.2) for <c>gliner2-base-v1</c> that the other classes do not cover: failed
/// initialization, malformed lock and snapshot, unsupported provider, queue saturation, disposal during in-flight work,
/// and timed-out requests with their cancellation latency. Cancellation then reuse and concurrency are in
/// <see cref="ExtractorBehaviorTests"/> and <see cref="CpuParityTests"/>; the one-hour soak with repeated load/unload is
/// ai-dotnet/tools/Starlings.Soak.Tool. The weight-free tests need only the pinned tokenizer.json.
/// </summary>
public sealed class OperationalTests(ITestOutputHelper output) : IDisposable
{
    private static readonly OnnxSessionOptions Fast = new() { IntraOpNumThreads = 1, ComputeModelSha256 = false };
    private static readonly string[] Labels = ["person", "city"];
    private readonly string _dir = Directory.CreateTempSubdirectory("starlings-gliner2-g4-").FullName;

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private static string Tokenizer()
    {
        var (tokenizer, reason) = Gliner2TestAssets.FindTokenizer();
        Assert.SkipWhen(tokenizer is null, reason);
        return tokenizer!;
    }

    private string Write(string name, byte[] bytes)
    {
        string path = Path.Combine(_dir, name);
        File.WriteAllBytes(path, bytes);
        return path;
    }

    private static Gliner2EntityExtractor Fake(string tokenizer, Gliner2ExtractorOptions? options = null) =>
        Gliner2EntityExtractor.Create(OnnxModelSession.Create(TestModels.FakeGliner2, Fast), tokenizer, options, ownsSession: true);

    [Fact]
    public void FailedInitializationIsTypedAndTheNextLoadWorks()
    {
        string tokenizer = Tokenizer();
        var options = new Gliner2ExtractorOptions { Session = Fast };
        Assert.Throws<FileNotFoundException>(() => Gliner2EntityExtractor.Create(Path.Combine(_dir, "absent.onnx"), tokenizer, options));
        Assert.Throws<OnnxRuntimeException>(() => Gliner2EntityExtractor.Create(Write("garbage.onnx", [.. Enumerable.Range(0, 4096).Select(i => (byte)(i * 31))]), tokenizer, options));
        Assert.Throws<OnnxRuntimeException>(() => Gliner2EntityExtractor.Create(Write("truncated.onnx", TestModels.FakeGliner2[..(TestModels.FakeGliner2.Length / 2)]), tokenizer, options));
        Assert.Throws<NotSupportedException>(() => Gliner2EntityExtractor.Create(Write("add.onnx", TestModels.Add), tokenizer, options));

        // The files overload pins the graph digest: the right signature with other bytes is an integrity failure.
        string fake = Write("fake.onnx", TestModels.FakeGliner2);
        Assert.Throws<OnnxModelIntegrityException>(() => Gliner2EntityExtractor.Create(fake, tokenizer, new Gliner2ExtractorOptions { Session = Fast with { ExpectedModelSha256 = Gliner2ModelAssets.ModelSha256 } }));
        Assert.Throws<FileNotFoundException>(() => Gliner2EntityExtractor.Create(fake, Path.Combine(_dir, "absent.json"), options));

        using var extractor = Gliner2EntityExtractor.Create(fake, tokenizer, options);
        Assert.NotEmpty(extractor.Extract("Ann met Bob.", Labels, cancellationToken: Ct).Entities);
    }

    [Fact]
    public void UnsupportedProvidersAndSettingsAreRejectedNotIgnored()
    {
        string tokenizer = Tokenizer();
        string fake = Write("fake.onnx", TestModels.FakeGliner2);
        var ex = Assert.Throws<OnnxProviderUnavailableException>(() => Gliner2EntityExtractor.Create(fake, tokenizer, new Gliner2ExtractorOptions { Session = Fast with { Provider = OnnxExecutionProvider.Cuda } }));
        output.WriteLine(ex.Message);
        Assert.Throws<ArgumentOutOfRangeException>(() => Gliner2EntityExtractor.Create(fake, tokenizer, new Gliner2ExtractorOptions { Session = Fast with { Provider = (OnnxExecutionProvider)42 } }));

        // Settings outside the profile are rejected, not clamped.
        Assert.Throws<ArgumentOutOfRangeException>(() => Gliner2EntityExtractor.Create(fake, tokenizer, new Gliner2ExtractorOptions { Session = Fast, MaxSequenceLength = 513 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => Gliner2EntityExtractor.Create(fake, tokenizer, new Gliner2ExtractorOptions { Session = Fast, Threshold = 1.5f }));
    }

    [Fact]
    public void MalformedLocksAndSnapshotsAreRejected()
    {
        OperationalChecks.MalformedLocksAreRejected(Gliner2ModelAssets.Lock, output);
        OperationalChecks.WrongSnapshotsAreRejected(Gliner2ModelAssets.Lock, dir => Gliner2ModelAssets.OpenDirectory(dir, CacheVerification.Size));
    }

    [Fact]
    public void SaturatedCallersOnOneExtractorAllComplete()
    {
        using var extractor = Fake(Tokenizer());
        OperationalChecks.SaturatedCallersComplete(
            16,
            (i, ct) => extractor.Extract($"Person {i} met person {i * 7} in city {i % 5}.", Labels, cancellationToken: ct).Entities,
            (expected, actual, i) => Assert.Equal(expected, actual),
            output,
            callsPerCaller: 20);
    }

    [Fact]
    public void DisposalDuringInFlightWorkFinishesOrThrowsObjectDisposed()
    {
        var extractor = Fake(Tokenizer(), new Gliner2ExtractorOptions { MaxBatchSize = 1 });
        var texts = Enumerable.Range(0, 1000).Select(i => $"Person {i} lives in city {i}.").ToList();
        OperationalChecks.DisposeDuringInFlightWork(
            () => extractor.ExtractBatch(texts, Labels, cancellationToken: Ct),
            extractor.Dispose,
            () => extractor.Extract("Ann.", Labels, cancellationToken: Ct),
            output);
    }

    [Fact]
    public void RealModelTimedOutRequestsReportCancellationLatencyThenReuse()
    {
        var (files, reason) = Gliner2TestAssets.FindModel();
        Assert.SkipWhen(files is null, reason);
        using var extractor = Gliner2EntityExtractor.Create(files!, new Gliner2ExtractorOptions { Session = new OnnxSessionOptions { ComputeModelSha256 = false } });
        string text = string.Join(' ', Enumerable.Repeat("Anna Larsen met Peter Holm in Copenhagen to discuss the harbour plan with the city council.", 12));
        var texts = Enumerable.Repeat(text, 32).ToList();
        string[] labels = ["person", "location", "organization"];
        var expected = extractor.Extract("Anna Larsen met Peter Holm in Copenhagen.", labels, cancellationToken: Ct).Entities;
        OperationalChecks.TimedOutRequestsThenReuse(
            ct => extractor.ExtractBatch(texts, labels, cancellationToken: ct),
            () => Assert.Equal(expected, extractor.Extract("Anna Larsen met Peter Holm in Copenhagen.", labels, cancellationToken: Ct).Entities),
            TimeSpan.FromMilliseconds(100),
            TimeSpan.FromSeconds(2),
            output);
    }

    [Fact]
    public void RealModelDisposalDuringALargeBatch()
    {
        var (files, reason) = Gliner2TestAssets.FindModel();
        Assert.SkipWhen(files is null, reason);
        var extractor = Gliner2EntityExtractor.Create(files!, new Gliner2ExtractorOptions { Session = new OnnxSessionOptions { ComputeModelSha256 = false }, MaxBatchSize = 2 });
        var texts = Enumerable.Repeat("Anna Larsen met Peter Holm in Copenhagen to discuss the harbour plan with the city council.", 64).ToList();
        OperationalChecks.DisposeDuringInFlightWork(
            () => extractor.ExtractBatch(texts, ["person", "location"], cancellationToken: Ct),
            extractor.Dispose,
            () => extractor.Extract("Ann.", ["person"], cancellationToken: Ct),
            output,
            settle: TimeSpan.FromMilliseconds(300));
    }
}
