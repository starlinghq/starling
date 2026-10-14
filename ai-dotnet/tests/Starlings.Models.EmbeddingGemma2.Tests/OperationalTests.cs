using Microsoft.ML.OnnxRuntime;
using Starlings.Acquisition;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>
/// Gate G4 operational tests (plan 10.2) for <c>embeddinggemma-2</c> that the other classes do not cover: failed
/// initialization (text and image graphs), malformed locks and snapshots, unsupported provider, queue saturation,
/// disposal during in-flight work, and timed-out requests with their cancellation latency. Cancellation then reuse and
/// concurrency are in <see cref="EncoderBehaviorTests"/> and <see cref="CpuParityTests"/>; repeated load and dispose in
/// <see cref="ModelMemoryTests"/>; the one-hour soak is ai-dotnet/tools/Starlings.Soak.Tool. The weight-free tests run in CI.
/// </summary>
public sealed class OperationalTests(ITestOutputHelper output) : IDisposable
{
    private static readonly OnnxSessionOptions Fast = new() { IntraOpNumThreads = 1, ComputeModelSha256 = false };
    private readonly string _dir = Directory.CreateTempSubdirectory("starlings-eg2-g4-").FullName;

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private string Write(string name, byte[] bytes)
    {
        string path = Path.Combine(_dir, name);
        File.WriteAllBytes(path, bytes);
        return path;
    }

    [Fact]
    public void FailedInitializationIsTypedAndTheNextLoadWorks()
    {
        var options = new EmbeddingGemma2EncoderOptions { Session = Fast };
        Assert.Throws<FileNotFoundException>(() => EmbeddingGemma2Encoder.Create(Path.Combine(_dir, "absent.onnx"), options));
        Assert.Throws<OnnxRuntimeException>(() => EmbeddingGemma2Encoder.Create(Write("garbage.onnx", [.. Enumerable.Range(0, 4096).Select(i => (byte)(i * 31))]), options));
        Assert.Throws<OnnxRuntimeException>(() => EmbeddingGemma2Encoder.Create(Write("truncated.onnx", TestModels.FakeEmbeddingGemma2[..(TestModels.FakeEmbeddingGemma2.Length / 2)]), options));
        Assert.Throws<NotSupportedException>(() => EmbeddingGemma2Encoder.Create(Write("add.onnx", TestModels.Add), options));
        string fake = Write("fake.onnx", TestModels.FakeEmbeddingGemma2);
        Assert.Throws<OnnxModelIntegrityException>(() => EmbeddingGemma2Encoder.Create(fake, new EmbeddingGemma2EncoderOptions { Session = Fast with { ExpectedModelSha256 = EmbeddingGemma2ModelAssets.ModelSha256 } }));
        Assert.Throws<InvalidDataException>(() => EmbeddingGemma2Tokenizer.FromFile(Write("tokenizer.json", "{}"u8.ToArray())));

        using var encoder = EmbeddingGemma2Encoder.Create(fake, options);
        Assert.Equal(768, encoder.Embed(new EmbeddingGemma2Input(new long[] { 2, 5, 1 }), Ct).Length);
    }

    [Fact]
    public void ImageEmbedderWithAMissingOrCorruptGraphFailsTypedAndReleasesWhatItLoaded()
    {
        var (tokenizer, reason) = EmbeddingGemma2TestAssets.FindTokenizer();
        Assert.SkipWhen(tokenizer is null, reason);
        var textFiles = new EmbeddingGemma2ModelFiles(Write("text.onnx", TestModels.FakeEmbeddingGemma2), tokenizer!, Path.Combine(_dir, "prompts.json"), null!);
        var options = new EmbeddingGemma2MultimodalOptions { Session = Fast, VerifyModelSha256 = false };
        Assert.Throws<FileNotFoundException>(() => EmbeddingGemma2MultimodalEmbedder.Create(new EmbeddingGemma2ImageModelFiles(textFiles, Path.Combine(_dir, "absent-vision.onnx"), textFiles.ModelPath), options));
        Assert.Throws<OnnxRuntimeException>(() => EmbeddingGemma2MultimodalEmbedder.Create(new EmbeddingGemma2ImageModelFiles(textFiles, Write("garbage-vision.onnx", [1, 2, 3, 4]), textFiles.ModelPath), options));

        // The default options pin the graph digests.
        Assert.Throws<OnnxModelIntegrityException>(() => EmbeddingGemma2MultimodalEmbedder.Create(
            new EmbeddingGemma2ImageModelFiles(textFiles, textFiles.ModelPath, textFiles.ModelPath), options with { VerifyModelSha256 = true }));
    }

    [Fact]
    public void UnsupportedProvidersAreRejectedNotIgnored()
    {
        string fake = Write("fake.onnx", TestModels.FakeEmbeddingGemma2);
        var ex = Assert.Throws<OnnxProviderUnavailableException>(() => EmbeddingGemma2Encoder.Create(fake, new EmbeddingGemma2EncoderOptions { Session = Fast with { Provider = OnnxExecutionProvider.Cuda } }));
        output.WriteLine(ex.Message);
        Assert.Throws<ArgumentOutOfRangeException>(() => EmbeddingGemma2Encoder.Create(fake, new EmbeddingGemma2EncoderOptions { Session = Fast with { Provider = (OnnxExecutionProvider)42 } }));
        Assert.Throws<ArgumentOutOfRangeException>(() => EmbeddingGemma2Encoder.Create(fake, new EmbeddingGemma2EncoderOptions { Session = Fast, Dimension = 300 }));
    }

    [Fact]
    public void MalformedLocksAndSnapshotsAreRejected()
    {
        OperationalChecks.MalformedLocksAreRejected(EmbeddingGemma2ModelAssets.Lock, output);
        OperationalChecks.MalformedLocksAreRejected(EmbeddingGemma2ModelAssets.ImageLock, output);
        OperationalChecks.WrongSnapshotsAreRejected(EmbeddingGemma2ModelAssets.Lock, dir => EmbeddingGemma2ModelAssets.OpenDirectory(dir, CacheVerification.Size));
        OperationalChecks.WrongSnapshotsAreRejected(EmbeddingGemma2ModelAssets.ImageLock, dir => EmbeddingGemma2ModelAssets.OpenImageDirectory(dir, CacheVerification.Size));
    }

    [Fact]
    public void SaturatedCallersOnOneEncoderAllComplete()
    {
        using var encoder = EncoderBehaviorTests.FakeEncoder();
        OperationalChecks.SaturatedCallersComplete(
            24,
            (i, ct) => encoder.Embed(new EmbeddingGemma2Input(Enumerable.Range(0, 2 + i).Select(t => (long)(t + i)).ToArray()), ct),
            (expected, actual, i) => Assert.Equal(expected, actual),
            output,
            callsPerCaller: 20);
    }

    [Fact]
    public void DisposalDuringInFlightWorkFinishesOrThrowsObjectDisposed()
    {
        var encoder = EncoderBehaviorTests.FakeEncoder(new EmbeddingGemma2EncoderOptions { MaxBatchSize = 1 });
        var inputs = Enumerable.Range(0, 1000).Select(i => new EmbeddingGemma2Input(new long[] { 2, i + 3, 1 })).ToList();
        OperationalChecks.DisposeDuringInFlightWork(
            () => encoder.EmbedBatch(inputs, Ct),
            encoder.Dispose,
            () => encoder.Embed(new EmbeddingGemma2Input(new long[] { 2, 1 }), Ct),
            output);
    }

    [Fact]
    public void RealModelTimedOutRequestsReportCancellationLatencyThenReuse()
    {
        var (files, reason) = EmbeddingGemma2TestAssets.FindModel();
        Assert.SkipWhen(files is null, reason);
        using var encoder = EmbeddingGemma2Encoder.Create(files!.ModelPath, new EmbeddingGemma2EncoderOptions { Session = new OnnxSessionOptions { ComputeModelSha256 = false } });
        var longInput = new EmbeddingGemma2Input(Enumerable.Range(0, 2048).Select(i => (long)(1000 + (i % 5000))).ToArray());
        var shortInput = new EmbeddingGemma2Input(new long[] { 2, 1596, 1 });
        var expected = encoder.Embed(shortInput, Ct);
        OperationalChecks.TimedOutRequestsThenReuse(
            ct => encoder.EmbedBatch([longInput, longInput, longInput, longInput], ct),
            () => Assert.Equal(expected, encoder.Embed(shortInput, Ct)),
            TimeSpan.FromMilliseconds(150),
            // Cancellation lands at the next graph node boundary. One node over 4 x 2,048 tokens took up to 4.9 s on a
            // 12-thread CPU shared with the whole test suite (docs/evidence/g4-soak/README.md), so the bound only
            // asserts that the run stops well before an uncancelled one would finish.
            TimeSpan.FromSeconds(10),
            output);
    }

    [Fact]
    public void RealModelDisposalDuringALargeBatch()
    {
        var (files, reason) = EmbeddingGemma2TestAssets.FindModel();
        Assert.SkipWhen(files is null, reason);
        var encoder = EmbeddingGemma2Encoder.Create(files!.ModelPath, new EmbeddingGemma2EncoderOptions { MaxBatchSize = 2, Session = new OnnxSessionOptions { ComputeModelSha256 = false } });
        var input = new EmbeddingGemma2Input(Enumerable.Range(0, 512).Select(i => (long)(1000 + i)).ToArray());
        var inputs = Enumerable.Repeat(input, 16).ToList();
        OperationalChecks.DisposeDuringInFlightWork(
            () => encoder.EmbedBatch(inputs, Ct),
            encoder.Dispose,
            () => encoder.Embed(input, Ct),
            output,
            settle: TimeSpan.FromMilliseconds(500));
    }
}
