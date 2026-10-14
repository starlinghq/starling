using Microsoft.ML.OnnxRuntime;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.E5.Tests;

/// <summary>
/// Gate G4 operational tests (plan 10.2) for <c>multilingual-e5-small</c> that the other classes do not cover: failed
/// initialization, malformed lock and snapshot, unsupported provider, queue saturation and disposal during in-flight
/// work. Cancellation, timed-out requests with cancellation latency, and cancellation then reuse on the real model are in
/// <see cref="CpuParityTests"/>; repeated load and dispose in <see cref="ModelMemoryTests"/>; the one-hour soak is
/// ai-dotnet/tools/Starlings.Soak.Tool. The weight-free tests run in CI; the real-model ones skip without the model.
/// </summary>
public sealed class OperationalTests(ITestOutputHelper output) : IDisposable
{
    private static readonly OnnxSessionOptions Fast = new() { IntraOpNumThreads = 1, ComputeModelSha256 = false };
    private readonly string _dir = Directory.CreateTempSubdirectory("starlings-e5-g4-").FullName;

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
        var options = new E5EncoderOptions { Session = Fast };
        Assert.Throws<FileNotFoundException>(() => E5Encoder.Create(Path.Combine(_dir, "absent.onnx"), options));
        Assert.Throws<OnnxRuntimeException>(() => E5Encoder.Create(Write("garbage.onnx", [.. Enumerable.Range(0, 4096).Select(i => (byte)(i * 31))]), options));
        Assert.Throws<OnnxRuntimeException>(() => E5Encoder.Create(Write("truncated.onnx", TestModels.FakeE5[..(TestModels.FakeE5.Length / 2)]), options));
        Assert.Throws<OnnxRuntimeException>(() => E5Encoder.Create(Write("empty.onnx", []), options));

        // A valid graph that is not e5, and the right graph with the wrong digest.
        Assert.Throws<NotSupportedException>(() => E5Encoder.Create(Write("add.onnx", TestModels.Add), options));
        string fake = Write("fake-e5.onnx", TestModels.FakeE5);
        Assert.Throws<OnnxModelIntegrityException>(() => E5Encoder.Create(fake, new E5EncoderOptions { Session = Fast with { ExpectedModelSha256 = E5ModelAssets.ModelSha256 } }));

        // A corrupt tokenizer.json is rejected by its digest before anything else loads.
        Assert.Throws<InvalidDataException>(() => E5Tokenizer.FromFile(Write("tokenizer.json", "{}"u8.ToArray()), E5ModelAssets.TokenizerSha256));

        using var encoder = E5Encoder.Create(fake, options);
        Assert.Equal(4, encoder.Embed(new EncoderInput(new long[] { 0, 5, 2 }), Ct).Length);
    }

    [Fact]
    public void UnsupportedProvidersAreRejectedNotIgnored()
    {
        string fake = Write("fake-e5.onnx", TestModels.FakeE5);

        // This project loads the CPU native package: CUDA fails the load; there is no silent CPU fallback.
        var ex = Assert.Throws<OnnxProviderUnavailableException>(() => E5Encoder.Create(fake, new E5EncoderOptions { Session = Fast with { Provider = OnnxExecutionProvider.Cuda } }));
        output.WriteLine(ex.Message);
        Assert.Throws<ArgumentOutOfRangeException>(() => E5Encoder.Create(fake, new E5EncoderOptions { Session = Fast with { Provider = (OnnxExecutionProvider)42 } }));
        Assert.Throws<ArgumentOutOfRangeException>(() => E5Encoder.Create(fake, new E5EncoderOptions { Session = Fast with { CudaDeviceId = -1 } }));
    }

    [Fact]
    public void MalformedLocksAndSnapshotsAreRejected()
    {
        OperationalChecks.MalformedLocksAreRejected(E5ModelAssets.Lock, output);
        OperationalChecks.WrongSnapshotsAreRejected(E5ModelAssets.Lock, dir => E5ModelAssets.OpenDirectory(dir, Acquisition.CacheVerification.Size));
    }

    [Fact]
    public void SaturatedCallersOnOneEncoderAllComplete()
    {
        using var encoder = new E5Encoder(OnnxModelSession.Create(TestModels.FakeE5, Fast), ownsSession: true);
        OperationalChecks.SaturatedCallersComplete(
            32,
            (i, ct) => encoder.EmbedBatch([new EncoderInput(new long[] { 0, i, i + 1, 2 }), new EncoderInput(new long[] { 0, 7 * i, 2 })], ct),
            (expected, actual, i) => Assert.Equal(expected, actual),
            output,
            callsPerCaller: 50);
    }

    [Fact]
    public void DisposalDuringInFlightWorkFinishesOrThrowsObjectDisposed()
    {
        var encoder = new E5Encoder(OnnxModelSession.Create(TestModels.FakeE5, Fast), new E5EncoderOptions { MaxBatchSize = 1 }, ownsSession: true);
        var inputs = Enumerable.Range(0, 2000).Select(i => new EncoderInput(new long[] { 0, i, 2 })).ToList();
        OperationalChecks.DisposeDuringInFlightWork(
            () => encoder.EmbedBatch(inputs, Ct),
            encoder.Dispose,
            () => encoder.Embed(new EncoderInput(new long[] { 0, 2 }), Ct),
            output);
    }

    [Fact]
    public void RealModelSaturatedCallersMatchTheirSingleCallResults()
    {
        var (path, reason) = E5TestAssets.FindModel();
        Assert.SkipWhen(path is null, reason);
        using var encoder = E5Encoder.Create(path!, new E5EncoderOptions { Session = new OnnxSessionOptions { ComputeModelSha256 = false } });
        var cases = E5Fixtures.Cases.Where(c => c.InputIds.Length <= 64).Take(8).ToList();
        OperationalChecks.SaturatedCallersComplete(
            cases.Count,
            (i, ct) => encoder.Embed(new EncoderInput(cases[i].InputIds), ct),
            (expected, actual, i) =>
            {
                var stats = new ErrorStats();
                stats.Add(actual, expected);
                Assert.True(stats.Max <= 1e-5, $"case {cases[i].Id}: {stats}");
            },
            output,
            callers: Environment.ProcessorCount * 2,
            callsPerCaller: 3);
    }

    [Fact]
    public void RealModelDisposalDuringALargeBatch()
    {
        var (path, reason) = E5TestAssets.FindModel();
        Assert.SkipWhen(path is null, reason);
        var encoder = E5Encoder.Create(path!, new E5EncoderOptions { MaxBatchSize = 4, Session = new OnnxSessionOptions { ComputeModelSha256 = false } });
        var longCase = E5Fixtures.Cases.Single(c => c.Id == "long-english");
        var inputs = Enumerable.Repeat(new EncoderInput(longCase.InputIds), 32).ToList();
        OperationalChecks.DisposeDuringInFlightWork(
            () => encoder.EmbedBatch(inputs, Ct),
            encoder.Dispose,
            () => encoder.Embed(new EncoderInput(longCase.InputIds), Ct),
            output,
            settle: TimeSpan.FromMilliseconds(300));
    }
}
