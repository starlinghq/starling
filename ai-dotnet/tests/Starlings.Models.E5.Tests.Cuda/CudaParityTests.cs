using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.E5.Tests.Cuda;

internal static class CudaGate
{
    public const string OptIn = "STARLING_CUDA_TESTS";

    /// <summary>Null when CUDA tests should run; otherwise the skip reason. Throws on an unsafe opt-in.</summary>
    public static string? Check()
    {
        if (Environment.GetEnvironmentVariable(OptIn) != "1")
        {
            return $"CUDA tests are opt-in: set {OptIn}=1, CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES to the GPU to use.";
        }

        // Never let the CUDA runtime pick a default device on a shared workstation.
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES")) ||
            Environment.GetEnvironmentVariable("CUDA_DEVICE_ORDER") != "PCI_BUS_ID")
        {
            throw new InvalidOperationException($"{OptIn}=1 requires CUDA_DEVICE_ORDER=PCI_BUS_ID and an explicit CUDA_VISIBLE_DEVICES.");
        }

        return null;
    }
}

public sealed class E5CudaFixture() : E5ModelFixture(
    new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cuda, CudaDeviceId = 0 },
    CudaGate.Check);

/// <summary>
/// Profile <c>multilingual-e5-small/onnx-fp32-cuda</c> (<c>use_tf32=0</c>): the real model on the CUDA execution
/// provider against the protected embeddings, using the fixture ids and the CUDA profile's binding tolerances.
/// </summary>
public class CudaParityTests(E5CudaFixture fixture, ITestOutputHelper output) : IClassFixture<E5CudaFixture>
{
    private static readonly E5Tolerances Tolerances = E5Tolerances.Load(E5TestAssets.CudaFixtureDirectory);

    [Fact]
    public void SessionUsesCudaWithTf32Disabled()
    {
        var d = fixture.Encoder.Session.Diagnostics;
        output.WriteLine($"ORT {d.OrtVersion}; available {string.Join(", ", d.AvailableProviders)}; session {string.Join(", ", d.SessionProviders)}");
        output.WriteLine($"CUDA_VISIBLE_DEVICES={Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES")}; provider options: {string.Join(", ", d.ProviderOptions.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => $"{kv.Key}={kv.Value}"))}");
        Assert.Equal(OnnxExecutionProvider.Cuda, d.RequestedProvider);
        Assert.Equal(OnnxRuntimeInfo.CudaProviderName, d.SessionProviders[0]);
        Assert.Equal("0", d.ProviderOptions["use_tf32"]);
        Assert.Equal("0", d.ProviderOptions["device_id"]);
        Assert.Equal(E5TestAssets.ModelSha256, d.ModelSha256);
    }

    [Fact]
    public void ParityWithProtectedEmbeddings()
    {
        var encoder = fixture.Encoder;
        var singles = E5ParityChecks.SingleInputs(encoder, Tolerances, output);
        E5ParityChecks.PaddedBatchFixture(encoder, Tolerances, singles, output);
        E5ParityChecks.AllCasesInBatches(encoder, 16, Tolerances, singles, output);
    }

    [Fact]
    public void PlacementMatchesTheAcceptedM0Placement()
    {
        var encoder = fixture.Encoder;
        encoder.Embed(new EncoderInput(E5Fixtures.Cases[2].InputIds), TestContext.Current.CancellationToken);
        var placement = encoder.Session.Placement;
        Assert.NotNull(placement);
        foreach (var (provider, ops) in placement.OperatorsByProvider)
        {
            output.WriteLine($"{provider}: {placement.NodesByProvider[provider]} nodes; " + string.Join(", ", ops.OrderBy(o => o.Key, StringComparer.Ordinal).Select(o => $"{o.Key} {o.Value}")));
        }

        output.WriteLine("CPU nodes: " + string.Join(", ", placement.NodeNamesByProvider.GetValueOrDefault(OnnxRuntimeInfo.CpuProviderName) ?? []));

        // Accepted expectation (PROPOSED.md decision 2): ORT puts two shape-path nodes on CPU by design.
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(E5TestAssets.CudaFixtureDirectory, "placement.json")));
        var accepted = doc.RootElement.GetProperty("ort-cuda-fp32-tf32-off");
        var nodes = accepted.GetProperty("nodes_per_provider");
        var cpuOps = accepted.GetProperty("ops_per_provider").GetProperty(OnnxRuntimeInfo.CpuProviderName)
            .EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetInt32(), StringComparer.Ordinal);

        Assert.Equal(nodes.GetProperty(OnnxRuntimeInfo.CudaProviderName).GetInt32(), placement.NodesByProvider[OnnxRuntimeInfo.CudaProviderName]);
        Assert.Equal(nodes.GetProperty(OnnxRuntimeInfo.CpuProviderName).GetInt32(), placement.NodesByProvider[OnnxRuntimeInfo.CpuProviderName]);
        Assert.Equal(cpuOps, placement.OperatorsByProvider[OnnxRuntimeInfo.CpuProviderName]);
    }

    [Fact]
    public void ConcurrentCallsOnSharedEncoderMatchSequentialResults()
    {
        var encoder = fixture.Encoder;
        var cases = E5Fixtures.Cases.Where(c => c.InputIds.Length <= 128).ToList();
        var sequential = cases.Select(c => encoder.Embed(new EncoderInput(c.InputIds), TestContext.Current.CancellationToken)).ToList();
        var stats = new ErrorStats[4];
        Parallel.For(0, stats.Length, t =>
        {
            stats[t] = new ErrorStats();
            for (int i = 0; i < cases.Count; i++)
            {
                int k = (i + (t * 7)) % cases.Count;
                stats[t].Add(encoder.Embed(new EncoderInput(cases[k].InputIds)), sequential[k]);
            }
        });

        double max = stats.Max(s => s.Max);
        output.WriteLine($"4 threads x {cases.Count} cases on one CUDA encoder vs sequential: max abs {max:E3}");
        Assert.True(max <= Tolerances.PaddedBatchVsSingleMaxAbs);
    }

    [Fact]
    public void CancellationStopsALargeBatchAndEncoderStaysUsable()
    {
        var encoder = fixture.Encoder;
        var longCase = E5Fixtures.Cases.Single(c => c.Id == "long-english");
        var inputs = Enumerable.Repeat(new EncoderInput(longCase.InputIds), 96).ToList();

        var full = Stopwatch.StartNew();
        encoder.EmbedBatch(inputs, TestContext.Current.CancellationToken);
        full.Stop();

        using var cts = new CancellationTokenSource();

        // The cancellation time is read on the cancelling thread before Cancel() and read here after joining it, not
        // from a callback registered on the token: callbacks run newest first, so the run's own (later) cancellation
        // callback can stop it before a recorder registered here has run.
        var sw = Stopwatch.StartNew();
        var cancelled = new StrongBox<TimeSpan>();
        var canceller = CancelAfter(cts, sw, full.Elapsed / 4, cancelled);
        TimeSpan returnedAt;
        try
        {
            Assert.ThrowsAny<OperationCanceledException>(() => encoder.EmbedBatch(inputs, cts.Token));
        }
        finally
        {
            returnedAt = sw.Elapsed;
            canceller.Join();
        }

        var cancelledAt = cancelled.Value;
        var latency = returnedAt - cancelledAt;
        output.WriteLine($"96 x 512 (3 chunks of 32): uncancelled {full.Elapsed.TotalMilliseconds:F0} ms; cancelled at {cancelledAt.TotalMilliseconds:F0} ms; threw {latency.TotalMilliseconds:F1} ms later");
        Assert.True(returnedAt < full.Elapsed, "the cancelled run took as long as an uncancelled one");

        var c0 = E5Fixtures.Cases[0];
        var stats = new ErrorStats();
        stats.Add(encoder.Embed(new EncoderInput(c0.InputIds), TestContext.Current.CancellationToken), c0.Embedding);
        Assert.True(stats.Max <= Tolerances.EmbeddingMaxAbs, $"after cancellation: {stats}");
    }

    /// <summary>
    /// Regression (docs/evidence/gliner2-cuda-wrong-output/; same shared session code as GLiNER2): a cancelled call
    /// returns only after the device work it queued has finished, and the encoder then gives the protected embedding.
    /// </summary>
    [Fact]
    public void ACancelledCallLeavesNoDeviceWorkQueued()
    {
        var encoder = fixture.Encoder;
        var longCase = E5Fixtures.Cases.Single(c => c.Id == "long-english");
        var inputs = Enumerable.Repeat(new EncoderInput(longCase.InputIds), 32).ToList();
        var probe = CudaQueueProbe.CancelledCalls(ct => encoder.EmbedBatch(inputs, ct), cancellations: 300);
        output.WriteLine($"32 x 512 (one chunk): {probe}");
        Assert.True(probe.Pending == 0, probe.ToString());
        Assert.True(probe.Drains.Count >= 100, probe.ToString());

        var c0 = E5Fixtures.Cases[0];
        var stats = new ErrorStats();
        stats.Add(encoder.Embed(new EncoderInput(c0.InputIds), TestContext.Current.CancellationToken), c0.Embedding);
        Assert.True(stats.Max <= Tolerances.EmbeddingMaxAbs, $"after cancellation: {stats}");
    }

    // Cancels on a dedicated thread after the delay, recording the elapsed time just before Cancel(). Join the thread
    // before reading the result.
    private static Thread CancelAfter(CancellationTokenSource cts, Stopwatch sw, TimeSpan delay, StrongBox<TimeSpan> cancelledAt)
    {
        var thread = new Thread(() =>
        {
            while (sw.Elapsed < delay)
            {
                Thread.Sleep(1);
            }

            cancelledAt.Value = sw.Elapsed;
            cts.Cancel();
        })
        { IsBackground = true, Name = "canceller" };
        thread.Start();
        return thread;
    }
}

/// <summary>
/// Provider failures must surface at session creation, never as a silent CPU fallback.
/// </summary>
public class CudaUnavailableTests(ITestOutputHelper output)
{
    /// <summary>
    /// Run from a copy of the test output with <c>runtimes/linux-x64/native/libonnxruntime_providers_cuda.so</c>
    /// deleted (or on a machine without the CUDA 13 runtime) and STARLING_EXPECT_CUDA_UNAVAILABLE=1.
    /// </summary>
    [Fact]
    public void MissingCudaProviderLibraryFailsClearly()
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("STARLING_EXPECT_CUDA_UNAVAILABLE") == "1", "Set STARLING_EXPECT_CUDA_UNAVAILABLE=1 in an environment where the CUDA provider cannot load.");
        var (path, reason) = E5TestAssets.FindModel();
        Assert.SkipWhen(path is null, reason);

        var ex = Assert.Throws<OnnxProviderUnavailableException>(() => E5Encoder.Create(path!, new E5EncoderOptions
        {
            Session = new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cuda, ComputeModelSha256 = false },
        }));
        output.WriteLine(ex.Message);
    }

    [Fact]
    public void DeviceThatIsNotVisibleFailsClearly()
    {
        string? skip = CudaGate.Check();
        Assert.SkipWhen(skip is not null, skip ?? string.Empty);
        int visible = Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES")!.Split(',').Length;

        var ex = Assert.Throws<OnnxProviderUnavailableException>(() => OnnxModelSession.Create(
            E5TestAssets.FindModel().Path ?? throw new InvalidOperationException("model missing"),
            new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cuda, CudaDeviceId = visible, ComputeModelSha256 = false }));
        output.WriteLine($"device_id {visible} with CUDA_VISIBLE_DEVICES={Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES")}: {ex.Message}");
    }
}
