using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests.Cuda;

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

public sealed class EmbeddingGemma2CudaFixture() : EmbeddingGemma2ModelFixture(
    new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cuda, CudaDeviceId = 0 },
    CudaGate.Check);

/// <summary>
/// Profile <c>embeddinggemma-2/onnx-fp32-cuda</c>: the locked graph on the CUDA execution provider (<c>use_tf32=0</c>,
/// forced by Runtime.Onnx) against the CPU PyTorch reference fixtures, with the profile's protected tolerances.yaml.
/// Small batches: the GPU is shared.
/// </summary>
public class CudaParityTests(EmbeddingGemma2CudaFixture fixture, ITestOutputHelper output) : IClassFixture<EmbeddingGemma2CudaFixture>
{
    private static readonly EmbeddingGemma2Tolerances Tolerances = EmbeddingGemma2Tolerances.Load("onnx-fp32-cuda");

    [Fact]
    public void SessionUsesCudaWithTf32Disabled()
    {
        var d = fixture.Encoder.Session.Diagnostics;
        output.WriteLine($"ORT {d.OrtVersion}; session {string.Join(", ", d.SessionProviders)}; CUDA_VISIBLE_DEVICES={Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES")}");
        output.WriteLine("provider options: " + string.Join(", ", d.ProviderOptions.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => $"{kv.Key}={kv.Value}")));
        Assert.Equal(OnnxRuntimeInfo.CudaProviderName, d.SessionProviders[0]);
        Assert.Equal("0", d.ProviderOptions["use_tf32"]);
        Assert.Equal(EmbeddingGemma2ModelAssets.ModelSha256, d.ModelSha256);
    }

    [Fact]
    public void ParityWithTheReference()
    {
        var encoder = fixture.Encoder;
        var singles = EmbeddingGemma2ParityChecks.SingleRows(encoder, fixture.Rows, Tolerances, output);
        EmbeddingGemma2ParityChecks.PaddedBatchFixture(encoder, fixture.Rows, Tolerances, singles, output);
        EmbeddingGemma2ParityChecks.AllRowsInBatches(encoder, fixture.Rows, 4, Tolerances, singles, output);
        EmbeddingGemma2ParityChecks.Matryoshka(encoder, fixture.Rows, Tolerances, singles, output);
    }

    [Fact]
    public void TextParityWithTheOwnedTokenizer() =>
        EmbeddingGemma2ParityChecks.TextRows(fixture.Encoder, EmbeddingGemma2TestAssets.Tokenizer, fixture.Rows, Tolerances, output);

    /// <summary>
    /// Capability <c>prompts_task</c>: every one of the 20 named prompts on four public texts (proposed fixture from
    /// sentence-transformers, judged with this profile's protected text bounds).
    /// </summary>
    [Fact]
    public void TaskPromptParityWithTheOwnedTokenizer()
    {
        var rows = EmbeddingGemma2Fixtures.LoadTaskPromptRows();
        Assert.Equal(80, rows.Count);
        Assert.Equal(EmbeddingGemma2Prompt.Named.Keys.Order(StringComparer.Ordinal), rows.Select(r => r.Prompt).Distinct().Order(StringComparer.Ordinal));
        Assert.All(rows, r => Assert.Equal(EmbeddingGemma2Prompt.Named[r.Prompt], r.PromptText));
        EmbeddingGemma2ParityChecks.TextRows(fixture.Encoder, EmbeddingGemma2TestAssets.Tokenizer, rows, Tolerances, output);
    }

    [Fact]
    public void BatchIndependenceAcrossTheSlidingWindow() =>
        EmbeddingGemma2ParityChecks.SlidingWindowBatchIndependence(fixture.Encoder, fixture.Rows, Tolerances, output);

    [Fact]
    public void InputsOver8192TokensAreRejected() =>
        EmbeddingGemma2ParityChecks.InputsOver8192TokensAreRejected(fixture.Encoder, fixture.Rows, output);

    [Fact]
    public void PlacementPutsMostNodesOnCuda()
    {
        fixture.Encoder.Embed(new EmbeddingGemma2Input(fixture.Rows[0].InputIds), TestContext.Current.CancellationToken);
        var placement = fixture.Encoder.Session.Placement;
        Assert.NotNull(placement);
        foreach (var (provider, count) in placement.NodesByProvider)
        {
            output.WriteLine($"{provider}: {count} nodes");
        }

        // Spike (onnx-fp32-cuda/parity.json): 2,423 nodes on CUDA, 130 shape-arithmetic nodes on CPU.
        int cuda = placement.NodesByProvider.GetValueOrDefault(OnnxRuntimeInfo.CudaProviderName);
        int cpu = placement.NodesByProvider.GetValueOrDefault(OnnxRuntimeInfo.CpuProviderName);
        Assert.True(cuda > 10 * cpu, $"CUDA {cuda} nodes, CPU {cpu} nodes");
    }

    /// <summary>
    /// Capability <c>concurrent_calls</c> over CUDA: 12 rows on four threads, three rounds, give the sequential results
    /// bit for bit.
    /// </summary>
    [Fact]
    public void ConcurrentCallsGiveTheSequentialResults()
    {
        var encoder = fixture.Encoder;
        var rows = fixture.Rows.Where(r => r.InputIds.Length < 200).Take(12).ToList();
        var ct = TestContext.Current.CancellationToken;
        var sequential = rows.Select(r => encoder.Embed(new EmbeddingGemma2Input(r.InputIds), ct)).ToList();
        for (int round = 0; round < 3; round++)
        {
            var parallel = new float[rows.Count][];
            Parallel.For(0, rows.Count, new ParallelOptions { MaxDegreeOfParallelism = 4, CancellationToken = ct }, i =>
                parallel[i] = encoder.Embed(new EmbeddingGemma2Input(rows[i].InputIds), ct));
            for (int i = 0; i < rows.Count; i++)
            {
                Assert.Equal(sequential[i], parallel[i]);
            }
        }

        output.WriteLine($"{rows.Count} rows x 3 rounds on 4 threads over CUDA: equal to the sequential run");
    }

    [Fact]
    public void CancellationDuringARunThrowsAndTheEncoderStaysUsable()
    {
        var encoder = fixture.Encoder;
        var longRow = fixture.Rows.MaxBy(r => r.InputIds.Length)!;
        var inputs = Enumerable.Repeat(new EmbeddingGemma2Input(longRow.InputIds), 64).ToList();
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(50));
        Assert.ThrowsAny<OperationCanceledException>(() => encoder.EmbedBatch(inputs, cts.Token));
        var after = encoder.Embed(new EmbeddingGemma2Input(fixture.Rows[5].InputIds), TestContext.Current.CancellationToken);
        var stats = new EmbeddingGemma2ErrorStats();
        stats.Add(fixture.Rows[5].Key, after, fixture.Rows[5].Embedding);
        Assert.True(stats.Max <= Tolerances.EmbeddingMaxAbs);
    }

    /// <summary>
    /// Regression (docs/evidence/gliner2-cuda-wrong-output/; same shared session code as GLiNER2): a cancelled call
    /// returns only after the device work it queued has finished, and the encoder then gives the reference embedding.
    /// </summary>
    [Fact]
    public void ACancelledCallLeavesNoDeviceWorkQueued()
    {
        var encoder = fixture.Encoder;
        var longRow = fixture.Rows.MaxBy(r => r.InputIds.Length)!;
        var inputs = Enumerable.Repeat(new EmbeddingGemma2Input(longRow.InputIds), 4).ToList();
        var probe = CudaQueueProbe.CancelledCalls(ct => encoder.EmbedBatch(inputs, ct), cancellations: 150);
        output.WriteLine($"4 x {longRow.InputIds.Length} tokens (one chunk): {probe}");
        Assert.True(probe.Drains.Count >= 75, probe.ToString());
        Assert.Equal(0, probe.Pending);

        var after = encoder.Embed(new EmbeddingGemma2Input(fixture.Rows[5].InputIds), TestContext.Current.CancellationToken);
        var stats = new EmbeddingGemma2ErrorStats();
        stats.Add(fixture.Rows[5].Key, after, fixture.Rows[5].Embedding);
        Assert.True(stats.Max <= Tolerances.EmbeddingMaxAbs);
    }
}
