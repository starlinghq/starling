using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.Gliner2.Tests.Cuda;

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

public sealed class Gliner2CudaFixture() : Gliner2ModelFixture(
    new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cuda, CudaDeviceId = 0 },
    CudaGate.Check);

/// <summary>
/// The GLiNER2 graph on the CUDA execution provider (<c>use_tf32=0</c>, forced by Runtime.Onnx) against the CPU
/// PyTorch reference fixtures, with the profile's protected tolerances.yaml.
/// </summary>
public class CudaParityTests(Gliner2CudaFixture fixture, ITestOutputHelper output) : IClassFixture<Gliner2CudaFixture>
{
    private static readonly Gliner2Tolerances Tolerances = Gliner2Tolerances.Load("onnx-fp32-cuda");

    [Fact]
    public void SessionUsesCudaWithTf32Disabled()
    {
        var d = fixture.Extractor.Session.Diagnostics;
        output.WriteLine($"ORT {d.OrtVersion}; session {string.Join(", ", d.SessionProviders)}; CUDA_VISIBLE_DEVICES={Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES")}");
        output.WriteLine("provider options: " + string.Join(", ", d.ProviderOptions.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => $"{kv.Key}={kv.Value}")));
        Assert.Equal(OnnxRuntimeInfo.CudaProviderName, d.SessionProviders[0]);
        Assert.Equal("0", d.ProviderOptions["use_tf32"]);
        Assert.Equal(Gliner2ModelAssets.ModelSha256, d.ModelSha256);
    }

    [Fact]
    public void ParityWithTheReference()
    {
        var summary = Gliner2ModelParity.SingleCases(fixture.Extractor, fixture.Cases, Tolerances, output);
        Assert.Equal(54, summary.Cases);
        Gliner2ModelParity.PaddedBatches(fixture.Extractor, fixture.Cases, summary.Outputs, Tolerances, output);
    }

    /// <summary>
    /// Capabilities <c>count_gate_closed</c> and <c>label_descriptions</c> (proposed cases, protected bounds of this
    /// profile): 12 cases where the reference's count gate closes (no entities) and 10 with label descriptions.
    /// </summary>
    [Fact]
    public void CountGateAndDescriptionCasesMatchTheReference()
    {
        _ = fixture.Extractor;
        var cases = Gliner2Fixtures.LoadCapabilityCases();
        Assert.Equal(12, cases.Count(c => !c.GateOpen));
        Assert.Equal(11, cases.Count(c => c.Descriptions is { Count: > 0 }));
        var summary = Gliner2ModelParity.SingleCases(fixture.Extractor, cases, Tolerances, output);
        Assert.Equal(22, summary.Cases);
        Assert.Equal(summary.Cases, summary.InputsExact);
        Assert.Equal(summary.Cases, summary.EntitiesExact);
        foreach (var c in cases.Where(c => !c.GateOpen))
        {
            var result = fixture.Extractor.ExtractBatch([c.Text], c.LabelsRequested, c.Descriptions, c.Threshold, TestContext.Current.CancellationToken)[0];
            Assert.Empty(result.Entities);
            Assert.False(result.GateOpen);
        }

        Gliner2ModelParity.PaddedBatches(fixture.Extractor, cases, summary.Outputs, Tolerances, output);
    }

    [Fact]
    public void InputsOver512TokensAreRejected() =>
        Gliner2ModelParity.LongInputsAreRejected(fixture.Extractor, fixture.Cases, output);

    /// <summary>
    /// Capability <c>concurrent_calls</c> over CUDA, serialized (the CUDA default): four threads give the sequential
    /// results. Unserialized, the same test returned wrong entities in 6 of 8 runs.
    /// </summary>
    [Fact]
    public void ConcurrentExtractionGivesTheSequentialResults()
    {
        var extractor = fixture.Extractor;
        Assert.True(extractor.SerializesNativeCalls);
        var cases = fixture.Cases.Where(c => c.OutOfContract is null).Take(16).ToList();
        var ct = TestContext.Current.CancellationToken;
        var sequential = cases.Select(c => extractor.Extract(c.Text, c.LabelsRequested, c.Threshold, ct).Entities).ToList();
        for (int round = 0; round < 3; round++)
        {
            var parallel = new IReadOnlyList<Gliner2Entity>[cases.Count];
            Parallel.For(0, cases.Count, new ParallelOptions { MaxDegreeOfParallelism = 4, CancellationToken = ct }, i =>
                parallel[i] = extractor.Extract(cases[i].Text, cases[i].LabelsRequested, cases[i].Threshold, ct).Entities);
            for (int i = 0; i < cases.Count; i++)
            {
                Assert.Equal(sequential[i], parallel[i]);
            }
        }

        output.WriteLine($"{cases.Count} cases x 3 rounds on 4 threads over CUDA: entities equal to the sequential run");
    }

    /// <summary>
    /// Capability <c>cancellation</c> over CUDA: a long batch cancelled after 50 ms throws, returns promptly, and the
    /// extractor gives the reference entities afterwards.
    /// </summary>
    [Fact]
    public void CancellationDuringARunThrowsAndTheExtractorStaysUsable()
    {
        var extractor = fixture.Extractor;
        var c = fixture.Cases.First(c => c.Id == "basic-multi-label");
        var texts = Enumerable.Repeat(c.Text, 4096).ToList();
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(50));
        var sw = System.Diagnostics.Stopwatch.StartNew();
        Assert.ThrowsAny<OperationCanceledException>(() => extractor.ExtractBatch(texts, c.LabelsRequested, c.Threshold, cts.Token));
        output.WriteLine($"4,096 texts cancelled after 50 ms: threw after {sw.Elapsed.TotalMilliseconds:F0} ms");
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(5), $"cancellation took {sw.Elapsed.TotalMilliseconds:F0} ms");
        var after = extractor.Extract(c.Text, c.LabelsRequested, c.Threshold, TestContext.Current.CancellationToken);
        Assert.Equal(c.Entities.Count, after.Entities.Count);
    }

    /// <summary>
    /// Regression (docs/evidence/gliner2-cuda-wrong-output/): a cancelled call returns only after the device work it
    /// queued has finished, so no later or concurrent run can compute on memory that abandoned work still writes. Before
    /// the fix, runs were terminated at a node boundary and about one cancelled call in 20 returned with up to 33 ms
    /// of work still queued; requests served right after such calls returned empty entities. Afterwards the
    /// extractor gives the reference entities.
    /// </summary>
    [Fact]
    public void ACancelledCallLeavesNoDeviceWorkQueued()
    {
        var extractor = fixture.Extractor;
        var c = fixture.Cases.First(c => c.Id == "basic-multi-label");
        var batch = Enumerable.Repeat(fixture.Cases.Where(x => x.OutOfContract is null).MaxBy(x => x.Text.Length)!.Text, 16).ToList();
        var probe = CudaQueueProbe.CancelledCalls(ct => extractor.ExtractBatch(batch, c.LabelsRequested, c.Threshold, ct), cancellations: 300);
        output.WriteLine($"16 texts in batches of 8: {probe}");
        Assert.True(probe.Drains.Count >= 100, probe.ToString());
        Assert.Equal(0, probe.Pending);
        var after = extractor.Extract(c.Text, c.LabelsRequested, c.Threshold, TestContext.Current.CancellationToken);
        Assert.Equal(c.Entities.Count, after.Entities.Count);
    }

    [Fact]
    public void PlacementPutsMostNodesOnCuda()
    {
        var c = fixture.Cases[0];
        fixture.Extractor.Extract(c.Text, c.LabelsRequested, c.Threshold, TestContext.Current.CancellationToken);
        var placement = fixture.Extractor.Session.Placement;
        Assert.NotNull(placement);
        foreach (var (provider, count) in placement.NodesByProvider)
        {
            output.WriteLine($"{provider}: {count} nodes");
        }

        // P-014 (onnx-fp32-cuda/placement.json): 759 nodes on CUDA, 33 shape-arithmetic nodes on CPU.
        int cuda = placement.NodesByProvider.GetValueOrDefault(OnnxRuntimeInfo.CudaProviderName);
        int cpu = placement.NodesByProvider.GetValueOrDefault(OnnxRuntimeInfo.CpuProviderName);
        Assert.True(cuda > 10 * cpu, $"CUDA {cuda} nodes, CPU {cpu} nodes");
    }
}
