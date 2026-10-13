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

    [Fact]
    public void InputsOver512TokensAreRejected() =>
        Gliner2ModelParity.LongInputsAreRejected(fixture.Extractor, fixture.Cases, output);

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
