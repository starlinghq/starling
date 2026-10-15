using Starlings.TestSupport;
using Starlings.Threading;

namespace Starlings.Runtime.Onnx.Tests;

/// <summary>Container-aware default intra-op thread count (docs/evidence/cuda-soak-growth/).</summary>
public class IntraOpThreadDefaultsTests
{
    [Theory]
    [InlineData(3, 31, 256, 3)] // the caller's count always wins
    [InlineData(64, 31, 256, 64)]
    [InlineData(null, 31, 256, 31)] // container quota of 31 CPUs on a 256-CPU host (the RunPod soak pods)
    [InlineData(null, 4, 12, 4)] // affinity mask or DOTNET_PROCESSOR_COUNT below the host count
    public void UsesRequestedOrConstrainedProcessorCount(int? requested, int processorCount, int? host, int expected) =>
        Assert.Equal(expected, IntraOpThreadDefaults.Resolve(requested, processorCount, host));

    [Theory]
    [InlineData(12, 12)] // unconstrained: keep the ONNX Runtime default the CPU profiles were qualified with
    [InlineData(12, null)] // host count unknown (not Linux, or /sys unreadable)
    [InlineData(0, 12)]
    public void KeepsRuntimeDefaultWhenUnconstrainedOrUnknown(int processorCount, int? host) =>
        Assert.Null(IntraOpThreadDefaults.Resolve(null, processorCount, host));

    [Theory]
    [InlineData("0-255\n", 256)]
    [InlineData("0", 1)]
    [InlineData("0,2-5,8", 6)]
    [InlineData("0-3,4-7", 8)]
    public void ParsesLinuxCpuLists(string text, int expected) => Assert.Equal(expected, IntraOpThreadDefaults.ParseCpuList(text));

    [Theory]
    [InlineData("")]
    [InlineData("a-b")]
    [InlineData("5-2")]
    [InlineData("-1")]
    public void RejectsMalformedCpuLists(string text) => Assert.Null(IntraOpThreadDefaults.ParseCpuList(text));

    [Fact]
    public void SessionRecordsTheThreadCountItConfigured()
    {
        using (var session = OnnxModelSession.Create(TestModels.Add, new OnnxSessionOptions { ComputeModelSha256 = false }))
        {
            Assert.Equal(IntraOpThreadDefaults.ForProcess(), session.Diagnostics.IntraOpNumThreads);
        }

        using (var session = OnnxModelSession.Create(TestModels.Add, new OnnxSessionOptions { IntraOpNumThreads = 3, ComputeModelSha256 = false }))
        {
            Assert.Equal(3, session.Diagnostics.IntraOpNumThreads);
        }
    }

    [Fact]
    public void ProcessDefaultFollowsEnvironmentProcessorCount()
    {
        // In a constrained process the default is exactly Environment.ProcessorCount; otherwise the runtime default.
        int? value = IntraOpThreadDefaults.ForProcess();
        Assert.True(value is null || value == Environment.ProcessorCount);
    }
}
