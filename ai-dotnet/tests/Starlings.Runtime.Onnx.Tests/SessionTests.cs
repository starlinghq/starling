using System.Security.Cryptography;
using Starlings.TestSupport;
using Microsoft.ML.OnnxRuntime;

namespace Starlings.Runtime.Onnx.Tests;

public class SessionTests
{
    [Fact]
    public void NativeRuntimeMatchesManagedBindingAndPinnedVersion()
    {
        Assert.Equal("1.30.0", OnnxRuntimeInfo.ManagedVersion);
        Assert.Equal(OnnxRuntimeInfo.ManagedVersion, OnnxRuntimeInfo.NativeVersion);
        Assert.Contains(OnnxRuntimeInfo.CpuProviderName, OnnxRuntimeInfo.AvailableProviders);
    }

    [Fact]
    public void DiagnosticsDescribeGraphProviderAndIdentity()
    {
        using var session = OnnxModelSession.Create(TestModels.Add, new OnnxSessionOptions { IntraOpNumThreads = 2 });
        var d = session.Diagnostics;

        Assert.Equal("1.30.0", d.OrtVersion);
        Assert.Equal(OnnxExecutionProvider.Cpu, d.RequestedProvider);
        Assert.Equal([OnnxRuntimeInfo.CpuProviderName], d.SessionProviders);
        Assert.Empty(d.ProviderOptions);
        Assert.Equal(2, d.IntraOpNumThreads);
        Assert.Null(d.InterOpNumThreads);
        Assert.Equal("ORT_ENABLE_ALL", d.GraphOptimizationLevel);
        Assert.Equal("<memory>", d.ModelSource);
        Assert.Equal(TestModels.Add.LongLength, d.ModelSizeBytes);
        Assert.Equal(Convert.ToHexStringLower(SHA256.HashData(TestModels.Add)), d.ModelSha256);
        Assert.Equal("starlings-test", d.Model.ProducerName);
        Assert.Equal("add", d.Model.GraphName);
        Assert.Equal("runtime-tests", d.Model.CustomMetadata["purpose"]);

        Assert.Equal(["x", "y"], d.Inputs.Select(i => i.Name));
        var x = d.Inputs[0];
        Assert.Equal("Float", x.ElementType);
        Assert.Equal([-1L, 4L], x.Dimensions);
        Assert.Equal("batch", x.SymbolicDimensions[0]);
        Assert.Equal("z", Assert.Single(d.Outputs).Name);
    }

    [Fact]
    public void Sha256CanBeSkippedAndIsVerifiedWhenExpected()
    {
        using (var session = OnnxModelSession.Create(TestModels.Add, new OnnxSessionOptions { ComputeModelSha256 = false }))
        {
            Assert.Null(session.Diagnostics.ModelSha256);
        }

        string actual = Convert.ToHexStringLower(SHA256.HashData(TestModels.Add));
        using (var verified = OnnxModelSession.Create(TestModels.Add, new OnnxSessionOptions { ExpectedModelSha256 = actual, ComputeModelSha256 = false }))
        {
            Assert.Equal(actual, verified.Diagnostics.ModelSha256);
        }

        var wrong = new OnnxSessionOptions { ExpectedModelSha256 = new string('0', 64) };
        Assert.Throws<OnnxModelIntegrityException>(() => OnnxModelSession.Create(TestModels.Add, wrong));
    }

    [Fact]
    public void KnownDigestReplacesTheReadAndSatisfiesTheExpectedDigest()
    {
        string actual = Convert.ToHexStringLower(SHA256.HashData(TestModels.Add));
        using (var computed = OnnxModelSession.Create(TestModels.Add))
        {
            Assert.Equal(OnnxModelDigestSource.Computed, computed.Diagnostics.ModelSha256Source);
        }

        // A digest the caller vouches for is reported as given: the bytes are not hashed (a different digest shows it).
        string other = new('a', 64);
        using (var known = OnnxModelSession.Create(TestModels.Add, new OnnxSessionOptions { KnownModelSha256 = other }))
        {
            Assert.Equal(other, known.Diagnostics.ModelSha256);
            Assert.Equal(OnnxModelDigestSource.Known, known.Diagnostics.ModelSha256Source);
        }

        using (var expected = OnnxModelSession.Create(TestModels.Add, new OnnxSessionOptions { KnownModelSha256 = actual, ExpectedModelSha256 = actual }))
        {
            Assert.Equal(actual, expected.Diagnostics.ModelSha256);
            Assert.Equal(OnnxModelDigestSource.Known, expected.Diagnostics.ModelSha256Source);
        }

        var mismatch = Assert.Throws<OnnxModelIntegrityException>(() => OnnxModelSession.Create(
            TestModels.Add, new OnnxSessionOptions { KnownModelSha256 = other, ExpectedModelSha256 = actual }));
        Assert.Contains("KnownModelSha256", mismatch.Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentException>(() => OnnxModelSession.Create(TestModels.Add, new OnnxSessionOptions { KnownModelSha256 = "ABC" }));

        using (var none = OnnxModelSession.Create(TestModels.Add, new OnnxSessionOptions { ComputeModelSha256 = false }))
        {
            Assert.Null(none.Diagnostics.ModelSha256Source);
        }
    }

    [Fact]
    public void KnownDigestSkipsReadingTheFile()
    {
        string path = Path.Combine(Path.GetTempPath(), $"starlings-add-{Guid.NewGuid():N}.onnx");
        File.WriteAllBytes(path, TestModels.Add);
        // Not the file's digest: had the file been hashed, the session would report the real one.
        string known = new('b', 64);
        try
        {
            using var session = OnnxModelSession.Create(path, new OnnxSessionOptions { KnownModelSha256 = known, ExpectedModelSha256 = known });
            Assert.Equal(known, session.Diagnostics.ModelSha256);
            Assert.Equal(OnnxModelDigestSource.Known, session.Diagnostics.ModelSha256Source);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void FileModelIsHashedFromDisk()
    {
        string path = Path.Combine(Path.GetTempPath(), $"starlings-add-{Guid.NewGuid():N}.onnx");
        File.WriteAllBytes(path, TestModels.Add);
        try
        {
            using var session = OnnxModelSession.Create(path);
            Assert.Equal(Path.GetFullPath(path), session.Diagnostics.ModelSource);
            Assert.Equal(Convert.ToHexStringLower(SHA256.HashData(TestModels.Add)), session.Diagnostics.ModelSha256);
        }
        finally
        {
            File.Delete(path);
        }

        Assert.Throws<FileNotFoundException>(() => OnnxModelSession.Create(path));
    }

    [Theory]
    [InlineData(0, null, 0)]
    [InlineData(null, -1, 0)]
    [InlineData(null, null, -1)]
    public void InvalidOptionsAreRejected(int? intra, int? inter, int device)
    {
        var options = new OnnxSessionOptions { IntraOpNumThreads = intra, InterOpNumThreads = inter, CudaDeviceId = device };
        Assert.Throws<ArgumentOutOfRangeException>(() => OnnxModelSession.Create(TestModels.Add, options));
    }

    [Fact]
    public void MalformedDigestIsRejected()
    {
        var options = new OnnxSessionOptions { ExpectedModelSha256 = "ABC" };
        Assert.Throws<ArgumentException>(() => OnnxModelSession.Create(TestModels.Add, options));
    }

    [Fact]
    public void CudaOnCpuPackageFailsClearlyWithoutFallback()
    {
        // This test project references the CPU native package, which has no CUDA execution provider.
        var ex = Assert.Throws<OnnxProviderUnavailableException>(
            () => OnnxModelSession.Create(TestModels.Add, new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cuda }));
        Assert.Contains("Microsoft.ML.OnnxRuntime.Gpu.Linux", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void RunReturnsCallerOwnedOutputs()
    {
        using var session = OnnxModelSession.Create(TestModels.Add);
        float[] x = [1, 2, 3, 4, 5, 6, 7, 8];
        float[] y = [10, 20, 30, 40, 50, 60, 70, 80];
        using var xv = OrtValue.CreateTensorValueFromMemory(x, [2, 4]);
        using var yv = OrtValue.CreateTensorValueFromMemory(y, [2, 4]);

        using var outputs = session.Run(["x", "y"], [xv, yv], ["z"], TestContext.Current.CancellationToken);

        Assert.Equal([11f, 22, 33, 44, 55, 66, 77, 88], outputs[0].GetTensorDataAsSpan<float>().ToArray());
    }

    [Fact]
    public void RunWritesIntoPreallocatedOutputs()
    {
        using var session = OnnxModelSession.Create(TestModels.Add);
        float[] x = [1, 2, 3, 4];
        float[] y = [1, 1, 1, 1];
        float[] z = new float[4];
        using var xv = OrtValue.CreateTensorValueFromMemory(x, [1, 4]);
        using var yv = OrtValue.CreateTensorValueFromMemory(y, [1, 4]);
        using var zv = OrtValue.CreateTensorValueFromMemory(z, [1, 4]);

        session.Run(["x", "y"], [xv, yv], ["z"], [zv], TestContext.Current.CancellationToken);

        Assert.Equal([2f, 3, 4, 5], z);
    }

    [Fact]
    public void PlacementIsCapturedOnFirstRunOnly()
    {
        using var session = OnnxModelSession.Create(TestModels.Add, new OnnxSessionOptions { CapturePlacement = true });
        Assert.Null(session.Placement);
        Run(session, 3);
        var placement = session.Placement;
        Assert.NotNull(placement);
        Assert.Equal(1, placement.NodesByProvider[OnnxRuntimeInfo.CpuProviderName]);
        Assert.Equal(1, placement.OperatorsByProvider[OnnxRuntimeInfo.CpuProviderName]["Add"]);
        Assert.Single(placement.NodesByProvider);

        Run(session, 3);
        Assert.Same(placement, session.Placement);
    }

    [Fact]
    public void ConcurrentSessionsCapturePlacementFromTheirOwnProfileFiles()
    {
        // Each session's profile goes to its own temporary prefix, never ONNX Runtime's default name in the working
        // directory: that name only has millisecond resolution, so sessions started together shared (and deleted) one file.
        string[] before = Directory.GetFiles(Environment.CurrentDirectory, "onnxruntime_profile_*");
        Parallel.For(0, 16, new ParallelOptions { MaxDegreeOfParallelism = 16 }, _ =>
        {
            using var session = OnnxModelSession.Create(TestModels.Add, new OnnxSessionOptions { CapturePlacement = true });
            Run(session, 1);
            Assert.NotNull(session.Placement);
        });
        Assert.Equal(before, Directory.GetFiles(Environment.CurrentDirectory, "onnxruntime_profile_*"));
        Assert.Empty(Directory.GetFiles(Path.GetTempPath(), "starlings-ort-placement-*"));
    }

    [Fact]
    public void PlacementProfileIsRemovedWhenNoRunCompletes()
    {
        string[] before = Directory.GetFiles(Path.GetTempPath(), "starlings-ort-placement-*");

        // Disposed before any run.
        using (var session = OnnxModelSession.Create(TestModels.Add, new OnnxSessionOptions { CapturePlacement = true }))
        {
            Assert.Null(session.Placement);
        }

        Assert.Equal(before, Directory.GetFiles(Path.GetTempPath(), "starlings-ort-placement-*"));

        // Every run fails (wrong input name): the profile is never read, and still goes away with the session.
        using (var session = OnnxModelSession.Create(TestModels.Add, new OnnxSessionOptions { CapturePlacement = true }))
        {
            float[] x = [1, 1, 1, 1];
            using var xv = OrtValue.CreateTensorValueFromMemory(x, [1, 4]);
            Assert.Throws<OnnxRuntimeException>(() => session.Run(["x", "nope"], [xv, xv], ["z"], TestContext.Current.CancellationToken));
            Assert.Null(session.Placement);
        }

        Assert.Equal(before, Directory.GetFiles(Path.GetTempPath(), "starlings-ort-placement-*"));
    }

    internal static void Run(OnnxModelSession session, float value)
    {
        float[] x = [value, value, value, value];
        using var xv = OrtValue.CreateTensorValueFromMemory(x, [1, 4]);
        using var outputs = session.Run(["x", "y"], [xv, xv], ["z"]);
        Assert.Equal(2 * value, outputs[0].GetTensorDataAsSpan<float>()[3]);
    }
}
