namespace Starlings.Models.EmbeddingGemma2.Tests.Cuda;

/// <summary>
/// The CUDA free-memory probe behind <see cref="EmbeddingGemma2MultimodalOptions.AutoBatchLimits"/> (Experimental,
/// capability <c>image.auto_batch_limits</c>): opt-in like every CUDA test, on the device chosen through
/// <c>CUDA_VISIBLE_DEVICES</c>.
/// </summary>
[Collection("cuda")]
public sealed class CudaAutoBatchLimitsTests(ITestOutputHelper output)
{
    [Fact]
    public void CudaProbeReadsTheVisibleDevicesFreeMemory()
    {
        string? gate = CudaGate.Check();
        Assert.SkipWhen(gate is not null, gate ?? string.Empty);

        long? free = EmbeddingGemma2MemoryProbe.CudaFreeBytes(0);
        output.WriteLine($"CUDA_VISIBLE_DEVICES={Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES")}; free {free} bytes");
        Assert.NotNull(free);
        Assert.InRange(free.Value, 1L, 1L << 40);

        // A device index past the visible devices is unavailable, not an error.
        Assert.Null(EmbeddingGemma2MemoryProbe.CudaFreeBytes(64));
    }
}
