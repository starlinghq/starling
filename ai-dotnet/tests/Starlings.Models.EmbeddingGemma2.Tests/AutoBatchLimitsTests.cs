using Starlings.Runtime.Onnx;
using Starlings.TestSupport;
using Status = Starlings.Models.CapabilityStatus;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>
/// <see cref="EmbeddingGemma2MultimodalOptions.AutoBatchLimits"/>: the limits chosen for a given free memory (the probe
/// replaced through the test seam), the probe parsers, and the embedder on the weight-free graphs. Runs in CI.
/// </summary>
public class AutoBatchLimitsTests
{
    private const long GiB = 1L << 30;

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static EmbeddingGemma2AutoBatchLimits Fixed(long? free) => new() { Probe = (_, _) => free };

    [Fact]
    public void OffByDefaultAndExperimental()
    {
        var options = new EmbeddingGemma2MultimodalOptions();
        Assert.Null(options.AutoBatchLimits);
        Assert.Equal(1, options.MaxVisionBatchSize);
        Assert.Equal(2_520, options.MaxVisionBatchPatches);
        Assert.Equal(2, options.MaxVideoFramesPerVisionCall);
        Assert.Equal(Status.Experimental, EmbeddingGemma2Capabilities.Statuses["image.auto_batch_limits"]);

        using var embedder = FakeMultimodal.Create();
        Assert.Equal(options, embedder.Options);
        Assert.Null(embedder.AutoBatchAvailableBytes);
    }

    [Fact]
    public void EstimateMatchesTheDocumentedFormula()
    {
        Assert.Equal(3 * GiB, EmbeddingGemma2AutoBatchLimits.EstimateImageBytes(2_520));
        // A video frame: 1,260 patches, 0.25 * 0.5 + 0.75 * 0.25 = 0.3125 of the full image.
        Assert.Equal((long)Math.Ceiling(3 * GiB * 0.3125), EmbeddingGemma2AutoBatchLimits.EstimateImageBytes(1_260));
        Assert.Equal(1_260, EmbeddingGemma2AutoBatchLimits.VideoFramePatches);
        Assert.Throws<ArgumentOutOfRangeException>(() => EmbeddingGemma2AutoBatchLimits.EstimateImageBytes(0));
    }

    [Theory]
    // free, images, frames, patches (defaults: 1 image, 2 frames, 2,520 patches; margin 1 GiB then half)
    [InlineData(0L, 1, 2, 2_520)]
    [InlineData(5L * GiB, 1, 2, 2_520)] // 2 GiB budget: no full image fits twice; 2 frames need 1.875 GiB
    [InlineData(7L * GiB, 1, 3, 3_780)] // 3 GiB: one image, three frames
    [InlineData(13L * GiB, 2, 6, 7_560)] // 6 GiB: two images (5,040), six frames (7,560)
    [InlineData(25L * GiB, 4, 12, 15_120)]
    [InlineData(200L * GiB, 8, 16, 20_160)] // capped: 8 images (20,160), 16 frames (20,160)
    public void LimitsFollowTheBudget(long free, int images, int frames, int patches)
    {
        var options = new EmbeddingGemma2MultimodalOptions();
        var chosen = new EmbeddingGemma2AutoBatchLimits().Apply(options, free);
        Assert.Equal(images, chosen.MaxVisionBatchSize);
        Assert.Equal(frames, chosen.MaxVideoFramesPerVisionCall);
        Assert.Equal(patches, chosen.MaxVisionBatchPatches);
        Assert.Equal(options with { MaxVisionBatchSize = images, MaxVideoFramesPerVisionCall = frames, MaxVisionBatchPatches = patches }, chosen);
    }

    [Fact]
    public void ChosenCallsFitTheBudgetAndGrowWithFreeMemory()
    {
        var limits = new EmbeddingGemma2AutoBatchLimits();
        var options = new EmbeddingGemma2MultimodalOptions();
        int lastImages = 0, lastFrames = 0;
        for (long free = 0; free <= 64 * GiB; free += GiB / 4)
        {
            var chosen = limits.Apply(options, free);
            Assert.True(chosen.MaxVisionBatchSize >= lastImages && chosen.MaxVideoFramesPerVisionCall >= lastFrames);
            lastImages = chosen.MaxVisionBatchSize;
            lastFrames = chosen.MaxVideoFramesPerVisionCall;
            double budget = Math.Max(0, free - limits.ReservedBytes) * limits.UsableFraction;
            if (chosen.MaxVisionBatchSize > options.MaxVisionBatchSize)
            {
                Assert.True(chosen.MaxVisionBatchSize * EmbeddingGemma2AutoBatchLimits.EstimateImageBytes(2_520) <= budget);
            }

            if (chosen.MaxVideoFramesPerVisionCall > options.MaxVideoFramesPerVisionCall)
            {
                Assert.True(chosen.MaxVideoFramesPerVisionCall * EmbeddingGemma2AutoBatchLimits.EstimateImageBytes(1_260) <= budget);
            }

            Assert.True(chosen.MaxVisionBatchPatches >= chosen.MaxVisionBatchSize * 2_520);
            Assert.True(chosen.MaxVisionBatchPatches >= chosen.MaxVideoFramesPerVisionCall * 1_260);
        }
    }

    [Fact]
    public void ConfiguredValuesAreFloorsAndTheCapsAndBudgetApply()
    {
        var options = new EmbeddingGemma2MultimodalOptions { MaxVisionBatchSize = 3, MaxVideoFramesPerVisionCall = 5, MaxVisionBatchPatches = 30_000 };
        var low = new EmbeddingGemma2AutoBatchLimits().Apply(options, 2 * GiB);
        Assert.Equal((3, 5, 30_000), (low.MaxVisionBatchSize, low.MaxVideoFramesPerVisionCall, low.MaxVisionBatchPatches));

        var capped = new EmbeddingGemma2AutoBatchLimits { MaxImagesPerCall = 2, MaxFramesPerCall = 4 }.Apply(new EmbeddingGemma2MultimodalOptions(), 100 * GiB);
        Assert.Equal((2, 4, 5_040), (capped.MaxVisionBatchSize, capped.MaxVideoFramesPerVisionCall, capped.MaxVisionBatchPatches));

        // A larger image budget (560 soft tokens: 5,040 patches, 10.5 GiB estimated) fits fewer images; the frames (16, capped) set the patches.
        var big = new EmbeddingGemma2AutoBatchLimits { UsableFraction = 1, ReservedBytes = 0 }.Apply(new EmbeddingGemma2MultimodalOptions { MaxImageSoftTokens = 560 }, 25 * GiB);
        Assert.Equal(2, big.MaxVisionBatchSize);
        Assert.Equal(20_160, big.MaxVisionBatchPatches);
    }

    [Fact]
    public void UnavailableProbeKeepsTheConfiguredLimits()
    {
        var options = new EmbeddingGemma2MultimodalOptions { MaxVisionBatchSize = 2 };
        Assert.Same(options, new EmbeddingGemma2AutoBatchLimits().Apply(options, null));

        using var embedder = FakeMultimodal.Create(options with { AutoBatchLimits = Fixed(null) });
        Assert.Null(embedder.AutoBatchAvailableBytes);
        Assert.Equal(2, embedder.Options.MaxVisionBatchSize);
        Assert.Equal(2_520, embedder.Options.MaxVisionBatchPatches);
        Assert.Equal(2, embedder.Options.MaxVideoFramesPerVisionCall);
    }

    [Fact]
    public void InvalidSettingsAreRejected()
    {
        foreach (var bad in new[]
        {
            new EmbeddingGemma2AutoBatchLimits { UsableFraction = 0 },
            new EmbeddingGemma2AutoBatchLimits { UsableFraction = 1.5 },
            new EmbeddingGemma2AutoBatchLimits { UsableFraction = double.NaN },
            new EmbeddingGemma2AutoBatchLimits { ReservedBytes = -1 },
            new EmbeddingGemma2AutoBatchLimits { MaxImagesPerCall = 0 },
            new EmbeddingGemma2AutoBatchLimits { MaxFramesPerCall = 0 },
        })
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => FakeMultimodal.Create(new EmbeddingGemma2MultimodalOptions { AutoBatchLimits = bad }));
        }
    }

    [Fact]
    public void EmbedderAppliesTheProbeAndBatchesEqualSingleCalls()
    {
        OnnxExecutionProvider? provider = null;
        var limits = new EmbeddingGemma2AutoBatchLimits
        {
            Probe = (p, _) =>
            {
                provider = p;
                return 13 * GiB;
            },
        };
        using var auto = FakeMultimodal.Create(new EmbeddingGemma2MultimodalOptions { AutoBatchLimits = limits });
        Assert.Equal(OnnxExecutionProvider.Cpu, provider);
        Assert.Equal(13 * GiB, auto.AutoBatchAvailableBytes);
        Assert.Equal(2, auto.Options.MaxVisionBatchSize);
        Assert.Equal(6, auto.Options.MaxVideoFramesPerVisionCall);
        Assert.Equal(7_560, auto.Options.MaxVisionBatchPatches);
        Assert.Same(limits, auto.Options.AutoBatchLimits);

        using var single = FakeMultimodal.Create();
        var images = Enumerable.Range(0, 5).Select(i => FakeMultimodal.Image(40 + (i * 30), 50 + (i * 10), i + 1)).ToArray();
        var expected = images.Select(i => single.Embed(i, Ct)).ToArray();
        var actual = auto.EmbedBatch(images, Ct);
        for (int i = 0; i < images.Length; i++)
        {
            Assert.Equal(expected[i], actual[i]);
        }
    }

    [Fact]
    public void HostProbeParsers()
    {
        const string meminfo = "MemTotal:       98000000 kB\nMemFree:         1000000 kB\nMemAvailable:   47000000 kB\nBuffers:          100 kB\n";
        Assert.Equal(47_000_000L * 1024, EmbeddingGemma2MemoryProbe.ParseMemAvailable(meminfo));
        Assert.Null(EmbeddingGemma2MemoryProbe.ParseMemAvailable("MemTotal: 1 kB\n"));
        Assert.Null(EmbeddingGemma2MemoryProbe.CgroupRoom("max\n", "12345\n"));
        Assert.Equal(8L * GiB - 1000, EmbeddingGemma2MemoryProbe.CgroupRoom($"{8L * GiB}\n", "1000\n"));
        Assert.Equal(0L, EmbeddingGemma2MemoryProbe.CgroupRoom("1000\n", "5000\n"));
    }

    [Fact]
    public void RealHostProbeReturnsAPlausibleValueOnLinux()
    {
        long? free = EmbeddingGemma2MemoryProbe.HostAvailableBytes();
        if (!OperatingSystem.IsLinux())
        {
            Assert.Null(free);
            return;
        }

        Assert.NotNull(free);
        Assert.InRange(free.Value, 0, GC.GetGCMemoryInfo().TotalAvailableMemoryBytes + (64 * GiB));
    }
}
