using System.Security.Cryptography;
using System.Text.Json;
using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>
/// <see cref="EmbeddingGemma2VideoSampling"/> against upstream's <c>sample_frames</c> (binding key
/// <c>discrete.frame_indices: exact</c>): the committed subset of the spike's seeded cases (CI), the full 20,849 cases when
/// <c>STARLING_EMBEDDINGGEMMA2_SAMPLING_CASES</c> points at <c>sampling-cases.json</c>, and every fixture video's
/// metadata.
/// </summary>
public sealed class VideoSamplingTests(ITestOutputHelper output)
{
    private const string FullCasesEnvironmentVariable = "STARLING_EMBEDDINGGEMMA2_SAMPLING_CASES";
    private const string FullCasesSha256 = "b9ef083bff2fb37aec9e2d66bd1d1bf9ea7e2f54b68ef64f20415361a362113a";

    private static readonly EmbeddingGemma2Tolerances Tolerances = EmbeddingGemma2Tolerances.Load("video-fp32-cpu");

    [Fact]
    public void CommittedCasesEqualUpstream()
    {
        Assert.True(Tolerances.IsExact("frame_indices"));
        string path = Path.Combine(EmbeddingGemma2TestAssets.RepositoryRoot, "ai-dotnet", "tests", "Starlings.Models.EmbeddingGemma2.Tests", "Data", "video-sampling-cases.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        Assert.Equal(FullCasesSha256, doc.RootElement.GetProperty("source_sha256").GetString());
        int n = Check(doc.RootElement.GetProperty("cases"));
        output.WriteLine($"{n} committed cases (structured + every 40th seeded case of {doc.RootElement.GetProperty("source_cases").GetInt32()}): indices equal upstream's sample_frames");
        Assert.Equal(1_349, n);
    }

    [Fact]
    public void AllSpikeCasesEqualUpstream()
    {
        string? path = Environment.GetEnvironmentVariable(FullCasesEnvironmentVariable);
        Assert.SkipWhen(string.IsNullOrEmpty(path) || !File.Exists(path), $"set {FullCasesEnvironmentVariable} to make_eg2_sampling_cases.py's sampling-cases.json for the full 20,849-case comparison.");
        byte[] bytes = File.ReadAllBytes(path!);
        Assert.Equal(FullCasesSha256, Convert.ToHexStringLower(SHA256.HashData(bytes)));
        using var doc = JsonDocument.Parse(bytes);
        int n = Check(doc.RootElement);
        output.WriteLine($"{n} cases: indices equal upstream's sample_frames");
        Assert.Equal(20_849, n);
    }

    [Fact]
    public void FixtureVideosSampleAsTheReference()
    {
        int n = 0;
        foreach (var (id, frames) in EmbeddingGemma2VideoTestAssets.Frames)
        {
            var m = frames.Metadata;
            Assert.True(frames.SampledIndices.AsSpan().SequenceEqual(EmbeddingGemma2VideoSampling.SampleIndices(m.NumFrames, m.AverageFps, m.DurationSeconds)), id);
            n++;
        }

        // The metadata-less row: every frame, thinned to 32.
        var row = EmbeddingGemma2VideoTestAssets.Rows.Single(r => r.Id == EmbeddingGemma2VideoTestAssets.NoMetadataRowId);
        Assert.Equal(row.Videos[0].SampledIndices, EmbeddingGemma2VideoSampling.SampleIndices(150, null, null));
        Assert.Equal(22, n);
    }

    [Theory]
    [InlineData(15, 30.0, 0.5, new[] { 0 })] // under 2 s: one frame
    [InlineData(57, 30.0, 1.9, new[] { 0 })]
    [InlineData(360, 29.97002997002997, 12.012, new[] { 0, 29, 59, 89, 119, 149, 179, 209, 239, 269, 299, 329 })]
    [InlineData(8, 1.0, 8.0, new[] { 0, 1, 2, 3, 4, 5, 6, 7 })]
    public void QuirksAreKept(int total, double fps, double duration, int[] expected) =>
        Assert.Equal(expected, EmbeddingGemma2VideoSampling.SampleIndices(total, fps, duration));

    [Fact]
    public void LongVideosAreThinnedTo32()
    {
        var idx = EmbeddingGemma2VideoSampling.SampleIndices(1_200, 30, 40);
        Assert.Equal(32, idx.Length);
        Assert.Equal([0, 30, 60, 90, 150], idx[..5]);
        Assert.Equal(1_170, idx[^1]);

        // Durations far beyond the frame count do not allocate per candidate.
        var huge = EmbeddingGemma2VideoSampling.SampleIndices(100, 1, 1e12);
        Assert.Equal(32, huge.Length);
        Assert.Equal(99, huge[^1]);
    }

    [Fact]
    public void InvalidArgumentsAreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => EmbeddingGemma2VideoSampling.SampleIndices(0, 30, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => EmbeddingGemma2VideoSampling.SampleIndices(10, double.NaN, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => EmbeddingGemma2VideoSampling.SampleIndices(10, 30, -1));
    }

    private static int Check(JsonElement cases)
    {
        int n = 0;
        foreach (var c in cases.EnumerateArray())
        {
            int total = c.GetProperty("total").GetInt32();
            double? fps = c.GetProperty("fps").ValueKind == JsonValueKind.Null ? null : c.GetProperty("fps").GetDouble();
            double? duration = c.GetProperty("duration").ValueKind == JsonValueKind.Null ? null : c.GetProperty("duration").GetDouble();
            int[] expected = c.GetProperty("expected").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            int[] actual = EmbeddingGemma2VideoSampling.SampleIndices(total, fps, duration);
            Assert.True(expected.AsSpan().SequenceEqual(actual), $"total {total}, fps {fps}, duration {duration}: expected [{string.Join(", ", expected)}], got [{string.Join(", ", actual)}]");
            n++;
        }

        return n;
    }
}
