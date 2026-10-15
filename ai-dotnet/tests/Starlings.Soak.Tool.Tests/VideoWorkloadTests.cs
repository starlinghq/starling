using Starlings.Models.EmbeddingGemma2;

namespace Starlings.Soak.Tool.Tests;

/// <summary>The EmbeddingGemma 2 video soak target's registration and its decoded-frame inputs (no model needed).</summary>
public sealed class VideoWorkloadTests
{
    [Fact]
    public void VideoTargetIsRegisteredWithTheVideoProfiles()
    {
        var workload = Assert.Single(SoakWorkload.All, w => w.Name == "embeddinggemma2-video");
        Assert.Equal("embeddinggemma-2/video-fp32-cpu", workload.Profile(new SoakOptions { Out = "x" }));
        Assert.Equal("embeddinggemma-2/video-fp32-cuda", workload.Profile(new SoakOptions { Out = "x", Cuda = true }));
        Assert.Equal(SoakWorkload.All.Count, SoakWorkload.All.Select(w => w.Name).Distinct().Count());
    }

    [Fact]
    public void ClipsAreDecodedFramesWithTimingSampledAtOneFramePerSecond()
    {
        for (int input = 0; input < SoakWorkload.InputCount; input++)
        {
            int seconds = 2 + (input % 2);
            var clip = EmbeddingGemma2VideoWorkload.Clip(input, seconds);
            Assert.Equal(EmbeddingGemma2VideoFrameSource.AllFramesWithTiming, clip.FrameSource);
            Assert.Equal(4 * seconds, clip.Frames.Count);
            Assert.Equal(EmbeddingGemma2VideoSampling.SampleIndices(4 * seconds, 4, seconds), clip.GetSampledFrameIndices());
            Assert.Equal(seconds, clip.GetSampledFrameIndices().Length);
            Assert.Single(clip.Frames.Select(f => (f.Width, f.Height)).Distinct());
        }
    }

    [Fact]
    public void ClipsAreDeterministicAndTheirFramesDiffer()
    {
        var a = EmbeddingGemma2VideoWorkload.Clip(5, 3);
        var b = EmbeddingGemma2VideoWorkload.Clip(5, 3);
        for (int f = 0; f < a.Frames.Count; f++)
        {
            Assert.True(a.Frames[f].Pixels.Span.SequenceEqual(b.Frames[f].Pixels.Span));
        }

        Assert.False(a.Frames[0].Pixels.Span.SequenceEqual(a.Frames[4].Pixels.Span));
        Assert.False(a.Frames[0].Pixels.Span.SequenceEqual(EmbeddingGemma2VideoWorkload.Clip(6, 3).Frames[0].Pixels.Span));
    }
}
