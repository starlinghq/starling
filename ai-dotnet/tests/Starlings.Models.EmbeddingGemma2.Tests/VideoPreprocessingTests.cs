using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>
/// Per-frame preprocessing of every fixture video at 140 soft tokens, from adapter-decoded frames (decoded frames,
/// resized bytes, pixel values and positions exact against the reference digests: binding keys
/// <c>discrete.decoded_frames</c>, <c>resized_pixels</c>, <c>pixel_values</c>, <c>video_positions</c>). Needs the video set
/// and FFmpeg 8; no model.
/// </summary>
public sealed class VideoPreprocessingTests(ITestOutputHelper output)
{
    private static readonly EmbeddingGemma2Tolerances Tolerances = EmbeddingGemma2Tolerances.Load("video-fp32-cpu");

    [Fact]
    public void EveryVideoInputPreprocessesBitExact()
    {
        Assert.True(Tolerances.IsExact("decoded_frames") && Tolerances.IsExact("resized_pixels") && Tolerances.IsExact("pixel_values") && Tolerances.IsExact("video_positions"));
        int inputs = 0, frames = 0;
        foreach (var row in EmbeddingGemma2VideoTestAssets.Rows)
        {
            var input = EmbeddingGemma2VideoTestAssets.BuildInput(row);
            var videos = input.Videos.ToList();
            Assert.Equal(row.Videos.Count, videos.Count);
            for (int k = 0; k < videos.Count; k++)
            {
                var expected = row.Videos[k];
                int[] indices = videos[k].GetSampledFrameIndices(allowWithoutTiming: true);
                var sampled = indices.Select(i => videos[k].Frames[i]).ToList();
                if (videos[k].FrameSource != EmbeddingGemma2VideoFrameSource.PreSampled)
                {
                    Assert.Equal(expected.SampledIndices, indices);
                }

                Assert.True(expected.FramesSha256 == EmbeddingGemma2VideoTestAssets.Sha(sampled.Select(f => f.Pixels)), $"{row.Id}/video{k}: decoded frames differ");
                var patches = sampled.Select(f => EmbeddingGemma2ImageProcessor.Preprocess(f, EmbeddingGemma2MultimodalEmbedder.VideoFrameSoftTokens)).ToList();
                Assert.Equal((expected.ResizedWidth, expected.ResizedHeight), (patches[0].ResizedWidth, patches[0].ResizedHeight));
                Assert.Equal(expected.SoftTokensPerFrame, patches[0].SoftTokenCount);
                Assert.True(expected.ResizedSha256 == EmbeddingGemma2VideoTestAssets.Sha(patches.Select(p => p.ResizedPixels)), $"{row.Id}/video{k}: resized frames differ");
                Assert.True(expected.PixelValuesSha256 == EmbeddingGemma2VideoTestAssets.Sha(patches.Select(p => p.PixelValues)), $"{row.Id}/video{k}: pixel values differ");
                Assert.True(expected.PositionsSha256 == EmbeddingGemma2VideoTestAssets.Sha(patches.Select(p => p.Positions)), $"{row.Id}/video{k}: positions differ");
                inputs++;
                frames += sampled.Count;
            }

            if (row.Image is not null)
            {
                var still = input.Images.Single();
                Assert.Equal(EmbeddingGemma2VideoTestAssets.TensorSha256(row.Id, "image0_rgb"), EmbeddingGemma2VideoTestAssets.Sha(still.Pixels.Span));
            }
        }

        output.WriteLine($"{inputs} video inputs ({frames} sampled frames, the metadata-less row included): decoded frames, resized frames, pixel values and positions equal the reference digests");
        Assert.Equal(32, inputs);
    }
}
