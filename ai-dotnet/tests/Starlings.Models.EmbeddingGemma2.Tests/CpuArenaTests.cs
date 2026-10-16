using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>
/// ONNX Runtime's CPU memory arena changes where buffers live, not what is computed: text and video embeddings with the
/// arena on (ONNX Runtime's default) and off (this library's CPU default, docs/evidence/eg2-hosted-soak/) are equal bit
/// for bit on the same machine.
/// </summary>
public class CpuArenaTests(ITestOutputHelper output)
{
    [Fact]
    public void TextEmbeddingsAreBitIdenticalWithAndWithoutTheArena()
    {
        var (files, reason) = EmbeddingGemma2TestAssets.FindModel();
        Assert.SkipWhen(files is null, reason);
        var rows = EmbeddingGemma2Fixtures.LoadRows();
        var inputs = rows.Take(8).Select(r => new EmbeddingGemma2Input(r.InputIds)).ToArray();

        float[][] Embed(bool arena)
        {
            using var encoder = EmbeddingGemma2Encoder.Create(files!.ModelPath, new EmbeddingGemma2EncoderOptions
            {
                Session = new OnnxSessionOptions { ComputeModelSha256 = false, CpuMemoryArena = arena },
            });
            Assert.Equal(arena, encoder.Session.Diagnostics.CpuMemoryArena);
            return inputs.Select(i => encoder.Embed(i, TestContext.Current.CancellationToken)).ToArray();
        }

        var on = Embed(true);
        var off = Embed(false);
        for (int i = 0; i < on.Length; i++)
        {
            Assert.Equal(on[i], off[i]);
        }

        output.WriteLine($"{on.Length} text rows bit-identical with the arena on and off");
    }

    [Fact]
    public void VideoEmbeddingsAreBitIdenticalWithAndWithoutTheArena()
    {
        var (files, reason) = EmbeddingGemma2ImageTestAssets.FindImageModel();
        Assert.SkipWhen(files is null, reason);
        var videos = new[] { Clip(0, 2), Clip(1, 3) };

        float[][] Embed(bool arena)
        {
            using var embedder = EmbeddingGemma2MultimodalEmbedder.Create(files!, new EmbeddingGemma2MultimodalOptions
            {
                Session = new OnnxSessionOptions { ComputeModelSha256 = false, CpuMemoryArena = arena },
                VerifyModelSha256 = false,
            });
            Assert.Equal(arena, embedder.VisionSession.Diagnostics.CpuMemoryArena);
            Assert.Equal(arena, embedder.MultimodalSession.Diagnostics.CpuMemoryArena);
            return videos.Select(v => embedder.Embed(v, TestContext.Current.CancellationToken)).ToArray();
        }

        var on = Embed(true);
        var off = Embed(false);
        for (int i = 0; i < on.Length; i++)
        {
            Assert.Equal(on[i], off[i]);
        }

        output.WriteLine($"{on.Length} video embeddings bit-identical with the arena on and off");
    }

    /// <summary>A synthetic decoded clip (gradients and a moving square) at 4 frames per second.</summary>
    private static EmbeddingGemma2Video Clip(int input, int seconds)
    {
        const int fps = 4;
        int width = 96 + (input * 37 % 160);
        int height = 80 + (input * 53 % 144);
        var frames = new EmbeddingGemma2Image[fps * seconds];
        for (int index = 0; index < frames.Length; index++)
        {
            var rgb = new byte[width * height * 3];
            int side = Math.Min(width, height) / 3;
            int x0 = index * 7 % (width - side);
            int y0 = index * 5 % (height - side);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int o = ((y * width) + x) * 3;
                    bool square = x >= x0 && x < x0 + side && y >= y0 && y < y0 + side;
                    rgb[o] = square ? (byte)250 : (byte)((x * 255 / width) ^ (input * 17));
                    rgb[o + 1] = square ? (byte)250 : (byte)(y * 255 / height);
                    rgb[o + 2] = (byte)(((x / (4 + input)) + (y / 8) + index) % 2 == 0 ? 200 : 40);
                }
            }

            frames[index] = EmbeddingGemma2Image.FromRgb24(width, height, rgb);
        }

        return EmbeddingGemma2Video.FromDecodedFrames(frames, fps, seconds);
    }
}
