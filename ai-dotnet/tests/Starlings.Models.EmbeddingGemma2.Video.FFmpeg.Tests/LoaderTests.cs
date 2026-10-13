using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Video.FFmpeg.Tests;

/// <summary>The library loader's typed errors and rules that need no FFmpeg (CI), and what the machine's FFmpeg is (reported).</summary>
public sealed class LoaderTests(ITestOutputHelper output)
{
    [Fact]
    public void AMissingDirectoryIsATypedNotFoundError()
    {
        var ex = Assert.Throws<EmbeddingGemma2FFmpegNotFoundException>(() => EmbeddingGemma2FFmpeg.Load(new EmbeddingGemma2FFmpegOptions { LibraryDirectory = "/nonexistent/ffmpeg-8/lib" }));
        Assert.Contains("does not exist", ex.Message, StringComparison.Ordinal);
        Assert.IsAssignableFrom<EmbeddingGemma2FFmpegException>(ex);
        Assert.Throws<EmbeddingGemma2FFmpegNotFoundException>(() => new EmbeddingGemma2VideoDecoder(new EmbeddingGemma2FFmpegOptions { LibraryDirectory = "/nonexistent/ffmpeg-8/lib" }));
    }

    [Fact]
    public void LicensesAreClassified()
    {
        static EmbeddingGemma2FFmpegLibraries With(params string[] licenses) =>
            new(null, "n8.1", new Dictionary<string, string>(), licenses.Select((l, i) => (l, i)).ToDictionary(x => $"lib{x.i}", x => x.l), string.Empty);

        Assert.True(With("LGPL version 3 or later", "LGPL version 2.1 or later").IsLgpl);
        Assert.False(With("LGPL version 3 or later", "GPL version 3 or later").IsLgpl);
        Assert.False(With("nonfree and unredistributable").IsLgpl);
    }

    [Fact]
    public void RotationMapsToTorchcodecTransforms()
    {
        Assert.Equal(0, EmbeddingGemma2VideoDecoder.QuarterTurns(null));
        Assert.Equal(0, EmbeddingGemma2VideoDecoder.QuarterTurns(0));
        Assert.Equal(1, EmbeddingGemma2VideoDecoder.QuarterTurns(90));
        Assert.Equal(1, EmbeddingGemma2VideoDecoder.QuarterTurns(89.9));
        Assert.Equal(-1, EmbeddingGemma2VideoDecoder.QuarterTurns(-90));
        Assert.Equal(2, EmbeddingGemma2VideoDecoder.QuarterTurns(180));
        Assert.Equal(2, EmbeddingGemma2VideoDecoder.QuarterTurns(-180));
        Assert.Equal(1, EmbeddingGemma2VideoDecoder.QuarterTurns(45)); // rounded to the nearest quarter turn (untested upstream)
        Assert.Throws<EmbeddingGemma2VideoDecodeException>(() => EmbeddingGemma2VideoDecoder.QuarterTurns(450));
    }

    [Fact]
    public void MetadataDerivesDurationAndRate()
    {
        var m = new EmbeddingGemma2VideoMetadata(150, 0, 5, 640, 360, null, "h264", "yuv420p", 0);
        Assert.Equal(5, m.DurationSeconds);
        Assert.Equal(30, m.AverageFramesPerSecond);
        Assert.Null((m with { EndSeconds = 0 }).AverageFramesPerSecond);
    }

    [Fact]
    public void OptionsAreValidated()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingGemma2VideoDecoder(new EmbeddingGemma2FFmpegOptions { MaxFramePixels = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingGemma2VideoDecoder(new EmbeddingGemma2FFmpegOptions { MaxAllFrames = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingGemma2VideoDecoder(new EmbeddingGemma2FFmpegOptions { DecoderThreads = -1 }));
    }

    /// <summary>
    /// Report (not gated): an FFmpeg 9 installation (libavcodec.so.63) is not loadable by this adapter version; the error
    /// names the other major. The torchcodec-on-FFmpeg-9 comparison of the spike (22 of 22 equal) is evidence only.
    /// </summary>
    [Fact]
    public void AnotherFFmpegMajorIsNamedInTheError()
    {
        string message = EmbeddingGemma2FFmpeg.NotFoundMessage("/usr/lib", "avcodec", 62);
        output.WriteLine(message);
        Assert.Contains("libavcodec.so.62", message, StringComparison.Ordinal);
        if (File.Exists("/usr/lib/libavcodec.so.63"))
        {
            Assert.Contains("libavcodec.so.63", message, StringComparison.Ordinal);
        }
    }
}
