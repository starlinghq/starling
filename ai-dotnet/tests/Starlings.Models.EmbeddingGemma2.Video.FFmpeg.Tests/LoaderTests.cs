using Starlings.TestSupport;
#if STARLINGS_FFMPEG9
namespace Starlings.Models.EmbeddingGemma2.Video.FFmpeg9.Tests;
#else
namespace Starlings.Models.EmbeddingGemma2.Video.FFmpeg.Tests;
#endif

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
    /// The not-found error names the bound library (FFmpeg 8: libavcodec.so.62; FFmpeg 9: libavcodec.so.63) and every other
    /// major it finds in the directory, so a user with the other FFmpeg major learns which adapter package to use.
    /// </summary>
    [Fact]
    public void AnotherFFmpegMajorIsNamedInTheError()
    {
        int bound = FFmpegNative.AVCODEC_MAJOR;
        Assert.Equal(EmbeddingGemma2FFmpeg.SupportedFFmpegMajor == 8 ? 62 : 63, bound);

        // The running system's naming: avcodec-62.dll on Windows, libavcodec.so.62 on Linux.
        string message = EmbeddingGemma2FFmpeg.NotFoundMessage("/usr/lib", "avcodec", bound);
        output.WriteLine(message);
        Assert.Contains(FFmpegNative.FileName("avcodec", bound), message, StringComparison.Ordinal);
        foreach (int other in Enumerable.Range(62, 2).Where(m => m != bound && File.Exists(Path.Combine("/usr/lib", FFmpegNative.FileName("avcodec", m)))))
        {
            Assert.Contains(FFmpegNative.FileName("avcodec", other), message, StringComparison.Ordinal);
        }
    }

    /// <summary>
    /// Every operating system's naming, on any: a directory holding other majors (and full-version and unrelated files)
    /// is reported with that system's file names only.
    /// </summary>
    [Theory]
    [InlineData("Windows", "avcodec-62.dll", "avcodec-63.dll", "avcodec-61.dll")]
    [InlineData("MacOS", "libavcodec.62.dylib", "libavcodec.63.dylib", "libavcodec.61.dylib")]
    [InlineData("Unix", "libavcodec.so.62", "libavcodec.so.63", "libavcodec.so.61")]
    public void AnotherMajorIsNamedForEachOperatingSystem(string system, string expected, string ffmpeg9, string ffmpeg7)
    {
        var naming = Enum.Parse<LibraryNaming>(system);
        string dir = Directory.CreateTempSubdirectory("ffmpeg-naming-").FullName;
        try
        {
            foreach (string file in new[] { ffmpeg9, ffmpeg7, "libavcodec.so.63.3.100", "libavcodec.63.3.100.dylib", "avcodec-63.lib", "avformat-63.dll", "libavformat.so.63" })
            {
                File.WriteAllBytes(Path.Combine(dir, file), []);
            }

            string message = EmbeddingGemma2FFmpeg.NotFoundMessage(dir, "avcodec", 62, naming);
            output.WriteLine(message);
            Assert.Contains($"shared library {expected} could not be loaded", message, StringComparison.Ordinal);
            Assert.Contains($"Found {ffmpeg7}, {ffmpeg9} instead", message, StringComparison.Ordinal);
            Assert.Contains($", {expected}; the adapter package of the installed major loads it", message, StringComparison.Ordinal);
            Assert.DoesNotContain("63.3.100", message, StringComparison.Ordinal);
            Assert.DoesNotContain("avformat", message, StringComparison.Ordinal);

            // The bound major itself (present but not loadable) is not another major.
            File.WriteAllBytes(Path.Combine(dir, expected), []);
            Assert.Contains($"Found {ffmpeg7}, {ffmpeg9} instead", EmbeddingGemma2FFmpeg.NotFoundMessage(dir, "avcodec", 62, naming), StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Theory]
    [InlineData("Windows", "avcodec-62.dll", 62)]
    [InlineData("Windows", "avcodec-62.lib", null)]
    [InlineData("Windows", "avcodec-.dll", null)]
    [InlineData("MacOS", "libavcodec.62.dylib", 62)]
    [InlineData("MacOS", "libavcodec.62.3.100.dylib", null)]
    [InlineData("Unix", "libavcodec.so.62", 62)]
    [InlineData("Unix", "libavcodec.so.62.3.100", null)]
    [InlineData("Unix", "libavformat.so.62", null)]
    public void MajorIsParsedFromExactFileNames(string system, string file, int? major) =>
        Assert.Equal(major, FFmpegNative.MajorOf(file, "avcodec", Enum.Parse<LibraryNaming>(system)));
}
