using System.Security.Cryptography;
using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Video.FFmpeg.Tests;

/// <summary>
/// The FFmpeg adapter against torchcodec (decoder profile of <c>embeddinggemma-2/video-fp32-cpu</c>, binding key
/// <c>decoder_ffmpeg.decoded_rgb_vs_torchcodec: exact</c> on FFmpeg 8 x86-64): metadata, sampled indices and the decoded
/// RGB of every frame of every fixture video. Needs the video set and an FFmpeg 8.x LGPL shared build
/// (<c>STARLING_FFMPEG_DIR</c>); skips otherwise.
/// </summary>
public sealed class DecoderTests(ITestOutputHelper output)
{
    private static readonly EmbeddingGemma2Tolerances Tolerances = EmbeddingGemma2Tolerances.Load("video-fp32-cpu");

    [Fact]
    public void TheLoadedFFmpegIsTheGatedMajorAndLgpl()
    {
        var decoder = EmbeddingGemma2VideoTestAssets.Decoder;
        var libs = decoder.Libraries;
        output.WriteLine($"FFmpeg {libs.Version} from {libs.Directory}: {string.Join(", ", libs.LibraryVersions.Select(kv => $"lib{kv.Key} {kv.Value}"))}");
        output.WriteLine($"licenses: {string.Join(", ", libs.Licenses.Select(kv => $"lib{kv.Key}: {kv.Value}"))}");
        Assert.Equal(Tolerances["decoder_ffmpeg.supported_ffmpeg_major"], EmbeddingGemma2FFmpeg.SupportedFFmpegMajor.ToString(System.Globalization.CultureInfo.InvariantCulture));
        Assert.StartsWith("n8.", libs.Version, StringComparison.Ordinal);
        Assert.True(libs.IsLgpl);
        Assert.Same(libs, EmbeddingGemma2FFmpeg.Load(new EmbeddingGemma2FFmpegOptions { LibraryDirectory = libs.Directory }));
    }

    [Fact]
    public void MetadataAndSampledIndicesEqualTorchcodec()
    {
        var decoder = EmbeddingGemma2VideoTestAssets.Decoder;
        foreach (var m in EmbeddingGemma2VideoTestAssets.Manifest)
        {
            var expected = EmbeddingGemma2VideoTestAssets.Frames[m.Id];
            var meta = decoder.Probe(EmbeddingGemma2VideoTestAssets.PathOf(m.Id));
            Assert.Equal(expected.Metadata.NumFrames, meta.FrameCount);
            Assert.Equal(expected.Metadata.DurationSeconds, meta.DurationSeconds);
            Assert.Equal(expected.Metadata.BeginSeconds, meta.BeginSeconds);
            Assert.Equal(expected.Metadata.EndSeconds, meta.EndSeconds);
            Assert.Equal(expected.Metadata.AverageFps, meta.AverageFramesPerSecond);
            Assert.Equal((expected.Metadata.Width, expected.Metadata.Height), (meta.Width, meta.Height));
            Assert.Equal(expected.Metadata.Codec, meta.Codec);
            Assert.Equal(expected.Metadata.Rotation, meta.Rotation);
            Assert.Equal(expected.SampledIndices, EmbeddingGemma2VideoSampling.SampleIndices(meta.FrameCount, meta.AverageFramesPerSecond, meta.DurationSeconds));
        }

        output.WriteLine($"{EmbeddingGemma2VideoTestAssets.Manifest.Count} videos: frame count, duration, begin/end, average rate, size, codec, rotation and sampled indices equal torchcodec 0.17.0");
    }

    [Fact]
    public void SampledFramesEqualTorchcodecBitForBit()
    {
        Assert.Equal("exact", Tolerances["decoder_ffmpeg.decoded_rgb_vs_torchcodec"]);
        Assert.True(Tolerances.IsExact("decoded_frames"));
        Assert.True(Tolerances.IsExact("frame_indices"));
        int frames = 0;
        foreach (var m in EmbeddingGemma2VideoTestAssets.Manifest)
        {
            var expected = EmbeddingGemma2VideoTestAssets.Frames[m.Id];
            var decoded = EmbeddingGemma2VideoTestAssets.Decode(m.Id);
            Assert.Equal(EmbeddingGemma2VideoFrameSource.PreSampled, decoded.Video.FrameSource);
            Assert.Equal(expected.SampledIndices, decoded.FrameIndices);
            Assert.Equal(expected.SampledIndices, decoded.Video.SourceFrameIndices);
            for (int k = 0; k < decoded.FrameIndices.Count; k++)
            {
                var frame = decoded.Video.Frames[k];
                Assert.Equal((expected.Metadata.Width, expected.Metadata.Height), (frame.Width, frame.Height));
                Assert.True(expected.AllFrameSha256[decoded.FrameIndices[k]] == Sha(frame.Pixels.Span), $"{m.Id} frame {decoded.FrameIndices[k]}: decoded RGB differs from torchcodec's");
                frames++;
            }

            // The row digest of the sampled frames (T x H x W x 3) as well.
            var row = EmbeddingGemma2VideoTestAssets.Rows.First(r => r.Videos.Count == 1 && r.Videos[0].Video == m.Id && r.Videos[0].InputForm == "path");
            Assert.Equal(row.Videos[0].FramesSha256, EmbeddingGemma2VideoTestAssets.Sha(decoded.Video.Frames.Select(f => f.Pixels)));
        }

        output.WriteLine($"{EmbeddingGemma2VideoTestAssets.Manifest.Count} videos, {frames} sampled frames: decoded RGB equal to torchcodec on FFmpeg 8 (bound: exact)");
    }

    [Fact]
    public void EveryDecodedFrameEqualsTorchcodecBitForBit()
    {
        var decoder = EmbeddingGemma2VideoTestAssets.Decoder;
        int total = 0;
        foreach (var m in EmbeddingGemma2VideoTestAssets.Manifest)
        {
            var expected = EmbeddingGemma2VideoTestAssets.Frames[m.Id].AllFrameSha256;

            // In chunks, so at most 64 decoded frames are held at a time.
            for (int start = 0; start < expected.Length; start += 64)
            {
                int[] indices = Enumerable.Range(start, Math.Min(64, expected.Length - start)).ToArray();
                var frames = decoder.DecodeFramesAt(EmbeddingGemma2VideoTestAssets.PathOf(m.Id), indices, TestContext.Current.CancellationToken);
                for (int k = 0; k < indices.Length; k++)
                {
                    Assert.True(expected[indices[k]] == Sha(frames[k].Pixels.Span), $"{m.Id} frame {indices[k]} ({m.Codec}, {m.Bits}-bit, rotation {m.Rotation}): decoded RGB differs from torchcodec's");
                }
            }

            total += expected.Length;
        }

        output.WriteLine($"{EmbeddingGemma2VideoTestAssets.Manifest.Count} videos, all {total} frames: decoded RGB equal to torchcodec on FFmpeg 8");
    }

    [Fact]
    public void StreamsAndMemoryDecodeLikeFiles()
    {
        var decoder = EmbeddingGemma2VideoTestAssets.Decoder;
        foreach (string id in new[] { "h264-bars-640x360-30fps-5s", "vp9-odd-321x179-30fps-3s", "h264-rot90-640x360-30fps-3s", "av1-640x360-30fps-3s" })
        {
            var file = EmbeddingGemma2VideoTestAssets.Decode(id);
            byte[] bytes = File.ReadAllBytes(EmbeddingGemma2VideoTestAssets.PathOf(id));
            using var stream = new MemoryStream(bytes);
            stream.Position = 0;
            var fromStream = decoder.Decode(stream, TestContext.Current.CancellationToken);
            var fromMemory = decoder.Decode(bytes.AsMemory(), TestContext.Current.CancellationToken);
            foreach (var other in new[] { fromStream, fromMemory })
            {
                Assert.Equal(file.Metadata, other.Metadata);
                Assert.Equal(file.FrameIndices, other.FrameIndices);
                Assert.Equal(file.Video.Frames.Select(f => Sha(f.Pixels.Span)), other.Video.Frames.Select(f => Sha(f.Pixels.Span)));
            }

            Assert.Equal(file.Metadata, decoder.Probe(new MemoryStream(bytes)));
        }
    }

    [Fact]
    public void AllFramesCarryTheTimingAndSampleLikeTheFile()
    {
        var all = EmbeddingGemma2VideoTestAssets.AllFrames("h264-bars-640x360-30fps-5s");
        Assert.Equal(EmbeddingGemma2VideoFrameSource.AllFramesWithTiming, all.Video.FrameSource);
        Assert.Equal(150, all.Video.Frames.Count);
        Assert.Equal(30, all.Video.FramesPerSecond);
        Assert.Equal(5, all.Video.DurationSeconds);
        Assert.Equal(EmbeddingGemma2VideoTestAssets.Decode("h264-bars-640x360-30fps-5s").FrameIndices, all.Video.GetSampledFrameIndices());
    }

    [Fact]
    public void NonVideoInputIsATypedError()
    {
        var decoder = EmbeddingGemma2VideoTestAssets.Decoder;
        string path = Path.Combine(EmbeddingGemma2TestAssets.RepositoryRoot, "ai-dotnet", "README.md");
        Assert.Throws<EmbeddingGemma2VideoDecodeException>(() => decoder.DecodeFile(path, TestContext.Current.CancellationToken));
        Assert.Throws<EmbeddingGemma2VideoDecodeException>(() => decoder.Decode(new byte[] { 1, 2, 3, 4 }.AsMemory(), TestContext.Current.CancellationToken));
        Assert.Throws<FileNotFoundException>(() => decoder.DecodeFile(path + ".missing.mp4", TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentOutOfRangeException>(() => decoder.DecodeFramesAt(EmbeddingGemma2VideoTestAssets.PathOf("h264-1s-320x240-30fps"), [30], TestContext.Current.CancellationToken));
    }

    [Fact]
    public void CancellationStopsDecoding()
    {
        var decoder = EmbeddingGemma2VideoTestAssets.Decoder;
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.ThrowsAny<OperationCanceledException>(() => decoder.DecodeFile(EmbeddingGemma2VideoTestAssets.PathOf("h264-long-320x240-30fps-40s"), cts.Token));
    }

    [Fact]
    public void FrameLimitsAreEnforced()
    {
        _ = EmbeddingGemma2VideoTestAssets.Decoder; // skips without FFmpeg or videos
        var small = new EmbeddingGemma2VideoDecoder(new EmbeddingGemma2FFmpegOptions { LibraryDirectory = Environment.GetEnvironmentVariable(EmbeddingGemma2VideoTestAssets.FFmpegEnvironmentVariable), MaxFramePixels = 1_000, MaxAllFrames = 10 });
        Assert.Throws<EmbeddingGemma2ImageTooLargeException>(() => small.DecodeFile(EmbeddingGemma2VideoTestAssets.PathOf("h264-1s-320x240-30fps"), TestContext.Current.CancellationToken));
        Assert.Throws<EmbeddingGemma2VideoDecodeException>(() => small.DecodeAllFrames(EmbeddingGemma2VideoTestAssets.PathOf("h264-1s-320x240-30fps"), TestContext.Current.CancellationToken));
    }

    private static string Sha(ReadOnlySpan<byte> bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
}
