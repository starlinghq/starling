using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using Starlings.TestSupport;
#if STARLINGS_FFMPEG9
namespace Starlings.Models.EmbeddingGemma2.Video.FFmpeg9.Tests;
#else
namespace Starlings.Models.EmbeddingGemma2.Video.FFmpeg.Tests;
#endif

/// <summary>
/// The FFmpeg adapter against torchcodec (decoder profile of <c>embeddinggemma-2/video-fp32-cpu</c>, binding key
/// <c>decoder_ffmpeg.decoded_rgb_vs_torchcodec: exact</c> on FFmpeg 8 x86-64): metadata, sampled indices and the decoded
/// RGB of every frame of every fixture video. Needs the video set and an FFmpeg 8.x LGPL shared build
/// (<c>STARLING_FFMPEG_DIR</c>); skips otherwise.
/// </summary>
/// <remarks>
/// Shared with the FFmpeg 9 adapter's tests (compiled there with STARLINGS_FFMPEG9; <c>STARLING_FFMPEG9_DIR</c>): the same
/// checks on FFmpeg 9 against the same protected digests, which torchcodec 0.17.0 on FFmpeg 9.0.1 reproduces for all
/// 5,367 frames (<c>model-pipeline/reference/embeddinggemma2/torchcodec_eg2_frames.py</c>). The FFmpeg 9 bounds
/// (<c>decoder_ffmpeg9.*</c>) are proposed, not yet in the protected file; until they are, the tests hold FFmpeg 9 to
/// <c>exact</c> and say so.
/// </remarks>
public sealed class DecoderTests(ITestOutputHelper output)
{
    private static readonly EmbeddingGemma2Tolerances Tolerances = EmbeddingGemma2Tolerances.Load("video-fp32-cpu");

    private static readonly int Major = EmbeddingGemma2FFmpeg.SupportedFFmpegMajor;

    /// <summary>The tolerance section of this adapter's decoder profile: <c>decoder_ffmpeg</c> (8), <c>decoder_ffmpeg9</c> (9).</summary>
    private static readonly string Section = Major == 8 ? "decoder_ffmpeg" : $"decoder_ffmpeg{Major}";

    [Fact]
    public void TheLoadedFFmpegIsTheGatedMajorAndLgpl()
    {
        var decoder = EmbeddingGemma2VideoTestAssets.Decoder;
        var libs = decoder.Libraries;
        output.WriteLine($"FFmpeg {libs.Version} from {libs.Directory}: {string.Join(", ", libs.LibraryVersions.Select(kv => $"lib{kv.Key} {kv.Value}"))}");
        output.WriteLine($"licenses: {string.Join(", ", libs.Licenses.Select(kv => $"lib{kv.Key}: {kv.Value}"))}");
        if (Tolerances.Contains($"{Section}.supported_ffmpeg_major"))
        {
            Assert.Equal(Tolerances[$"{Section}.supported_ffmpeg_major"], Major.ToString(CultureInfo.InvariantCulture));
        }
        else
        {
            output.WriteLine($"{Section}.supported_ffmpeg_major is not in {Tolerances.Source} yet (proposed); the adapter binds FFmpeg {Major}");
        }

        Assert.StartsWith($"n{Major}.", libs.Version, StringComparison.Ordinal);
        if (EmbeddingGemma2VideoTestAssets.AllowNonLgplBuild)
        {
            output.WriteLine($"{EmbeddingGemma2VideoTestAssets.AllowNonLgplEnvironmentVariable}=1: license not required to be LGPL (IsLgpl {libs.IsLgpl})");
            if (!libs.IsLgpl)
            {
                // Without the opt-in, the same loaded build is refused.
                Assert.Throws<EmbeddingGemma2FFmpegLicenseException>(() => EmbeddingGemma2FFmpeg.Load(new EmbeddingGemma2FFmpegOptions { LibraryDirectory = libs.Directory }));
            }
        }
        else
        {
            Assert.True(libs.IsLgpl);
        }

        Assert.Same(libs, EmbeddingGemma2FFmpeg.Load(new EmbeddingGemma2FFmpegOptions { LibraryDirectory = libs.Directory, AllowNonLgplBuild = EmbeddingGemma2VideoTestAssets.AllowNonLgplBuild }));
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
        if (Tolerances.Contains($"{Section}.decoded_rgb_vs_torchcodec"))
        {
            Assert.Equal("exact", Tolerances[$"{Section}.decoded_rgb_vs_torchcodec"]);
        }
        else
        {
            output.WriteLine($"{Section}.decoded_rgb_vs_torchcodec is not in {Tolerances.Source} yet (proposed: exact); held to exact");
        }

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

        output.WriteLine($"{EmbeddingGemma2VideoTestAssets.Manifest.Count} videos, {frames} sampled frames: decoded RGB equal to torchcodec on FFmpeg {Major} (bound: exact)");
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

        output.WriteLine($"{EmbeddingGemma2VideoTestAssets.Manifest.Count} videos, all {total} frames: decoded RGB equal to torchcodec on FFmpeg {Major}");
    }

    /// <summary>
    /// Capabilities <c>video_decode_rotation_other</c>, <c>video_hdr</c>, <c>video_interlaced</c> and
    /// <c>video_variable_frame_rate</c>: the nine edge-case videos of <c>make_eg2_edge_videos.py</c> (rotation -90 and
    /// 180, HEVC Main 10 BT.2020 PQ and HLG, H.264 interlaced top and bottom field first, H.264 with varying frame
    /// durations) against torchcodec 0.17.0 on FFmpeg 8 (proposed <c>video-edge/</c>): metadata, sampled indices and
    /// the RGB of every frame. Needs the edge videos (<c>STARLING_EMBEDDINGGEMMA2_EDGE_VIDEOS</c>); skips otherwise.
    /// </summary>
    [Fact]
    public void EdgeCaseVideosEqualTorchcodec()
    {
        var decoder = EmbeddingGemma2VideoTestAssets.Decoder;
        var (videos, frames, dir) = EmbeddingGemma2VideoTestAssets.LoadEdge();
        string first = Path.Combine(dir, videos[0].File);
        Assert.SkipWhen(!File.Exists(first), $"edge videos not found ({first}); write them with make_eg2_edge_videos.py --out <dir> and set {EmbeddingGemma2VideoTestAssets.EdgeVideosEnvironmentVariable}.");
        var failures = new List<string>();
        int total = 0;
        foreach (var v in videos)
        {
            string path = Path.Combine(dir, v.File);
            Assert.True(v.Sha256 == Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path))), $"{v.File}: not the proposed file");
            var expected = frames[v.Id];
            var meta = decoder.Probe(path);
            var problems = new List<string>();
            if (expected.Metadata.NumFrames != meta.FrameCount || expected.Metadata.DurationSeconds != meta.DurationSeconds ||
                expected.Metadata.BeginSeconds != meta.BeginSeconds || expected.Metadata.EndSeconds != meta.EndSeconds ||
                expected.Metadata.AverageFps != meta.AverageFramesPerSecond || (expected.Metadata.Width, expected.Metadata.Height) != (meta.Width, meta.Height) ||
                expected.Metadata.Codec != meta.Codec || expected.Metadata.Rotation != meta.Rotation)
            {
                problems.Add($"metadata {meta} vs torchcodec {expected.Metadata}");
            }

            if (!expected.SampledIndices.SequenceEqual(EmbeddingGemma2VideoSampling.SampleIndices(meta.FrameCount, meta.AverageFramesPerSecond, meta.DurationSeconds)))
            {
                problems.Add("sampled indices differ");
            }

            int differing = 0;
            for (int start = 0; start < expected.AllFrameSha256.Length; start += 64)
            {
                int[] indices = Enumerable.Range(start, Math.Min(64, expected.AllFrameSha256.Length - start)).ToArray();
                var decoded = decoder.DecodeFramesAt(path, indices, TestContext.Current.CancellationToken);
                for (int k = 0; k < indices.Length; k++)
                {
                    differing += expected.AllFrameSha256[indices[k]] == Sha(decoded[k].Pixels.Span) ? 0 : 1;
                }
            }

            if (differing > 0)
            {
                problems.Add($"{differing} of {expected.AllFrameSha256.Length} frames differ");
            }

            var sampled = decoder.DecodeFile(path, TestContext.Current.CancellationToken);
            if (!sampled.FrameIndices.SequenceEqual(expected.SampledIndices))
            {
                problems.Add("DecodeFile sampled other frames");
            }

            total += expected.AllFrameSha256.Length;
            output.WriteLine($"{v.Id}: {(problems.Count == 0 ? $"metadata, {expected.SampledIndices.Length} sampled indices and all {expected.AllFrameSha256.Length} frames equal torchcodec" : string.Join("; ", problems))}");
            if (problems.Count > 0)
            {
                failures.Add(v.Id);
            }
        }

        output.WriteLine($"{videos.Count} edge-case videos, {total} frames, FFmpeg {Major}; differing: {(failures.Count == 0 ? "none" : string.Join(", ", failures))}");
        Assert.Empty(failures);
        Assert.Equal(9, videos.Count);
    }

    /// <summary>Directory holding the four P-030 release videos (<c>fetch_eg2_release_media.py</c>); the test skips without it.</summary>
    private const string ReleaseVideosEnvironmentVariable = "STARLING_EMBEDDINGGEMMA2_RELEASE_VIDEOS";

    /// <summary>
    /// The P-030 video release set's sampled frames, which torchcodec decodes by seeking: on <c>commons-video-02</c> (Ogg
    /// Theora) its seek to frame 24 yields a frame that differs from a sequential decode, and the adapter must return
    /// torchcodec's frame. Checked against the protected per-row frame digests (<c>release/video-fp32-release</c>); the
    /// sequential frames are checked against <c>video-frames.json</c>. Needs the release videos and FFmpeg 8 (the
    /// protected digests are torchcodec's on FFmpeg 8.1.3); skips otherwise.
    /// </summary>
    [Fact]
    public void ReleaseVideosSampleTorchcodecsSeekedFrames()
    {
        Assert.SkipWhen(Major != 8, "the release digests are torchcodec's on FFmpeg 8");
        string? dir = Environment.GetEnvironmentVariable(ReleaseVideosEnvironmentVariable);
        Assert.SkipWhen(dir is not { Length: > 0 } || !Directory.Exists(dir), $"set {ReleaseVideosEnvironmentVariable} to the release videos (fetch_eg2_release_media.py)");
        Assert.SkipWhen(EmbeddingGemma2VideoTestAssets.FFmpegSkipReason is not null, EmbeddingGemma2VideoTestAssets.FFmpegSkipReason ?? string.Empty);
        var decoder = new EmbeddingGemma2VideoDecoder(new EmbeddingGemma2FFmpegOptions { AllowNonLgplBuild = EmbeddingGemma2VideoTestAssets.AllowNonLgplBuild });
        string release = Path.Combine(EmbeddingGemma2TestAssets.ProtectedRoot, "release", "video-fp32-release");
        using var frames = JsonDocument.Parse(File.ReadAllText(Path.Combine(release, "video-frames.json")));
        int videos = 0;
        foreach (string line in File.ReadLines(Path.Combine(release, "video-embeddings.jsonl")))
        {
            using var row = JsonDocument.Parse(line);
            var r = row.RootElement;
            if (r.GetProperty("input_form").GetString() != "path" || r.GetProperty("videos").GetArrayLength() != 1 || r.GetProperty("prompt").GetString() != "none" ||
                r.GetProperty("text").ValueKind != JsonValueKind.Null || r.GetProperty("image").ValueKind != JsonValueKind.Null)
            {
                continue;
            }

            var v = r.GetProperty("videos")[0];
            string id = v.GetProperty("video").GetString()!;
            string path = Directory.GetFiles(dir!, id + ".*").Single();
            string[] want = [.. v.GetProperty("frame_sha256").EnumerateArray().Select(e => e.GetString()!)];
            var decoded = decoder.DecodeFile(path, TestContext.Current.CancellationToken);
            Assert.Equal(want, decoded.Video.Frames.Select(f => Sha(f.Pixels.Span)));
            videos++;
            if (id == "commons-video-02")
            {
                // Sequential decoding (torchcodec's get_frames_in_range) gives another frame 24.
                string[] sequential = [.. frames.RootElement.GetProperty(id).GetProperty("all_frame_sha256").EnumerateArray().Select(e => e.GetString()!)];
                var all = decoder.DecodeAllFrames(path, TestContext.Current.CancellationToken);
                Assert.Equal(sequential, all.Video.Frames.Select(f => Sha(f.Pixels.Span)));
                Assert.NotEqual(sequential[24], want[1]);
                output.WriteLine($"{id}: sampled frame 24 {want[1][..12]} (seek) vs {sequential[24][..12]} (sequential); every frame of the sequential decode equals torchcodec's");
            }
        }

        Assert.Equal(4, videos);
        output.WriteLine($"{videos} release videos: sampled frames equal torchcodec's (seek-based get_frames_at)");
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
        var small = new EmbeddingGemma2VideoDecoder(new EmbeddingGemma2FFmpegOptions { LibraryDirectory = Environment.GetEnvironmentVariable(EmbeddingGemma2VideoTestAssets.FFmpegEnvironmentVariable), MaxFramePixels = 1_000, MaxAllFrames = 10, AllowNonLgplBuild = EmbeddingGemma2VideoTestAssets.AllowNonLgplBuild });
        Assert.Throws<EmbeddingGemma2ImageTooLargeException>(() => small.DecodeFile(EmbeddingGemma2VideoTestAssets.PathOf("h264-1s-320x240-30fps"), TestContext.Current.CancellationToken));
        Assert.Throws<EmbeddingGemma2VideoDecodeException>(() => small.DecodeAllFrames(EmbeddingGemma2VideoTestAssets.PathOf("h264-1s-320x240-30fps"), TestContext.Current.CancellationToken));
    }

    private static string Sha(ReadOnlySpan<byte> bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
}
