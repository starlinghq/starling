using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>
/// Video inputs without the graphs: rendered ids against the accepted video fixtures (tokenizer only; CI), the three
/// frame forms and their rules (metadata-less frames rejected by default, upstream's behaviour on opt-in), limits and
/// validation.
/// </summary>
public sealed class VideoInputTests(ITestOutputHelper output)
{
    private static readonly EmbeddingGemma2Tolerances Tolerances = EmbeddingGemma2Tolerances.Load("video-fp32-cpu");

    private static EmbeddingGemma2Image Frame(int width, int height) => EmbeddingGemma2Image.FromRgb24(width, height, new byte[width * height * 3]);

    [Fact]
    public void RenderedIdsEqualTheFixtureOnEveryRow()
    {
        Assert.True(Tolerances.IsExact("token_ids"));
        var tokenizer = EmbeddingGemma2TestAssets.Tokenizer;
        var options = new EmbeddingGemma2MultimodalOptions { AllowVideosWithoutTiming = true };
        int n = 0;
        foreach (var row in EmbeddingGemma2VideoTestAssets.Rows)
        {
            // Rendering depends only on each video's frame size and sampled count, and the still's size.
            var videos = row.Videos.Select(v => v.InputForm == "array"
                ? EmbeddingGemma2Video.FromFramesWithoutTiming(Enumerable.Repeat(Frame(v.DecodedWidth, v.DecodedHeight), 150).ToArray())
                : EmbeddingGemma2VideoTestAssets.SizedPlaceholder(v)).ToList();
            var still = row.Image is null ? null : Frame(1280, 720);
            var input = EmbeddingGemma2VideoTestAssets.BuildInput(row, videos, still);
            var prepared = EmbeddingGemma2MultimodalEmbedder.PrepareCore(tokenizer, options, input, row.PackagePrompt, 0);
            Assert.True(row.InputIds.AsSpan().SequenceEqual(prepared.Tokens.InputIds.Span), $"{row.Id}: rendered ids differ from the fixture");
            Assert.Equal(row.Videos.Select(v => v.SoftTokensPerFrame), prepared.Videos.Select(v => v.SoftTokensPerFrame));
            Assert.Equal(row.Videos.Select(v => v.SampledIndices.Length), prepared.Videos.Select(v => v.Frames.Length));
            n++;
        }

        output.WriteLine($"{n} rows (22 video-only, 6 interleaved or prompted, 3 input forms, 5 text): ids exact with the owned tokenizer");
        Assert.Equal(36, n);
    }

    [Fact]
    public void FramesWithoutTimingAreRejectedByDefault()
    {
        var tokenizer = EmbeddingGemma2TestAssets.Tokenizer;
        var frames = Enumerable.Repeat(Frame(64, 48), 150).ToArray();
        var video = EmbeddingGemma2Video.FromFramesWithoutTiming(frames);
        Assert.Equal(EmbeddingGemma2VideoFrameSource.AllFramesWithoutTiming, video.FrameSource);
        Assert.Throws<EmbeddingGemma2VideoTimingRequiredException>(() => video.GetSampledFrameIndices());
        var ex = Assert.Throws<EmbeddingGemma2VideoTimingRequiredException>(() =>
            EmbeddingGemma2MultimodalEmbedder.PrepareCore(tokenizer, new EmbeddingGemma2MultimodalOptions(), EmbeddingGemma2MultimodalInput.FromVideo(video), EmbeddingGemma2Prompt.None, 0));
        Assert.Contains("AllowVideosWithoutTiming", ex.Message, StringComparison.Ordinal);
        Assert.IsAssignableFrom<ArgumentException>(ex);

        // Opt-in: upstream's behaviour (every frame, thinned to 32) instead of 1 frame per second.
        var prepared = EmbeddingGemma2MultimodalEmbedder.PrepareCore(tokenizer, new EmbeddingGemma2MultimodalOptions { AllowVideosWithoutTiming = true }, EmbeddingGemma2MultimodalInput.FromVideo(video), EmbeddingGemma2Prompt.None, 0);
        Assert.Equal(EmbeddingGemma2VideoSampling.SampleIndices(150, null, null), prepared.Videos[0].FrameIndices);
        Assert.Equal(32, prepared.Videos[0].Frames.Length);

        // The same frames with timing: 5 frames, as the file.
        var timed = EmbeddingGemma2Video.FromDecodedFrames(frames, 30, 5);
        Assert.Equal([0, 30, 60, 90, 120], timed.GetSampledFrameIndices());
    }

    [Fact]
    public void PreSampledFramesAreUsedAsGiven()
    {
        var a = Frame(64, 48);
        var video = EmbeddingGemma2Video.FromSampledFrames([a, a, a], [0, 8, 16]);
        Assert.Equal([0, 1, 2], video.GetSampledFrameIndices());
        Assert.Equal([0, 8, 16], video.SourceFrameIndices);
        Assert.Throws<ArgumentException>(() => EmbeddingGemma2Video.FromSampledFrames(Enumerable.Repeat(a, 33).ToArray()));
        Assert.Throws<ArgumentException>(() => EmbeddingGemma2Video.FromSampledFrames([]));
        Assert.Throws<ArgumentException>(() => EmbeddingGemma2Video.FromSampledFrames([a], [1, 2]));
    }

    [Fact]
    public void OnlyTheSampledFramesAreRead()
    {
        var tokenizer = EmbeddingGemma2TestAssets.Tokenizer;
        var lazy = new CountingFrames(1_200, 64, 48);
        var video = EmbeddingGemma2Video.FromDecodedFrames(lazy, 30, 40);
        var prepared = EmbeddingGemma2MultimodalEmbedder.PrepareCore(tokenizer, new EmbeddingGemma2MultimodalOptions(), EmbeddingGemma2MultimodalInput.FromVideo(video), EmbeddingGemma2Prompt.None, 0);
        Assert.Equal(32, prepared.Videos[0].Frames.Length);
        Assert.Equal(prepared.Videos[0].FrameIndices.Distinct().Order(), lazy.Read.Order());
    }

    [Fact]
    public void TimingIsValidated()
    {
        var frames = new[] { Frame(8, 8) };
        Assert.Throws<ArgumentOutOfRangeException>(() => EmbeddingGemma2Video.FromDecodedFrames(frames, 0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => EmbeddingGemma2Video.FromDecodedFrames(frames, double.NaN, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => EmbeddingGemma2Video.FromDecodedFrames(frames, 30, -1));
        Assert.Throws<ArgumentException>(() => EmbeddingGemma2Video.FromDecodedFrames([], 30, 1));
        Assert.Throws<ArgumentException>(() => EmbeddingGemma2Video.FromFramesWithoutTiming([]));
    }

    [Fact]
    public void FramesOfOneVideoMustShareASize()
    {
        var tokenizer = EmbeddingGemma2TestAssets.Tokenizer;
        var video = EmbeddingGemma2Video.FromSampledFrames([Frame(64, 48), Frame(48, 64)]);
        Assert.Throws<ArgumentException>(() =>
            EmbeddingGemma2MultimodalEmbedder.PrepareCore(tokenizer, new EmbeddingGemma2MultimodalOptions(), EmbeddingGemma2MultimodalInput.FromVideo(video), EmbeddingGemma2Prompt.None, 0));
        var large = EmbeddingGemma2Video.FromSampledFrames([Frame(100, 100)]);
        Assert.Throws<EmbeddingGemma2ImageTooLargeException>(() =>
            EmbeddingGemma2MultimodalEmbedder.PrepareCore(tokenizer, new EmbeddingGemma2MultimodalOptions { MaxImagePixels = 9_999 }, EmbeddingGemma2MultimodalInput.FromVideo(large), EmbeddingGemma2Prompt.None, 0));
    }

    [Fact]
    public void InputsOverTheContextAreRejectedBeforeInference()
    {
        var tokenizer = EmbeddingGemma2TestAssets.Tokenizer;

        // 672x480 frames reach the full 140 tokens per frame: a 32-frame video is 32 x 142 = 4,544 tokens.
        var full = EmbeddingGemma2Video.FromSampledFrames(Enumerable.Repeat(Frame(672, 480), 32).ToArray());
        var one = EmbeddingGemma2MultimodalEmbedder.PrepareCore(tokenizer, new EmbeddingGemma2MultimodalOptions(), EmbeddingGemma2MultimodalInput.FromVideo(full), EmbeddingGemma2Prompt.None, 0);
        Assert.Equal(4_544 + 2, one.Tokens.Length);
        Assert.Equal(140, one.Videos[0].SoftTokensPerFrame);

        // Two maximal videos: 9,090 tokens > 8,192.
        var ex = Assert.Throws<EmbeddingGemma2InputTooLongException>(() =>
            EmbeddingGemma2MultimodalEmbedder.PrepareCore(tokenizer, new EmbeddingGemma2MultimodalOptions(), EmbeddingGemma2MultimodalInput.Create(full, full), EmbeddingGemma2Prompt.None, 2));
        Assert.Equal(2, ex.InputIndex);
        Assert.Equal(8_192, ex.MaximumTokens);
    }

    [Fact]
    public void VideosInterleaveWithTextAndImages()
    {
        var tokenizer = EmbeddingGemma2TestAssets.Tokenizer;
        var image = Frame(48, 48);
        var video = EmbeddingGemma2Video.FromSampledFrames([Frame(48, 48), Frame(48, 48)]); // 2 frames
        var input = EmbeddingGemma2MultimodalInput.Create("a", video, "b", image);
        Assert.Equal((1, 1, true, true, true), (input.ImageCount, input.VideoCount, input.HasImages, input.HasVideos, input.HasMedia));
        var p = EmbeddingGemma2MultimodalEmbedder.PrepareCore(tokenizer, new EmbeddingGemma2MultimodalOptions(), input, EmbeddingGemma2Prompt.None, 0);
        long[] ids = p.Tokens.InputIds.ToArray();
        const long S = EmbeddingGemma2MultimodalEmbedder.ImageStartTokenId, E = EmbeddingGemma2MultimodalEmbedder.ImageEndTokenId;
        const long V = EmbeddingGemma2MultimodalEmbedder.VideoTokenId, I = EmbeddingGemma2MultimodalEmbedder.ImageTokenId;
        var media = ids.Where(id => id is S or E or V or I).ToArray();
        var runs = media.Where((id, k) => k == 0 || media[k - 1] != id).ToArray(); // collapse runs of soft-token ids
        Assert.Equal([S, V, E, S, V, E, S, I, E], runs);
        Assert.Equal(2 * p.Videos[0].SoftTokensPerFrame, media.Count(id => id == V));
        Assert.Equal(p.SoftTokenCounts[0], media.Count(id => id == I));
        Assert.Equal(258_884, EmbeddingGemma2MultimodalEmbedder.VideoTokenId);
        Assert.Equal(140, EmbeddingGemma2MultimodalEmbedder.VideoFrameSoftTokens);
        Assert.Throws<ArgumentException>(() => EmbeddingGemma2MultimodalInput.Create("x <|video|>", video));
    }

    [Fact]
    public void OptionsValidate()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingGemma2MultimodalOptions { MaxVideoFramesPerVisionCall = 0 }.Validate());
        var defaults = new EmbeddingGemma2MultimodalOptions();
        Assert.Equal((2, false), (defaults.MaxVideoFramesPerVisionCall, defaults.AllowVideosWithoutTiming));
    }

    /// <summary>A lazy frame list that records which frames were read.</summary>
    private sealed class CountingFrames(int count, int width, int height) : IReadOnlyList<EmbeddingGemma2Image>
    {
        private readonly EmbeddingGemma2Image _frame = Frame(width, height);

        public HashSet<int> Read { get; } = [];

        public int Count => count;

        public EmbeddingGemma2Image this[int index]
        {
            get
            {
                Read.Add(index);
                return _frame;
            }
        }

        public IEnumerator<EmbeddingGemma2Image> GetEnumerator() => throw new InvalidOperationException("enumerated");

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
