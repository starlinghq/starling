namespace Starlings.Models.EmbeddingGemma2;

/// <summary>How the frames of an <see cref="EmbeddingGemma2Video"/> relate to the source video.</summary>
public enum EmbeddingGemma2VideoFrameSource
{
    /// <summary>
    /// Every decoded frame with the video's average frame rate and duration: the package samples the frames exactly as
    /// upstream (<see cref="EmbeddingGemma2VideoSampling"/>).
    /// </summary>
    AllFramesWithTiming,

    /// <summary>1 to 32 frames the caller already sampled, used as given (the caller's choice of frames).</summary>
    PreSampled,

    /// <summary>
    /// Every decoded frame without timing metadata. Rejected unless
    /// <see cref="EmbeddingGemma2MultimodalOptions.AllowVideosWithoutTiming"/> is set; then upstream's behaviour for a
    /// decoded array without metadata applies (every frame a candidate, thinned to 32), which gives a different embedding
    /// than the same file (cosine 0.965 on a 5 s fixture clip).
    /// </summary>
    AllFramesWithoutTiming,
}

/// <summary>
/// An already-decoded video for EmbeddingGemma 2: 8-bit RGB frames (<see cref="EmbeddingGemma2Image"/>, all of one size)
/// and how they were obtained. Upstream decodes with torchcodec (FFmpeg) and samples 1 frame per second up to 32 frames;
/// this package starts from the decoded frames and reproduces everything after decoding exactly (sampling, the 140-token
/// per-frame resize, patches, placeholders).
/// </summary>
/// <remarks>
/// <para>
/// Three forms: <see cref="FromDecodedFrames"/> (every frame plus the average frame rate and duration; the package
/// samples), <see cref="FromSampledFrames"/> (up to 32 frames the caller sampled), and <see cref="FromFramesWithoutTiming"/>
/// (rejected by default). Decode with your own stack, or with <c>Starlings.Models.EmbeddingGemma2.Video.FFmpeg</c>,
/// which reproduces torchcodec's metadata, rotation and RGB conversion on the user's own FFmpeg 8.x libraries.
/// </para>
/// <para>
/// <b>Large videos.</b> Only the sampled frames of <see cref="Frames"/> are read (at most 32). <see cref="Frames"/> may be a
/// lazy <see cref="IReadOnlyList{T}"/> that produces a frame when indexed, so a caller does not have to hold every decoded
/// frame in memory. Frames are read on the embedding call and must not change while it runs. Instances are immutable.
/// </para>
/// </remarks>
public sealed class EmbeddingGemma2Video
{
    private EmbeddingGemma2Video(IReadOnlyList<EmbeddingGemma2Image> frames, EmbeddingGemma2VideoFrameSource source, double? fps, double? duration, IReadOnlyList<int>? sourceIndices)
    {
        Frames = frames;
        FrameSource = source;
        FramesPerSecond = fps;
        DurationSeconds = duration;
        SourceFrameIndices = sourceIndices;
    }

    /// <summary>The frames: every decoded frame, or the pre-sampled frames (see <see cref="FrameSource"/>).</summary>
    public IReadOnlyList<EmbeddingGemma2Image> Frames { get; }

    /// <summary>How <see cref="Frames"/> relate to the source video.</summary>
    public EmbeddingGemma2VideoFrameSource FrameSource { get; }

    /// <summary>Average frame rate (frames / duration), when the frames carry timing.</summary>
    public double? FramesPerSecond { get; }

    /// <summary>Duration in seconds, when the frames carry timing.</summary>
    public double? DurationSeconds { get; }

    /// <summary>
    /// For <see cref="EmbeddingGemma2VideoFrameSource.PreSampled"/>: the source frame index of each frame when the caller
    /// gave them (informational; never used for embedding). Null otherwise.
    /// </summary>
    public IReadOnlyList<int>? SourceFrameIndices { get; }

    /// <summary>
    /// Every decoded frame of a video plus its average frame rate and duration (as torchcodec reports them: packet count,
    /// <c>max(pts + duration) - min(pts)</c> in seconds, <c>frames / duration</c>). The package samples the frames exactly as
    /// upstream. <paramref name="frames"/> may be lazy; only the sampled frames are read.
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="frames"/> is empty.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A timing value is not finite, the frame rate is not positive, or the duration is negative.</exception>
    public static EmbeddingGemma2Video FromDecodedFrames(IReadOnlyList<EmbeddingGemma2Image> frames, double framesPerSecond, double durationSeconds)
    {
        ArgumentNullException.ThrowIfNull(frames);
        if (frames.Count == 0)
        {
            throw new ArgumentException("A video needs at least one frame.", nameof(frames));
        }

        if (!double.IsFinite(framesPerSecond) || framesPerSecond <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(framesPerSecond), framesPerSecond, "The frame rate must be finite and positive.");
        }

        if (!double.IsFinite(durationSeconds) || durationSeconds < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(durationSeconds), durationSeconds, "The duration must be finite and not negative.");
        }

        return new EmbeddingGemma2Video(frames, EmbeddingGemma2VideoFrameSource.AllFramesWithTiming, framesPerSecond, durationSeconds, null);
    }

    /// <summary>
    /// 1 to 32 frames the caller already sampled, embedded in the order given (no further sampling). This equals the file's
    /// embedding when the frames are the ones upstream samples (<see cref="EmbeddingGemma2VideoSampling.SampleIndices"/>);
    /// any other choice is the caller's and gives a different embedding.
    /// </summary>
    /// <param name="frames">The sampled frames.</param>
    /// <param name="sourceFrameIndices">Optional source frame index of each frame (informational).</param>
    /// <exception cref="ArgumentException">No frames, more than 32, or <paramref name="sourceFrameIndices"/> of another length.</exception>
    public static EmbeddingGemma2Video FromSampledFrames(IReadOnlyList<EmbeddingGemma2Image> frames, IReadOnlyList<int>? sourceFrameIndices = null)
    {
        ArgumentNullException.ThrowIfNull(frames);
        if (frames.Count is 0 or > EmbeddingGemma2VideoSampling.MaxFrames)
        {
            throw new ArgumentException($"Pre-sampled videos hold 1 to {EmbeddingGemma2VideoSampling.MaxFrames} frames (the checkpoint's max_frames); got {frames.Count}. Pass every frame with FromDecodedFrames to let the package sample.", nameof(frames));
        }

        if (sourceFrameIndices is not null && sourceFrameIndices.Count != frames.Count)
        {
            throw new ArgumentException($"{sourceFrameIndices.Count} source indices for {frames.Count} frames.", nameof(sourceFrameIndices));
        }

        return new EmbeddingGemma2Video(frames, EmbeddingGemma2VideoFrameSource.PreSampled, null, null, sourceFrameIndices?.ToArray());
    }

    /// <summary>
    /// Every decoded frame with no timing metadata. Embedding it throws <see cref="EmbeddingGemma2VideoTimingRequiredException"/>
    /// unless <see cref="EmbeddingGemma2MultimodalOptions.AllowVideosWithoutTiming"/> is set, in which case upstream's
    /// behaviour for an array without metadata applies (all frames, thinned to 32): a different embedding than the same
    /// file, which upstream samples at 1 frame per second.
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="frames"/> is empty.</exception>
    public static EmbeddingGemma2Video FromFramesWithoutTiming(IReadOnlyList<EmbeddingGemma2Image> frames)
    {
        ArgumentNullException.ThrowIfNull(frames);
        if (frames.Count == 0)
        {
            throw new ArgumentException("A video needs at least one frame.", nameof(frames));
        }

        return new EmbeddingGemma2Video(frames, EmbeddingGemma2VideoFrameSource.AllFramesWithoutTiming, null, null, null);
    }

    /// <summary>
    /// The indices into <see cref="Frames"/> that are embedded: upstream's sampling for
    /// <see cref="EmbeddingGemma2VideoFrameSource.AllFramesWithTiming"/>, every frame for
    /// <see cref="EmbeddingGemma2VideoFrameSource.PreSampled"/>, and (when allowed) every frame thinned to 32 for
    /// <see cref="EmbeddingGemma2VideoFrameSource.AllFramesWithoutTiming"/>.
    /// </summary>
    /// <param name="allowWithoutTiming">Whether frames without timing are accepted (<see cref="EmbeddingGemma2MultimodalOptions.AllowVideosWithoutTiming"/>).</param>
    /// <exception cref="EmbeddingGemma2VideoTimingRequiredException">The frames carry no timing and <paramref name="allowWithoutTiming"/> is false.</exception>
    public int[] GetSampledFrameIndices(bool allowWithoutTiming = false) => FrameSource switch
    {
        EmbeddingGemma2VideoFrameSource.AllFramesWithTiming => EmbeddingGemma2VideoSampling.SampleIndices(Frames.Count, FramesPerSecond, DurationSeconds),
        EmbeddingGemma2VideoFrameSource.PreSampled => Enumerable.Range(0, Frames.Count).ToArray(),
        _ => allowWithoutTiming
            ? EmbeddingGemma2VideoSampling.SampleIndices(Frames.Count, null, null)
            : throw new EmbeddingGemma2VideoTimingRequiredException(),
    };
}

/// <summary>
/// A video's frames carry no frame rate or duration and
/// <see cref="EmbeddingGemma2MultimodalOptions.AllowVideosWithoutTiming"/> is not set. Upstream would keep every frame
/// and thin to 32, which differs from embedding the same file (1 frame per second). Pass the timing
/// (<see cref="EmbeddingGemma2Video.FromDecodedFrames"/>), pre-sampled frames (<see cref="EmbeddingGemma2Video.FromSampledFrames"/>),
/// or opt in to upstream's behaviour. Nothing was embedded.
/// </summary>
public sealed class EmbeddingGemma2VideoTimingRequiredException : ArgumentException
{
    /// <summary>Creates the exception with the default explanation.</summary>
    public EmbeddingGemma2VideoTimingRequiredException()
        : base("The video's frames carry no frame rate or duration. Upstream would keep every frame and thin to 32 frames, which gives a different embedding than the same file (sampled at 1 frame per second). " +
               "Use EmbeddingGemma2Video.FromDecodedFrames(frames, fps, duration) or FromSampledFrames(frames), or set EmbeddingGemma2MultimodalOptions.AllowVideosWithoutTiming to opt in to upstream's behaviour.")
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public EmbeddingGemma2VideoTimingRequiredException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public EmbeddingGemma2VideoTimingRequiredException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
