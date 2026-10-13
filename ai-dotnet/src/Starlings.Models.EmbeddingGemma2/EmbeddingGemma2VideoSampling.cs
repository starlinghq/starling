namespace Starlings.Models.EmbeddingGemma2;

/// <summary>
/// Frame sampling of the pinned revision (<c>EmbeddingGemma2VideoProcessor.sample_frames</c>, transformers 5.19.0) with
/// the checkpoint's settings: 1 frame per second, at most 32 frames, overflow strategy "uniform". Restated exactly,
/// upstream quirks included (equal to upstream on 20,849 cases, <c>docs/m0/embeddinggemma2-video-spike.md</c> section
/// 2.2).
/// </summary>
/// <remarks>
/// <para>
/// With a frame rate and a duration: <c>n = max(1, int(duration))</c> candidates (truncated: clips shorter than 2 s give
/// one frame), candidate <c>i</c> is frame <c>min(total - 1, int(i * fps))</c> (by frame index from the average rate, not
/// by timestamp), then more than 32 candidates are thinned to 32 as <c>np.linspace(0, n - 1, 32, dtype=int)</c> picks them.
/// Without a frame rate or duration (upstream's decoded array without metadata) every frame is a candidate before the
/// 32-frame thinning, which gives a different embedding than the same file; this package rejects such input unless
/// <see cref="EmbeddingGemma2MultimodalOptions.AllowVideosWithoutTiming"/> is set.
/// </para>
/// <para>
/// For parity with upstream, pass the metadata torchcodec computes (exact seek mode): the frame count is the number of
/// video packets, the duration is <c>max(pts + duration) - min(pts)</c> of those packets in seconds, and the average
/// rate is <c>frames / duration</c>. <c>Starlings.Models.EmbeddingGemma2.Video.FFmpeg</c> computes exactly that.
/// </para>
/// </remarks>
public static class EmbeddingGemma2VideoSampling
{
    /// <summary>Sampling rate of the checkpoint's video processor (<c>fps</c> 1).</summary>
    public const double TargetFramesPerSecond = 1;

    /// <summary>Frames per video at most (<c>max_frames</c> 32).</summary>
    public const int MaxFrames = 32;

    /// <summary>
    /// The indices of the frames upstream keeps, ascending (repeats are possible when the frame count is short of the
    /// duration, as upstream).
    /// </summary>
    /// <param name="totalFrames">Frames in the video (at least 1).</param>
    /// <param name="framesPerSecond">Average frame rate, or null when unknown.</param>
    /// <param name="durationSeconds">Duration in seconds, or null when unknown.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="totalFrames"/> is below 1, or a timing value is not finite or negative.</exception>
    public static int[] SampleIndices(int totalFrames, double? framesPerSecond, double? durationSeconds)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(totalFrames, 1);
        if (framesPerSecond is { } f && (!double.IsFinite(f) || f < 0))
        {
            throw new ArgumentOutOfRangeException(nameof(framesPerSecond), f, "The frame rate must be finite and not negative.");
        }

        if (durationSeconds is { } d && (!double.IsFinite(d) || d < 0))
        {
            throw new ArgumentOutOfRangeException(nameof(durationSeconds), d, "The duration must be finite and not negative.");
        }

        int[] indices;
        if (framesPerSecond is null || durationSeconds is null)
        {
            indices = new int[totalFrames];
            for (int i = 0; i < totalFrames; i++)
            {
                indices[i] = i;
            }
        }
        else
        {
            double step = framesPerSecond.Value / TargetFramesPerSecond;

            // int(duration * fps): truncation toward zero, at least one frame.
            double candidates = Math.Truncate(durationSeconds.Value * TargetFramesPerSecond);
            int count = (int)Math.Clamp(candidates, 1, int.MaxValue);
            if (count > MaxFrames)
            {
                // Only the thinned candidates are ever used; compute them directly (same values, no large array).
                return Thin(count, i => CandidateIndex(i, step, totalFrames));
            }

            indices = new int[count];
            for (int i = 0; i < count; i++)
            {
                indices[i] = CandidateIndex(i, step, totalFrames);
            }
        }

        return indices.Length <= MaxFrames ? indices : Thin(indices.Length, i => indices[i]);
    }

    /// <summary>Candidate <paramref name="i"/>: <c>min(total - 1, int(i * step))</c> in float64.</summary>
    private static int CandidateIndex(long i, double step, int totalFrames)
    {
        double value = Math.Truncate(i * step);
        return value >= totalFrames - 1 ? totalFrames - 1 : (int)value;
    }

    /// <summary>
    /// <c>np.linspace(0, n - 1, 32, dtype=int)</c> over <paramref name="n"/> candidates: <c>floor(i * ((n - 1) / 31))</c>
    /// in float64 for <c>i &lt; 31</c>, the last exactly <c>n - 1</c>.
    /// </summary>
    private static int[] Thin(int n, Func<int, int> candidate)
    {
        double s = (double)(n - 1) / (MaxFrames - 1);
        var picked = new int[MaxFrames];
        for (int i = 0; i < MaxFrames - 1; i++)
        {
            picked[i] = candidate((int)Math.Floor(i * s));
        }

        picked[MaxFrames - 1] = candidate(n - 1);
        return picked;
    }
}
