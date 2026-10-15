using System.Runtime.InteropServices;

#if STARLINGS_FFMPEG9
namespace Starlings.Models.EmbeddingGemma2.Video.FFmpeg9;
#else
namespace Starlings.Models.EmbeddingGemma2.Video.FFmpeg;
#endif

/// <summary>
/// Decodes video files and streams into <see cref="EmbeddingGemma2Video"/> exactly as upstream sees them (torchcodec
/// 0.17.0 on CPU, uint8 RGB, <c>seek_mode="exact"</c>), on the user's own FFmpeg shared libraries of the major this package binds
/// (<see cref="EmbeddingGemma2FFmpeg"/>; none are shipped).
/// </summary>
/// <remarks>
/// <para>
/// <b>Metadata</b> (<see cref="Probe(string)"/>): torchcodec's packet scan without decoding, on the best video stream: the
/// frame count is the stream's packets (packets flagged discard skipped), the duration is
/// <c>max(pts + duration) - min(pts)</c> in seconds (pts, or dts when pts is unset), the average rate is
/// <c>frames / duration</c>, the rotation comes from the stream's display matrix.
/// </para>
/// <para>
/// <b>Sampling and decoding</b> (<see cref="DecodeFile(string, CancellationToken)"/>): upstream's indices
/// (<see cref="EmbeddingGemma2VideoSampling.SampleIndices"/> on that metadata; 1 frame per second, at most 32), then
/// torchcodec's <c>get_frames_at</c> on the same demuxer: in index order, each frame is reached by decoding on or by
/// seeking to the scanned keyframe before it (<c>avformat_seek_file</c>, then <c>avcodec_flush_buffers</c>), with torchcodec's
/// rule for when a seek is avoided (which depends on the decoder's frame threads), and only the sampled frames are converted
/// (at most 32 in memory). The decoder state after a seek is part of upstream's output: on an Ogg Theora video of the
/// release set, torchcodec's frame after a seek differs from a sequential decode, and this adapter returns torchcodec's.
/// </para>
/// <para>
/// <b>RGB conversion</b>, as torchcodec calls FFmpeg: when the frame width is a multiple of 32 and no rotation applies,
/// one <c>sws_scale</c> to RGB24 with flags 0 (as torchcodec passes them), the source and destination coefficients
/// <c>sws_getCoefficients(frame colorspace)</c> (BT.601 when untagged) and the source range from the frame when tagged;
/// otherwise a filter graph <c>buffer</c> -&gt; <c>copy</c> (or <c>format=rgb24,transpose=cclock</c>,
/// <c>format=rgb24,transpose=clock</c>, <c>format=rgb24,hflip,vflip</c> for a display matrix of 90, -90 or 180 degrees)
/// -&gt; <c>buffersink</c> RGB24. This reproduces torchcodec's decoded RGB bit for bit on the 22 fixture videos (H.264
/// 8- and 10-bit, HEVC, VP9, AV1; rotation 90; odd sizes) on x86-64 with FFmpeg 8.1 (this code with the FFmpeg 8 bindings)
/// and FFmpeg 9.0.1 (with the FFmpeg 9 bindings), against torchcodec on the same FFmpeg. Other CPUs (swscale's SIMD paths)
/// and FFmpeg builds are not covered by that evidence.
/// </para>
/// <para>Instances are immutable and thread-safe; each call opens its own FFmpeg contexts.</para>
/// </remarks>
public sealed unsafe class EmbeddingGemma2VideoDecoder
{
    private const int IoBufferSize = 64 * 1024;

    private readonly EmbeddingGemma2FFmpegOptions _options;

    /// <summary>Loads and checks the FFmpeg libraries (<see cref="EmbeddingGemma2FFmpeg.Load"/>).</summary>
    /// <exception cref="EmbeddingGemma2FFmpegException">FFmpeg of the bound major is missing, has another major, or is a GPL or nonfree build that is not allowed.</exception>
    public EmbeddingGemma2VideoDecoder(EmbeddingGemma2FFmpegOptions? options = null)
    {
        options ??= new EmbeddingGemma2FFmpegOptions();
        options.Validate();
        _options = options;
        Libraries = EmbeddingGemma2FFmpeg.Load(options);
    }

    /// <summary>Options in effect.</summary>
    public EmbeddingGemma2FFmpegOptions Options => _options;

    /// <summary>The FFmpeg libraries in use.</summary>
    public EmbeddingGemma2FFmpegLibraries Libraries { get; }

#pragma warning disable CA1822 // Instance members: the constructor loads and checks the FFmpeg libraries first.

    /// <summary>torchcodec's metadata of a video file (packet scan, no decoding).</summary>
    /// <exception cref="EmbeddingGemma2VideoDecodeException">The file does not open or has no video stream.</exception>
    public EmbeddingGemma2VideoMetadata Probe(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        using var input = Input.OpenFile(path);
        return Scan(input, CancellationToken.None);
    }

    /// <summary>torchcodec's metadata of a video in a seekable stream (read from its current position to the end).</summary>
    public EmbeddingGemma2VideoMetadata Probe(Stream stream)
    {
        using var input = Input.OpenStream(stream);
        return Scan(input, CancellationToken.None);
    }

#pragma warning restore CA1822

    /// <summary>Decodes a video file: upstream's sampled frames (at most 32), with the metadata and the sampled indices.</summary>
    /// <exception cref="EmbeddingGemma2VideoDecodeException">The file does not open, has no video stream, or fails to decode.</exception>
    public EmbeddingGemma2DecodedVideo DecodeFile(string path, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        return Decode(() => Input.OpenFile(path), cancellationToken);
    }

    /// <summary>
    /// Decodes a video from a seekable stream (read from its current position; the stream is not disposed): upstream's
    /// sampled frames, as <see cref="DecodeFile(string, CancellationToken)"/>. The container is detected from the content.
    /// </summary>
    public EmbeddingGemma2DecodedVideo Decode(Stream stream, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(stream);
        long start = stream.CanSeek ? stream.Position : throw new ArgumentException("The stream must be seekable (the metadata scan reads it twice).", nameof(stream));
        return Decode(
            () =>
            {
                stream.Position = start;
                return Input.OpenStream(stream);
            },
            cancellationToken);
    }

    /// <summary>Decodes an encoded video held in memory (for example a <c>DataContent</c>'s bytes).</summary>
    public EmbeddingGemma2DecodedVideo Decode(ReadOnlyMemory<byte> data, CancellationToken cancellationToken = default)
    {
        using var stream = MemoryMarshal.TryGetArray(data, out var segment)
            ? new MemoryStream(segment.Array!, segment.Offset, segment.Count, writable: false)
            : new MemoryStream(data.ToArray(), writable: false);
        return Decode(stream, cancellationToken);
    }

    /// <summary>
    /// Decodes every frame of a video file (at most <see cref="EmbeddingGemma2FFmpegOptions.MaxAllFrames"/>), for your own
    /// sampling or for <see cref="EmbeddingGemma2Video.FromFramesWithoutTiming"/>. <see cref="EmbeddingGemma2DecodedVideo.Video"/>
    /// is <see cref="EmbeddingGemma2Video.FromDecodedFrames"/> with the metadata's rate and duration (the package then
    /// samples exactly as <see cref="DecodeFile(string, CancellationToken)"/> does), or every frame without timing when
    /// the duration is zero.
    /// </summary>
    /// <exception cref="EmbeddingGemma2VideoDecodeException">The video has more frames than allowed, or fails to decode.</exception>
    public EmbeddingGemma2DecodedVideo DecodeAllFrames(string path, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        using var input = Input.OpenFile(path);
        var metadata = Scan(input, cancellationToken, out var table);
        if (metadata.FrameCount > _options.MaxAllFrames)
        {
            throw new EmbeddingGemma2VideoDecodeException($"{path} has {metadata.FrameCount} frames; DecodeAllFrames keeps at most {_options.MaxAllFrames} (EmbeddingGemma2FFmpegOptions.MaxAllFrames). Use DecodeFile, which keeps only the sampled frames.");
        }

        int[] all = Enumerable.Range(0, metadata.FrameCount).ToArray();
        input.SeekToStartAfterScan();
        var frames = DecodeFrames(input, metadata, table, all, cancellationToken);
        var video = metadata.AverageFramesPerSecond is { } fps
            ? EmbeddingGemma2Video.FromDecodedFrames(frames, fps, metadata.DurationSeconds)
            : EmbeddingGemma2Video.FromFramesWithoutTiming(frames);
        return new EmbeddingGemma2DecodedVideo(video, metadata, all);
    }

    /// <summary>
    /// The frames at <paramref name="indices"/> (presentation order indices, as torchcodec's <c>get_frames_at</c>), in the
    /// order given; repeats allowed. Decodes as torchcodec does (seeking where it seeks) and keeps only the requested frames.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">An index is negative or not below the scanned frame count.</exception>
    public IReadOnlyList<EmbeddingGemma2Image> DecodeFramesAt(string path, IReadOnlyList<int> indices, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        ArgumentNullException.ThrowIfNull(indices);
        if (indices.Count == 0)
        {
            return [];
        }

        using var input = Input.OpenFile(path);
        var metadata = Scan(input, cancellationToken, out var table);
        foreach (int i in indices)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(i, nameof(indices));
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(i, metadata.FrameCount, nameof(indices));
        }

        input.SeekToStartAfterScan();
        return DecodeFrames(input, metadata, table, [.. indices], cancellationToken);
    }

    private EmbeddingGemma2DecodedVideo Decode(Func<Input> open, CancellationToken cancellationToken)
    {
        using var input = open();
        var metadata = Scan(input, cancellationToken, out var table);

        // Upstream: sample from the scanned metadata (average_fps is None when the duration is 0: every frame, capped).
        int[] indices = EmbeddingGemma2VideoSampling.SampleIndices(metadata.FrameCount, metadata.AverageFramesPerSecond, metadata.AverageFramesPerSecond is null ? null : metadata.DurationSeconds);
        input.SeekToStartAfterScan();
        var frames = DecodeFrames(input, metadata, table, indices, cancellationToken);
        return new EmbeddingGemma2DecodedVideo(EmbeddingGemma2Video.FromSampledFrames(frames, indices), metadata, indices);
    }

    private static EmbeddingGemma2VideoMetadata Scan(Input input, CancellationToken cancellationToken) => Scan(input, cancellationToken, out _);

    /// <summary>
    /// torchcodec's exact-mode scan (<c>scan_file_and_update_metadata_and_index</c>): every packet of the video stream (packets
    /// flagged discard skipped), its pts (dts when unset) and keyframe flag; the frame table is sorted by pts.
    /// </summary>
    private static EmbeddingGemma2VideoMetadata Scan(Input input, CancellationToken cancellationToken, out FrameTable table)
    {
        var fmt = input.Format;
        int si = input.StreamIndex;
        var st = fmt->streams[si];
        long begin = long.MaxValue, end = long.MinValue;
        int n = 0;
        var pts = new List<long>();
        var keys = new List<bool>();
        var pkt = FFmpegNative.av_packet_alloc();
        try
        {
            int r;
            while ((r = FFmpegNative.av_read_frame(fmt, pkt)) >= 0)
            {
                if (pkt->stream_index == si && (pkt->flags & FFmpegNative.AV_PKT_FLAG_DISCARD) == 0)
                {
                    long p = pkt->pts != FFmpegNative.AV_NOPTS_VALUE ? pkt->pts : pkt->dts;
                    begin = Math.Min(begin, p);
                    end = Math.Max(end, p + pkt->duration);
                    pts.Add(p);
                    keys.Add((pkt->flags & FFmpegNative.AV_PKT_FLAG_KEY) != 0);
                    n++;
                    if ((n & 1023) == 0)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                    }
                }

                FFmpegNative.av_packet_unref(pkt);
            }

            if (r != FFmpegNative.AVERROR_EOF)
            {
                throw new EmbeddingGemma2VideoDecodeException($"{input.Name}: reading packets failed: {Error(r)}.");
            }
        }
        finally
        {
            FFmpegNative.av_packet_free(&pkt);
        }

        if (n == 0)
        {
            throw new EmbeddingGemma2VideoDecodeException($"{input.Name}: the video stream has no packets.");
        }

        table = FrameTable.Create(pts, keys, st->time_base);
        var tb = st->time_base;
        double Seconds(long p) => (double)p * tb.num / tb.den;
        double? rotation = null;
        var sd = FFmpegNative.av_packet_side_data_get(st->codecpar->coded_side_data, st->codecpar->nb_coded_side_data, FFmpegNative.AV_PKT_DATA_DISPLAYMATRIX);
        if (sd != null && sd->size >= 9 * sizeof(int))
        {
            double degrees = FFmpegNative.av_display_rotation_get((int*)sd->data);
            if (!double.IsNaN(degrees))
            {
                rotation = degrees;
            }
        }

        double beginSeconds = Seconds(begin), endSeconds = Seconds(end);
        bool quarter = rotation is { } deg && Math.Abs(Math.Round(deg / 90.0, MidpointRounding.AwayFromZero)) % 2 == 1;
        return new EmbeddingGemma2VideoMetadata(
            n,
            beginSeconds,
            endSeconds,
            quarter ? st->codecpar->height : st->codecpar->width,
            quarter ? st->codecpar->width : st->codecpar->height,
            rotation,
            FFmpegNative.Utf8(FFmpegNative.avcodec_get_name(st->codecpar->codec_id)),
            PixelFormatName(st->codecpar->format),
            si);
    }

    /// <summary>
    /// torchcodec's <c>get_frames_at</c> on the decoder that made the scan (<c>SingleStreamDecoder</c>, exact mode, CPU): the
    /// indices are decoded in sorted order (a repeated index is copied); an index that does not follow the previous one sets
    /// the cursor to its scanned pts, and the cursor is reached either by decoding on or by seeking
    /// (<c>avformat_seek_file</c> to the scanned keyframe at or before it, then <c>avcodec_flush_buffers</c>) with
    /// torchcodec's rule for when a seek is avoided; the frame returned is the first decoded frame whose pts (dts when unset)
    /// is at or after the cursor. The decoder state after a seek is part of what upstream returns (an Ogg Theora seek, for
    /// example, can yield a frame that differs from a sequential decode), so the same calls are made in the same order.
    /// </summary>
    private List<EmbeddingGemma2Image> DecodeFrames(Input input, EmbeddingGemma2VideoMetadata metadata, FrameTable table, int[] indices, CancellationToken cancellationToken)
    {
        int quarterTurns = QuarterTurns(metadata.Rotation);
        var fmt = input.Format;
        int si = input.StreamIndex;
        var st = fmt->streams[si];
        int[] order = Enumerable.Range(0, indices.Length).OrderBy(k => indices[k]).ToArray();
        var output = new EmbeddingGemma2Image?[indices.Length];
        var codec = FFmpegNative.avcodec_find_decoder(st->codecpar->codec_id);
        if (codec == null)
        {
            throw new EmbeddingGemma2VideoDecodeException($"{input.Name}: the FFmpeg build has no decoder for {metadata.Codec}.");
        }

        AVCodecContext* cc = FFmpegNative.avcodec_alloc_context3(codec);
        AVPacket* pkt = FFmpegNative.av_packet_alloc();
        AVFrame* frame = FFmpegNative.av_frame_alloc();
        using var converter = new Converter(quarterTurns, _options.MaxFramePixels, input.Name);
        try
        {
            Check(FFmpegNative.avcodec_parameters_to_context(cc, st->codecpar), input.Name, "avcodec_parameters_to_context");
            cc->thread_count = _options.DecoderThreads;
            cc->pkt_timebase = st->time_base;
            Check(FFmpegNative.avcodec_open2(cc, codec, null), input.Name, "avcodec_open2");

            // torchcodec reads only the active stream once the codec is open.
            for (uint s = 0; s < fmt->nb_streams; s++)
            {
                if (s != (uint)si)
                {
                    fmt->streams[s]->discard = FFmpegNative.AVDISCARD_ALL;
                }
            }

            var state = new SeekState();
            EmbeddingGemma2Image? previous = null;
            int previousIndex = -1;
            foreach (int slot in order)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int index = indices[slot];
                if (previous is not null && index == previousIndex)
                {
                    output[slot] = previous;
                    continue;
                }

                if (index != state.LastIndex + 1)
                {
                    state.Cursor = table.SecondsToClosestPts(table.PtsToSeconds(table.Pts[index]));
                    state.CursorJustSet = true;
                }

                DecodeAt(input, cc, pkt, frame, table, state, cancellationToken);
                try
                {
                    previous = output[slot] = converter.Convert(frame, st->time_base);
                }
                finally
                {
                    FFmpegNative.av_frame_unref(frame);
                }

                previousIndex = index;
                state.LastIndex = index;
            }
        }
        finally
        {
            FFmpegNative.av_frame_free(&frame);
            FFmpegNative.av_packet_free(&pkt);
            FFmpegNative.avcodec_free_context(&cc);
        }

        return output.Select(f => f!).ToList();
    }

    /// <summary>torchcodec's <c>decode_av_frame</c> with the filter <c>pts &gt;= cursor</c>: on return <paramref name="frame"/> holds that frame.</summary>
    private static void DecodeAt(Input input, AVCodecContext* cc, AVPacket* pkt, AVFrame* frame, FrameTable table, SeekState state, CancellationToken cancellationToken)
    {
        var fmt = input.Format;
        int si = input.StreamIndex;
        if (!CanAvoidSeeking(cc, table, state))
        {
            long desired = state.Cursor;
            int key = table.KeyFrameFor(desired);
            if (key >= 0)
            {
                desired = table.KeyPts[key];
            }

            int s = FFmpegNative.avformat_seek_file(fmt, si, long.MinValue, desired, desired, 0);
            if (s < 0)
            {
                throw new EmbeddingGemma2VideoDecodeException($"{input.Name}: seeking to pts {desired} failed: {Error(s)} (torchcodec seeks as well and fails the same way).");
            }

            FFmpegNative.avcodec_flush_buffers(cc);
        }

        state.CursorJustSet = false;
        bool reachedEof = false;
        int r;
        int received = 0;
        while (true)
        {
            r = FFmpegNative.avcodec_receive_frame(cc, frame);
            if (r != 0 && r != FFmpegNative.AVERROR_EAGAIN)
            {
                break;
            }

            if (r == 0)
            {
                if (PtsOrDts(frame) >= state.Cursor)
                {
                    break;
                }

                FFmpegNative.av_frame_unref(frame);
                if ((++received & 63) == 0)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                }

                continue;
            }

            if (reachedEof)
            {
                continue;
            }

            int p;
            do
            {
                FFmpegNative.av_packet_unref(pkt);
                p = FFmpegNative.av_read_frame(fmt, pkt);
            }
            while (p >= 0 && pkt->stream_index != si);

            if (p == FFmpegNative.AVERROR_EOF)
            {
                Check(FFmpegNative.avcodec_send_packet(cc, null), input.Name, "avcodec_send_packet (flush)");
                reachedEof = true;
                continue;
            }

            if (p < 0)
            {
                throw new EmbeddingGemma2VideoDecodeException($"{input.Name}: reading packets failed: {Error(p)}.");
            }

            try
            {
                Check(FFmpegNative.avcodec_send_packet(cc, pkt), input.Name, "avcodec_send_packet");
            }
            finally
            {
                FFmpegNative.av_packet_unref(pkt);
            }
        }

        if (r < 0)
        {
            if (reachedEof || r == FFmpegNative.AVERROR_EOF)
            {
                throw new EmbeddingGemma2VideoDecodeException($"{input.Name}: the decoder has no frame at or after pts {state.Cursor} (the packet scan counted {table.Pts.Length} frames).");
            }

            Check(r, input.Name, "avcodec_receive_frame");
        }

        state.LastPts = PtsOrDts(frame);
    }

    /// <summary>torchcodec's <c>can_we_avoid_seeking</c> (video, CPU).</summary>
    private static bool CanAvoidSeeking(AVCodecContext* cc, FrameTable table, SeekState state)
    {
        if (!state.CursorJustSet)
        {
            return true;
        }

        if (state.Cursor < state.LastPts || state.Cursor == state.LastPts)
        {
            return false;
        }

        int lastKey = table.KeyFrameFor(state.LastPts);
        int targetKey = table.KeyFrameFor(state.Cursor);
        if (lastKey < 0 || targetKey < 0)
        {
            return false;
        }

        if (lastKey == targetKey)
        {
            return true;
        }

        long x = table.SecondsToIndexLowerBound(table.PtsToSeconds(state.LastPts));
        long j = table.KeyFrameIndex[targetKey];
        long reorder = Math.Max(cc->has_b_frames, 0);
        long inFlight = (cc->active_thread_type & FFmpegNative.FF_THREAD_FRAME) != 0 ? Math.Max(cc->thread_count, 1) - 1 : 0;
        return j - x <= reorder + inFlight;
    }

    private static long PtsOrDts(AVFrame* frame) => frame->pts == FFmpegNative.AV_NOPTS_VALUE ? frame->pkt_dts : frame->pts;

    /// <summary>torchcodec's decoding position: the cursor pts, whether it was just set, the last decoded frame.</summary>
    private sealed class SeekState
    {
        public long Cursor { get; set; } = long.MinValue;

        public bool CursorJustSet { get; set; }

        public long LastPts { get; set; } = long.MinValue;

        public long LastIndex { get; set; } = long.MinValue;
    }

    /// <summary>torchcodec's scanned index (<c>all_frames</c>, <c>key_frames</c> after <c>sort_all_frames</c>).</summary>
    private sealed class FrameTable
    {
        private FrameTable(long[] pts, long[] nextPts, long[] keyPts, long[] keyFrameIndex, AVRational timeBase)
        {
            Pts = pts;
            NextPts = nextPts;
            KeyPts = keyPts;
            KeyFrameIndex = keyFrameIndex;
            TimeBase = timeBase;
        }

        /// <summary>Frame pts in presentation order (the frame index is the position).</summary>
        public long[] Pts { get; }

        /// <summary>The next frame's pts (<see cref="long.MaxValue"/> for the last frame).</summary>
        public long[] NextPts { get; }

        /// <summary>Keyframe pts, sorted.</summary>
        public long[] KeyPts { get; }

        /// <summary>The frame index of each keyframe.</summary>
        public long[] KeyFrameIndex { get; }

        public AVRational TimeBase { get; }

        public static FrameTable Create(List<long> pts, List<bool> keys, AVRational timeBase)
        {
            // Sorted by pts (std::sort in torchcodec; equal pts keep their packet order here).
            int[] order = Enumerable.Range(0, pts.Count).OrderBy(i => pts[i]).ToArray();
            long[] sorted = order.Select(i => pts[i]).ToArray();
            long[] next = new long[sorted.Length];
            for (int i = 0; i < sorted.Length; i++)
            {
                next[i] = i + 1 < sorted.Length ? sorted[i + 1] : long.MaxValue;
            }

            var keyPts = new List<long>();
            var keyIndex = new List<long>();
            for (int i = 0; i < order.Length; i++)
            {
                if (keys[order[i]])
                {
                    keyPts.Add(sorted[i]);
                    keyIndex.Add(i);
                }
            }

            return new FrameTable(sorted, next, [.. keyPts], [.. keyIndex], timeBase);
        }

        public double PtsToSeconds(long pts) => (double)pts * TimeBase.num / TimeBase.den;

        public long SecondsToClosestPts(double seconds) => (long)Math.Round(seconds * TimeBase.den / TimeBase.num, MidpointRounding.AwayFromZero);

        /// <summary>The last keyframe at or before <paramref name="pts"/> (its position in <see cref="KeyPts"/>), or -1.</summary>
        public int KeyFrameFor(long pts)
        {
            int lo = 0, hi = KeyPts.Length;
            while (lo < hi)
            {
                int mid = (lo + hi) >>> 1;
                if (pts < KeyPts[mid])
                {
                    hi = mid;
                }
                else
                {
                    lo = mid + 1;
                }
            }

            return lo - 1;
        }

        /// <summary>The first frame whose next pts (in seconds) is after <paramref name="seconds"/>.</summary>
        public long SecondsToIndexLowerBound(double seconds)
        {
            int lo = 0, hi = NextPts.Length;
            while (lo < hi)
            {
                int mid = (lo + hi) >>> 1;
                if (PtsToSeconds(NextPts[mid]) <= seconds)
                {
                    lo = mid + 1;
                }
                else
                {
                    hi = mid;
                }
            }

            return lo;
        }
    }

    /// <summary>torchcodec's rotation handling: the display matrix angle rounded to a multiple of 90 degrees.</summary>
    internal static int QuarterTurns(double? rotation)
    {
        if (rotation is null)
        {
            return 0;
        }

        int rounded = (int)Math.Round(rotation.Value / 90.0, MidpointRounding.AwayFromZero) * 90;
        return rounded switch
        {
            0 => 0,
            90 => 1,
            -90 => -1,
            180 or -180 => 2,
            _ => throw new EmbeddingGemma2VideoDecodeException($"Unsupported display rotation {rotation} degrees."),
        };
    }

    private static void Check(int result, string name, string what)
    {
        if (result < 0)
        {
            throw new EmbeddingGemma2VideoDecodeException($"{name}: {what} failed: {Error(result)}.");
        }
    }

    internal static string Error(int code)
    {
        const int Size = 256;
        byte* buffer = stackalloc byte[Size];
        _ = FFmpegNative.av_strerror(code, buffer, Size); // FFmpeg writes a message even for an unknown code.
        return Marshal.PtrToStringUTF8((IntPtr)buffer) ?? code.ToString(System.Globalization.CultureInfo.InvariantCulture);
    }

    internal static string PixelFormatName(int format)
    {
        byte* name = FFmpegNative.av_get_pix_fmt_name(format);
        return name == null ? "unknown" : FFmpegNative.Utf8(name);
    }

    /// <summary>An opened container (file or stream) with its best video stream; can be rewound for a second pass.</summary>
    private sealed class Input : IDisposable
    {
        private readonly string? _path;
        private readonly Stream? _stream;
        private AVFormatContext* _format;
        private AVIOContext* _io;
        private GCHandle _self;

        private Input(string? path, Stream? stream)
        {
            _path = path;
            _stream = stream;
            Name = path ?? "stream";
        }

        public string Name { get; }

        public AVFormatContext* Format => _format;

        public int StreamIndex { get; private set; }

        public static Input OpenFile(string path)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException("Video file not found.", path);
            }

            var input = new Input(path, null);
            input.Open();
            return input;
        }

        public static Input OpenStream(Stream stream)
        {
            ArgumentNullException.ThrowIfNull(stream);
            if (!stream.CanRead || !stream.CanSeek)
            {
                throw new ArgumentException("The stream must be readable and seekable.", nameof(stream));
            }

            var input = new Input(null, stream);
            input.Open();
            return input;
        }

        /// <summary>
        /// torchcodec after its scan: <c>avformat_seek_file(ctx, 0, INT64_MIN, 0, 0, 0)</c> on the same demuxer, which then
        /// decodes. A container that cannot seek fails here, as it does in torchcodec.
        /// </summary>
        public void SeekToStartAfterScan()
        {
            int r = FFmpegNative.avformat_seek_file(_format, 0, long.MinValue, 0, 0, 0);
            if (r < 0)
            {
                throw new EmbeddingGemma2VideoDecodeException($"{Name}: seeking back to the start after the packet scan failed: {Error(r)} (torchcodec makes the same seek and fails the same way).");
            }
        }

        private long _streamStart = -1;

        private void Open()
        {
            AVFormatContext* fmt = null;
            try
            {
                if (_stream is not null)
                {
                    if (_streamStart < 0)
                    {
                        _streamStart = _stream.Position;
                    }

                    fmt = FFmpegNative.avformat_alloc_context();
                    _self = GCHandle.Alloc(this);
                    byte* buffer = (byte*)FFmpegNative.av_malloc(IoBufferSize);
                    _io = FFmpegNative.avio_alloc_context(buffer, IoBufferSize, 0, (void*)GCHandle.ToIntPtr(_self), &ReadPacket, null, &SeekPacket);
                    fmt->pb = _io;
                    fmt->flags |= FFmpegNative.AVFMT_FLAG_CUSTOM_IO;
                }

                int r = FFmpegNative.avformat_open_input(&fmt, _path, null, null);
                if (r < 0)
                {
                    // avformat_open_input frees the context on failure.
                    fmt = null;
                    throw new EmbeddingGemma2VideoDecodeException($"{Name}: not a readable video container: {Error(r)}.");
                }

                _format = fmt;
                Check(FFmpegNative.avformat_find_stream_info(_format, null), Name, "avformat_find_stream_info");
                int si = FFmpegNative.av_find_best_stream(_format, FFmpegNative.AVMEDIA_TYPE_VIDEO, -1, -1, null, 0);
                if (si < 0)
                {
                    throw new EmbeddingGemma2VideoDecodeException($"{Name}: no video stream ({Error(si)}).");
                }

                StreamIndex = si;
            }
            catch
            {
                if (fmt != null && _format == null)
                {
                    FFmpegNative.avformat_free_context(fmt);
                }

                Close();
                throw;
            }
        }

        [UnmanagedCallersOnly]
        private static int ReadPacket(void* opaque, byte* buffer, int size) =>
            ((Input)GCHandle.FromIntPtr((IntPtr)opaque).Target!).Read(buffer, size);

        [UnmanagedCallersOnly]
        private static long SeekPacket(void* opaque, long offset, int whence) =>
            ((Input)GCHandle.FromIntPtr((IntPtr)opaque).Target!).Seek(offset, whence);

        private int Read(byte* buffer, int size)
        {
            try
            {
                int n = _stream!.Read(new Span<byte>(buffer, size));
                return n == 0 ? FFmpegNative.AVERROR_EOF : n;
            }
#pragma warning disable CA1031 // Errors cannot cross the native boundary; FFmpeg receives an error code.
            catch
#pragma warning restore CA1031
            {
                return FFmpegNative.AVERROR_EIO;
            }
        }

        private long Seek(long offset, int whence)
        {
            try
            {
                var s = _stream!;
                long length = s.Length - _streamStart;
                if ((whence & FFmpegNative.AVSEEK_SIZE) != 0)
                {
                    return length;
                }

                long target = (whence & 3) switch
                {
                    0 => offset,
                    1 => s.Position - _streamStart + offset,
                    2 => length + offset,
                    _ => -1,
                };
                if (target < 0)
                {
                    return FFmpegNative.AVERROR_EINVAL;
                }

                s.Position = _streamStart + target;
                return target;
            }
#pragma warning disable CA1031 // Errors cannot cross the native boundary; FFmpeg receives an error code.
            catch
#pragma warning restore CA1031
            {
                return FFmpegNative.AVERROR_EIO;
            }
        }

        private void Close()
        {
            if (_format != null)
            {
                var f = _format;
                FFmpegNative.avformat_close_input(&f);
                _format = null;
            }

            if (_io != null)
            {
                FFmpegNative.av_freep(&_io->buffer);
                var io = _io;
                FFmpegNative.avio_context_free(&io);
                _io = null;
            }

            if (_self.IsAllocated)
            {
                _self.Free();
            }
        }

        public void Dispose() => Close();
    }

    /// <summary>torchcodec's CPU RGB24 conversion: swscale or a filter graph, rebuilt when the frame format or size changes.</summary>
    private sealed class Converter(int quarterTurns, long maxPixels, string name) : IDisposable
    {
        private SwsContext* _sws;
        private (int W, int H, int Format, int Space, int Range) _swsKey;
        private AVFilterGraph* _graph;
        private AVFilterContext* _source;
        private AVFilterContext* _sink;
        private (int W, int H, int Format) _graphKey;

        public EmbeddingGemma2Image Convert(AVFrame* frame, AVRational timeBase)
        {
            if ((long)frame->width * frame->height > maxPixels)
            {
                throw new EmbeddingGemma2ImageTooLargeException(frame->width, frame->height, maxPixels);
            }

            return quarterTurns == 0 && frame->width % 32 == 0 ? Swscale(frame) : FilterGraph(frame, timeBase);
        }

        private EmbeddingGemma2Image Swscale(AVFrame* f)
        {
            int w = f->width, h = f->height;
            var key = (w, h, f->format, f->colorspace, f->color_range);
            if (_sws == null || key != _swsKey)
            {
                if (_sws != null)
                {
                    FFmpegNative.sws_freeContext(_sws);
                }

                _sws = FFmpegNative.sws_getContext(w, h, f->format, w, h, FFmpegNative.AV_PIX_FMT_RGB24, 0, null, null, null);
                if (_sws == null)
                {
                    throw new EmbeddingGemma2VideoDecodeException($"{name}: sws_getContext failed for {PixelFormatName(f->format)} {w}x{h}.");
                }

                int* invTable;
                int* table;
                int srcRange, dstRange, brightness, contrast, saturation;
                Check(FFmpegNative.sws_getColorspaceDetails(_sws, &invTable, &srcRange, &table, &dstRange, &brightness, &contrast, &saturation), name, "sws_getColorspaceDetails");
                if (f->color_range != FFmpegNative.AVCOL_RANGE_UNSPECIFIED)
                {
                    srcRange = f->color_range == FFmpegNative.AVCOL_RANGE_JPEG ? 1 : 0;
                }

                int* coefficients = FFmpegNative.sws_getCoefficients(f->colorspace);
                Check(FFmpegNative.sws_setColorspaceDetails(_sws, coefficients, srcRange, coefficients, dstRange, brightness, contrast, saturation), name, "sws_setColorspaceDetails");
                _swsKey = key;
            }

            var rgb = new byte[checked(w * h * 3)];
            fixed (byte* p = rgb)
            {
                byte** dst = stackalloc byte*[4];
                int* dstStride = stackalloc int[4];
                dst[0] = p;
                dst[1] = dst[2] = dst[3] = null;
                dstStride[0] = w * 3;
                dstStride[1] = dstStride[2] = dstStride[3] = 0;
                int rows = FFmpegNative.sws_scale(_sws, (byte**)f->data, f->linesize, 0, h, dst, dstStride);
                if (rows != h)
                {
                    throw new EmbeddingGemma2VideoDecodeException($"{name}: sws_scale converted {rows} of {h} rows.");
                }
            }

            return EmbeddingGemma2Image.FromRgb24(w, h, rgb);
        }

        private EmbeddingGemma2Image FilterGraph(AVFrame* f, AVRational timeBase)
        {
            var key = (f->width, f->height, f->format);
            if (_graph == null || key != _graphKey)
            {
                FreeGraph();
                BuildGraph(f, timeBase);
                _graphKey = key;
            }

            Check(FFmpegNative.av_buffersrc_write_frame(_source, f), name, "av_buffersrc_write_frame");
            AVFrame* o = FFmpegNative.av_frame_alloc();
            try
            {
                Check(FFmpegNative.av_buffersink_get_frame(_sink, o), name, "av_buffersink_get_frame");
                if (o->format != FFmpegNative.AV_PIX_FMT_RGB24)
                {
                    throw new EmbeddingGemma2VideoDecodeException($"{name}: the filter graph produced {PixelFormatName(o->format)}, not rgb24.");
                }

                int w = o->width, h = o->height;
                var rgb = new byte[checked(w * h * 3)];
                for (int y = 0; y < h; y++)
                {
                    new ReadOnlySpan<byte>(((byte**)o->data)[0] + ((long)y * o->linesize[0]), w * 3).CopyTo(rgb.AsSpan(y * w * 3, w * 3));
                }

                return EmbeddingGemma2Image.FromRgb24(w, h, rgb);
            }
            finally
            {
                FFmpegNative.av_frame_free(&o);
            }
        }

        private void BuildGraph(AVFrame* f, AVRational timeBase)
        {
            _graph = FFmpegNative.avfilter_graph_alloc();
            var parameters = FFmpegNative.av_buffersrc_parameters_alloc();
            parameters->format = f->format;
            parameters->width = f->width;
            parameters->height = f->height;
            parameters->sample_aspect_ratio = f->sample_aspect_ratio;
            parameters->time_base = timeBase;
            _source = FFmpegNative.avfilter_graph_alloc_filter(_graph, FFmpegNative.avfilter_get_by_name("buffer"), "in");
            int r = FFmpegNative.av_buffersrc_parameters_set(_source, parameters);
            FFmpegNative.av_free(parameters);
            Check(r, name, "av_buffersrc_parameters_set");
            Check(FFmpegNative.avfilter_init_str(_source, null), name, "buffer init");
            _sink = FFmpegNative.avfilter_graph_alloc_filter(_graph, FFmpegNative.avfilter_get_by_name("buffersink"), "out");
            Check(FFmpegNative.av_opt_set(_sink, "pixel_formats", "rgb24", FFmpegNative.AV_OPT_SEARCH_CHILDREN), name, "buffersink pixel_formats");
            Check(FFmpegNative.avfilter_init_str(_sink, null), name, "buffersink init");
            var outputs = FFmpegNative.avfilter_inout_alloc();
            outputs->name = FFmpegNative.av_strdup("in");
            outputs->filter_ctx = _source;
            outputs->pad_idx = 0;
            outputs->next = null;
            var inputs = FFmpegNative.avfilter_inout_alloc();
            inputs->name = FFmpegNative.av_strdup("out");
            inputs->filter_ctx = _sink;
            inputs->pad_idx = 0;
            inputs->next = null;
            string description = quarterTurns switch
            {
                0 => "copy",
                1 => "format=rgb24,transpose=cclock",
                -1 => "format=rgb24,transpose=clock",
                _ => "format=rgb24,hflip,vflip",
            };
            try
            {
                Check(FFmpegNative.avfilter_graph_parse_ptr(_graph, description, &inputs, &outputs, null), name, "avfilter_graph_parse_ptr");
            }
            finally
            {
                FFmpegNative.avfilter_inout_free(&inputs);
                FFmpegNative.avfilter_inout_free(&outputs);
            }

            Check(FFmpegNative.avfilter_graph_config(_graph, null), name, "avfilter_graph_config");
        }

        private void FreeGraph()
        {
            if (_graph != null)
            {
                var g = _graph;
                FFmpegNative.avfilter_graph_free(&g);
                _graph = null;
                _source = null;
                _sink = null;
            }
        }

        public void Dispose()
        {
            if (_sws != null)
            {
                FFmpegNative.sws_freeContext(_sws);
                _sws = null;
            }

            FreeGraph();
        }
    }
}

/// <summary>A video's metadata as torchcodec computes it (exact seek mode: a packet scan without decoding).</summary>
/// <param name="FrameCount">Packets of the video stream (packets flagged discard skipped): torchcodec's <c>num_frames</c>.</param>
/// <param name="BeginSeconds">Smallest packet pts in seconds.</param>
/// <param name="EndSeconds">Largest packet pts + duration in seconds.</param>
/// <param name="Width">Frame width after the display rotation (as torchcodec reports it).</param>
/// <param name="Height">Frame height after the display rotation.</param>
/// <param name="Rotation">Display-matrix rotation in degrees (<c>av_display_rotation_get</c>), or null.</param>
/// <param name="Codec">Codec name (for example <c>h264</c>).</param>
/// <param name="PixelFormat">Stream pixel format (for example <c>yuv420p</c>).</param>
/// <param name="StreamIndex">Index of the best video stream in the container.</param>
public sealed record EmbeddingGemma2VideoMetadata(
    int FrameCount,
    double BeginSeconds,
    double EndSeconds,
    int Width,
    int Height,
    double? Rotation,
    string Codec,
    string PixelFormat,
    int StreamIndex)
{
    /// <summary>Duration in seconds: <see cref="EndSeconds"/> - <see cref="BeginSeconds"/>.</summary>
    public double DurationSeconds => EndSeconds - BeginSeconds;

    /// <summary>Average frame rate (<see cref="FrameCount"/> / <see cref="DurationSeconds"/>), or null for a zero duration.</summary>
    public double? AverageFramesPerSecond => DurationSeconds != 0 ? FrameCount / DurationSeconds : null;
}

/// <summary>A decoded video: the frames for the embedder, the metadata and the source frame index of each frame.</summary>
/// <param name="Video">The video to embed.</param>
/// <param name="Metadata">torchcodec's metadata.</param>
/// <param name="FrameIndices">Source frame index of each frame of <see cref="EmbeddingGemma2Video.Frames"/>.</param>
public sealed record EmbeddingGemma2DecodedVideo(EmbeddingGemma2Video Video, EmbeddingGemma2VideoMetadata Metadata, IReadOnlyList<int> FrameIndices);
