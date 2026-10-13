using System.Runtime.InteropServices;
using FFmpeg.AutoGen;

namespace Starlings.Models.EmbeddingGemma2.Video.FFmpeg;

/// <summary>
/// Decodes video files and streams into <see cref="EmbeddingGemma2Video"/> exactly as upstream sees them (torchcodec
/// 0.17.0 on CPU, uint8 RGB, <c>seek_mode="exact"</c>), on the user's own FFmpeg 8.x shared libraries
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
/// (<see cref="EmbeddingGemma2VideoSampling.SampleIndices"/> on that metadata; 1 frame per second, at most 32), then one
/// sequential decode in presentation order that converts and keeps only the sampled frames (at most 32 in memory).
/// </para>
/// <para>
/// <b>RGB conversion</b>, as torchcodec calls FFmpeg: when the frame width is a multiple of 32 and no rotation applies,
/// one <c>sws_scale</c> to RGB24 with flags 0 (FFmpeg 8.1: bicubic), the source and destination coefficients
/// <c>sws_getCoefficients(frame colorspace)</c> (BT.601 when untagged) and the source range from the frame when tagged;
/// otherwise a filter graph <c>buffer</c> -&gt; <c>copy</c> (or <c>format=rgb24,transpose=cclock</c>,
/// <c>format=rgb24,transpose=clock</c>, <c>format=rgb24,hflip,vflip</c> for a display matrix of 90, -90 or 180 degrees)
/// -&gt; <c>buffersink</c> RGB24. This reproduces torchcodec's decoded RGB bit for bit on the 22 fixture videos (H.264
/// 8- and 10-bit, HEVC, VP9, AV1; rotation 90; odd sizes) with FFmpeg 8.1 on x86-64. Other CPUs (swscale's SIMD paths)
/// and FFmpeg builds are not covered by that evidence.
/// </para>
/// <para>Instances are immutable and thread-safe; each call opens its own FFmpeg contexts.</para>
/// </remarks>
public sealed unsafe class EmbeddingGemma2VideoDecoder
{
    private const int IoBufferSize = 64 * 1024;
    private const int Eio = 5; // POSIX EIO (FFmpeg.AutoGen does not define it).

    private readonly EmbeddingGemma2FFmpegOptions _options;

    /// <summary>Loads and checks the FFmpeg libraries (<see cref="EmbeddingGemma2FFmpeg.Load"/>).</summary>
    /// <exception cref="EmbeddingGemma2FFmpegException">FFmpeg 8.x is missing, has another major, or is a GPL or nonfree build that is not allowed.</exception>
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
        var metadata = Scan(input, cancellationToken);
        if (metadata.FrameCount > _options.MaxAllFrames)
        {
            throw new EmbeddingGemma2VideoDecodeException($"{path} has {metadata.FrameCount} frames; DecodeAllFrames keeps at most {_options.MaxAllFrames} (EmbeddingGemma2FFmpegOptions.MaxAllFrames). Use DecodeFile, which keeps only the sampled frames.");
        }

        int[] all = Enumerable.Range(0, metadata.FrameCount).ToArray();
        input.Rewind();
        var frames = DecodeFrames(input, metadata, all, cancellationToken);
        var video = metadata.AverageFramesPerSecond is { } fps
            ? EmbeddingGemma2Video.FromDecodedFrames(frames, fps, metadata.DurationSeconds)
            : EmbeddingGemma2Video.FromFramesWithoutTiming(frames);
        return new EmbeddingGemma2DecodedVideo(video, metadata, all);
    }

    /// <summary>
    /// The frames at <paramref name="indices"/> (presentation order indices, as torchcodec's <c>get_frames_at</c>), in the
    /// order given; repeats allowed. Decodes sequentially up to the largest index and keeps only the requested frames.
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
        var metadata = Scan(input, cancellationToken);
        foreach (int i in indices)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(i, nameof(indices));
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(i, metadata.FrameCount, nameof(indices));
        }

        input.Rewind();
        return DecodeFrames(input, metadata, [.. indices], cancellationToken);
    }

    private EmbeddingGemma2DecodedVideo Decode(Func<Input> open, CancellationToken cancellationToken)
    {
        using var input = open();
        var metadata = Scan(input, cancellationToken);

        // Upstream: sample from the scanned metadata (average_fps is None when the duration is 0: every frame, capped).
        int[] indices = EmbeddingGemma2VideoSampling.SampleIndices(metadata.FrameCount, metadata.AverageFramesPerSecond, metadata.AverageFramesPerSecond is null ? null : metadata.DurationSeconds);
        input.Rewind();
        var frames = DecodeFrames(input, metadata, indices, cancellationToken);
        return new EmbeddingGemma2DecodedVideo(EmbeddingGemma2Video.FromSampledFrames(frames, indices), metadata, indices);
    }

    private static EmbeddingGemma2VideoMetadata Scan(Input input, CancellationToken cancellationToken)
    {
        var fmt = input.Format;
        int si = input.StreamIndex;
        var st = fmt->streams[si];
        long begin = long.MaxValue, end = long.MinValue;
        int n = 0;
        var pkt = ffmpeg.av_packet_alloc();
        try
        {
            int r;
            while ((r = ffmpeg.av_read_frame(fmt, pkt)) >= 0)
            {
                if (pkt->stream_index == si && (pkt->flags & ffmpeg.AV_PKT_FLAG_DISCARD) == 0)
                {
                    long pts = pkt->pts != ffmpeg.AV_NOPTS_VALUE ? pkt->pts : pkt->dts;
                    begin = Math.Min(begin, pts);
                    end = Math.Max(end, pts + pkt->duration);
                    n++;
                    if ((n & 1023) == 0)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                    }
                }

                ffmpeg.av_packet_unref(pkt);
            }

            if (r != ffmpeg.AVERROR_EOF)
            {
                throw new EmbeddingGemma2VideoDecodeException($"{input.Name}: reading packets failed: {Error(r)}.");
            }
        }
        finally
        {
            ffmpeg.av_packet_free(&pkt);
        }

        if (n == 0)
        {
            throw new EmbeddingGemma2VideoDecodeException($"{input.Name}: the video stream has no packets.");
        }

        var tb = st->time_base;
        double Seconds(long pts) => (double)pts * tb.num / tb.den;
        double? rotation = null;
        var sd = ffmpeg.av_packet_side_data_get(st->codecpar->coded_side_data, st->codecpar->nb_coded_side_data, AVPacketSideDataType.AV_PKT_DATA_DISPLAYMATRIX);
        if (sd != null && sd->size >= 9 * sizeof(int))
        {
            var matrix = new int_array9();
            for (uint k = 0; k < 9; k++)
            {
                matrix[k] = ((int*)sd->data)[k];
            }

            double degrees = ffmpeg.av_display_rotation_get(matrix);
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
            ffmpeg.avcodec_get_name(st->codecpar->codec_id),
            ffmpeg.av_get_pix_fmt_name((AVPixelFormat)st->codecpar->format) ?? "unknown",
            si);
    }

    private List<EmbeddingGemma2Image> DecodeFrames(Input input, EmbeddingGemma2VideoMetadata metadata, int[] indices, CancellationToken cancellationToken)
    {
        int quarterTurns = QuarterTurns(metadata.Rotation);
        var fmt = input.Format;
        int si = input.StreamIndex;
        var st = fmt->streams[si];
        var wanted = new Dictionary<int, List<int>>();
        for (int k = 0; k < indices.Length; k++)
        {
            if (!wanted.TryGetValue(indices[k], out var slots))
            {
                wanted[indices[k]] = slots = [];
            }

            slots.Add(k);
        }

        int last = indices.Max();
        var output = new EmbeddingGemma2Image?[indices.Length];
        var codec = ffmpeg.avcodec_find_decoder(st->codecpar->codec_id);
        if (codec == null)
        {
            throw new EmbeddingGemma2VideoDecodeException($"{input.Name}: the FFmpeg build has no decoder for {metadata.Codec}.");
        }

        AVCodecContext* cc = ffmpeg.avcodec_alloc_context3(codec);
        AVPacket* pkt = ffmpeg.av_packet_alloc();
        AVFrame* frame = ffmpeg.av_frame_alloc();
        using var converter = new Converter(quarterTurns, _options.MaxFramePixels, input.Name);
        try
        {
            Check(ffmpeg.avcodec_parameters_to_context(cc, st->codecpar), input.Name, "avcodec_parameters_to_context");
            cc->thread_count = _options.DecoderThreads;
            cc->pkt_timebase = st->time_base;
            Check(ffmpeg.avcodec_open2(cc, codec, null), input.Name, "avcodec_open2");
            int index = 0;
            bool flushing = false;
            while (index <= last)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!flushing)
                {
                    int r = ffmpeg.av_read_frame(fmt, pkt);
                    if (r < 0)
                    {
                        if (r != ffmpeg.AVERROR_EOF)
                        {
                            throw new EmbeddingGemma2VideoDecodeException($"{input.Name}: reading packets failed: {Error(r)}.");
                        }

                        flushing = true;
                        Check(ffmpeg.avcodec_send_packet(cc, null), input.Name, "avcodec_send_packet (flush)");
                    }
                    else
                    {
                        try
                        {
                            if (pkt->stream_index == si)
                            {
                                Check(ffmpeg.avcodec_send_packet(cc, pkt), input.Name, "avcodec_send_packet");
                            }
                        }
                        finally
                        {
                            ffmpeg.av_packet_unref(pkt);
                        }
                    }
                }

                while (index <= last)
                {
                    int r = ffmpeg.avcodec_receive_frame(cc, frame);
                    if (r == ffmpeg.AVERROR(ffmpeg.EAGAIN))
                    {
                        break;
                    }

                    if (r == ffmpeg.AVERROR_EOF)
                    {
                        throw new EmbeddingGemma2VideoDecodeException($"{input.Name}: the decoder produced {index} frames; frame {last} was requested (the packet scan counted {metadata.FrameCount}).");
                    }

                    Check(r, input.Name, "avcodec_receive_frame");
                    try
                    {
                        if (wanted.TryGetValue(index, out var slots))
                        {
                            var image = converter.Convert(frame, st->time_base);
                            foreach (int slot in slots)
                            {
                                output[slot] = image;
                            }
                        }
                    }
                    finally
                    {
                        ffmpeg.av_frame_unref(frame);
                    }

                    index++;
                }
            }
        }
        finally
        {
            ffmpeg.av_frame_free(&frame);
            ffmpeg.av_packet_free(&pkt);
            ffmpeg.avcodec_free_context(&cc);
        }

        return output.Select(f => f!).ToList();
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
        ffmpeg.av_strerror(code, buffer, Size);
        return Marshal.PtrToStringUTF8((IntPtr)buffer) ?? code.ToString(System.Globalization.CultureInfo.InvariantCulture);
    }

    /// <summary>An opened container (file or stream) with its best video stream; can be rewound for a second pass.</summary>
    private sealed class Input : IDisposable
    {
        private readonly string? _path;
        private readonly Stream? _stream;
        private AVFormatContext* _format;
        private AVIOContext* _io;
        private GCHandle _self;
        private avio_alloc_context_read_packet? _read;
        private avio_alloc_context_seek? _seek;

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

        /// <summary>Closes and reopens the container (torchcodec's decoder starts from a fresh demuxer after the scan).</summary>
        public void Rewind()
        {
            Close();
            if (_stream is not null)
            {
                _stream.Position = _streamStart;
            }

            Open();
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

                    fmt = ffmpeg.avformat_alloc_context();
                    _self = GCHandle.Alloc(this);
                    _read = Read;
                    _seek = Seek;
                    byte* buffer = (byte*)ffmpeg.av_malloc(IoBufferSize);
                    _io = ffmpeg.avio_alloc_context(buffer, IoBufferSize, 0, (void*)GCHandle.ToIntPtr(_self), _read, null, _seek);
                    fmt->pb = _io;
                    fmt->flags |= ffmpeg.AVFMT_FLAG_CUSTOM_IO;
                }

                int r = ffmpeg.avformat_open_input(&fmt, _path, null, null);
                if (r < 0)
                {
                    // avformat_open_input frees the context on failure.
                    fmt = null;
                    throw new EmbeddingGemma2VideoDecodeException($"{Name}: not a readable video container: {Error(r)}.");
                }

                _format = fmt;
                Check(ffmpeg.avformat_find_stream_info(_format, null), Name, "avformat_find_stream_info");
                int si = ffmpeg.av_find_best_stream(_format, AVMediaType.AVMEDIA_TYPE_VIDEO, -1, -1, null, 0);
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
                    ffmpeg.avformat_free_context(fmt);
                }

                Close();
                throw;
            }
        }

        private int Read(void* opaque, byte* buffer, int size)
        {
            try
            {
                int n = _stream!.Read(new Span<byte>(buffer, size));
                return n == 0 ? ffmpeg.AVERROR_EOF : n;
            }
#pragma warning disable CA1031 // Errors cannot cross the native boundary; FFmpeg receives an error code.
            catch
#pragma warning restore CA1031
            {
                return ffmpeg.AVERROR(Eio);
            }
        }

        private long Seek(void* opaque, long offset, int whence)
        {
            try
            {
                var s = _stream!;
                long length = s.Length - _streamStart;
                if ((whence & ffmpeg.AVSEEK_SIZE) != 0)
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
                    return ffmpeg.AVERROR(ffmpeg.EINVAL);
                }

                s.Position = _streamStart + target;
                return target;
            }
#pragma warning disable CA1031 // Errors cannot cross the native boundary; FFmpeg receives an error code.
            catch
#pragma warning restore CA1031
            {
                return ffmpeg.AVERROR(Eio);
            }
        }

        private void Close()
        {
            if (_format != null)
            {
                var f = _format;
                ffmpeg.avformat_close_input(&f);
                _format = null;
            }

            if (_io != null)
            {
                ffmpeg.av_freep(&_io->buffer);
                var io = _io;
                ffmpeg.avio_context_free(&io);
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
        private (int W, int H, int Format, AVColorSpace Space, AVColorRange Range) _swsKey;
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
                    ffmpeg.sws_freeContext(_sws);
                }

                _sws = ffmpeg.sws_getContext(w, h, (AVPixelFormat)f->format, w, h, AVPixelFormat.AV_PIX_FMT_RGB24, 0, null, null, null);
                if (_sws == null)
                {
                    throw new EmbeddingGemma2VideoDecodeException($"{name}: sws_getContext failed for {ffmpeg.av_get_pix_fmt_name((AVPixelFormat)f->format)} {w}x{h}.");
                }

                int* invTable;
                int* table;
                int srcRange, dstRange, brightness, contrast, saturation;
                Check(ffmpeg.sws_getColorspaceDetails(_sws, &invTable, &srcRange, &table, &dstRange, &brightness, &contrast, &saturation), name, "sws_getColorspaceDetails");
                if (f->color_range != AVColorRange.AVCOL_RANGE_UNSPECIFIED)
                {
                    srcRange = f->color_range == AVColorRange.AVCOL_RANGE_JPEG ? 1 : 0;
                }

                int* coefficients = ffmpeg.sws_getCoefficients((int)f->colorspace);
                var c4 = new int_array4();
                for (uint i = 0; i < 4; i++)
                {
                    c4[i] = coefficients[i];
                }

                Check(ffmpeg.sws_setColorspaceDetails(_sws, c4, srcRange, c4, dstRange, brightness, contrast, saturation), name, "sws_setColorspaceDetails");
                _swsKey = key;
            }

            var rgb = new byte[checked(w * h * 3)];
            fixed (byte* p = rgb)
            {
                var dst = new byte_ptrArray4 { [0] = p };
                var dstStride = new int_array4 { [0] = w * 3 };
                int rows = ffmpeg.sws_scale(_sws, f->data, f->linesize, 0, h, dst, dstStride);
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

            Check(ffmpeg.av_buffersrc_write_frame(_source, f), name, "av_buffersrc_write_frame");
            AVFrame* o = ffmpeg.av_frame_alloc();
            try
            {
                Check(ffmpeg.av_buffersink_get_frame(_sink, o), name, "av_buffersink_get_frame");
                if ((AVPixelFormat)o->format != AVPixelFormat.AV_PIX_FMT_RGB24)
                {
                    throw new EmbeddingGemma2VideoDecodeException($"{name}: the filter graph produced {ffmpeg.av_get_pix_fmt_name((AVPixelFormat)o->format)}, not rgb24.");
                }

                int w = o->width, h = o->height;
                var rgb = new byte[checked(w * h * 3)];
                for (int y = 0; y < h; y++)
                {
                    new ReadOnlySpan<byte>(o->data[0] + ((long)y * o->linesize[0]), w * 3).CopyTo(rgb.AsSpan(y * w * 3, w * 3));
                }

                return EmbeddingGemma2Image.FromRgb24(w, h, rgb);
            }
            finally
            {
                ffmpeg.av_frame_free(&o);
            }
        }

        private void BuildGraph(AVFrame* f, AVRational timeBase)
        {
            _graph = ffmpeg.avfilter_graph_alloc();
            var parameters = ffmpeg.av_buffersrc_parameters_alloc();
            parameters->format = f->format;
            parameters->width = f->width;
            parameters->height = f->height;
            parameters->sample_aspect_ratio = f->sample_aspect_ratio;
            parameters->time_base = timeBase;
            _source = ffmpeg.avfilter_graph_alloc_filter(_graph, ffmpeg.avfilter_get_by_name("buffer"), "in");
            int r = ffmpeg.av_buffersrc_parameters_set(_source, parameters);
            ffmpeg.av_free(parameters);
            Check(r, name, "av_buffersrc_parameters_set");
            Check(ffmpeg.avfilter_init_str(_source, null), name, "buffer init");
            _sink = ffmpeg.avfilter_graph_alloc_filter(_graph, ffmpeg.avfilter_get_by_name("buffersink"), "out");
            Check(ffmpeg.av_opt_set(_sink, "pixel_formats", "rgb24", ffmpeg.AV_OPT_SEARCH_CHILDREN), name, "buffersink pixel_formats");
            Check(ffmpeg.avfilter_init_str(_sink, null), name, "buffersink init");
            var outputs = ffmpeg.avfilter_inout_alloc();
            outputs->name = ffmpeg.av_strdup("in");
            outputs->filter_ctx = _source;
            outputs->pad_idx = 0;
            outputs->next = null;
            var inputs = ffmpeg.avfilter_inout_alloc();
            inputs->name = ffmpeg.av_strdup("out");
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
                Check(ffmpeg.avfilter_graph_parse_ptr(_graph, description, &inputs, &outputs, null), name, "avfilter_graph_parse_ptr");
            }
            finally
            {
                ffmpeg.avfilter_inout_free(&inputs);
                ffmpeg.avfilter_inout_free(&outputs);
            }

            Check(ffmpeg.avfilter_graph_config(_graph, null), name, "avfilter_graph_config");
        }

        private void FreeGraph()
        {
            if (_graph != null)
            {
                var g = _graph;
                ffmpeg.avfilter_graph_free(&g);
                _graph = null;
                _source = null;
                _sink = null;
            }
        }

        public void Dispose()
        {
            if (_sws != null)
            {
                ffmpeg.sws_freeContext(_sws);
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
