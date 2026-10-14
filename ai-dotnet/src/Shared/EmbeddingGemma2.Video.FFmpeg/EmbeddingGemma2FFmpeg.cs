// Shared source of the FFmpeg adapter packages, one per FFmpeg major (decision D3): compiled into
// Starlings.Models.EmbeddingGemma2.Video.FFmpeg (FFmpeg 8) and, with STARLINGS_FFMPEG9 defined,
// Starlings.Models.EmbeddingGemma2.Video.FFmpeg9 (FFmpeg 9). Each package has its own generated bindings (FFmpegNative.g.cs,
// tools/generate-ffmpeg-bindings.py --major 8|9); everything else is this code.
#if STARLINGS_FFMPEG9
namespace Starlings.Models.EmbeddingGemma2.Video.FFmpeg9;
#else
namespace Starlings.Models.EmbeddingGemma2.Video.FFmpeg;
#endif

/// <summary>
/// Loads the user's own FFmpeg shared libraries of the major this package binds (<see cref="SupportedFFmpegMajor"/>), once
/// per process, through this package's bindings and checks them: the library majors the bindings were generated for
/// (FFmpeg 8: libavcodec 62, libavformat 62, libavutil 60, libswscale 9, libavfilter 11; FFmpeg 9: libavcodec 63,
/// libavformat 63, libavutil 61, libswscale 10, libavfilter 12), and the build's license (LGPL expected; GPL and nonfree
/// builds are rejected unless allowed). This package ships no FFmpeg binaries.
/// </summary>
/// <remarks>
/// <para>
/// <b>Where the libraries come from.</b> <see cref="EmbeddingGemma2FFmpegOptions.LibraryDirectory"/>, else the directory in
/// <see cref="LibraryDirectoryEnvironmentVariable"/>, else the operating system's standard library search (on Linux
/// <c>dlopen("libavcodec.so.62")</c> for FFmpeg 8, <c>libavcodec.so.63</c> for FFmpeg 9: <c>LD_LIBRARY_PATH</c>, the loader
/// cache, <c>/usr/lib</c>). The runtime keeps the libraries it loaded for the process, so every decoder in a process must
/// use the same directory.
/// </para>
/// <para>
/// <b>Versions.</b> One FFmpeg major per package: the ABI the bindings were generated for, from that major's headers by
/// <c>tools/generate-ffmpeg-bindings.py</c> (<c>Starlings.Models.EmbeddingGemma2.Video.FFmpeg</c>: FFmpeg 8.x;
/// <c>Starlings.Models.EmbeddingGemma2.Video.FFmpeg9</c>: FFmpeg 9.x). Another major has other library majors and is
/// rejected with <see cref="EmbeddingGemma2FFmpegNotFoundException"/> or <see cref="EmbeddingGemma2FFmpegVersionException"/>;
/// use the package for that major.
/// </para>
/// </remarks>
public static unsafe class EmbeddingGemma2FFmpeg
{
#if STARLINGS_FFMPEG9
    /// <summary>Environment variable naming the directory of the FFmpeg 9 shared libraries (used when the options give none).</summary>
    public const string LibraryDirectoryEnvironmentVariable = "STARLING_FFMPEG9_DIR";

    /// <summary>The FFmpeg major release this adapter binds.</summary>
    public const int SupportedFFmpegMajor = 9;
#else
    /// <summary>Environment variable naming the directory of the FFmpeg shared libraries (used when the options give none).</summary>
    public const string LibraryDirectoryEnvironmentVariable = "STARLING_FFMPEG_DIR";

    /// <summary>The FFmpeg major release this adapter binds.</summary>
    public const int SupportedFFmpegMajor = 8;
#endif

    private static readonly (string Name, int Major)[] Libraries =
    [
        ("avutil", FFmpegNative.AVUTIL_MAJOR),
        ("avcodec", FFmpegNative.AVCODEC_MAJOR),
        ("avformat", FFmpegNative.AVFORMAT_MAJOR),
        ("swscale", FFmpegNative.SWSCALE_MAJOR),
        ("avfilter", FFmpegNative.AVFILTER_MAJOR),
    ];

    private static readonly Lock Gate = new();
    private static EmbeddingGemma2FFmpegLibraries? _loaded;
    private static bool _bound;
    private static string? _boundDirectory;

    /// <summary>The loaded libraries, or null before the first successful <see cref="Load"/>.</summary>
    public static EmbeddingGemma2FFmpegLibraries? Loaded
    {
        get
        {
            lock (Gate)
            {
                return _loaded;
            }
        }
    }

    /// <summary>
    /// Loads and checks the FFmpeg libraries (idempotent for the same directory and license choice).
    /// </summary>
    /// <param name="options">Where to load from and whether a non-LGPL build is accepted. Null: defaults.</param>
    /// <returns>What was loaded: version, library versions, licenses, configuration.</returns>
    /// <exception cref="EmbeddingGemma2FFmpegNotFoundException">The libraries of the bound FFmpeg major are not found.</exception>
    /// <exception cref="EmbeddingGemma2FFmpegVersionException">A library has another major than this adapter binds.</exception>
    /// <exception cref="EmbeddingGemma2FFmpegLicenseException">The build is GPL or nonfree and that is not allowed.</exception>
    /// <exception cref="InvalidOperationException">Libraries from another directory are already loaded in this process.</exception>
    public static EmbeddingGemma2FFmpegLibraries Load(EmbeddingGemma2FFmpegOptions? options = null)
    {
        options ??= new EmbeddingGemma2FFmpegOptions();
        string? directory = ResolveDirectory(options.LibraryDirectory);
        if (directory is not null && !System.IO.Directory.Exists(directory))
        {
            throw new EmbeddingGemma2FFmpegNotFoundException($"The FFmpeg library directory {directory} does not exist. Install FFmpeg {SupportedFFmpegMajor}.x shared libraries (an LGPL build) and point EmbeddingGemma2FFmpegOptions.LibraryDirectory or {LibraryDirectoryEnvironmentVariable} at the directory holding {FFmpegNative.FileName("avcodec", FFmpegNative.AVCODEC_MAJOR)}.");
        }

        lock (Gate)
        {
            if (_loaded is { } loaded)
            {
                if (!string.Equals(loaded.Directory, directory, StringComparison.Ordinal))
                {
                    throw new InvalidOperationException($"FFmpeg is already loaded from {Describe(loaded.Directory)} in this process; the runtime keeps the libraries it loaded for the process, so {Describe(directory)} cannot be loaded as well.");
                }

                CheckLicense(loaded, options.AllowNonLgplBuild);
                return loaded;
            }

            if (_bound && !string.Equals(_boundDirectory, directory, StringComparison.Ordinal))
            {
                throw new InvalidOperationException($"FFmpeg libraries were already loaded from {Describe(_boundDirectory)} in this process (the load failed its checks); the runtime keeps loaded libraries for the process, so {Describe(directory)} needs a new process.");
            }

            if (!FFmpegNative.Configure(directory))
            {
                throw new InvalidOperationException($"FFmpeg libraries were already configured for another directory in this process; the runtime keeps loaded libraries for the process, so {Describe(directory)} needs a new process.");
            }

            var versions = new Dictionary<string, string>(StringComparer.Ordinal);
            var licenses = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (var (name, major) in Libraries)
            {
                uint version;
                string license;
                try
                {
                    (version, license) = name switch
                    {
                        "avutil" => (FFmpegNative.avutil_version(), FFmpegNative.Utf8(FFmpegNative.avutil_license())),
                        "avcodec" => (FFmpegNative.avcodec_version(), FFmpegNative.Utf8(FFmpegNative.avcodec_license())),
                        "avformat" => (FFmpegNative.avformat_version(), FFmpegNative.Utf8(FFmpegNative.avformat_license())),
                        "swscale" => (FFmpegNative.swscale_version(), FFmpegNative.Utf8(FFmpegNative.swscale_license())),
                        _ => (FFmpegNative.avfilter_version(), FFmpegNative.Utf8(FFmpegNative.avfilter_license())),
                    };
                }
                catch (Exception ex) when (ex is NotSupportedException or DllNotFoundException or EntryPointNotFoundException or BadImageFormatException)
                {
                    throw new EmbeddingGemma2FFmpegNotFoundException(NotFoundMessage(directory, name, major), ex);
                }

                _bound = true;
                _boundDirectory = directory;
                int found = (int)(version >> 16);
                versions[name] = $"{found}.{(version >> 8) & 0xFF}.{version & 0xFF}";
                licenses[name] = license;
                if (found != major)
                {
                    throw new EmbeddingGemma2FFmpegVersionException($"lib{name} {versions[name]} was loaded from {Describe(directory)}; this adapter binds FFmpeg {SupportedFFmpegMajor}.x (lib{name} major {major}). Install FFmpeg {SupportedFFmpegMajor}.x shared libraries.");
                }
            }

            var libraries = new EmbeddingGemma2FFmpegLibraries(directory, FFmpegNative.Utf8(FFmpegNative.av_version_info()), versions, licenses, FFmpegNative.Utf8(FFmpegNative.avcodec_configuration()));
            FFmpegNative.av_log_set_level(FFmpegNative.AV_LOG_ERROR);

            // Recorded before the license check: the libraries stay loaded in the process either way.
            _loaded = libraries;
            CheckLicense(libraries, options.AllowNonLgplBuild);
            return libraries;
        }
    }

    private static void CheckLicense(EmbeddingGemma2FFmpegLibraries libraries, bool allowNonLgpl)
    {
        if (!libraries.IsLgpl && !allowNonLgpl)
        {
            string kinds = string.Join(", ", libraries.Licenses.Select(kv => $"lib{kv.Key}: {kv.Value}"));
            throw new EmbeddingGemma2FFmpegLicenseException($"The FFmpeg build at {Describe(libraries.Directory)} is not LGPL ({kinds}). A GPL build makes the program that loads it subject to the GPL, and a nonfree build is not redistributable. Use an LGPL build (configured without --enable-gpl and --enable-nonfree), or set EmbeddingGemma2FFmpegOptions.AllowNonLgplBuild if that is your deliberate choice.");
        }
    }

    private static string? ResolveDirectory(string? configured)
    {
        string? directory = configured is { Length: > 0 } ? configured : Environment.GetEnvironmentVariable(LibraryDirectoryEnvironmentVariable);
        return directory is { Length: > 0 } ? Path.GetFullPath(directory) : null;
    }

    private static string Describe(string? directory) => directory ?? "the system library search path";

    internal static string NotFoundMessage(string? directory, string name, int major) =>
        NotFoundMessage(directory, name, major, FFmpegNative.CurrentNaming);

    /// <summary>The not-found message under a naming convention (testable for every operating system on any).</summary>
    internal static string NotFoundMessage(string? directory, string name, int major, LibraryNaming naming)
    {
        string expected = FFmpegNative.FileName(name, major, naming);
        string hint = string.Empty;
        string[] dirs = directory is not null ? [directory] : DefaultSearchDirectories(naming);
        var other = new List<string>();
        foreach (string dir in dirs.Where(System.IO.Directory.Exists))
        {
            try
            {
                other.AddRange(System.IO.Directory.GetFiles(dir, FFmpegNative.FilePattern(name, naming))
                    .Select(f => Path.GetFileName(f)!)
                    .Where(f => FFmpegNative.MajorOf(f, name, naming) is int found && found != major));
            }
            catch (UnauthorizedAccessException)
            {
            }
        }

        if (other.Count > 0)
        {
            hint = $" Found {string.Join(", ", other.Distinct().Order(StringComparer.Ordinal))} instead (another FFmpeg major; this package binds FFmpeg {SupportedFFmpegMajor}, {expected}; the adapter package of the installed major loads it: Starlings.Models.EmbeddingGemma2.Video.FFmpeg for FFmpeg 8, Starlings.Models.EmbeddingGemma2.Video.FFmpeg9 for FFmpeg 9).";
        }

        return $"FFmpeg {SupportedFFmpegMajor}.x shared library {expected} could not be loaded from {Describe(directory)}.{hint} Install FFmpeg {SupportedFFmpegMajor}.x shared libraries (an LGPL build; this package ships none) and set EmbeddingGemma2FFmpegOptions.LibraryDirectory or {LibraryDirectoryEnvironmentVariable}.";
    }

    /// <summary>Where the system search usually finds FFmpeg, scanned only for the other-major hint.</summary>
    private static string[] DefaultSearchDirectories(LibraryNaming naming) => naming switch
    {
        LibraryNaming.Windows => [AppContext.BaseDirectory],
        LibraryNaming.MacOS => ["/opt/homebrew/lib", "/usr/local/lib"],
        _ => ["/usr/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/local/lib"],
    };
}

/// <summary>The FFmpeg libraries a process loaded.</summary>
/// <param name="Directory">The directory loaded from, or null for the system library search path.</param>
/// <param name="Version">FFmpeg's version string (<c>av_version_info</c>), for example <c>n8.1.3</c>.</param>
/// <param name="LibraryVersions">Version per library (<c>avcodec</c>, <c>avformat</c>, <c>avutil</c>, <c>swscale</c>, <c>avfilter</c>).</param>
/// <param name="Licenses">License string per library (<c>avcodec_license()</c> and the others).</param>
/// <param name="Configuration">The build's configure line (<c>avcodec_configuration()</c>).</param>
public sealed record EmbeddingGemma2FFmpegLibraries(
    string? Directory,
    string Version,
    IReadOnlyDictionary<string, string> LibraryVersions,
    IReadOnlyDictionary<string, string> Licenses,
    string Configuration)
{
    /// <summary>True when every library reports an LGPL license (no <c>--enable-gpl</c>, no <c>--enable-nonfree</c>).</summary>
    public bool IsLgpl => Licenses.Values.All(l => l.StartsWith("LGPL", StringComparison.Ordinal));
}

/// <summary>Configuration of the FFmpeg adapter.</summary>
public sealed record EmbeddingGemma2FFmpegOptions
{
    /// <summary>
    /// Directory of the FFmpeg shared libraries of the major this package binds (<see cref="EmbeddingGemma2FFmpeg.SupportedFFmpegMajor"/>). Null (default): <see cref="EmbeddingGemma2FFmpeg.LibraryDirectoryEnvironmentVariable"/>,
    /// else the operating system's standard library search.
    /// </summary>
    public string? LibraryDirectory { get; init; }

    /// <summary>
    /// Accept a GPL or nonfree FFmpeg build. Default <see langword="false"/>: such builds throw
    /// <see cref="EmbeddingGemma2FFmpegLicenseException"/>, because loading a GPL FFmpeg makes the program subject to the
    /// GPL and a nonfree build is not redistributable. Decoding is identical either way.
    /// </summary>
    public bool AllowNonLgplBuild { get; init; }

    /// <summary>Decoded pixels per frame at most (memory bound; upstream has none). Default 50,000,000.</summary>
    public long MaxFramePixels { get; init; } = 50_000_000;

    /// <summary>
    /// Frames at most for <see cref="EmbeddingGemma2VideoDecoder.DecodeAllFrames(string, CancellationToken)"/>, which keeps
    /// every frame in memory. Default 3,600 (2 minutes at 30 fps). Sampled decoding keeps at most 32 frames and has no
    /// such limit.
    /// </summary>
    public int MaxAllFrames { get; init; } = 3_600;

    /// <summary>Decoder threads; 0 (default) lets FFmpeg choose, as torchcodec (<c>num_ffmpeg_threads=0</c>) does.</summary>
    public int DecoderThreads { get; init; }

    internal void Validate()
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxFramePixels, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxAllFrames, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(DecoderThreads);
    }
}

/// <summary>Base of the FFmpeg adapter's errors about the FFmpeg libraries.</summary>
public class EmbeddingGemma2FFmpegException : InvalidOperationException
{
    /// <summary>Creates the exception.</summary>
    public EmbeddingGemma2FFmpegException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public EmbeddingGemma2FFmpegException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public EmbeddingGemma2FFmpegException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}

/// <summary>The FFmpeg shared libraries of the bound major could not be loaded (not installed, wrong directory, or another major only).</summary>
public sealed class EmbeddingGemma2FFmpegNotFoundException : EmbeddingGemma2FFmpegException
{
    /// <summary>Creates the exception.</summary>
    public EmbeddingGemma2FFmpegNotFoundException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public EmbeddingGemma2FFmpegNotFoundException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public EmbeddingGemma2FFmpegNotFoundException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}

/// <summary>A loaded FFmpeg library has another major version than this adapter binds (<see cref="EmbeddingGemma2FFmpeg.SupportedFFmpegMajor"/>).</summary>
public sealed class EmbeddingGemma2FFmpegVersionException : EmbeddingGemma2FFmpegException
{
    /// <summary>Creates the exception.</summary>
    public EmbeddingGemma2FFmpegVersionException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public EmbeddingGemma2FFmpegVersionException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public EmbeddingGemma2FFmpegVersionException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}

/// <summary>The FFmpeg build is GPL or nonfree and <see cref="EmbeddingGemma2FFmpegOptions.AllowNonLgplBuild"/> is not set.</summary>
public sealed class EmbeddingGemma2FFmpegLicenseException : EmbeddingGemma2FFmpegException
{
    /// <summary>Creates the exception.</summary>
    public EmbeddingGemma2FFmpegLicenseException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public EmbeddingGemma2FFmpegLicenseException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public EmbeddingGemma2FFmpegLicenseException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}

/// <summary>A video could not be decoded: it does not open, has no video stream, fails to decode, or needs an unsupported transform.</summary>
public sealed class EmbeddingGemma2VideoDecodeException : Exception
{
    /// <summary>Creates the exception.</summary>
    public EmbeddingGemma2VideoDecodeException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public EmbeddingGemma2VideoDecodeException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and the underlying failure.</summary>
    public EmbeddingGemma2VideoDecodeException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
