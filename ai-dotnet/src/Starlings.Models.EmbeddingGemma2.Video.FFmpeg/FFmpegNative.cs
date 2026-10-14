using System.Reflection;
using System.Runtime.InteropServices;

namespace Starlings.Models.EmbeddingGemma2.Video.FFmpeg;

/// <summary>
/// Where the bound FFmpeg libraries come from: the generated declarations (<c>FFmpegNative.g.cs</c>) name the libraries
/// <c>avutil</c>, <c>avcodec</c>, <c>avformat</c>, <c>swscale</c> and <c>avfilter</c>; this resolver maps them to the
/// user's shared libraries of the bound majors.
/// </summary>
internal static unsafe partial class FFmpegNative
{
    /// <summary>Load order: dependencies first, so each library's needed libraries are already loaded (no RUNPATH needed).</summary>
    private static readonly (string Name, int Major)[] LoadOrder =
    [
        ("avutil", AVUTIL_MAJOR),
        ("swresample", SWRESAMPLE_MAJOR),
        ("swscale", SWSCALE_MAJOR),
        ("avcodec", AVCODEC_MAJOR),
        ("avformat", AVFORMAT_MAJOR),
        ("avfilter", AVFILTER_MAJOR),
    ];

    private static readonly Lock Gate = new();
    private static readonly Dictionary<string, IntPtr> Handles = new(StringComparer.Ordinal);
    private static bool _configured;
    private static string? _directory;

    /// <summary>
    /// <c>AVERROR(EAGAIN)</c> on the running platform: errno <c>EAGAIN</c> is 11 on Linux and Windows and 35 on macOS and
    /// FreeBSD (the generated constant is the generating platform's).
    /// </summary>
    public static int AVERROR_EAGAIN => OperatingSystem.IsMacOS() || OperatingSystem.IsFreeBSD() ? -35 : AVERROR_EAGAIN_LINUX;

    /// <summary>
    /// Sets the library directory (null: the operating system's standard search) and installs the resolver. Once per
    /// process: the runtime keeps the libraries it resolved, so a second, different directory is refused.
    /// </summary>
    /// <returns>False when another directory was configured earlier in this process.</returns>
    public static bool Configure(string? directory)
    {
        lock (Gate)
        {
            if (_configured)
            {
                return string.Equals(_directory, directory, StringComparison.Ordinal);
            }

            if (!Environment.Is64BitProcess)
            {
                throw new PlatformNotSupportedException("The FFmpeg adapter binds 64-bit FFmpeg layouts only; run in a 64-bit process.");
            }

            _directory = directory;
            NativeLibrary.SetDllImportResolver(typeof(FFmpegNative).Assembly, Resolve);
            _configured = true;
            return true;
        }
    }

    /// <summary>The running operating system's shared-library naming.</summary>
    public static LibraryNaming CurrentNaming =>
        OperatingSystem.IsWindows() ? LibraryNaming.Windows
        : OperatingSystem.IsMacOS() ? LibraryNaming.MacOS
        : LibraryNaming.Unix;

    /// <summary>File name of an FFmpeg shared library of a given major on the running operating system.</summary>
    public static string FileName(string name, int major) => FileName(name, major, CurrentNaming);

    /// <summary>File name of an FFmpeg shared library of a given major under a naming convention.</summary>
    public static string FileName(string name, int major, LibraryNaming naming) => Format(name, major.ToString(System.Globalization.CultureInfo.InvariantCulture), naming);

    /// <summary>Search pattern for the major-named files of a library (<c>avcodec-*.dll</c>, <c>libavcodec.*.dylib</c>, <c>libavcodec.so.*</c>).</summary>
    public static string FilePattern(string name, LibraryNaming naming) => Format(name, "*", naming);

    /// <summary>The major of a file named exactly like <see cref="FileName(string, int, LibraryNaming)"/>, else null (other files, full-version names).</summary>
    public static int? MajorOf(string fileName, string name, LibraryNaming naming)
    {
        string prefix = Format(name, "|", naming);
        int bar = prefix.IndexOf('|', StringComparison.Ordinal);
        string head = prefix[..bar];
        string tail = prefix[(bar + 1)..];
        if (fileName.Length <= head.Length + tail.Length
            || !fileName.StartsWith(head, StringComparison.Ordinal)
            || !fileName.EndsWith(tail, StringComparison.Ordinal))
        {
            return null;
        }

        string digits = fileName[head.Length..^tail.Length];
        return digits.All(char.IsAsciiDigit) && int.TryParse(digits, System.Globalization.CultureInfo.InvariantCulture, out int major) ? major : null;
    }

    private static string Format(string name, string major, LibraryNaming naming) => naming switch
    {
        LibraryNaming.Windows => $"{name}-{major}.dll",
        LibraryNaming.MacOS => $"lib{name}.{major}.dylib",
        _ => $"lib{name}.so.{major}",
    };

    /// <summary>A string FFmpeg owns (static or context-owned): copied, never freed.</summary>
    public static string Utf8(byte* value) => Marshal.PtrToStringUTF8((IntPtr)value) ?? string.Empty;

    private static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        lock (Gate)
        {
            if (!Array.Exists(LoadOrder, l => l.Name == libraryName))
            {
                return IntPtr.Zero;
            }

            foreach (var (name, major) in LoadOrder)
            {
                if (!Handles.ContainsKey(name) && TryLoad(name, major, out IntPtr handle))
                {
                    Handles[name] = handle;
                }

                if (name == libraryName)
                {
                    break;
                }
            }

            // Zero: the runtime throws DllNotFoundException, which EmbeddingGemma2FFmpeg.Load turns into a typed error.
            return Handles.GetValueOrDefault(libraryName);
        }
    }

    private static bool TryLoad(string name, int major, out IntPtr handle)
    {
        string file = FileName(name, major);
        return _directory is not null
            ? NativeLibrary.TryLoad(Path.Combine(_directory, file), out handle)
            : NativeLibrary.TryLoad(file, out handle);
    }
}

/// <summary>How an operating system names FFmpeg's shared libraries of a major.</summary>
internal enum LibraryNaming
{
    /// <summary>Linux and other ELF systems: <c>libavcodec.so.62</c>.</summary>
    Unix,

    /// <summary>Windows: <c>avcodec-62.dll</c>.</summary>
    Windows,

    /// <summary>macOS: <c>libavcodec.62.dylib</c>.</summary>
    MacOS,
}
