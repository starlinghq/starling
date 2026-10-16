using System.Diagnostics;
using System.Globalization;

/// <summary>One memory observation. Sizes in MiB; null where the platform does not report the value.</summary>
internal sealed record MemorySample(
    double Seconds,
    string Phase,
    int Cycle,
    bool Settled,
    double ManagedHeapMb,
    double WorkingSetMb,
    double PrivateMb,
    double? RssAnonMb,
    double? RssFileMb,
    double? VmDataMb,
    double? GpuMb,
    double? AnonHugePagesMb = null)
{
    public static readonly string[] Metrics = ["managed_heap", "working_set", "private", "rss_anon", "gpu"];

    /// <summary>
    /// Whether a growth verdict on <paramref name="metric"/> fails the run. On Linux, .NET's private bytes is the
    /// <c>VmData</c> address space (glibc malloc arenas reserve 64 MiB each without touching it), not resident memory,
    /// so it is reported but does not gate; <c>rss_anon</c> is the resident equivalent. On Windows private bytes is the
    /// commit charge and gates.
    /// </summary>
    public static bool Gates(string metric) => metric != "private" || !OperatingSystem.IsLinux();

    public double? Get(string metric) => metric switch
    {
        "managed_heap" => ManagedHeapMb,
        "working_set" => WorkingSetMb,
        "private" => PrivateMb,
        "rss_anon" => RssAnonMb,
        "gpu" => GpuMb,
        _ => throw new ArgumentOutOfRangeException(nameof(metric)),
    };
}

/// <summary>
/// Reads process memory. Managed heap: <see cref="GC.GetTotalMemory(bool)"/> (live objects after a full collection for
/// settled samples). Working set and private bytes: <see cref="Process"/>. On Linux also <c>RssAnon</c> (anonymous
/// resident pages: managed heap plus native allocations such as ONNX Runtime arenas, the best leak signal) and
/// <c>RssFile</c> (file-backed resident pages, for example mapped model files) from <c>/proc/self/status</c>. GPU:
/// this process's <c>used_memory</c> from <c>nvidia-smi --query-compute-apps</c>, when nvidia-smi exists.
/// </summary>
internal static class MemoryProbe
{
    private static readonly Lazy<bool> s_nvidiaSmi = new(() => RunNvidiaSmi("--query-gpu=name --format=csv,noheader") is not null);

    public static bool GpuSamplingAvailable => s_nvidiaSmi.Value;

    /// <summary>Calls glibc <c>malloc_trim(0)</c>; false where it does not exist (not Linux, or not glibc).</summary>
    public static bool MallocTrim()
    {
        if (!OperatingSystem.IsLinux())
        {
            return false;
        }

        try
        {
            // The return value only says whether any memory was released; both outcomes are valid.
            _ = NativeMethods.malloc_trim(0);
            return true;
        }
        catch (EntryPointNotFoundException)
        {
            return false;
        }
        catch (DllNotFoundException)
        {
            return false;
        }
    }

    /// <summary>glibc's <c>gnu_get_libc_version()</c>, or null where it does not exist.</summary>
    public static string? GlibcVersion()
    {
        if (!OperatingSystem.IsLinux())
        {
            return null;
        }

        try
        {
            return System.Runtime.InteropServices.Marshal.PtrToStringUTF8(NativeMethods.gnu_get_libc_version());
        }
        catch (EntryPointNotFoundException)
        {
            return null;
        }
        catch (DllNotFoundException)
        {
            return null;
        }
    }

    /// <summary>
    /// Appends one JSON line: the anonymous mappings of <c>/proc/self/smaps</c> with more than 2 MiB resident (start,
    /// end, permissions, name, Rss, Anonymous, AnonHugePages in KiB), the totals over all mappings, and glibc's
    /// <c>mallinfo2</c> totals over all arenas where it exists (bytes in use, free in the heaps, in mmapped chunks),
    /// <c>malloc_info</c> per arena, and the GC's committed bytes and fragmentation.
    /// </summary>
    public static void AppendSmaps(string path, double seconds, int cycle, string phase)
    {
        var maps = new System.Text.Json.Nodes.JsonArray();
        long rss = 0, anon = 0, huge = 0;
        string? header = null;
        long mRss = 0, mAnon = 0, mHuge = 0;
        void Flush()
        {
            if (header is not null && mAnon > 2048)
            {
                var parts = header.Split(' ', 6, StringSplitOptions.RemoveEmptyEntries);
                maps.Add(new System.Text.Json.Nodes.JsonObject
                {
                    ["range"] = parts[0],
                    ["perms"] = parts[1],
                    ["name"] = parts.Length > 5 ? parts[5].Trim() : "",
                    ["rss_kb"] = mRss,
                    ["anon_kb"] = mAnon,
                    ["anon_huge_kb"] = mHuge,
                });
            }

            rss += mRss;
            anon += mAnon;
            huge += mHuge;
            mRss = mAnon = mHuge = 0;
        }

        try
        {
            foreach (string line in File.ReadLines("/proc/self/smaps"))
            {
                // A mapping header starts with "<start>-<end> "; field lines start with "<Name>:".
                int space = line.IndexOf(' ', StringComparison.Ordinal);
                if (space > 0 && line[space - 1] != ':' && line.AsSpan(0, space).Contains('-'))
                {
                    Flush();
                    header = line;
                }
                else if (line.StartsWith("Rss:", StringComparison.Ordinal))
                {
                    mRss = KbOf(line);
                }
                else if (line.StartsWith("Anonymous:", StringComparison.Ordinal))
                {
                    mAnon = KbOf(line);
                }
                else if (line.StartsWith("AnonHugePages:", StringComparison.Ordinal))
                {
                    mHuge = KbOf(line);
                }
            }

            Flush();
        }
        catch (IOException)
        {
            return;
        }
        catch (UnauthorizedAccessException)
        {
            return;
        }

        var row = new System.Text.Json.Nodes.JsonObject
        {
            ["t_s"] = Math.Round(seconds, 1),
            ["cycle"] = cycle,
            ["phase"] = phase,
            ["rss_kb"] = rss,
            ["anon_kb"] = anon,
            ["anon_huge_kb"] = huge,
            ["maps"] = maps,
        };
        if (MallInfo() is { } mi)
        {
            row["mallinfo2_kb"] = new System.Text.Json.Nodes.JsonObject
            {
                ["arena"] = (ulong)mi.Arena / 1024,
                ["hblkhd"] = (ulong)mi.Hblkhd / 1024,
                ["hblks"] = (ulong)mi.Hblks,
                ["uordblks"] = (ulong)mi.Uordblks / 1024,
                ["fordblks"] = (ulong)mi.Fordblks / 1024,
                ["keepcost"] = (ulong)mi.Keepcost / 1024,
            };
        }

        // The managed heap's committed memory (the GC keeps freed regions committed for reuse) and fragmentation.
        var gc = GC.GetGCMemoryInfo(GCKind.Any);
        var generations = new System.Text.Json.Nodes.JsonArray();
        foreach (var g in gc.GenerationInfo)
        {
            generations.Add(new System.Text.Json.Nodes.JsonObject
            {
                ["size_after"] = g.SizeAfterBytes / 1024,
                ["fragmentation_after"] = g.FragmentationAfterBytes / 1024,
            });
        }

        row["gc_kb"] = new System.Text.Json.Nodes.JsonObject
        {
            ["committed"] = gc.TotalCommittedBytes / 1024,
            ["heap_size"] = gc.HeapSizeBytes / 1024,
            ["fragmented"] = gc.FragmentedBytes / 1024,
            ["generations"] = generations,
        };
        if (MallocInfoXml() is { } xml)
        {
            row["malloc_info"] = xml;
        }

        File.AppendAllText(path, row.ToJsonString() + "\n");

        static long KbOf(string line) =>
            long.Parse(line.Split(':', 2)[1].Trim().Split(' ')[0], CultureInfo.InvariantCulture);
    }

    /// <summary>
    /// glibc's <c>malloc_info(0, fp)</c> XML: per arena (the main heap and each thread arena), the free chunks by size,
    /// the top chunk and the memory obtained from the system. Null where it does not exist.
    /// </summary>
    private static string? MallocInfoXml()
    {
        if (!OperatingSystem.IsLinux())
        {
            return null;
        }

        string file = Path.Combine(Path.GetTempPath(), $"starlings-soak-malloc-info-{Environment.ProcessId}.xml");
        try
        {
            IntPtr fp = NativeMethods.fopen(file, "w");
            if (fp == IntPtr.Zero)
            {
                return null;
            }

            try
            {
                _ = NativeMethods.malloc_info(0, fp);
            }
            finally
            {
                _ = NativeMethods.fclose(fp);
            }

            return File.ReadAllText(file);
        }
        catch (EntryPointNotFoundException)
        {
            return null;
        }
        catch (DllNotFoundException)
        {
            return null;
        }
        catch (IOException)
        {
            return null;
        }
        finally
        {
            try
            {
                File.Delete(file);
            }
            catch (IOException)
            {
            }
        }
    }

    /// <summary>glibc's <c>mallinfo2()</c> (glibc 2.33 and later), or null where it does not exist.</summary>
    private static MallInfo2? MallInfo()
    {
        if (!OperatingSystem.IsLinux())
        {
            return null;
        }

        try
        {
            return NativeMethods.mallinfo2();
        }
        catch (EntryPointNotFoundException)
        {
            return null;
        }
        catch (DllNotFoundException)
        {
            return null;
        }
    }

    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    internal struct MallInfo2
    {
        public nuint Arena;
        public nuint Ordblks;
        public nuint Smblks;
        public nuint Hblks;
        public nuint Hblkhd;
        public nuint Usmblks;
        public nuint Fsmblks;
        public nuint Uordblks;
        public nuint Fordblks;
        public nuint Keepcost;
    }

    private static class NativeMethods
    {
        [System.Runtime.InteropServices.DllImport("libc", EntryPoint = "mallinfo2")]
        [System.Runtime.InteropServices.DefaultDllImportSearchPaths(System.Runtime.InteropServices.DllImportSearchPath.SafeDirectories)]
        internal static extern MallInfo2 mallinfo2();

        [System.Runtime.InteropServices.DllImport("libc", EntryPoint = "gnu_get_libc_version")]
        [System.Runtime.InteropServices.DefaultDllImportSearchPaths(System.Runtime.InteropServices.DllImportSearchPath.SafeDirectories)]
        internal static extern IntPtr gnu_get_libc_version();

        [System.Runtime.InteropServices.DllImport("libc", EntryPoint = "malloc_info")]
        [System.Runtime.InteropServices.DefaultDllImportSearchPaths(System.Runtime.InteropServices.DllImportSearchPath.SafeDirectories)]
        internal static extern int malloc_info(int options, IntPtr fp);

        [System.Runtime.InteropServices.DllImport("libc", EntryPoint = "fopen", CharSet = System.Runtime.InteropServices.CharSet.Ansi, BestFitMapping = false, ThrowOnUnmappableChar = true)]
        [System.Runtime.InteropServices.DefaultDllImportSearchPaths(System.Runtime.InteropServices.DllImportSearchPath.SafeDirectories)]
        internal static extern IntPtr fopen(string path, string mode);

        [System.Runtime.InteropServices.DllImport("libc", EntryPoint = "fclose")]
        [System.Runtime.InteropServices.DefaultDllImportSearchPaths(System.Runtime.InteropServices.DllImportSearchPath.SafeDirectories)]
        internal static extern int fclose(IntPtr fp);

        [System.Runtime.InteropServices.DllImport("libc", EntryPoint = "malloc_trim")]
        [System.Runtime.InteropServices.DefaultDllImportSearchPaths(System.Runtime.InteropServices.DllImportSearchPath.SafeDirectories)]
        internal static extern int malloc_trim(nuint pad);
    }

    public static MemorySample Sample(double seconds, string phase, int cycle, bool settle)
    {
        if (settle)
        {
            for (int i = 0; i < 2; i++)
            {
                GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: true);
                GC.WaitForPendingFinalizers();
            }
        }

        using var process = Process.GetCurrentProcess();
        process.Refresh();
        var (anon, file, data) = ProcStatus();
        double? hugePages = settle ? AnonHugePages() : null;
        return new MemorySample(
            seconds,
            phase,
            cycle,
            settle,
            GC.GetTotalMemory(false) / 1048576.0,
            process.WorkingSet64 / 1048576.0,
            process.PrivateMemorySize64 / 1048576.0,
            anon,
            file,
            data,
            GpuMb(process.Id),
            hugePages);
    }

    /// <summary>
    /// <c>AnonHugePages</c> of <c>/proc/self/smaps_rollup</c> (anonymous memory backed by transparent huge pages): with
    /// THP <c>always</c>, a freed hole inside a huge page stays resident until the whole 2 MiB is free. Settled samples
    /// only (the rollup walks every mapping).
    /// </summary>
    private static double? AnonHugePages()
    {
        try
        {
            foreach (string line in File.ReadLines("/proc/self/smaps_rollup"))
            {
                if (line.StartsWith("AnonHugePages:", StringComparison.Ordinal))
                {
                    return double.Parse(line.Split(':', 2)[1].Trim().Split(' ')[0], CultureInfo.InvariantCulture) / 1024.0;
                }
            }
        }
        catch (IOException)
        {
        }
        catch (UnauthorizedAccessException)
        {
        }

        return null;
    }

    private static (double? Anon, double? File, double? Data) ProcStatus()
    {
        try
        {
            double? anon = null;
            double? file = null;
            double? data = null;
            foreach (string line in File.ReadLines("/proc/self/status"))
            {
                if (line.StartsWith("RssAnon:", StringComparison.Ordinal))
                {
                    anon = Kb(line);
                }
                else if (line.StartsWith("RssFile:", StringComparison.Ordinal))
                {
                    file = Kb(line);
                }
                else if (line.StartsWith("VmData:", StringComparison.Ordinal))
                {
                    data = Kb(line);
                }
            }

            return (anon, file, data);
        }
        catch (IOException)
        {
            return (null, null, null);
        }
        catch (UnauthorizedAccessException)
        {
            return (null, null, null);
        }

        static double Kb(string line) =>
            double.Parse(line.Split(':', 2)[1].Trim().Split(' ')[0], CultureInfo.InvariantCulture) / 1024.0;
    }

    private static double? GpuMb(int pid)
    {
        if (!GpuSamplingAvailable)
        {
            return null;
        }

        string? output = RunNvidiaSmi("--query-compute-apps=pid,used_memory --format=csv,noheader,nounits");
        if (output is null)
        {
            return null;
        }

        double total = 0;
        foreach (string line in output.Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            var parts = line.Split(',');
            if (parts.Length == 2 && int.TryParse(parts[0].Trim(), CultureInfo.InvariantCulture, out int p) && p == pid &&
                double.TryParse(parts[1].Trim(), CultureInfo.InvariantCulture, out double mb))
            {
                total += mb;
            }
        }

        return total;
    }

    public static string? RunNvidiaSmi(string arguments)
    {
        try
        {
            var info = new ProcessStartInfo("nvidia-smi", arguments) { RedirectStandardOutput = true, RedirectStandardError = true };
            using var process = Process.Start(info);
            if (process is null)
            {
                return null;
            }

            string output = process.StandardOutput.ReadToEnd();
            process.WaitForExit(10_000);
            return process.ExitCode == 0 ? output.Trim() : null;
        }
        catch (System.ComponentModel.Win32Exception)
        {
            return null;
        }
    }
}

/// <summary>Samples memory on a background thread at a fixed interval (unsettled: no forced collection).</summary>
internal sealed class BackgroundSampler : IDisposable
{
    private readonly Thread _thread;
    private readonly TimeSpan _interval;
    private readonly Stopwatch _clock;
    private readonly ManualResetEventSlim _stop = new();
    private volatile string _phase = "start";
    private volatile int _cycle = -1;

    public BackgroundSampler(Stopwatch clock, TimeSpan interval)
    {
        _clock = clock;
        _interval = interval;
        _thread = new Thread(Loop) { IsBackground = true, Name = "soak-sampler" };
        _thread.Start();
    }

    public List<MemorySample> Samples { get; } = [];

    public void SetPhase(string phase, int cycle)
    {
        _phase = phase;
        _cycle = cycle;
    }

    private void Loop()
    {
        do
        {
            var s = MemoryProbe.Sample(_clock.Elapsed.TotalSeconds, _phase, _cycle, settle: false);
            lock (Samples)
            {
                Samples.Add(s);
            }
        }
        while (!_stop.Wait(_interval));
    }

    public void Dispose()
    {
        _stop.Set();
        _thread.Join();
        _stop.Dispose();
    }
}
