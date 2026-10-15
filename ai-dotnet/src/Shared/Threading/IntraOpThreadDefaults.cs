// Compiled into Starlings.Runtime.Onnx and Starlings.Generation.OnnxGenAI (linked source, internal to each).
using System.Globalization;

namespace Starlings.Threading;

/// <summary>
/// Default intra-op thread count for ONNX Runtime sessions in a CPU-constrained process.
/// </summary>
/// <remarks>
/// ONNX Runtime's default intra-op pool has one thread per physical core of the host, and ignores a container's CPU
/// quota (cgroup <c>cpu.max</c>). In a container limited to 31 CPUs on a 256-CPU host, a CUDA session of e5 started
/// 147 threads, each with its own glibc malloc arena (docs/evidence/cuda-soak-growth/). .NET's
/// <see cref="Environment.ProcessorCount"/> honours the quota and the affinity mask, so when it is below the host's
/// online CPU count the process is constrained and that count is used. On an unconstrained host the result is
/// <see langword="null"/>, which keeps ONNX Runtime's default, the configuration the CPU profiles were qualified with.
/// </remarks>
internal static class IntraOpThreadDefaults
{
    private static readonly Lazy<int?> s_hostOnlineCpus = new(ReadHostOnlineCpus);

    /// <summary>The count to configure: the caller's value if set, else <see cref="ForProcess"/>.</summary>
    public static int? Resolve(int? requested) => requested ?? ForProcess();

    /// <summary><see cref="Resolve(int?, int, int?)"/> for this process.</summary>
    public static int? ForProcess() => Resolve(null, Environment.ProcessorCount, s_hostOnlineCpus.Value);

    /// <summary>
    /// <paramref name="requested"/> when set; otherwise <paramref name="processorCount"/> when it is below
    /// <paramref name="hostOnlineCpus"/> (a CPU quota or affinity mask limits the process); otherwise
    /// <see langword="null"/> (ONNX Runtime default).
    /// </summary>
    public static int? Resolve(int? requested, int processorCount, int? hostOnlineCpus)
    {
        if (requested is int r)
        {
            return r;
        }

        return hostOnlineCpus is int host && processorCount >= 1 && processorCount < host ? processorCount : null;
    }

    /// <summary>
    /// Counts the CPUs in a Linux CPU list such as <c>0-255</c> or <c>0,2-5,8</c>
    /// (<c>/sys/devices/system/cpu/online</c>); <see langword="null"/> if it does not parse.
    /// </summary>
    public static int? ParseCpuList(string text)
    {
        int count = 0;
        foreach (string part in text.Trim().Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            int dash = part.IndexOf('-', StringComparison.Ordinal);
            if (dash < 0)
            {
                if (!int.TryParse(part, NumberStyles.None, CultureInfo.InvariantCulture, out _))
                {
                    return null;
                }

                count++;
            }
            else if (int.TryParse(part.AsSpan(0, dash), NumberStyles.None, CultureInfo.InvariantCulture, out int lo) &&
                     int.TryParse(part.AsSpan(dash + 1), NumberStyles.None, CultureInfo.InvariantCulture, out int hi) && hi >= lo)
            {
                count += hi - lo + 1;
            }
            else
            {
                return null;
            }
        }

        return count > 0 ? count : null;
    }

    private static int? ReadHostOnlineCpus()
    {
        if (!OperatingSystem.IsLinux())
        {
            return null;
        }

        try
        {
            return ParseCpuList(File.ReadAllText("/sys/devices/system/cpu/online"));
        }
        catch (IOException)
        {
            return null;
        }
        catch (UnauthorizedAccessException)
        {
            return null;
        }
    }
}
