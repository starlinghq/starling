using System.Globalization;
using System.Runtime.InteropServices;
using Starlings.Runtime.Onnx;

namespace Starlings.Models.EmbeddingGemma2;

/// <summary>
/// Opt-in vision batch limits chosen from free memory when the embedder loads
/// (<see cref="EmbeddingGemma2MultimodalOptions.AutoBatchLimits"/>). Experimental (capability
/// <c>image.auto_batch_limits</c>); off by default, and with it off the configured limits apply unchanged.
/// </summary>
/// <remarks>
/// <para>
/// After the vision and multimodal graphs are loaded, the embedder reads the free memory of the device the vision graph
/// runs on: on CPU the host's <c>MemAvailable</c> (<c>/proc/meminfo</c>), lowered to the least room left under a cgroup
/// v2 <c>memory.max</c> of the process's cgroup or any of its ancestors; on CUDA the device's free memory
/// (<c>cuMemGetInfo</c> of the CUDA driver, in the visible-device order that <see cref="OnnxSessionOptions.CudaDeviceId"/>
/// uses), capped at the CUDA arena limit (<see cref="OnnxCudaProviderOptions.GpuMemoryLimitBytes"/>,
/// <c>gpu_mem_limit</c>) when one is set. cgroup v1 limits are not read. The probe is Linux only
/// on CPU; when it is unavailable (another OS, no driver, an error) the configured limits apply unchanged and
/// <see cref="EmbeddingGemma2MultimodalEmbedder.AutoBatchAvailableBytes"/> is <see langword="null"/>.
/// </para>
/// <para>
/// The budget is <c>(free - ReservedBytes) * UsableFraction</c>. One vision-graph image of <c>P</c> padded patch rows
/// is estimated at <c>FullImageBytes * (0.25 * P / 2520 + 0.75 * (P / 2520)^2)</c>: 3 GiB at the full 280-token
/// budget (2,520 patches) and 0.94 GiB for a video frame (1,260 patches). The estimate is conservative and is
/// derived from the ONNX Runtime CUDA observations behind the defaults (one full-budget image needs a 610 MB attention
/// buffer per layer; two failed with 5.5 GB free; two frames ran with about 5 GB free); the same estimate is used on
/// CPU. It is not a measured bound, so this option is Experimental and not part of any Stable contract.
/// </para>
/// <para>
/// From the budget: <see cref="EmbeddingGemma2MultimodalOptions.MaxVisionBatchSize"/> is the number of images of
/// the configured budget's maximum size that fit, <see cref="EmbeddingGemma2MultimodalOptions.MaxVideoFramesPerVisionCall"/>
/// the number of 1,260-patch frames, and <see cref="EmbeddingGemma2MultimodalOptions.MaxVisionBatchPatches"/> the larger
/// of the two calls' patch rows. The configured values are floors (the defaults 1, 2 and 2,520 when unset), so the
/// probe only ever raises them, and <see cref="MaxImagesPerCall"/> and <see cref="MaxFramesPerCall"/> cap the raise.
/// <see cref="EmbeddingGemma2MultimodalEmbedder.Options"/> reports the limits chosen. The text-graph limits
/// (<see cref="EmbeddingGemma2MultimodalOptions.MaxBatchSize"/>, <see cref="EmbeddingGemma2MultimodalOptions.MaxBatchTokens"/>)
/// are not changed. Memory used by other processes after load is not tracked: the probe runs once.
/// </para>
/// <para>
/// Because every vision call is sized to the whole budget, the embedder runs its vision-graph calls one at a time when
/// this option is set (concurrent callers wait for each other's vision calls; text-graph calls are not serialized).
/// </para>
/// </remarks>
public sealed record EmbeddingGemma2AutoBatchLimits
{
    /// <summary>Estimated peak bytes of one vision-graph image at the full 280-token budget (2,520 patch rows): 3 GiB.</summary>
    public const long FullImageBytes = 3L << 30;

    /// <summary>Patch rows of one full-budget image (280 soft tokens).</summary>
    internal const int FullImagePatches = 2_520;

    /// <summary>Patch rows of one video frame (140 soft tokens).</summary>
    internal const int VideoFramePatches = EmbeddingGemma2MultimodalEmbedder.VideoFrameSoftTokens * EmbeddingGemma2ImageProcessor.PoolingKernelSize * EmbeddingGemma2ImageProcessor.PoolingKernelSize;

    /// <summary>
    /// Share of the free memory left after <see cref="ReservedBytes"/> that vision calls may use: the safety margin.
    /// Greater than 0 and at most 1. Default 0.5 (half of it stays free for the text graphs, the arena's growth and
    /// other processes).
    /// </summary>
    public double UsableFraction { get; init; } = 0.5;

    /// <summary>Bytes taken off the free memory before <see cref="UsableFraction"/> applies. Default 1 GiB.</summary>
    public long ReservedBytes { get; init; } = 1L << 30;

    /// <summary>Images per vision-graph call at most, whatever the probe finds. Default 8.</summary>
    public int MaxImagesPerCall { get; init; } = 8;

    /// <summary>Video frames per vision-graph call at most, whatever the probe finds. Default 16.</summary>
    public int MaxFramesPerCall { get; init; } = 16;

    /// <summary>Test seam: replaces the memory probe (provider, CUDA device id) to free bytes, or null for unavailable.</summary>
    internal Func<OnnxExecutionProvider, int, long?>? Probe { get; init; }

    /// <summary>Estimated peak bytes of one vision-graph image of <paramref name="patches"/> padded patch rows.</summary>
    public static long EstimateImageBytes(int patches)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(patches, 1);
        double r = (double)patches / FullImagePatches;
        return (long)Math.Ceiling(FullImageBytes * ((0.25 * r) + (0.75 * r * r)));
    }

    /// <summary>
    /// <paramref name="freeBytes"/> capped at the CUDA arena limit (<c>gpu_mem_limit</c>) in
    /// <paramref name="providerOptions"/>, when one is set. The cap ignores what the arena already holds.
    /// </summary>
    internal static long? CapToArenaLimit(long? freeBytes, IReadOnlyDictionary<string, string> providerOptions)
    {
        if (freeBytes is { } free
            && providerOptions.TryGetValue("gpu_mem_limit", out string? text)
            && long.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out long limit)
            && limit > 0)
        {
            return Math.Min(free, limit);
        }

        return freeBytes;
    }

    internal void Validate()
    {
        if (!(UsableFraction > 0 && UsableFraction <= 1))
        {
            throw new ArgumentOutOfRangeException(nameof(UsableFraction), UsableFraction, "Must be greater than 0 and at most 1.");
        }

        ArgumentOutOfRangeException.ThrowIfNegative(ReservedBytes);
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxImagesPerCall, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxFramesPerCall, 1);
    }

    /// <summary>
    /// The options with the vision limits chosen for <paramref name="freeBytes"/> free; unchanged when
    /// <paramref name="freeBytes"/> is null.
    /// </summary>
    internal EmbeddingGemma2MultimodalOptions Apply(EmbeddingGemma2MultimodalOptions options, long? freeBytes)
    {
        if (freeBytes is not { } free)
        {
            return options;
        }

        double budget = Math.Max(0, free - ReservedBytes) * UsableFraction;
        int imagePatches = EmbeddingGemma2ImageProcessor.MaxPatches(options.MaxImageSoftTokens);
        int images = Fit(budget, imagePatches, options.MaxVisionBatchSize, MaxImagesPerCall);
        int frames = Fit(budget, VideoFramePatches, options.MaxVideoFramesPerVisionCall, MaxFramesPerCall);
        long patches = Math.Max(options.MaxVisionBatchPatches, Math.Max((long)images * imagePatches, (long)frames * VideoFramePatches));
        return options with
        {
            MaxVisionBatchSize = images,
            MaxVideoFramesPerVisionCall = frames,
            MaxVisionBatchPatches = (int)Math.Min(patches, int.MaxValue),
        };
    }

    private static int Fit(double budget, int patches, int floor, int ceiling)
    {
        double fit = Math.Floor(budget / EstimateImageBytes(patches));
        int count = fit >= int.MaxValue ? int.MaxValue : (int)fit;
        return Math.Max(floor, Math.Min(count, ceiling));
    }

    /// <summary>Free bytes on the device of <paramref name="provider"/>, or null when the probe is unavailable.</summary>
    internal long? ProbeFreeBytes(OnnxExecutionProvider provider, int cudaDeviceId)
    {
        if (Probe is { } probe)
        {
            return probe(provider, cudaDeviceId);
        }

        try
        {
            return provider switch
            {
                OnnxExecutionProvider.Cpu => EmbeddingGemma2MemoryProbe.HostAvailableBytes(),
                OnnxExecutionProvider.Cuda => EmbeddingGemma2MemoryProbe.CudaFreeBytes(cudaDeviceId),
                _ => null,
            };
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or FormatException or OverflowException or DllNotFoundException or EntryPointNotFoundException or BadImageFormatException)
        {
            return null;
        }
    }
}

/// <summary>Free-memory probes behind <see cref="EmbeddingGemma2AutoBatchLimits"/>.</summary>
internal static class EmbeddingGemma2MemoryProbe
{
    /// <summary>
    /// Linux: <c>MemAvailable</c> from <c>/proc/meminfo</c>, lowered to the least <c>memory.max - memory.current</c>
    /// over the process's cgroup v2 and its ancestors that set a limit. Null on other systems or when <c>MemAvailable</c> is missing.
    /// </summary>
    internal static long? HostAvailableBytes()
    {
        if (!OperatingSystem.IsLinux() || !File.Exists("/proc/meminfo"))
        {
            return null;
        }

        long? available = ParseMemAvailable(File.ReadAllText("/proc/meminfo"));
        if (available is null)
        {
            return null;
        }

        return CgroupRoom() is { } room ? Math.Min(available.Value, room) : available;
    }

    /// <summary>The <c>MemAvailable</c> line of a <c>/proc/meminfo</c> text, in bytes.</summary>
    internal static long? ParseMemAvailable(string meminfo)
    {
        foreach (string line in meminfo.Split('\n'))
        {
            if (line.StartsWith("MemAvailable:", StringComparison.Ordinal))
            {
                string[] parts = line["MemAvailable:".Length..].Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
                return parts.Length > 0 && long.TryParse(parts[0], NumberStyles.None, CultureInfo.InvariantCulture, out long kb) ? checked(kb * 1024) : null;
            }
        }

        return null;
    }

    /// <summary>The room under a cgroup v2 limit: <c>memory.max</c> minus <c>memory.current</c>; null when unlimited.</summary>
    internal static long? CgroupRoom(string max, string current)
    {
        max = max.Trim();
        if (max == "max" || !long.TryParse(max, NumberStyles.None, CultureInfo.InvariantCulture, out long limit))
        {
            return null;
        }

        return long.TryParse(current.Trim(), NumberStyles.None, CultureInfo.InvariantCulture, out long used) ? Math.Max(0, limit - used) : null;
    }

    private static long? CgroupRoom()
    {
        if (!File.Exists("/proc/self/cgroup"))
        {
            return null;
        }

        // cgroup v2: a single "0::/path" line.
        string? path = File.ReadAllLines("/proc/self/cgroup").FirstOrDefault(l => l.StartsWith("0::", StringComparison.Ordinal))?[3..];
        if (path is null)
        {
            return null;
        }

        return CgroupRoomOverAncestors("/sys/fs/cgroup", path.Trim(), (dir, file) =>
        {
            string full = Path.Combine(dir, file);
            return File.Exists(full) ? File.ReadAllText(full) : null;
        });
    }

    /// <summary>
    /// The least room under a cgroup v2 limit over <paramref name="path"/> and its ancestors below
    /// <paramref name="root"/> (a slice's or pod's limit sits above the process's own cgroup); null when none is limited.
    /// <paramref name="read"/> gives a file of a cgroup directory, or null when it is missing.
    /// </summary>
    internal static long? CgroupRoomOverAncestors(string root, string path, Func<string, string, string?> read)
    {
        long? least = null;
        string relative = path.Trim('/');
        while (true)
        {
            string dir = relative.Length == 0 ? root : root.TrimEnd('/') + "/" + relative;
            if (read(dir, "memory.max") is { } max && read(dir, "memory.current") is { } current && CgroupRoom(max, current) is { } room)
            {
                least = least is { } l ? Math.Min(l, room) : room;
            }

            if (relative.Length == 0)
            {
                return least;
            }

            int slash = relative.LastIndexOf('/');
            relative = slash < 0 ? string.Empty : relative[..slash];
        }
    }

    /// <summary>
    /// Free bytes on CUDA device <paramref name="deviceId"/> (visible-device order) from the CUDA driver
    /// (<c>libcuda.so.1</c> or <c>nvcuda.dll</c>), through the device's primary context, which ONNX Runtime's CUDA
    /// provider also uses. Null when the driver or the device is unavailable.
    /// </summary>
    internal static unsafe long? CudaFreeBytes(int deviceId)
    {
        string name = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(name, out IntPtr lib))
        {
            return null;
        }

        try
        {
            if (!NativeLibrary.TryGetExport(lib, "cuInit", out IntPtr init)
                || !NativeLibrary.TryGetExport(lib, "cuDeviceGet", out IntPtr deviceGet)
                || !NativeLibrary.TryGetExport(lib, "cuDevicePrimaryCtxRetain", out IntPtr retain)
                || !NativeLibrary.TryGetExport(lib, "cuDevicePrimaryCtxRelease_v2", out IntPtr release)
                || !NativeLibrary.TryGetExport(lib, "cuCtxPushCurrent_v2", out IntPtr push)
                || !NativeLibrary.TryGetExport(lib, "cuCtxPopCurrent_v2", out IntPtr pop)
                || !NativeLibrary.TryGetExport(lib, "cuMemGetInfo_v2", out IntPtr memGetInfo))
            {
                return null;
            }

            if (((delegate* unmanaged<uint, int>)init)(0) != 0)
            {
                return null;
            }

            int device;
            if (((delegate* unmanaged<int*, int, int>)deviceGet)(&device, deviceId) != 0)
            {
                return null;
            }

            IntPtr context;
            if (((delegate* unmanaged<IntPtr*, int, int>)retain)(&context, device) != 0)
            {
                return null;
            }

            try
            {
                if (((delegate* unmanaged<IntPtr, int>)push)(context) != 0)
                {
                    return null;
                }

                try
                {
                    nuint free;
                    nuint total;
                    return ((delegate* unmanaged<nuint*, nuint*, int>)memGetInfo)(&free, &total) == 0 ? (long)free : null;
                }
                finally
                {
                    IntPtr popped;
                    _ = ((delegate* unmanaged<IntPtr*, int>)pop)(&popped);
                }
            }
            finally
            {
                _ = ((delegate* unmanaged<int, int>)release)(device);
            }
        }
        finally
        {
            NativeLibrary.Free(lib);
        }
    }
}
