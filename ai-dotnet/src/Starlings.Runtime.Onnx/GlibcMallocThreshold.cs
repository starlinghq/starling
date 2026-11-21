using System.Runtime.InteropServices;

namespace Starlings.Runtime.Onnx;

/// <summary>Outcome of the process-wide glibc mmap-threshold opt-in (<see cref="GlibcMallocThreshold.Apply"/>).</summary>
/// <param name="Applied">Whether <c>mallopt(M_MMAP_THRESHOLD, 65536)</c> was applied to this process.</param>
/// <param name="Description">
/// Human-readable diagnostic describing what was done, or why nothing was done (not Linux, no glibc). Empty when
/// nothing was attempted.
/// </param>
public sealed record GlibcMallocThresholdOutcome(bool Applied, string Description);

/// <summary>
/// Opt-in, process-wide glibc malloc tuning for Linux hosts that cannot set the <c>MALLOC_MMAP_THRESHOLD_</c>
/// environment variable before the process starts. <see cref="Apply"/> calls <c>mallopt(M_MMAP_THRESHOLD, 65536)</c>
/// and prints a loud diagnostic to standard error. It is off by default: nothing in this package calls it unless the
/// host asks for it, with <see cref="OnnxSessionOptions.ApplyGlibcMmapThreshold"/> on a session or <see cref="Apply"/>
/// directly.
/// </summary>
/// <remarks>
/// <para>
/// Why: with glibc's default 32 MiB mmap threshold, the large allocations of a model load go onto the main heap and
/// freed memory of them stays resident across load/unload cycles. Setting the threshold to 64 KiB makes glibc mmap
/// those allocations instead, so the kernel reclaims them at <c>free</c> (<c>munmap</c>). Setting
/// <c>MALLOC_MMAP_THRESHOLD_=65536</c> in the process environment remains the primary recommendation — it is read
/// before <c>Main</c> and needs no code. This opt-in exists for hosts that cannot control the environment of the
/// process they start (for example Corvid, which re-executes itself today to set the variable).
/// </para>
/// <para>
/// <c>mallopt</c> changes the allocator of the whole host process, and it only raises the dynamic mmap threshold
/// pairing in one direction (a lowered <c>M_MMAP_THRESHOLD</c> is honored; a later raise is capped by glibc's
/// <c>DEFAULT_MMAP_THRESHOLD_MAX</c> pairing rule). It has no effect on musl, Windows or macOS, and after the first
/// call it is cached for the process: further calls, from any session, change nothing. Callers that need the
/// environment-variable form instead: see the package README, "Setting <c>MALLOC_MMAP_THRESHOLD_</c>".
/// </para>
/// </remarks>
public static class GlibcMallocThreshold
{
    /// <summary>The threshold <see cref="Apply"/> sets, in bytes: the same 64 KiB the environment-variable form uses.</summary>
    public const int MmapThresholdBytes = 65536;

    /// <summary><c>malloc.h</c>'s <c>M_MMAP_THRESHOLD</c>.</summary>
    private const int MallocMmapThresholdParam = -3;

    // 0 unknown, 1 applied, -1 unavailable (not Linux, or glibc's mallopt not found).
    private static int s_state = OperatingSystem.IsLinux() ? 0 : -1;
    private static GlibcMallocThresholdOutcome? s_outcome;
    private static readonly object s_gate = new();

    /// <summary>
    /// Whether the opt-in has been applied to this process, cannot apply (not Linux, no glibc), or has not been
    /// attempted (<see langword="null"/>).
    /// </summary>
    public static bool? Applied
    {
        get
        {
            int state = Volatile.Read(ref s_state);
            return state == 1 ? true : state == -1 ? false : null;
        }
    }

    /// <summary>
    /// Applies <c>mallopt(M_MMAP_THRESHOLD, 65536)</c> once per process and writes a loud diagnostic to standard
    /// error (or to <paramref name="log"/> when given). Thread-safe and idempotent: after the first call, later calls
    /// return the first outcome and log nothing. Never throws.
    /// </summary>
    /// <returns>What was done, or why nothing was done.</returns>
    public static GlibcMallocThresholdOutcome Apply(Action<string>? log = null)
    {
        lock (s_gate)
        {
            if (s_outcome is not null)
            {
                return s_outcome;
            }

            s_outcome = Call();
            s_state = s_outcome.Applied ? 1 : -1;
            if (s_outcome.Description.Length > 0)
            {
                try
                {
                    (log ?? Console.Error.WriteLine).Invoke(s_outcome.Description);
                }
                catch
                {
                    // A failing sink must not break session creation; the outcome record still carries the message.
                }
            }

            return s_outcome;
        }
    }

    /// <summary>
    /// Applies the opt-in when a session requests it (<see cref="OnnxSessionOptions.ApplyGlibcMmapThreshold"/>),
    /// returning the outcome's description for the session's diagnostics. Never throws.
    /// </summary>
    internal static string EnsureApplied()
    {
        return Apply().Description;
    }

    private static GlibcMallocThresholdOutcome Call()
    {
        if (!OperatingSystem.IsLinux())
        {
            return new GlibcMallocThresholdOutcome(false, string.Empty);
        }

        try
        {
            // glibc's soname; musl and other C libraries do not provide it, so the import fails there.
            int result = NativeMethods.mallopt(MallocMmapThresholdParam, MmapThresholdBytes);
            string what = result != 0
                ? $"[Starlings.Runtime.Onnx] Applied mallopt(M_MMAP_THRESHOLD, {MmapThresholdBytes}) process-wide: glibc mmaps large allocations, so freed model memory returns to the OS. Opt-in for hosts that cannot set MALLOC_MMAP_THRESHOLD_ in the environment; the environment form remains the primary recommendation (issue #178)."
                : $"[Starlings.Runtime.Onnx] mallopt(M_MMAP_THRESHOLD, {MmapThresholdBytes}) returned 0 (not applied by glibc); set MALLOC_MMAP_THRESHOLD_={MmapThresholdBytes} in the process environment instead (issue #178).";
            return new GlibcMallocThresholdOutcome(result != 0, what);
        }
        catch (DllNotFoundException)
        {
            // No libc.so.6: not glibc (for example musl).
            return new GlibcMallocThresholdOutcome(false, string.Empty);
        }
        catch (EntryPointNotFoundException)
        {
            return new GlibcMallocThresholdOutcome(false, string.Empty);
        }
    }

    private static class NativeMethods
    {
        // glibc's soname; musl and other C libraries do not provide it, so the import fails there and the opt-in is
        // unavailable. Blittable signature: no marshalling, Native AOT compatible.
        [DllImport("libc.so.6", EntryPoint = "mallopt", ExactSpelling = true)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        internal static extern int mallopt(int param, int value);
    }
}
