using System.Runtime.InteropServices;

namespace Starlings.Runtime.Onnx;

/// <summary>
/// Returns freed native heap memory to the operating system after a session is released, on Linux with glibc.
/// </summary>
/// <remarks>
/// <para>
/// A session's native allocations (graph, initializers, ONNX Runtime's CPU arena, per-run buffers) are freed when it
/// is released, but glibc malloc keeps freed memory of its heaps resident for reuse: across many load/unload cycles
/// the process's anonymous resident memory climbs to a high-water mark that no live object uses
/// (docs/evidence/gliner2-soak-growth/, docs/evidence/scheduled-regression-fix/). <c>malloc_trim(0)</c> hands those
/// free pages back to the kernel. It only releases memory that is already free: it changes no allocator setting
/// (unlike <c>mallopt</c> or the <c>MALLOC_*</c> environment variables) and no live allocation.
/// </para>
/// <para>
/// Elsewhere (Windows, macOS, musl) this is a no-op. A host turns it off for the whole process with the
/// <see cref="AppContext"/> switch <c>Starlings.Runtime.Onnx.DisableNativeHeapTrim</c> (for example
/// <c>&lt;RuntimeHostConfigurationOption Include="Starlings.Runtime.Onnx.DisableNativeHeapTrim" Value="true" /&gt;</c>
/// in the project file), or per session with <see cref="OnnxSessionOptions.TrimNativeHeapOnRelease"/>.
/// </para>
/// </remarks>
internal static class NativeHeap
{
    /// <summary>The <see cref="AppContext"/> switch that disables the trim for the process.</summary>
    internal const string DisableSwitch = "Starlings.Runtime.Onnx.DisableNativeHeapTrim";

    // 0 unknown, 1 available, -1 unavailable (not Linux, not glibc).
    private static int s_state = OperatingSystem.IsLinux() ? 0 : -1;

    /// <summary>Whether a trim would run in this process: Linux, glibc not ruled out, and the switch not set.</summary>
    internal static bool Enabled =>
        Volatile.Read(ref s_state) >= 0 && !(AppContext.TryGetSwitch(DisableSwitch, out bool disabled) && disabled);

    /// <summary>
    /// Calls <c>malloc_trim(0)</c> when <see cref="Enabled"/>. Returns whether the call was made. Never throws.
    /// </summary>
    internal static bool Trim()
    {
        if (!Enabled)
        {
            return false;
        }

        try
        {
            // The return value only says whether any memory was released; both outcomes are fine.
            _ = NativeMethods.malloc_trim(0);
            Volatile.Write(ref s_state, 1);
            return true;
        }
        catch (DllNotFoundException)
        {
            // No libc.so.6: not glibc (for example musl).
            Volatile.Write(ref s_state, -1);
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            Volatile.Write(ref s_state, -1);
            return false;
        }
    }

    private static class NativeMethods
    {
        // glibc's soname; musl and other C libraries do not provide it, so the import fails there and the trim is
        // disabled. Blittable signature: no marshalling, Native AOT compatible.
        [DllImport("libc.so.6", EntryPoint = "malloc_trim", ExactSpelling = true)]
        [DefaultDllImportSearchPaths(DllImportSearchPath.SafeDirectories)]
        internal static extern int malloc_trim(nuint pad);
    }
}
