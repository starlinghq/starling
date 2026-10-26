using System.Runtime.ConstrainedExecution;

namespace Starlings.Generation.OnnxGenAI;

/// <summary>
/// The owner of one native release (a GenAI generator, or the model's own reference), in the style of
/// <see cref="System.Runtime.InteropServices.SafeHandle"/>: <see cref="Dispose"/> runs the release once, on the calling
/// thread, and suppresses finalization; if the owner becomes unreachable without it, the finalizer does not run the
/// release itself but queues it to the thread pool.
/// </summary>
/// <remarks>
/// <para>
/// Why a finalizer, and why not a <see cref="System.Runtime.InteropServices.SafeHandle"/>: the ONNX Runtime GenAI
/// 0.17.1 managed types (<c>Config</c>, <c>Model</c>, <c>GeneratorParams</c>, <c>Generator</c>) each declare their
/// own finalizer (read by reflection from the 0.17.1 assembly), which runs on the finalizer thread, in no defined order
/// between a model and its generators, outside the model's native-call lock, and without the model's reference count;
/// an undisposed sequence also kept the model's count up, so the model was never released by
/// <see cref="GenAIModel.Dispose"/>. Their handles are private, so they cannot be moved into
/// a <see cref="System.Runtime.InteropServices.SafeHandle"/>. The adapter therefore suppresses their finalizers and owns
/// their release here: the deterministic path (<see cref="IDisposable.Dispose"/>) is unchanged, and the finalizer path
/// runs the same release on a thread-pool thread, so the native-call lock and the reference count are taken as on
/// every other path and never on the finalizer thread, and a native model is destroyed only after its last generator.
/// </para>
/// <para>
/// The finalizer path is a safety net for an abandoned request (for example a streaming enumerator dropped without
/// <c>Dispose</c>); it runs only after a garbage collection finds the owner unreachable, and the .NET runtime does not run
/// finalizers at process exit. Disposal remains the documented way to release native memory promptly.
/// </para>
/// </remarks>
internal sealed class NativeLifetime : CriticalFinalizerObject, IDisposable
{
    private static long s_finalizerReleases;
    private Action? _release;

    /// <summary>Creates an owner of <paramref name="release"/>, which must not throw and must be safe on any thread.</summary>
    public NativeLifetime(Action release)
    {
        ArgumentNullException.ThrowIfNull(release);
        _release = release;
    }

    /// <summary>Releases queued by finalizers since the process started (diagnostics and tests).</summary>
    public static long FinalizerReleases => Interlocked.Read(ref s_finalizerReleases);

    /// <summary>True once the release has run or been queued.</summary>
    public bool IsReleased => Volatile.Read(ref _release) is null;

    /// <summary>
    /// Suppresses the finalizer of a GenAI managed object whose release this adapter owns through a
    /// <see cref="NativeLifetime"/> (see the class remarks).
    /// </summary>
    public static void TakeOverFinalization(object native)
    {
#pragma warning disable CA1816 // On purpose: the GenAI object's own finalizer is replaced by a NativeLifetime.
        GC.SuppressFinalize(native);
#pragma warning restore CA1816
    }

    /// <summary>Runs the release on this thread, once. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _release, null) is { } release)
        {
            GC.SuppressFinalize(this);
            release();
        }
    }

    ~NativeLifetime()
    {
        if (Interlocked.Exchange(ref _release, null) is { } release)
        {
            Interlocked.Increment(ref s_finalizerReleases);
            ThreadPool.UnsafeQueueUserWorkItem(static r => RunQueued(r), release, preferLocal: false);
        }
    }

    private static void RunQueued(Action release)
    {
        try
        {
            release();
        }
#pragma warning disable CA1031 // An exception here would end the process; nothing is waiting for this release.
        catch (Exception)
#pragma warning restore CA1031
        {
        }
    }
}
