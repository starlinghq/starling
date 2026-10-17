namespace Starlings.Runtime.Onnx;

/// <summary>Execution provider requested for a session.</summary>
public enum OnnxExecutionProvider
{
    /// <summary>ONNX Runtime's CPU execution provider. Available in every ONNX Runtime native package.</summary>
    Cpu,

    /// <summary>
    /// NVIDIA CUDA execution provider, fp32 with TF32 disabled (<c>use_tf32=0</c>). Requires the
    /// <c>Microsoft.ML.OnnxRuntime.Gpu.Linux</c> native package plus the CUDA 13 runtime and cuDNN 9 on the
    /// loader path. Session creation fails when any of these is missing; there is no silent CPU fallback.
    /// ONNX Runtime still assigns some shape-path nodes to the CPU provider by design; enable
    /// <see cref="OnnxSessionOptions.CapturePlacement"/> to see the actual node placement.
    /// </summary>
    Cuda,
}

/// <summary>
/// Immutable session configuration. Every setting that changes numerical results or native resource use is
/// explicit here; settings ONNX Runtime would accept but this profile does not support are not exposed.
/// </summary>
public sealed record OnnxSessionOptions
{
    /// <summary>Execution provider. Default <see cref="OnnxExecutionProvider.Cpu"/>.</summary>
    public OnnxExecutionProvider Provider { get; init; } = OnnxExecutionProvider.Cpu;

    /// <summary>
    /// Threads used inside one operator (<c>intra_op_num_threads</c>). Must be positive when set.
    /// <see langword="null"/> selects the default: when a CPU quota or affinity mask limits the process
    /// (<see cref="Environment.ProcessorCount"/> below the host's online CPUs, as in most containers), that
    /// processor count; otherwise ONNX Runtime's default (one per physical core). ONNX Runtime's own default ignores a
    /// container's CPU quota and would start one thread per physical core of the host. The count used is in
    /// <see cref="OnnxSessionDiagnostics.IntraOpNumThreads"/>.
    /// </summary>
    public int? IntraOpNumThreads { get; init; }

    /// <summary>
    /// Threads used across independent operators (<c>inter_op_num_threads</c>). Only relevant with parallel
    /// execution, which this component does not enable (execution mode is sequential); exposed so the value is
    /// recorded. <see langword="null"/> keeps ONNX Runtime's default. Must be positive when set.
    /// </summary>
    public int? InterOpNumThreads { get; init; }

    /// <summary>
    /// CUDA device ordinal as the CUDA runtime enumerates it (after <c>CUDA_VISIBLE_DEVICES</c> and
    /// <c>CUDA_DEVICE_ORDER</c>). Ignored for CPU. Default 0.
    /// </summary>
    public int CudaDeviceId { get; init; }

    /// <summary>
    /// Records which execution provider ran each node. Enables ONNX Runtime profiling for the first
    /// <see cref="OnnxModelSession.Run(IReadOnlyCollection{string}, IReadOnlyCollection{Microsoft.ML.OnnxRuntime.OrtValue}, IReadOnlyCollection{string}, CancellationToken)"/>
    /// only, then stops it and publishes <see cref="OnnxModelSession.Placement"/>. The first run is serialized
    /// with other runs while profiling is active. Default <see langword="false"/>.
    /// </summary>
    public bool CapturePlacement { get; init; }

    /// <summary>
    /// Expected lowercase hex SHA-256 of the model bytes. When set, the model is hashed before loading and a
    /// mismatch throws <see cref="OnnxModelIntegrityException"/>.
    /// </summary>
    public string? ExpectedModelSha256 { get; init; }

    /// <summary>
    /// Hash the model bytes and report the digest in <see cref="OnnxSessionDiagnostics.ModelSha256"/>.
    /// Default <see langword="true"/>. Costs one sequential read of the model file at load
    /// (about 1 s for a 470 MB file). Always on when <see cref="ExpectedModelSha256"/> is set.
    /// </summary>
    public bool ComputeModelSha256 { get; init; } = true;

    /// <summary>
    /// On Linux with glibc, call <c>malloc_trim(0)</c> when the session's native memory is released (after
    /// <see cref="OnnxModelSession.Dispose"/> and the last in-flight run), so freed memory goes back to the operating
    /// system instead of staying resident in glibc's heaps. Default <see langword="true"/>. It releases only free
    /// memory and changes no allocator setting; it costs a few milliseconds per release. A no-op on other platforms
    /// and when the process sets the <see cref="AppContext"/> switch
    /// <c>Starlings.Runtime.Onnx.DisableNativeHeapTrim</c>. See the package README, "Memory on Linux".
    /// </summary>
    public bool TrimNativeHeapOnRelease { get; init; } = true;

    /// <summary>
    /// ONNX Runtime's CPU memory arena (<c>enable_cpu_mem_arena</c>). <see langword="null"/> (the default) leaves the
    /// choice to the model library, and ONNX Runtime's default (on) applies when the session is created directly. With
    /// the arena on, ONNX Runtime keeps every CPU buffer a run needed in regions it never returns while the session
    /// lives; their resident size climbs toward the region size as runs touch different parts of it. With it off, each
    /// buffer is allocated and freed per run. Results are the same either way. See the package README,
    /// "Memory on Linux".
    /// </summary>
    public bool? CpuMemoryArena { get; init; }

    /// <summary>
    /// Execute runs on a small process-wide set of long-lived threads owned by this package instead of the calling
    /// thread; the caller blocks until its run completes. <see langword="null"/> (the default) leaves the choice to the
    /// model library, and runs execute on the calling thread when the session is created directly. On Linux with glibc,
    /// a CPU run without ONNX Runtime's arena allocates its buffers in the calling thread's malloc arena, which keeps up
    /// to 64 MiB of freed memory resident; callers drawn from a rotating thread pool then leave that memory in more and
    /// more arenas. On these threads the same few arenas are reused. At most <see cref="Environment.ProcessorCount"/>
    /// threads are created (only when all are busy); further concurrent runs wait for one. Results are the same either
    /// way. See the package README, "Memory on Linux".
    /// </summary>
    public bool? DedicatedRunThreads { get; init; }

    /// <summary>
    /// On Linux with glibc, call <c>malloc_trim(0)</c> once the session is created, so memory that session creation
    /// allocated and freed again (graph transformation and initializer copies) goes back to the operating system instead
    /// of staying resident in glibc's heaps while the session lives. <see langword="null"/> (the default) leaves the
    /// choice to the model library, and no trim is made when the session is created directly. It releases only free
    /// memory and changes no allocator setting; it costs milliseconds per load, never a request. A no-op on other
    /// platforms and when the process sets the <see cref="AppContext"/> switch
    /// <c>Starlings.Runtime.Onnx.DisableNativeHeapTrim</c>. See the package README, "Memory on Linux".
    /// </summary>
    public bool? TrimNativeHeapAfterLoad { get; init; }

    /// <summary>
    /// On Linux with glibc, call <c>malloc_trim(0)</c> on the thread that ran, after each run returns, so the run's
    /// freed buffers go back to the operating system instead of staying resident as free chunks in that thread's glibc
    /// heap. <see langword="null"/> (the default) leaves the choice to the model library, and no trim is made when the
    /// session is created directly. It releases only free memory and changes no allocator setting; it adds the trim
    /// (milliseconds) to each run and the next run faults the returned pages in again, so it suits runs that take
    /// seconds and allocate hundreds of MiB, not short ones. A no-op on other platforms and when the process sets the
    /// <see cref="AppContext"/> switch <c>Starlings.Runtime.Onnx.DisableNativeHeapTrim</c>. See the package README,
    /// "Memory on Linux".
    /// </summary>
    public bool? TrimNativeHeapAfterRun { get; init; }

    internal void Validate()
    {
        if (!Enum.IsDefined(Provider))
        {
            throw new ArgumentOutOfRangeException(nameof(Provider), Provider, "Unknown execution provider.");
        }

        if (IntraOpNumThreads is <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(IntraOpNumThreads), IntraOpNumThreads, "Must be positive, or null for the ONNX Runtime default.");
        }

        if (InterOpNumThreads is <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(InterOpNumThreads), InterOpNumThreads, "Must be positive, or null for the ONNX Runtime default.");
        }

        if (CudaDeviceId < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(CudaDeviceId), CudaDeviceId, "Must not be negative.");
        }

        if (ExpectedModelSha256 is { } sha && (sha.Length != 64 || !sha.All(char.IsAsciiHexDigitLower)))
        {
            throw new ArgumentException("Expected a 64-character lowercase hex SHA-256.", nameof(ExpectedModelSha256));
        }
    }
}
