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

/// <summary>How ONNX Runtime schedules the nodes of one run (<c>ExecutionMode</c>).</summary>
public enum OnnxExecutionMode
{
    /// <summary>One node at a time, in topological order (<c>ORT_SEQUENTIAL</c>). The default, and the mode every profile was qualified in.</summary>
    Sequential,

    /// <summary>
    /// Independent nodes run concurrently on an inter-op thread pool of <see cref="OnnxSessionOptions.InterOpNumThreads"/>
    /// threads (<c>ORT_PARALLEL</c>). Each node computes the same values as in sequential mode. It helps only graphs with
    /// independent branches. Experimental on CPU; untested on CUDA.
    /// </summary>
    Parallel,
}

/// <summary>Graph optimizations ONNX Runtime applies when it creates the session (<c>GraphOptimizationLevel</c>).</summary>
public enum OnnxGraphOptimizationLevel
{
    /// <summary>No graph optimization (<c>ORT_DISABLE_ALL</c>).</summary>
    None,

    /// <summary>Semantics-preserving rewrites such as constant folding and redundant-node removal (<c>ORT_ENABLE_BASIC</c>).</summary>
    Basic,

    /// <summary>Basic plus node fusions (<c>ORT_ENABLE_EXTENDED</c>).</summary>
    Extended,

    /// <summary>Extended plus layout optimizations (<c>ORT_ENABLE_LAYOUT</c>).</summary>
    Layout,

    /// <summary>Every optimization (<c>ORT_ENABLE_ALL</c>). The default, and the level every profile was qualified at.</summary>
    All,
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
    /// Threads used across independent operators (<c>inter_op_num_threads</c>). ONNX Runtime uses them only with
    /// <see cref="ExecutionMode"/> <see cref="OnnxExecutionMode.Parallel"/>; in the default sequential mode it creates no
    /// inter-op pool and the value is only recorded in <see cref="OnnxSessionDiagnostics.InterOpNumThreads"/>.
    /// <see langword="null"/> keeps ONNX Runtime's default. Must be positive when set.
    /// </summary>
    public int? InterOpNumThreads { get; init; }

    /// <summary>
    /// Node scheduling within a run. Default <see cref="OnnxExecutionMode.Sequential"/>, as before this option existed.
    /// <see cref="OnnxExecutionMode.Parallel"/> makes <see cref="InterOpNumThreads"/> take effect.
    /// </summary>
    public OnnxExecutionMode ExecutionMode { get; init; } = OnnxExecutionMode.Sequential;

    /// <summary>
    /// Graph optimization level. Default <see cref="OnnxGraphOptimizationLevel.All"/>, the level every profile was
    /// qualified at. A lower level keeps the graph closer to the file (useful to inspect it or to rule out a fusion) and
    /// can change results in the last bits; no parity claim of a model package covers another level.
    /// </summary>
    public OnnxGraphOptimizationLevel GraphOptimizationLevel { get; init; } = OnnxGraphOptimizationLevel.All;

    /// <summary>
    /// Path where ONNX Runtime writes the graph after optimization (<c>optimized_model_filepath</c>), for inspection or to
    /// load later with <see cref="OnnxGraphOptimizationLevel.None"/>. <see langword="null"/> (the default) writes nothing.
    /// An existing file is overwritten. It must not be the model file being loaded. The optimized graph can contain
    /// provider-specific nodes, so it loads only with the provider it was written for.
    /// </summary>
    public string? OptimizedModelFilePath { get; init; }

    /// <summary>
    /// Logger id of this session's ONNX Runtime messages (<c>session_log_id</c>), so messages of several sessions can be
    /// told apart. <see langword="null"/> keeps ONNX Runtime's default (empty). See
    /// <see cref="OnnxRuntimeInfo.ConfigureLogging"/>.
    /// </summary>
    public string? LogId { get; init; }

    /// <summary>
    /// Lowest severity of this session's ONNX Runtime messages that are logged. Default <see cref="OnnxLogSeverity.Error"/>,
    /// as before this option existed. Messages go to standard error unless a sink is set with
    /// <see cref="OnnxRuntimeInfo.ConfigureLogging"/>.
    /// </summary>
    public OnnxLogSeverity LogSeverity { get; init; } = OnnxLogSeverity.Error;

    /// <summary>
    /// CUDA execution provider options beyond the device id: memory limit, arena growth, cuDNN algorithm search and CUDA
    /// graphs. Used when <see cref="Provider"/> is <see cref="OnnxExecutionProvider.Cuda"/>, ignored for CPU, validated
    /// either way. <see langword="null"/> (the default) keeps ONNX Runtime's defaults. Untested on CUDA.
    /// </summary>
    public OnnxCudaProviderOptions? Cuda { get; init; }

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
    /// Lowercase hex SHA-256 of the model bytes as the caller already established it, for example from an acquisition
    /// snapshot verified with <c>CacheVerification.Sha256</c> (<c>ModelSnapshot.GetVerifiedSha256</c>). When set, the
    /// model is not read for hashing: the digest is reported in <see cref="OnnxSessionDiagnostics.ModelSha256"/> (with
    /// <see cref="OnnxSessionDiagnostics.ModelSha256Source"/> <see cref="OnnxModelDigestSource.Known"/>),
    /// <see cref="ExpectedModelSha256"/> is compared with it, and <see cref="ComputeModelSha256"/> has no effect. The
    /// caller vouches that the bytes have not changed since they were hashed. Default <see langword="null"/>.
    /// </summary>
    public string? KnownModelSha256 { get; init; }

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

    /// <summary>
    /// Throws if an option is out of range or malformed, including <see cref="Cuda"/> (validated whatever the
    /// <see cref="Provider"/>). Session creation and <see cref="OnnxRuntimeInfo.TryInitializeProvider(OnnxSessionOptions, out string?)"/>
    /// call it; call it directly to check a configuration early, for example at startup. Needs no model, GPU or CUDA
    /// libraries, so it cannot tell whether a provider or device accepts the options.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">A value is out of range or an enum value is undefined.</exception>
    /// <exception cref="ArgumentException">A path, hash or additional CUDA provider option is malformed.</exception>
    public void Validate()
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

        if (!Enum.IsDefined(ExecutionMode))
        {
            throw new ArgumentOutOfRangeException(nameof(ExecutionMode), ExecutionMode, "Unknown execution mode.");
        }

        if (!Enum.IsDefined(GraphOptimizationLevel))
        {
            throw new ArgumentOutOfRangeException(nameof(GraphOptimizationLevel), GraphOptimizationLevel, "Unknown graph optimization level.");
        }

        if (!Enum.IsDefined(LogSeverity))
        {
            throw new ArgumentOutOfRangeException(nameof(LogSeverity), LogSeverity, "Unknown log severity.");
        }

        if (OptimizedModelFilePath is { Length: 0 })
        {
            throw new ArgumentException("Must not be empty; use null to write no optimized model.", nameof(OptimizedModelFilePath));
        }

        Cuda?.Validate();

        if (ExpectedModelSha256 is { } sha && (sha.Length != 64 || !sha.All(char.IsAsciiHexDigitLower)))
        {
            throw new ArgumentException("Expected a 64-character lowercase hex SHA-256.", nameof(ExpectedModelSha256));
        }

        if (KnownModelSha256 is { } known && (known.Length != 64 || !known.All(char.IsAsciiHexDigitLower)))
        {
            throw new ArgumentException("Expected a 64-character lowercase hex SHA-256.", nameof(KnownModelSha256));
        }
    }
}
