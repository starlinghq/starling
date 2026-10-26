using System.Diagnostics;
using System.Security.Cryptography;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Starlings.Runtime.Onnx;

/// <summary>
/// One loaded ONNX model and the native resources behind it (inference session, session options, provider
/// options). Create it once and reuse it: loading is expensive and <see cref="Run(IReadOnlyCollection{string}, IReadOnlyCollection{OrtValue}, IReadOnlyCollection{string}, CancellationToken)"/>
/// is safe to call concurrently from multiple threads.
/// </summary>
/// <remarks>
/// <para>
/// <b>Ownership.</b> The instance owns its native session. <see cref="Dispose"/> is idempotent; calls made after it
/// throw <see cref="ObjectDisposedException"/>. A run that is already executing when <see cref="Dispose"/> is called
/// completes normally, and the native session is released when the last such run returns, so native memory is never
/// freed under an active run. Output values returned by <see cref="Run(IReadOnlyCollection{string}, IReadOnlyCollection{OrtValue}, IReadOnlyCollection{string}, CancellationToken)"/>
/// belong to the caller, who must dispose them (before disposing the session).
/// </para>
/// <para>
/// <b>Cancellation.</b> Runs are synchronous; there is no asynchronous API, because ONNX Runtime's C# binding has no
/// truly asynchronous run for these providers and wrapping a blocking call in a task would not make it cancellable.
/// A token that is already cancelled prevents the run. On CPU, a token cancelled during a run sets ONNX Runtime's
/// terminate flag, which the executor checks between graph nodes: the operator currently executing finishes first,
/// then the run stops and throws <see cref="OperationCanceledException"/>, so cancellation latency is bounded by the
/// longest single node. On CUDA a run is never terminated mid-graph: ONNX Runtime returns from a terminated CUDA run
/// with device work still pending on buffers it has already released, which corrupted later runs (wrong entities in a
/// GLiNER2 soak; <c>docs/evidence/gliner2-cuda-wrong-output/</c>). There a token cancelled during a run takes effect
/// when the run completes (its outputs are discarded and the call throws <see cref="OperationCanceledException"/>), so
/// cancellation latency is bounded by one run; callers that split work into chunks check the token between them. A
/// CUDA run that fails for another reason drains the device before the exception propagates. The session stays usable
/// after a cancelled run.
/// </para>
/// </remarks>
public sealed class OnnxModelSession : IDisposable
{
    private const string MemorySource = "<memory>";

    private readonly InferenceSession _session;
    private readonly SessionOptions _sessionOptions;
    private readonly OrtCUDAProviderOptions? _cudaOptions;
    private readonly int _cudaDeviceId;
    private readonly bool _trimOnRelease;
    private readonly bool _dedicatedRunThreads;
    private readonly bool _trimAfterRun;
    private readonly object _placementGate = new();
    private volatile bool _placementPending;
    private OnnxPlacementReport? _placement;

    // One reference held by the owner plus one per in-flight run. The native session is released at zero.
    private int _references = 1;
    private int _disposed;

    private OnnxModelSession(
        InferenceSession session,
        SessionOptions sessionOptions,
        OrtCUDAProviderOptions? cudaOptions,
        int cudaDeviceId,
        OnnxSessionDiagnostics diagnostics,
        bool capturePlacement)
    {
        _trimOnRelease = diagnostics.TrimNativeHeapOnRelease;
        _dedicatedRunThreads = diagnostics.DedicatedRunThreads;
        _trimAfterRun = diagnostics.TrimNativeHeapAfterRun;
        _session = session;
        _sessionOptions = sessionOptions;
        _cudaOptions = cudaOptions;
        _cudaDeviceId = cudaDeviceId;
        Diagnostics = diagnostics;
        _placementPending = capturePlacement;
    }

    /// <summary>Identity, configuration and graph signature of this session.</summary>
    public OnnxSessionDiagnostics Diagnostics { get; }

    /// <summary>Graph inputs.</summary>
    public IReadOnlyList<OnnxTensorInfo> Inputs => Diagnostics.Inputs;

    /// <summary>Graph outputs.</summary>
    public IReadOnlyList<OnnxTensorInfo> Outputs => Diagnostics.Outputs;

    /// <summary>
    /// Node placement observed on the first completed run when <see cref="OnnxSessionOptions.CapturePlacement"/>
    /// is set; otherwise, or before that run, <see langword="null"/>.
    /// </summary>
    public OnnxPlacementReport? Placement => Volatile.Read(ref _placement);

    /// <summary><see langword="true"/> after <see cref="Dispose"/>.</summary>
    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

    /// <summary>
    /// Loads a model file. The file is read once for its SHA-256 when <see cref="OnnxSessionOptions.ComputeModelSha256"/>
    /// or <see cref="OnnxSessionOptions.ExpectedModelSha256"/> asks for it, unless
    /// <see cref="OnnxSessionOptions.KnownModelSha256"/> supplies the digest.
    /// </summary>
    /// <exception cref="OnnxRuntimeUnavailableException">The native runtime is missing or mismatched.</exception>
    /// <exception cref="OnnxProviderUnavailableException">The requested provider cannot be used.</exception>
    /// <exception cref="OnnxModelIntegrityException">The file does not match <see cref="OnnxSessionOptions.ExpectedModelSha256"/>.</exception>
    public static OnnxModelSession Create(string modelPath, OnnxSessionOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelPath);
        options ??= new OnnxSessionOptions();
        options.Validate();
        var file = new FileInfo(modelPath);
        if (!file.Exists)
        {
            throw new FileNotFoundException("Model file not found.", modelPath);
        }

        // A digest the caller already established (an acquisition snapshot verified with SHA-256) replaces the read.
        string? sha = options.KnownModelSha256;
        if (sha is null && (options.ComputeModelSha256 || options.ExpectedModelSha256 is not null))
        {
            using var stream = file.OpenRead();
            sha = Convert.ToHexStringLower(SHA256.HashData(stream));
        }

        return CreateCore(options, file.FullName, file.Length, sha, so => new InferenceSession(file.FullName, so));
    }

    /// <summary>Loads a model from memory. The bytes are copied by ONNX Runtime and not retained.</summary>
    /// <exception cref="OnnxRuntimeUnavailableException">The native runtime is missing or mismatched.</exception>
    /// <exception cref="OnnxProviderUnavailableException">The requested provider cannot be used.</exception>
    /// <exception cref="OnnxModelIntegrityException">The bytes do not match <see cref="OnnxSessionOptions.ExpectedModelSha256"/>.</exception>
    public static OnnxModelSession Create(byte[] model, OnnxSessionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        options ??= new OnnxSessionOptions();
        options.Validate();
        string? sha = options.KnownModelSha256
            ?? (options.ComputeModelSha256 || options.ExpectedModelSha256 is not null ? Convert.ToHexStringLower(SHA256.HashData(model)) : null);
        return CreateCore(options, MemorySource, model.LongLength, sha, so => new InferenceSession(model, so));
    }

    /// <summary>
    /// Runs the model and returns outputs allocated by ONNX Runtime. The caller owns and must dispose the result.
    /// Input values stay owned by the caller and must remain alive for the duration of the call.
    /// </summary>
    /// <exception cref="OperationCanceledException">The token was cancelled before or during the run.</exception>
    /// <exception cref="ObjectDisposedException">The session was disposed.</exception>
    public IDisposableReadOnlyCollection<OrtValue> Run(
        IReadOnlyCollection<string> inputNames,
        IReadOnlyCollection<OrtValue> inputValues,
        IReadOnlyCollection<string> outputNames,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputNames);
        ArgumentNullException.ThrowIfNull(inputValues);
        ArgumentNullException.ThrowIfNull(outputNames);
        IDisposableReadOnlyCollection<OrtValue>? result = null;
        try
        {
            Execute(runOptions => result = _session.Run(runOptions, inputNames, inputValues, outputNames), cancellationToken);
        }
        catch
        {
            // A CUDA run cancelled while executing completes, then throws: its outputs are not returned.
            result?.Dispose();
            throw;
        }

        return result!;
    }

    /// <summary>
    /// Runs the model writing into caller-allocated output values (for example tensors over pooled buffers),
    /// avoiding per-call output allocation. Shapes must match what the graph produces for these inputs.
    /// </summary>
    /// <exception cref="OperationCanceledException">The token was cancelled before or during the run.</exception>
    /// <exception cref="ObjectDisposedException">The session was disposed.</exception>
    public void Run(
        IReadOnlyCollection<string> inputNames,
        IReadOnlyCollection<OrtValue> inputValues,
        IReadOnlyCollection<string> outputNames,
        IReadOnlyCollection<OrtValue> outputValues,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputNames);
        ArgumentNullException.ThrowIfNull(inputValues);
        ArgumentNullException.ThrowIfNull(outputNames);
        ArgumentNullException.ThrowIfNull(outputValues);
        Execute(runOptions => _session.Run(runOptions, inputNames, inputValues, outputNames, outputValues), cancellationToken);
    }

    /// <summary>
    /// Runs the model with inputs keyed by graph input name and returns outputs allocated by ONNX Runtime, in the order
    /// of <paramref name="outputNames"/> (or of <see cref="Outputs"/> when it is <see langword="null"/>). The caller owns
    /// and must dispose the result; input values stay owned by the caller. Same cancellation and lifetime rules as
    /// <see cref="Run(IReadOnlyCollection{string}, IReadOnlyCollection{OrtValue}, IReadOnlyCollection{string}, CancellationToken)"/>.
    /// </summary>
    /// <exception cref="OperationCanceledException">The token was cancelled before or during the run.</exception>
    /// <exception cref="ObjectDisposedException">The session was disposed.</exception>
    public IDisposableReadOnlyCollection<OrtValue> Run(
        IReadOnlyDictionary<string, OrtValue> inputs,
        IReadOnlyCollection<string>? outputNames = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        IReadOnlyCollection<string> names = outputNames ?? Diagnostics.Outputs.Select(o => o.Name).ToArray();
        IDisposableReadOnlyCollection<OrtValue>? result = null;
        try
        {
            Execute(runOptions => result = _session.Run(runOptions, inputs, names), cancellationToken);
        }
        catch
        {
            result?.Dispose();
            throw;
        }

        return result!;
    }

    /// <summary>
    /// Creates an ONNX Runtime I/O binding for this session. Bind inputs and outputs on it (for CUDA, values already in
    /// device memory, or outputs left on the device with <c>BindOutputToDevice</c> and
    /// <see cref="CreateDeviceMemoryInfo"/>), then run with <see cref="RunWithBinding"/>, so inputs are not copied per
    /// run. The caller owns the binding and must dispose it before disposing the session. A binding is not thread-safe:
    /// use one per concurrent caller. On CUDA this path is untested (no GPU evidence).
    /// </summary>
    /// <exception cref="ObjectDisposedException">The session was disposed.</exception>
    public OrtIoBinding CreateIoBinding()
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);
        return _session.CreateIoBinding();
    }

    /// <summary>
    /// Memory description of the session's device for <c>OrtIoBinding.BindOutputToDevice</c> and device tensors: the
    /// CUDA device of <see cref="OnnxSessionOptions.CudaDeviceId"/> for a CUDA session, CPU memory otherwise. The caller
    /// owns and must dispose it.
    /// </summary>
    public OrtMemoryInfo CreateDeviceMemoryInfo() => _cudaOptions is not null
        ? new OrtMemoryInfo(OrtMemoryInfo.allocatorCUDA, OrtAllocatorType.DeviceAllocator, _cudaDeviceId, OrtMemType.Default)
        : new OrtMemoryInfo(OrtMemoryInfo.allocatorCPU, OrtAllocatorType.DeviceAllocator, 0, OrtMemType.Default);

    /// <summary>
    /// Runs the model on the inputs and outputs bound in <paramref name="binding"/> (from <see cref="CreateIoBinding"/>).
    /// Outputs bound to a device are read afterwards with <c>OrtIoBinding.GetOutputValues</c>. Same cancellation,
    /// placement capture and lifetime rules as the other runs; bound values must stay alive for the duration of the call.
    /// </summary>
    /// <exception cref="OperationCanceledException">The token was cancelled before or during the run.</exception>
    /// <exception cref="ObjectDisposedException">The session was disposed.</exception>
    public void RunWithBinding(OrtIoBinding binding, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(binding);
        Execute(runOptions => _session.RunWithBinding(runOptions, binding), cancellationToken);
    }

    /// <summary>
    /// Runs the model <paramref name="runs"/> times on the given inputs and discards the outputs, so first-run costs
    /// (allocator growth, kernel selection, cuDNN algorithm search, CUDA graph capture) are paid before serving. With
    /// <see cref="OnnxSessionOptions.CapturePlacement"/> the first warmup run is the profiled one.
    /// </summary>
    /// <param name="inputs">Inputs keyed by graph input name; owned by the caller.</param>
    /// <param name="runs">Number of runs, at least 1. Default 1.</param>
    /// <param name="cancellationToken">Checked before each run, and during it as for <see cref="Run(IReadOnlyDictionary{string, OrtValue}, IReadOnlyCollection{string}?, CancellationToken)"/>.</param>
    /// <returns>The wall-clock duration of each run.</returns>
    /// <exception cref="OperationCanceledException">The token was cancelled.</exception>
    /// <exception cref="ObjectDisposedException">The session was disposed.</exception>
    public IReadOnlyList<TimeSpan> Warmup(IReadOnlyDictionary<string, OrtValue> inputs, int runs = 1, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentOutOfRangeException.ThrowIfLessThan(runs, 1);
        var durations = new TimeSpan[runs];
        for (int i = 0; i < runs; i++)
        {
            long start = Stopwatch.GetTimestamp();
            using (Run(inputs, null, cancellationToken))
            {
            }

            durations[i] = Stopwatch.GetElapsedTime(start);
        }

        return durations;
    }

    /// <summary>
    /// <see cref="Warmup(IReadOnlyDictionary{string, OrtValue}, int, CancellationToken)"/> with zero-filled inputs built
    /// from <see cref="Inputs"/>: each symbolic dimension takes its value from <paramref name="symbolicDimensions"/>
    /// (by dimension name) or 1. Zeros are valid token ids and masks for most graphs but not for every one; a graph that
    /// needs meaningful values (or has string or non-tensor inputs, which throw <see cref="NotSupportedException"/>)
    /// takes the overload with explicit inputs. Warm up with the shapes that will be served: allocator growth and
    /// kernel selection depend on them.
    /// </summary>
    /// <param name="symbolicDimensions">Values of symbolic dimensions by name, for example <c>batch</c> and <c>sequence</c>; each positive.</param>
    /// <param name="runs">Number of runs, at least 1. Default 1.</param>
    /// <param name="cancellationToken">As for the other overload.</param>
    /// <returns>The wall-clock duration of each run.</returns>
    /// <exception cref="NotSupportedException">A graph input is not a numeric or boolean tensor.</exception>
    /// <exception cref="OperationCanceledException">The token was cancelled.</exception>
    /// <exception cref="ObjectDisposedException">The session was disposed.</exception>
    public IReadOnlyList<TimeSpan> WarmupWithZeroInputs(
        IReadOnlyDictionary<string, long>? symbolicDimensions = null,
        int runs = 1,
        CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(runs, 1);
        if (symbolicDimensions is not null)
        {
            foreach (var (name, value) in symbolicDimensions)
            {
                if (value <= 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(symbolicDimensions), value, $"Dimension '{name}' must be positive.");
                }
            }
        }

        ObjectDisposedException.ThrowIf(IsDisposed, this);
        var inputs = new Dictionary<string, OrtValue>(StringComparer.Ordinal);
        try
        {
            foreach (var (name, metadata) in _session.InputMetadata)
            {
                if (metadata.OnnxValueType != OnnxValueType.ONNX_TYPE_TENSOR || metadata.ElementDataType == TensorElementType.String)
                {
                    throw new NotSupportedException(
                        $"Input '{name}' is not a numeric or boolean tensor ({metadata.OnnxValueType}, {metadata.ElementDataType}); warm up with explicit inputs.");
                }

                long[] shape = new long[metadata.Dimensions.Length];
                for (int d = 0; d < shape.Length; d++)
                {
                    shape[d] = metadata.Dimensions[d] >= 0
                        ? metadata.Dimensions[d]
                        : symbolicDimensions is not null && symbolicDimensions.TryGetValue(metadata.SymbolicDimensions[d], out long v) ? v : 1;
                }

                var value = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, metadata.ElementDataType, shape);
                inputs.Add(name, value);
                value.GetTensorMutableRawData().Clear();
            }

            return Warmup(inputs, runs, cancellationToken);
        }
        finally
        {
            foreach (var value in inputs.Values)
            {
                value.Dispose();
            }
        }
    }

    /// <summary>
    /// Releases the native session once no run is executing. Idempotent. Runs already executing complete first. A
    /// placement profile that was started (<see cref="OnnxSessionOptions.CapturePlacement"/>) but never read, because no
    /// run completed, is ended and its temporary file deleted.
    /// </summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            Release();
        }
    }

    private static OnnxModelSession CreateCore(
        OnnxSessionOptions options,
        string source,
        long size,
        string? sha,
        Func<SessionOptions, InferenceSession> load)
    {
        if (options.ExpectedModelSha256 is { } expected && !string.Equals(expected, sha, StringComparison.Ordinal))
        {
            string origin = options.KnownModelSha256 is null ? string.Empty : " (the caller's KnownModelSha256, not re-read)";
            throw new OnnxModelIntegrityException($"Model {source} has SHA-256 {sha}{origin}; expected {expected}.");
        }

        var env = OnnxRuntimeInfo.Environment;
        var available = env.GetAvailableProviders();
        bool cuda = options.Provider == OnnxExecutionProvider.Cuda;
        if (cuda && !available.Contains(OnnxRuntimeInfo.CudaProviderName, StringComparer.Ordinal))
        {
            throw new OnnxProviderUnavailableException(OnnxRuntimeInfo.NoCudaProvider(env.GetVersionString(), available));
        }

        string? optimizedModelPath = options.OptimizedModelFilePath is { } optimized ? Path.GetFullPath(optimized) : null;
        if (optimizedModelPath is not null && string.Equals(optimizedModelPath, source, OperatingSystem.IsLinux() ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException("The optimized model path must not be the model file being loaded.", nameof(options));
        }

        var sessionOptions = new SessionOptions
        {
            LogSeverityLevel = OnnxLogging.ToOrt(options.LogSeverity),
            ExecutionMode = options.ExecutionMode == OnnxExecutionMode.Parallel ? ExecutionMode.ORT_PARALLEL : ExecutionMode.ORT_SEQUENTIAL,
            GraphOptimizationLevel = ToOrt(options.GraphOptimizationLevel),
        };
        if (options.CpuMemoryArena is bool arena)
        {
            sessionOptions.EnableCpuMemArena = arena;
        }

        if (options.LogId is { } logId)
        {
            sessionOptions.LogId = logId;
        }

        if (optimizedModelPath is not null)
        {
            sessionOptions.OptimizedModelFilePath = optimizedModelPath;
        }

        OrtCUDAProviderOptions? cudaOptions = null;
        InferenceSession? session = null;
        try
        {
            // ONNX Runtime's own default ignores a container CPU quota; see IntraOpThreadDefaults.
            int? intraOpThreads = Starlings.Threading.IntraOpThreadDefaults.Resolve(options.IntraOpNumThreads);
            if (intraOpThreads is int intra)
            {
                sessionOptions.IntraOpNumThreads = intra;
            }

            // ONNX Runtime creates an inter-op pool only in parallel execution mode; in sequential mode the value is recorded only.
            if (options.InterOpNumThreads is int inter)
            {
                sessionOptions.InterOpNumThreads = inter;
            }

            if (options.CapturePlacement)
            {
                // The prefix first: the EnableProfiling setter passes the current prefix to native code, and ONNX
                // Runtime's default writes to the working directory with a millisecond timestamp that concurrent
                // sessions share.
                sessionOptions.ProfileOutputPathPrefix = Path.Combine(Path.GetTempPath(), "starlings-ort-placement-" + Guid.NewGuid().ToString("N"));
                sessionOptions.EnableProfiling = true;
            }

            var providerOptions = new Dictionary<string, string>(StringComparer.Ordinal);
            if (cuda)
            {
                // Creating or updating the provider options already loads libonnxruntime_providers_cuda and its CUDA
                // dependencies, so failures can surface at any of these calls.
                cudaOptions = OnnxCudaProviderOptions.CreateNative(options.CudaDeviceId, options.Cuda);
                try
                {
                    sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
                }
                catch (OnnxRuntimeException ex)
                {
                    throw OnnxCudaProviderOptions.CudaUnavailable(ex);
                }

                providerOptions = ParseProviderOptions(cudaOptions.GetOptions());
                if (!providerOptions.TryGetValue("use_tf32", out var tf32) || tf32 != "0")
                {
                    throw new InvalidOperationException($"CUDA provider reports use_tf32={tf32 ?? "<missing>"} after configuration; the fp32 profile requires 0.");
                }
            }

            try
            {
                session = load(sessionOptions);
            }
            catch (OnnxRuntimeException ex) when (cuda)
            {
                throw OnnxCudaProviderOptions.CudaUnavailable(ex);
            }

            // Session creation frees much of what it allocates (graph transformation, initializer copies); on the
            // creating thread's glibc heap that memory stays resident for the life of the session unless trimmed.
            bool trimmedAfterLoad = options.TrimNativeHeapAfterLoad == true && NativeHeap.Trim();
            var diagnostics = new OnnxSessionDiagnostics
            {
                OrtVersion = env.GetVersionString(),
                ManagedBindingVersion = OnnxRuntimeInfo.ManagedVersion,
                AvailableProviders = available,
                RequestedProvider = options.Provider,
                SessionProviders = cuda
                    ? [OnnxRuntimeInfo.CudaProviderName, OnnxRuntimeInfo.CpuProviderName]
                    : [OnnxRuntimeInfo.CpuProviderName],
                ProviderOptions = providerOptions,
                IntraOpNumThreads = intraOpThreads,
                InterOpNumThreads = options.InterOpNumThreads,
                TrimNativeHeapOnRelease = options.TrimNativeHeapOnRelease && NativeHeap.Enabled,
                TrimNativeHeapAfterLoad = trimmedAfterLoad,
                TrimNativeHeapAfterRun = options.TrimNativeHeapAfterRun == true && NativeHeap.Enabled,
                CpuMemoryArena = sessionOptions.EnableCpuMemArena,
                DedicatedRunThreads = options.DedicatedRunThreads ?? false,
                GraphOptimizationLevel = sessionOptions.GraphOptimizationLevel.ToString(),
                ExecutionMode = options.ExecutionMode,
                OptimizedModelFilePath = optimizedModelPath,
                LogId = options.LogId,
                LogSeverity = options.LogSeverity,
                ModelSource = source,
                ModelSizeBytes = size,
                ModelSha256 = sha,
                ModelSha256Source = sha is null ? null : options.KnownModelSha256 is null ? OnnxModelDigestSource.Computed : OnnxModelDigestSource.Known,
                Model = ReadModelMetadata(session),
                Inputs = ReadTensorInfo(session.InputMetadata),
                Outputs = ReadTensorInfo(session.OutputMetadata),
            };
            return new OnnxModelSession(session, sessionOptions, cudaOptions, options.CudaDeviceId, diagnostics, options.CapturePlacement);
        }
        catch
        {
            session?.Dispose();
            cudaOptions?.Dispose();
            sessionOptions.Dispose();
            throw;
        }
    }

    private static GraphOptimizationLevel ToOrt(OnnxGraphOptimizationLevel level) => level switch
    {
        OnnxGraphOptimizationLevel.None => GraphOptimizationLevel.ORT_DISABLE_ALL,
        OnnxGraphOptimizationLevel.Basic => GraphOptimizationLevel.ORT_ENABLE_BASIC,
        OnnxGraphOptimizationLevel.Extended => GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        OnnxGraphOptimizationLevel.Layout => GraphOptimizationLevel.ORT_ENABLE_LAYOUT,
        _ => GraphOptimizationLevel.ORT_ENABLE_ALL,
    };

    private static Dictionary<string, string> ParseProviderOptions(string options)
    {
        // OrtCUDAProviderOptions.GetOptions returns "key=value;key=value".
        var result = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var pair in options.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            int eq = pair.IndexOf('=', StringComparison.Ordinal);
            if (eq > 0)
            {
                result[pair[..eq]] = pair[(eq + 1)..];
            }
        }

        return result;
    }

    private static OnnxModelMetadata ReadModelMetadata(InferenceSession session)
    {
        var m = session.ModelMetadata;
        return new OnnxModelMetadata(
            m.ProducerName,
            m.GraphName,
            m.Domain,
            m.Description,
            m.Version,
            new Dictionary<string, string>(m.CustomMetadataMap, StringComparer.Ordinal));
    }

    private static List<OnnxTensorInfo> ReadTensorInfo(IReadOnlyDictionary<string, NodeMetadata> metadata) =>
        metadata.Select(kv => new OnnxTensorInfo(
            kv.Key,
            kv.Value.OnnxValueType == OnnxValueType.ONNX_TYPE_TENSOR
                ? kv.Value.ElementDataType.ToString()
                : kv.Value.OnnxValueType.ToString(),
            kv.Value.Dimensions.Select(d => (long)d).ToArray(),
            kv.Value.SymbolicDimensions.ToArray()))
        .ToList();

    private void Execute(Action<RunOptions> run, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        Acquire();
        try
        {
            if (_placementPending)
            {
                lock (_placementGate)
                {
                    if (_placementPending)
                    {
                        RunOnce(run, cancellationToken);
                        CapturePlacement();
                        return;
                    }
                }
            }

            RunOnce(run, cancellationToken);
        }
        finally
        {
            Release();
        }
    }

    private void RunOnce(Action<RunOptions> run, CancellationToken cancellationToken)
    {
        // RunOptions is per call: its terminate flag must not leak into concurrent runs.
        using var runOptions = new RunOptions();
        if (_cudaOptions is not null)
        {
            RunOnceCuda(run, runOptions, cancellationToken);
            return;
        }

        using var registration = cancellationToken.CanBeCanceled
            ? cancellationToken.UnsafeRegister(static state => ((RunOptions)state!).Terminate = true, runOptions)
            : default;
        try
        {
            Invoke(run, runOptions);
        }
        catch (OnnxRuntimeException ex) when (cancellationToken.IsCancellationRequested)
        {
            throw new OperationCanceledException("The ONNX Runtime run was cancelled at a node boundary.", ex, cancellationToken);
        }
    }

    /// <summary>Calls ONNX Runtime, on a dedicated run thread when the session uses them (<see cref="RunThreads"/>).</summary>
    private void Invoke(Action<RunOptions> run, RunOptions runOptions)
    {
        if (_dedicatedRunThreads)
        {
            RunThreads.Invoke(() => RunAndTrim(run, runOptions));
        }
        else
        {
            RunAndTrim(run, runOptions);
        }
    }

    /// <summary>
    /// Runs, then (<see cref="OnnxSessionOptions.TrimNativeHeapAfterRun"/>) returns the run's freed buffers, which
    /// glibc keeps as free chunks in this thread's heap, to the operating system.
    /// </summary>
    private void RunAndTrim(Action<RunOptions> run, RunOptions runOptions)
    {
        try
        {
            run(runOptions);
        }
        finally
        {
            if (_trimAfterRun)
            {
                _ = NativeHeap.Trim();
            }
        }
    }

    /// <summary>
    /// A CUDA run is never terminated mid-graph. ONNX Runtime 1.30 returns from a terminated CUDA run while kernels and
    /// copies it already queued are still pending, and the run's buffers are back in its allocators: the next run (or a
    /// concurrent one) can then compute on memory that the abandoned work still writes. A GLiNER2 soak returned empty
    /// entities after such runs (docs/evidence/gliner2-cuda-wrong-output/). So the run completes, and a token cancelled
    /// meanwhile takes effect when it returns. Any other failure drains the device before the exception propagates.
    /// </summary>
    private void RunOnceCuda(Action<RunOptions> run, RunOptions runOptions, CancellationToken cancellationToken)
    {
        try
        {
            Invoke(run, runOptions);
        }
        catch (OnnxRuntimeException)
        {
            DrainDevice();
            throw;
        }

        if (cancellationToken.IsCancellationRequested)
        {
            throw new OperationCanceledException("The ONNX Runtime run was cancelled; on CUDA the run completes before the cancellation takes effect.", cancellationToken);
        }
    }

    /// <summary>
    /// Waits for all work queued on the session's device (the CUDA provider's <c>Sync</c>, a device synchronization,
    /// reached through the I/O binding's synchronize calls, which sync every non-CPU provider of the graph's inputs and
    /// outputs).
    /// </summary>
    private void DrainDevice()
    {
        try
        {
            using var binding = _session.CreateIoBinding();
            binding.SynchronizeBoundInputs();
            binding.SynchronizeBoundOutputs();
        }
        catch (OnnxRuntimeException)
        {
            // The run's own exception is the one to report; a device that cannot synchronize fails the next run.
        }
    }

    private void CapturePlacement()
    {
        string path = _session.EndProfiling();
        _placementPending = false;
        try
        {
            using var stream = File.OpenRead(path);
            Volatile.Write(ref _placement, PlacementProfileParser.Parse(stream));
        }
        finally
        {
            File.Delete(path);
        }
    }

    private void Acquire()
    {
        while (true)
        {
            int current = Volatile.Read(ref _references);
            ObjectDisposedException.ThrowIf(current == 0 || IsDisposed, this);
            if (Interlocked.CompareExchange(ref _references, current + 1, current) == current)
            {
                break;
            }
        }

        if (IsDisposed)
        {
            Release();
            ObjectDisposedException.ThrowIf(true, this);
        }
    }

    /// <summary>
    /// Ends a profile that no completed run consumed (the session is being released before its first run, or every run
    /// failed) so its temporary file does not stay behind. Runs at reference count zero, so no capture is in progress.
    /// </summary>
    private void DiscardPendingPlacementProfile()
    {
        if (!_placementPending)
        {
            return;
        }

        _placementPending = false;
        try
        {
            File.Delete(_session.EndProfiling());
        }
        catch (Exception ex) when (ex is OnnxRuntimeException or IOException or UnauthorizedAccessException)
        {
            // Disposal must not throw; a profile file left behind is the only consequence.
        }
    }

    private void Release()
    {
        if (Interlocked.Decrement(ref _references) == 0)
        {
            DiscardPendingPlacementProfile();
            _session.Dispose();
            _cudaOptions?.Dispose();
            _sessionOptions.Dispose();

            // glibc keeps the freed session memory resident; hand the free pages back (NativeHeap).
            if (_trimOnRelease)
            {
                NativeHeap.Trim();
            }
        }
    }
}
