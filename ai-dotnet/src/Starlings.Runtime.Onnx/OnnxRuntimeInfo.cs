using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using Microsoft.ML.OnnxRuntime;

namespace Starlings.Runtime.Onnx;

/// <summary>Process-wide facts about the loaded ONNX Runtime.</summary>
public static class OnnxRuntimeInfo
{
    /// <summary>Provider name ONNX Runtime uses for the CPU execution provider.</summary>
    public const string CpuProviderName = "CPUExecutionProvider";

    /// <summary>Provider name ONNX Runtime uses for the CUDA execution provider.</summary>
    public const string CudaProviderName = "CUDAExecutionProvider";

    private static readonly Lock LoggingGate = new();

    // The native environment keeps calling the logging delegate for the life of the process.
    private static DOrtLoggingFunction? s_loggingFunction;

    /// <summary>
    /// Version of the managed binding assembly (<c>Microsoft.ML.OnnxRuntime.Managed</c>), without build metadata.
    /// </summary>
    public static string ManagedVersion { get; } = ReadManagedVersion();

    /// <summary>
    /// Version string reported by the loaded native library. Throws
    /// <see cref="OnnxRuntimeUnavailableException"/> when no native library can be loaded.
    /// </summary>
    public static string NativeVersion => Environment.GetVersionString();

    /// <summary>
    /// Execution providers compiled into the loaded native library. A listed provider can still fail to
    /// initialize when its own native dependencies (for CUDA: CUDA runtime, cuBLAS, cuDNN) are missing.
    /// </summary>
    public static IReadOnlyList<string> AvailableProviders => Environment.GetAvailableProviders();

    /// <summary>
    /// The ONNX Runtime environment, after checking that the native library loads and matches
    /// <see cref="ManagedVersion"/>.
    /// </summary>
    internal static OrtEnv Environment => LoadEnvironment(OrtEnv.Instance);

    /// <summary>
    /// Checks that an execution provider can be initialized in this process without loading a model: the native
    /// library loads and matches <see cref="ManagedVersion"/>, and for <see cref="OnnxExecutionProvider.Cuda"/> the
    /// native library includes the CUDA provider and <c>libonnxruntime_providers_cuda</c>, with its CUDA and cuDNN
    /// dependencies, loads and accepts the provider options. No model is loaded and no device memory is allocated, so a
    /// missing, busy or full device can still fail at session creation. On a CPU-only native package the CUDA check
    /// returns <see langword="false"/> with the reason. The CUDA path has no GPU evidence yet (untested).
    /// </summary>
    /// <param name="provider">Provider to check.</param>
    /// <param name="deviceId">CUDA device ordinal (ignored for CPU). Must not be negative.</param>
    /// <param name="problem">Why the provider cannot be used; <see langword="null"/> when it can.</param>
    /// <returns><see langword="true"/> when the provider initialized.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="provider"/> is unknown or <paramref name="deviceId"/> is negative.</exception>
    public static bool TryInitializeProvider(OnnxExecutionProvider provider, int deviceId, [NotNullWhen(false)] out string? problem) =>
        TryInitializeProvider(new OnnxSessionOptions { Provider = provider, CudaDeviceId = deviceId }, out problem);

    /// <summary>
    /// <see cref="TryInitializeProvider(OnnxExecutionProvider, int, out string?)"/> for the provider, device and
    /// <see cref="OnnxSessionOptions.Cuda"/> options of a session configuration, so the provider options reach ONNX
    /// Runtime exactly as session creation passes them.
    /// </summary>
    /// <param name="options">Session configuration; validated first, as session creation validates it.</param>
    /// <param name="problem">Why the provider cannot be used; <see langword="null"/> when it can.</param>
    /// <returns><see langword="true"/> when the provider initialized.</returns>
    /// <exception cref="ArgumentException">An option is invalid.</exception>
    public static bool TryInitializeProvider(OnnxSessionOptions options, [NotNullWhen(false)] out string? problem)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        try
        {
            var env = Environment;
            if (options.Provider == OnnxExecutionProvider.Cpu)
            {
                problem = null;
                return true;
            }

            var available = env.GetAvailableProviders();
            if (!available.Contains(CudaProviderName, StringComparer.Ordinal))
            {
                problem = NoCudaProvider(env.GetVersionString(), available);
                return false;
            }

            using var cuda = OnnxCudaProviderOptions.CreateNative(options.CudaDeviceId, options.Cuda);
            using var sessionOptions = new SessionOptions();
            try
            {
                sessionOptions.AppendExecutionProvider_CUDA(cuda);
            }
            catch (OnnxRuntimeException ex)
            {
                throw OnnxCudaProviderOptions.CudaUnavailable(ex, options.Cuda);
            }

            problem = null;
            return true;
        }
        catch (OnnxSessionException ex) when (ex is OnnxRuntimeUnavailableException or OnnxProviderUnavailableException)
        {
            problem = ex.Message;
            return false;
        }
    }

    /// <summary>
    /// Sends ONNX Runtime's log messages to <paramref name="sink"/> instead of standard error. ONNX Runtime accepts a
    /// logging function only when it creates its process-wide environment, so call this once, before anything else in
    /// the process uses ONNX Runtime (session creation, <see cref="NativeVersion"/>, <see cref="AvailableProviders"/>,
    /// <see cref="TryInitializeProvider(OnnxExecutionProvider, int, out string?)"/>, or another library calling ONNX
    /// Runtime directly). The sink runs on ONNX Runtime's threads, possibly concurrently, and should not block; an
    /// exception it throws is dropped. A session's messages are filtered by <see cref="OnnxSessionOptions.LogSeverity"/>
    /// and carry <see cref="OnnxSessionOptions.LogId"/>; environment messages are filtered by
    /// <paramref name="minimumSeverity"/>.
    /// </summary>
    /// <param name="sink">Receives each message.</param>
    /// <param name="minimumSeverity">Lowest severity of environment-level messages. Default <see cref="OnnxLogSeverity.Warning"/>.</param>
    /// <param name="logId">Logger id of environment-level messages. <see langword="null"/> or empty selects <c>starlings</c>.</param>
    /// <exception cref="InvalidOperationException">The ONNX Runtime environment already exists in this process.</exception>
    /// <exception cref="OnnxRuntimeUnavailableException">The native runtime is missing or mismatched.</exception>
    public static void ConfigureLogging(Action<OnnxRuntimeLogMessage> sink, OnnxLogSeverity minimumSeverity = OnnxLogSeverity.Warning, string? logId = null)
    {
        ArgumentNullException.ThrowIfNull(sink);
        if (!Enum.IsDefined(minimumSeverity))
        {
            throw new ArgumentOutOfRangeException(nameof(minimumSeverity), minimumSeverity, "Unknown log severity.");
        }

        lock (LoggingGate)
        {
            if (OrtEnv.IsCreated)
            {
                throw new InvalidOperationException(
                    "The ONNX Runtime environment already exists in this process, and ONNX Runtime accepts a logging function only when it creates it. " +
                    "Call OnnxRuntimeInfo.ConfigureLogging before any session is created or any other ONNX Runtime call.");
            }

            var function = OnnxLogging.Adapt(sink);
            var creation = new EnvironmentCreationOptions
            {
                logId = string.IsNullOrEmpty(logId) ? "starlings" : logId,
                logLevel = OnnxLogging.ToOrt(minimumSeverity),
                loggingFunction = function,
            };
            s_loggingFunction = function;
            try
            {
                _ = LoadEnvironment(() => OrtEnv.CreateInstanceWithOptions(ref creation));
            }
            catch (OnnxRuntimeException ex) when (OrtEnv.IsCreated)
            {
                // Another thread created the environment between the check above and this call.
                throw new InvalidOperationException(
                    "The ONNX Runtime environment was created by another caller while logging was being configured, and ONNX Runtime accepts a logging function only when it creates it. " +
                    "Call OnnxRuntimeInfo.ConfigureLogging before any session is created or any other ONNX Runtime call.",
                    ex);
            }
        }
    }

    /// <summary>Whether <see cref="ConfigureLogging"/> set a logging function in this process.</summary>
    internal static bool LoggingConfigured => Volatile.Read(ref s_loggingFunction) is not null;

    /// <summary>The message for a native library without the CUDA provider (the CPU package).</summary>
    internal static string NoCudaProvider(string nativeVersion, IEnumerable<string> available) =>
        $"The loaded ONNX Runtime {nativeVersion} native library has no CUDA execution provider (available: {string.Join(", ", available)}). " +
        $"Reference Microsoft.ML.OnnxRuntime.Gpu.Linux {ManagedVersion} instead of Microsoft.ML.OnnxRuntime.";

    private static OrtEnv LoadEnvironment(Func<OrtEnv> create)
    {
        OrtEnv env;
        try
        {
            env = create();
        }
        catch (Exception ex) when (ex is DllNotFoundException or TypeInitializationException or EntryPointNotFoundException or BadImageFormatException)
        {
            throw new OnnxRuntimeUnavailableException(
                "The ONNX Runtime native library could not be loaded. Reference exactly one native package in the application: " +
                $"Microsoft.ML.OnnxRuntime {ManagedVersion} (CPU) or Microsoft.ML.OnnxRuntime.Gpu.Linux {ManagedVersion} (CUDA, Linux x64).",
                ex);
        }

        string native = env.GetVersionString();
        if (!string.Equals(native, ManagedVersion, StringComparison.Ordinal))
        {
            throw new OnnxRuntimeUnavailableException(
                $"ONNX Runtime native library {native} does not match the managed binding {ManagedVersion}. " +
                "Reference native packages of the same version as Microsoft.ML.OnnxRuntime.Managed.");
        }

        return env;
    }

    private static string ReadManagedVersion()
    {
        var assembly = typeof(InferenceSession).Assembly;
        string? informational = assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
        if (informational is null)
        {
            return assembly.GetName().Version?.ToString(3) ?? "unknown";
        }

        int plus = informational.IndexOf('+', StringComparison.Ordinal);
        return plus < 0 ? informational : informational[..plus];
    }
}
