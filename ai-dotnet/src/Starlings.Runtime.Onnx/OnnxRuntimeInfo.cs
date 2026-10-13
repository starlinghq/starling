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
    internal static OrtEnv Environment
    {
        get
        {
            OrtEnv env;
            try
            {
                env = OrtEnv.Instance();
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
