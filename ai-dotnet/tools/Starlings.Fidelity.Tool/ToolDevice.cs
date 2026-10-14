using System.Globalization;
using Starlings.Runtime.Onnx;

/// <summary>
/// The execution provider an EmbeddingGemma 2 dump runs on: <c>--device cpu</c> (the default) or <c>--device cuda[:N]</c>.
/// The CPU build of the tool (<c>Starlings.Fidelity.Tool</c>, in AI.slnx and CI) carries the CPU native runtime only;
/// <c>cuda</c> needs the CUDA build (<c>ai-dotnet/tools/Starlings.Fidelity.Tool.Cuda</c>: the same sources with
/// <c>Microsoft.ML.OnnxRuntime.Gpu.Linux</c>, which carries the CPU provider too, and its own lock file). Session creation
/// fails when the CUDA runtime or cuDNN is missing; there is no silent CPU fallback (Starlings.Runtime.Onnx), and
/// Starlings.Runtime.Onnx always sets <c>use_tf32=0</c> on CUDA.
/// </summary>
internal readonly record struct ToolDevice(OnnxExecutionProvider Provider, int CudaDeviceId)
{
    public static ToolDevice Cpu => new(OnnxExecutionProvider.Cpu, 0);

    public bool IsCuda => Provider == OnnxExecutionProvider.Cuda;

    /// <summary>The stage environment's <c>provider</c>: <c>cpu</c> or <c>cuda</c>.</summary>
    public string Name => IsCuda ? "cuda" : "cpu";

    /// <summary>The provider as written in the stage results' <c>candidate</c> text.</summary>
    public string Label => IsCuda ? "CUDA" : "CPU";

    /// <summary>Parses <c>cpu</c>, <c>cuda</c> or <c>cuda:N</c> (N: the device index among the visible devices).</summary>
    public static bool TryParse(string value, out ToolDevice device)
    {
        device = Cpu;
        if (value == "cpu")
        {
            return true;
        }

        if (value == "cuda")
        {
            device = new ToolDevice(OnnxExecutionProvider.Cuda, 0);
            return true;
        }

        if (value.StartsWith("cuda:", StringComparison.Ordinal)
            && int.TryParse(value.AsSpan(5), NumberStyles.None, CultureInfo.InvariantCulture, out int id))
        {
            device = new ToolDevice(OnnxExecutionProvider.Cuda, id);
            return true;
        }

        return false;
    }

    /// <summary>Null when the loaded native runtime has this device's provider; otherwise why not.</summary>
    public string? Unavailable()
    {
        if (!IsCuda)
        {
            return null;
        }

        var providers = OnnxRuntimeInfo.AvailableProviders;
        return providers.Contains(OnnxRuntimeInfo.CudaProviderName)
            ? null
            : $"--device cuda: the loaded ONNX Runtime has no CUDA execution provider (available: {string.Join(", ", providers)}). "
              + "This is the CPU build; run the CUDA build: dotnet run --project ai-dotnet/tools/Starlings.Fidelity.Tool.Cuda -c Release -- ...";
    }

    /// <summary>The session options with this device's provider and device id.</summary>
    public OnnxSessionOptions Apply(OnnxSessionOptions options) => options with { Provider = Provider, CudaDeviceId = CudaDeviceId };

    /// <summary>The default profile id: the CPU profile on CPU, its <c>-cuda</c> sibling on CUDA.</summary>
    public string Profile(string cpuProfile) =>
        IsCuda && cpuProfile.EndsWith("-cpu", StringComparison.Ordinal) ? cpuProfile[..^"-cpu".Length] + "-cuda" : cpuProfile;

    /// <summary>Adds the device facts to a stage environment (nothing on CPU, so CPU results are unchanged).</summary>
    public void Describe(IDictionary<string, object?> environment)
    {
        if (IsCuda)
        {
            environment["cuda_device_id"] = CudaDeviceId;
            environment["cuda_visible_devices"] = Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES");
            environment["cuda_device_order"] = Environment.GetEnvironmentVariable("CUDA_DEVICE_ORDER");
        }
    }
}
