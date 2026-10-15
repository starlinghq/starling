using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime;
using System.Runtime.InteropServices;
using System.Text.Json.Nodes;
using Starlings.Generation.OnnxGenAI;
using Starlings.Runtime.Onnx;

/// <summary>Environment manifest of a soak run (machine, runtime, native packages, source state, machine load).</summary>
internal static class SoakEnvironment
{
    public static JsonObject Load()
    {
        var parts = Read("/proc/loadavg")?.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var mem = MemInfo();
        return new JsonObject
        {
            ["utc"] = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture),
            ["loadavg"] = parts is { Length: >= 3 }
                ? new JsonArray(parts.Take(3).Select(p => (JsonNode)double.Parse(p, CultureInfo.InvariantCulture)).ToArray())
                : null,
            ["mem_available_mb"] = mem.Available,
            ["gpus"] = MemoryProbe.RunNvidiaSmi("--query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader"),
        };
    }

    public static JsonObject Collect(JsonObject loadBefore, string? note)
    {
        var (commit, dirty) = GitState();
        var mem = MemInfo();
        return new JsonObject
        {
            ["cpu_model"] = CpuModel(),
            ["logical_cpus"] = Environment.ProcessorCount,
            ["ram_total_mb"] = mem.Total,
            ["os"] = OsName(),
            ["kernel"] = Read("/proc/sys/kernel/osrelease") ?? RuntimeInformation.OSDescription,
            ["rid"] = RuntimeInformation.RuntimeIdentifier,
            ["runtime"] = RuntimeInformation.FrameworkDescription,
            ["server_gc"] = GCSettings.IsServerGC,
            ["concurrent_gc"] = AppContext.TryGetSwitch("System.GC.Concurrent", out bool concurrent) ? concurrent : null,
            ["gc_latency_mode"] = GCSettings.LatencyMode.ToString(),
            ["ort_version"] = OnnxRuntimeInfo.NativeVersion,
            ["ort_available_providers"] = new JsonArray(OnnxRuntimeInfo.AvailableProviders.Select(p => (JsonNode)p).ToArray()),
            ["genai_version"] = GenAIRuntimeInfo.ManagedVersion,
            ["genai_native_package"] = GenAIRuntimeInfo.NativePackage,
            ["packages"] = new JsonObject
            {
                ["Starlings.Runtime.Onnx"] = Version(typeof(OnnxModelSession).Assembly),
                ["Starlings.Generation.OnnxGenAI"] = Version(typeof(GenAIModel).Assembly),
                ["Starlings.Tokenizers"] = Version(typeof(Starlings.Tokenizers.Tokenizer).Assembly),
            },
            ["gpu_identity"] = MemoryProbe.RunNvidiaSmi("--query-gpu=index,name,driver_version,memory.total --format=csv,noheader"),
            ["cuda_visible_devices"] = Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES"),
            ["allocator_and_runtime_env"] = new JsonObject(Environment.GetEnvironmentVariables().Keys.Cast<string>()
                .Where(k => k.StartsWith("MALLOC_", StringComparison.Ordinal) || k == "GLIBC_TUNABLES" || k.StartsWith("DOTNET_", StringComparison.Ordinal))
                .Where(k => k != "DOTNET_ROOT")
                .Order(StringComparer.Ordinal)
                .Select(k => KeyValuePair.Create(k, (JsonNode?)Environment.GetEnvironmentVariable(k)))),
            ["cpu_governor"] = Read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
            ["transparent_hugepage"] = Read("/sys/kernel/mm/transparent_hugepage/enabled"),
            ["transparent_hugepage_defrag"] = Read("/sys/kernel/mm/transparent_hugepage/defrag"),
            ["git_commit"] = commit,
            ["git_dirty"] = dirty,
            ["command_line"] = string.Join(' ', Environment.GetCommandLineArgs().Skip(1)),
            // Collect runs at every checkpoint; a JsonNode can have one parent, so attach a copy.
            ["load_before"] = loadBefore.DeepClone(),
            ["load_after"] = Load(),
            ["machine_note"] = note,
        };
    }

    private static string? Version(Assembly assembly) => assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;

    private static string? CpuModel() =>
        Read("/proc/cpuinfo")?.Split('\n').FirstOrDefault(l => l.StartsWith("model name", StringComparison.Ordinal))?.Split(':', 2)[1].Trim()
        ?? RuntimeInformation.ProcessArchitecture.ToString();

    private static (double? Total, double? Available) MemInfo()
    {
        double? total = null;
        double? available = null;
        foreach (string line in (Read("/proc/meminfo") ?? string.Empty).Split('\n'))
        {
            var p = line.Split(':', 2);
            if (p.Length == 2 && double.TryParse(p[1].Trim().Split(' ')[0], CultureInfo.InvariantCulture, out double kb))
            {
                if (p[0] == "MemTotal")
                {
                    total = Math.Round(kb / 1024);
                }
                else if (p[0] == "MemAvailable")
                {
                    available = Math.Round(kb / 1024);
                }
            }
        }

        return (total, available);
    }

    private static string OsName()
    {
        foreach (string line in (Read("/etc/os-release") ?? string.Empty).Split('\n'))
        {
            if (line.StartsWith("PRETTY_NAME=", StringComparison.Ordinal))
            {
                return line["PRETTY_NAME=".Length..].Trim('"');
            }
        }

        return RuntimeInformation.OSDescription;
    }

    private static (string? Commit, bool? Dirty) GitState()
    {
        try
        {
            string head = Git("rev-parse HEAD");
            string status = Git("status --porcelain --untracked-files=no");
            return head.Length == 0 ? (null, null) : (head, status.Length > 0);
        }
        catch (System.ComponentModel.Win32Exception)
        {
            return (null, null);
        }

        static string Git(string arguments)
        {
            var info = new ProcessStartInfo("git", arguments) { RedirectStandardOutput = true, RedirectStandardError = true, WorkingDirectory = AppContext.BaseDirectory };
            using var process = Process.Start(info)!;
            string output = process.StandardOutput.ReadToEnd().Trim();
            process.WaitForExit();
            return process.ExitCode == 0 ? output : string.Empty;
        }
    }

    private static string? Read(string path)
    {
        try
        {
            return File.Exists(path) ? File.ReadAllText(path).Trim() : null;
        }
        catch (IOException)
        {
            return null;
        }
        catch (UnauthorizedAccessException)
        {
            return null;
        }
    }
}
