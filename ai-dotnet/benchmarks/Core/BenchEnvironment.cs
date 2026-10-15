using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json.Nodes;
using Starlings.Runtime.Onnx;

namespace Starlings.Benchmarks;

/// <summary>Repository paths for the harnesses.</summary>
internal static class BenchPaths
{
    public static string RepositoryRoot { get; } = FindRepositoryRoot();

    public static string Workload(string fileName) => Path.Combine(RepositoryRoot, "ai-dotnet", "benchmarks", "workloads", fileName);

    private static string FindRepositoryRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "model-pipeline", "protected")) &&
                Directory.Exists(Path.Combine(dir.FullName, "ai-dotnet")))
            {
                return dir.FullName;
            }
        }

        throw new InvalidOperationException("Repository root not found above " + AppContext.BaseDirectory);
    }
}

/// <summary>Quality of embeddings against reference rows (max abs difference, min cosine, all finite).</summary>
internal readonly record struct Quality(double MaxAbs, double MinCosine, bool Finite)
{
    public static Quality Empty => new(0, 1, true);

    public static Quality Of(IReadOnlyList<float[]> actual, IReadOnlyList<float[]> expected)
    {
        double maxAbs = 0;
        double minCos = 1;
        bool finite = true;
        for (int r = 0; r < expected.Count; r++)
        {
            var a = actual[r];
            var e = expected[r];
            double dot = 0, na = 0, ne = 0;
            for (int i = 0; i < e.Length; i++)
            {
                finite &= float.IsFinite(a[i]);
                maxAbs = Math.Max(maxAbs, Math.Abs((double)a[i] - e[i]));
                dot += (double)a[i] * e[i];
                na += (double)a[i] * a[i];
                ne += (double)e[i] * e[i];
            }

            minCos = Math.Min(minCos, dot / (Math.Sqrt(na) * Math.Sqrt(ne)));
        }

        return new Quality(maxAbs, minCos, finite);
    }

    public Quality Merge(Quality other) => new(Math.Max(MaxAbs, other.MaxAbs), Math.Min(MinCosine, other.MinCosine), Finite && other.Finite);

    public JsonObject ToJson() => new() { ["max_abs"] = MaxAbs, ["min_cosine"] = MinCosine, ["finite"] = Finite };
}

/// <summary>Artifact identity and environment blocks of the raw result format <c>starlings-bench-raw/1</c>.</summary>
internal static class BenchEnvironment
{
    public const string Schema = "starlings-bench-raw/1";

    private static readonly string[] s_cpuFlags =
    [
        "sse4_2", "avx", "avx2", "fma", "f16c", "avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512_bf16",
        "avx_vnni", "amx_tile", "amx_bf16", "amx_int8",
    ];

    public static string Sha256File(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }

    public static JsonObject LoadSnapshot()
    {
        var parts = Read("/proc/loadavg")?.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        return new JsonObject
        {
            ["loadavg"] = parts is { Length: >= 3 }
                ? new JsonArray(parts.Take(3).Select(p => (JsonNode)double.Parse(p, CultureInfo.InvariantCulture)).ToArray())
                : null,
            ["gpus"] = GpuLoad(),
        };
    }

    /// <summary>
    /// The artifact block of a run record: model and tokenizer digests, workload and fixture digests, the commit, and
    /// the package versions (<paramref name="packages"/>, usually <see cref="CommonPackages"/> plus the model package).
    /// </summary>
    public static JsonObject Artifacts(string modelPath, string? modelSha256, string tokenizerPath, string workloadPath, JsonObject fixtureSha256, JsonObject packages)
    {
        var (commit, dirty) = GitState();
        return new JsonObject
        {
            ["model_path"] = modelPath,
            ["model_sha256"] = modelSha256 ?? Sha256File(modelPath),
            ["tokenizer_sha256"] = Sha256File(tokenizerPath),
            ["workload_file"] = RepoRelative(workloadPath),
            ["workload_sha256"] = Sha256File(workloadPath),
            ["fixture_sha256"] = fixtureSha256,
            ["git_commit"] = commit,
            ["git_dirty"] = dirty,
            ["packages"] = packages,
        };
    }

    /// <summary>ONNX Runtime (managed and the native package), the ONNX runtime layer and the owned tokenizer.</summary>
    public static JsonObject CommonPackages(string nativePackage) => new()
    {
        ["Microsoft.ML.OnnxRuntime.Managed"] = OnnxRuntimeInfo.ManagedVersion,
        [nativePackage] = OnnxRuntimeInfo.NativeVersion,
        ["Starlings.Runtime.Onnx"] = InformationalVersion(typeof(OnnxModelSession).Assembly),
        ["Starlings.Tokenizers"] = TokenizerVersion,
    };

    /// <summary>The owned tokenizer package version in this process.</summary>
    public static string? TokenizerVersion => InformationalVersion(typeof(Starlings.Tokenizers.Tokenizer).Assembly);

    /// <summary>SHA-256 of each file, keyed by its repository-relative path.</summary>
    public static JsonObject FileDigests(params string[] paths)
    {
        var result = new JsonObject();
        foreach (string path in paths)
        {
            result[RepoRelative(path)] = Sha256File(path);
        }

        return result;
    }

    public static string RepoRelative(string path) => Path.GetRelativePath(BenchPaths.RepositoryRoot, path).Replace('\\', '/');

    public static JsonObject Environment(bool cuda, IReadOnlyList<string>? sessionProviders, JsonObject loadBefore, string? note) =>
        Environment(cuda, sessionProviders, loadBefore, note, managedOrt: true);

    /// <summary>
    /// The environment block. With <paramref name="managedOrt"/> false the managed ONNX Runtime environment is never
    /// created (a process whose native runtime is driven by ONNX Runtime GenAI): <c>ort_version</c> is then the
    /// <c>Microsoft.ML.OnnxRuntime</c> package version from the dependency manifest and the provider list is empty.
    /// </summary>
    public static JsonObject Environment(bool cuda, IReadOnlyList<string>? sessionProviders, JsonObject loadBefore, string? note, bool managedOrt)
    {
        var (model, flags, physical) = CpuInfo();
        string? ortVersion = managedOrt ? OnnxRuntimeInfo.NativeVersion : PackageVersion("Microsoft.ML.OnnxRuntime");
        IReadOnlyList<string> providers = managedOrt ? OnnxRuntimeInfo.AvailableProviders : [];
        return new JsonObject
        {
            ["cpu_model"] = model,
            ["cpu_flags"] = new JsonArray(s_cpuFlags.Where(flags.Contains).Select(f => (JsonNode)f).ToArray()),
            ["physical_cores"] = physical,
            ["logical_cpus"] = System.Environment.ProcessorCount,
            ["ram_bytes"] = RamBytes(),
            ["os"] = OsName(),
            ["kernel"] = Read("/proc/sys/kernel/osrelease") ?? RuntimeInformation.OSDescription,
            ["runtime"] = RuntimeInformation.FrameworkDescription,
            ["runtime_details"] = new JsonObject
            {
                ["rid"] = RuntimeInformation.RuntimeIdentifier,
                ["server_gc"] = GCSettings.IsServerGC,
                ["gc_latency_mode"] = GCSettings.LatencyMode.ToString(),
                ["tiered_pgo"] = System.Environment.GetEnvironmentVariable("DOTNET_TieredPGO"),
                ["tiered_compilation"] = System.Environment.GetEnvironmentVariable("DOTNET_TieredCompilation"),
            },
            ["ort_version"] = ortVersion,
            ["ort_version_source"] = managedOrt ? "native library (OrtGetVersionString)" : "package version in the dependency manifest",
            ["ort_available_providers"] = new JsonArray(providers.Select(p => (JsonNode)p).ToArray()),
            ["session_providers"] = sessionProviders is null ? null : new JsonArray(sessionProviders.Select(p => (JsonNode)p).ToArray()),
            ["gpu"] = cuda ? GpuIdentity() : null,
            ["native_libraries"] = new JsonArray(LoadedNativeLibraries().Select(p => (JsonNode)p).ToArray()),
            ["native_library_digests"] = NativeLibraryDigests.Collect(),
            ["cuda_visible_devices"] = System.Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES"),
            ["cuda_device_order"] = System.Environment.GetEnvironmentVariable("CUDA_DEVICE_ORDER"),
            ["cpu_governor"] = Read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
            ["energy_performance_preference"] = Read("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference"),
            ["no_turbo"] = Read("/sys/devices/system/cpu/intel_pstate/no_turbo"),
            ["load_before"] = loadBefore,
            ["load_after"] = LoadSnapshot(),
            ["machine_note"] = note,
        };
    }

    /// <summary>The version of a package in this application's dependency manifest (<c>*.deps.json</c>), or null.</summary>
    public static string? PackageVersion(string packageId)
    {
        if (AppContext.GetData("APP_CONTEXT_DEPS_FILES") is not string files)
        {
            return null;
        }

        foreach (string path in files.Split(';', StringSplitOptions.RemoveEmptyEntries).Where(File.Exists))
        {
            using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllBytes(path));
            if (!doc.RootElement.TryGetProperty("libraries", out var libraries))
            {
                continue;
            }

            foreach (var library in libraries.EnumerateObject())
            {
                if (library.Name.StartsWith(packageId + "/", StringComparison.OrdinalIgnoreCase))
                {
                    return library.Name[(packageId.Length + 1)..];
                }
            }
        }

        return null;
    }

    public static string? InformationalVersion(Assembly assembly) =>
        assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;

    private static (string? Model, HashSet<string> Flags, int? Physical) CpuInfo()
    {
        string? model = null;
        var flags = new HashSet<string>(StringComparer.Ordinal);
        var cores = new HashSet<string>(StringComparer.Ordinal);
        string? phys = null;
        foreach (var line in (Read("/proc/cpuinfo") ?? string.Empty).Split('\n'))
        {
            int colon = line.IndexOf(':', StringComparison.Ordinal);
            if (colon < 0)
            {
                continue;
            }

            string key = line[..colon].Trim();
            string value = line[(colon + 1)..].Trim();
            switch (key)
            {
                case "model name" when model is null:
                    model = value;
                    break;
                case "flags" when flags.Count == 0:
                    flags.UnionWith(value.Split(' ', StringSplitOptions.RemoveEmptyEntries));
                    break;
                case "physical id":
                    phys = value;
                    break;
                case "core id":
                    cores.Add(phys + "/" + value);
                    break;
                default:
                    break;
            }
        }

        return (model, flags, cores.Count == 0 ? null : cores.Count);
    }

    private static long? RamBytes()
    {
        var line = (Read("/proc/meminfo") ?? string.Empty).Split('\n').FirstOrDefault(l => l.StartsWith("MemTotal:", StringComparison.Ordinal));
        return line is null ? null : long.Parse(line.Split(' ', StringSplitOptions.RemoveEmptyEntries)[1], CultureInfo.InvariantCulture) * 1024;
    }

    private static string OsName()
    {
        foreach (var line in (Read("/etc/os-release") ?? string.Empty).Split('\n'))
        {
            if (line.StartsWith("PRETTY_NAME=", StringComparison.Ordinal))
            {
                return line["PRETTY_NAME=".Length..].Trim('"');
            }
        }

        return RuntimeInformation.OSDescription;
    }

    private static List<string> LoadedNativeLibraries()
    {
        string[] names = ["libonnxruntime", "libcudart", "libcublas", "libcudnn", "libcufft", "libcurand"];
        return (Read("/proc/self/maps") ?? string.Empty).Split('\n')
            .Select(l => l.Split(' ', StringSplitOptions.RemoveEmptyEntries).LastOrDefault() ?? string.Empty)
            .Where(p => p.StartsWith('/') && names.Any(n => p.Contains(n, StringComparison.Ordinal)))
            .Distinct(StringComparer.Ordinal)
            .Order(StringComparer.Ordinal)
            .ToList();
    }

    private static JsonArray? GpuLoad()
    {
        string? output = RunTool("nvidia-smi", "--query-gpu=index,name,pci.bus_id,utilization.gpu,memory.used,memory.total,clocks.sm,pstate,power.draw,temperature.gpu --format=csv,noheader,nounits");
        if (output is null)
        {
            return null;
        }

        string[] keys = ["index", "name", "pci_bus_id", "util_pct", "mem_used_mib", "mem_total_mib", "sm_clock_mhz", "pstate", "power_w", "temp_c"];
        return new JsonArray(output.Split('\n', StringSplitOptions.RemoveEmptyEntries).Select(line => (JsonNode)ToObject(keys, line)).ToArray());
    }

    private static JsonObject? GpuIdentity()
    {
        string? visible = System.Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES")?.Trim();
        if (visible is null || !visible.All(char.IsAsciiDigit) || visible.Length == 0)
        {
            return null;
        }

        string? output = RunTool("nvidia-smi", $"-i {visible} --query-gpu=index,name,pci.bus_id,driver_version,memory.total,compute_cap --format=csv,noheader,nounits");
        if (string.IsNullOrWhiteSpace(output))
        {
            return null;
        }

        var gpu = ToObject(["index", "name", "pci_bus_id", "driver_version", "memory_total_mib", "compute_capability"], output.Split('\n')[0]);
        string header = RunTool("nvidia-smi", string.Empty) ?? string.Empty;
        // "CUDA Version: x.y" in older drivers, "CUDA UMD Version: x.y" in newer ones.
        var match = System.Text.RegularExpressions.Regex.Match(header, @"CUDA (?:UMD )?Version:\s*([0-9.]+)");
        if (match.Success)
        {
            gpu["cuda_driver_api_version"] = match.Groups[1].Value;
        }

        return gpu;
    }

    private static JsonObject ToObject(string[] keys, string csvLine)
    {
        var values = csvLine.Split(',');
        var obj = new JsonObject();
        for (int i = 0; i < keys.Length && i < values.Length; i++)
        {
            obj[keys[i]] = values[i].Trim();
        }

        return obj;
    }

    private static (string? Commit, bool? Dirty) GitState()
    {
        string root = BenchPaths.RepositoryRoot;
        string? commit = RunTool("git", $"-C \"{root}\" rev-parse HEAD")?.Trim();
        string? status = RunTool("git", $"-C \"{root}\" status --porcelain --untracked-files=no");
        return (commit, status is null ? null : status.Trim().Length > 0);
    }

    private static string? RunTool(string file, string arguments)
    {
        try
        {
            using var process = Process.Start(new ProcessStartInfo(file, arguments)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            });
            if (process is null)
            {
                return null;
            }

            string output = process.StandardOutput.ReadToEnd();
            process.WaitForExit(30_000);
            return process.ExitCode == 0 ? output : null;
        }
        catch (System.ComponentModel.Win32Exception)
        {
            return null;
        }
    }

    private static string? Read(string path)
    {
        try
        {
            return File.ReadAllText(path).Trim();
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
