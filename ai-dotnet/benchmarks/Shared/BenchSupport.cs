using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using Starlings.Acquisition;
using Starlings.Models.E5;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Benchmarks;

/// <summary>One protected fixture case: ids and reference embedding from <c>embeddings.jsonl</c>, text from <c>tokenizer.jsonl</c>.</summary>
internal sealed record BenchCase(string Id, long[] Ids, string Text, float[] Embedding);

/// <summary>One workload of the shared workload file (<c>ai-dotnet/benchmarks/workloads/e5-v1.json</c>).</summary>
internal sealed record BenchWorkload(string Name, string Input, int Batch, int[] Concurrency, IReadOnlyDictionary<string, int> Samples, BenchCase[][] Requests);

/// <summary>The workload file, resolved against the protected fixtures.</summary>
internal sealed record BenchWorkloadSpec(string Path, int WarmupCalls, BenchCase[] ColdRequest, IReadOnlyList<BenchWorkload> Workloads)
{
    public static string DefaultPath => System.IO.Path.Combine(E5TestAssets.RepositoryRoot, "ai-dotnet", "benchmarks", "workloads", "e5-v1.json");

    public static BenchWorkloadSpec Load(string? path = null)
    {
        path = System.IO.Path.GetFullPath(path ?? DefaultPath);
        var cases = BenchFixtures.Cases;
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var root = doc.RootElement;
        BenchCase[] Resolve(JsonElement ids) => ids.EnumerateArray().Select(e => cases[e.GetString()!]).ToArray();
        var workloads = root.GetProperty("workloads").EnumerateArray().Select(w => new BenchWorkload(
            w.GetProperty("name").GetString()!,
            w.GetProperty("input").GetString()!,
            w.GetProperty("batch").GetInt32(),
            w.GetProperty("concurrency").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
            w.GetProperty("samples").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetInt32(), StringComparer.Ordinal),
            w.GetProperty("requests").EnumerateArray().Select(Resolve).ToArray())).ToList();
        return new BenchWorkloadSpec(
            path,
            root.GetProperty("warmup_calls").GetInt32(),
            Resolve(root.GetProperty("cold").GetProperty("request")),
            workloads);
    }
}

/// <summary>The protected e5 fixtures as benchmark cases (read only).</summary>
internal static class BenchFixtures
{
    private static readonly Lazy<IReadOnlyDictionary<string, BenchCase>> s_cases = new(() =>
        E5Fixtures.Cases.ToDictionary(
            c => c.Id,
            c => new BenchCase(c.Id, c.InputIds, E5Fixtures.Texts[c.Id], c.Embedding),
            StringComparer.Ordinal));

    /// <summary>Case id to case, in <c>embeddings.jsonl</c> order when enumerated through <see cref="Ordered"/>.</summary>
    public static IReadOnlyDictionary<string, BenchCase> Cases => s_cases.Value;

    public static IReadOnlyList<BenchCase> Ordered => E5Fixtures.Cases.Select(c => Cases[c.Id]).ToList();

    public static string FixtureDirectory(OnnxExecutionProvider provider) =>
        provider == OnnxExecutionProvider.Cuda ? E5TestAssets.CudaFixtureDirectory : E5TestAssets.CpuFixtureDirectory;

    /// <summary>Model and tokenizer paths: <c>$STARLING_E5_SNAPSHOT</c>, else the acquisition cache. Never downloads.</summary>
    public static E5ModelFiles Files(string? snapshot)
    {
        if (!string.IsNullOrEmpty(snapshot))
        {
            return E5ModelAssets.OpenDirectory(snapshot, CacheVerification.Size);
        }

        var (files, reason) = E5TestAssets.FindFiles();
        return files ?? throw new InvalidOperationException(reason);
    }
}

/// <summary>Quality of embeddings against the protected reference rows.</summary>
internal readonly record struct Quality(double MaxAbs, double MinCosine, bool Finite)
{
    public static Quality Empty => new(0, 1, true);

    public static Quality Of(IReadOnlyList<float[]> actual, IReadOnlyList<BenchCase> expected)
    {
        double maxAbs = 0;
        double minCos = 1;
        bool finite = true;
        for (int r = 0; r < expected.Count; r++)
        {
            var a = actual[r];
            var e = expected[r].Embedding;
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

/// <summary>Monotonic clock in nanoseconds (CLOCK_MONOTONIC on Linux, the clock Python's <c>time.monotonic_ns</c> reads).</summary>
internal static class MonotonicClock
{
    private static readonly double s_nsPerTick = 1e9 / Stopwatch.Frequency;

    public static long NowNs() => (long)(Stopwatch.GetTimestamp() * s_nsPerTick);

    public static double ElapsedMs(long startTimestamp) => Stopwatch.GetElapsedTime(startTimestamp).TotalMilliseconds;
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

    public static JsonObject Artifacts(E5ModelFiles files, string? modelSha256, string workloadPath, string nativePackage)
    {
        var (commit, dirty) = GitState();
        string root = E5TestAssets.RepositoryRoot;
        return new JsonObject
        {
            ["model_path"] = files.ModelPath,
            ["model_sha256"] = modelSha256 ?? Sha256File(files.ModelPath),
            ["tokenizer_sha256"] = Sha256File(files.TokenizerPath),
            ["workload_file"] = Path.GetRelativePath(root, workloadPath).Replace('\\', '/'),
            ["workload_sha256"] = Sha256File(workloadPath),
            ["fixture_sha256"] = new JsonObject
            {
                ["embeddings.jsonl"] = Sha256File(Path.Combine(E5TestAssets.CpuFixtureDirectory, "embeddings.jsonl")),
                ["tokenizer.jsonl"] = Sha256File(Path.Combine(E5TestAssets.CpuFixtureDirectory, "tokenizer.jsonl")),
            },
            ["git_commit"] = commit,
            ["git_dirty"] = dirty,
            ["packages"] = new JsonObject
            {
                ["Microsoft.ML.OnnxRuntime.Managed"] = OnnxRuntimeInfo.ManagedVersion,
                [nativePackage] = OnnxRuntimeInfo.NativeVersion,
                ["Starlings.Runtime.Onnx"] = InformationalVersion(typeof(OnnxModelSession).Assembly),
                ["Starlings.Models.E5"] = InformationalVersion(typeof(E5Encoder).Assembly),
                ["Starlings.Tokenizers"] = InformationalVersion(typeof(Starlings.Tokenizers.Tokenizer).Assembly),
            },
        };
    }

    public static JsonObject Environment(bool cuda, IReadOnlyList<string>? sessionProviders, JsonObject loadBefore, string? note)
    {
        var (model, flags, physical) = CpuInfo();
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
            ["ort_version"] = OnnxRuntimeInfo.NativeVersion,
            ["ort_available_providers"] = new JsonArray(OnnxRuntimeInfo.AvailableProviders.Select(p => (JsonNode)p).ToArray()),
            ["session_providers"] = sessionProviders is null ? null : new JsonArray(sessionProviders.Select(p => (JsonNode)p).ToArray()),
            ["gpu"] = cuda ? GpuIdentity() : null,
            ["native_libraries"] = new JsonArray(LoadedNativeLibraries().Select(p => (JsonNode)p).ToArray()),
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
        string root = E5TestAssets.RepositoryRoot;
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
