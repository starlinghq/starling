using System.Diagnostics;
using System.Globalization;

namespace Starlings.Benchmarks;

/// <summary>Chunk and padding arithmetic of the batching sweep.</summary>
internal static class SweepMath
{
    /// <summary>Padded cells (rows x longest row per ORT run) and the number of runs for the target's chunk plan.</summary>
    internal static (long Cells, int Runs) Padding(IReadOnlyList<int> lengths, IEnumerable<int> chunks)
    {
        long cells = 0;
        int start = 0, runs = 0;
        foreach (int count in chunks)
        {
            int width = 0;
            for (int k = start; k < start + count; k++)
            {
                width = Math.Max(width, lengths[k]);
            }

            cells += (long)count * width;
            start += count;
            runs++;
        }

        return (cells, runs);
    }

    /// <summary>
    /// Padded cells and runs of the libraries' explicit-batch chunk plan (Starlings.Batching.BatchPlan, internal to each
    /// model assembly; mirrored here): rows in input order, or by ascending length (stable) with
    /// <paramref name="sortByLength"/>, grouped greedily while a chunk has fewer than <paramref name="maxRows"/> rows and,
    /// with a budget, rows x longest row stays within <paramref name="maxCells"/>. Every chunk has at least one row.
    /// </summary>
    internal static (long Cells, int Runs) Plan(IReadOnlyList<int> lengths, int maxRows, int? maxCells, bool sortByLength)
    {
        IEnumerable<int> ordered = sortByLength ? lengths.Order() : lengths;
        long cells = 0;
        int runs = 0, count = 0, width = 0;
        foreach (int length in ordered)
        {
            int newWidth = Math.Max(width, length);
            if (count > 0 && (count >= maxRows || (maxCells is int budget && (long)(count + 1) * newWidth > budget)))
            {
                cells += (long)count * width;
                runs++;
                count = 0;
                newWidth = length;
            }

            count++;
            width = newWidth;
        }

        if (count > 0)
        {
            cells += (long)count * width;
            runs++;
        }

        return (cells, runs);
    }

    /// <summary>Consecutive chunks of at most <paramref name="maxRows"/> rows (the plan without sorting or a budget).</summary>
    internal static IEnumerable<int> FixedChunks(int count, int maxRows)
    {
        for (int start = 0; start < count; start += maxRows)
        {
            yield return Math.Min(maxRows, count - start);
        }
    }
}

/// <summary>Options of the sweep command.</summary>
internal sealed record SweepOptions
{
    public required string Out { get; init; }

    public string[] Sets { get; init; } = ["short", "long", "mixed"];

    public int[] Batches { get; init; } = [1, 4, 8, 16, 32, 64];

    public int[] Concurrency { get; init; } = [1, 4, 16];

    public double MinSeconds { get; init; } = 2;

    public int MinCalls { get; init; } = 10;

    public int MaxCalls { get; init; } = 5000;

    public int Warmup { get; init; } = 3;

    public int? Pool { get; init; }

    /// <summary>Rows per ORT run for the library: null for its default.</summary>
    public int? MaxBatchSize { get; init; }

    public string MaxBatchSizeLabel { get; init; } = "call";

    public bool Preliminary { get; init; } = true;

    public bool Vram { get; init; } = true;

    /// <summary>--max-batch-tokens: the library's padded-cell budget per ORT run (MaxBatchTokens); null for its default.</summary>
    public int? MaxBatchTokens { get; init; }

    /// <summary>--sort-by-length: the library's SortByLength (length-sorted chunking within one call).</summary>
    public bool SortByLength { get; init; }

    /// <summary>--microbatch-wait-ms: MicroBatchingOptions.MaxWait in milliseconds; any micro-batch flag turns micro-batching on.</summary>
    public double? MicroBatchWaitMs { get; init; }

    /// <summary>--microbatch-max-rows: MicroBatchingOptions.MaxRows.</summary>
    public int? MicroBatchMaxRows { get; init; }

    /// <summary>--microbatch-max-tokens: MicroBatchingOptions.MaxTokens (padded cells per dispatched batch).</summary>
    public int? MicroBatchMaxTokens { get; init; }

    /// <summary>
    /// Rows per dispatched micro-batch when --microbatch-max-rows is absent: MicroBatchingOptions.MaxRows's default,
    /// mirrored so the harness core does not depend on Starlings.Runtime.Onnx (the targets pass this value explicitly).
    /// </summary>
    public const int DefaultMicroBatchMaxRows = 8;

    /// <summary>
    /// Micro-batching on: each call's items are issued as one-item calls through the product's micro-batched entry
    /// point (e5 and EmbeddingGemma 2 IEmbeddingGenerator.GenerateAsync with one value, GLiNER2 ExtractAsync), so the
    /// library's batcher, not the caller, forms the batches. Compare with batch 1 at the same concurrency without it.
    /// </summary>
    public bool MicroBatching => MicroBatchWaitMs is not null || MicroBatchMaxRows is not null || MicroBatchMaxTokens is not null;

    /// <summary>Rows per dispatched micro-batch in effect.</summary>
    public int MicroBatchRows => MicroBatchMaxRows ?? DefaultMicroBatchMaxRows;

    /// <summary>The batching configuration's label (summary tables are grouped by it): baseline, sorted, tokens2048, micro-w0-r8, ...</summary>
    public string Config
    {
        get
        {
            var parts = new List<string>();
            if (SortByLength)
            {
                parts.Add("sorted");
            }

            if (MaxBatchTokens is int t)
            {
                parts.Add(string.Create(CultureInfo.InvariantCulture, $"tokens{t}"));
            }

            if (MicroBatching)
            {
                string micro = string.Create(CultureInfo.InvariantCulture, $"micro-w{MicroBatchWaitMs ?? 0:0.##}-r{MicroBatchRows}");
                parts.Add(MicroBatchMaxTokens is int mt ? string.Create(CultureInfo.InvariantCulture, $"{micro}-t{mt}") : micro);
            }

            return parts.Count == 0 ? "baseline" : string.Join('+', parts);
        }
    }

    /// <summary>--max-inflight-items: cells with batch x concurrency above this are skipped (recorded in the end record).</summary>
    public int? MaxInflightItems { get; init; }

    /// <summary>
    /// The option-contract hooks. A driver lists the ones it maps to library options (ISweepDriver.WiredSweepHooks);
    /// <see cref="Parse"/> rejects the others.
    /// </summary>
    public static readonly (string Flag, string Contract)[] Hooks =
    [
        ("sort-by-length", "SortByLength (E5EncoderOptions, Gliner2ExtractorOptions, EmbeddingGemma2EncoderOptions)"),
        ("max-batch-tokens", "MaxBatchTokens (E5EncoderOptions, Gliner2ExtractorOptions, EmbeddingGemma2EncoderOptions)"),
        ("microbatch-wait-ms", "MicroBatching.MaxWait (Starlings.Runtime.Onnx.Batching.MicroBatchingOptions)"),
        ("microbatch-max-rows", "MicroBatching.MaxRows (Starlings.Runtime.Onnx.Batching.MicroBatchingOptions)"),
        ("microbatch-max-tokens", "MicroBatching.MaxTokens (Starlings.Runtime.Onnx.Batching.MicroBatchingOptions)"),
    ];

    /// <summary>Every hook flag: the drivers that map all of them (e5, GLiNER2, EmbeddingGemma 2) return this.</summary>
    public static IReadOnlyCollection<string> AllHooks { get; } = Hooks.Select(h => h.Flag).ToArray();

    /// <summary>
    /// Parses the sweep options. <paramref name="wiredHooks"/> names the hook flags this target already maps to an
    /// existing option; every other hook flag is rejected with the option it will map to.
    /// </summary>
    public static SweepOptions Parse(ArgMap map, IReadOnlyCollection<string>? wiredHooks = null)
    {
        foreach (var (flag, contract) in Hooks)
        {
            if (map.Get(flag) is not null && !(wiredHooks?.Contains(flag) ?? false))
            {
                throw new ArgumentException(
                    $"--{flag} is a hook of the batching option contract ({contract}) and is not wired in this build yet: " +
                    "the integrate step maps it to the option once the library has it. Rerun without it.");
            }
        }

        bool quick = map.Get("quick", "false") == "true";
        int[] Ints(string key, int[] fallback) => map.GetList(key)?.Select(s => int.Parse(s, CultureInfo.InvariantCulture)).ToArray() ?? fallback;
        var batches = Ints("batches", quick ? [1, 8, 32] : [1, 4, 8, 16, 32, 64]);
        var concurrency = Ints("concurrency", quick ? [1, 4] : [1, 4, 16]);
        if (batches.Length == 0 || batches.Any(b => b < 1) || concurrency.Length == 0 || concurrency.Any(c => c < 1))
        {
            throw new ArgumentException("--batches and --concurrency take positive integers.");
        }

        int? microRows = map.GetInt("microbatch-max-rows");
        int? microTokens = map.GetInt("microbatch-max-tokens");
        string? waitText = map.Get("microbatch-wait-ms");
        double? wait = waitText is null ? null : double.Parse(waitText, NumberStyles.Float, CultureInfo.InvariantCulture);
        if (microRows is < 1 || microTokens is < 1 || wait is < 0 || (wait is double w && !double.IsFinite(w)))
        {
            throw new ArgumentException("--microbatch-max-rows and --microbatch-max-tokens take positive integers, --microbatch-wait-ms a number >= 0.");
        }

        bool micro = wait is not null || microRows is not null || microTokens is not null;
        string sort = map.Get("sort-by-length", "false")!;
        if (sort is not ("true" or "false"))
        {
            throw new ArgumentException("--sort-by-length true|false.");
        }

        // "call": one ORT run per call or, with micro-batching, one ORT run per dispatched batch (whose rows can exceed
        // the caller's batch).
        string mbs = map.Get("max-batch-size", "call");
        int callRows = micro ? Math.Max(batches.Max(), microRows ?? DefaultMicroBatchMaxRows) : batches.Max();
        int? maxBatch = mbs switch
        {
            "default" => null,
            "call" => callRows,
            _ => int.TryParse(mbs, NumberStyles.None, CultureInfo.InvariantCulture, out int v) && v > 0 ? v : throw new ArgumentException("--max-batch-size default|call|N."),
        };
        string? output = map.Get("out");
        if (string.IsNullOrEmpty(output))
        {
            throw new ArgumentException("--out <sweep.jsonl> is required.");
        }

        int? tokens = map.GetInt("max-batch-tokens");
        if (tokens is < 1)
        {
            throw new ArgumentException("--max-batch-tokens takes a positive integer.");
        }

        return new SweepOptions
        {
            Out = output,
            Sets = map.GetList("sets") ?? ["short", "long", "mixed"],
            Batches = batches,
            Concurrency = concurrency,
            MinSeconds = map.GetDouble("min-seconds", quick ? 1 : 2),
            MinCalls = map.GetInt("min-calls", quick ? 5 : 10)!.Value,
            MaxCalls = map.GetInt("max-calls", 5000)!.Value,
            Warmup = map.GetInt("warmup", quick ? 2 : 3)!.Value,
            Pool = map.GetInt("pool", quick ? 32 : null),
            MaxBatchSize = maxBatch,
            MaxBatchSizeLabel = mbs == "call" ? $"call ({callRows})" : mbs,
            Preliminary = map.Get("preliminary", "true") != "false",
            Vram = map.Get("vram", "auto") != "off",
            MaxBatchTokens = tokens,
            SortByLength = sort == "true",
            MicroBatchWaitMs = wait,
            MicroBatchMaxRows = microRows,
            MicroBatchMaxTokens = microTokens,
            MaxInflightItems = map.GetInt("max-inflight-items"),
        };
    }
}

/// <summary>
/// Counters of the library's micro-batcher (Starlings.Runtime.Onnx.Batching.MicroBatcherStatistics, copied by each target
/// so the harness core does not depend on the runtime package).
/// </summary>
internal readonly record struct SweepBatcherStats(long Batches, long Items, long TokenCells, long Tokens, double QueueMs, double RunMs)
{
    public SweepBatcherStats Since(SweepBatcherStats b) =>
        new(Batches - b.Batches, Items - b.Items, TokenCells - b.TokenCells, Tokens - b.Tokens, QueueMs - b.QueueMs, RunMs - b.RunMs);
}

/// <summary>Result of comparing outputs with single-call references: worst difference, finiteness, discrete mismatches.</summary>
internal readonly record struct SweepCheck(long Items, double MaxDiff, bool Finite, int Mismatches, string? Worst)
{
    public static SweepCheck Empty => new(0, 0, true, 0, null);

    public SweepCheck Merge(SweepCheck other) => new(
        Items + other.Items,
        Math.Max(MaxDiff, other.MaxDiff),
        Finite && other.Finite,
        Mismatches + other.Mismatches,
        other.MaxDiff > MaxDiff ? other.Worst : Worst ?? other.Worst);

    /// <summary>Embeddings: max abs difference of each row with its single-call embedding.</summary>
    public static SweepCheck Embeddings(SweepItem[] items, float[][] output, IReadOnlyDictionary<string, object> singles)
    {
        double max = 0;
        bool finite = true;
        string? worst = null;
        for (int r = 0; r < items.Length; r++)
        {
            var a = output[r];
            var e = (float[])singles[items[r].Id];
            for (int i = 0; i < e.Length; i++)
            {
                finite &= float.IsFinite(a[i]);
                double d = Math.Abs((double)a[i] - e[i]);
                if (d > max)
                {
                    max = d;
                    worst = items[r].Id;
                }
            }
        }

        return new SweepCheck(items.Length, max, finite, 0, worst);
    }
}

/// <summary>Peak resident memory of this process (Linux: VmHWM, reset through <c>/proc/self/clear_refs</c>).</summary>
internal static class ProcessMemory
{
    /// <summary>Resets the peak to the current RSS (when the kernel allows it) and returns the current RSS in bytes.</summary>
    public static long ResetPeak()
    {
        try
        {
            File.WriteAllText("/proc/self/clear_refs", "5");
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            // Not resettable here: the peak is then the process's lifetime peak.
        }

        return Read("VmRSS:");
    }

    public static long PeakBytes() => Read("VmHWM:");

    private static long Read(string key)
    {
        try
        {
            foreach (string line in File.ReadLines("/proc/self/status"))
            {
                if (line.StartsWith(key, StringComparison.Ordinal))
                {
                    var parts = line[key.Length..].Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    return long.Parse(parts[0], CultureInfo.InvariantCulture) * 1024;
                }
            }
        }
        catch (IOException)
        {
        }

        return Environment.WorkingSet;
    }
}

/// <summary>
/// Device memory in use (MiB), sampled every 100 ms by one long-running <c>nvidia-smi -lms 100</c> process on the first
/// device of <c>CUDA_VISIBLE_DEVICES</c> (whole GPU, every process). Null when nvidia-smi is unavailable.
/// </summary>
internal sealed class GpuMemorySampler : IDisposable
{
    private readonly Process _process;
    private readonly object _gate = new();
    private double _current = double.NaN;
    private double _peak = double.NaN;

    private GpuMemorySampler(Process process)
    {
        _process = process;
        _process.OutputDataReceived += (_, e) =>
        {
            if (double.TryParse(e.Data?.Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out double mib))
            {
                lock (_gate)
                {
                    _current = mib;
                    _peak = double.IsNaN(_peak) ? mib : Math.Max(_peak, mib);
                }
            }
        };
        _process.BeginOutputReadLine();
    }

    public static GpuMemorySampler? TryStart()
    {
        string device = (Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES") ?? "0").Split(',')[0].Trim();
        try
        {
            var psi = new ProcessStartInfo("nvidia-smi") { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
            foreach (string a in new[] { "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", device, "-lms", "100" })
            {
                psi.ArgumentList.Add(a);
            }

            var p = Process.Start(psi);
            if (p is null)
            {
                return null;
            }

            var sampler = new GpuMemorySampler(p);
            for (int i = 0; i < 30 && double.IsNaN(sampler.Current); i++)
            {
                Thread.Sleep(50);
            }

            return sampler;
        }
        catch (System.ComponentModel.Win32Exception)
        {
            return null;
        }
    }

    private double Current
    {
        get
        {
            lock (_gate)
            {
                return _current;
            }
        }
    }

    /// <summary>Restarts the peak at the current reading; returns the current reading (null if none yet).</summary>
    public double? Reset()
    {
        lock (_gate)
        {
            _peak = _current;
            return double.IsNaN(_current) ? null : _current;
        }
    }

    public double? Peak()
    {
        Thread.Sleep(150); // one more sample after the cell's last call
        lock (_gate)
        {
            return double.IsNaN(_peak) ? null : _peak;
        }
    }

    public void Dispose()
    {
        try
        {
            if (!_process.HasExited)
            {
                _process.Kill();
            }
        }
        catch (InvalidOperationException)
        {
        }

        _process.Dispose();
    }
}
