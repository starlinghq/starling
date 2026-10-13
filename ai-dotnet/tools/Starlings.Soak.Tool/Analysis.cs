using System.Text.Json.Nodes;

/// <summary>
/// Post-warmup plateau and growth analysis of one series (one metric, settled samples taken at the same point of every
/// load/unload cycle).
/// </summary>
/// <remarks>
/// <para>
/// The growth is the least-squares slope times the post-warmup window ("fitted growth"). A series is
/// <c>growth</c> (unbounded growth, a blocker) when the fitted growth exceeds the limit
/// <c>max(--growth-limit-mb, 2% of the plateau median)</c> and the growth continues in both halves of the window (each
/// half's fitted growth above a quarter of the limit). The second condition reproduces the trend inside the run: a
/// single step, such as an allocator keeping a high-water mark (plan 10.2: not automatically a leak), raises one half
/// and leaves the other flat, and is reported as <c>plateau</c> with its band. Fewer than 4 post-warmup points is
/// <c>insufficient-data</c>. The thresholds are a screening heuristic; the raw samples are in the JSON for review.
/// </para>
/// </remarks>
internal sealed record SeriesVerdict(
    string Series,
    string Metric,
    int Points,
    double? First,
    double? Last,
    double? Median,
    double? Min,
    double? Max,
    double? SlopeMbPerHour,
    double? FittedGrowthMb,
    double? FirstHalfGrowthMb,
    double? SecondHalfGrowthMb,
    double LimitMb,
    string Verdict)
{
    /// <summary>Whether a growth verdict on this series fails the run (<see cref="MemorySample.Gates"/>).</summary>
    public bool Gates { get; init; } = true;

    public const string Growth = "growth";
    public const string Plateau = "plateau";
    public const string Insufficient = "insufficient-data";
    public const string Retention = "retention";

    public static SeriesVerdict Analyze(string series, string metric, IReadOnlyList<(double Seconds, double Value)> points, double limitFloorMb)
    {
        if (points.Count == 0)
        {
            return new SeriesVerdict(series, metric, 0, null, null, null, null, null, null, null, null, null, limitFloorMb, Insufficient);
        }

        var values = points.Select(p => p.Value).ToArray();
        double median = MedianOf(values);
        double limit = Math.Max(limitFloorMb, 0.02 * median);
        if (points.Count < 4)
        {
            return new SeriesVerdict(series, metric, points.Count, values[0], values[^1], median, values.Min(), values.Max(), null, null, null, null, limit, Insufficient);
        }

        var (slope, growth) = Fit(points);
        int half = points.Count / 2;
        var (_, g1) = Fit(points.Take(half + (points.Count % 2)).ToList());
        var (_, g2) = Fit(points.Skip(half).ToList());
        string verdict = growth > limit && g1 > limit / 4 && g2 > limit / 4 ? Growth : Plateau;
        return new SeriesVerdict(series, metric, points.Count, values[0], values[^1], median, values.Min(), values.Max(), slope, growth, g1, g2, limit, verdict);
    }

    /// <summary>Least-squares slope in MiB per hour, and slope times the window in MiB.</summary>
    private static (double SlopePerHour, double Growth) Fit(IReadOnlyList<(double Seconds, double Value)> points)
    {
        if (points.Count < 2)
        {
            return (0, 0);
        }

        double mx = points.Average(p => p.Seconds);
        double my = points.Average(p => p.Value);
        double sxx = points.Sum(p => (p.Seconds - mx) * (p.Seconds - mx));
        double sxy = points.Sum(p => (p.Seconds - mx) * (p.Value - my));
        double perSecond = sxx == 0 ? 0 : sxy / sxx;
        return (perSecond * 3600, perSecond * (points[^1].Seconds - points[0].Seconds));
    }

    public static double MedianOf(IReadOnlyList<double> values)
    {
        var sorted = values.Order().ToArray();
        int n = sorted.Length;
        return n == 0 ? double.NaN : n % 2 == 1 ? sorted[n / 2] : (sorted[(n / 2) - 1] + sorted[n / 2]) / 2;
    }

    public static double PercentileOf(IReadOnlyList<double> values, double p)
    {
        if (values.Count == 0)
        {
            return double.NaN;
        }

        var sorted = values.Order().ToArray();
        double rank = p / 100 * (sorted.Length - 1);
        int lo = (int)Math.Floor(rank);
        int hi = (int)Math.Ceiling(rank);
        return sorted[lo] + ((sorted[hi] - sorted[lo]) * (rank - lo));
    }

    public JsonObject ToJson() => new()
    {
        ["series"] = Series,
        ["metric"] = Metric,
        ["points"] = Points,
        ["first_mb"] = First,
        ["last_mb"] = Last,
        ["median_mb"] = Median,
        ["min_mb"] = Min,
        ["max_mb"] = Max,
        ["slope_mb_per_hour"] = SlopeMbPerHour,
        ["fitted_growth_mb"] = FittedGrowthMb,
        ["first_half_growth_mb"] = FirstHalfGrowthMb,
        ["second_half_growth_mb"] = SecondHalfGrowthMb,
        ["limit_mb"] = LimitMb,
        ["verdict"] = Verdict,
        ["gates"] = Gates,
    };
}
