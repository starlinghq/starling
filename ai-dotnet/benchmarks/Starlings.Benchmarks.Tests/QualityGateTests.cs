using System.Text.Json.Nodes;

namespace Starlings.Benchmarks.Tests;

public sealed class QualityGateTests
{
    private const double Tolerance = 1e-5;

    private static JsonObject Series(string workload, string path, double maxAbs, bool finite = true) => new()
    {
        ["mode"] = "warm",
        ["path"] = path,
        ["workload"] = workload,
        ["concurrency"] = 1,
        ["quality"] = new JsonObject { ["max_abs"] = maxAbs, ["min_cosine"] = 1.0, ["finite"] = finite },
    };

    private static JsonObject PassingSweep() => new() { ["pass"] = true, ["cases"] = 46 };

    [Fact]
    public void AllSeriesWithinToleranceKeepTheRunPassing()
    {
        var quality = PassingSweep();
        var series = new[] { Series("short-b1", "ids", 2e-7), Series("short-b32", "text", 2.3e-7) };

        Assert.True(QualityGate.Apply(quality, series, Tolerance));
        Assert.True(quality["pass"]!.GetValue<bool>());
        Assert.Empty(quality["failed_series"]!.AsArray());
        Assert.All(series, s => Assert.True(s["quality"]!["pass"]!.GetValue<bool>()));
    }

    [Fact]
    public void OneFailingSeriesFailsTheRunEvenWhenTheSweepPassed()
    {
        var quality = PassingSweep();
        var series = new[]
        {
            Series("short-b1", "ids", 2e-7),
            Series("long512-b32", "text", 3e-3),
            Series("short-b8", "ids", 1e-7),
        };

        Assert.False(QualityGate.Apply(quality, series, Tolerance));
        Assert.False(quality["pass"]!.GetValue<bool>());
        Assert.True(quality["preliminary_pass"]!.GetValue<bool>());
        Assert.False(quality["series_pass"]!.GetValue<bool>());
        Assert.Equal("warm/text/long512-b32/c1", Assert.Single(quality["failed_series"]!.AsArray())!.GetValue<string>());
        Assert.False(series[1]["quality"]!["pass"]!.GetValue<bool>());
        Assert.True(series[0]["quality"]!["pass"]!.GetValue<bool>());
    }

    [Fact]
    public void NonFiniteOutputFailsTheSeries()
    {
        var quality = PassingSweep();
        Assert.False(QualityGate.Apply(quality, [Series("short-b8", "ids", 0, finite: false)], Tolerance));
    }

    [Fact]
    public void AFailedSweepStaysFailed()
    {
        var quality = new JsonObject { ["pass"] = false };
        Assert.False(QualityGate.Apply(quality, [Series("short-b1", "ids", 1e-7)], Tolerance));
        Assert.True(quality["series_pass"]!.GetValue<bool>());
    }
}
