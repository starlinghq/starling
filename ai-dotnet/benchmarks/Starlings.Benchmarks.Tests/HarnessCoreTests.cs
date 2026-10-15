using System.Text.Json;
using System.Text.Json.Nodes;

namespace Starlings.Benchmarks.Tests;

public sealed class HarnessCoreTests
{
    private static readonly string s_workloads = FindWorkloads();

    [Fact]
    public void RecordedVerdictsDecideTheRun()
    {
        var quality = new JsonObject { ["pass"] = true };
        var series = new[]
        {
            Series("short-l3", new JsonObject { ["pass"] = true }),
            Series("short-decode64", new JsonObject { ["pass"] = false, ["failures"] = new JsonArray("en-list: diverges at step 3") }),
            Series("long-prefill", new JsonObject()),
        };

        Assert.False(QualityGate.ApplyRecorded(quality, series));
        Assert.False(quality["pass"]!.GetValue<bool>());
        Assert.True(quality["preliminary_pass"]!.GetValue<bool>());
        Assert.Equal(["warm/text/short-decode64/c1", "warm/text/long-prefill/c1"], quality["failed_series"]!.AsArray().Select(n => n!.GetValue<string>()));
        Assert.False(series[2]["quality"]!["pass"]!.GetValue<bool>()); // a missing verdict fails
    }

    [Fact]
    public void TimerRecordsEverySampleWithItsStagesAndMetrics()
    {
        var timed = SeriesTimer.Time(25, 3, 2, 1, (i, worker, stages, metrics) =>
        {
            stages[0] = i;
            stages[1] = worker;
            metrics[0] = i * 10;
        });

        Assert.Equal(25, timed.Totals.Length);
        Assert.All(timed.Totals, t => Assert.True(t >= 0));
        for (int i = 0; i < 25; i++)
        {
            Assert.Equal(i, timed.Stages[i, 0]);
            Assert.InRange(timed.Stages[i, 1], 0, 2);
            Assert.Equal(i * 10, timed.Metrics[i, 0]);
        }

        var stages = timed.StagesJson(["a", "worker"]);
        Assert.Equal(25, stages["a"]!.AsArray().Count);
        Assert.Equal(25, timed.MetricsJson(["m"])["m"]!.AsArray().Count);
        Assert.True(timed.WallMs >= 0);
    }

    [Fact]
    public void ColdStageArraysSkipNonTimingFieldsAndAddTheProcessWall()
    {
        var children = new List<JsonObject>
        {
            new() { ["spawn_to_main"] = 20.0, ["session_create"] = 500.0, ["spawn_to_result"] = 900.0, ["quality"] = new JsonObject() },
            new() { ["spawn_to_main"] = 22.0, ["session_create"] = 510.0, ["spawn_to_result"] = 910.0, ["quality"] = new JsonObject() },
        };

        var stages = ColdRunner.StageArrays(children, [950.0, 960.0], ["quality", "spawn_to_result"]);

        Assert.Equal(["spawn_to_main", "session_create", "process_wall"], stages.Select(kv => kv.Key));
        Assert.Equal([500.0, 510.0], stages["session_create"]!.AsArray().Select(n => n!.GetValue<double>()));
    }

    [Fact]
    public void OptionsParseAndRejectAStrayValue()
    {
        var map = ArgMap.Parse(["--threads", "6", "--only", "a,b", "--samples-scale", "0.5"], "usage");
        Assert.Equal(6, map.GetInt("threads"));
        Assert.Equal(["a", "b"], map.GetList("only")!);
        Assert.Equal(0.5, map.GetDouble("samples-scale", 1));
        Assert.Null(map.Get("out"));
        Assert.Throws<ArgumentException>(() => ArgMap.Parse(["--threads"], "usage"));
        Assert.Throws<ArgumentException>(() => ArgMap.Parse(["threads", "6"], "usage"));
    }

    [Theory]
    [InlineData("gliner2-v1.json", "text")]
    [InlineData("embeddinggemma2-v1.json", "text,image,video")]
    [InlineData("qwen35-v1.json", "ids,text")]
    public void WorkloadFilesParseWithUniqueNamesAndKnownPaths(string file, string paths)
    {
        var spec = WorkloadSpec.Load(Path.Combine(s_workloads, file));
        Assert.NotEmpty(spec.Workloads);
        Assert.True(spec.WarmupCalls >= 1);
        Assert.Equal(spec.Workloads.Count, spec.Workloads.Select(w => w.Name).Distinct().Count());
        Assert.All(spec.Workloads, w =>
        {
            Assert.Contains(w.Path, paths.Split(','));
            Assert.True(w.Samples >= 1);
            Assert.True(w.Batch >= 1);
            bool hasRequests = w.Raw.TryGetProperty("requests", out var requests) && requests.ValueKind == JsonValueKind.Array && requests.GetArrayLength() > 0;
            Assert.True(hasRequests || w.Raw.TryGetProperty("synthetic", out _), $"{w.Name}: no requests");
            if (hasRequests && requests[0].ValueKind == JsonValueKind.Array)
            {
                Assert.All(requests.EnumerateArray(), r => Assert.Equal(w.Batch, r.GetArrayLength()));
            }
        });
    }

    private static JsonObject Series(string workload, JsonObject quality) => new()
    {
        ["mode"] = "warm",
        ["path"] = "text",
        ["workload"] = workload,
        ["concurrency"] = 1,
        ["quality"] = quality,
    };

    private static string FindWorkloads()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "ai-dotnet", "benchmarks", "workloads");
            if (Directory.Exists(candidate))
            {
                return candidate;
            }
        }

        throw new InvalidOperationException("ai-dotnet/benchmarks/workloads not found above " + AppContext.BaseDirectory);
    }
}
