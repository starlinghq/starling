using System.Text.Json.Nodes;

namespace Starlings.Soak.Tool.Tests;

/// <summary>The soak loop end to end on a fake model: checkpoints, final report, reanalysis.</summary>
public sealed class SoakRunTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("starlings-soak-tests-").FullName;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    [Fact]
    public void EnvironmentManifestCanBeCollectedRepeatedlyFromOneLoadSnapshot()
    {
        // Regression: the manifest attached the same load_before node at every checkpoint ("node already has a parent").
        var loadBefore = SoakEnvironment.Load();
        var first = new JsonObject { ["environment"] = SoakEnvironment.Collect(loadBefore, "a") };
        var second = new JsonObject { ["environment"] = SoakEnvironment.Collect(loadBefore, "b") };
        Assert.Equal(first["environment"]!["load_before"]!.ToJsonString(), second["environment"]!["load_before"]!.ToJsonString());
        Assert.NotNull(JsonNode.Parse(second.ToJsonString()));
    }

    [Fact]
    public void ARunWithSeveralCheckpointsFinishesAndCanBeReanalysed()
    {
        // The loop is bounded by wall-clock time only. A fake cycle takes about 0.2 s on an idle machine and up to 2 s on a
        // loaded CI runner (2 cycles in 4 s observed), so the run gets enough time for at least 4 cycles on a loaded runner.
        var options = new SoakOptions
        {
            Duration = TimeSpan.FromSeconds(12),
            Out = _dir,
            CheckpointInterval = TimeSpan.FromMilliseconds(200),
            SampleInterval = TimeSpan.FromMilliseconds(200),
            RequestsPerCycle = 20,
            DeadlineEvery = 5,
            Note = "unit test",
        };

        int exit = new SoakRunner(new FakeWorkload(), options).Run();

        string json = Assert.Single(Directory.GetFiles(_dir, "fake-*.json"));
        var raw = JsonNode.Parse(File.ReadAllText(json))!;
        string verdict = (string)raw["verdict"]!;
        Assert.False(verdict.StartsWith("INCOMPLETE", StringComparison.Ordinal), verdict);
        Assert.Equal(0, exit);
        Assert.True(raw["cycles"]!.AsArray().Count >= 4);
        Assert.Equal(0, (int)raw["wrong_output_count"]!);
        Assert.Equal(0, (int)raw["error_count"]!);
        Assert.True((int)raw["cancellation"]!["deadline_requests_cancelled"]! > 0);
        Assert.Contains("Short development run", File.ReadAllText(Path.ChangeExtension(json, ".md")), StringComparison.Ordinal);

        Assert.Equal(0, Reanalysis.Run([json]));
        Assert.Equal(0, Reanalysis.Run([json]));
    }

    [Fact]
    public void KeepLoadedLoadsOnceServesEveryCycleAndDisposesAtTheEnd()
    {
        var options = SoakOptions.Parse(["--out", _dir, "--duration", "2s", "--keep-loaded", "--requests-per-cycle", "10", "--deadline-every", "0", "--sample-interval", "200ms"]);
        Assert.True(options.KeepLoaded);
        var workload = new FakeWorkload();

        int exit = new SoakRunner(workload, options).Run();

        Assert.Equal(0, exit);
        Assert.Equal(1, workload.Loads);
        Assert.Equal(1, workload.Disposals);
        string json = Assert.Single(Directory.GetFiles(_dir, "fake-*.json"));
        var raw = JsonNode.Parse(File.ReadAllText(json))!;
        Assert.True((bool)raw["options"]!["keep_loaded"]!);
        Assert.True(raw["cycles"]!.AsArray().Count >= 2);
        Assert.Contains("Keep loaded (diagnostic)", File.ReadAllText(Path.ChangeExtension(json, ".md")), StringComparison.Ordinal);
    }

    private sealed class FakeWorkload : SoakWorkload
    {
        public int Loads;
        public int Disposals;

        public override string Name => "fake";

        public override int DefaultRequestsPerCycle => 20;

        public override string Profile(SoakOptions options) => "fake/cpu";

        public override string? Locate(SoakOptions options) => null;

        public override JsonObject Artifacts() => new() { ["repository"] = "fake/fake", ["revision"] = new string('0', 40) };

        public override ISoakInstance Load(SoakOptions options)
        {
            Interlocked.Increment(ref Loads);
            return new Instance(this);
        }

        private sealed class Instance(FakeWorkload owner) : ISoakInstance
        {
            private int _disposed;

            public IReadOnlyList<string> SessionProviders => ["fake"];

            public SoakResult Run(int input, CancellationToken cancellationToken)
            {
                ObjectDisposedException.ThrowIf(_disposed != 0, this);
                return new SoakResult($"input {input}", [input, input * 0.5f]);
            }

            public void RunLong(CancellationToken cancellationToken)
            {
                cancellationToken.WaitHandle.WaitOne(TimeSpan.FromSeconds(2));
                cancellationToken.ThrowIfCancellationRequested();
            }

            public void Dispose()
            {
                _disposed = 1;
                Interlocked.Increment(ref owner.Disposals);
            }
        }
    }
}
