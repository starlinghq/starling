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
        var options = new SoakOptions
        {
            Duration = TimeSpan.FromSeconds(4),
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

    private sealed class FakeWorkload : SoakWorkload
    {
        public override string Name => "fake";

        public override int DefaultRequestsPerCycle => 20;

        public override string Profile(SoakOptions options) => "fake/cpu";

        public override string? Locate(SoakOptions options) => null;

        public override JsonObject Artifacts() => new() { ["repository"] = "fake/fake", ["revision"] = new string('0', 40) };

        public override ISoakInstance Load(SoakOptions options) => new Instance();

        private sealed class Instance : ISoakInstance
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

            public void Dispose() => _disposed = 1;
        }
    }
}
