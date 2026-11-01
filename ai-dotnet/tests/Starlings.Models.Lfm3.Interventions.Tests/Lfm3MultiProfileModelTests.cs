using Starlings.Acquisition;
using Starlings.Interventions;
using Starlings.Runtime.Onnx;

namespace Starlings.Models.Lfm3.Interventions.Tests;

/// <summary>
/// The derived LFM3 fp32 graph of two synthetic profiles with one strength vector (manifest schema version 2): index 0
/// <c>lfm3-synthetic-test</c> (projection after layers 8 and 16, the profile of the single-profile entry), index 1
/// <c>lfm3-synthetic-add</c> (addition after layers 12 and 16). Needs <see cref="Lfm3InterventionModelTests.SnapshotVariable"/>
/// and <see cref="MultiVariable"/>; with <see cref="Lfm3InterventionModelTests.InterventionVariable"/> also set, profile 0
/// alone is compared with the single-profile graph. References are computed first, one session at a time (original,
/// then single-profile graph, each disposed), before the two-profile graph loads.
/// </summary>
/// <remarks>
/// Run this class in its own test process (for example with <c>-class</c>): the fixture of
/// <see cref="Lfm3InterventionModelTests"/> holds its own fp32 session for the life of the process, and both classes
/// together would hold two (about 14 GB). The two classes share a collection so they never run in parallel.
/// </remarks>
[Collection(Lfm3SessionGroup.Name)]
public sealed class Lfm3MultiProfileModelTests
{
    /// <summary>A cache entry printed by <c>python -m model_pipeline.interventions.lfm3_synthetic --multi ...</c>.</summary>
    public const string MultiVariable = "STARLING_LFM3_INTERVENTION_MULTI";

    private const string Projection = "lfm3-synthetic-test";
    private const string Addition = "lfm3-synthetic-add";

    // The prompt of Lfm3InterventionModelTests (template corpus case "arithmetic", no thinking).
    private static readonly int[] PromptIds =
        [124894, 124899, 5922, 207, 2992, 355, 229, 26, 1510, 229, 26, 39, 41774, 21422, 22, 124900, 207, 124899, 63514, 207, 124901, 124902];
    private static readonly Lazy<Fixture?> _fixture = new(Fixture.Load);
    private static readonly float[] OneElement = [1f];

    private static Fixture Require()
    {
        Fixture? fixture = _fixture.Value;
        Assert.SkipWhen(fixture is null, $"Set {Lfm3InterventionModelTests.SnapshotVariable} to an LFM3-1.7B-ONNX fp32 snapshot and {MultiVariable} to a compiled two-profile cache entry.");
        return fixture!;
    }

    private static InterventionSelection Select(params (string Id, float Strength)[] entries) =>
        new([.. entries.Select(e => new InterventionProfileStrength(e.Id, e.Strength))]);

    [Fact]
    public void TheZeroVectorMatchesTheOriginalGraph()
    {
        Fixture f = Require();
        float[] none = f.PrefillLogits(null);
        float[] zeros = f.PrefillLogits(Select((Projection, 0f), (Addition, 0f)));
        float[] oneZero = f.PrefillLogits(new InterventionSelection(Addition, 0f));
        float diff = MaxAbs(f.BaseLogits, none);
        TestContext.Current.SendDiagnosticMessage($"max|original - (0,0)| = {diff:g3}");
        Assert.True(diff <= 1e-4f, $"the zero vector differs from the original graph by {diff}");
        Assert.Equal(none, zeros);
        Assert.Equal(none, oneZero);
    }

    [Fact]
    public void OneProfileAloneMatchesTheSingleProfileGraph()
    {
        Fixture f = Require();
        Assert.SkipWhen(f.SingleLogits is null, $"Set {Lfm3InterventionModelTests.InterventionVariable} to the single-profile entry to compare.");
        float[] first = f.PrefillLogits(Select((Projection, 1f), (Addition, 0f)));
        float diff = MaxAbs(f.SingleLogits!, first);
        TestContext.Current.SendDiagnosticMessage($"max|single-profile graph at 1 - (1,0)| = {diff:g3}");
        Assert.True(diff <= 1e-4f, $"(1,0) differs from the single-profile graph at strength 1 by {diff}");
    }

    [Fact]
    public void ProfilesCombineAndSelectionOrderDoesNotMatter()
    {
        Fixture f = Require();
        float[] first = f.PrefillLogits(new InterventionSelection(Projection, 1f));
        float[] second = f.PrefillLogits(new InterventionSelection(Addition, 1f));
        float[] both = f.PrefillLogits(Select((Projection, 1f), (Addition, 1f)));
        float[] reversed = f.PrefillLogits(Select((Addition, 1f), (Projection, 1f)));
        TestContext.Current.SendDiagnosticMessage(
            $"max|original - (0,1)| = {MaxAbs(f.BaseLogits, second):g3}, max|(1,0) - (1,1)| = {MaxAbs(first, both):g3}");
        Assert.True(MaxAbs(f.BaseLogits, second) > 1e-2f);
        Assert.True(MaxAbs(first, both) > 1e-2f);
        Assert.True(MaxAbs(second, both) > 1e-2f);
        Assert.Equal(both, reversed); // the graph applies profiles in strength-vector order, whatever the selection order
    }

    [Fact]
    public void ConcurrentSequencesWithDifferentVectorsAreIsolated()
    {
        Fixture f = Require();
        InterventionSelection?[] selections =
        [
            null,
            new InterventionSelection(Projection, 1f),
            new InterventionSelection(Addition, 1f),
            Select((Projection, 0.5f), (Addition, -1f)),
            Select((Projection, -1f), (Addition, 2f)),
        ];
        float[][] reference = [.. selections.Select(f.PrefillLogits)];
        var results = new float[10][];
        Parallel.For(0, results.Length, new ParallelOptions { MaxDegreeOfParallelism = 5 }, i =>
            results[i] = f.PrefillLogits(selections[i % selections.Length]));
        for (int i = 0; i < results.Length; i++)
        {
            float diff = MaxAbs(reference[i % selections.Length], results[i]);
            Assert.True(diff <= 1e-5f, $"sequence {i}: {diff}");
        }

        Assert.True(MaxAbs(reference[1], reference[2]) > 1e-2f);
    }

    [Fact]
    public void ASelectionOfAnUnknownProfileOrAWrongVectorIsRejected()
    {
        Fixture f = Require();
        Assert.Throws<InterventionMismatchException>(() => f.Model.CreateSequence(new InterventionSelection("lfm3-other", 1f)));
        Assert.Throws<InterventionMismatchException>(() => f.Model.CreateSequence(Select((Projection, 1f), ("lfm3-other", 1f))));
        Assert.Throws<ArgumentException>(() => f.Model.CreateSequence(OneElement));
    }

    [Fact]
    public void DiagnosticsListEveryProfile()
    {
        Fixture f = Require();
        OnnxInterventionDiagnostics d = f.Model.GetInterventionDiagnostics();
        Assert.True(d.Enabled);
        Assert.Equal([Projection, Addition], d.ProfileIds);
        OnnxTensorInfo strength = Assert.Single(d.InterventionInputs);
        Assert.Equal("intervention_strength", strength.Name);
        InterventionArtifact artifact = f.Model.GetInterventionArtifact()!;
        Assert.Equal(2, artifact.Manifest.SchemaVersion);
        Assert.Equal([Projection, Addition], artifact.Profiles.Select(p => p.Profile.Id));
        Assert.Equal(Lfm3Interventions.BaseModelSha256(Lfm3Precision.Fp32), d.OriginalModelSha256);
        Assert.Equal(artifact.Manifest.DerivedGraph.Sha256, d.LoadedModelSha256);
    }

    private static float MaxAbs(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float max = 0;
        for (int i = 0; i < a.Length; i++)
        {
            max = Math.Max(max, Math.Abs(a[i] - b[i]));
        }

        return max;
    }

    private sealed class Fixture
    {
        public required Lfm3Model Model { get; init; }

        public required float[] BaseLogits { get; init; }

        public required float[]? SingleLogits { get; init; }

        public float[] PrefillLogits(InterventionSelection? selection) => PrefillLogits(Model.CreateSequence(selection));

        private static float[] PrefillLogits(Lfm3Sequence sequence)
        {
            using (sequence)
            {
                sequence.AppendTokens(PromptIds);
                var logits = new float[sequence.VocabularySize];
                sequence.CopyLastLogits(logits);
                return logits;
            }
        }

        public static Fixture? Load()
        {
            string? snapshot = Environment.GetEnvironmentVariable(Lfm3InterventionModelTests.SnapshotVariable);
            string? entry = Environment.GetEnvironmentVariable(MultiVariable);
            string? single = Environment.GetEnvironmentVariable(Lfm3InterventionModelTests.InterventionVariable);
            if (string.IsNullOrEmpty(snapshot) || !Directory.Exists(snapshot) || string.IsNullOrEmpty(entry) || !Directory.Exists(entry))
            {
                return null;
            }

            Lfm3ModelFiles files = Lfm3ModelAssets.OpenDirectory(snapshot, Lfm3Precision.Fp32, CacheVerification.Size);
            var options = new Lfm3Options { ContextLength = 512 };
            float[] baseLogits;
            using (Lfm3Model original = Lfm3Model.Create(files, options))
            {
                baseLogits = PrefillLogits(original.CreateSequence());
            }

            float[]? singleLogits = null;
            if (!string.IsNullOrEmpty(single) && Directory.Exists(single))
            {
                using Lfm3Model singleModel = Lfm3Interventions.CreateModel(files, single, options);
                singleLogits = PrefillLogits(singleModel.CreateSequence(new InterventionSelection(Projection, 1f)));
            }

            return new Fixture
            {
                Model = Lfm3Interventions.CreateModel(files, entry, options),
                BaseLogits = baseLogits,
                SingleLogits = singleLogits,
            };
        }
    }
}

/// <summary>The real-model LFM3 classes run one at a time (each fixture holds an fp32 session).</summary>
[CollectionDefinition(Name)]
public sealed class Lfm3SessionGroup
{
    /// <summary>The collection name.</summary>
    public const string Name = "LFM3 fp32 session";
}
