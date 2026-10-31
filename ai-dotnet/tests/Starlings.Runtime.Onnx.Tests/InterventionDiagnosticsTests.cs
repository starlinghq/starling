using System.Security.Cryptography;
using Microsoft.ML.OnnxRuntime;
using Starlings.TestSupport;

namespace Starlings.Runtime.Onnx.Tests;

/// <summary>
/// The intervention companion diagnostics on a weight-free graph with the shape of a derived intervention graph
/// (<see cref="TestModels.AddWithStrength"/>: the Add graph plus one float [1] strength input).
/// </summary>
public class InterventionDiagnosticsTests
{
    private const string Strength = "intervention_strength";
    private static readonly string AddSha = Convert.ToHexStringLower(SHA256.HashData(TestModels.Add));
    private static readonly string DerivedSha = Convert.ToHexStringLower(SHA256.HashData(TestModels.AddWithStrength));

    [Fact]
    public void OriginalGraphReportsNoIntervention()
    {
        using var session = OnnxModelSession.Create(TestModels.Add);
        var d = OnnxInterventionDiagnostics.ForOriginalGraph(session);

        Assert.False(d.Enabled);
        Assert.Equal(AddSha, d.OriginalModelSha256);
        Assert.Equal(AddSha, d.LoadedModelSha256);
        Assert.Equal(OnnxModelDigestSource.Computed, d.LoadedModelSha256Source);
        Assert.Equal("<memory>", d.LoadedModelSource);
        Assert.Empty(d.ProfileIds);
        Assert.Empty(d.InterventionInputs);
        Assert.Equal(["x", "y"], d.ModelInputs.Select(i => i.Name));
    }

    [Fact]
    public void DerivedGraphReportsIdentitiesProfilesAndInputs()
    {
        using var session = OnnxModelSession.Create(TestModels.AddWithStrength);
        var d = OnnxInterventionDiagnostics.ForDerivedGraph(session, AddSha, DerivedSha, ["synthetic-demo"], [Strength]);

        Assert.True(d.Enabled);
        Assert.Equal(AddSha, d.OriginalModelSha256);
        Assert.Equal(DerivedSha, d.LoadedModelSha256);
        Assert.Equal(OnnxModelDigestSource.Computed, d.LoadedModelSha256Source);
        Assert.Equal(["synthetic-demo"], d.ProfileIds);
        OnnxTensorInfo strength = Assert.Single(d.InterventionInputs);
        Assert.Equal(Strength, strength.Name);
        Assert.Equal("Float", strength.ElementType);
        Assert.Equal([1L], strength.Dimensions);
        Assert.Equal(["x", "y"], d.ModelInputs.Select(i => i.Name));

        // The ordinary session diagnostics are untouched: all three inputs, the derived graph's own digest.
        Assert.Equal(["x", "y", Strength], session.Inputs.Select(i => i.Name));
        Assert.Equal(DerivedSha, session.Diagnostics.ModelSha256);
    }

    [Fact]
    public void DigestSourceSaysWhetherTheSessionHashedTheFile()
    {
        using var session = OnnxModelSession.Create(TestModels.AddWithStrength, new OnnxSessionOptions { ComputeModelSha256 = false });
        var d = OnnxInterventionDiagnostics.ForDerivedGraph(session, AddSha, DerivedSha, ["p"], [Strength]);
        Assert.Equal(DerivedSha, d.LoadedModelSha256);
        Assert.Null(d.LoadedModelSha256Source);

        // A caller-supplied digest is reported as Known, never as computed: the session did not read the file for it.
        using var known = OnnxModelSession.Create(TestModels.AddWithStrength, new OnnxSessionOptions { KnownModelSha256 = DerivedSha });
        Assert.Equal(
            OnnxModelDigestSource.Known,
            OnnxInterventionDiagnostics.ForDerivedGraph(known, AddSha, DerivedSha, ["p"], [Strength]).LoadedModelSha256Source);
        using var knownOriginal = OnnxModelSession.Create(TestModels.Add, new OnnxSessionOptions { KnownModelSha256 = AddSha });
        Assert.Equal(OnnxModelDigestSource.Known, OnnxInterventionDiagnostics.ForOriginalGraph(knownOriginal).LoadedModelSha256Source);
    }

    [Fact]
    public void DerivedGraphFileLoadsThroughTheOrdinaryPathApi()
    {
        string dir = Directory.CreateTempSubdirectory("starlings-intervention-diag-").FullName;
        try
        {
            string path = Path.Combine(dir, "model.onnx");
            File.WriteAllBytes(path, TestModels.AddWithStrength);
            using var session = OnnxModelSession.Create(path, new OnnxSessionOptions { ExpectedModelSha256 = DerivedSha });
            var d = OnnxInterventionDiagnostics.ForDerivedGraph(session, AddSha, DerivedSha, ["p"], [Strength]);
            Assert.Equal(path, d.LoadedModelSource);
            Assert.True(d.Enabled);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void WrongDerivedDigestIsAnIntegrityFailure()
    {
        using var session = OnnxModelSession.Create(TestModels.AddWithStrength);
        Assert.Throws<OnnxModelIntegrityException>(
            () => OnnxInterventionDiagnostics.ForDerivedGraph(session, AddSha, new string('0', 64), ["p"], [Strength]));
    }

    [Fact]
    public void UndeclaredOrMissingInputsAreRejected()
    {
        // The original graph has no strength input: describing it as derived fails (the digest check passes: no digest).
        using var unhashed = OnnxModelSession.Create(TestModels.Add, new OnnxSessionOptions { ComputeModelSha256 = false });
        var missing = Assert.Throws<ArgumentException>(
            () => OnnxInterventionDiagnostics.ForDerivedGraph(unhashed, AddSha, DerivedSha, ["p"], [Strength]));
        Assert.Contains(Strength, missing.Message, StringComparison.Ordinal);

        using var session = OnnxModelSession.Create(TestModels.AddWithStrength);
        Assert.Throws<ArgumentException>(() => OnnxInterventionDiagnostics.ForDerivedGraph(session, AddSha, DerivedSha, ["p"], []));
        Assert.Throws<ArgumentException>(() => OnnxInterventionDiagnostics.ForDerivedGraph(session, AddSha, DerivedSha, ["p"], [Strength, Strength]));
        Assert.Throws<ArgumentException>(() => OnnxInterventionDiagnostics.ForDerivedGraph(session, AddSha, DerivedSha, [], [Strength]));
        Assert.Throws<ArgumentException>(() => OnnxInterventionDiagnostics.ForDerivedGraph(session, AddSha, DerivedSha, ["p", "p"], [Strength]));
        Assert.Throws<ArgumentException>(() => OnnxInterventionDiagnostics.ForDerivedGraph(session, AddSha, DerivedSha, [""], [Strength]));
        Assert.Throws<ArgumentException>(() => OnnxInterventionDiagnostics.ForDerivedGraph(session, AddSha.ToUpperInvariant(), DerivedSha, ["p"], [Strength]));
        Assert.Throws<ArgumentException>(() => OnnxInterventionDiagnostics.ForDerivedGraph(session, DerivedSha, DerivedSha, ["p"], [Strength]));
    }

    [Fact]
    public void StrengthZeroMatchesTheOriginalGraphAndStrengthIsPerRun()
    {
        float[] x = [1, 2, 3, 4, 5, 6, 7, 8];
        float[] y = [10, 20, 30, 40, 50, 60, 70, 80];
        long[] shape = [2, 4];
        using var original = OnnxModelSession.Create(TestModels.Add);
        using var derived = OnnxModelSession.Create(TestModels.AddWithStrength);

        float[] Run(OnnxModelSession session, float? strength)
        {
            using var xv = OrtValue.CreateTensorValueFromMemory(x, shape);
            using var yv = OrtValue.CreateTensorValueFromMemory(y, shape);
            using var sv = OrtValue.CreateTensorValueFromMemory(new[] { strength ?? 0f }, [1L]);
            string[] names = strength is null ? ["x", "y"] : ["x", "y", Strength];
            OrtValue[] values = strength is null ? [xv, yv] : [xv, yv, sv];
            using var outputs = session.Run(names, values, ["z"]);
            return outputs[0].GetTensorDataAsSpan<float>().ToArray();
        }

        float[] baseline = Run(original, null);
        Assert.Equal(baseline, Run(derived, 0f));
        Assert.Equal(baseline.Select(v => v * 0.5f), Run(derived, 0.5f));
        Assert.Equal(baseline, Run(derived, 0f)); // nothing sticks between runs

        // Concurrent runs with different strengths on one session do not interfere.
        Parallel.For(0, 64, i =>
        {
            float s = (i % 4) / 4f;
            Assert.Equal(baseline.Select(v => v - (s * v)), Run(derived, s));
        });
    }
}
