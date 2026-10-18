using System.Collections.Concurrent;
using System.Text.Json;
using Starlings.Models.Kev;
using Starlings.Runtime.Onnx;

namespace Starlings.Models.Kev.Tests;

/// <summary>
/// The model against Kev's PyTorch fp32 path (Data/probs-*.json, make_dotnet_cases.py probs) on the request goldens:
/// probabilities within the profile bound (2e-4, protected/kev-0.8b/onnx-fp32/tolerances.yaml; proposed for 4B), no
/// argmax flip, cached equal to uncached within 1e-5. Skips without a bundle (see <see cref="KevTestAssets"/>).
/// </summary>
public sealed class ModelParityTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static readonly ConcurrentDictionary<KevModel, Lazy<(KevDecisionModel? Model, string Reason)>> Models = new();

    private static (KevDecisionModel? Model, string Reason) Load(KevModel model) => Models.GetOrAdd(model, m => new(() =>
    {
        var (files, reason) = KevTestAssets.FindBundle(m);
        if (files is null)
        {
            return (null, reason);
        }

        // The bundle was size-checked; the session checks the graph digest against the lock.
        return (KevDecisionModel.Create(files, new KevOptions { Session = new OnnxSessionOptions { IntraOpNumThreads = 4 } }), string.Empty);
    })).Value;

    public static TheoryData<KevModel, string> Cases()
    {
        var data = new TheoryData<KevModel, string>();
        foreach (var model in new[] { KevModel.Kev08B, KevModel.Kev4B })
        {
            foreach (var c in KevTestAssets.ReadData("requests.json").EnumerateArray())
            {
                data.Add(model, c.GetProperty("id").GetString()!);
            }
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void MatchesPyTorchReference(KevModel model, string id)
    {
        var (kev, reason) = Load(model);
        Assert.SkipWhen(kev is null, reason);
        string? missing = KevParity.MissingReference(model);
        Assert.SkipWhen(missing is not null, missing ?? string.Empty);
        KevParity.AssertMatchesReference(kev!, id, Ct);
    }

    [Theory]
    [InlineData(KevModel.Kev08B)]
    [InlineData(KevModel.Kev4B)]
    public void CachedEqualsUncached(KevModel model)
    {
        var (kev, reason) = Load(model);
        Assert.SkipWhen(kev is null, reason);
        KevParity.AssertCachedEqualsUncached(kev!, Ct);
    }

    /// <summary>Capability <c>concurrent_calls</c> on CPU (not serialized): four callers, bit-identical to sequential.</summary>
    [Theory]
    [InlineData(KevModel.Kev08B)]
    [InlineData(KevModel.Kev4B)]
    public void ConcurrentCallsEqualSequential(KevModel model)
    {
        var (kev, reason) = Load(model);
        Assert.SkipWhen(kev is null, reason);
        Assert.False(kev!.SerializesNativeCalls);
        Assert.Equal(KevModelProfile.For(model).Id, kev.ProfileId);
        KevParity.AssertConcurrentEqualsSequential(kev, callers: 4, rounds: 2, Ct);
    }

    [Theory]
    [InlineData(KevModel.Kev08B)]
    [InlineData(KevModel.Kev4B)]
    public void QuestionsAreIsolated(KevModel model)
    {
        var (kev, reason) = Load(model);
        Assert.SkipWhen(kev is null, reason);
        var q1 = KevQuestion.Noul("Is this about billing?");
        var q2 = KevQuestion.Choice("Which team?", "billing", "shipping", "returns");
        const string state = "I was charged twice for order 1182.";
        var alone = kev!.Decide(new KevRequest(state, [new("a", q1)]), Ct);
        var together = kev.Decide(new KevRequest(state, [new("b", q2), new("a", q1)]), Ct);
        Assert.Equal(alone["a"].ProbabilityList, together["a"].ProbabilityList);
    }

    [Theory]
    [InlineData(KevModel.Kev08B)]
    [InlineData(KevModel.Kev4B)]
    public void CancelledCallThrowsAndModelStaysUsable(KevModel model)
    {
        var (kev, reason) = Load(model);
        Assert.SkipWhen(kev is null, reason);
        var request = new KevRequest("text", [new("q", KevQuestion.Noul("Is it text?"))]);
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.ThrowsAny<OperationCanceledException>(() => kev!.Decide(request, cts.Token));
        Assert.Equal(2, kev!.Decide(request, Ct).Answers[0].ProbabilityList.Count);
    }

    [Theory]
    [InlineData(KevModel.Kev08B)]
    [InlineData(KevModel.Kev4B)]
    public void RawTemperatureSharpensOrFlattensButKeepsOrder(KevModel model)
    {
        var (kev, reason) = Load(model);
        Assert.SkipWhen(kev is null, reason);
        var (files, _) = KevTestAssets.FindBundle(model);
        using var raw = KevDecisionModel.Create(files!, new KevOptions { Temperature = 1.0, Session = new OnnxSessionOptions { IntraOpNumThreads = 4 } });
        var request = new KevRequest("I was charged twice for order 1182.", [new("t", KevQuestion.Choice("Which team?", "billing", "shipping", "returns"))]);
        var calibrated = kev!.Decide(request, Ct)["t"];
        var uncalibrated = raw.Decide(request, Ct)["t"];
        Assert.Equal(1.0, raw.Temperature);
        Assert.Equal(calibrated.Choice, uncalibrated.Choice);
        Assert.True(uncalibrated.ProbabilityList.Max() >= calibrated.ProbabilityList.Max()); // T > 1 flattens
    }
}
