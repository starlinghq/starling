using Starlings.Acquisition;
using Starlings.Models.Kev;
using Starlings.Runtime.Onnx;

namespace Starlings.Models.Kev.Tests;

/// <summary>
/// Branch budget, over-long questions in one pass, placement, calibration flag and state reuse across calls. Contract
/// checks run without assets; encoding checks need the pinned tokenizer; model checks need a bundle (see
/// <see cref="KevTestAssets"/>).
/// </summary>
[Collection(KevModelGroup.Name)]
public sealed class ApiTests
{
    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static readonly Lazy<(KevEncoder? Encoder, string Reason)> Encoder = new(() =>
    {
        var (path, reason) = KevTestAssets.FindTokenizer();
        return path is null ? (null, reason) : (KevEncoder.FromFile(path), string.Empty);
    });

    private static readonly string LongInstruction = string.Concat(Enumerable.Repeat("why ", 4000));

    [Fact]
    public void BranchBudgetIsValidated()
    {
        Assert.Null(new KevOptions().MaxBranchTokens);
        Assert.False(new KevOptions().DeclineOverLongQuestions);
        new KevOptions { MaxBranchTokens = 1000 }.Validate();
        Assert.Throws<ArgumentOutOfRangeException>(() => new KevOptions { MaxBranchTokens = 0 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new KevOptions { MaxBranchTokens = KevModelProfile.MaxRowTokens + 1 }.Validate());
    }

    [Fact]
    public void NewCapabilitiesHaveExplicitStatuses()
    {
        Assert.Equal(KevCapabilityStatus.Experimental, KevCapabilities.All["branch_token_limit"]);
        Assert.Equal(KevCapabilityStatus.Experimental, KevCapabilities.All["decline_over_long_questions"]);
        Assert.Equal(KevCapabilityStatus.Experimental, KevCapabilities.All["placement_report"]);
        Assert.Equal(KevCapabilityStatus.Experimental, KevCapabilities.All["calibration_flag"]);
        Assert.Equal(KevCapabilityStatus.Experimental, KevCapabilities.All["state_reuse_across_calls"]);
        Assert.Equal(KevCapabilityStatus.Experimental, KevCapabilities.All["state_reuse_across_calls.cuda"]);
        Assert.Equal(KevCapabilityStatus.Unsupported, KevCapabilities.All["state_prefix_reuse_across_requests"]);
    }

    [Fact]
    public void CanLoadChecksTheProviderWithoutABundle()
    {
        Assert.Equal(KevCapabilityStatus.Experimental, KevCapabilities.All["load_check"]);
        Assert.Equal(KevCapabilityStatus.Untested, KevCapabilities.All["load_check.cuda"]);
        Assert.True(KevDecisionModel.CanLoad(null, out string? reason), reason);
        Assert.Null(reason);
        Assert.True(KevDecisionModel.CanLoad(new KevOptions { Session = new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cpu } }, out reason), reason);

        // The test project references the CPU-only native package, so CUDA cannot load and the reason says why.
        Assert.SkipWhen(
            OnnxRuntimeInfo.AvailableProviders.Contains(OnnxRuntimeInfo.CudaProviderName),
            "The loaded ONNX Runtime has the CUDA provider; the CPU-only negative case does not apply.");
        Assert.False(KevDecisionModel.CanLoad(new KevOptions { Session = new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cuda } }, out reason));
        Assert.Contains("CUDA", reason, StringComparison.Ordinal);
    }

    [Fact]
    public void CanLoadRejectsInvalidOptionsWithAReason()
    {
        Assert.False(KevDecisionModel.CanLoad(new KevOptions { MaxBranchTokens = 0 }, out string? reason));
        Assert.Contains(nameof(KevOptions.MaxBranchTokens), reason, StringComparison.Ordinal);
        Assert.False(KevDecisionModel.CanLoad(new KevOptions { Session = new OnnxSessionOptions { CudaDeviceId = -1 } }, out reason));
        Assert.Contains(nameof(OnnxSessionOptions.CudaDeviceId), reason, StringComparison.Ordinal);
    }

    [Fact]
    public void DecisionLookupIsByOrdinalIdAndFollowsCopies()
    {
        var a = new KevAnswer("a", KevQuestion.Noul("q"), [0.2f, 0.8f]);
        var b = new KevAnswer("b", KevQuestion.Noul("q"), [0.6f, 0.4f]);
        var first = new KevAnswer("dup", KevQuestion.Noul("q"), [0.1f, 0.9f]);
        var second = new KevAnswer("dup", KevQuestion.Noul("q"), [0.9f, 0.1f]);
        var decision = new KevDecision([a, b, first, second], 10, 20)
        {
            Declined = [new KevOverLongQuestion(4, "long", KevTokenLimit.Branch, 2000, 1000)],
        };

        Assert.Same(a, decision["a"]);
        Assert.Same(b, decision["b"]);
        Assert.Same(first, decision["dup"]);
        Assert.Contains("No question", Assert.Throws<KeyNotFoundException>(() => decision["A"]).Message, StringComparison.Ordinal);
        Assert.Contains("declined", Assert.Throws<KeyNotFoundException>(() => decision["long"]).Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentNullException>(() => decision[null!]);

        // A copy with other answers has its own index; the original keeps its own.
        var copy = decision with { Answers = [b] };
        Assert.Same(b, copy["b"]);
        Assert.Throws<KeyNotFoundException>(() => copy["a"]);
        Assert.Same(a, decision["a"]);
        Assert.Equal(new KevDecision(decision.Answers, 10, 20) { Declined = decision.Declined }, decision);
    }

    [Fact]
    public void NoulProbabilitiesAreInNoYesOrder()
    {
        var answer = new KevAnswer("u", KevQuestion.Noul("q"), [0.3f, 0.7f]);
        Assert.Equal(["false", "true"], answer.Keys);
        Assert.Equal(0.7f, answer.Yes);
        Assert.Equal(0.7f, answer.Probabilities["true"]);
        Assert.Equal(0.3f, answer.Probabilities["false"]);
    }

    [Fact]
    public void EveryOverLongQuestionIsReportedInOnePass()
    {
        Assert.SkipWhen(Encoder.Value.Encoder is null, Encoder.Value.Reason);
        var encoder = Encoder.Value.Encoder!;
        string state = string.Concat(Enumerable.Repeat("word ", 5000));
        var ok = new KevRecordQuestion("ok", "Is it long?", ["no", "yes"]);
        var longA = new KevRecordQuestion("a", LongInstruction, ["no", "yes"]);
        var longB = new KevRecordQuestion("b", LongInstruction, ["no", "yes"]);
        var e = Assert.Throws<KevInputTooLongException>(() => encoder.Encode(new KevRecord(state, [ok, longA, ok, longB])));
        Assert.Equal(1, e.QuestionIndex);
        Assert.Equal(KevTokenLimit.Row, e.Limit);
        Assert.Equal([1, 3], e.OverLongQuestions.Select(q => q.Index));
        Assert.Equal(["a", "b"], e.OverLongQuestions.Select(q => q.Id));
        Assert.All(e.OverLongQuestions, q => Assert.True(q.TokenCount > q.MaxTokens));
        Assert.Contains("1, 3", e.Message, StringComparison.Ordinal);

        var state2 = Assert.Throws<KevInputTooLongException>(() => encoder.Encode(new KevRecord(string.Concat(Enumerable.Repeat("word ", 9000)), [ok])));
        Assert.Equal(KevTokenLimit.State, state2.Limit);
        Assert.Empty(state2.OverLongQuestions);
    }

    [Fact]
    public void DeclineModeKeepsTheOtherQuestionsInOrder()
    {
        Assert.SkipWhen(Encoder.Value.Encoder is null, Encoder.Value.Reason);
        var encoder = Encoder.Value.Encoder!;
        string state = string.Concat(Enumerable.Repeat("word ", 5000));
        var q0 = new KevRecordQuestion("q0", "First?", ["no", "yes"]);
        var q2 = new KevRecordQuestion("q2", "Third?", ["a", "b", "c"]);
        var record = new KevRecord(state, [q0, new KevRecordQuestion("long", LongInstruction, ["no", "yes"]), q2]);
        var full = encoder.Encode(new KevRecord(state, [q0, q2]));
        var declined = encoder.Encode(record, decline: true);
        Assert.Equal([0, 2], declined.QuestionIndices!);
        Assert.Equal(0, declined.QuestionIndexOf(0));
        Assert.Equal(2, declined.QuestionIndexOf(1));
        var only = Assert.Single(declined.Declined);
        Assert.Equal((1, "long", KevTokenLimit.Row), (only.Index, only.Id, only.Limit));
        Assert.Equal(full.StateIds, declined.StateIds);
        Assert.Equal(full.Rows.Select(r => r.Ids), declined.Rows.Select(r => r.Ids));
        Assert.Equal(full.Rows.Select(r => r.Positions), declined.Rows.Select(r => r.Positions));

        var none = encoder.Encode(new KevRecord(state, [record.Questions[1]]), decline: true);
        Assert.Empty(none.Rows);
        Assert.Single(none.Declined);

        var complete = encoder.Encode(new KevRecord(state, [q0]), decline: true);
        Assert.Null(complete.QuestionIndices);
        Assert.Empty(complete.Declined);
    }

    [Fact]
    public void BranchBudgetCountsTheBranchAlone()
    {
        Assert.SkipWhen(Encoder.Value.Encoder is null, Encoder.Value.Reason);
        var encoder = Encoder.Value.Encoder!;
        var question = new KevRecordQuestion("q", "Is this about billing or about shipping?", ["no", "yes"]);
        int branch = encoder.Encode(new KevRecord("short", [question])).Rows[0].Ids.Length;
        string longState = string.Concat(Enumerable.Repeat("word ", 3000));

        // The state does not count: the same branch fits at its own length whatever the state.
        Assert.Single(encoder.Encode(new KevRecord(longState, [question]), maxBranchTokens: branch).Rows);
        var e = Assert.Throws<KevInputTooLongException>(() => encoder.Encode(new KevRecord("short", [question]), maxBranchTokens: branch - 1));
        Assert.Equal((KevTokenLimit.Branch, branch, branch - 1, 0), (e.Limit, e.TokenCount, e.MaxTokens, e.QuestionIndex));
        var declined = encoder.Encode(new KevRecord("short", [question, question with { Id = "r" }]), maxBranchTokens: branch - 1, decline: true);
        Assert.Equal(["q", "r"], declined.Declined.Select(d => d.Id));
        Assert.All(declined.Declined, d => Assert.Equal(KevTokenLimit.Branch, d.Limit));
    }

    [Theory]
    [InlineData(KevModel.Kev08B)]
    [InlineData(KevModel.Kev4B)]
    public void DeclinedDecisionAnswersTheRestWithUnchangedProbabilities(KevModel model)
    {
        var (files, reason) = KevTestAssets.FindBundle(model);
        Assert.SkipWhen(files is null, reason);
        using var declining = KevDecisionModel.Create(files!, new KevOptions
        {
            DeclineOverLongQuestions = true,
            MaxBranchTokens = 64,
            Session = new OnnxSessionOptions { IntraOpNumThreads = 4 },
        });
        const string state = "I was charged twice for order 1182. Please refund one of the charges.";
        var team = KevQuestion.Choice("Which team should handle this?", "billing", "shipping", "returns");
        var urgent = KevQuestion.Noul("Does this need a reply today?");
        var tooLong = KevQuestion.Noul(string.Concat(Enumerable.Repeat("Is it really about the charge? ", 20)));
        var decision = declining.Decide(new KevRequest(state, [new("team", team), new("long", tooLong), new("urgent", urgent)]), Ct);
        Assert.Equal(["team", "urgent"], decision.Answers.Select(a => a.Id));
        var declined = Assert.Single(decision.Declined);
        Assert.Equal(("long", 1, KevTokenLimit.Branch, 64), (declined.Id, declined.Index, declined.Limit, declined.MaxTokens));
        Assert.Throws<KeyNotFoundException>(() => decision["long"]);

        // The same questions without the declined one: nothing to decline, the plain path.
        var reference = declining.Decide(new KevRequest(state, [new("team", team), new("urgent", urgent)]), Ct);
        Assert.Empty(reference.Declined);
        Assert.Equal(reference["team"].ProbabilityList, decision["team"].ProbabilityList);
        Assert.Equal(reference["urgent"].ProbabilityList, decision["urgent"].ProbabilityList);
        Assert.Equal(reference.BranchTokens, decision.BranchTokens);

        var allDeclined = declining.Decide(new KevRequest(state, [new("long", tooLong)]), Ct);
        Assert.Empty(allDeclined.Answers);
        Assert.Single(allDeclined.Declined);
    }

    [Theory]
    [InlineData(KevModel.Kev08B)]
    [InlineData(KevModel.Kev4B)]
    public void PlacementAndCalibrationAreReported(KevModel model)
    {
        var (kev, reason) = ModelParityTests.Load(model);
        Assert.SkipWhen(kev is null, reason);
        var placement = kev!.Placement;
        Assert.Equal(OnnxExecutionProvider.Cpu, placement.Provider);
        Assert.Null(placement.CudaDeviceId);
        Assert.Equal(KevModelProfile.For(model).Id, placement.ProfileId);
        Assert.False(placement.SerializesNativeCalls);
        Assert.True(placement.StateCacheInHostMemory);
        Assert.Null(placement.Nodes); // CapturePlacement not set

        Assert.True(kev.IsCalibrated);
        var answer = kev.Decide(new KevRequest("text", [new("q", KevQuestion.Noul("Is it text?"))]), Ct).Answers[0];
        Assert.True(answer.IsCalibrated);
        Assert.Equal(KevModelProfile.For(model).Temperature, answer.Temperature);
    }

    [Theory]
    [InlineData(KevModel.Kev08B)]
    [InlineData(KevModel.Kev4B)]
    public void OnlyTheCheckpointTemperatureIsCalibrated(KevModel model)
    {
        var (files, reason) = KevTestAssets.FindBundle(model);
        Assert.SkipWhen(files is null, reason);
        using (var raw = KevDecisionModel.Create(files!, new KevOptions { Temperature = 1.0, Session = new OnnxSessionOptions { IntraOpNumThreads = 4 } }))
        {
            Assert.False(raw.IsCalibrated);
            var rawAnswer = raw.Decide(new KevRequest("text", [new("q", KevQuestion.Noul("Is it text?"))]), Ct).Answers[0];
            Assert.False(rawAnswer.IsCalibrated);
            Assert.Equal(1.0, rawAnswer.Temperature);
        }

        using var explicitCalibrated = KevDecisionModel.Create(files!, new KevOptions
        {
            Temperature = KevModelProfile.For(model).Temperature,
            Session = new OnnxSessionOptions { IntraOpNumThreads = 4 },
        });
        Assert.True(explicitCalibrated.IsCalibrated);
        Assert.True(explicitCalibrated.Decide(new KevRequest("text", [new("q", KevQuestion.Noul("Is it text?"))]), Ct).Answers[0].IsCalibrated);
    }

    /// <summary>Capability <c>state_reuse_across_calls</c> on CPU: every request golden, bit-identical to Decide.</summary>
    [Theory]
    [InlineData(KevModel.Kev08B)]
    [InlineData(KevModel.Kev4B)]
    public void StateReuseEqualsDecideBitForBit(KevModel model)
    {
        var (kev, reason) = ModelParityTests.Load(model);
        Assert.SkipWhen(kev is null, reason);
        foreach (var c in KevTestAssets.ReadData("requests.json").EnumerateArray())
        {
            var request = KevRequest.ParseSystemOne(c.GetProperty("request").GetString()!);
            var expected = kev!.Decide(request, Ct);
            using var state = kev.CreateState(request.State, Ct);
            Assert.Equal(expected.StateTokens, state.TokenCount);
            for (int round = 0; round < 2; round++)
            {
                var actual = kev.Decide(state, request.Questions, Ct);
                Assert.Equal(expected.StateTokens, actual.StateTokens);
                Assert.Equal(expected.BranchTokens, actual.BranchTokens);
                Assert.Equal(expected.Answers.Select(a => a.Id), actual.Answers.Select(a => a.Id));
                for (int i = 0; i < expected.Answers.Count; i++)
                {
                    Assert.Equal(expected.Answers[i].ProbabilityList, actual.Answers[i].ProbabilityList);
                }
            }

            // A different question set on the same state, then each question alone.
            foreach (var question in request.Questions)
            {
                var alone = kev.Decide(state, [question], Ct);
                Assert.Equal(expected[question.Key].ProbabilityList, alone[question.Key].ProbabilityList);
            }
        }
    }

    [Theory]
    [InlineData(KevModel.Kev08B)]
    [InlineData(KevModel.Kev4B)]
    public void StateIsSharedByConcurrentCallsAndRefusedAfterDispose(KevModel model)
    {
        var (kev, reason) = ModelParityTests.Load(model);
        Assert.SkipWhen(kev is null, reason);
        const string text = "I was charged twice for order 1182. Please refund one of the charges.";
        KeyValuePair<string, KevQuestion>[] questions =
        [
            new("team", KevQuestion.Choice("Which team should handle this?", "billing", "shipping", "returns")),
            new("urgent", KevQuestion.Noul("Does this need a reply today?")),
            new("severity", KevQuestion.Score("How severe is the problem?", "minor", "moderate", "serious")),
        ];
        var expected = kev!.Decide(new KevRequest(text, questions), Ct);
        var state = kev.CreateState(text, Ct);
        var results = new KevDecision[8];
        Parallel.For(0, results.Length, new ParallelOptions { MaxDegreeOfParallelism = 4, CancellationToken = Ct }, i =>
            results[i] = kev.Decide(state, questions, Ct));
        Assert.All(results, r =>
        {
            for (int i = 0; i < questions.Length; i++)
            {
                Assert.Equal(expected.Answers[i].ProbabilityList, r.Answers[i].ProbabilityList);
            }
        });

        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.ThrowsAny<OperationCanceledException>(() => kev.Decide(state, questions, cts.Token));
        Assert.Equal(expected["urgent"].ProbabilityList, kev.Decide(state, [questions[1]], Ct)["urgent"].ProbabilityList);

        state.Dispose();
        state.Dispose();
        Assert.True(state.IsDisposed);
        Assert.Throws<ObjectDisposedException>(() => kev.Decide(state, questions, Ct));
        Assert.Throws<ArgumentException>(() => kev.Decide(state, Array.Empty<KeyValuePair<string, KevQuestion>>(), Ct));
    }

    [Fact]
    public void StateFromAnotherModelIsRefused()
    {
        var (files, reason) = KevTestAssets.FindBundle(KevModel.Kev08B);
        Assert.SkipWhen(files is null, reason);
        var (kev, _) = ModelParityTests.Load(KevModel.Kev08B);
        using var other = KevDecisionModel.Create(files!, new KevOptions { Session = new OnnxSessionOptions { IntraOpNumThreads = 2 } });
        using var state = other.CreateState("text", Ct);
        Assert.Throws<ArgumentException>(() => kev!.Decide(state, [new("q", KevQuestion.Noul("Is it text?"))], Ct));
    }

    /// <summary>C7: a bundle opened with SHA-256 verification hands its digest to the session, which does not hash again.</summary>
    [Fact]
    public void VerifiedBundleDigestIsNotComputedAgain()
    {
        var (sized, reason) = KevTestAssets.FindBundle(KevModel.Kev08B);
        Assert.SkipWhen(sized is null, reason);
        Assert.Null(sized!.ModelSha256);
        var files = KevModelAssets.OpenDirectory(KevModel.Kev08B, sized.Snapshot.Directory, CacheVerification.Sha256);
        Assert.Equal(KevModelAssets.ModelSha256(KevModel.Kev08B), files.ModelSha256);
        using var kev = KevDecisionModel.Create(files, new KevOptions { Session = new OnnxSessionOptions { IntraOpNumThreads = 2 } });
        Assert.Equal(KevModelAssets.ModelSha256(KevModel.Kev08B), kev.Session.Diagnostics.ModelSha256);
        Assert.Equal(OnnxModelDigestSource.Known, kev.Session.Diagnostics.ModelSha256Source);
    }
}
