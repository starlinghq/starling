using Starlings.Acquisition;
using Starlings.Generation;
using Starlings.Interventions;
using Starlings.Runtime.Onnx;

namespace Starlings.Models.Lfm3.Interventions.Tests;

/// <summary>
/// The derived LFM3 fp32 graph of the synthetic profile (random orthonormal directions, rank 8, after layers 8 and 16)
/// against the original graph, on CPU. Needs <see cref="SnapshotVariable"/> and <see cref="InterventionVariable"/>; skips
/// without them. The original graph is loaded first, its reference values kept, and disposed before the derived graph
/// loads, so only one fp32 session (about 7 GB) exists at a time. The prompt is the template corpus case
/// <c>arithmetic</c> as token ids. The chat-level test loads the snapshot's <c>tokenizer.json</c> with the pinned
/// <c>Starlings.Tokenizers</c> (profile <c>lfm3-bpe-chat</c>, from 0.3.3) and checks that the text path encodes the same
/// conversation to those ids.
/// </summary>
[Collection(Lfm3SessionGroup.Name)]
public sealed class Lfm3InterventionModelTests
{
    /// <summary>The LFM3-1.7B-ONNX fp32 snapshot directory (as in Starlings.Models.Lfm3.Tests).</summary>
    public const string SnapshotVariable = "STARLING_LFM3_SNAPSHOT";

    /// <summary>
    /// A cache entry directory printed by <c>python -m model_pipeline.interventions.lfm3_synthetic --snapshot
    /// $STARLING_LFM3_SNAPSHOT --cache &lt;dir&gt;</c>.
    /// </summary>
    public const string InterventionVariable = "STARLING_LFM3_INTERVENTION";

    private const string ProfileId = "lfm3-synthetic-test";
    private const int NewTokens = 6;

    // "What is 2 + 2? Answer briefly." rendered by the LFM3 chat template without thinking (template-reference.jsonl).
    private static readonly int[] PromptIds =
        [124894, 124899, 5922, 207, 2992, 355, 229, 26, 1510, 229, 26, 39, 41774, 21422, 22, 124900, 207, 124899, 63514, 207, 124901, 124902];
    private static readonly Lazy<Fixture?> _fixture = new(Fixture.Load);

    private static Fixture Require()
    {
        Fixture? fixture = _fixture.Value;
        Assert.SkipWhen(fixture is null, $"Set {SnapshotVariable} to an LFM3-1.7B-ONNX fp32 snapshot and {InterventionVariable} to a compiled cache entry.");
        return fixture!;
    }

    [Fact]
    public void StrengthZeroMatchesTheOriginalGraph()
    {
        Fixture f = Require();
        float[] zero = f.PrefillLogits(new InterventionSelection(ProfileId, 0f));
        float[] none = f.PrefillLogits(null);
        float diff = MaxAbs(f.BaseLogits, zero);
        TestContext.Current.SendDiagnosticMessage($"max|original - strength 0| = {diff:g3}, max|strength 0 - no selection| = {MaxAbs(zero, none):g3}");
        Assert.True(diff <= 1e-4f, $"strength 0 differs from the original graph by {diff}");
        Assert.Equal(zero, none);
        Assert.Equal(f.BaseTokens, f.Greedy(new InterventionSelection(ProfileId, 0f)));
    }

    [Fact]
    public void StrengthOneChangesTheLogits()
    {
        Fixture f = Require();
        float[] one = f.PrefillLogits(new InterventionSelection(ProfileId, 1f));
        float[] half = f.PrefillLogits(new InterventionSelection(ProfileId, 0.5f));
        float diffOne = MaxAbs(f.BaseLogits, one);
        float diffHalf = MaxAbs(f.BaseLogits, half);
        TestContext.Current.SendDiagnosticMessage($"max|original - strength 1| = {diffOne:g3}, strength 0.5: {diffHalf:g3}");
        Assert.True(diffOne > 1e-2f, $"strength 1 changed the logits by only {diffOne}");
        Assert.True(diffHalf > 1e-3f && diffHalf < diffOne);
        TestContext.Current.SendDiagnosticMessage(
            $"greedy ids: original [{string.Join(", ", f.BaseTokens)}], strength 1 [{string.Join(", ", f.Greedy(new InterventionSelection(ProfileId, 1f)))}]");
    }

    [Fact]
    public void ConcurrentSequencesWithDifferentStrengthsAreIsolated()
    {
        Fixture f = Require();
        float[] strengths = [0f, 0.5f, 1f, -1f, 2f];
        float[][] reference = [.. strengths.Select(s => f.PrefillLogits(new InterventionSelection(ProfileId, s)))];
        var results = new float[strengths.Length * 2][];
        Parallel.For(0, results.Length, new ParallelOptions { MaxDegreeOfParallelism = results.Length }, i =>
            results[i] = f.PrefillLogits(new InterventionSelection(ProfileId, strengths[i % strengths.Length])));
        for (int i = 0; i < results.Length; i++)
        {
            float diff = MaxAbs(reference[i % strengths.Length], results[i]);
            Assert.True(diff <= 1e-5f, $"strength {strengths[i % strengths.Length]} run {i}: {diff} from its sequential value");
        }

        Assert.True(MaxAbs(reference[0], reference[2]) > 1e-2f);
    }

    [Fact]
    public void PrefillAndTokenByTokenAgree()
    {
        Fixture f = Require();
        var selection = new InterventionSelection(ProfileId, 1f);
        float[] prefill = f.PrefillLogits(selection);
        using Lfm3Sequence sequence = f.Model.CreateSequence(selection);
        foreach (int id in f.Prompt)
        {
            sequence.AppendTokens([id], TestContext.Current.CancellationToken);
        }

        var stepped = new float[sequence.VocabularySize];
        sequence.CopyLastLogits(stepped);
        float diff = MaxAbs(prefill, stepped);
        TestContext.Current.SendDiagnosticMessage($"strength 1: max|prefill - token by token| = {diff:g3}");
        Assert.True(diff <= 1e-3f, $"prefill and token-by-token differ by {diff}");
        Assert.Equal(Lfm3Sequence.ArgMax(prefill), Lfm3Sequence.ArgMax(stepped));
    }

    [Fact]
    public void ChatGenerateAndStreamingPassTheSelection()
    {
        Fixture f = Require();
        Lfm3ChatModel chat = f.Chat;
        var selection = new InterventionSelection(ProfileId, 1f);
        var options = new GenerationOptions { MaxNewTokens = NewTokens };
        GenerationResult generated = chat.Generate(f.Prompt, selection, options, TestContext.Current.CancellationToken);
        GenerationResult? streamed = null;
        foreach (GenerationUpdate update in chat.GenerateStreaming(f.Prompt, selection, options, TestContext.Current.CancellationToken))
        {
            streamed = update.Result ?? streamed;
        }

        Assert.Equal(generated.TokenIds.ToArray(), streamed!.TokenIds.ToArray());
        Assert.Equal(f.Greedy(selection).TakeWhile(id => id != Lfm3Checkpoint.ImEndTokenId), generated.TokenIds);
        GenerationResult zero = chat.Generate(f.Prompt, null, options, TestContext.Current.CancellationToken);
        Assert.Equal(f.BaseTokens.TakeWhile(id => id != Lfm3Checkpoint.ImEndTokenId), zero.TokenIds);
        Assert.Throws<InterventionMismatchException>(() => chat.GenerateStreaming(f.Prompt, new InterventionSelection("another-profile", 1f), cancellationToken: TestContext.Current.CancellationToken));

        // The text path: the pinned tokenizer encodes the rendered conversation to the reference prompt ids, and
        // Chat(turns, selection) decodes the same ids as Generate on those ids.
        IReadOnlyList<ChatTurn> turns = [new ChatTurn(ChatTurnRole.User, "What is 2 + 2? Answer briefly.")];
        Assert.Equal(f.Prompt, chat.EncodePrompt(turns, enableThinking: false));
        GenerationResult chatted = chat.Chat(turns, selection, options, enableThinking: false, TestContext.Current.CancellationToken);
        Assert.Equal(generated.TokenIds, chatted.TokenIds);
        Assert.Equal(generated.Text, chatted.Text);
    }

    [Fact]
    public void SelectionsAreCheckedAgainstTheLoadedGraph()
    {
        Fixture f = Require();
        Assert.IsType<InvalidOperationException>(f.PlainModelSelectionError);
        Assert.Throws<InterventionMismatchException>(() => f.Model.CreateSequence(new InterventionSelection("another-profile", 1f)));
    }

    [Fact]
    public void DiagnosticsReportBothGraphs()
    {
        Fixture f = Require();
        OnnxInterventionDiagnostics plain = f.PlainDiagnostics;
        Assert.False(plain.Enabled);
        Assert.Equal(39, plain.ModelInputs.Count);

        OnnxInterventionDiagnostics d = f.Model.GetInterventionDiagnostics();
        InterventionArtifact artifact = f.Model.GetInterventionArtifact()!;
        Assert.True(d.Enabled);
        Assert.Equal(Lfm3Interventions.BaseModelSha256(Lfm3Precision.Fp32), d.OriginalModelSha256);
        Assert.Equal(artifact.Manifest.DerivedGraph.Sha256, d.LoadedModelSha256);
        Assert.Equal(OnnxModelDigestSource.Known, d.LoadedModelSha256Source);
        Assert.Equal([ProfileId], d.ProfileIds);
        OnnxTensorInfo strength = Assert.Single(d.InterventionInputs);
        Assert.Equal("intervention_strength", strength.Name);
        Assert.Equal(plain.ModelInputs.Select(i => i.Name), d.ModelInputs.Select(i => i.Name));
        Assert.Equal("intervention_strength", f.Model.InputNames[^1]);
        Assert.Equal(plain.ModelInputs.Count + 1, f.Model.InputNames.Length);
    }

    [Fact]
    public void TheArtifactIsRejectedForAnotherBaseGraphBeforeAnySession()
    {
        Fixture f = Require();
        InterventionArtifact artifact = f.Model.GetInterventionArtifact()!;
        Assert.Throws<InterventionMismatchException>(() => Lfm3Interventions.CheckArtifact(artifact, Lfm3Precision.Fp16));
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

        /// <summary>A chat model over <see cref="Model"/> with the snapshot's tokenizer.</summary>
        public required Lfm3ChatModel Chat { get; init; }

        public required int[] Prompt { get; init; }

        public required float[] BaseLogits { get; init; }

        public required int[] BaseTokens { get; init; }

        public required Exception? PlainModelSelectionError { get; init; }

        public required OnnxInterventionDiagnostics PlainDiagnostics { get; init; }

        public float[] PrefillLogits(InterventionSelection? selection) => PrefillLogits(Model.CreateSequence(selection), Prompt);

        public int[] Greedy(InterventionSelection? selection) => Greedy(Model.CreateSequence(selection), Prompt);

        private static float[] PrefillLogits(Lfm3Sequence sequence, int[] prompt)
        {
            using (sequence)
            {
                sequence.AppendTokens(prompt);
                var logits = new float[sequence.VocabularySize];
                sequence.CopyLastLogits(logits);
                return logits;
            }
        }

        /// <summary>Greedy ids for <see cref="NewTokens"/> steps (end tokens included): every step is a native call with the strength.</summary>
        private static int[] Greedy(Lfm3Sequence sequence, int[] prompt)
        {
            using (sequence)
            {
                sequence.AppendTokens(prompt);
                var ids = new int[NewTokens];
                for (int i = 0; i < ids.Length; i++)
                {
                    ids[i] = sequence.Step();
                }

                return ids;
            }
        }

        public static Fixture? Load()
        {
            string? snapshot = Environment.GetEnvironmentVariable(SnapshotVariable);
            string? entry = Environment.GetEnvironmentVariable(InterventionVariable);
            if (string.IsNullOrEmpty(snapshot) || !Directory.Exists(snapshot) || string.IsNullOrEmpty(entry) || !Directory.Exists(entry))
            {
                return null;
            }

            Lfm3ModelFiles files = Lfm3ModelAssets.OpenDirectory(snapshot, Lfm3Precision.Fp32, CacheVerification.Size);
            var options = new Lfm3Options { ContextLength = 512 };
            float[] baseLogits;
            int[] baseTokens;
            Exception? plainError;
            OnnxInterventionDiagnostics plainDiagnostics;
            using (Lfm3Model original = Lfm3Model.Create(files, options))
            {
                baseLogits = PrefillLogits(original.CreateSequence(selection: null), PromptIds);
                baseTokens = Greedy(original.CreateSequence(), PromptIds);
                plainError = Record.Exception(() => original.CreateSequence(new InterventionSelection(ProfileId, 1f)));
                plainDiagnostics = original.GetInterventionDiagnostics();
            }

            Lfm3Model model = Lfm3Interventions.CreateModel(files, entry, options);
            var chat = new Lfm3ChatModel(model, Tokenizers.Tokenizer.FromFile(files.TokenizerPath));

            return new Fixture
            {
                Model = model,
                Chat = chat,
                Prompt = PromptIds,
                BaseLogits = baseLogits,
                BaseTokens = baseTokens,
                PlainModelSelectionError = plainError,
                PlainDiagnostics = plainDiagnostics,
            };
        }
    }
}
