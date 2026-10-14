using Starlings.Generation.OnnxGenAI;
using Starlings.Models.Qwen35;
using Starlings.TestSupport;

namespace Starlings.Models.Qwen35.Tests.Cuda;

internal static class CudaGate
{
    public const string OptIn = "STARLING_CUDA_TESTS";

    /// <summary>Null when CUDA tests should run; otherwise the skip reason. Throws on an unsafe opt-in.</summary>
    public static string? Check()
    {
        if (Environment.GetEnvironmentVariable(OptIn) != "1")
        {
            return $"CUDA tests are opt-in: set {OptIn}=1, CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES to the GPU to use.";
        }

        // Never let the CUDA runtime pick a default device on a shared workstation.
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES")) ||
            Environment.GetEnvironmentVariable("CUDA_DEVICE_ORDER") != "PCI_BUS_ID")
        {
            throw new InvalidOperationException($"{OptIn}=1 requires CUDA_DEVICE_ORDER=PCI_BUS_ID and an explicit CUDA_VISIBLE_DEVICES.");
        }

        return null;
    }
}

public abstract class CudaModelFixture : IDisposable
{
    protected CudaModelFixture(Qwen35Variant variant)
    {
        Reason = CudaGate.Check() ?? string.Empty;
        if (Reason.Length > 0)
        {
            return;
        }

        var (files, reason) = Qwen35TestAssets.FindModel(variant);
        Reason = reason;
        if (files is not null)
        {
            Model = Qwen35ChatModel.Load(files, new Qwen35ModelOptions { Provider = GenAIExecutionProvider.Cuda });
        }
    }

    public Qwen35ChatModel? Model { get; }

    public string Reason { get; }

    public void Dispose()
    {
        Model?.Dispose();
        GC.SuppressFinalize(this);
    }
}

public sealed class Fp32xCudaFixture() : CudaModelFixture(Qwen35Variant.Fp32x);

public sealed class Fp16xCudaFixture() : CudaModelFixture(Qwen35Variant.Fp16x);

[CollectionDefinition("cuda", DisableParallelization = true)]
public sealed class CudaTestGroup;

/// <summary>The fp32x graph on the CUDA provider (TF32 off) against the transformers fp32 CPU reference: all 22 prompts.</summary>
[Collection("cuda")]
public sealed class Fp32xCudaTests(Fp32xCudaFixture fixture) : IClassFixture<Fp32xCudaFixture>
{
    private Qwen35ChatModel Model
    {
        get
        {
            Assert.SkipWhen(fixture.Model is null, fixture.Reason);
            return fixture.Model!;
        }
    }

    public static TheoryData<string> CaseIds() => [.. Qwen35TestAssets.LoadCases().Select(c => c.Id)];

    private static GenerationCase Case(string id) => Qwen35TestAssets.LoadCases().Single(c => c.Id == id);

    [Fact]
    public void RunsOnCudaWithTf32Off()
    {
        Assert.Equal(GenAIExecutionProvider.Cuda, Model.Info.Provider);
        Assert.Equal("0", Model.Info.ProviderOptions["use_tf32"]);
        Assert.Equal("Microsoft.ML.OnnxRuntimeGenAI.Cuda/0.17.1", Model.Info.GenAINativePackage);
    }

    [Theory]
    [MemberData(nameof(CaseIds))]
    // 128-token prefill chunks: on the shared workstation GPU (another process holds about 10.7 of 16 GB) the fp32 logits
    // of a default 512-token chunk (508 MB) did not fit next to the 3.1 GB fp32x weights.
    public void GreedyIdsEqualTheReference(string id) => Qwen35ModelChecks.GreedyFromReferenceIds(Model, Case(id), prefillChunkTokens: 128);

    [Fact]
    public void StreamingReassembles() => Qwen35ModelChecks.StreamingReassembles(Model, Case("emoji"), 64);

    [Fact]
    public void ConcurrentRequestsAreIndependent()
    {
        // Serialized by default on CUDA: unserialized concurrent GenAI generators produced wrong tokens.
        Assert.True(Model.Info.SerializesNativeCalls);
        Qwen35ModelChecks.ConcurrentRequestsAreIndependent(
            Model, [Case("zh"), Case("code-py"), Case("en-math"), Case("ja"), Case("en-list"), Case("th")], 32, rounds: 4, threads: 4);
    }

    [Fact]
    public void CancellationBetweenStepsThenReuse() => Qwen35ModelChecks.CancellationThenReuse(Model, Case("en-list"), Case("en-short"));

    /// <summary>
    /// Capability <c>long_context_1667_to_8192_tokens</c> on fp32x CUDA: greedy ids, step-0 logits and cached versus
    /// uncached decoding against the transformers fp32 reference on prompts-long-v1 (proposed fixture), with this
    /// profile's protected tolerances. The 8,213-token prompt plus 32 tokens is beyond the accepted 8,192, so it runs with
    /// the limit lifted (test seam); it covers every length the package accepts.
    /// </summary>
    [Theory]
    [InlineData("long-4096")]
    [InlineData("long-8192")]
    public void LongContextGreedyAndLogitsEqualTheReference(string id)
    {
        var c = Qwen35TestAssets.LoadLongContextCases().Single(x => x.Id == id);
        var tolerances = Qwen35TestAssets.Tolerances("ortgenai-fp32x-cuda");
        var model = Model;
        model.EnforcedContextTokens = int.MaxValue;
        try
        {
            Qwen35ModelChecks.GreedyFromReferenceIds(model, c, prefillChunkTokens: 128);
            Qwen35LogitChecks.Step0LogitsMatchTheReference(model, c, tolerances, prefillChunkTokens: 128);
            Qwen35LogitChecks.CachedDecodingEqualsAFreshPrefill(model, c, 16, tolerances, prefillChunkTokens: 128);
        }
        finally
        {
            model.EnforcedContextTokens = Qwen35ChatModel.MaxContextTokens;
        }
    }

    [Fact]
    public void TextPathWithThinking() => Qwen35ModelChecks.TextPathMatches(Model, Case("thinking-on"), 32);
}

/// <summary>
/// The fp16x graph on CUDA. fp16 is not expected to reproduce fp32 at near ties: ids must equal the reference up to the
/// first divergence, which must fall where the reference top-1 minus top-2 margin is below 0.05 (a proposed bound,
/// docs/evidence/p016-qwen35/README.md; the binding tolerance is the maintainer's).
/// </summary>
[Collection("cuda")]
public sealed class Fp16xCudaTests(Fp16xCudaFixture fixture) : IClassFixture<Fp16xCudaFixture>
{
    private const double ProposedNearTieMargin = 0.05;

    private Qwen35ChatModel Model
    {
        get
        {
            Assert.SkipWhen(fixture.Model is null, fixture.Reason);
            return fixture.Model!;
        }
    }

    public static TheoryData<string> CaseIds() => [.. Qwen35TestAssets.LoadCases().Select(c => c.Id)];

    private static GenerationCase Case(string id) => Qwen35TestAssets.LoadCases().Single(c => c.Id == id);

    [Theory]
    [MemberData(nameof(CaseIds))]
    public void GreedyIdsDivergeOnlyAtNearTies(string id) => Qwen35ModelChecks.GreedyFromReferenceIds(Model, Case(id), nearTieMargin: ProposedNearTieMargin);

    [Fact]
    public void StreamingReassembles()
    {
        var c = Case("emoji");
        var updates = Model.GenerateStreaming(Qwen35Prompt.FromTokenIds(c.PromptIds), new Qwen35ChatOptions { MaxNewTokens = 64 }, TestContext.Current.CancellationToken).ToList();
        var result = updates[^1].Result!;
        Assert.Equal(result.Text, string.Concat(updates.Select(u => u.Text)));
        Assert.Equal(Model.Tokenizer.Decode(result.TokenIds.AsSpan(), false), result.Text);
    }

    /// <summary>
    /// Capability <c>long_context_1667_to_8192_tokens</c> on fp16x CUDA: greedy ids equal the reference up to a divergence
    /// at a near tie (the profile's binding rule), step-0 logits within this profile's protected tolerances. The
    /// 8,213-token prompt runs with the limit lifted (test seam).
    /// </summary>
    [Theory]
    [InlineData("long-4096")]
    [InlineData("long-8192")]
    public void LongContextGreedyAndLogitsMatchTheReference(string id)
    {
        var c = Qwen35TestAssets.LoadLongContextCases().Single(x => x.Id == id);
        var tolerances = Qwen35TestAssets.Tolerances("ortgenai-fp16x-cuda");
        var model = Model;
        model.EnforcedContextTokens = int.MaxValue;
        try
        {
            Qwen35ModelChecks.GreedyFromReferenceIds(model, c, nearTieMargin: ProposedNearTieMargin, prefillChunkTokens: 128);
            Qwen35LogitChecks.Step0LogitsMatchTheReference(model, c, tolerances, prefillChunkTokens: 128);
        }
        finally
        {
            model.EnforcedContextTokens = Qwen35ChatModel.MaxContextTokens;
        }
    }

    [Fact]
    public void Fp16xIsRejectedOnCpu()
    {
        var (files, reason) = Qwen35TestAssets.FindModel(Qwen35Variant.Fp16x);
        Assert.SkipWhen(files is null, reason);
        Assert.Throws<NotSupportedException>(() => Qwen35ChatModel.Load(files!));
    }
}
