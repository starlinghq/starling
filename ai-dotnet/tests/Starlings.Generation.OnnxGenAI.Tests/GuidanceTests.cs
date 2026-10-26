using Starlings.Generation.OnnxGenAI;
using Starlings.TestSupport;
using Starlings.Tokenizers;

namespace Starlings.Generation.OnnxGenAI.Tests;

/// <summary>
/// Guided decoding in the decode loop with scripted sequences (no model weights): the constraint reaches the sequence
/// options, unguided requests are unchanged, and fast-forward steps (several tokens per step) go through the same
/// limits, stops and detokenization.
/// </summary>
public sealed class GuidanceTests
{
    private const int ImEnd = 248046;
    private const int EndOfText = 248044;
    private static readonly int[] Eos = [ImEnd, EndOfText];
    private static readonly int[] Prompt = [248045, 846, 198];

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    private static Tokenizer RequireTokenizer()
    {
        var (tokenizer, reason) = Qwen35TestAssets.Tokenizer;
        Assert.SkipWhen(tokenizer is null, reason);
        return tokenizer!;
    }

    private static int[] Ids(Tokenizer tokenizer, string text) => [.. tokenizer.Encode(text, addSpecialTokens: false).Ids];

    private static (TextGenerator Generator, List<GenAISequenceOptions> Options) CreateGrouped(Tokenizer tokenizer, IReadOnlyList<int[]> steps, int contextLength = 1000)
    {
        var options = new List<GenAISequenceOptions>();
        var generator = new TextGenerator(
            o =>
            {
                options.Add(o);
                return new FastForwardSequence(o, steps);
            },
            tokenizer,
            Eos,
            contextLength,
            null);
        return (generator, options);
    }

    [Fact]
    public void FactoriesSetKindDataAndTheNativeType()
    {
        var json = GenerationGuidance.JsonSchema("""{"type":"object"}""");
        var lark = GenerationGuidance.Lark("start: \"a\"");
        var regex = GenerationGuidance.Regex("[0-9]+");
        Assert.Equal((GenerationGuidanceKind.JsonSchema, "json_schema", """{"type":"object"}"""), (json.Kind, json.NativeType, json.Data));
        Assert.Equal((GenerationGuidanceKind.Lark, "lark_grammar"), (lark.Kind, lark.NativeType));
        Assert.Equal((GenerationGuidanceKind.Regex, "regex"), (regex.Kind, regex.NativeType));
        Assert.False(regex.FastForward);
        Assert.Null(regex.MaxFastForwardTokens);
        var ff = regex.WithFastForward(7);
        Assert.True(ff.FastForward);
        Assert.Equal(7, ff.MaxFastForwardTokens);
        Assert.Equal(regex, ff.WithoutFastForward());
        Assert.Throws<ArgumentException>(() => GenerationGuidance.Regex(""));
        Assert.Throws<ArgumentNullException>(() => GenerationGuidance.Lark(null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => regex.WithFastForward(0));
    }

    [Fact]
    public void UnguidedRequestsKeepTheirSequenceOptions()
    {
        var tok = RequireTokenizer();
        int[] content = Ids(tok, "Paris.");
        var (gen, options) = CreateGrouped(tok, [.. content.Select(id => new[] { id }), [ImEnd]]);
        var r = gen.Generate(Prompt, new GenerationOptions { MaxNewTokens = 16 }, Ct);
        Assert.Equal("Paris.", r.Text);
        var o = Assert.Single(options);
        Assert.Null(o.Guidance);
        Assert.Equal(Prompt.Length + 16, o.MaxLength);
    }

    [Fact]
    public void GuidanceWithoutFastForwardReachesTheSequenceWithoutHeadroom()
    {
        var tok = RequireTokenizer();
        int[] content = Ids(tok, "FORCED-100");
        var guidance = GenerationGuidance.Regex("FORCED-[0-9]{3}");
        var (gen, options) = CreateGrouped(tok, [.. content.Select(id => new[] { id }), [ImEnd]]);
        var r = gen.Generate(Prompt, new GenerationOptions { MaxNewTokens = 16, Guidance = guidance }, Ct);
        Assert.Equal("FORCED-100", r.Text);
        var o = Assert.Single(options);
        Assert.Same(guidance, o.Guidance);
        Assert.Equal(Prompt.Length + 16, o.MaxLength);
    }

    [Fact]
    public void FastForwardTokensAreCompletionTokensInOrder()
    {
        var tok = RequireTokenizer();
        int[] a = Ids(tok, "<tool_call>\n<function=");
        int[] b = Ids(tok, "get_weather>\n</function>\n</tool_call>");
        // Step 1 selects a[0]; step 2 appends the rest of a as forced tokens, then selects b[0]; and so on.
        int[][] steps = [[a[0]], [.. a[1..], b[0]], [.. b[1..], ImEnd]];
        var guidance = GenerationGuidance.Lark("start: \"x\"").WithFastForward(40);
        var (gen, options) = CreateGrouped(tok, steps);
        var r = gen.Generate(Prompt, new GenerationOptions { MaxNewTokens = 64, Guidance = guidance }, Ct);
        Assert.Equal(GenerationFinishReason.Stop, r.FinishReason);
        Assert.Equal(ImEnd, r.StopTokenId);
        Assert.Equal([.. a, .. b], r.TokenIds);
        Assert.Equal("<tool_call>\n<function=get_weather>\n</function>\n</tool_call>", r.Text);
        Assert.Equal(a.Length + b.Length, r.Usage.CompletionTokens);
        Assert.Equal(Prompt.Length + 64 + 40, Assert.Single(options).MaxLength);

        // Streaming reassembles to the same text and ids.
        var updates = gen.GenerateStreaming(Prompt, new GenerationOptions { MaxNewTokens = 64, Guidance = guidance }, Ct).ToList();
        Assert.Equal(r.Text, string.Concat(updates.Select(u => u.Text)));
        Assert.Equal(r.TokenIds, updates.SelectMany(u => u.TokenIds));
    }

    [Fact]
    public void ForcedTokensBeyondMaxNewTokensAreCutWithFinishReasonLength()
    {
        var tok = RequireTokenizer();
        int[] text = Ids(tok, "one two three four five six");
        int[][] steps = [[text[0]], [.. text[1..], ImEnd]];
        var guidance = GenerationGuidance.Regex(".*").WithFastForward(20);
        var (gen, _) = CreateGrouped(tok, steps);
        var r = gen.Generate(Prompt, new GenerationOptions { MaxNewTokens = 3, Guidance = guidance }, Ct);
        Assert.Equal(GenerationFinishReason.Length, r.FinishReason);
        Assert.Equal(text[..3], r.TokenIds);
        Assert.Equal(tok.Decode(text.AsSpan(0, 3), false), r.Text);
    }

    [Fact]
    public void AStopIdAmongForcedTokensEndsGeneration()
    {
        var tok = RequireTokenizer();
        int[] text = Ids(tok, "done");
        int[][] steps = [[text[0]], [.. text[1..], EndOfText, 11, 12]];
        var (gen, _) = CreateGrouped(tok, steps);
        var r = gen.Generate(Prompt, new GenerationOptions { MaxNewTokens = 32, Guidance = GenerationGuidance.Regex("done").WithFastForward(8) }, Ct);
        Assert.Equal(GenerationFinishReason.Stop, r.FinishReason);
        Assert.Equal(EndOfText, r.StopTokenId);
        Assert.Equal(text, r.TokenIds);
    }

    [Fact]
    public void FastForwardPastItsLimitFails()
    {
        var tok = RequireTokenizer();
        int[][] steps = [[11], [12, 13, 14, 15]];
        var (gen, _) = CreateGrouped(tok, steps);
        var ex = Assert.Throws<GenAIModelException>(() =>
            gen.Generate(Prompt, new GenerationOptions { MaxNewTokens = 32, Guidance = GenerationGuidance.Regex(".*").WithFastForward(2) }, Ct));
        Assert.Contains("at most 2 forced tokens", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheContextCheckCountsTheFastForwardHeadroom()
    {
        var tok = RequireTokenizer();
        var (gen, _) = CreateGrouped(tok, [[ImEnd]], contextLength: 100);
        var unguided = gen.Generate(Prompt, new GenerationOptions { MaxNewTokens = 90 }, Ct);
        Assert.Equal(GenerationFinishReason.Stop, unguided.FinishReason);
        var ex = Assert.Throws<GenerationContextLengthException>(() =>
            gen.Generate(Prompt, new GenerationOptions { MaxNewTokens = 90, Guidance = GenerationGuidance.Regex(".*").WithFastForward(10) }, Ct));
        Assert.Contains("fast-forward headroom", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void GuidanceCannotBeCombinedWithLogprobs()
    {
        var options = new GenerationOptions { Guidance = GenerationGuidance.Regex("a"), ReturnLogprobs = true };
        Assert.Throws<ArgumentException>(options.Validate);
    }

    [Fact]
    public void TheUnavailableMessageNamesTheMissingFiles()
    {
        Assert.Equal(["tokenizer.json", "tokenizer_config.json"], GenAIModel.GuidanceFiles);
        string one = GenAIModel.GuidanceUnavailableMessage("/m", ["tokenizer_config.json"]);
        Assert.Contains("tokenizer_config.json is missing", one, StringComparison.Ordinal);
        string two = GenAIModel.GuidanceUnavailableMessage("/m", GenAIModel.GuidanceFiles);
        Assert.Contains("tokenizer.json and tokenizer_config.json are missing", two, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("tokenizer")]
    public void TheConfigSummaryReadsTheTokenizerDirectory(string? tokenizerDirectory)
    {
        string directory = Directory.CreateTempSubdirectory("starling-genai-config-").FullName;
        try
        {
            string field = tokenizerDirectory is null ? string.Empty : $", \"tokenizer_dir\": \"{tokenizerDirectory}\"";
            string path = Path.Combine(directory, "genai_config.json");
            File.WriteAllText(path, $$$"""{"model": {"type": "qwen3_5", "context_length": 1000, "vocab_size": 10, "eos_token_id": [1, 2]{{{field}}}}}""");
            Assert.Equal(tokenizerDirectory, GenAIConfigSummary.Read(path).TokenizerDirectory);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    /// <summary>A scripted sequence whose steps append groups of tokens (forced tokens, then the selected one).</summary>
    private sealed class FastForwardSequence(GenAISequenceOptions options, IReadOnlyList<int[]> steps) : IGenerationSequence
    {
        private int _length;
        private int _step;

        public int VocabularySize => 248320;

        public bool IsDone { get; private set; }

        public void AppendTokens(ReadOnlySpan<int> tokenIds) => _length += tokenIds.Length;

        public int GenerateNextToken()
        {
            int[] group = GenerateNextTokens();
            return group.Length == 1 ? group[0] : throw new InvalidOperationException("A step without fast-forward appended several tokens.");
        }

        public int[] GenerateNextTokens()
        {
            if (IsDone || _length >= options.MaxLength)
            {
                throw new InvalidOperationException("Stepped past the end of the sequence.");
            }

            int[] group = steps[_step++];
            _length += group.Length;
            if (_length > options.MaxLength)
            {
                throw new InvalidOperationException("Forced tokens overran the sequence's maximum length.");
            }

            IsDone = group.Any(Eos.Contains) || _length == options.MaxLength;
            return group;
        }

        public void CopyLastLogits(Span<float> destination) => destination.Clear();

        public void Dispose()
        {
        }
    }
}
