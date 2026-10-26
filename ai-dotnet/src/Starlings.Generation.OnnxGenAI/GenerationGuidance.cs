namespace Starlings.Generation.OnnxGenAI;

/// <summary>The constraint language of a <see cref="GenerationGuidance"/>.</summary>
public enum GenerationGuidanceKind
{
    /// <summary>A JSON Schema: the completion is one JSON value that validates against it (GenAI type <c>json_schema</c>).</summary>
    JsonSchema,

    /// <summary>A Lark grammar in llguidance's dialect (GenAI type <c>lark_grammar</c>).</summary>
    Lark,

    /// <summary>A regular expression the whole completion matches (GenAI type <c>regex</c>).</summary>
    Regex,
}

/// <summary>
/// Experimental: a constraint on the completion, applied by ONNX Runtime GenAI's llguidance integration
/// (<c>GeneratorParams.SetGuidance</c>). At every step GenAI masks the tokens the constraint does not allow before it
/// selects one, and once the constraint is complete it allows only an end-of-sequence id.
/// </summary>
/// <remarks>
/// <para>
/// GenAI builds the constraint's token table from <c>tokenizer.json</c> and <c>tokenizer_config.json</c> in the model
/// directory (<see cref="GenAIModel.GuidanceFiles"/>). A model loaded from a directory without them rejects a guided
/// request with <see cref="NotSupportedException"/> before any native call (<see cref="GenAIModel.GuidanceMissingFiles"/>).
/// </para>
/// <para>
/// The constraint is compiled by GenAI when the request's generator is created; a constraint GenAI rejects fails that
/// request with <see cref="GenAIModelException"/>. Unguided requests (<see cref="GenerationOptions.Guidance"/> null) are
/// not affected by any of this.
/// </para>
/// </remarks>
public sealed record GenerationGuidance
{
    private GenerationGuidance(GenerationGuidanceKind kind, string data)
    {
        Kind = kind;
        Data = data;
    }

    /// <summary>The constraint language.</summary>
    public GenerationGuidanceKind Kind { get; }

    /// <summary>The schema, grammar or pattern text.</summary>
    public string Data { get; }

    /// <summary>
    /// The most tokens one decode step may append through fast-forwarding (see <see cref="WithFastForward"/>), or null
    /// (default) for none: every step then appends exactly one selected token.
    /// </summary>
    public int? MaxFastForwardTokens { get; private init; }

    /// <summary>True when fast-forward tokens are on (<see cref="MaxFastForwardTokens"/> is set).</summary>
    public bool FastForward => MaxFastForwardTokens is not null;

    /// <summary>The type string GenAI's <c>SetGuidance</c> takes.</summary>
    internal string NativeType => Kind switch
    {
        GenerationGuidanceKind.JsonSchema => "json_schema",
        GenerationGuidanceKind.Lark => "lark_grammar",
        _ => "regex",
    };

    /// <summary>A JSON Schema constraint.</summary>
    /// <param name="schema">The schema as JSON text (non-empty).</param>
    public static GenerationGuidance JsonSchema(string schema) => new(GenerationGuidanceKind.JsonSchema, Require(schema, nameof(schema)));

    /// <summary>A Lark grammar constraint (llguidance dialect; the grammar's <c>start</c> rule is the completion).</summary>
    /// <param name="grammar">The grammar text (non-empty).</param>
    public static GenerationGuidance Lark(string grammar) => new(GenerationGuidanceKind.Lark, Require(grammar, nameof(grammar)));

    /// <summary>A regular expression constraint (llguidance regex syntax; the whole completion matches).</summary>
    /// <param name="pattern">The pattern (non-empty).</param>
    public static GenerationGuidance Regex(string pattern) => new(GenerationGuidanceKind.Regex, Require(pattern, nameof(pattern)));

    /// <summary>
    /// Experimental: this constraint with fast-forward tokens on. When the constraint leaves only one possible
    /// continuation (for example the closing tags of a tool call), GenAI appends those tokens at the start of the next
    /// step without selecting each one, and the decode loop returns them as completion tokens in order, before that
    /// step's selected token.
    /// </summary>
    /// <param name="maxTokensPerStep">
    /// The most tokens the constraint can force in a row (at least 1). The decode loop sizes GenAI's sequence buffers for
    /// <see cref="GenerationOptions.MaxNewTokens"/> plus this many tokens, because GenAI appends forced tokens without
    /// checking the sequence's maximum length; a request whose prompt, <see cref="GenerationOptions.MaxNewTokens"/> and
    /// this headroom exceed the model's context length fails with <see cref="GenerationContextLengthException"/>. A step
    /// that appends more forced tokens than this fails the request with <see cref="GenAIModelException"/>, but GenAI has
    /// written them by then, so the bound must hold: the caller derives it from the constraint (for example
    /// <c>Qwen35ToolGrammar</c> in Starlings.Models.Qwen35 does so for its tool-call grammar). Every token holds at
    /// least one byte, so the longest text the constraint can force in a row, in UTF-8 bytes, is always a bound.
    /// </param>
    /// <remarks>
    /// Forced tokens have no log probabilities, so a request with fast-forward guidance cannot set
    /// <see cref="GenerationOptions.ReturnLogprobs"/>. A completion that crosses <see cref="GenerationOptions.MaxNewTokens"/>
    /// inside a step's forced tokens is cut at that many tokens and ends with <see cref="GenerationFinishReason.Length"/>.
    /// </remarks>
    public GenerationGuidance WithFastForward(int maxTokensPerStep)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxTokensPerStep);
        return this with { MaxFastForwardTokens = maxTokensPerStep };
    }

    /// <summary>This constraint with fast-forward tokens off (the default).</summary>
    public GenerationGuidance WithoutFastForward() => this with { MaxFastForwardTokens = null };

    private static string Require(string value, string name)
    {
        ArgumentException.ThrowIfNullOrEmpty(value, name);
        return value;
    }
}
