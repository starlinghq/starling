using System.Security.Cryptography;
using System.Text.Json;
using Starlings.Tokenizers;

namespace Starlings.Models.EmbeddingGemma2;

/// <summary>
/// The owned tokenizer (<c>Starlings.Tokenizers</c>, profile <c>gemma-bpe-embeddinggemma2</c>) as an
/// <see cref="IEmbeddingGemma2Tokenizer"/>: the pinned <c>tokenizer.json</c>, <c>&lt;bos&gt;</c> and <c>&lt;eos&gt;</c>
/// added, no truncation, no padding (the encoder pads and rejects inputs above 8,192 tokens).
/// </summary>
/// <remarks>
/// <para>
/// The text is tokenized exactly as given; a prompt is part of the text (<see cref="EmbeddingGemma2TextEmbedder"/>
/// prepends it). Literal media placeholders in the text (<c>&lt;|image|&gt;</c>, <c>&lt;|audio|&gt;</c>,
/// <c>&lt;|video|&gt;</c> and the boundary tokens) tokenize to their special ids, which the encoder rejects: this
/// package embeds text only.
/// </para>
/// <para>
/// <b>Cheap rejection of long text.</b> <see cref="Tokenize(string, int, int)"/> and <see cref="Validate"/> reject text
/// whose length alone proves it has more tokens than the limit, without tokenizing it: no token of the pinned file
/// covers more than <see cref="MaxCharactersPerToken"/> UTF-16 code units (31), so a text of <c>n</c> code units has at
/// least <c>ceil(n / 31)</c> text tokens plus <c>&lt;bos&gt;</c> and <c>&lt;eos&gt;</c>. With the default limit of 8,192
/// tokens, text above 253,890 code units is rejected at once; shorter text is tokenized and checked exactly. The bound
/// is computed from <c>tokenizer.json</c> at load and is off (null) for a file whose normalizer, pre-tokenizer, model or
/// added tokens could drop characters.
/// </para>
/// <para>Instances are immutable and safe for concurrent calls (the owned tokenizer is immutable).</para>
/// </remarks>
public sealed class EmbeddingGemma2Tokenizer : IEmbeddingGemma2Tokenizer, IChunkingTokenizer
{
    private readonly Tokenizer _tokenizer;

    private EmbeddingGemma2Tokenizer(Tokenizer tokenizer, int? maxCharactersPerToken)
    {
        _tokenizer = tokenizer.WithPadding(null).WithTruncation(null);
        SpecialTokenCount = _tokenizer.Encode(string.Empty).Count;
        MaxCharactersPerToken = maxCharactersPerToken;
    }

    /// <summary>Tokens added around every sequence: <c>&lt;bos&gt;</c> and <c>&lt;eos&gt;</c> (2).</summary>
    public int SpecialTokenCount { get; }

    /// <summary>
    /// UTF-16 code units the longest token of the file covers (31 for the pinned file), or null when the file's pipeline
    /// could drop characters and no length bound holds. Used to reject text that is certainly too long without
    /// tokenizing it.
    /// </summary>
    public int? MaxCharactersPerToken { get; }

    /// <summary>
    /// Loads <c>tokenizer.json</c> from <paramref name="path"/>. The file must match <paramref name="expectedSha256"/>
    /// (lower-case hex; default <see cref="EmbeddingGemma2ModelAssets.TokenizerSha256"/>, the pinned file) or
    /// <see cref="InvalidDataException"/> is thrown. Pass <see langword="null"/> to skip the check (another tokenizer
    /// file is not the qualified profile).
    /// </summary>
    public static EmbeddingGemma2Tokenizer FromFile(string path, string? expectedSha256 = EmbeddingGemma2ModelAssets.TokenizerSha256)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        byte[] bytes = File.ReadAllBytes(path);
        if (expectedSha256 is not null)
        {
            string actual = Convert.ToHexStringLower(SHA256.HashData(bytes));
            if (!string.Equals(actual, expectedSha256, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidDataException($"{path}: SHA-256 {actual} does not match the expected {expectedSha256}.");
            }
        }

        return new EmbeddingGemma2Tokenizer(Tokenizer.FromUtf8Json(bytes), ComputeMaxCharactersPerToken(bytes));
    }

    /// <inheritdoc />
    public EmbeddingGemma2Input Tokenize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var encoding = _tokenizer.Encode(text);
        var ids = new long[encoding.Count];
        var mask = new long[encoding.Count];
        for (int i = 0; i < ids.Length; i++)
        {
            ids[i] = encoding.Ids[i];
            mask[i] = encoding.AttentionMask[i];
        }

        return new EmbeddingGemma2Input(ids, mask);
    }

    /// <summary>
    /// <see cref="Tokenize(string)"/> with a limit: text whose length alone proves it has more than
    /// <paramref name="maxTokens"/> tokens is rejected before it is tokenized (the exception's
    /// <see cref="InputTooLongException.TokenCountIsLowerBound"/> is then true), and any other text above the limit after
    /// tokenizing it.
    /// </summary>
    /// <param name="text">The text exactly as it is embedded (prompt included).</param>
    /// <param name="maxTokens">Tokens at most, special tokens included.</param>
    /// <param name="inputIndex">Position reported in the exception.</param>
    /// <exception cref="EmbeddingGemma2InputTooLongException">The text has more than <paramref name="maxTokens"/> tokens.</exception>
    public EmbeddingGemma2Input Tokenize(string text, int maxTokens, int inputIndex = 0)
    {
        ArgumentNullException.ThrowIfNull(text);
        int bound = LowerBound(text);
        if (bound > maxTokens)
        {
            throw new EmbeddingGemma2InputTooLongException(inputIndex, bound, maxTokens, isLowerBound: true);
        }

        var input = Tokenize(text);
        if (input.Length > maxTokens)
        {
            throw new EmbeddingGemma2InputTooLongException(inputIndex, input.Length, maxTokens);
        }

        return input;
    }

    /// <summary>
    /// Checks <paramref name="text"/> against <paramref name="maxTokens"/> without throwing: from its length alone when
    /// that proves it too long (<see cref="TokenLimitCheck.TokenCountIsLowerBound"/>), else from its exact count.
    /// </summary>
    /// <param name="text">The text exactly as it is embedded (prompt included).</param>
    /// <param name="maxTokens">Tokens at most, special tokens included.</param>
    public TokenLimitCheck Validate(string text, int maxTokens)
    {
        ArgumentNullException.ThrowIfNull(text);
        int bound = LowerBound(text);
        return bound > maxTokens
            ? new TokenLimitCheck(bound, maxTokens, TokenCountIsLowerBound: true)
            : new TokenLimitCheck(CountTokens(text), maxTokens);
    }

    /// <summary>Tokens of <paramref name="text"/>, <c>&lt;bos&gt;</c> and <c>&lt;eos&gt;</c> included. Tokenizes the whole text.</summary>
    public int CountTokens(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return _tokenizer.Encode(text).Count;
    }

    /// <inheritdoc />
    public IReadOnlyList<TokenSpan> GetTokenSpans(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var encoding = _tokenizer.Encode(text, addSpecialTokens: false);
        var spans = new TokenSpan[encoding.Count];
        for (int i = 0; i < spans.Length; i++)
        {
            var offset = encoding.Offsets[i];
            spans[i] = new TokenSpan(offset.Start, offset.End);
        }

        return spans;
    }

    /// <summary>Tokens <paramref name="text"/> has at least (special tokens included), or 0 when no bound is known.</summary>
    internal int LowerBound(string text) =>
        MaxCharactersPerToken is { } max && text.Length > 0
            ? (int)Math.Min(int.MaxValue, SpecialTokenCount + (((long)text.Length + max - 1) / max))
            : 0;

    /// <summary>
    /// The longest token of a BPE <c>tokenizer.json</c> in UTF-16 code units (vocabulary and added tokens), when every
    /// character of the input ends up inside some token: a normalizer of string replacements that never shorten the
    /// text, a pre-tokenizer that only splits, byte fallback or an unknown token that is not fused, and added tokens that
    /// strip nothing. Otherwise null.
    /// </summary>
    internal static int? ComputeMaxCharactersPerToken(byte[] json)
    {
        try
        {
            using var document = JsonDocument.Parse(json);
            var root = document.RootElement;
            if (!KeepsEveryCharacter(root, "normalizer", NormalizerKeeps) || !KeepsEveryCharacter(root, "pre_tokenizer", PreTokenizerKeeps))
            {
                return null;
            }

            var model = root.GetProperty("model");
            if (model.GetProperty("type").GetString() != "BPE")
            {
                return null;
            }

            // Without byte fallback an unknown character becomes the unknown token, one per character unless fused, or
            // is dropped when the model has no unknown token.
            bool hasUnknownToken = model.TryGetProperty("unk_token", out var unknown) && unknown.ValueKind == JsonValueKind.String;
            if (!IsTrue(model, "byte_fallback") && (IsTrue(model, "fuse_unk") || !hasUnknownToken))
            {
                return null;
            }

            // At least 2, so one unknown token per character covers a surrogate pair.
            int longest = 2;
            foreach (var entry in model.GetProperty("vocab").EnumerateObject())
            {
                longest = Math.Max(longest, entry.Name.Length);
            }

            if (root.TryGetProperty("added_tokens", out var added) && added.ValueKind == JsonValueKind.Array)
            {
                foreach (var token in added.EnumerateArray())
                {
                    if (IsTrue(token, "lstrip") || IsTrue(token, "rstrip") || IsTrue(token, "single_word"))
                    {
                        return null;
                    }

                    longest = Math.Max(longest, token.GetProperty("content").GetString()?.Length ?? 0);
                }
            }

            return longest;
        }
        catch (Exception ex) when (ex is JsonException or KeyNotFoundException or InvalidOperationException)
        {
            return null;
        }
    }

    private static bool IsTrue(JsonElement element, string name) =>
        element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.True;

    private static bool KeepsEveryCharacter(JsonElement root, string name, Func<JsonElement, bool> keeps) =>
        !root.TryGetProperty(name, out var step) || step.ValueKind == JsonValueKind.Null || keeps(step);

    private static bool NormalizerKeeps(JsonElement step) => step.GetProperty("type").GetString() switch
    {
        "Sequence" => step.GetProperty("normalizers").EnumerateArray().All(NormalizerKeeps),
        "Replace" => step.GetProperty("pattern").TryGetProperty("String", out var pattern)
            && (step.GetProperty("content").GetString()?.Length ?? 0) >= (pattern.GetString()?.Length ?? int.MaxValue),
        _ => false,
    };

    private static bool PreTokenizerKeeps(JsonElement step) => step.GetProperty("type").GetString() switch
    {
        "Sequence" => step.GetProperty("pretokenizers").EnumerateArray().All(PreTokenizerKeeps),
        "Split" => step.GetProperty("behavior").GetString() is "MergedWithPrevious" or "MergedWithNext" or "Isolated" or "Contiguous",
        _ => false,
    };
}
