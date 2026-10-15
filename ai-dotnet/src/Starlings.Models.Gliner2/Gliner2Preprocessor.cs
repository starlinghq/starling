using System.Collections.Concurrent;
using System.Security.Cryptography;
using Starlings.Tokenizers;

namespace Starlings.Models.Gliner2;

/// <summary>
/// Builds <see cref="Gliner2ModelInput"/> as GLiNER2 2.0.0 does for one entity schema (<c>processor.py:1024</c>,
/// <c>:1204</c>, <c>:1374</c>), with the owned tokenizer (profile <c>gliner2-base-v1-unigram</c>) in place of
/// <c>DebertaV2TokenizerFast</c>: each schema token and each word is tokenized alone without special tokens, and each
/// piece is mapped with the vocabulary (unknown pieces to <c>[UNK]</c>, id 3), like <c>tokenize</c> followed by
/// <c>convert_tokens_to_ids</c>.
/// </summary>
/// <remarks>
/// Safe to use from several threads (the owned tokenizer is immutable). Each token is tokenized alone, so its ids
/// depend on the token string only: the ids of up to <see cref="MaxCachedTokens"/> distinct tokens (labels, schema
/// tokens, words) of at most <see cref="MaxCachedTokenLength"/> UTF-16 units are memoized, as Hugging Face tokenizers
/// cache per-word results; a cached token gives the ids it gave when it was computed.
/// </remarks>
internal sealed class Gliner2Preprocessor
{
    /// <summary>Most distinct tokens whose ids are memoized; later tokens are tokenized on every call.</summary>
    internal const int MaxCachedTokens = 16_384;

    /// <summary>Longest token (UTF-16 units) whose ids are memoized.</summary>
    internal const int MaxCachedTokenLength = 64;

    /// <summary><c>[UNK]</c> id in the GLiNER2 vocabulary.</summary>
    public const long UnknownId = 3;

    /// <summary>The schema's prompt token for entities.</summary>
    public const string EntitiesPrompt = "entities";

    private readonly Tokenizer _tokenizer;
    private readonly ConcurrentDictionary<string, long[]> _ids = new(StringComparer.Ordinal);

    private Gliner2Preprocessor(Tokenizer tokenizer)
    {
        // Per-word encoding never truncates or pads (tokenize() does neither).
        _tokenizer = tokenizer.WithPadding(null).WithTruncation(null);
    }

    /// <summary>Loads <c>tokenizer.json</c>, optionally checking its SHA-256 (lower-case hex).</summary>
    /// <exception cref="InvalidDataException">The file does not match <paramref name="expectedSha256"/>.</exception>
    public static Gliner2Preprocessor FromFile(string path, string? expectedSha256)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        byte[] bytes = File.ReadAllBytes(path);
        if (expectedSha256 is not null)
        {
            string actual = Convert.ToHexStringLower(SHA256.HashData(bytes));
            if (!string.Equals(actual, expectedSha256, StringComparison.Ordinal))
            {
                throw new InvalidDataException($"{path}: SHA-256 {actual} does not match the expected {expectedSha256}.");
            }
        }

        return new Gliner2Preprocessor(Tokenizer.FromUtf8Json(bytes));
    }

    /// <summary>
    /// Validates the labels and returns them deduplicated in caller order (GLiNER2 keeps them as dict keys: first
    /// occurrence wins, case is kept).
    /// </summary>
    /// <exception cref="ArgumentException">No labels, or a label that is null, empty or whitespace only.</exception>
    public static IReadOnlyList<string> NormalizeLabels(IReadOnlyList<string> labels, string parameterName)
    {
        ArgumentNullException.ThrowIfNull(labels, parameterName);
        if (labels.Count == 0)
        {
            throw new ArgumentException("At least one entity label is required.", parameterName);
        }

        var seen = new HashSet<string>(StringComparer.Ordinal);
        var result = new List<string>(labels.Count);
        for (int i = 0; i < labels.Count; i++)
        {
            string? label = labels[i];
            if (string.IsNullOrWhiteSpace(label))
            {
                throw new ArgumentException($"Label {i} is null, empty or whitespace; labels must name an entity type.", parameterName);
            }

            if (seen.Add(label))
            {
                result.Add(label);
            }
        }

        return result.AsReadOnly();
    }

    /// <summary>Builds the model input for one text and one label set.</summary>
    /// <param name="text">The caller's text (valid Unicode).</param>
    /// <param name="labels">Labels already normalized by <see cref="NormalizeLabels"/>.</param>
    /// <param name="descriptions">Optional label descriptions (label to text); labels without one get none.</param>
    /// <param name="maxWords">
    /// GLiNER2's <c>max_len</c> word cap (words after the first <paramref name="maxWords"/> are dropped before
    /// encoding). Outside the contract; used only for fidelity evidence. Null: no cap (GLiNER2's default).
    /// </param>
    public Gliner2ModelInput Build(string text, IReadOnlyList<string> labels, IReadOnlyDictionary<string, string>? descriptions, int? maxWords = null)
    {
        int originalScalars = Gliner2Text.ToScalars(text, nameof(text)).Length;
        string normalized = Gliner2Text.Normalize(text);
        int[] scalars = Gliner2Text.ToScalars(normalized, nameof(text));
        var words = Gliner2Text.SplitWords(scalars);
        if (maxWords is int cap && words.Count > cap)
        {
            words.RemoveRange(cap, words.Count - cap);
        }

        // ( [P] prompt ( [E] l1 [E] l2 ... ) ), then [SEP_TEXT]. With descriptions the prompt is one string with the
        // special token inside: "entities [DESCRIPTION] label: text ...", in label order.
        string prompt = EntitiesPrompt;
        if (descriptions is not null)
        {
            foreach (string label in labels)
            {
                if (descriptions.TryGetValue(label, out string? description) && description is not null)
                {
                    prompt += $" [DESCRIPTION] {label}: {description}";
                }
            }
        }

        var schemaTokens = new List<string>(labels.Count * 2 + 6) { "(", "[P]", prompt, "(" };
        foreach (string label in labels)
        {
            schemaTokens.Add("[E]");
            schemaTokens.Add(label);
        }

        schemaTokens.Add(")");
        schemaTokens.Add(")");

        var ids = new List<long>(words.Count * 2 + schemaTokens.Count + 1);
        var schemaIndex = new long[labels.Count + 1];
        int slot = 0;
        for (int j = 0; j < schemaTokens.Count; j++)
        {
            // Marker positions by structure, not by text: index 1 is [P], then [E] at 4, 6, ... before the closing ")" ")".
            bool marker = j == 1 || (j >= 4 && j < schemaTokens.Count - 2 && (j - 4) % 2 == 0);
            if (marker)
            {
                schemaIndex[slot++] = ids.Count;
            }

            AddPieces(schemaTokens[j], ids);
        }

        AddPieces("[SEP_TEXT]", ids);
        var wordIndex = new long[words.Count];
        var starts = new int[words.Count];
        var ends = new int[words.Count];
        var wordTexts = new string[words.Count];
        for (int w = 0; w < words.Count; w++)
        {
            // A word with no pieces keeps the next position as a placeholder, as GLiNER2 does.
            wordIndex[w] = ids.Count;
            starts[w] = words[w].Start;
            ends[w] = words[w].End;
            wordTexts[w] = words[w].Word;
            AddPieces(words[w].Word, ids);
        }

        return new Gliner2ModelInput(
            text,
            normalized,
            scalars,
            originalScalars,
            labels,
            schemaTokens.AsReadOnly(),
            wordTexts,
            starts,
            ends,
            ids.ToArray(),
            wordIndex,
            schemaIndex);
    }

    private void AddPieces(string token, List<long> ids) => ids.AddRange(PieceIds(token));

    /// <summary>The token's pieces mapped with the vocabulary (unknown pieces to <see cref="UnknownId"/>); do not modify.</summary>
    internal long[] PieceIds(string token)
    {
        if (_ids.TryGetValue(token, out long[]? cached))
        {
            return cached;
        }

        var encoding = _tokenizer.Encode(token, addSpecialTokens: false);
        var pieces = encoding.Tokens;
        var result = new long[encoding.Count];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = _tokenizer.TokenToId(pieces[i]) ?? UnknownId;
        }

        if (token.Length <= MaxCachedTokenLength && _ids.Count < MaxCachedTokens)
        {
            _ids.TryAdd(token, result);
        }

        return result;
    }
}
