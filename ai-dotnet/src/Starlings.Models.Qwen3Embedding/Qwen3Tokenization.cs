using System.Security.Cryptography;
using Starlings.Tokenizers;

namespace Starlings.Models.Qwen3Embedding;

/// <summary>
/// The inputs both profiles build, without a graph: an embedding row is <c>prompt + text</c> encoded with the pinned
/// embedding tokenizer (which appends <c>&lt;|endoftext|&gt;</c>), truncated from the right to the token limit with
/// the appended token kept; a reranker row is <c>prefix + encode("&lt;Instruct&gt;: {i}\n&lt;Query&gt;: {q}\n&lt;Document&gt;: {d}") + suffix</c>,
/// each part encoded without special tokens, and only the middle part truncated (the model card's transformers snippet).
/// </summary>
internal sealed class Qwen3Tokenization
{
    private readonly Tokenizer _truncating;
    private readonly Tokenizer _untruncated;
    private readonly long[] _prefix;
    private readonly long[] _suffix;

    public Qwen3Tokenization(Qwen3EmbeddingProfile profile, Tokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentNullException.ThrowIfNull(tokenizer);
        Profile = profile;
        _untruncated = tokenizer.WithPadding(null).WithTruncation(null);
        if (_untruncated.TokenToId("<|endoftext|>") != Qwen3EmbeddingProfile.EndOfTextTokenId
            || _untruncated.TokenToId("yes") != Qwen3EmbeddingProfile.YesTokenId
            || _untruncated.TokenToId("no") != Qwen3EmbeddingProfile.NoTokenId
            || _untruncated.GetVocabularySize(true) != Qwen3EmbeddingProfile.VocabularySize)
        {
            throw new InvalidDataException($"tokenizer.json does not have the special tokens and vocabulary of {profile.Id}.");
        }

        if (profile.Task == Qwen3EmbeddingTask.Embedding)
        {
            _truncating = _untruncated.WithTruncation(new TruncationOptions(profile.MaxTokens, TruncationStrategy.LongestFirst, TruncationDirection.Right));
            _prefix = [];
            _suffix = [];
        }
        else
        {
            _prefix = Ids(_untruncated.Encode(Qwen3EmbeddingProfile.RerankPrefix, addSpecialTokens: false).Ids);
            _suffix = Ids(_untruncated.Encode(Qwen3EmbeddingProfile.RerankSuffix, addSpecialTokens: false).Ids);
            int budget = profile.MaxTokens - _prefix.Length - _suffix.Length;
            _truncating = _untruncated.WithTruncation(new TruncationOptions(budget, TruncationStrategy.LongestFirst, TruncationDirection.Right));
        }
    }

    public Qwen3EmbeddingProfile Profile { get; }

    /// <summary>The reranker template's tokens before the content (empty for the embedding profile).</summary>
    public IReadOnlyList<long> RerankPrefixIds => _prefix;

    /// <summary>The reranker template's tokens after the content (empty for the embedding profile).</summary>
    public IReadOnlyList<long> RerankSuffixIds => _suffix;

    /// <summary>Opens <c>tokenizer.json</c> after checking its SHA-256 against the profile's lock.</summary>
    public static Qwen3Tokenization Open(Qwen3EmbeddingProfile profile, string path, bool verifyDigest = true)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentNullException.ThrowIfNull(path);
        if (verifyDigest)
        {
            using FileStream stream = File.OpenRead(path);
            string expected = profile.RuntimeLock.GetFile(Qwen3EmbeddingProfile.TokenizerFile).Sha256;
            if (!string.Equals(Convert.ToHexStringLower(SHA256.HashData(stream)), expected, StringComparison.Ordinal))
            {
                throw new InvalidDataException($"{path} is not the pinned tokenizer.json of {profile.Repository}@{profile.Revision}.");
            }
        }

        return new Qwen3Tokenization(profile, Tokenizer.FromFile(path));
    }

    /// <summary>The prompt text of a named prompt.</summary>
    public string PromptText(string prompt)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        return Profile.Prompts.TryGetValue(prompt, out string? text)
            ? text
            : throw new ArgumentException($"Prompt '{prompt}' is not declared by {Profile.Id} (declared: {string.Join(", ", Profile.Prompts.Keys)}).", nameof(prompt));
    }

    /// <summary>An embedding row: <paramref name="prefix"/> + <paramref name="text"/>, truncated, ending in <c>&lt;|endoftext|&gt;</c>.</summary>
    public Qwen3Input TokenizeText(string text, string prefix)
    {
        RequireTask(Qwen3EmbeddingTask.Embedding);
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(prefix);
        return Row(Ids(_truncating.Encode(prefix + text).Ids));
    }

    /// <summary>Tokens of an embedding row before truncation.</summary>
    public int CountText(string text, string prefix)
    {
        RequireTask(Qwen3EmbeddingTask.Embedding);
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(prefix);
        return _untruncated.Encode(prefix + text).Count;
    }

    /// <summary>The reranker's user content for one pair.</summary>
    public static string RerankContent(string instruction, string query, string document) =>
        $"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}";

    /// <summary>A reranker row: prefix, the content truncated to the remaining budget, suffix.</summary>
    public Qwen3Input TokenizePair(string instruction, string query, string document)
    {
        RequireTask(Qwen3EmbeddingTask.Reranking);
        ArgumentNullException.ThrowIfNull(instruction);
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(document);
        long[] content = Ids(_truncating.Encode(RerankContent(instruction, query, document), addSpecialTokens: false).Ids);
        return Row([.. _prefix, .. content, .. _suffix]);
    }

    /// <summary>Tokens of a reranker row before truncation.</summary>
    public int CountPair(string instruction, string query, string document)
    {
        RequireTask(Qwen3EmbeddingTask.Reranking);
        return _prefix.Length + _suffix.Length
            + _untruncated.Encode(RerankContent(instruction, query, document), addSpecialTokens: false).Count;
    }

    /// <summary>Checks a row before it reaches the graph: lengths, id range, mask 1 then 0 (right padding with the pad id).</summary>
    public void Validate(Qwen3Input input)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (input.Length < 1 || input.Length > Profile.MaxTokens || input.AttentionMask.Length != input.Length)
        {
            throw new ArgumentException($"A row needs 1 to {Profile.MaxTokens} ids and a mask of the same length.", nameof(input));
        }

        var ids = input.InputIds.Span;
        var mask = input.AttentionMask.Span;
        if (mask[0] != 1)
        {
            throw new ArgumentException("The first token of a row must be attended (right padding only).", nameof(input));
        }

        bool padding = false;
        for (int i = 0; i < ids.Length; i++)
        {
            if (ids[i] < 0 || ids[i] >= Qwen3EmbeddingProfile.VocabularySize || mask[i] is not (0 or 1)
                || (mask[i] == 1 && padding) || (mask[i] == 0 && ids[i] != Qwen3EmbeddingProfile.PadTokenId))
            {
                throw new ArgumentException($"Invalid token id, mask value or padding at position {i}.", nameof(input));
            }

            padding |= mask[i] == 0;
        }
    }

    private void RequireTask(Qwen3EmbeddingTask task)
    {
        if (Profile.Task != task)
        {
            throw new InvalidOperationException($"{Profile.Id} is a {Profile.Task} profile, not {task}.");
        }
    }

    private static Qwen3Input Row(long[] ids)
    {
        var mask = new long[ids.Length];
        Array.Fill(mask, 1L);
        return new Qwen3Input(ids, mask);
    }

    private static long[] Ids(IReadOnlyList<int> ids)
    {
        var result = new long[ids.Count];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = ids[i];
        }

        return result;
    }
}
