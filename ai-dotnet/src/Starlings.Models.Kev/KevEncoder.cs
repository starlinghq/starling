using System.Collections.Immutable;
using Starlings.Tokenizers;

namespace Starlings.Models.Kev;

/// <summary>One question's branch: the tokens that follow the state in its causal row.</summary>
/// <param name="Ids">Branch token ids: <c>&lt;q&gt; instruction (&lt;opt&gt; option &lt;/opt&gt;)* &lt;decide&gt;</c>.</param>
/// <param name="Positions">Position ids, continuing the state's (the state has positions 0 to its length minus one).</param>
/// <param name="Decide">Offset of <c>&lt;decide&gt;</c> within the branch (its last token).</param>
/// <param name="Options">Offset of each option's <c>&lt;/opt&gt;</c> within the branch.</param>
public sealed record KevRow(ImmutableArray<int> Ids, ImmutableArray<int> Positions, int Decide, ImmutableArray<int> Options);

/// <summary>A record encoded as <c>kev.model.rows_of(encode(...))</c> gives it: the state, then one branch per question.</summary>
/// <param name="StateIds"><c>&lt;state&gt;</c> followed by the state's tokens.</param>
/// <param name="Rows">One branch per question, in order.</param>
public sealed record KevEncoding(ImmutableArray<int> StateIds, IReadOnlyList<KevRow> Rows);

/// <summary>A state or question row needs more tokens than the profile allows. Nothing is truncated.</summary>
public sealed class KevInputTooLongException : ArgumentException
{
    /// <summary>Creates the exception.</summary>
    public KevInputTooLongException()
    {
    }

    /// <summary>Creates the exception with a message.</summary>
    public KevInputTooLongException(string message)
        : base(message)
    {
    }

    /// <summary>Creates the exception with a message and an inner exception.</summary>
    public KevInputTooLongException(string message, Exception innerException)
        : base(message, innerException)
    {
    }

    /// <summary>Tokens needed: the state (with <c>&lt;state&gt;</c>), or the state plus the offending question's branch.</summary>
    public int TokenCount { get; init; }

    /// <summary>The limit in effect.</summary>
    public int MaxTokens { get; init; }

    /// <summary>Index of the offending question, or null when the state itself is too long.</summary>
    public int? QuestionIndex { get; init; }
}

/// <summary>
/// Kev's encoder (<c>kev/model.py</c> <c>encode</c> and <c>rows_of</c>, strict, row form) over the owned tokenizer.
/// Each text segment is escaped (<see cref="KevText.EscapeSpecialTokens"/>) and tokenized on its own without special
/// tokens; delimiters are inserted by id. The decision runtime (<c>KevDecisionModel</c> in Starlings.Models.Kev.Onnx)
/// runs the rows it returns; another runtime can run the same rows. Thread-safe.
/// </summary>
public sealed class KevEncoder
{
    private static readonly (string Token, int Id)[] Delimiters =
    [
        ("<|fim_prefix|>", KevModelProfile.StateTokenId),
        ("<|fim_middle|>", KevModelProfile.QuestionTokenId),
        ("<|box_start|>", KevModelProfile.OptionTokenId),
        ("<|box_end|>", KevModelProfile.OptionEndTokenId),
        ("<|fim_suffix|>", KevModelProfile.DecideTokenId),
        ("<|endoftext|>", KevModelProfile.PadTokenId),
    ];

    private readonly Tokenizer _tokenizer;

    private KevEncoder(Tokenizer tokenizer) => _tokenizer = tokenizer;

    /// <summary>Loads the owned tokenizer and checks Kev's delimiter ids.</summary>
    /// <param name="tokenizerPath">The <c>qwen2-bpe-kev</c> profile's <c>tokenizer.json</c> (both checkpoints ship it).</param>
    /// <exception cref="InvalidDataException">A delimiter token is not at Kev's id.</exception>
    public static KevEncoder FromFile(string tokenizerPath)
    {
        var tokenizer = Tokenizer.FromFile(tokenizerPath);
        foreach (var (token, id) in Delimiters)
        {
            if (tokenizer.TokenToId(token) != id)
            {
                throw new InvalidDataException($"tokenizer.json is not the {KevModelProfile.TokenizerProfile} profile asset: {token} is not id {id}.");
            }
        }

        return new KevEncoder(tokenizer);
    }

    /// <summary>The owned tokenizer.</summary>
    public Tokenizer Tokenizer => _tokenizer;

    /// <summary><c>user_tokens</c>: escape, then encode without special tokens.</summary>
    public int[] UserTokens(string text) => [.. _tokenizer.Encode(KevText.EscapeSpecialTokens(text), addSpecialTokens: false).Ids];

    /// <summary>
    /// <c>encode</c> and <c>rows_of</c>: the state ids (the <c>&lt;state&gt;</c> token first) and one row per question
    /// (positions continuing after the state, the <c>&lt;decide&gt;</c> index and each option's <c>&lt;/opt&gt;</c> index).
    /// </summary>
    /// <param name="record">The rendered request (<see cref="KevRequest.ToRecord"/>).</param>
    /// <param name="maxStateTokens">Longest accepted state, the <c>&lt;state&gt;</c> token included.</param>
    /// <param name="maxRowTokens">Longest accepted state plus one question row.</param>
    /// <exception cref="ArgumentException">No question, a question without 1 to 255 options, or a null option.</exception>
    /// <exception cref="KevInputTooLongException">The state or a row exceeds the limits.</exception>
    public KevEncoding Encode(KevRecord record, int maxStateTokens = KevModelProfile.MaxStateTokens, int maxRowTokens = KevModelProfile.MaxRowTokens)
    {
        ArgumentNullException.ThrowIfNull(record);
        ArgumentNullException.ThrowIfNull(record.State);
        ArgumentNullException.ThrowIfNull(record.Questions);
        if (record.Questions.Count == 0)
        {
            throw new ArgumentException("A record needs at least one question.", nameof(record));
        }

        int[] stateTokens = UserTokens(record.State);
        int stateLength = stateTokens.Length + 1;
        if (stateLength > maxStateTokens)
        {
            throw new KevInputTooLongException(
                $"state is {stateLength:N0} tokens, over the {maxStateTokens:N0}-token limit (the <state> token included): shorten the document or split it across requests")
            {
                TokenCount = stateLength, MaxTokens = maxStateTokens,
            };
        }

        var state = ImmutableArray.CreateBuilder<int>(stateLength);
        state.Add(KevModelProfile.StateTokenId);
        state.AddRange(stateTokens);
        var rows = new KevRow[record.Questions.Count];
        for (int k = 0; k < rows.Length; k++)
        {
            var q = record.Questions[k] ?? throw new ArgumentException($"Question {k} is null.", nameof(record));
            if (q.Options is null || q.Options.Count is < 1 or > KevModelProfile.MaxOptions)
            {
                throw new ArgumentException($"Question {k} needs 1 to {KevModelProfile.MaxOptions} options.", nameof(record));
            }

            var ids = new List<int> { KevModelProfile.QuestionTokenId };
            ids.AddRange(UserTokens(q.Instruction ?? string.Empty));
            var ends = ImmutableArray.CreateBuilder<int>(q.Options.Count);
            foreach (string option in q.Options)
            {
                ids.Add(KevModelProfile.OptionTokenId);
                ids.AddRange(UserTokens(option ?? throw new ArgumentException($"Question {k} has a null option.", nameof(record))));
                ids.Add(KevModelProfile.OptionEndTokenId);
                ends.Add(ids.Count - 1);
            }

            ids.Add(KevModelProfile.DecideTokenId);
            if (ids.Count > maxRowTokens - stateLength)
            {
                throw new KevInputTooLongException(
                    $"branch too long: {ids.Count} tokens with a {stateLength}-token state (row limit {maxRowTokens})")
                {
                    TokenCount = stateLength + ids.Count, MaxTokens = maxRowTokens, QuestionIndex = k,
                };
            }

            rows[k] = new KevRow(
                [.. ids], [.. Enumerable.Range(stateLength, ids.Count)], ids.Count - 1, ends.MoveToImmutable());
        }

        return new KevEncoding(state.MoveToImmutable(), rows);
    }
}
