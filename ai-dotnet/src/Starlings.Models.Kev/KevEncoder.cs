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
public sealed record KevEncoding(ImmutableArray<int> StateIds, IReadOnlyList<KevRow> Rows)
{
    /// <summary>
    /// The question index of each row, in row order, or null when row <c>k</c> is question <c>k</c> (every question
    /// encoded). Set when <c>KevOptions.DeclineOverLongQuestions</c> in Starlings.Models.Kev.Onnx left questions out.
    /// </summary>
    public IReadOnlyList<int>? QuestionIndices { get; init; }

    /// <summary>
    /// Questions left out because they are over a limit (<c>KevOptions.DeclineOverLongQuestions</c>), in question
    /// order; empty otherwise.
    /// </summary>
    public IReadOnlyList<KevOverLongQuestion> Declined { get; init; } = [];

    /// <summary>The question index of row <paramref name="row"/>.</summary>
    public int QuestionIndexOf(int row) => QuestionIndices is { } indices ? indices[row] : row;
}

/// <summary>Which token limit a state or question is over.</summary>
public enum KevTokenLimit
{
    /// <summary>The state, <c>&lt;state&gt;</c> included, is over <c>KevOptions.MaxStateTokens</c>.</summary>
    State,

    /// <summary>The question's row (state plus branch) is over <c>KevOptions.MaxRowTokens</c>.</summary>
    Row,

    /// <summary>The question's branch alone is over <c>KevOptions.MaxBranchTokens</c>.</summary>
    Branch,
}

/// <summary>One question that is over a token limit. Nothing is truncated.</summary>
/// <param name="Index">Index of the question in the request or record.</param>
/// <param name="Id">The question's id.</param>
/// <param name="Limit">The limit it is over (<see cref="KevTokenLimit.Row"/> is checked before <see cref="KevTokenLimit.Branch"/>).</param>
/// <param name="TokenCount">Tokens counted against the limit: state plus branch for <see cref="KevTokenLimit.Row"/>, the branch for <see cref="KevTokenLimit.Branch"/>.</param>
/// <param name="MaxTokens">The limit in effect.</param>
public sealed record KevOverLongQuestion(int Index, string Id, KevTokenLimit Limit, int TokenCount, int MaxTokens);

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

    /// <summary>
    /// Tokens counted against <see cref="Limit"/>: the state (with <c>&lt;state&gt;</c>), the state plus the first
    /// offending question's branch (<see cref="KevTokenLimit.Row"/>), or that branch alone (<see cref="KevTokenLimit.Branch"/>).
    /// </summary>
    public int TokenCount { get; init; }

    /// <summary>The limit in effect.</summary>
    public int MaxTokens { get; init; }

    /// <summary>Index of the first offending question, or null when the state itself is too long.</summary>
    public int? QuestionIndex { get; init; }

    /// <summary>The limit that <see cref="TokenCount"/> is over.</summary>
    public KevTokenLimit Limit { get; init; }

    /// <summary>
    /// Every question over a limit, found in one pass, in question order (the first is <see cref="QuestionIndex"/>);
    /// empty when the state itself is too long.
    /// </summary>
    public IReadOnlyList<KevOverLongQuestion> OverLongQuestions { get; init; } = [];
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
    /// <param name="maxBranchTokens">Longest accepted question branch alone; null for no branch limit.</param>
    /// <param name="decline">Leave over-long questions out (<see cref="KevEncoding.Declined"/>) instead of throwing.</param>
    /// <exception cref="ArgumentException">No question, a question without 1 to 255 options, or a null option.</exception>
    /// <exception cref="KevInputTooLongException">
    /// The state is over the limit, or a question is and <paramref name="decline"/> is false.
    /// </exception>
    public KevEncoding Encode(
        KevRecord record,
        int maxStateTokens = KevModelProfile.MaxStateTokens,
        int maxRowTokens = KevModelProfile.MaxRowTokens,
        int? maxBranchTokens = null,
        bool decline = false)
    {
        ArgumentNullException.ThrowIfNull(record);
        ArgumentNullException.ThrowIfNull(record.State);
        ArgumentNullException.ThrowIfNull(record.Questions);
        if (record.Questions.Count == 0)
        {
            throw new ArgumentException("A record needs at least one question.", nameof(record));
        }

        var state = EncodeState(record.State, maxStateTokens);
        return EncodeRows(state, record.Questions, maxRowTokens, maxBranchTokens, decline);
    }

    /// <summary><c>&lt;state&gt;</c> followed by the state's tokens.</summary>
    /// <exception cref="KevInputTooLongException">The state is over <paramref name="maxStateTokens"/>.</exception>
    public ImmutableArray<int> EncodeState(string text, int maxStateTokens = KevModelProfile.MaxStateTokens)
    {
        ArgumentNullException.ThrowIfNull(text);
        int[] stateTokens = UserTokens(text);
        int stateLength = stateTokens.Length + 1;
        if (stateLength > maxStateTokens)
        {
            throw new KevInputTooLongException(
                $"state is {stateLength:N0} tokens, over the {maxStateTokens:N0}-token limit (the <state> token included): shorten the document or split it across requests")
            {
                TokenCount = stateLength, MaxTokens = maxStateTokens, Limit = KevTokenLimit.State,
            };
        }

        var state = ImmutableArray.CreateBuilder<int>(stateLength);
        state.Add(KevModelProfile.StateTokenId);
        state.AddRange(stateTokens);
        return state.MoveToImmutable();
    }

    /// <summary>
    /// One branch per question after an encoded state. Every question is checked in one pass; the over-long ones are
    /// thrown together (<see cref="KevInputTooLongException.OverLongQuestions"/>) or, with <paramref name="decline"/>,
    /// left out and listed in <see cref="KevEncoding.Declined"/>.
    /// </summary>
    /// <exception cref="KevInputTooLongException">A question is over a limit and <paramref name="decline"/> is false.</exception>
    public KevEncoding EncodeRows(
        ImmutableArray<int> state, IReadOnlyList<KevRecordQuestion> questions, int maxRowTokens, int? maxBranchTokens, bool decline)
    {
        ArgumentNullException.ThrowIfNull(questions);
        int stateLength = state.Length;
        var rows = new List<KevRow>(questions.Count);
        var indices = new List<int>(questions.Count);
        var overLong = new List<KevOverLongQuestion>();
        string? firstMessage = null;
        for (int k = 0; k < questions.Count; k++)
        {
            var q = questions[k] ?? throw new ArgumentException($"Question {k} is null.", nameof(questions));
            if (q.Options is null || q.Options.Count is < 1 or > KevModelProfile.MaxOptions)
            {
                throw new ArgumentException($"Question {k} needs 1 to {KevModelProfile.MaxOptions} options.", nameof(questions));
            }

            var ids = new List<int> { KevModelProfile.QuestionTokenId };
            ids.AddRange(UserTokens(q.Instruction ?? string.Empty));
            var ends = ImmutableArray.CreateBuilder<int>(q.Options.Count);
            foreach (string option in q.Options)
            {
                ids.Add(KevModelProfile.OptionTokenId);
                ids.AddRange(UserTokens(option ?? throw new ArgumentException($"Question {k} has a null option.", nameof(questions))));
                ids.Add(KevModelProfile.OptionEndTokenId);
                ends.Add(ids.Count - 1);
            }

            ids.Add(KevModelProfile.DecideTokenId);
            if (ids.Count > maxRowTokens - stateLength)
            {
                overLong.Add(new KevOverLongQuestion(k, q.Id, KevTokenLimit.Row, stateLength + ids.Count, maxRowTokens));
                firstMessage ??= $"branch too long: {ids.Count} tokens with a {stateLength}-token state (row limit {maxRowTokens})";
                continue;
            }

            if (maxBranchTokens is { } maxBranch && ids.Count > maxBranch)
            {
                overLong.Add(new KevOverLongQuestion(k, q.Id, KevTokenLimit.Branch, ids.Count, maxBranch));
                firstMessage ??= $"branch too long: {ids.Count} tokens (branch limit {maxBranch})";
                continue;
            }

            rows.Add(new KevRow([.. ids], [.. Enumerable.Range(stateLength, ids.Count)], ids.Count - 1, ends.MoveToImmutable()));
            indices.Add(k);
        }

        if (overLong.Count > 0 && !decline)
        {
            var first = overLong[0];
            string others = overLong.Count == 1
                ? string.Empty
                : $"; {overLong.Count} questions are over the limits: {string.Join(", ", overLong.Select(o => o.Index))}";
            throw new KevInputTooLongException($"question {first.Index}: {firstMessage}{others}")
            {
                TokenCount = first.TokenCount,
                MaxTokens = first.MaxTokens,
                QuestionIndex = first.Index,
                Limit = first.Limit,
                OverLongQuestions = overLong.AsReadOnly(),
            };
        }

        return overLong.Count == 0
            ? new KevEncoding(state, rows)
            : new KevEncoding(state, rows) { QuestionIndices = indices.AsReadOnly(), Declined = overLong.AsReadOnly() };
    }
}
