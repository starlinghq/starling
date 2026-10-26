namespace Starlings.Models.Kev;

// State reuse across calls: the state pass of RunCached, kept in a KevState, and the rows of later calls run from it.
public sealed partial class KevDecisionModel
{
    /// <summary>
    /// Runs a state once and keeps its cache for later question sets
    /// (<see cref="Decide(KevState, IEnumerable{KeyValuePair{string, KevQuestion}}, CancellationToken)"/>). The state is
    /// rendered, escaped and tokenized as <see cref="Decide(KevRequest, CancellationToken)"/> does. The caller owns the
    /// result and disposes it before the model.
    /// </summary>
    /// <exception cref="KevInputTooLongException">The state is over <see cref="KevOptions.MaxStateTokens"/>.</exception>
    public KevState CreateState(KevContent state, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(state);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        var stateIds = _encoder.EncodeState(state.Render(), _options.MaxStateTokens);
        _runGate?.Wait(cancellationToken);
        try
        {
            return new KevState(this, stateIds, RunState(stateIds, cancellationToken));
        }
        finally
        {
            _runGate?.Release();
        }
    }

    /// <summary>
    /// Answers questions about a state created by <see cref="CreateState"/>: the state pass is skipped, every question
    /// runs as its own row from the state's cache. Probabilities equal
    /// <see cref="Decide(KevRequest, CancellationToken)"/> on the same state and questions. Limits and
    /// <see cref="KevOptions.DeclineOverLongQuestions"/> apply as there.
    /// </summary>
    /// <param name="state">A state of this model, not disposed.</param>
    /// <param name="questions">Question id to question, in order; at least one, ids unique.</param>
    /// <param name="cancellationToken">Checked before and inside each graph run.</param>
    /// <exception cref="ArgumentException">The state belongs to another model, or the questions are invalid.</exception>
    /// <exception cref="ObjectDisposedException">The state or the model is disposed.</exception>
    /// <exception cref="KevInputTooLongException">A question is over a limit and <see cref="KevOptions.DeclineOverLongQuestions"/> is false.</exception>
    public KevDecision Decide(KevState state, IEnumerable<KeyValuePair<string, KevQuestion>> questions, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(state);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!ReferenceEquals(state.Model, this))
        {
            throw new ArgumentException("The state was created by another model.", nameof(state));
        }

        // Checked here as well as in Acquire, so a disposed state is refused even when every question is declined.
        ObjectDisposedException.ThrowIf(state.IsDisposed, state);

        var checkedQuestions = KevRequest.CheckQuestions(questions);
        var encoding = _encoder.EncodeRows(
            state.StateIds, KevRequest.RecordQuestions(checkedQuestions), _options.MaxRowTokens, _options.MaxBranchTokens, _options.DeclineOverLongQuestions);
        float[][] probs = [];
        if (encoding.Rows.Count > 0)
        {
            CheckEncoding(encoding);
            state.Acquire();
            try
            {
                _runGate?.Wait(cancellationToken);
                try
                {
                    probs = RunRows(encoding, state.Past, cancellationToken);
                }
                finally
                {
                    _runGate?.Release();
                }
            }
            finally
            {
                state.Release();
            }
        }

        return ToDecision(checkedQuestions, encoding, probs);
    }
}
