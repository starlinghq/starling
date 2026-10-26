namespace Starlings.Generation;

/// <summary>
/// One request's decoding state on a generation backend: the seam a decode loop drives (append the prompt, step, read
/// the logits that selected a token). Not thread-safe; one instance per request, disposed when the request ends.
/// </summary>
/// <remarks>
/// <para>
/// Implemented by <c>GenAISequence</c> in Starlings.Generation.OnnxGenAI, which completes every call synchronously
/// (ONNX Runtime GenAI's work is blocking native code), and open to other backends, for example a GPU worker whose
/// steps complete asynchronously.
/// </para>
/// <para>
/// Cancellation is checked before a call starts. Whether an implementation can stop work already running is its own
/// contract; <c>GenAISequence</c> cannot.
/// </para>
/// </remarks>
public interface IGenerationSequence : IDisposable
{
    /// <summary>Width of the logits.</summary>
    int VocabularySize { get; }

    /// <summary>True once the backend considers the sequence finished (an end-of-sequence id or its maximum length).</summary>
    bool IsDone { get; }

    /// <summary>
    /// Appends token ids and runs the model over them (the prefill for a prompt, or a chunk of it). Ids must be smaller
    /// than <see cref="VocabularySize"/>.
    /// </summary>
    /// <param name="tokenIds">At least one token id.</param>
    /// <param name="cancellationToken">Checked before the work starts.</param>
    ValueTask AppendTokensAsync(ReadOnlyMemory<int> tokenIds, CancellationToken cancellationToken = default);

    /// <summary>
    /// Runs one decode step and returns every token id it appended, in order, the selected token last. Without
    /// constrained fast-forwarding that is one id.
    /// </summary>
    /// <param name="cancellationToken">Checked before the work starts.</param>
    ValueTask<int[]> StepAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Copies the most recent logits into <paramref name="destination"/> as float32: after
    /// <see cref="AppendTokensAsync"/>, those of the last appended position (the step-0 logits of a prompt); after
    /// <see cref="StepAsync"/>, those that selected the token it returned last.
    /// </summary>
    /// <param name="destination">Exactly <see cref="VocabularySize"/> values.</param>
    /// <param name="cancellationToken">Checked before the work starts.</param>
    ValueTask CopyLastLogitsAsync(Memory<float> destination, CancellationToken cancellationToken = default);
}
