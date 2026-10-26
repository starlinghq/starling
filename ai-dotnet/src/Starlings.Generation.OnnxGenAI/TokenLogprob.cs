using System.Collections.Immutable;
using Starlings.Tokenizers;

namespace Starlings.Generation.OnnxGenAI;

/// <summary>
/// The log probability of one generated token (<see cref="GenerationOptions.ReturnLogprobs"/>), with the most likely
/// tokens of the same step. The Microsoft.Extensions.AI adapters put lists of it under their <c>logprobs</c> additional
/// property.
/// </summary>
/// <param name="TokenId">The generated token id.</param>
/// <param name="Text">
/// The token decoded on its own with the owned tokenizer (special tokens kept). A byte-level token that carries part
/// of a multi-byte UTF-8 character decodes to U+FFFD here; the streamed text never splits a character.
/// </param>
/// <param name="Logprob">Natural log of the token's probability: the log-softmax of the step's raw logits at the token.</param>
/// <param name="TopLogprobs">
/// The <see cref="GenerationOptions.TopLogprobs"/> most likely tokens of the step, most likely first (ties by lower id);
/// empty when <see cref="GenerationOptions.TopLogprobs"/> is 0. The generated token appears in it only when it is
/// among them.
/// </param>
/// <remarks>Equality compares <see cref="TopLogprobs"/> element by element.</remarks>
public sealed record TokenLogprob(int TokenId, string Text, double Logprob, ImmutableArray<TopTokenLogprob> TopLogprobs)
{
    /// <inheritdoc />
    public bool Equals(TokenLogprob? other) =>
        other is not null &&
        TokenId == other.TokenId &&
        string.Equals(Text, other.Text, StringComparison.Ordinal) &&
        Logprob.Equals(other.Logprob) &&
        TopLogprobs.AsSpan().SequenceEqual(other.TopLogprobs.AsSpan());

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(TokenId, Text, Logprob, TopLogprobs.IsDefault ? 0 : TopLogprobs.Length);
}

/// <summary>One of the most likely tokens of a step (<see cref="TokenLogprob.TopLogprobs"/>).</summary>
/// <param name="TokenId">The token id.</param>
/// <param name="Text">The token decoded on its own (see <see cref="TokenLogprob.Text"/>).</param>
/// <param name="Logprob">Natural log of its probability at that step.</param>
public sealed record TopTokenLogprob(int TokenId, string Text, double Logprob);

/// <summary>Computes <see cref="TokenLogprob"/> entries from a logits row. One instance per request; not thread-safe.</summary>
internal sealed class LogprobCalculator
{
    private readonly Tokenizer _tokenizer;
    private readonly int _top;
    private readonly float[] _logits;
    private readonly int[] _topIds;

    public LogprobCalculator(Tokenizer tokenizer, int vocabularySize, int topLogprobs)
    {
        _tokenizer = tokenizer;
        _top = Math.Min(topLogprobs, vocabularySize);
        _logits = new float[vocabularySize];
        _topIds = new int[_top];
    }

    /// <summary>The buffer the caller fills with the step's logits before <see cref="Compute"/>.</summary>
    public Span<float> Logits => _logits;

    /// <summary>Log-softmax of <see cref="Logits"/> at <paramref name="tokenId"/>, with the top entries.</summary>
    public TokenLogprob Compute(int tokenId)
    {
        ReadOnlySpan<float> logits = _logits;

        // log p(i) = x_i - max - log(sum_j exp(x_j - max)), accumulated in double.
        float max = float.NegativeInfinity;
        foreach (float x in logits)
        {
            if (x > max)
            {
                max = x;
            }
        }

        double sum = 0;
        foreach (float x in logits)
        {
            sum += Math.Exp((double)x - max);
        }

        double logNormalizer = max + Math.Log(sum);
        var top = ImmutableArray<TopTokenLogprob>.Empty;
        if (_top > 0)
        {
            int count = SelectTop(logits, _topIds);
            var builder = ImmutableArray.CreateBuilder<TopTokenLogprob>(count);
            for (int i = 0; i < count; i++)
            {
                int id = _topIds[i];
                builder.Add(new TopTokenLogprob(id, Decode(id), logits[id] - logNormalizer));
            }

            top = builder.MoveToImmutable();
        }

        return new TokenLogprob(tokenId, Decode(tokenId), logits[tokenId] - logNormalizer, top);
    }

    // The ids of the largest logits, largest first and ties by lower id; one pass with an insertion-sorted window.
    internal static int SelectTop(ReadOnlySpan<float> logits, Span<int> ids)
    {
        int count = 0;
        if (ids.IsEmpty)
        {
            return count;
        }

        for (int id = 0; id < logits.Length; id++)
        {
            float x = logits[id];
            if (float.IsNaN(x))
            {
                continue;
            }

            if (count == ids.Length && !(x > logits[ids[count - 1]]))
            {
                continue;
            }

            int position = count < ids.Length ? count++ : count - 1;
            while (position > 0 && x > logits[ids[position - 1]])
            {
                ids[position] = ids[position - 1];
                position--;
            }

            ids[position] = id;
        }

        return count;
    }

    private string Decode(int id) => _tokenizer.Decode([id], false);
}
