using System.Collections.Immutable;
using Starlings.Tokenizers;

namespace Starlings.Generation.OnnxGenAI;

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
