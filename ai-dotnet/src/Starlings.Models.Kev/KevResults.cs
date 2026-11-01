using System.Runtime.CompilerServices;

namespace Starlings.Models.Kev;

/// <summary>The answers to a request, in question order.</summary>
/// <param name="Answers">One answer per question, in request order (questions in <see cref="Declined"/> have none).</param>
/// <param name="StateTokens">Tokens of the state, <c>&lt;state&gt;</c> included.</param>
/// <param name="BranchTokens">Tokens of all answered question branches together.</param>
public sealed record KevDecision(IReadOnlyList<KevAnswer> Answers, int StateTokens, int BranchTokens)
{
    /// <summary>
    /// Questions left unanswered because they are over a token limit (<c>KevOptions.DeclineOverLongQuestions</c>),
    /// in question order; empty otherwise. <see cref="Answers"/> then holds the other questions, still in request order.
    /// </summary>
    public IReadOnlyList<KevOverLongQuestion> Declined { get; init; } = [];

    /// <summary>The answer to a question by id.</summary>
    /// <exception cref="ArgumentNullException"><paramref name="id"/> is null.</exception>
    /// <exception cref="KeyNotFoundException">No question has this id, or it was declined (<see cref="Declined"/>).</exception>
    /// <remarks>
    /// A hash lookup: an ordinal index of <see cref="Answers"/> is built on first use and kept with that list, so a copy
    /// with other answers (<c>with { Answers = ... }</c>) gets its own. On a repeated id the first answer wins.
    /// Pass an answers list that is not changed afterwards (the model's are arrays): a list mutated after the first
    /// lookup keeps its old index.
    /// </remarks>
    public KevAnswer this[string id]
    {
        get
        {
            ArgumentNullException.ThrowIfNull(id);
            return AnswerIndex.GetValue(Answers, BuildIndex).TryGetValue(id, out var answer) ? answer : throw new KeyNotFoundException(
                Declined.Any(d => d.Id == id) ? $"Question '{id}' was declined: it is over a token limit." : $"No question '{id}'.");
        }
    }

    // Keyed by the Answers list itself rather than held in a field, so record equality, copies and hashing are unchanged.
    private static readonly ConditionalWeakTable<IReadOnlyList<KevAnswer>, Dictionary<string, KevAnswer>> AnswerIndex = new();

    private static Dictionary<string, KevAnswer> BuildIndex(IReadOnlyList<KevAnswer> answers)
    {
        var index = new Dictionary<string, KevAnswer>(answers.Count, StringComparer.Ordinal);
        foreach (var answer in answers)
        {
            index.TryAdd(answer.Id, answer);
        }

        return index;
    }
}

/// <summary>
/// One question's calibrated distribution and its System One reading (<c>kev/api.py</c> <c>to_answers</c>), unrounded:
/// noul p(yes); choice argmax and confidence <c>(p_max - 1/K) / (1 - 1/K)</c>; score expected level and confidence
/// <c>max(0, 1 - E|level - mode| / D)</c>.
/// </summary>
public sealed class KevAnswer
{
    internal KevAnswer(string id, KevQuestion question, float[] probabilities, double temperature = 1.0, bool isCalibrated = false)
    {
        Id = id;
        Temperature = temperature;
        IsCalibrated = isCalibrated;
        Type = question.Type;
        Keys = question.Keys;
        Legend = question.Legend;
        ProbabilityList = probabilities;
        Probabilities = Keys.Select((k, i) => new KeyValuePair<string, float>(k, probabilities[i])).ToDictionary(StringComparer.Ordinal);
        double[] p = [.. probabilities.Select(x => (double)x)];
        int mode = 0;
        for (int i = 1; i < p.Length; i++)
        {
            if (p[i] > p[mode])
            {
                mode = i;
            }
        }

        Choice = Keys[mode];
        Confidence = Type == KevQuestionType.Score ? ScoreConfidence(p) : ChoiceConfidence(p);
        ExpectedScore = p.Select((x, i) => i * x).Sum();
    }

    /// <summary>Question id.</summary>
    public string Id { get; }

    /// <summary>Question type.</summary>
    public KevQuestionType Type { get; }

    /// <summary>Keys in option order: option names (choice), <c>false</c>/<c>true</c> (noul), level indices (score).</summary>
    public IReadOnlyList<string> Keys { get; }

    /// <summary>Score questions: rendered level descriptions; otherwise null.</summary>
    public IReadOnlyList<string>? Legend { get; }

    /// <summary>Probabilities in option order (sum to 1 up to rounding).</summary>
    public IReadOnlyList<float> ProbabilityList { get; }

    /// <summary>Probabilities by key.</summary>
    public IReadOnlyDictionary<string, float> Probabilities { get; }

    /// <summary>Noul: p(yes). Otherwise the probability of the last option.</summary>
    public float Yes => ProbabilityList[^1];

    /// <summary>The most likely key (the first one on ties); for noul <c>true</c> or <c>false</c>.</summary>
    public string Choice { get; }

    /// <summary>Choice and noul: <c>(p_max - 1/K) / (1 - 1/K)</c>, 1 for one option. Score: the score confidence.</summary>
    public double Confidence { get; }

    /// <summary>Score: the expected level index <c>sum(i * p_i)</c>.</summary>
    public double ExpectedScore { get; }

    /// <summary>The temperature the probabilities were divided by (<c>KevDecisionModel.Temperature</c>).</summary>
    public double Temperature { get; }

    /// <summary>
    /// True when <see cref="Temperature"/> is the checkpoint's calibrated temperature (<see cref="KevModelProfile.Temperature"/>,
    /// from <c>head.pt</c>), so the probabilities are the calibrated ones; false for any other
    /// <c>KevOptions.Temperature</c>, 1.0 (raw) included.
    /// </summary>
    public bool IsCalibrated { get; }

    // system-one-adapter 0.2.1 _normalize: p normalised to sum 1 (all zeros -> uniform).
    private static double[] Normalize(double[] p)
    {
        double t = p.Sum();
        return t == 0 ? [.. p.Select(_ => 1.0 / p.Length)] : [.. p.Select(x => x / t)];
    }

    private static double ChoiceConfidence(double[] p)
    {
        int k = p.Length;
        return k == 1 ? 1.0 : (Normalize(p).Max() - 1.0 / k) / (1 - 1.0 / k);
    }

    private static double ScoreConfidence(double[] p)
    {
        int l = p.Length;
        if (l == 1)
        {
            return 1.0;
        }

        p = Normalize(p);
        int mode = 0;
        for (int i = 1; i < l; i++)
        {
            if (p[i] > p[mode])
            {
                mode = i;
            }
        }

        double d = Enumerable.Range(0, l).Sum(i => Math.Abs(i - (l - 1) / 2.0)) / l;
        return Math.Max(0.0, 1.0 - p.Select((pi, i) => pi * Math.Abs(i - mode)).Sum() / d);
    }
}
