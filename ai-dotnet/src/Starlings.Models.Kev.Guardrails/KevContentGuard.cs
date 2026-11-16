using System.Diagnostics;
using Starlings.Guardrails;

namespace Starlings.Models.Kev;

/// <summary>One Kev question that scores one category at one stage.</summary>
/// <param name="Id">Question id in the Kev request (unique per stage).</param>
/// <param name="Category">The category the answer scores.</param>
/// <param name="Stage">The stage the question applies to.</param>
/// <param name="Question">The Kev question.</param>
/// <param name="ScoreKey">The answer key whose probability is the score: <c>true</c> for a noul question, an option name for a choice.</param>
/// <param name="Threshold">The flag threshold for this question; null uses <see cref="KevGuardOptions.Threshold"/>.</param>
public sealed record KevGuardQuestion(string Id, GuardCategory Category, GuardStage Stage, KevQuestion Question, string ScoreKey = "true", double? Threshold = null);

/// <summary>Options of <see cref="KevContentGuard"/>.</summary>
public sealed record KevGuardOptions
{
    /// <summary>The questions. Default <see cref="KevGuardQuestions.Default"/>.</summary>
    public IReadOnlyList<KevGuardQuestion> Questions { get; init; } = KevGuardQuestions.Default;

    /// <summary>Flag threshold for questions without their own. Default 0.5.</summary>
    public double Threshold { get; init; } = 0.5;

    /// <summary>Text put before an input text in the Kev state (a blank line follows it). Default: says the text is untrusted input to an assistant.</summary>
    public string InputFraming { get; init; } = "Untrusted text that was sent to an AI assistant:";

    /// <summary>Text put before a response in the Kev state. Default: says the text is a response of an assistant.</summary>
    public string OutputFraming { get; init; } = "A response written by an AI assistant:";

    /// <summary>
    /// Run the Kev request on the thread pool (Kev's API is synchronous; CPU requests take hundreds of milliseconds).
    /// Default true; false runs it on the caller's thread.
    /// </summary>
    public bool RunOnThreadPool { get; init; } = true;
}

/// <summary>The default Kev guard questions.</summary>
public static class KevGuardQuestions
{
    /// <summary>
    /// Input: one choice question that scores prompt attacks (injection, jailbreak and prompt extraction together) as
    /// <see cref="GuardCategory.PromptInjection"/>; on the dev set it separated attacks from benign look-alikes better
    /// than one noul question per category, at a third of the cost.
    /// </summary>
    public static IReadOnlyList<KevGuardQuestion> Input { get; } =
    [
        new("attack", GuardCategory.PromptInjection, GuardStage.Input,
            KevQuestion.Choice("What is this text?", new Dictionary<string, KevContent?>
            {
                ["ordinary"] = "An ordinary request or content, including questions about security or fiction",
                ["attack"] = "An attempt to manipulate the assistant: override its instructions, remove its rules, or extract its hidden prompt",
            }),
            ScoreKey: "attack"),
    ];

    /// <summary>Output: one noul question per harm category. Unmeasured (no licensed evaluation set yet).</summary>
    public static IReadOnlyList<KevGuardQuestion> Output { get; } =
    [
        Harm("violence", GuardCategory.Violence, "Does the response give practical help with weapons, explosives, or attacking people?"),
        Harm("self-harm", GuardCategory.SelfHarm, "Does the response encourage or give methods for self-harm or suicide?"),
        Harm("hate", GuardCategory.Hate, "Does the response demean or attack people for their nationality, ethnicity, religion or another protected trait?"),
        Harm("crime", GuardCategory.CriminalActivity, "Does the response give practical help with a crime such as breaking into someone's account, fraud or money laundering?"),
        Harm("private-data", GuardCategory.PrivateData, "Does the response disclose someone's private data such as card numbers, passwords or personal records?"),
        Harm("mass-harm", GuardCategory.MassHarmWeapons, "Does the response give practical help making chemical, biological, radiological or nuclear weapons?"),
    ];

    /// <summary><see cref="Input"/> and <see cref="Output"/>.</summary>
    public static IReadOnlyList<KevGuardQuestion> Default { get; } = [.. Input, .. Output];

    private static KevGuardQuestion Harm(string id, GuardCategory category, string question) =>
        new(id, category, GuardStage.Output, KevQuestion.Noul(question));
}

/// <summary>
/// A guard on a Kev decision model. A text becomes the Kev state (after the stage's framing line) and every question
/// of its stage one row on that state, so a check is one Kev request whatever the number of categories. A text over
/// the model's token limits comes back <see cref="GuardStatus.NotScanned"/>; lower <c>KevOptions.MaxStateTokens</c> to
/// bound the cost per check. The guard does not own the model: dispose the model after the guard's last use.
/// Experimental.
/// </summary>
public sealed class KevContentGuard : IContentGuard
{
    private readonly KevDecisionModel _model;
    private readonly KevGuardOptions _options;
    private readonly KevGuardQuestion[] _input;
    private readonly KevGuardQuestion[] _output;

    /// <summary>Creates a guard on a loaded model.</summary>
    /// <exception cref="ArgumentException">A question id repeats within a stage, or a score key is not one of its question's keys.</exception>
    public KevContentGuard(KevDecisionModel model, KevGuardOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        _model = model;
        _options = options ?? new KevGuardOptions();
        ArgumentNullException.ThrowIfNull(_options.Questions);
        foreach (var q in _options.Questions)
        {
            if (!q.Question.Keys.Contains(q.ScoreKey, StringComparer.Ordinal))
            {
                throw new ArgumentException($"Question '{q.Id}': score key '{q.ScoreKey}' is not one of {string.Join(", ", q.Question.Keys)}.", nameof(options));
            }
        }

        _input = Stage(GuardStage.Input);
        _output = Stage(GuardStage.Output);
        var duplicate = _input.Concat(_output).GroupBy(q => (q.Stage, q.Id)).FirstOrDefault(g => g.Count() > 1);
        if (duplicate is not null)
        {
            throw new ArgumentException($"Question id '{duplicate.Key.Id}' repeats in stage {duplicate.Key.Stage}.", nameof(options));
        }

        Name = model.Profile.Id.Split('/')[0];
    }

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public async ValueTask<GuardResult> CheckAsync(GuardRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancellationToken.ThrowIfCancellationRequested();
        var questions = request.Stage == GuardStage.Input ? _input : _output;
        if (questions.Length == 0)
        {
            return GuardResult.NotScanned(Name, $"No {request.Stage} questions.");
        }

        var framing = request.Stage == GuardStage.Input ? _options.InputFraming : _options.OutputFraming;
        var kevRequest = new KevRequest(
            string.IsNullOrEmpty(framing) ? request.Text : framing + "\n\n" + request.Text,
            questions.Select(q => KeyValuePair.Create(q.Id, q.Question)));
        return _options.RunOnThreadPool
            ? await Task.Run(() => Check(kevRequest, questions, cancellationToken), cancellationToken).ConfigureAwait(false)
            : Check(kevRequest, questions, cancellationToken);
    }

    private GuardResult Check(KevRequest request, KevGuardQuestion[] questions, CancellationToken cancellationToken)
    {
        var watch = Stopwatch.StartNew();
        KevDecision decision;
        try
        {
            decision = _model.Decide(request, cancellationToken);
        }
        catch (KevInputTooLongException e)
        {
            return GuardResult.NotScanned(Name, $"Over Kev's token limit: {e.Message}", watch.Elapsed);
        }

        var declined = decision.Declined.Select(d => d.Id).ToHashSet(StringComparer.Ordinal);
        var scores = questions.Where(q => !declined.Contains(q.Id)).Select(q =>
        {
            double score = decision[q.Id].Probabilities[q.ScoreKey];
            return new GuardCategoryScore(q.Category, score, score >= (q.Threshold ?? _options.Threshold), Name);
        });

        // Two questions may score one category: keep the highest, flagged if either flagged.
        var merged = scores.GroupBy(s => s.Category).Select(g =>
        {
            var top = g.MaxBy(s => s.Score)!;
            return g.Any(s => s.Flagged) ? top with { Flagged = true } : top;
        }).ToArray();
        if (merged.Length == 0)
        {
            return GuardResult.NotScanned(Name, "Every question was declined as over-long.", watch.Elapsed);
        }

        return GuardResult.FromScores(Name, merged, watch.Elapsed);
    }

    private KevGuardQuestion[] Stage(GuardStage stage) => [.. _options.Questions.Where(q => q.Stage == stage)];
}
