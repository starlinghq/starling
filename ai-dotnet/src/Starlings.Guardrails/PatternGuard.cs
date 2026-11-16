using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;

namespace Starlings.Guardrails;

/// <summary>One pattern rule: a match adds <see cref="Weight"/> to its category's score (capped at 1).</summary>
/// <param name="Category">The category a match counts toward.</param>
/// <param name="Pattern">The pattern, matched against the normalized text (NFKC, zero-width characters removed).</param>
/// <param name="Weight">Score added per matching rule, 0 to 1.</param>
public sealed record PatternRule(GuardCategory Category, Regex Pattern, double Weight = 0.5);

/// <summary>Options of <see cref="PatternGuard"/>.</summary>
public sealed record PatternGuardOptions
{
    /// <summary>Rules for input text (user messages, tool results, documents). Default: <see cref="PatternGuard.BuiltInRules"/>.</summary>
    public IReadOnlyList<PatternRule> Rules { get; init; } = PatternGuard.BuiltInRules;

    /// <summary>
    /// Canary strings for output text: put a unique string in your system prompt and list it here; a response that
    /// contains one (ordinal, case-insensitive) is flagged <see cref="GuardCategory.PromptExtraction"/> with score 1.
    /// Without canaries the pattern guard does not scan output.
    /// </summary>
    public IReadOnlyList<string> Canaries { get; init; } = [];

    /// <summary>A category is flagged at or above this score. Default 0.5: one built-in rule match flags.</summary>
    public double Threshold { get; init; } = 0.5;
}

/// <summary>
/// A model-free guard: regular expressions for common prompt-injection, jailbreak and prompt-extraction phrasings on input,
/// and canary strings on output. Microseconds per text and no model memory. High precision and low recall: it misses
/// paraphrased and indirect attacks, so use it as the first tier of a <see cref="CascadeGuard"/> or as the light level.
/// English, French, German and Spanish phrasings only. Experimental.
/// </summary>
public sealed partial class PatternGuard : IContentGuard
{
    private const int MatchTimeoutMs = 100;
    private readonly PatternGuardOptions _options;

    /// <summary>Creates a pattern guard.</summary>
    public PatternGuard(PatternGuardOptions? options = null)
    {
        _options = options ?? new PatternGuardOptions();
        ArgumentNullException.ThrowIfNull(_options.Rules);
        ArgumentNullException.ThrowIfNull(_options.Canaries);
        if (_options.Canaries.Any(string.IsNullOrEmpty))
        {
            throw new ArgumentException("A canary is null or empty.", nameof(options));
        }
    }

    /// <inheritdoc />
    public string Name => "patterns";

    /// <summary>The built-in input rules (8 rules over the injection, jailbreak and extraction categories).</summary>
    public static IReadOnlyList<PatternRule> BuiltInRules { get; } =
    [
        new(GuardCategory.PromptInjection, IgnorePrevious()),
        new(GuardCategory.PromptExtraction, HiddenPrompt()),
        new(GuardCategory.Jailbreak, NoRulesMode()),
        new(GuardCategory.Jailbreak, NewPersona()),
        new(GuardCategory.PromptInjection, ChatTemplateTokens()),
        new(GuardCategory.PromptExtraction, RepeatAbove()),
        new(GuardCategory.PromptInjection, IgnorePreviousOtherLanguages()),
        new(GuardCategory.PromptInjection, NoteToAgent()),
    ];

    /// <inheritdoc />
    public ValueTask<GuardResult> CheckAsync(GuardRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancellationToken.ThrowIfCancellationRequested();
        var watch = Stopwatch.StartNew();
        var text = Normalize(request.Text);
        return ValueTask.FromResult(request.Stage == GuardStage.Input ? CheckInput(text, watch) : CheckOutput(text, watch));
    }

    private GuardResult CheckInput(string text, Stopwatch watch)
    {
        if (_options.Rules.Count == 0)
        {
            return GuardResult.NotScanned(Name, "No input rules.", watch.Elapsed);
        }

        var totals = new Dictionary<GuardCategory, double>();
        foreach (var rule in _options.Rules)
        {
            bool match;
            try
            {
                match = rule.Pattern.IsMatch(text);
            }
            catch (RegexMatchTimeoutException)
            {
                return GuardResult.NotScanned(Name, $"A rule for {rule.Category} timed out.", watch.Elapsed);
            }

            if (match)
            {
                totals[rule.Category] = Math.Min(1.0, totals.GetValueOrDefault(rule.Category) + rule.Weight);
            }
        }

        return GuardResult.FromScores(Name, totals.Select(t => new GuardCategoryScore(t.Key, t.Value, t.Value >= _options.Threshold, Name)), watch.Elapsed);
    }

    private GuardResult CheckOutput(string text, Stopwatch watch)
    {
        if (_options.Canaries.Count == 0)
        {
            return GuardResult.NotScanned(Name, "No canaries configured; the pattern guard has no output rules.", watch.Elapsed);
        }

        bool leaked = _options.Canaries.Any(c => text.Contains(Normalize(c), StringComparison.OrdinalIgnoreCase));
        GuardCategoryScore[] scores = leaked ? [new(GuardCategory.PromptExtraction, 1.0, true, Name)] : [];
        return GuardResult.FromScores(Name, scores, watch.Elapsed);
    }

    /// <summary>NFKC (folds full-width and other compatibility forms) and removes zero-width characters.</summary>
    internal static string Normalize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var folded = text.Normalize(NormalizationForm.FormKC);
        var builder = new StringBuilder(folded.Length);
        foreach (var c in folded)
        {
            if (c is not ('​' or '‌' or '‍' or '⁠' or '﻿'))
            {
                builder.Append(c);
            }
        }

        return builder.ToString();
    }

    [GeneratedRegex(@"\b(ignore|disregard|forget)\b.{0,30}\b(previous|prior|above|earlier|all)\b.{0,20}\b(instructions?|rules|guidance|prompt)", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant, MatchTimeoutMs)]
    private static partial Regex IgnorePrevious();

    [GeneratedRegex(@"\b(system prompt|hidden instructions|initial (configuration|instructions)|confidential instructions)\b", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant, MatchTimeoutMs)]
    private static partial Regex HiddenPrompt();

    // DAN is matched case-sensitively, so the name Dan does not flag.
    [GeneratedRegex(@"\b(developer mode|(?-i:DAN)|do anything now|no (content )?policy|without restrictions|never refuses?)\b", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant, MatchTimeoutMs)]
    private static partial Regex NoRulesMode();

    [GeneratedRegex(@"\b(you are (now|no longer)|from now on you)\b", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant, MatchTimeoutMs)]
    private static partial Regex NewPersona();

    [GeneratedRegex(@"(<<SYS>>|\[INST\]|<\|im_start\|>|<system>)", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant, MatchTimeoutMs)]
    private static partial Regex ChatTemplateTokens();

    [GeneratedRegex(@"\b(repeat|print|output|reveal)\b.{0,30}\b(above|everything|words|instructions)\b", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant, MatchTimeoutMs)]
    private static partial Regex RepeatAbove();

    [GeneratedRegex(@"\b(ignorez|vergiss|ignora)\b.{0,30}\b(instructions|anweisungen|instrucciones)\b", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant, MatchTimeoutMs)]
    private static partial Regex IgnorePreviousOtherLanguages();

    [GeneratedRegex(@"\b(AI|assistant|agent)\b.{0,40}\b(new instruction|stop summarizing|forward|transfer)\b", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant, MatchTimeoutMs)]
    private static partial Regex NoteToAgent();
}
