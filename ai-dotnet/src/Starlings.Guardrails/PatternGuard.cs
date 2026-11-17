using System.Buffers;
using System.Diagnostics;
using System.Text.RegularExpressions;

namespace Starlings.Guardrails;

/// <summary>One pattern rule: a match adds <see cref="Weight"/> to its category's score in that view (capped at 1).</summary>
/// <param name="Category">The category a match counts toward.</param>
/// <param name="Pattern">The pattern, matched against every view of the text (see <see cref="PatternGuard"/>).</param>
/// <param name="Weight">Score added per matching rule, 0 to 1. Below the threshold, a rule is a signal that needs another.</param>
/// <param name="PlainViewsOnly">
/// Run the rule only on plain text (the input and decoded payloads), not on the folded and leet views. For rules on
/// scripts other than Latin, which folding and leet mapping garble; saves time.
/// </param>
/// <param name="Keywords">
/// Literals of which every match contains at least one (case-insensitive). The pattern runs only on a view that contains
/// one, found with one vectorized search, so rules cost little on long text. Null runs the pattern on every view. A
/// keyword list that misses a way the pattern can match loses those matches.
/// </param>
public sealed record PatternRule(GuardCategory Category, Regex Pattern, double Weight = 0.5, bool PlainViewsOnly = false, IReadOnlyList<string>? Keywords = null);

/// <summary>
/// A multi-word phrase matched as a substring of the squashed view (letters only, lower case, leetspeak mapped back), which
/// defeats spacing, punctuation, emoji and zero-width characters between letters. Keep phrases to several words: a single
/// squashed word matches inside unrelated text.
/// </summary>
/// <param name="Category">The category a match counts toward.</param>
/// <param name="Phrase">The phrase in plain words (squashed by the guard), for example <c>ignore previous instructions</c>.</param>
/// <param name="Weight">Score added per matching phrase, 0 to 1.</param>
public sealed record PatternPhrase(GuardCategory Category, string Phrase, double Weight = 0.5);

/// <summary>Options of <see cref="PatternGuard"/>.</summary>
public sealed record PatternGuardOptions
{
    /// <summary>Rules for input text (user messages, tool results, documents). Default: <see cref="PatternGuard.BuiltInRules"/>.</summary>
    public IReadOnlyList<PatternRule> Rules { get; init; } = PatternGuard.BuiltInRules;

    /// <summary>Phrases for the squashed view. Default: <see cref="PatternGuard.BuiltInPhrases"/>.</summary>
    public IReadOnlyList<PatternPhrase> Phrases { get; init; } = PatternGuard.BuiltInPhrases;

    /// <summary>
    /// Canary strings for output text: put a unique string in your system prompt and list it here; a response that
    /// contains one, also leet-spelled, spaced out, reversed, in ROT13, base64 or hex, is flagged
    /// <see cref="GuardCategory.PromptExtraction"/> with score 1. Use canaries of six letters or more. Without canaries
    /// the pattern guard does not scan output.
    /// </summary>
    public IReadOnlyList<string> Canaries { get; init; } = [];

    /// <summary>A category is flagged at or above this score. Default 0.5: one full-weight rule match flags.</summary>
    public double Threshold { get; init; } = 0.5;

    /// <summary>
    /// Look for hidden characters and decode obfuscated text (see <see cref="PatternGuard"/>). Default true; false runs the
    /// rules on the plain view only (no folded, leet, squashed or decoded views, no phrases).
    /// </summary>
    public bool DetectObfuscation { get; init; } = true;

    /// <summary>
    /// Characters of a text that the whole-text copies (ROT13, reversed, upside down) cover. Default 4,096. The plain,
    /// folded, leet and squashed views and the base64 and hex blob search always cover the whole text.
    /// </summary>
    public int MaxDecodeChars { get; init; } = 4_096;
}

/// <summary>
/// A model-free guard. On input it runs regular expressions for common prompt-injection, jailbreak and prompt-extraction
/// phrasings, known jailbreak-kit markers, and the same attack in 18 languages, over several views of the text:
/// <list type="bullet">
/// <item>plain: hidden characters removed, NFKC (full-width and mathematical letters folded);</item>
/// <item>folded: diacritics, underlines and zalgo marks removed, Cyrillic and Greek look-alikes mapped to Latin;</item>
/// <item>leet: digits and symbols mapped back to letters;</item>
/// <item>squashed: letters only, matched against <see cref="PatternGuardOptions.Phrases"/> (defeats spacing and emoji between letters);</item>
/// <item>decoded: text hidden in Unicode tag characters or emoji variation selectors, base64 and hex blobs, ROT13, reversed and upside-down text.</item>
/// </list>
/// Hidden-character smuggling, bidi overrides and runs of invisible separators score <see cref="GuardCategory.Obfuscation"/>
/// on their own; so does an attack found only after decoding or folding. On output it checks canary strings.
/// Microseconds to milliseconds per text and no model memory. High precision and low recall: it misses paraphrased and
/// novel attacks, so use it as the first tier of a <see cref="CascadeGuard"/> or as the light level. Experimental.
/// </summary>
public sealed partial class PatternGuard : IContentGuard
{
    private readonly PatternGuardOptions _options;
    private readonly (PatternRule Rule, SearchValues<string>? Keywords)[] _rules;
    private readonly (GuardCategory Category, string Squashed, double Weight)[] _phrases;
    private readonly (string Clean, string Squashed)[] _canaries;

    /// <summary>Creates a pattern guard.</summary>
    public PatternGuard(PatternGuardOptions? options = null)
    {
        _options = options ?? new PatternGuardOptions();
        ArgumentNullException.ThrowIfNull(_options.Rules);
        ArgumentNullException.ThrowIfNull(_options.Phrases);
        ArgumentNullException.ThrowIfNull(_options.Canaries);
        if (_options.Canaries.Any(string.IsNullOrEmpty))
        {
            throw new ArgumentException("A canary is null or empty.", nameof(options));
        }

        _rules = [.. _options.Rules.Select(r => (r, r.Keywords is { Count: > 0 } k ? SearchValues.Create([.. k], StringComparison.OrdinalIgnoreCase) : null))];
        _phrases = [.. _options.Phrases.Select(p => (p.Category, TextViews.SquashAll(p.Phrase), p.Weight)).Where(p => p.Item2.Length > 0)];
        _canaries = [.. _options.Canaries.Select(c => (TextViews.Clean(c), TextViews.SquashAll(c)))];
    }

    /// <inheritdoc />
    public string Name => "patterns";

    /// <inheritdoc />
    public ValueTask<GuardResult> CheckAsync(GuardRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        cancellationToken.ThrowIfCancellationRequested();
        var watch = Stopwatch.StartNew();
        try
        {
            return ValueTask.FromResult(request.Stage == GuardStage.Input ? CheckInput(request.Text, watch) : CheckOutput(request.Text, watch));
        }
        catch (RegexMatchTimeoutException e)
        {
            return ValueTask.FromResult(GuardResult.NotScanned(Name, $"A pattern timed out: {e.Pattern}", watch.Elapsed));
        }
    }

    /// <summary>NFKC with hidden and formatting characters removed (the plain view).</summary>
    internal static string Normalize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return TextViews.Clean(text);
    }

    private GuardResult CheckInput(string text, Stopwatch watch)
    {
        if (_options.Rules.Count == 0 && _phrases.Length == 0)
        {
            return GuardResult.NotScanned(Name, "No input rules.", watch.Elapsed);
        }

        var clean = TextViews.Clean(text);
        var plain = new Dictionary<GuardCategory, double>();
        ScoreRules(clean, plain, plainView: true);
        if (!_options.DetectObfuscation)
        {
            return Result(plain, watch);
        }

        var hidden = TextViews.FindHidden(text);
        var transformed = new Dictionary<GuardCategory, double>();
        ScoreTransformedViews(clean, transformed);
        foreach (var payload in new[] { hidden.TagPayload, hidden.SelectorPayload }.OfType<string>())
        {
            var payloadClean = TextViews.Clean(payload);
            ScoreRules(payloadClean, transformed, plainView: true);
            ScoreTransformedViews(payloadClean, transformed);
        }

        foreach (var decoded in TextViews.Decode(clean, _options.MaxDecodeChars))
        {
            if (decoded.Source is "base64" or "hex")
            {
                // Blobs are payloads: every view.
                ScoreRules(decoded.Text, transformed, plainView: true);
                ScoreTransformedViews(decoded.Text, transformed);
            }
            else
            {
                // Whole-text ROT13, reversed and upside-down copies: the Latin rules and the phrases only.
                ScoreRules(decoded.Text, transformed, plainView: false);
                ScorePhrases(TextViews.Squash(TextViews.Fold(decoded.Text)), transformed);
            }
        }

        // Hidden characters with no ordinary use, then attacks that only show once decoded or folded.
        double obfuscation = 0;
        if (hidden.TagCharacters > 0 || hidden.SmuggledSelectors > 0)
        {
            obfuscation = 1.0;
        }
        else if (hidden.BidiControls > 0 || hidden.InvisibleSeparators >= 3)
        {
            obfuscation = 0.5;
        }

        var merged = new Dictionary<GuardCategory, double>(plain);
        foreach (var (category, score) in transformed)
        {
            if (score >= _options.Threshold && plain.GetValueOrDefault(category) < _options.Threshold)
            {
                obfuscation = Math.Max(obfuscation, 0.75);
            }

            merged[category] = Math.Max(merged.GetValueOrDefault(category), score);
        }

        if (obfuscation > 0)
        {
            merged[GuardCategory.Obfuscation] = obfuscation;
        }

        return Result(merged, watch);
    }

    /// <summary>Folded and leet views get the rules; the squashed view gets the phrases. Each view is scored on its own; the highest wins.</summary>
    private void ScoreTransformedViews(string clean, Dictionary<GuardCategory, double> into)
    {
        // The rules rerun on the folded view only when the text shows a character trick; ordinary accents need only the phrases.
        var folded = TextViews.Fold(clean);
        if (TextViews.HasCharacterTricks(clean))
        {
            ScoreRules(folded, into, plainView: false);
        }

        // Leetspeak 1 stands for i or l: both readings. Rules run on words that mix letters with leet digits or symbols
        // (skipped when there are none); phrases on the squashed text with every leet character mapped (cheap, no regexes).
        foreach (var one in "il")
        {
            if (TextViews.Leet(folded, one) is { } leet)
            {
                ScoreRules(leet, into, plainView: false);
            }

            ScorePhrases(TextViews.Squash(TextViews.LeetAll(folded, one)), into);
        }
    }

    private void ScoreRules(string view, Dictionary<GuardCategory, double> into, bool plainView)
    {
        var totals = new Dictionary<GuardCategory, double>();
        foreach (var (rule, keywords) in _rules)
        {
            if ((plainView || !rule.PlainViewsOnly)
                && (keywords is null || view.AsSpan().IndexOfAny(keywords) >= 0)
                && rule.Pattern.IsMatch(view))
            {
                totals[rule.Category] = Math.Min(1.0, totals.GetValueOrDefault(rule.Category) + rule.Weight);
            }
        }

        Merge(totals, into);
    }

    private void ScorePhrases(string squashed, Dictionary<GuardCategory, double> into)
    {
        var totals = new Dictionary<GuardCategory, double>();
        foreach (var (category, phrase, weight) in _phrases)
        {
            if (squashed.Contains(phrase, StringComparison.Ordinal))
            {
                totals[category] = Math.Min(1.0, totals.GetValueOrDefault(category) + weight);
            }
        }

        Merge(totals, into);
    }

    private static void Merge(Dictionary<GuardCategory, double> view, Dictionary<GuardCategory, double> into)
    {
        foreach (var (category, score) in view)
        {
            into[category] = Math.Max(into.GetValueOrDefault(category), score);
        }
    }

    private GuardResult Result(Dictionary<GuardCategory, double> scores, Stopwatch watch) =>
        GuardResult.FromScores(Name, scores.Select(s => new GuardCategoryScore(s.Key, s.Value, s.Value >= _options.Threshold, Name)), watch.Elapsed);

    private GuardResult CheckOutput(string text, Stopwatch watch)
    {
        if (_canaries.Length == 0)
        {
            return GuardResult.NotScanned(Name, "No canaries configured; the pattern guard has no output rules.", watch.Elapsed);
        }

        var clean = TextViews.Clean(text);
        var hidden = TextViews.FindHidden(text);
        var sources = new List<string> { clean };
        sources.AddRange(new[] { hidden.TagPayload, hidden.SelectorPayload }.OfType<string>().Select(TextViews.Clean));
        sources.AddRange(TextViews.Decode(clean, _options.MaxDecodeChars).Select(d => d.Text));
        bool leaked = sources.Any(source =>
        {
            var squashed = TextViews.SquashAll(source);
            return _canaries.Any(c => source.Contains(c.Clean, StringComparison.OrdinalIgnoreCase)
                || (c.Squashed.Length >= 6 && squashed.Contains(c.Squashed, StringComparison.Ordinal)));
        });
        GuardCategoryScore[] scores = leaked ? [new(GuardCategory.PromptExtraction, 1.0, true, Name)] : [];
        return GuardResult.FromScores(Name, scores, watch.Elapsed);
    }
}
