using System.Diagnostics;

namespace Starlings.Guardrails;

/// <summary>One tier of a <see cref="CascadeGuard"/> and when the cascade stops after it.</summary>
/// <param name="Guard">The guard.</param>
/// <param name="StopWhenFlagged">Stop when this tier flags the text (later, costlier tiers do not run).</param>
/// <param name="StopBelow">Stop when this tier scanned the text and scored it below this value; null never stops on a low score.</param>
public sealed record CascadeTier(IContentGuard Guard, bool StopWhenFlagged = true, double? StopBelow = null);

/// <summary>
/// Runs guards in order, cheapest first, and stops as soon as a tier's stop rule says so, so costly guards run only on
/// the texts that need them. A tier that does not scan a text passes it on. The result keeps the highest score per
/// category over the tiers that ran (flagged if any of them flagged it), lists every tier that ran in
/// <see cref="GuardResult.Detectors"/>, and is not scanned only when no tier scanned the text.
/// </summary>
public sealed class CascadeGuard : IContentGuard
{
    private readonly CascadeTier[] _tiers;

    /// <summary>Creates a cascade.</summary>
    /// <exception cref="ArgumentException">No tiers.</exception>
    public CascadeGuard(params IEnumerable<CascadeTier> tiers)
    {
        ArgumentNullException.ThrowIfNull(tiers);
        _tiers = [.. tiers];
        if (_tiers.Length == 0)
        {
            throw new ArgumentException("A cascade needs at least one tier.", nameof(tiers));
        }

        if (_tiers.Any(t => t is null || t.Guard is null))
        {
            throw new ArgumentException("A tier or its guard is null.", nameof(tiers));
        }

        Name = string.Join('>', _tiers.Select(t => t.Guard.Name));
    }

    /// <inheritdoc />
    public string Name { get; }

    /// <summary>The tiers, in order.</summary>
    public IReadOnlyList<CascadeTier> Tiers => _tiers;

    /// <inheritdoc />
    public async ValueTask<GuardResult> CheckAsync(GuardRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        var watch = Stopwatch.StartNew();
        var best = new Dictionary<GuardCategory, GuardCategoryScore>();
        var flaggedBy = new HashSet<GuardCategory>();
        var detectors = new List<string>();
        var reasons = new List<string>();
        bool scanned = false;
        foreach (var tier in _tiers)
        {
            var result = await tier.Guard.CheckAsync(request, cancellationToken).ConfigureAwait(false);
            detectors.Add(tier.Guard.Name);
            if (result.Status == GuardStatus.NotScanned)
            {
                reasons.Add($"{tier.Guard.Name}: {result.Reason}");
                continue;
            }

            scanned = true;
            foreach (var score in result.Categories)
            {
                if (score.Flagged)
                {
                    flaggedBy.Add(score.Category);
                }

                if (!best.TryGetValue(score.Category, out var current) || score.Score > current.Score)
                {
                    best[score.Category] = score;
                }
            }

            if ((tier.StopWhenFlagged && result.Flagged) || (tier.StopBelow is { } low && result.Score < low))
            {
                break;
            }
        }

        if (!scanned)
        {
            return new GuardResult { Status = GuardStatus.NotScanned, Detectors = detectors, Reason = string.Join("; ", reasons), Elapsed = watch.Elapsed };
        }

        var merged = best.Values.Select(s => flaggedBy.Contains(s.Category) && !s.Flagged ? s with { Flagged = true } : s);
        return GuardResult.FromScores(Name, merged, watch.Elapsed) with { Detectors = detectors };
    }
}

/// <summary>Preset trade-offs between cost and coverage. Pick the model (and its size) separately.</summary>
public enum GuardLevel
{
    /// <summary>Patterns only: microseconds, no model. Catches common phrasings; misses paraphrased and indirect attacks.</summary>
    Light,

    /// <summary>Patterns first; the model runs only on texts the patterns did not flag.</summary>
    Balanced,

    /// <summary>Patterns and the model on every text; scores merged. Pair with the largest model you can afford.</summary>
    Strict,
}

/// <summary>Builds the guard for a <see cref="GuardLevel"/>.</summary>
public static class GuardLevels
{
    /// <summary>Creates the guard for a level.</summary>
    /// <param name="level">The level.</param>
    /// <param name="model">A model guard (for example a Kev guard); required for <see cref="GuardLevel.Balanced"/> and <see cref="GuardLevel.Strict"/>.</param>
    /// <param name="patterns">Pattern guard options (custom rules, canaries); null for the built-in rules.</param>
    /// <exception cref="ArgumentException">A model level without a model.</exception>
    public static IContentGuard Create(GuardLevel level, IContentGuard? model = null, PatternGuardOptions? patterns = null)
    {
        var patternGuard = new PatternGuard(patterns);
        if (level == GuardLevel.Light)
        {
            return patternGuard;
        }

        if (model is null)
        {
            throw new ArgumentException($"Level {level} needs a model guard.", nameof(model));
        }

        return level switch
        {
            GuardLevel.Balanced => new CascadeGuard(new CascadeTier(patternGuard), new CascadeTier(model)),
            GuardLevel.Strict => new CascadeGuard(new CascadeTier(patternGuard, StopWhenFlagged: false), new CascadeTier(model)),
            _ => throw new ArgumentOutOfRangeException(nameof(level), level, "Unknown level."),
        };
    }
}
