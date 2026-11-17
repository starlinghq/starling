namespace Starlings.Guardrails;

/// <summary>Where a text sits in a conversation, relative to the model being guarded.</summary>
public enum GuardStage
{
    /// <summary>Text going into the model: a user message, a tool result, a retrieved document.</summary>
    Input,

    /// <summary>Text the model produced: a response.</summary>
    Output,
}

/// <summary>Who or what a text came from. Guards may treat sources differently (a tool result is never an instruction).</summary>
public enum GuardSource
{
    /// <summary>A user message.</summary>
    User,

    /// <summary>A tool or function result.</summary>
    ToolResult,

    /// <summary>A retrieved or attached document (web page, email, file).</summary>
    Document,

    /// <summary>A response of the guarded model.</summary>
    Assistant,
}

/// <summary>A category of issue a guard reports. Compared by <see cref="Id"/> (ordinal).</summary>
/// <param name="Id">Stable, lower-case, hyphenated id, for example <c>prompt-injection</c>.</param>
public readonly record struct GuardCategory(string Id)
{
    /// <summary>Instructions aimed at the model that try to override or replace the ones it was given.</summary>
    public static GuardCategory PromptInjection { get; } = new("prompt-injection");

    /// <summary>Attempts to make the model drop its rules or safety guidelines (personas, "no restrictions" modes).</summary>
    public static GuardCategory Jailbreak { get; } = new("jailbreak");

    /// <summary>Attempts to reveal the system prompt or hidden configuration; on output, a leak of it.</summary>
    public static GuardCategory PromptExtraction { get; } = new("prompt-extraction");

    /// <summary>
    /// Text hidden or disguised from filters: instructions smuggled in invisible Unicode (tag characters, emoji variation
    /// selectors), bidi overrides, or an attack that only shows once decoded (base64, ROT13, leetspeak, look-alike letters).
    /// </summary>
    public static GuardCategory Obfuscation { get; } = new("obfuscation");

    /// <summary>Practical help with weapons or with attacking people.</summary>
    public static GuardCategory Violence { get; } = new("violence");

    /// <summary>Encouragement of, or methods for, self-harm or suicide.</summary>
    public static GuardCategory SelfHarm { get; } = new("self-harm");

    /// <summary>Content that demeans or attacks people for a protected trait.</summary>
    public static GuardCategory Hate { get; } = new("hate");

    /// <summary>Practical help with a crime (intrusion, fraud, laundering).</summary>
    public static GuardCategory CriminalActivity { get; } = new("criminal-activity");

    /// <summary>Disclosure of someone's private data (card numbers, credentials, personal records).</summary>
    public static GuardCategory PrivateData { get; } = new("private-data");

    /// <summary>Practical help with chemical, biological, radiological or nuclear weapons.</summary>
    public static GuardCategory MassHarmWeapons { get; } = new("mass-harm-weapons");

    /// <inheritdoc />
    public override string ToString() => Id;
}

/// <summary>One text to check.</summary>
/// <param name="Text">The text.</param>
/// <param name="Stage">Input to the model or output of it.</param>
/// <param name="Source">Where the text came from.</param>
public sealed record GuardRequest(string Text, GuardStage Stage, GuardSource Source)
{
    /// <summary>A user message going into the model.</summary>
    public static GuardRequest UserInput(string text) => new(text, GuardStage.Input, GuardSource.User);

    /// <summary>A model response.</summary>
    public static GuardRequest Response(string text) => new(text, GuardStage.Output, GuardSource.Assistant);
}

/// <summary>Whether a guard looked at the text.</summary>
public enum GuardStatus
{
    /// <summary>The guard checked the text; <see cref="GuardResult.Score"/> and the categories are its findings.</summary>
    Scanned,

    /// <summary>
    /// The guard did not check the text (too long, unsupported stage, no categories for the stage). Not a pass:
    /// <see cref="GuardResult.Reason"/> says why, and policies decide whether to allow or block it.
    /// </summary>
    NotScanned,
}

/// <summary>One category's score from one guard.</summary>
/// <param name="Category">The category.</param>
/// <param name="Score">0 to 1; higher means more likely. Not a calibrated probability unless the guard says so.</param>
/// <param name="Flagged">True when <paramref name="Score"/> reached the guard's threshold for the category.</param>
/// <param name="Detector">The guard that produced the score (<see cref="IContentGuard.Name"/>).</param>
public sealed record GuardCategoryScore(GuardCategory Category, double Score, bool Flagged, string Detector);

/// <summary>A guard's verdict on one text.</summary>
public sealed record GuardResult
{
    /// <summary>Whether the text was checked.</summary>
    public required GuardStatus Status { get; init; }

    /// <summary>The highest category score, 0 when none (or when not scanned).</summary>
    public double Score { get; init; }

    /// <summary>True when any category is flagged.</summary>
    public bool Flagged { get; init; }

    /// <summary>Category scores, highest first. A cascade keeps the highest score per category over its guards.</summary>
    public IReadOnlyList<GuardCategoryScore> Categories { get; init; } = [];

    /// <summary>The guards that ran, in order (a cascade lists each tier it reached).</summary>
    public IReadOnlyList<string> Detectors { get; init; } = [];

    /// <summary>Why the text was not scanned; null when scanned.</summary>
    public string? Reason { get; init; }

    /// <summary>Wall time spent checking.</summary>
    public TimeSpan Elapsed { get; init; }

    /// <summary>The flagged categories, highest first.</summary>
    public IEnumerable<GuardCategoryScore> FlaggedCategories => Categories.Where(c => c.Flagged);

    /// <summary>A scanned result from category scores (sorted highest first; score and flag derived).</summary>
    public static GuardResult FromScores(string detector, IEnumerable<GuardCategoryScore> scores, TimeSpan elapsed)
    {
        ArgumentNullException.ThrowIfNull(detector);
        ArgumentNullException.ThrowIfNull(scores);
        var sorted = scores.OrderByDescending(s => s.Score).ToArray();
        return new GuardResult
        {
            Status = GuardStatus.Scanned,
            Score = sorted.Length == 0 ? 0 : sorted[0].Score,
            Flagged = sorted.Any(s => s.Flagged),
            Categories = sorted,
            Detectors = [detector],
            Elapsed = elapsed,
        };
    }

    /// <summary>A not-scanned result.</summary>
    public static GuardResult NotScanned(string detector, string reason, TimeSpan elapsed = default) => new()
    {
        Status = GuardStatus.NotScanned,
        Detectors = [detector],
        Reason = reason,
        Elapsed = elapsed,
    };
}

/// <summary>
/// A guard: checks one text and returns a score and categories. Implementations are thread-safe and do not keep
/// per-call state; a guard that needs a model takes the loaded model and does not own its lifetime unless it says so.
/// </summary>
public interface IContentGuard
{
    /// <summary>Short id used in results, for example <c>patterns</c> or <c>kev-0.8b</c>.</summary>
    string Name { get; }

    /// <summary>Checks a text. Never throws for content; throws only for cancellation or a broken guard.</summary>
    ValueTask<GuardResult> CheckAsync(GuardRequest request, CancellationToken cancellationToken = default);
}
