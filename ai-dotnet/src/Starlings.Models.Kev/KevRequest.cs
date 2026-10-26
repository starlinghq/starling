using System.Collections.ObjectModel;
using System.Text.Json;

namespace Starlings.Models.Kev;

/// <summary>
/// A state, instruction or description: plain text, or a JSON value rendered as <see cref="KevText.Render(JsonElement)"/> does
/// (System One's <c>JSONContent</c>). Strings convert implicitly.
/// </summary>
public sealed class KevContent
{
    private readonly string? _text;
    private readonly JsonElement _json;
    private readonly bool _isJson;

    private KevContent(string text) => _text = text;

    private KevContent(JsonElement json)
    {
        _json = json.Clone();
        _isJson = true;
    }

    /// <summary>Plain text, passed to the model as it is (after delimiter escaping).</summary>
    public static KevContent Text(string text) => new(text ?? throw new ArgumentNullException(nameof(text)));

    /// <summary>A JSON value (object, array, string, number, boolean or null), rendered as upstream renders it. Experimental.</summary>
    public static KevContent Json(JsonElement value) => new(value);

    /// <summary>Parses JSON text into a JSON value. Experimental.</summary>
    public static KevContent ParseJson(string json)
    {
        using var doc = JsonDocument.Parse(json);
        return new(doc.RootElement);
    }

    /// <summary>Plain text.</summary>
    public static implicit operator KevContent(string text) => Text(text);

    /// <summary>True for JSON content.</summary>
    public bool IsJson => _isJson;

    /// <summary>JSON null or the empty string (<c>option_text</c>'s "no description").</summary>
    internal bool IsNullOrEmptyString => _isJson
        ? _json.ValueKind == JsonValueKind.Null || (_json.ValueKind == JsonValueKind.String && _json.GetString()!.Length == 0)
        : _text!.Length == 0;

    /// <summary>The text the model sees (before delimiter escaping).</summary>
    public string Render() => _isJson ? KevText.Render(_json) : _text!;

    /// <inheritdoc />
    public override string ToString() => Render();
}

/// <summary>Question types of the System One API.</summary>
public enum KevQuestionType
{
    /// <summary>Yes or no: options <c>no</c>, <c>yes</c>; the answer is p(yes).</summary>
    Noul,

    /// <summary>One of 1 to 255 named options.</summary>
    Choice,

    /// <summary>One of 1 to 255 ordered levels; the answer is the expected level.</summary>
    Score,
}

/// <summary>One typed question (System One <c>Noul</c>, <c>Choice</c> or <c>Score</c>).</summary>
public sealed class KevQuestion
{
    private KevQuestion(KevQuestionType type, KevContent? instructions, IReadOnlyList<string> keys, IReadOnlyList<string> options, IReadOnlyList<string>? legend)
    {
        Type = type;
        Instructions = instructions;
        Keys = keys;
        Options = options;
        Legend = legend;
    }

    /// <summary>The question type.</summary>
    public KevQuestionType Type { get; }

    /// <summary>The instruction, or null (rendered as empty).</summary>
    public KevContent? Instructions { get; }

    /// <summary>Keys the probabilities are reported under, in option order (<c>question_keys</c>).</summary>
    public IReadOnlyList<string> Keys { get; }

    /// <summary>Option texts the model reads, in order.</summary>
    public IReadOnlyList<string> Options { get; }

    /// <summary>Score questions: the rendered level descriptions (the System One <c>legend</c>); otherwise null.</summary>
    public IReadOnlyList<string>? Legend { get; }

    /// <summary>
    /// A yes/no question. Options are <c>no</c> and <c>yes</c>, each followed by its description when given
    /// (System One <c>criteria</c> keys <c>false</c> and <c>true</c>).
    /// </summary>
    public static KevQuestion Noul(KevContent? instructions, KevContent? noDescription = null, KevContent? yesDescription = null) =>
        new(KevQuestionType.Noul, instructions, ["false", "true"],
            [KevText.OptionText("no", noDescription), KevText.OptionText("yes", yesDescription)], null);

    /// <summary>A choice among named options (name to optional description, in order; 1 to 255).</summary>
    /// <exception cref="ArgumentException">No options, more than 255, or a repeated name.</exception>
    public static KevQuestion Choice(KevContent? instructions, IEnumerable<KeyValuePair<string, KevContent?>> criteria)
    {
        ArgumentNullException.ThrowIfNull(criteria);
        var keys = new List<string>();
        var options = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (name, description) in criteria)
        {
            ArgumentNullException.ThrowIfNull(name, nameof(criteria));
            if (!seen.Add(name))
            {
                throw new ArgumentException($"Option '{name}' appears twice.", nameof(criteria));
            }

            keys.Add(name);
            options.Add(KevText.OptionText(name, description));
        }

        CheckCount(options.Count, nameof(criteria));
        return new(KevQuestionType.Choice, instructions, keys.AsReadOnly(), options.AsReadOnly(), null);
    }

    /// <summary>A choice among option names without descriptions.</summary>
    public static KevQuestion Choice(KevContent? instructions, params IEnumerable<string> names) =>
        Choice(instructions, names.Select(n => new KeyValuePair<string, KevContent?>(n, null)));

    /// <summary>A score over ordered levels (level descriptions in order; 1 to 255). Keys are <c>"0"</c>, <c>"1"</c>, ….</summary>
    public static KevQuestion Score(KevContent? instructions, params IEnumerable<KevContent> levels)
    {
        ArgumentNullException.ThrowIfNull(levels);
        var options = levels.Select(l => (l ?? throw new ArgumentException("A level is null.", nameof(levels))).Render()).ToList();
        CheckCount(options.Count, nameof(levels));
        var keys = Enumerable.Range(0, options.Count).Select(i => i.ToString(System.Globalization.CultureInfo.InvariantCulture)).ToList();
        return new(KevQuestionType.Score, instructions, keys.AsReadOnly(), options.AsReadOnly(), options.AsReadOnly());
    }

    private static void CheckCount(int count, string name)
    {
        if (count is < 1 or > KevModelProfile.MaxOptions)
        {
            throw new ArgumentException($"A question needs 1 to {KevModelProfile.MaxOptions} options; got {count}.", name);
        }
    }
}

/// <summary>
/// A System One request: one state and named questions about it, answered independently (each question is its own row
/// on the shared state).
/// </summary>
public sealed class KevRequest
{
    /// <summary>Creates a request.</summary>
    /// <param name="state">The document (text or JSON).</param>
    /// <param name="questions">Question id to question, in order; at least one.</param>
    public KevRequest(KevContent state, IEnumerable<KeyValuePair<string, KevQuestion>> questions)
    {
        ArgumentNullException.ThrowIfNull(state);
        State = state;
        Questions = CheckQuestions(questions);
    }

    /// <summary>Questions as a request takes them: ids unique and not null, at least one.</summary>
    internal static ReadOnlyCollection<KeyValuePair<string, KevQuestion>> CheckQuestions(IEnumerable<KeyValuePair<string, KevQuestion>> questions)
    {
        ArgumentNullException.ThrowIfNull(questions);
        var list = new List<KeyValuePair<string, KevQuestion>>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var pair in questions)
        {
            ArgumentNullException.ThrowIfNull(pair.Key, nameof(questions));
            ArgumentNullException.ThrowIfNull(pair.Value, nameof(questions));
            if (!seen.Add(pair.Key))
            {
                throw new ArgumentException($"Question id '{pair.Key}' appears twice.", nameof(questions));
            }

            list.Add(pair);
        }

        if (list.Count == 0)
        {
            throw new ArgumentException("A request needs at least one question.", nameof(questions));
        }

        return new ReadOnlyCollection<KeyValuePair<string, KevQuestion>>(list);
    }

    /// <summary>
    /// Parses a System One request body (<c>POST /v1/systemone</c>: <c>state</c>, <c>questions</c> by id with
    /// <c>type</c> <c>noul</c>/<c>choice</c>/<c>score</c>, <c>instructions</c> and <c>criteria</c>; <c>model</c> is
    /// ignored) as <c>kev.api.SystemOneRequest</c> reads it. Experimental.
    /// </summary>
    /// <exception cref="JsonException">The body is not JSON.</exception>
    /// <exception cref="ArgumentException">A required field is missing or has the wrong shape.</exception>
    public static KevRequest ParseSystemOne(string json)
    {
        ArgumentNullException.ThrowIfNull(json);
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        if (root.ValueKind != JsonValueKind.Object || !root.TryGetProperty("state", out var state)
            || !root.TryGetProperty("questions", out var questions) || questions.ValueKind != JsonValueKind.Object)
        {
            throw new ArgumentException("A System One request is an object with 'state' and an object 'questions'.", nameof(json));
        }

        var list = new List<KeyValuePair<string, KevQuestion>>();
        foreach (var q in questions.EnumerateObject())
        {
            var body = q.Value;
            string? type = body.ValueKind == JsonValueKind.Object && body.TryGetProperty("type", out var t) ? t.GetString() : null;
            KevContent? instructions = body.TryGetProperty("instructions", out var ins) ? KevContent.Json(ins) : null;
            bool hasCriteria = body.TryGetProperty("criteria", out var criteria) && criteria.ValueKind != JsonValueKind.Null;
            KevQuestion question = type switch
            {
                "noul" when !hasCriteria => KevQuestion.Noul(instructions),
                "noul" when criteria.ValueKind == JsonValueKind.Object => KevQuestion.Noul(
                    instructions,
                    criteria.TryGetProperty("false", out var no) ? KevContent.Json(no) : null,
                    criteria.TryGetProperty("true", out var yes) ? KevContent.Json(yes) : null),
                "choice" when hasCriteria && criteria.ValueKind == JsonValueKind.Object => KevQuestion.Choice(
                    instructions, LastWins(criteria).Select(p => new KeyValuePair<string, KevContent?>(p.Name, KevContent.Json(p.Value)))),
                "score" when hasCriteria && criteria.ValueKind == JsonValueKind.Array => KevQuestion.Score(
                    instructions, criteria.EnumerateArray().Select(KevContent.Json)),
                _ => throw new ArgumentException($"Question '{q.Name}' is not a noul, choice or score question with valid criteria.", nameof(json)),
            };
            list.RemoveAll(p => p.Key == q.Name); // a repeated id: Python's dict keeps the last value at the first position
            list.Add(new(q.Name, question));
        }

        return new KevRequest(KevContent.Json(state), list);
    }

    // A JSON object as a Python dict: a repeated key keeps its first position and takes the last value.
    private static List<(string Name, JsonElement Value)> LastWins(JsonElement obj)
    {
        var entries = new List<(string Name, JsonElement Value)>();
        foreach (var p in obj.EnumerateObject())
        {
            int at = entries.FindIndex(e => e.Name == p.Name);
            if (at >= 0)
            {
                entries[at] = (p.Name, p.Value);
            }
            else
            {
                entries.Add((p.Name, p.Value));
            }
        }

        return entries;
    }

    /// <summary>The state.</summary>
    public KevContent State { get; }

    /// <summary>The questions, in request order.</summary>
    public IReadOnlyList<KeyValuePair<string, KevQuestion>> Questions { get; }

    /// <summary><c>to_record</c>: the internal record the encoder reads.</summary>
    public KevRecord ToRecord() => new(State.Render(), RecordQuestions(Questions));

    internal static KevRecordQuestion[] RecordQuestions(IReadOnlyList<KeyValuePair<string, KevQuestion>> questions) =>
        [.. questions.Select(q => new KevRecordQuestion(q.Key, q.Value.Instructions?.Render() ?? string.Empty, q.Value.Options))];
}

/// <summary>Kev's internal record (the input of <c>kev.model.encode</c>): rendered state and per-question instruction and options.</summary>
/// <param name="State">Rendered state text.</param>
/// <param name="Questions">Questions in order.</param>
public sealed record KevRecord(string State, IReadOnlyList<KevRecordQuestion> Questions);

/// <summary>One question of a <see cref="KevRecord"/>.</summary>
/// <param name="Id">Question id (not seen by the model).</param>
/// <param name="Instruction">Rendered instruction text.</param>
/// <param name="Options">Rendered option texts, in order (1 to 255).</param>
public sealed record KevRecordQuestion(string Id, string Instruction, IReadOnlyList<string> Options);
