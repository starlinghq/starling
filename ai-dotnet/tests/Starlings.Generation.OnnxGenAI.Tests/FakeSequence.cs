using Starlings.Generation.OnnxGenAI;

namespace Starlings.Generation.OnnxGenAI.Tests;

/// <summary>A scripted sequence: returns the given ids in order and reports done after an end-of-sequence id.</summary>
internal sealed class FakeSequence : IDecodeSequence
{
    private readonly IReadOnlyList<int> _script;
    private readonly HashSet<int> _eos;
    private readonly Action<int>? _onStep;

    public FakeSequence(GenAISequenceOptions options, IReadOnlyList<int> script, IEnumerable<int> eos, Action<int>? onStep = null)
    {
        Options = options;
        _script = script;
        _eos = [.. eos];
        _onStep = onStep;
    }

    public GenAISequenceOptions Options { get; }

    public int VocabularySize => 248320;

    public bool IsDone { get; private set; }

    public int PromptTokens { get; private set; }

    public int Steps { get; private set; }

    public bool Disposed { get; private set; }

    public List<int> Appends { get; } = [];

    public Action<int>? OnAppend { get; set; }

    public void AppendTokens(ReadOnlySpan<int> tokenIds)
    {
        PromptTokens += tokenIds.Length;
        Appends.Add(tokenIds.Length);
        OnAppend?.Invoke(Appends.Count);
    }

    public int GenerateNextToken()
    {
        if (IsDone)
        {
            throw new InvalidOperationException("Generator is done.");
        }

        if (PromptTokens + Steps >= Options.MaxLength)
        {
            throw new InvalidOperationException("Past max_length.");
        }

        int id = _script[Steps++];
        _onStep?.Invoke(Steps);
        if (_eos.Contains(id) || PromptTokens + Steps == Options.MaxLength)
        {
            IsDone = true;
        }

        return id;
    }

    /// <summary>
    /// When set, the logits after a step have <see cref="PeakLogit"/> at the id that step returned and
    /// <see cref="RunnerUpLogit"/> at id 0 (all others 0), as GenAI's logits after a step are those that selected it.
    /// </summary>
    public bool ScriptedLogits { get; set; }

    public const float PeakLogit = 4f;

    public const float RunnerUpLogit = 1f;

    /// <summary>Called on every logits copy with the number of steps run so far.</summary>
    public List<int> LogitCopiesAtStep { get; } = [];

    /// <summary>Optional release owned the way GenAISequence owns its native generator (lifetime tests).</summary>
    public NativeLifetime? Lifetime { get; set; }

    public void CopyLastLogits(Span<float> destination)
    {
        LogitCopiesAtStep.Add(Steps);
        destination.Clear();
        if (ScriptedLogits && Steps > 0)
        {
            destination[0] = RunnerUpLogit;
            destination[_script[Steps - 1]] = PeakLogit;
        }
    }

    public void Dispose()
    {
        Disposed = true;
        Lifetime?.Dispose();
    }
}
