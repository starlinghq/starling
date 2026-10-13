using Starlings.Generation.OnnxGenAI;

namespace Starlings.Generation.OnnxGenAI.Tests;

/// <summary>A scripted sequence: returns the given ids in order and reports done after an end-of-sequence id.</summary>
internal sealed class FakeSequence : IGenerationSequence
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

    public void CopyLastLogits(Span<float> destination) => destination.Clear();

    public void Dispose() => Disposed = true;
}
