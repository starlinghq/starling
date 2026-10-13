using System.Collections.Concurrent;
using System.Collections.Immutable;
using Starlings.Generation.OnnxGenAI;
using Starlings.Tokenizers;

namespace Starlings.Models.Qwen35.Extensions.AI.Tests;

/// <summary>
/// A <see cref="Qwen35ChatModel"/> whose native ONNX Runtime GenAI sequence is replaced by a script: the real template,
/// tokenizer, decode loop, stop handling and incremental detokenizer run; only the model's token choice is scripted.
/// </summary>
internal sealed class ScriptedBackend
{
    public const int ContextLength = 262144;

    public ScriptedBackend(Tokenizer tokenizer, Func<int[], IReadOnlyList<int>> script)
    {
        Tokenizer = tokenizer;
        Script = script;
        var generator = new TextGenerator(
            options => Track(new ScriptedSequence(options, this)),
            tokenizer,
            [Qwen35ChatModel.EndOfTextTokenId, Qwen35ChatModel.ImEndTokenId],
            ContextLength,
            Qwen35ChatModel.PadTokenIdOverlay);
        Model = Qwen35ChatModel.CreateWithoutBackend(generator, Info);
    }

    public static Qwen35ModelInfo Info { get; } = new()
    {
        Profile = "qwen3.5-0.8b/ortgenai-fp32x",
        Checkpoint = $"{Qwen35ModelAssets.Repository}@{Qwen35ModelAssets.Revision}",
        Variant = Qwen35Variant.Fp32x,
        Provider = GenAIExecutionProvider.Cpu,
        GenAIVersion = "scripted",
        Tokenizer = "qwen35-bpe-chat",
        ChatTemplate = "owned C# renderer",
        ModelDirectory = "/scripted",
        ConfigOverlays = [],
        SerializesNativeCalls = false,
        ProviderOptions = new Dictionary<string, string>(),
        ContextLength = ContextLength,
        StopTokenIds = [],
    };

    public Tokenizer Tokenizer { get; }

    public Func<int[], IReadOnlyList<int>> Script { get; set; }

    public Qwen35ChatModel Model { get; }

    public ConcurrentQueue<ScriptedSequence> Sequences { get; } = new();

    /// <summary>Called on the generating thread after each prompt chunk (1-based chunk number).</summary>
    public Action<ScriptedSequence, int>? OnAppend { get; set; }

    /// <summary>Called on the generating thread before each decode step returns (1-based step number).</summary>
    public Action<ScriptedSequence, int>? OnStep { get; set; }

    public ScriptedSequence Single() => Assert.Single(Sequences);

    private ScriptedSequence Track(ScriptedSequence s)
    {
        Sequences.Enqueue(s);
        return s;
    }
}

/// <summary>One request's scripted sequence: records the prompt, chunks, steps, threads and disposal.</summary>
internal sealed class ScriptedSequence(GenAISequenceOptions options, ScriptedBackend backend) : IGenerationSequence
{
    private readonly List<int> _prompt = [];
    private IReadOnlyList<int>? _script;
    private volatile bool _disposed;

    public GenAISequenceOptions Options { get; } = options;

    public int VocabularySize => 248320;

    public bool IsDone { get; private set; }

    public ImmutableArray<int> Prompt => [.. _prompt];

    public List<int> Chunks { get; } = [];

    public int Steps { get; private set; }

    public bool Disposed => _disposed;

    public ConcurrentBag<int> Threads { get; } = [Environment.CurrentManagedThreadId];

    public string? ThreadName { get; } = Thread.CurrentThread.Name;

    public void AppendTokens(ReadOnlySpan<int> tokenIds)
    {
        Threads.Add(Environment.CurrentManagedThreadId);
        _prompt.AddRange(tokenIds.ToArray());
        Chunks.Add(tokenIds.Length);
        backend.OnAppend?.Invoke(this, Chunks.Count);
    }

    public int GenerateNextToken()
    {
        Threads.Add(Environment.CurrentManagedThreadId);
        _script ??= backend.Script([.. _prompt]);
        if (IsDone || _prompt.Count + Steps >= Options.MaxLength)
        {
            throw new InvalidOperationException("Stepped past the end of the sequence.");
        }

        int id = _script[Steps++];
        backend.OnStep?.Invoke(this, Steps);
        if (id is Qwen35ChatModel.ImEndTokenId or Qwen35ChatModel.EndOfTextTokenId || _prompt.Count + Steps == Options.MaxLength)
        {
            IsDone = true;
        }

        return id;
    }

    public void CopyLastLogits(Span<float> destination) => destination.Clear();

    public void Dispose()
    {
        Threads.Add(Environment.CurrentManagedThreadId);
        _disposed = true;
    }
}
