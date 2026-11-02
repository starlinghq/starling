using Starlings.Runtime.Onnx;

namespace Starlings.Models.Qwen3Embedding;

/// <summary>
/// <c>Qwen/Qwen3-Embedding-0.6B</c> text embeddings on ONNX Runtime CPU (fp32): the pinned tokenizer (which appends
/// <c>&lt;|endoftext|&gt;</c>), the exported graph, the hidden state at the last token, L2 normalization; 1024 floats.
/// Experimental (<see cref="Qwen3EmbeddingCapabilities.Embeddings"/>).
/// </summary>
/// <remarks>
/// <para>
/// <b>Prompts.</b> Queries take an instruction, documents none: embed queries with the prompt <c>query</c>
/// (<see cref="Qwen3EmbeddingProfile.QueryPrompt"/>) or with <see cref="FormatQuery"/> for a task of your own, and
/// documents with <c>none</c> (the default). The <see cref="ITextEmbedder"/> members embed text exactly as given.
/// </para>
/// <para>
/// <b>Length.</b> Text longer than 8192 tokens (prompt and the appended token included) is truncated from the right;
/// the appended <c>&lt;|endoftext|&gt;</c> is kept. The attention cost grows with the square of the length, and a
/// 8192-token row needs several GB of memory.
/// </para>
/// <para><b>Threading.</b> Concurrent calls are safe; <see cref="Dispose"/> is idempotent.</para>
/// </remarks>
public sealed class Qwen3TextEmbedder : ITextEmbedder, IDisposable
{
    private readonly Qwen3Session _session;

    private Qwen3TextEmbedder(Qwen3Session session) => _session = session;

    /// <summary>The profile.</summary>
    public Qwen3EmbeddingProfile Profile => _session.Profile;

    /// <summary>Floats per embedding (1024).</summary>
    public int Dimension => Qwen3EmbeddingProfile.HiddenSize;

    /// <summary>Tokens per text at most (8192), prompt and the appended token included; longer text is truncated.</summary>
    public int MaxSequenceLength => Profile.MaxTokens;

    /// <summary>Loaded model and native runtime diagnostics.</summary>
    public OnnxSessionDiagnostics Diagnostics => _session.Session.Diagnostics;

    /// <summary>
    /// Verifies a released directory (<see cref="Qwen3EmbeddingProfile.OpenDirectory"/>) and creates a CPU session.
    /// Never downloads.
    /// </summary>
    /// <param name="directory">The directory <c>release_qwen3_onnx.py</c> wrote for <see cref="Qwen3EmbeddingProfile.Embedding06B"/>.</param>
    /// <param name="options">Threads, batching and verification.</param>
    /// <param name="profile">The profile; <see cref="Qwen3EmbeddingProfile.Embedding06B"/> when null.</param>
    public static Qwen3TextEmbedder Create(string directory, Qwen3EmbeddingOptions? options = null, Qwen3EmbeddingProfile? profile = null)
    {
        profile ??= Qwen3EmbeddingProfile.Embedding06B;
        if (profile.Task != Qwen3EmbeddingTask.Embedding)
        {
            throw new ArgumentException("Use Qwen3Reranker for a reranking profile.", nameof(profile));
        }

        return new Qwen3TextEmbedder(new Qwen3Session(profile, directory, options));
    }

    /// <summary>
    /// The query text for a custom task, in the model card's format: <c>"Instruct: {task}\nQuery:{query}"</c> (no space
    /// after <c>Query:</c>, as the pinned <c>query</c> prompt). Embed the result with the prompt <c>none</c>.
    /// </summary>
    /// <param name="task">A one-sentence task description.</param>
    /// <param name="query">The query.</param>
    public static string FormatQuery(string task, string query)
    {
        ArgumentNullException.ThrowIfNull(task);
        ArgumentNullException.ThrowIfNull(query);
        return $"Instruct: {task}\nQuery:{query}";
    }

    /// <summary>Tokenizes <paramref name="text"/> after a named prompt (<c>none</c>, <c>document</c> or <c>query</c>), truncated to 8192 tokens.</summary>
    public Qwen3Input Tokenize(string text, string prompt = "none") =>
        _session.Tokenization.TokenizeText(text, _session.Tokenization.PromptText(prompt));

    /// <summary>Embeds one text after a named prompt.</summary>
    public float[] Embed(string text, string prompt = "none", CancellationToken cancellationToken = default) =>
        EmbedTokens([Tokenize(text, prompt)], cancellationToken)[0];

    /// <summary>Embeds texts in input order, all after the same named prompt.</summary>
    public float[][] EmbedBatch(IReadOnlyList<string> texts, string prompt = "none", CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(texts);
        string prefix = _session.Tokenization.PromptText(prompt);
        var inputs = new Qwen3Input[texts.Count];
        for (int i = 0; i < inputs.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            inputs[i] = _session.Tokenization.TokenizeText(texts[i] ?? throw new ArgumentException($"Text {i} is null.", nameof(texts)), prefix);
        }

        return EmbedTokens(inputs, cancellationToken);
    }

    /// <summary>
    /// Embeds texts exactly as given (prompt <c>none</c>) and writes each text's token count (after truncation, the
    /// appended token included). The <see cref="ITextEmbedder"/> form.
    /// </summary>
    public float[][] EmbedBatch(IReadOnlyList<string> texts, Span<int> tokenCounts, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(texts);
        if (tokenCounts.Length != texts.Count)
        {
            throw new ArgumentException($"tokenCounts holds {tokenCounts.Length} values; {texts.Count} texts need {texts.Count}.", nameof(tokenCounts));
        }

        var inputs = new Qwen3Input[texts.Count];
        for (int i = 0; i < inputs.Length; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            inputs[i] = _session.Tokenization.TokenizeText(texts[i] ?? throw new ArgumentException($"Text {i} is null.", nameof(texts)), string.Empty);
            tokenCounts[i] = inputs[i].Length;
        }

        return EmbedTokens(inputs, cancellationToken);
    }

    /// <summary>Embeds tokenized rows (from <see cref="Tokenize"/>; right padding with the pad id is allowed).</summary>
    public float[][] EmbedTokens(IReadOnlyList<Qwen3Input> inputs, CancellationToken cancellationToken = default) =>
        _session.Predict(inputs, cancellationToken);

    /// <summary>Tokens <paramref name="text"/> needs with the prompt <c>none</c>, the appended token included, before truncation.</summary>
    public int CountTokens(string text) => _session.Tokenization.CountText(text, string.Empty);

    /// <summary>Tokens <paramref name="text"/> needs after a named prompt, before truncation.</summary>
    public int CountTokens(string text, string prompt) => _session.Tokenization.CountText(text, _session.Tokenization.PromptText(prompt));

    /// <summary>The token count of each text, as <see cref="CountTokens(string)"/> returns it.</summary>
    public int[] CountTokens(IReadOnlyList<string> texts)
    {
        ArgumentNullException.ThrowIfNull(texts);
        var counts = new int[texts.Count];
        for (int i = 0; i < counts.Length; i++)
        {
            counts[i] = CountTokens(texts[i] ?? throw new ArgumentException($"Text {i} is null.", nameof(texts)));
        }

        return counts;
    }

    /// <summary><see cref="CountTokens(string)"/> against <see cref="MaxSequenceLength"/>; longer text is embedded truncated.</summary>
    public TokenLimitCheck Validate(string text) => new(CountTokens(text), MaxSequenceLength);

    /// <summary>Disposes the native session. Idempotent.</summary>
    public void Dispose() => _session.Dispose();
}
