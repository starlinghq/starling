using Starlings.Runtime.Onnx;

namespace Starlings.Models.Bert;

/// <summary>A reusable owned-tokenizer plus ONNX text embedding session for an explicit BERT profile.</summary>
public sealed class BertTextEmbedder : IDisposable
{
    private readonly BertSession _model;
    private BertTextEmbedder(BertSession model) => _model = model;

    /// <summary>Profile and immutable source/runtime asset identities.</summary>
    public BertModelProfile Profile => _model.Profile;
    /// <summary>Embedding dimension.</summary>
    public int Dimension => (int)_model.Session.Outputs.Single(output => output.Name == "last_hidden_state").Dimensions[2];
    /// <summary>Loaded model and native runtime diagnostics.</summary>
    public OnnxSessionDiagnostics Diagnostics => _model.Session.Diagnostics;
    /// <summary>Advanced backend API for intermediate diagnostics. Owned by this embedder.</summary>
    public OnnxModelSession Session => _model.Session;

    /// <summary>Verifies local assets and creates a CPU session; never downloads.</summary>
    public static BertTextEmbedder Create(BertModelProfile profile, string snapshot, int threads = 2, int maxBatchSize = 32)
    {
        ArgumentNullException.ThrowIfNull(profile);
        if (profile.Task != BertTask.Embeddings)
        {
            throw new ArgumentException("Use BertReranker for paired-input scoring profiles.", nameof(profile));
        }

        return new BertTextEmbedder(new BertSession(profile, snapshot, threads, maxBatchSize));
    }

    /// <summary>Tokenizes an explicitly named prompt followed by text as one sequence.</summary>
    public BertInput Tokenize(string text, string prompt = "none")
    {
        ArgumentNullException.ThrowIfNull(text);
        if (!Profile.Prompts.TryGetValue(prompt, out string? prefix))
        {
            throw new ArgumentException($"Prompt '{prompt}' is not declared by {Profile.Id}.", nameof(prompt));
        }

        return _model.Tokenize(prefix + text);
    }

    /// <summary>Embeds one text with an explicit optional prompt; passages use none.</summary>
    public float[] Embed(string text, string prompt = "none", CancellationToken cancellationToken = default) =>
        EmbedTokens([Tokenize(text, prompt)], cancellationToken)[0];

    /// <summary>Embeds a batch in input order, right-padding bounded chunks.</summary>
    public float[][] EmbedBatch(IReadOnlyList<string> texts, string prompt = "none", CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(texts);
        cancellationToken.ThrowIfCancellationRequested();
        return EmbedTokens(texts.Select(text => Tokenize(text, prompt)).ToArray(), cancellationToken);
    }

    /// <summary>Embeds supplied tokenized inputs, retaining mask and segment ids.</summary>
    public float[][] EmbedTokens(IReadOnlyList<BertInput> inputs, CancellationToken cancellationToken = default) =>
        _model.Predict(inputs, cancellationToken);

    /// <summary>Disposes the owned native session. Idempotent.</summary>
    public void Dispose() => _model.Dispose();
}
