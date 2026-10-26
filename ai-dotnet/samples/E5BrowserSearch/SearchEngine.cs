using System.Diagnostics;
using Microsoft.Extensions.AI;
using Starlings.Models.E5.Browser;
using Starlings.Models.E5.Extensions.AI;
using Starlings.Runtime.OnnxWeb;
using Starlings.Runtime.OnnxWeb.WebGpu;

namespace E5BrowserSearch;

/// <summary>One passage of the corpus with its embedding.</summary>
public sealed record Passage(string Language, string Text, float[] Vector);

/// <summary>One search hit.</summary>
public sealed record Hit(Passage Passage, float Score);

/// <summary>
/// Loads e5 in the browser once, embeds the corpus with the <c>passage: </c> prefix, and ranks passages for a query
/// embedded with <c>query: </c>, by cosine (the vectors are L2-normalized, so the dot product).
/// </summary>
public sealed class SearchEngine : IDisposable
{
    private IEmbeddingGenerator<string, Embedding<float>>? _generator;
    private E5BrowserEmbedder? _embedder;
    private List<Passage> _passages = [];

    public bool IsLoaded => _generator is not null;

    public string? Profile => _embedder?.Profile;

    public OnnxWebRuntimeInfo? Runtime => OnnxWebRuntime.Info;

    public TimeSpan CorpusTime { get; private set; }

    public IReadOnlyList<Passage> Passages => _passages;

    /// <summary>
    /// Loads the model and embeds the corpus. The first load downloads the graph and tokenizer (verified by SHA-256)
    /// into Cache Storage; later page loads read them from there.
    /// </summary>
    public async Task LoadAsync(E5BrowserVariant variant, string? modelUrl, OnnxWebExecutionProvider provider, IProgress<E5BrowserLoadProgress> progress)
    {
        if (IsLoaded)
        {
            throw new InvalidOperationException("Already loaded; reload the page to switch models.");
        }

        var embedder = await E5BrowserEmbedder.LoadAsync(
            new E5BrowserLoadOptions
            {
                Variant = variant,
                ModelUrl = modelUrl,
                Provider = provider,
                // A page initializes onnxruntime-web once, so the bundle is chosen on the first load only. The WASM-only
                // bundle is the smaller download (2.3 MB brotli); the WebGPU adapter's (about 4 MB) is needed only for
                // WebGPU, and it also runs WASM, so a failed WebGPU attempt can be retried with WASM without a reload.
                Runtime = OnnxWebRuntime.Info is not null
                    ? null
                    : new OnnxWebRuntimeOptions
                    {
                        Bundle = provider == OnnxWebExecutionProvider.WebGpu ? OnnxWebGpuBundle.WebGpu : OnnxWebBundle.Wasm,
                    },
            },
            progress);
        var generator = new E5BrowserEmbeddingGenerator(embedder, ownsEmbedder: true);
        try
        {
            var stopwatch = Stopwatch.StartNew();
            var corpus = Corpus.Passages;
            var embeddings = await generator.GenerateAsync(corpus.Select(p => p.Text), PassageOptions);
            _passages = [.. corpus.Select((p, i) => new Passage(p.Language, p.Text, embeddings[i].Vector.ToArray()))];
            CorpusTime = stopwatch.Elapsed;
        }
        catch
        {
            // Not loaded: the caller can try again (for example with the other backend).
            generator.Dispose();
            throw;
        }

        _embedder = embedder;
        _generator = generator;
    }

    /// <summary>The <paramref name="top"/> passages closest to <paramref name="query"/>, and the time the query embedding took.</summary>
    public async Task<(IReadOnlyList<Hit> Hits, TimeSpan Elapsed)> SearchAsync(string query, int top = 5)
    {
        var generator = _generator ?? throw new InvalidOperationException("Load the model first.");
        var stopwatch = Stopwatch.StartNew();
        var embedding = await generator.GenerateAsync([query], QueryOptions);
        stopwatch.Stop();
        float[] q = embedding[0].Vector.ToArray();
        var hits = _passages
            .Select(p => new Hit(p, Dot(q, p.Vector)))
            .OrderByDescending(h => h.Score)
            .Take(top)
            .ToList();
        return (hits, stopwatch.Elapsed);
    }

    private static float Dot(float[] a, float[] b)
    {
        float sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }

        return sum;
    }

    public void Dispose() => _generator?.Dispose();

    private static EmbeddingGenerationOptions QueryOptions { get; } = InputType(E5InputType.Query);

    private static EmbeddingGenerationOptions PassageOptions { get; } = InputType(E5InputType.Passage);

    private static EmbeddingGenerationOptions InputType(E5InputType type) =>
        new() { AdditionalProperties = new() { [E5EmbeddingGenerator.InputTypePropertyName] = type } };
}
