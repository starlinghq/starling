using System.Text.Json;
using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>Named prompts and the text embedder's prompt handling. Runs in CI (weight-free graph, recording tokenizer).</summary>
public class PromptAndTextEmbedderTests(ITestOutputHelper output)
{
    /// <summary>Records every string it is asked to tokenize; ids are the UTF-16 code units plus 10, between bos and eos.</summary>
    private sealed class RecordingTokenizer : IEmbeddingGemma2Tokenizer
    {
        public List<string> Seen { get; } = [];

        public EmbeddingGemma2Input Tokenize(string text)
        {
            lock (Seen)
            {
                Seen.Add(text);
            }

            return new EmbeddingGemma2Input(new long[] { 2 }.Concat(text.Select(c => (long)c + 10)).Append(1).ToArray());
        }
    }

    [Fact]
    public void TheTwentyNamedPromptsAndTheDefaultNone()
    {
        Assert.Equal(20, EmbeddingGemma2Prompt.Named.Count);
        Assert.Equal("task: search result | query: ", EmbeddingGemma2Prompt.Query.Text);
        Assert.Equal("title: none | text: ", EmbeddingGemma2Prompt.Document.Text);
        Assert.Equal(string.Empty, EmbeddingGemma2Prompt.None.Text);
        Assert.Equal("none", EmbeddingGemma2Prompt.None.Name);
        Assert.DoesNotContain("none", EmbeddingGemma2Prompt.Named.Keys);
        Assert.Same(EmbeddingGemma2Prompt.Query, EmbeddingGemma2Prompt.FromName("query"));
        Assert.Same(EmbeddingGemma2Prompt.Document, EmbeddingGemma2Prompt.FromName("document"));
        Assert.Same(EmbeddingGemma2Prompt.None, EmbeddingGemma2Prompt.FromName("none"));
        Assert.Equal("task: code retrieval | query: ", EmbeddingGemma2Prompt.FromName("InstructionRetrieval").Text);
        Assert.Equal("title: none | text: ", EmbeddingGemma2Prompt.FromName("Retrieval-document").Text);
        Assert.Equal(EmbeddingGemma2Prompt.Document.Text, EmbeddingGemma2Prompt.FromName("Document").Text);
        Assert.Equal("Document", EmbeddingGemma2Prompt.FromName("Document").Name);
        Assert.Equal(EmbeddingGemma2Prompt.FromName("STS"), EmbeddingGemma2Prompt.FromName("STS"));
    }

    [Theory]
    [InlineData("Query")]
    [InlineData("QUERY")]
    [InlineData("None")]
    [InlineData("")]
    [InlineData("passage")]
    [InlineData("task: search result | query: ")]
    public void UnknownOrDifferentlyCasedNamesAreRejected(string name)
    {
        Assert.Throws<ArgumentException>(() => EmbeddingGemma2Prompt.FromName(name));
        Assert.False(EmbeddingGemma2Prompt.TryFromName(name, out _));
    }

    [Fact]
    public void PromptsMatchTheProtectedReference()
    {
        // The fixtures were computed with these prompt texts (reference.json, embeddings.jsonl prompt_text).
        using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(EmbeddingGemma2TestAssets.FixtureDirectory, "reference.json")));
        var prompts = doc.RootElement.GetProperty("prompts");
        Assert.Equal(prompts.GetProperty("query").GetString(), EmbeddingGemma2Prompt.Query.Text);
        Assert.Equal(prompts.GetProperty("document").GetString(), EmbeddingGemma2Prompt.Document.Text);
        Assert.Equal(prompts.GetProperty("none").GetString(), EmbeddingGemma2Prompt.None.Text);
        foreach (var row in EmbeddingGemma2Fixtures.LoadRows())
        {
            Assert.Equal(row.PromptText, row.PackagePrompt.Text);
        }
    }

    [Fact]
    public void PromptsEqualThePinnedConfigFileWhenAcquired()
    {
        var (files, reason) = EmbeddingGemma2TestAssets.FindModel();
        Assert.SkipWhen(files is null, reason);
        using var doc = JsonDocument.Parse(File.ReadAllText(files!.PromptConfigPath));
        var config = doc.RootElement.GetProperty("prompts").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetString()!, StringComparer.Ordinal);
        Assert.Equal(JsonValueKind.Null, doc.RootElement.GetProperty("default_prompt_name").ValueKind);
        Assert.Equal(config.OrderBy(kv => kv.Key, StringComparer.Ordinal), EmbeddingGemma2Prompt.Named.OrderBy(kv => kv.Key, StringComparer.Ordinal));
        output.WriteLine($"{config.Count} prompts in {files.PromptConfigPath} equal EmbeddingGemma2Prompt.Named; default_prompt_name null");
    }

    [Fact]
    public void TheTextEmbedderAddsNoPromptByDefault()
    {
        var tokenizer = new RecordingTokenizer();
        using var encoder = EncoderBehaviorTests.FakeEncoder();
        var embedder = new EmbeddingGemma2TextEmbedder(tokenizer, encoder);
        var ct = TestContext.Current.CancellationToken;
        embedder.Embed("hello", ct);
        embedder.EmbedBatch(["a", "b"], ct);
        Assert.Equal(["hello", "a", "b"], tokenizer.Seen);
    }

    [Fact]
    public void TheTextEmbedderPrependsTheChosenPromptExactlyOnce()
    {
        var tokenizer = new RecordingTokenizer();
        using var encoder = EncoderBehaviorTests.FakeEncoder();
        var embedder = new EmbeddingGemma2TextEmbedder(tokenizer, encoder);
        var ct = TestContext.Current.CancellationToken;
        var counts = new int[2];
        var q = embedder.Embed("hello", EmbeddingGemma2Prompt.Query, ct);
        embedder.EmbedBatch(["x", "task: search result | query: y"], EmbeddingGemma2Prompt.Query, counts, ct);
        embedder.Embed("doc", EmbeddingGemma2Prompt.FromName("Clustering"), ct);
        Assert.Equal(
            ["task: search result | query: hello", "task: search result | query: x", "task: search result | query: task: search result | query: y", "task: clustering | query: doc"],
            tokenizer.Seen);
        Assert.Equal("task: search result | query: x".Length + 2, counts[0]);
        Assert.Equal(EncoderBehaviorTests.Expected(tokenizer.Tokenize("task: search result | query: hello").InputIds.ToArray()).Length, q.Length);
    }

    [Fact]
    public void TheTextEmbedderValidatesArguments()
    {
        using var encoder = EncoderBehaviorTests.FakeEncoder();
        var embedder = new EmbeddingGemma2TextEmbedder(new RecordingTokenizer(), encoder);
        var ct = TestContext.Current.CancellationToken;
        Assert.Throws<ArgumentNullException>(() => embedder.Embed(null!, ct));
        Assert.Throws<ArgumentNullException>(() => embedder.Embed("x", null!, ct));
        Assert.Throws<ArgumentException>(() => embedder.EmbedBatch(["x", null!], ct));
        Assert.Throws<ArgumentException>(() => embedder.EmbedBatch(["x"], EmbeddingGemma2Prompt.None, new int[2], ct));
    }
}
