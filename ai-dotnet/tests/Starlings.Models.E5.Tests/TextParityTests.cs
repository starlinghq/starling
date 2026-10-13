using System.Text.Json;
using Starlings.TestSupport;

namespace Starlings.Models.E5.Tests;

/// <summary>
/// Text in, embedding out, through the owned tokenizer (<see cref="E5Tokenizer"/>) and the real model: ids and masks
/// must equal the protected <c>tokenizer.jsonl</c> exactly (G1), and embeddings must meet the binding tolerance against
/// <c>embeddings.jsonl</c> (G2). Skips when the snapshot (model and <c>tokenizer.json</c>) is not present.
/// </summary>
public class TextParityTests(E5CpuFixture fixture, ITestOutputHelper output) : IClassFixture<E5CpuFixture>
{
    private static readonly E5Tolerances Tolerances = E5Tolerances.Load(E5TestAssets.CpuFixtureDirectory);

    private static E5Tokenizer LoadTokenizer()
    {
        var (files, reason) = E5TestAssets.FindFiles();
        Assert.SkipWhen(files is null, reason);
        return E5Tokenizer.FromFile(files!.TokenizerPath, E5TestAssets.ReadManifestSha256("onnx-fp32-cpu", "tokenizer.json"));
    }

    /// <summary>Single-text cases of <c>tokenizer.jsonl</c>: id, text, and the 512-truncated ids and mask.</summary>
    private static IEnumerable<(string Id, string Text, long[] Ids, long[] Mask)> SingleTextCases()
    {
        foreach (var line in File.ReadLines(Path.Combine(E5TestAssets.CpuFixtureDirectory, "tokenizer.jsonl")))
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            using var doc = JsonDocument.Parse(line);
            var root = doc.RootElement;
            if (root.TryGetProperty("pair", out _))
            {
                continue;
            }

            var encoding = root.TryGetProperty("truncated", out var truncated)
                ? truncated.GetProperty("encoding")
                : root.GetProperty("encoding");
            yield return (
                root.GetProperty("id").GetString()!,
                root.GetProperty("text").GetString()!,
                encoding.GetProperty("ids").EnumerateArray().Select(e => e.GetInt64()).ToArray(),
                encoding.GetProperty("attention_mask").EnumerateArray().Select(e => e.GetInt64()).ToArray());
        }
    }

    [Fact]
    public void TokenizerIdsMatchTheProtectedFixtures()
    {
        var tokenizer = LoadTokenizer();
        int count = 0;
        foreach (var (id, text, ids, mask) in SingleTextCases())
        {
            var input = tokenizer.Tokenize(text);
            Assert.True(ids.AsSpan().SequenceEqual(input.InputIds.Span), $"{id}: ids differ");
            Assert.True(mask.AsSpan().SequenceEqual(input.AttentionMask.Span), $"{id}: attention mask differs");
            count++;
        }

        output.WriteLine($"{count} single-text cases: ids and masks exact");
        Assert.Equal(E5Fixtures.Cases.Count, count);
    }

    [Fact]
    public void TextEmbeddingsMeetTheBindingTolerance()
    {
        var tokenizer = LoadTokenizer();
        var embedder = new E5TextEmbedder(tokenizer, fixture.Encoder);
        var expected = E5Fixtures.Cases.ToDictionary(c => c.Id, c => c.Embedding, StringComparer.Ordinal);
        var cases = SingleTextCases().ToList();

        var single = new ErrorStats();
        foreach (var (id, text, _, _) in cases)
        {
            single.Add(embedder.Embed(text, TestContext.Current.CancellationToken), expected[id]);
        }

        var batched = new ErrorStats();
        var vectors = embedder.EmbedBatch(cases.Select(c => c.Text).ToList(), TestContext.Current.CancellationToken);
        for (int i = 0; i < cases.Count; i++)
        {
            batched.Add(vectors[i], expected[cases[i].Id]);
        }

        output.WriteLine($"text -> embedding, one per call: {single}");
        output.WriteLine($"text -> embedding, one batch call: {batched}");
        output.WriteLine($"bounds from {Tolerances.Source}");
        Assert.False(single.NonFinite || batched.NonFinite);
        Assert.InRange(single.Max, 0, Tolerances.EmbeddingMaxAbs);
        Assert.InRange(batched.Max, 0, Tolerances.PaddedBatchVsSingleMaxAbs);
    }

    [Fact]
    public void RejectsATokenizerFileWithTheWrongHash()
    {
        var (files, reason) = E5TestAssets.FindFiles();
        Assert.SkipWhen(files is null, reason);
        Assert.Throws<InvalidDataException>(() => E5Tokenizer.FromFile(files!.TokenizerPath, new string('0', 64)));
    }
}
