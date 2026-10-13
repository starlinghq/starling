using Starlings.Models.E5;

namespace Starlings.TestSupport;

/// <summary>
/// Parity checks against the protected e5 embeddings, shared by the CPU and CUDA test projects. Inputs are the
/// fixture ids; nothing is tokenized. Bounds come from the profile's protected tolerances.yaml.
/// </summary>
internal static class E5ParityChecks
{
    public static Dictionary<string, float[]> SingleInputs(E5Encoder encoder, E5Tolerances tolerances, ITestOutputHelper output)
    {
        var stats = new ErrorStats();
        var singles = new Dictionary<string, float[]>(StringComparer.Ordinal);
        string worstId = string.Empty;
        double worst = -1;
        foreach (var c in E5Fixtures.Cases)
        {
            var embedding = encoder.Embed(new EncoderInput(c.InputIds), TestContext.Current.CancellationToken);
            singles[c.Id] = embedding;
            var caseStats = new ErrorStats();
            caseStats.Add(embedding, c.Embedding);
            stats.Add(embedding, c.Embedding);
            if (caseStats.Max > worst)
            {
                worst = caseStats.Max;
                worstId = c.Id;
            }
        }

        output.WriteLine($"single inputs vs protected ({E5Fixtures.Cases.Count} cases): {stats}; worst case {worstId}");
        output.WriteLine($"bound: embedding_max_abs {tolerances.EmbeddingMaxAbs:E1} from {tolerances.Source}");
        Assert.Equal(46, E5Fixtures.Cases.Count);
        Assert.False(stats.NonFinite, "NaN or Inf in embeddings");
        Assert.True(stats.Max <= tolerances.EmbeddingMaxAbs, $"embedding max abs {stats.Max:E3} > {tolerances.EmbeddingMaxAbs:E1}");
        return singles;
    }

    public static void PaddedBatchFixture(E5Encoder encoder, E5Tolerances tolerances, IReadOnlyDictionary<string, float[]> singles, ITestOutputHelper output)
    {
        var expected = E5Fixtures.Cases.ToDictionary(c => c.Id, StringComparer.Ordinal);
        var rows = E5Fixtures.BatchRows;
        Assert.Equal(encoder.Options.PadTokenId, E5Fixtures.BatchPadId);

        // As the tokenizer will hand them over: already right-padded with explicit masks.
        var padded = encoder.EmbedBatch(rows.Select(r => new EncoderInput(r.Ids, r.AttentionMask)).ToList(), TestContext.Current.CancellationToken);

        // Unpadded rows, padded by the encoder.
        var unpadded = encoder.EmbedBatch(rows.Select(r => new EncoderInput(expected[r.Id].InputIds)).ToList(), TestContext.Current.CancellationToken);

        var vsProtected = new ErrorStats();
        var vsSingle = new ErrorStats();
        var encoderPaddedVsSingle = new ErrorStats();
        for (int i = 0; i < rows.Count; i++)
        {
            // The fixture's own padding must agree with the encoder's padding rule.
            int length = expected[rows[i].Id].InputIds.Length;
            Assert.Equal(expected[rows[i].Id].InputIds, rows[i].Ids[..length]);
            Assert.All(rows[i].Ids[length..], id => Assert.Equal(E5Fixtures.BatchPadId, id));

            vsProtected.Add(padded[i], expected[rows[i].Id].Embedding);
            vsSingle.Add(padded[i], singles[rows[i].Id]);
            encoderPaddedVsSingle.Add(unpadded[i], singles[rows[i].Id]);
        }

        output.WriteLine($"{rows.Count}-row padded batch fixture vs protected: {vsProtected}");
        output.WriteLine($"{rows.Count}-row padded batch fixture vs own single rows: {vsSingle}");
        output.WriteLine($"{rows.Count} rows padded by the encoder vs own single rows: {encoderPaddedVsSingle}");
        Assert.False(vsProtected.NonFinite || encoderPaddedVsSingle.NonFinite);
        Assert.True(vsProtected.Max <= tolerances.EmbeddingMaxAbs, $"batch vs protected {vsProtected.Max:E3}");
        Assert.True(vsSingle.Max <= tolerances.PaddedBatchVsSingleMaxAbs, $"batch vs single {vsSingle.Max:E3}");
        Assert.True(encoderPaddedVsSingle.Max <= tolerances.PaddedBatchVsSingleMaxAbs, $"encoder-padded batch vs single {encoderPaddedVsSingle.Max:E3}");
    }

    public static void AllCasesInBatches(E5Encoder encoder, int batchSize, E5Tolerances tolerances, IReadOnlyDictionary<string, float[]> singles, ITestOutputHelper output)
    {
        var cases = E5Fixtures.Cases;
        var vsProtected = new ErrorStats();
        var vsSingle = new ErrorStats();
        for (int start = 0; start < cases.Count; start += batchSize)
        {
            var chunk = cases.Skip(start).Take(batchSize).ToList();
            var embeddings = encoder.EmbedBatch(chunk.Select(c => new EncoderInput(c.InputIds)).ToList(), TestContext.Current.CancellationToken);
            for (int i = 0; i < chunk.Count; i++)
            {
                vsProtected.Add(embeddings[i], chunk[i].Embedding);
                vsSingle.Add(embeddings[i], singles[chunk[i].Id]);
            }
        }

        output.WriteLine($"{cases.Count} cases in padded batches of {batchSize} vs protected: {vsProtected}");
        output.WriteLine($"{cases.Count} cases in padded batches of {batchSize} vs own single rows: {vsSingle}");
        Assert.False(vsProtected.NonFinite);
        Assert.True(vsProtected.Max <= tolerances.EmbeddingMaxAbs, $"batches vs protected {vsProtected.Max:E3}");
        Assert.True(vsSingle.Max <= tolerances.PaddedBatchVsSingleMaxAbs, $"batches vs single {vsSingle.Max:E3}");
    }
}
