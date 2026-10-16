using System.Collections.Concurrent;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Starlings.Models.E5.MLNet.Tests;

/// <summary>
/// The transform's ML.NET contract on a weight-free graph with the e5 signature (hidden state
/// <c>[id, 0, 1, 0.5 * mask]</c>, dimension 4) and a counting stand-in tokenizer. Runs in CI; the real model is
/// covered by <see cref="RealModelTests"/>.
/// </summary>
public sealed class TransformContractTests : IDisposable
{
    private static readonly string[] Texts = ["a", "longer text", "", "zz", "mid", "the last one", "seven"];

    private readonly E5Encoder _encoder = new(
        OnnxModelSession.Create(TestModels.FakeE5, new OnnxSessionOptions { IntraOpNumThreads = 1, ComputeModelSha256 = false }),
        new E5EncoderOptions { MaxBatchSize = 2 },
        ownsSession: true);

    private readonly CountingTokenizer _tokenizer = new();
    private readonly MLContext _ml = new(seed: 1);
    private readonly ConcurrentQueue<int> _inferences = new();

    public TransformContractTests() => Embedder = new E5TextEmbedder(_tokenizer, _encoder);

    private E5TextEmbedder Embedder { get; }

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose() => _encoder.Dispose();

    /// <summary>normalize([mean of 0, code points, 2; 0; 1; 0.5]).</summary>
    private static float[] Expected(string text)
    {
        double sum = 2 + text.Sum(ch => (double)ch);
        double[] v = [sum / (text.Length + 2), 0, 1, 0.5];
        double norm = Math.Sqrt(v.Sum(x => x * x));
        return v.Select(x => (float)(x / norm)).ToArray();
    }

    /// <summary>A transformer whose inference calls (rows per call) are recorded in <see cref="_inferences"/>.</summary>
    private E5EmbeddingTransformer Observed(E5EmbeddingOptions? options = null, string output = "Embedding", string input = "Text")
    {
        options ??= new E5EmbeddingOptions { TokenCountColumnName = "TokenCount" };
        return new E5EmbeddingTransformer(Embedder, E5ColumnNames.Create(output, input, options.TokenCountColumnName), options, ownedEncoder: null, _inferences.Enqueue);
    }

    private IDataView Rows(params string?[] texts) =>
        _ml.Data.LoadFromEnumerable(texts.Select((t, i) => new TextRow { Text = t, Id = i }).ToList());

    [Fact]
    public void OutputSchemaAppendsAFixedSizeVectorAndTheTokenCount()
    {
        var input = Rows(Texts);
        using var transformer = Observed();
        var schema = transformer.GetOutputSchema(input.Schema);

        Assert.Equal(input.Schema.Count + 2, schema.Count);
        for (int i = 0; i < input.Schema.Count; i++)
        {
            Assert.Equal(input.Schema[i].Name, schema[i].Name);
            Assert.Equal(input.Schema[i].Type, schema[i].Type);
        }

        var embedding = schema["Embedding"];
        var vector = Assert.IsType<VectorDataViewType>(embedding.Type);
        Assert.Equal(NumberDataViewType.Single, vector.ItemType);
        Assert.True(vector.IsKnownSize);
        Assert.Equal(4, vector.Size);
        Assert.Equal(NumberDataViewType.Int32, schema["TokenCount"].Type);
        Assert.Equal(schema.Count, transformer.Transform(input).Schema.Count);

        // Without a token-count column only the embedding is added.
        using var plain = new E5EmbeddingTransformer(Embedder, "Embedding", "Text");
        Assert.Equal(input.Schema.Count + 1, plain.GetOutputSchema(input.Schema).Count);
        Assert.Null(plain.GetOutputSchema(input.Schema).GetColumnOrNull("TokenCount"));
        Assert.Empty(_inferences);
    }

    [Fact]
    public void EstimatorShapeMatchesTheTransformedSchema()
    {
        var input = Rows(Texts);
        var estimator = _ml.Transforms.ApplyE5Embedding(Embedder, "Embedding", "Text", new E5EmbeddingOptions { TokenCountColumnName = "TokenCount" });
        var shape = estimator.GetOutputSchema(ShapeOf(input.Schema));
        var embedding = shape.Single(c => c.Name == "Embedding");
        Assert.Equal(SchemaShape.Column.VectorKind.Vector, embedding.Kind);
        Assert.Equal(NumberDataViewType.Single, embedding.ItemType);
        Assert.False(embedding.IsKey);
        var tokens = shape.Single(c => c.Name == "TokenCount");
        Assert.Equal(SchemaShape.Column.VectorKind.Scalar, tokens.Kind);
        Assert.Equal(NumberDataViewType.Int32, tokens.ItemType);
        Assert.Equal(["Text", "Id", "Embedding", "TokenCount"], shape.Select(c => c.Name));

        // Fit trains nothing and reads no row.
        using var transformer = estimator.Fit(input);
        Assert.Equal(0, _tokenizer.Calls);
        Assert.Equal(shape.Select(c => c.Name), transformer.GetOutputSchema(input.Schema).Select(c => c.Name));
    }

    [Fact]
    public void OutputNamedLikeTheInputHidesTheTextColumn()
    {
        var input = Rows(Texts);
        using var transformer = new E5EmbeddingTransformer(Embedder, "Text");
        var output = transformer.Transform(input);
        Assert.True(output.Schema[0].IsHidden);
        Assert.IsType<VectorDataViewType>(output.Schema["Text"].Type);
        var embeddings = output.GetColumn<float[]>("Text").ToList();
        Assert.Equal(Texts.Select(Expected), embeddings);
    }

    public static TheoryData<string> WrongInputs => new() { "missing", "Id", "Vector" };

    [Theory]
    [MemberData(nameof(WrongInputs))]
    public void WrongInputColumnFailsWithASchemaError(string column)
    {
        var input = _ml.Transforms.Concatenate("Vector", "Text", "Text").Fit(Rows(Texts)).Transform(Rows(Texts));
        var estimator = _ml.Transforms.ApplyE5Embedding(Embedder, "Embedding", column);
        using var transformer = new E5EmbeddingTransformer(Embedder, "Embedding", column);

        var errors = new Exception[]
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => estimator.Fit(input)),
            Assert.Throws<ArgumentOutOfRangeException>(() => estimator.GetOutputSchema(ShapeOf(input.Schema))),
            Assert.Throws<ArgumentOutOfRangeException>(() => transformer.GetOutputSchema(input.Schema)),
            Assert.Throws<ArgumentOutOfRangeException>(() => transformer.Transform(input)),
            Assert.Throws<ArgumentOutOfRangeException>(() => transformer.GetRowToRowMapper(input.Schema)),
        };
        foreach (var error in errors)
        {
            Assert.Contains($"'{column}'", error.Message, StringComparison.Ordinal);
            Assert.Contains(column == "missing" ? "not found" : "Schema mismatch", error.Message, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void InvalidOptionsAndNamesAreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new E5EmbeddingTransformer(Embedder, "E", "Text", new E5EmbeddingOptions { BatchSize = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new E5EmbeddingTransformer(Embedder, "E", "Text", new E5EmbeddingOptions { Prefix = (E5TextPrefix)7 }));
        Assert.Throws<ArgumentException>(() => new E5EmbeddingTransformer(Embedder, "E", "Text", new E5EmbeddingOptions { TokenCountColumnName = "E" }));
        Assert.Throws<ArgumentException>(() => new E5EmbeddingTransformer(Embedder, " ", "Text"));
        Assert.Throws<ArgumentException>(() => new E5EmbeddingTransformer(Embedder, "E", "Text", new E5EmbeddingOptions { TokenCountColumnName = "" }));
    }

    [Fact]
    public void EachRowIsEmbeddedOnceHoweverManyGettersAreCalled()
    {
        var input = Rows(Texts);
        using var transformer = Observed();
        var output = transformer.Transform(input);
        var embeddingColumn = output.Schema["Embedding"];
        var tokenColumn = output.Schema["TokenCount"];

        using (var cursor = output.GetRowCursor(output.Schema))
        {
            var embedding1 = cursor.GetGetter<VBuffer<float>>(embeddingColumn);
            var embedding2 = cursor.GetGetter<VBuffer<float>>(embeddingColumn);
            var tokens = cursor.GetGetter<int>(tokenColumn);
            var text = cursor.GetGetter<ReadOnlyMemory<char>>(output.Schema["Text"]);
            VBuffer<float> a = default, b = default;
            int count = 0, row = 0;
            ReadOnlyMemory<char> t = default;
            while (cursor.MoveNext())
            {
                tokens(ref count);
                embedding1(ref a);
                embedding2(ref b);
                embedding1(ref a);
                tokens(ref count);
                text(ref t);
                Assert.Equal(Texts[row], t.ToString());
                Assert.Equal(Expected(Texts[row]), a.DenseValues().ToArray());
                Assert.Equal(a.DenseValues(), b.DenseValues());
                Assert.Equal(Texts[row].Length + 2, count);
                row++;
                Assert.Equal(row, _tokenizer.Calls);
            }
        }

        // One inference of one row per row: 5 getter calls on each of 7 rows.
        Assert.Equal(Enumerable.Repeat(1, Texts.Length), _inferences);
        Assert.Equal(Texts, _tokenizer.Seen);
    }

    [Fact]
    public void RowsWithoutOutputGettersAreNotEmbeddedAndPassThroughCursorsDoNotReadText()
    {
        var input = new TestRowsView(Texts);
        using var transformer = Observed();
        var output = transformer.Transform(input);

        // Only a pass-through column: neither the text nor the model is touched.
        using (var cursor = output.GetRowCursor([output.Schema["Index"]]))
        {
            var index = cursor.GetGetter<int>(output.Schema["Index"]);
            Assert.False(cursor.IsColumnActive(output.Schema["Embedding"]));
            Assert.Throws<InvalidOperationException>(() => cursor.GetGetter<VBuffer<float>>(output.Schema["Embedding"]));
            int i = 0, n = 0;
            while (cursor.MoveNext())
            {
                index(ref i);
                n++;
            }

            Assert.Equal(Texts.Length, n);
        }

        Assert.Equal(0, input.TextReads);
        Assert.Empty(_inferences);

        // The embedding is active, but its getter is called on every second row only.
        using (var cursor = output.GetRowCursor([output.Schema["Embedding"]]))
        {
            var embedding = cursor.GetGetter<VBuffer<float>>(output.Schema["Embedding"]);
            VBuffer<float> v = default;
            for (int row = 0; cursor.MoveNext(); row++)
            {
                if (row % 2 == 0)
                {
                    embedding(ref v);
                    Assert.Equal(Expected(Texts[row]), v.DenseValues().ToArray());
                }
            }
        }

        Assert.Equal((Texts.Length + 1) / 2, _inferences.Count);
    }

    [Fact]
    public void GetterTypeMismatchFails()
    {
        using var transformer = Observed();
        var output = transformer.Transform(Rows(Texts));
        using var cursor = output.GetRowCursor(output.Schema);
        Assert.Throws<InvalidOperationException>(() => cursor.GetGetter<float[]>(output.Schema["Embedding"]));
        Assert.Throws<InvalidOperationException>(() => cursor.GetGetter<long>(output.Schema["TokenCount"]));
        Assert.Throws<ArgumentOutOfRangeException>(() => cursor.GetGetter<int>(MakeForeignColumn()));
    }

    /// <summary>SchemaShape.Create is internal in ML.NET; tests reach it by reflection (the library does not).</summary>
    private static SchemaShape ShapeOf(DataViewSchema schema) =>
        (SchemaShape)typeof(SchemaShape).GetMethod("Create", System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public, [typeof(DataViewSchema)])!
            .Invoke(null, [schema])!;

    private static DataViewSchema.Column MakeForeignColumn()
    {
        var builder = new DataViewSchema.Builder();
        for (int i = 0; i < 10; i++)
        {
            builder.AddColumn($"c{i}", NumberDataViewType.Int32);
        }

        return builder.ToSchema()[9];
    }

    [Fact]
    public void NullAndEmptyTextAreEmbeddedAsTheEmptyString()
    {
        using var transformer = Observed();
        var rows = _ml.Data.CreateEnumerable<EmbeddedRow>(transformer.Transform(Rows(null, "", "x")), reuseRowObject: false).ToList();
        float[] empty = Embedder.Embed(string.Empty, Ct);
        Assert.Equal(empty, rows[0].Embedding);
        Assert.Equal(empty, rows[1].Embedding);
        Assert.Equal(2, rows[0].TokenCount);
        Assert.Equal(2, rows[1].TokenCount);
        Assert.Equal(Expected("x"), rows[2].Embedding);
    }

    [Fact]
    public void ValuesEqualTheTextEmbedderExactlyAndRoundTripThroughCreateEnumerable()
    {
        using var transformer = Observed();
        var rows = _ml.Data.CreateEnumerable<EmbeddedRow>(transformer.Transform(Rows(Texts)), reuseRowObject: false).ToList();
        Assert.Equal(Texts.Length, rows.Count);
        for (int i = 0; i < Texts.Length; i++)
        {
            Assert.Equal(Texts[i], rows[i].Text);
            Assert.Equal(i, rows[i].Id);
            Assert.Equal(Embedder.Embed(Texts[i], Ct), rows[i].Embedding);
            Assert.Equal(Texts[i].Length + 2, rows[i].TokenCount);
        }

        // Back into a view and through the transform's output schema again.
        var back = _ml.Data.LoadFromEnumerable(rows);
        Assert.Equal(rows.Select(r => r.Embedding), back.GetColumn<float[]>("Embedding"));
    }

    [Fact]
    public void PrefixIsExplicitAndCountedInTokens()
    {
        foreach (var (prefix, text) in new[] { (E5TextPrefix.None, ""), (E5TextPrefix.Query, "query: "), (E5TextPrefix.Passage, "passage: ") })
        {
            using var transformer = Observed(new E5EmbeddingOptions { Prefix = prefix, TokenCountColumnName = "TokenCount" });
            var rows = _ml.Data.CreateEnumerable<EmbeddedRow>(transformer.Transform(Rows("query: x", "y")), reuseRowObject: false).ToList();
            Assert.Equal(Expected(text + "query: x"), rows[0].Embedding);
            Assert.Equal(Expected(text + "y"), rows[1].Embedding);
            Assert.Equal(text.Length + 3, rows[1].TokenCount);
        }

        Assert.Equal(E5TextEmbedder.QueryPrefix, "query: ");
        Assert.Equal(E5TextEmbedder.PassagePrefix, "passage: ");
    }

    [Fact]
    public void PipelineFitTransformAndPreview()
    {
        var input = Rows(Texts);
        var pipeline = _ml.Transforms.CopyColumns("Copy", "Text")
            .Append(_ml.Transforms.ApplyE5Embedding(Embedder, "Embedding", "Copy", new E5EmbeddingOptions { TokenCountColumnName = "TokenCount" }));
        using var model = pipeline.Fit(input);
        var output = model.Transform(input);

        var preview = output.Preview(maxRows: 3);
        Assert.Equal(3, preview.RowView.Length);
        var first = preview.RowView[1].Values.ToDictionary(kv => kv.Key, kv => kv.Value);
        Assert.Equal(Expected(Texts[1]), ((VBuffer<float>)first["Embedding"]).DenseValues().ToArray());
        Assert.Equal(Texts[1].Length + 2, (int)first["TokenCount"]);
        Assert.Contains(preview.ColumnView, c => c.Column.Name == "Embedding" && c.Values.Length == 3);

        // Preview of the estimator itself fits on a sample and transforms it.
        var estimatorPreview = _ml.Transforms.ApplyE5Embedding(Embedder, "Embedding", "Text").Preview(input, maxRows: 2);
        Assert.Equal(2, estimatorPreview.RowView.Length);
    }

    [Fact]
    public async Task ParallelCursorSetEmbedsEachRowOnceConcurrently()
    {
        var texts = Enumerable.Range(0, 64).Select(i => new string((char)('a' + (i % 26)), 1 + (i % 9))).ToArray();
        var input = new TestRowsView(texts);
        var barrier = new Barrier(4);
        var threads = new ConcurrentDictionary<int, byte>();
        using var transformer = Observed();
        var output = transformer.Transform(input);
        var cursors = output.GetRowCursorSet([output.Schema["Embedding"], output.Schema["Index"]], 4);
        Assert.Equal(4, cursors.Length);

        var results = new ConcurrentDictionary<int, float[]>();
        var tasks = cursors.Select(cursor => Task.Factory.StartNew(() =>
        {
            using (cursor)
            {
                barrier.SignalAndWait(Ct);
                var embedding = cursor.GetGetter<VBuffer<float>>(output.Schema["Embedding"]);
                var index = cursor.GetGetter<int>(output.Schema["Index"]);
                var id = cursor.GetIdGetter();
                VBuffer<float> v = default;
                DataViewRowId rowId = default;
                int i = 0;
                bool first = true;
                while (cursor.MoveNext())
                {
                    threads[Environment.CurrentManagedThreadId] = 0;
                    if (first)
                    {
                        // Every cursor holds a row before any takes a second, so the four embed concurrently on any
                        // scheduler: without this rendezvous one thread can drain the shared input before the others
                        // are scheduled (seen on Windows runners), and the threads assertion measures the scheduler.
                        first = false;
                        barrier.SignalAndWait(Ct);
                    }

                    embedding(ref v);
                    embedding(ref v);
                    index(ref i);
                    id(ref rowId);
                    Assert.Equal((ulong)i, rowId.Low);
                    Assert.True(results.TryAdd(i, v.DenseValues().ToArray()), $"row {i} seen twice");
                }

                if (first)
                {
                    barrier.RemoveParticipant();
                }
            }
        }, Ct, TaskCreationOptions.LongRunning, TaskScheduler.Default)).ToArray();
        await Task.WhenAll(tasks);

        Assert.Equal(texts.Length, results.Count);
        for (int i = 0; i < texts.Length; i++)
        {
            Assert.Equal(Expected(texts[i]), results[i]);
        }

        Assert.Equal(texts.Length, _inferences.Count);
        Assert.Equal(texts.Length, _tokenizer.Calls);
        Assert.Equal(4, threads.Count);
    }

    [Fact]
    public void LookAheadBatchingEmbedsInBatchesAndEqualsTheBatchedEmbedder()
    {
        var input = new TestRowsView(Texts);
        using var transformer = Observed(new E5EmbeddingOptions { BatchSize = 3, TokenCountColumnName = "TokenCount" });
        var output = transformer.Transform(input);
        using (var cursor = output.GetRowCursor(output.Schema))
        {
            var embedding = cursor.GetGetter<VBuffer<float>>(output.Schema["Embedding"]);
            var tokens = cursor.GetGetter<int>(output.Schema["TokenCount"]);
            var index = cursor.GetGetter<int>(output.Schema["Index"]);
            VBuffer<float> v = default;
            int count = 0, i = 0;
            for (int row = 0; cursor.MoveNext(); row++)
            {
                embedding(ref v);
                tokens(ref count);
                embedding(ref v);
                index(ref i);
                Assert.Equal(row, i);
                var chunk = Texts.Skip(row / 3 * 3).Take(3).ToList();
                Assert.Equal(Embedder.EmbedBatch(chunk, Ct)[row % 3], v.DenseValues().ToArray());
                Assert.Equal(Texts[row].Length + 2, count);
            }
        }

        // Three inference calls of 3, 3 and 1 rows; the fake graph is exact, so values also equal row-wise ones.
        Assert.Equal([3, 3, 1], _inferences);

        // A cursor set with batching is one batched cursor; a shuffled cursor is row-wise.
        Assert.Single(output.GetRowCursorSet(output.Schema, 4));
        _inferences.Clear();
        int shuffledRows = 0;
        using (var cursor = output.GetRowCursor(output.Schema, new Random(3)))
        {
            var embedding = cursor.GetGetter<VBuffer<float>>(output.Schema["Embedding"]);
            var index = cursor.GetGetter<int>(output.Schema["Index"]);
            VBuffer<float> v = default;
            int i = 0;
            while (cursor.MoveNext())
            {
                embedding(ref v);
                index(ref i);
                Assert.Equal(Expected(Texts[i]), v.DenseValues().ToArray());
                shuffledRows++;
            }
        }

        Assert.Equal(Texts.Length, shuffledRows);
        Assert.Equal(Enumerable.Repeat(1, Texts.Length), _inferences);
    }

    [Fact]
    public void LookAheadBatchingSkipsRowsWithoutGettersAndStillAligns()
    {
        using var transformer = Observed(new E5EmbeddingOptions { BatchSize = 2 });
        var output = transformer.Transform(new TestRowsView(Texts));
        using var cursor = output.GetRowCursor(output.Schema);
        var embedding = cursor.GetGetter<VBuffer<float>>(output.Schema["Embedding"]);
        VBuffer<float> v = default;
        var read = new List<int>();
        for (int row = 0; cursor.MoveNext(); row++)
        {
            // Rows 0, 3, 4 and 6 only: batches start at 0, 3 and 6.
            if (row is 0 or 3 or 4 or 6)
            {
                embedding(ref v);
                Assert.Equal(Expected(Texts[row]), v.DenseValues().ToArray());
                read.Add(row);
            }
        }

        Assert.Equal([2, 2, 1], _inferences);
    }

    [Fact]
    public void LookAheadBatchingFailsWhenTwoCursorsDisagreeOnRowOrder()
    {
        using var transformer = Observed(new E5EmbeddingOptions { BatchSize = 3 });
        var output = transformer.Transform(new TestRowsView(Texts) { ReverseEverySecondCursor = true });
        using var cursor = output.GetRowCursor(output.Schema);
        var embedding = cursor.GetGetter<VBuffer<float>>(output.Schema["Embedding"]);
        VBuffer<float> v = default;
        Assert.True(cursor.MoveNext());
        var error = Assert.Throws<InvalidOperationException>(() => embedding(ref v));
        Assert.Contains("BatchSize = 1", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void PredictionEngineIsRowWiseAndNeverReturnsAStaleRow()
    {
        using var transformer = Observed();
        var engine = _ml.Model.CreatePredictionEngine<TextRow, EmbeddedRow>(transformer, inputSchema: Rows("x").Schema);
        var a = engine.Predict(new TextRow { Text = "abc" });
        var b = engine.Predict(new TextRow { Text = "zz" });
        var c = engine.Predict(new TextRow { Text = "zz" });
        Assert.Equal(Expected("abc"), a.Embedding);
        Assert.Equal(5, a.TokenCount);
        Assert.Equal(Expected("zz"), b.Embedding);
        Assert.Equal(Expected("zz"), c.Embedding);
        Assert.InRange(_inferences.Count, 2, 3);
        Assert.All(_inferences, n => Assert.Equal(1, n));
    }

    [Fact]
    public void SaveIsUnsupportedAlsoInsideAChain()
    {
        var input = Rows(Texts);
        using var transformer = Observed();
        using var stream = new MemoryStream();
        var direct = Assert.ThrowsAny<Exception>(() => _ml.Model.Save(transformer, input.Schema, stream));
        AssertUnsupported(direct);

        using var chain = _ml.Transforms.CopyColumns("Copy", "Text")
            .Append(_ml.Transforms.ApplyE5Embedding(Embedder, "Embedding", "Copy"))
            .Fit(input);
        AssertUnsupported(Assert.ThrowsAny<Exception>(() => _ml.Model.Save(chain, input.Schema, new MemoryStream())));
        Assert.Throws<NotSupportedException>(() => ((ICanSaveModel)transformer).Save(null!));
    }

    private static void AssertUnsupported(Exception error)
    {
        var inner = error;
        while (inner is not NotSupportedException && inner.InnerException is not null)
        {
            inner = inner.InnerException;
        }

        var unsupported = Assert.IsType<NotSupportedException>(inner);
        Assert.Contains("mlnet.persistence: unsupported", unsupported.Message, StringComparison.Ordinal);
        Assert.Contains("E5ModelAssets", unsupported.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void WrappingTransformerOwnsNothingAndDisposeIsIdempotent()
    {
        var transformer = new E5EmbeddingTransformer(Embedder, "Embedding", "Text");
        var output = transformer.Transform(Rows(Texts));
        transformer.Dispose();
        transformer.Dispose();
        Assert.Throws<ObjectDisposedException>(() => transformer.Transform(Rows(Texts)));
        Assert.Throws<ObjectDisposedException>(() => transformer.GetRowToRowMapper(Rows(Texts).Schema));

        // The embedder's encoder is not owned: an existing view and the embedder still work.
        Assert.Equal(Texts.Select(Expected), output.GetColumn<float[]>("Embedding"));
        Assert.Equal(Expected("a"), Embedder.Embed("a", Ct));
        Assert.Same(Embedder, transformer.Embedder);
        Assert.Equal(4, transformer.Dimension);
        Assert.Equal("Text", transformer.InputColumnName);
        Assert.Equal("Embedding", transformer.OutputColumnName);
        Assert.True(transformer.IsRowToRowMapper);
    }

    [Fact]
    public void CapabilityDescriptorReportsTheMlNetStatuses()
    {
        Assert.Equal(CapabilityStatus.Supported, E5Capabilities.All["mlnet.embedding_transform"]);
        Assert.Equal(CapabilityStatus.Supported, E5Capabilities.All["mlnet.batching"]);
        Assert.Equal(CapabilityStatus.Unsupported, E5Capabilities.All["mlnet.persistence"]);
        Assert.Equal(CapabilityStatus.Unsupported, E5Capabilities.All["mlnet.cancellation"]);
        Assert.Equal(CapabilityStatus.Unsupported, E5Capabilities.All["mlnet.training"]);
        Assert.Equal(CapabilityStatus.Supported, E5Capabilities.All["mlnet.embedding_transform.cuda"]); // Stable CUDA profile (RTX 3090)
    }
}
