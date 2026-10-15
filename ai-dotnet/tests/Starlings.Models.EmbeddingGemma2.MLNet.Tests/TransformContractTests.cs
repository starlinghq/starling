using System.Collections.Concurrent;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Starlings.Models.EmbeddingGemma2.MLNet.Tests;

/// <summary>
/// The transform's ML.NET contract on a weight-free graph with the EmbeddingGemma 2 signature (token embeddings
/// <c>id * (1 + d / 768) + mask * cos(d)</c>, dimension 768) and a counting stand-in tokenizer. Runs in CI; the real model is
/// covered by <see cref="RealModelTests"/>.
/// </summary>
public sealed class TransformContractTests : IDisposable
{
    private static readonly string[] Texts = ["a", "longer text", "", "zz", "mid", "the last one", "seven"];

    private readonly EmbeddingGemma2Encoder _encoder = new(
        OnnxModelSession.Create(TestModels.FakeEmbeddingGemma2, new OnnxSessionOptions { IntraOpNumThreads = 1, ComputeModelSha256 = false }),
        new EmbeddingGemma2EncoderOptions { MaxBatchSize = 2 },
        ownsSession: true);

    private readonly CountingTokenizer _tokenizer = new();
    private readonly MLContext _ml = new(seed: 1);
    private readonly ConcurrentQueue<int> _inferences = new();

    public TransformContractTests()
    {
        Embedder = new EmbeddingGemma2TextEmbedder(_tokenizer, _encoder);
        _reference = new EmbeddingGemma2TextEmbedder(new CountingTokenizer(), _encoder);
    }

    private readonly EmbeddingGemma2TextEmbedder _reference;

    private EmbeddingGemma2TextEmbedder Embedder { get; }

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose() => _encoder.Dispose();

    /// <summary>
    /// The row-wise embedding of <paramref name="text"/> through a separate, uncounted tokenizer on the same encoder.
    /// The fake graph pools each row with the same operations in a batch or alone, so values compare exactly;
    /// <see cref="ExpectedMatchesTheClosedForm"/> checks this reference against the closed form.
    /// </summary>
    private float[] Expected(string text) => _reference.Embed(text, Ct);

    [Fact]
    public void ExpectedMatchesTheClosedForm()
    {
        foreach (string text in Texts)
        {
            var ids = CountingTokenizer.Ids(text);
            var sum = new double[768];
            foreach (long id in ids)
            {
                for (int d = 0; d < 768; d++)
                {
                    sum[d] += (float)((float)id * (float)(1.0 + (d / 768.0))) + (float)Math.Cos(d);
                }
            }

            double norm = Math.Sqrt(sum.Sum(x => (x / ids.Length) * (x / ids.Length)));
            var actual = Expected(text);
            for (int d = 0; d < 768; d++)
            {
                Assert.True(Math.Abs((sum[d] / ids.Length / norm) - actual[d]) <= 2e-6);
            }
        }
    }

    /// <summary>A transformer whose inference calls (rows per call) are recorded in <see cref="_inferences"/>.</summary>
    private EmbeddingGemma2EmbeddingTransformer Observed(EmbeddingGemma2EmbeddingOptions? options = null, string output = "Embedding", string input = "Text")
    {
        options ??= new EmbeddingGemma2EmbeddingOptions { TokenCountColumnName = "TokenCount" };
        return new EmbeddingGemma2EmbeddingTransformer(Embedder, EmbeddingGemma2ColumnNames.Create(output, input, options.TokenCountColumnName), options, ownedEncoder: null, _inferences.Enqueue);
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
        Assert.Equal(768, vector.Size);
        Assert.Equal(NumberDataViewType.Int32, schema["TokenCount"].Type);
        Assert.Equal(schema.Count, transformer.Transform(input).Schema.Count);

        // Without a token-count column only the embedding is added.
        using var plain = new EmbeddingGemma2EmbeddingTransformer(Embedder, "Embedding", "Text");
        Assert.Equal(input.Schema.Count + 1, plain.GetOutputSchema(input.Schema).Count);
        Assert.Null(plain.GetOutputSchema(input.Schema).GetColumnOrNull("TokenCount"));
        Assert.Empty(_inferences);
    }

    [Fact]
    public void EstimatorShapeMatchesTheTransformedSchema()
    {
        var input = Rows(Texts);
        var estimator = _ml.Transforms.ApplyEmbeddingGemma2Embedding(Embedder, "Embedding", "Text", new EmbeddingGemma2EmbeddingOptions { TokenCountColumnName = "TokenCount" });
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
        using var transformer = new EmbeddingGemma2EmbeddingTransformer(Embedder, "Text");
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
        var estimator = _ml.Transforms.ApplyEmbeddingGemma2Embedding(Embedder, "Embedding", column);
        using var transformer = new EmbeddingGemma2EmbeddingTransformer(Embedder, "Embedding", column);

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
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingGemma2EmbeddingTransformer(Embedder, "E", "Text", new EmbeddingGemma2EmbeddingOptions { BatchSize = 0 }));
        Assert.Throws<ArgumentNullException>(() => new EmbeddingGemma2EmbeddingTransformer(Embedder, "E", "Text", new EmbeddingGemma2EmbeddingOptions { Prompt = null! }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new EmbeddingGemma2EmbeddingTransformer(Embedder, "E", "Text", new EmbeddingGemma2EmbeddingOptions { Encoder = new() { Dimension = 300 } }));
        Assert.Throws<ArgumentException>(() => new EmbeddingGemma2EmbeddingTransformer(Embedder, "E", "Text", new EmbeddingGemma2EmbeddingOptions { TokenCountColumnName = "E" }));
        Assert.Throws<ArgumentException>(() => new EmbeddingGemma2EmbeddingTransformer(Embedder, " ", "Text"));
        Assert.Throws<ArgumentException>(() => new EmbeddingGemma2EmbeddingTransformer(Embedder, "E", "Text", new EmbeddingGemma2EmbeddingOptions { TokenCountColumnName = "" }));
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
    public void PromptIsExplicitAndCountedInTokens()
    {
        foreach (var prompt in new[] { EmbeddingGemma2Prompt.None, EmbeddingGemma2Prompt.Query, EmbeddingGemma2Prompt.Document, EmbeddingGemma2Prompt.FromName("STS") })
        {
            using var transformer = Observed(new EmbeddingGemma2EmbeddingOptions { Prompt = prompt, TokenCountColumnName = "TokenCount" });
            var rows = _ml.Data.CreateEnumerable<EmbeddedRow>(transformer.Transform(Rows("task: search result | query: x", "y")), reuseRowObject: false).ToList();
            Assert.Equal(Expected(prompt.Text + "task: search result | query: x"), rows[0].Embedding);
            Assert.Equal(Expected(prompt.Text + "y"), rows[1].Embedding);
            Assert.Equal(prompt.Text.Length + 3, rows[1].TokenCount);
        }

        Assert.Equal(EmbeddingGemma2Prompt.None, new EmbeddingGemma2EmbeddingOptions().Prompt);
    }

    [Fact]
    public void TheEmbeddingColumnTakesTheEncodersMatryoshkaSize()
    {
        using var encoder256 = new EmbeddingGemma2Encoder(_encoder.Session, new EmbeddingGemma2EncoderOptions { Dimension = 256 });
        var embedder = new EmbeddingGemma2TextEmbedder(_tokenizer, encoder256);
        var estimator = _ml.Transforms.ApplyEmbeddingGemma2Embedding(embedder, "Embedding", "Text");
        Assert.Equal(256, estimator.Dimension);
        using var transformer = estimator.Fit(Rows(Texts));
        var output = transformer.Transform(Rows(Texts));
        Assert.Equal(256, ((VectorDataViewType)output.Schema["Embedding"].Type).Size);
        var values = output.GetColumn<float[]>("Embedding").ToList();
        for (int i = 0; i < Texts.Length; i++)
        {
            Assert.Equal(EmbeddingGemma2Encoder.Truncate(Expected(Texts[i]), 256).Length, values[i].Length);
            Assert.True(EmbeddingGemma2Encoder.Truncate(Expected(Texts[i]), 256).Zip(values[i]).All(p => Math.Abs(p.First - p.Second) <= 1e-6));
        }
    }

    [Fact]
    public void ATooLongRowFailsFromTheGetterWithTheTypedException()
    {
        using var transformer = Observed();
        var output = transformer.Transform(Rows("ok", new string('a', 8191)));
        using var cursor = output.GetRowCursor(output.Schema);
        var embedding = cursor.GetGetter<VBuffer<float>>(output.Schema["Embedding"]);
        VBuffer<float> v = default;
        Assert.True(cursor.MoveNext());
        embedding(ref v);
        Assert.True(cursor.MoveNext());
        Assert.Throws<EmbeddingGemma2InputTooLongException>(() => embedding(ref v));
    }

    [Fact]
    public void PipelineFitTransformAndPreview()
    {
        var input = Rows(Texts);
        var pipeline = _ml.Transforms.CopyColumns("Copy", "Text")
            .Append(_ml.Transforms.ApplyEmbeddingGemma2Embedding(Embedder, "Embedding", "Copy", new EmbeddingGemma2EmbeddingOptions { TokenCountColumnName = "TokenCount" }));
        using var model = pipeline.Fit(input);
        var output = model.Transform(input);

        var preview = output.Preview(maxRows: 3);
        Assert.Equal(3, preview.RowView.Length);
        var first = preview.RowView[1].Values.ToDictionary(kv => kv.Key, kv => kv.Value);
        Assert.Equal(Expected(Texts[1]), ((VBuffer<float>)first["Embedding"]).DenseValues().ToArray());
        Assert.Equal(Texts[1].Length + 2, (int)first["TokenCount"]);
        Assert.Contains(preview.ColumnView, c => c.Column.Name == "Embedding" && c.Values.Length == 3);

        // Preview of the estimator itself fits on a sample and transforms it.
        var estimatorPreview = _ml.Transforms.ApplyEmbeddingGemma2Embedding(Embedder, "Embedding", "Text").Preview(input, maxRows: 2);
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
        using var transformer = Observed(new EmbeddingGemma2EmbeddingOptions { BatchSize = 3, TokenCountColumnName = "TokenCount" });
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
        using var transformer = Observed(new EmbeddingGemma2EmbeddingOptions { BatchSize = 2 });
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
        using var transformer = Observed(new EmbeddingGemma2EmbeddingOptions { BatchSize = 3 });
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
            .Append(_ml.Transforms.ApplyEmbeddingGemma2Embedding(Embedder, "Embedding", "Copy"))
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
        Assert.Contains("EmbeddingGemma2ModelAssets", unsupported.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void WrappingTransformerOwnsNothingAndDisposeIsIdempotent()
    {
        var transformer = new EmbeddingGemma2EmbeddingTransformer(Embedder, "Embedding", "Text");
        var output = transformer.Transform(Rows(Texts));
        transformer.Dispose();
        transformer.Dispose();
        Assert.Throws<ObjectDisposedException>(() => transformer.Transform(Rows(Texts)));
        Assert.Throws<ObjectDisposedException>(() => transformer.GetRowToRowMapper(Rows(Texts).Schema));

        // The embedder's encoder is not owned: an existing view and the embedder still work.
        Assert.Equal(Texts.Select(Expected), output.GetColumn<float[]>("Embedding"));
        Assert.Equal(Expected("a"), Embedder.Embed("a", Ct));
        Assert.Same(Embedder, transformer.Embedder);
        Assert.Equal(768, transformer.Dimension);
        Assert.Equal("Text", transformer.InputColumnName);
        Assert.Equal("Embedding", transformer.OutputColumnName);
        Assert.True(transformer.IsRowToRowMapper);
    }

    [Fact]
    public void CapabilityDescriptorReportsTheMlNetStatuses()
    {
        var all = EmbeddingGemma2Capabilities.All;
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Experimental, all["mlnet.embedding_transform"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Experimental, all["mlnet.batching"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Unsupported, all["mlnet.persistence"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Unsupported, all["mlnet.cancellation"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Unsupported, all["mlnet.training"]);
        Assert.Equal(EmbeddingGemma2CapabilityStatus.Experimental, all["mlnet.embedding_transform.cuda"]);
    }
}
