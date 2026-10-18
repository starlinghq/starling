// Multilingual support-ticket triage: local e5 embeddings as features in an ML.NET pipeline, GLiNER2 extraction.
//
//   dotnet run -- download [--cache <dir>]                          explicit acquisition of e5 and GLiNER2 (network)
//   dotnet run -- train    [--cache <dir>] [--model <zip>]          train on English tickets, evaluate per language
//   dotnet run -- triage   [--cache <dir>] [--model <zip>] [text]   route new tickets with the saved classifier
//
// train and triage load verified local files only and never download.
using System.Globalization;
using System.Numerics.Tensors;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Starlings.Acquisition;
using Starlings.Models.E5;
using Starlings.Models.E5.MLNet;
using Starlings.Models.Gliner2;

try
{
    return await RunAsync();
}
catch (Exception ex) when (ex is not (OutOfMemoryException or StackOverflowException))
{
    Console.Error.WriteLine($"error: {ex.Message}");
    return 1;
}

async Task<int> RunAsync()
{
    if (args.Length == 0 || args[0] is not ("download" or "train" or "triage"))
        return Usage();

    string? cache = null;
    string modelPath = "triage-model.zip";
    var texts = new List<string>();
    for (int i = 1; i < args.Length; i++)
    {
        if (args[i] is "--cache" or "--model")
        {
            if (i + 1 == args.Length || args[i + 1].StartsWith("--", StringComparison.Ordinal) || args[0] == "download" && args[i] == "--model")
                return Usage();
            if (args[i] == "--cache") cache = args[++i];
            else modelPath = args[++i];
        }
        else if (args[0] == "triage" && !args[i].StartsWith("--", StringComparison.Ordinal))
            texts.Add(args[i]);
        else
            return Usage();
    }

    var acquisition = new ModelAcquisitionOptions { CacheDirectory = cache };
    if (args[0] == "download")
    {
        var e5 = await E5ModelAssets.DownloadAsync(acquisition);
        Console.WriteLine($"Verified e5: {e5.Snapshot.Directory}");
        var gliner2 = await Gliner2ModelAssets.DownloadAsync(acquisition);
        Console.WriteLine($"Verified GLiNER2: {gliner2.Snapshot.Directory}");
        return 0;
    }

    // Seeded, with single-threaded trainers (SingleThreaded below), so training is reproducible run to run.
    var ml = new MLContext(seed: 1);
    string dataDir = Path.Combine(AppContext.BaseDirectory, "data");
    IDataView history = LoadTickets(ml, Path.Combine(dataDir, "train.tsv"));

    // The embedding is a pretrained transform: Fit loads the verified model and trains nothing. "query: " is e5's
    // prefix for symmetric tasks such as classification and ticket-to-ticket similarity (model card).
    using E5EmbeddingTransformer embedding = ml.Transforms
        .ApplyE5Embedding(E5ModelAssets.OpenLocal(acquisition), outputColumnName: "Embedding", inputColumnName: nameof(Ticket.Text),
            new E5EmbeddingOptions { Prefix = E5TextPrefix.Query })
        .Fit(history);

    return args[0] == "train"
        ? Train(ml, embedding, history, LoadTickets(ml, Path.Combine(dataDir, "test.tsv")), modelPath)
        : Triage(ml, embedding, history, modelPath, acquisition, texts);
}

static int Train(MLContext ml, E5EmbeddingTransformer embedding, IDataView train, IDataView test, string modelPath)
{
    // The classifier: label to key, the 384 embedding values as the feature vector, a linear multiclass model, and the
    // predicted key back to the queue name. Everything here is an ordinary ML.NET estimator and can be saved.
    var classifier = ml.Transforms.Conversion.MapValueToKey("Label", nameof(Ticket.Queue))
        .Append(ml.Transforms.Concatenate("Features", "Embedding"))
        .Append(ml.MulticlassClassification.Trainers.SdcaMaximumEntropy(SingleThreaded()))
        .Append(ml.Transforms.Conversion.MapKeyToValue(nameof(Prediction.PredictedQueue), "PredictedLabel"));

    // SDCA makes many passes over its input, and the embedding column is computed lazily on every pass. Caching the
    // embedded rows runs e5 once per ticket instead of once per ticket per epoch.
    IDataView embeddedTrain = ml.Data.Cache(embedding.Transform(train));
    var stopwatch = System.Diagnostics.Stopwatch.StartNew();
    ITransformer model = classifier.Fit(embeddedTrain);
    Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
        $"Trained on {Count(train)} English tickets (embedding and training {stopwatch.Elapsed.TotalSeconds:F1} s)."));

    // Baseline: the same label mapping and trainer over ML.NET's own bag of word and character n-grams.
    ITransformer baseline = ml.Transforms.Conversion.MapValueToKey("Label", nameof(Ticket.Queue))
        .Append(ml.Transforms.Text.FeaturizeText("Features", nameof(Ticket.Text)))
        .Append(ml.MulticlassClassification.Trainers.SdcaMaximumEntropy(SingleThreaded()))
        .Fit(train);

    // The e5 transformer cannot be saved (its model is the pinned asset lock, resolved at run time), so evaluation
    // and prediction compose it in front of the classifier as a chain.
    var pipeline = new TransformerChain<ITransformer>(embedding, model);
    Console.WriteLine();
    Console.WriteLine("Accuracy on held-out tickets (the training set is English only)");
    Console.WriteLine("language  tickets  bag-of-n-grams  e5 embeddings");
    var languages = ml.Data.CreateEnumerable<Ticket>(test, reuseRowObject: false).Select(t => t.Language).Distinct().ToArray();
    foreach (string language in languages.Append("all non-English"))
    {
        // FilterByCustomPredicate drops the rows for which the predicate is true.
        IDataView subset = ml.Data.FilterByCustomPredicate<Ticket>(test,
            t => language == "all non-English" ? t.Language == "en" : t.Language != language);
        var bag = ml.MulticlassClassification.Evaluate(baseline.Transform(subset));
        var e5 = ml.MulticlassClassification.Evaluate(pipeline.Transform(subset));
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"{language,-16}  {Count(subset),2}  {bag.MicroAccuracy,7:P0}  {e5.MicroAccuracy,13:P0}"));
    }

    IDataView foreign = ml.Data.FilterByCustomPredicate<Ticket>(test, t => t.Language == "en");
    var metrics = ml.MulticlassClassification.Evaluate(pipeline.Transform(foreign));
    Console.WriteLine();
    Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
        $"e5 pipeline on non-English tickets: macro accuracy {metrics.MacroAccuracy:F3}, log-loss {metrics.LogLoss:F3}"));
    Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

    // Save the classifier with the schema it consumes (the input columns plus Embedding). triage re-attaches e5.
    ml.Model.Save(model, embeddedTrain.Schema, modelPath);
    Console.WriteLine($"Saved classifier: {Path.GetFullPath(modelPath)}");
    return 0;
}

static int Triage(MLContext ml, E5EmbeddingTransformer embedding, IDataView history, string modelPath,
    ModelAcquisitionOptions acquisition, List<string> texts)
{
    if (!File.Exists(modelPath))
        throw new FileNotFoundException($"No trained classifier at {modelPath}; run 'train' first.");
    ITransformer classifier = ml.Model.Load(modelPath, out _);

    // GLiNER2 joins the pipeline as a custom mapping: any local model can be a row-wise ML.NET transform. Not saved.
    using var extractor = Gliner2EntityExtractor.Create(Gliner2ModelAssets.OpenLocal(acquisition));
    string[] labels = ["order number", "product", "date"];
    var entities = ml.Transforms.CustomMapping<Ticket, Extracted>((ticket, output) =>
        output.Entities = [.. extractor.Extract(ticket.Text, labels).Entities.Select(e => $"{e.Label}: {e.Text}")],
        contractName: null);

    // e5 (re-attached) -> saved classifier -> GLiNER2, as one chain and one prediction engine.
    var pipeline = new TransformerChain<ITransformer>(embedding, classifier, entities.Fit(history));
    using var engine = ml.Model.CreatePredictionEngine<Ticket, Prediction>(pipeline);
    var slotNames = default(VBuffer<ReadOnlyMemory<char>>);
    engine.OutputSchema["Score"].GetSlotNames(ref slotNames);
    string[] queues = [.. slotNames.DenseValues().Select(n => n.ToString())];

    // Past tickets, embedded by the same transform, for "similar resolved ticket" links.
    var past = ml.Data.CreateEnumerable<EmbeddedTicket>(embedding.Transform(history), reuseRowObject: false).ToArray();

    if (texts.Count == 0)
    {
        texts =
        [
            "Hallo, meine Bestellung 40917 sollte am 12. Oktober ankommen, ist aber immer noch nicht da.",
            "Bonjour, le grille-pain TurboToast 3000 ne chauffe plus, je voudrais le renvoyer.",
            "ログインしようとするとパスワードが違うと言われます。",
            "My SmartHub 2 stopped responding after the firmware update on October 3.",
            "Hi, quick question about my account and an order.",
        ];
    }

    // A ticket goes to a person when the two likeliest queues are close, not merely when the top score is low: with five
    // queues and 80 training tickets, a correct prediction can hold well under half of the probability.
    const float ReviewMargin = 0.15f;
    foreach (string text in texts)
    {
        Prediction prediction = engine.Predict(new Ticket { Text = text });
        int[] order = [.. Enumerable.Range(0, queues.Length).OrderByDescending(i => prediction.Score[i])];
        float margin = prediction.Score[order[0]] - prediction.Score[order[1]];
        // Embeddings are L2-normalized, so the dot product is the cosine similarity.
        var similar = past.MaxBy(p => TensorPrimitives.Dot(p.Embedding, prediction.Embedding))!;
        Console.WriteLine($"ticket:   {text}");
        string review = margin < ReviewMargin ? "  -> needs human review" : "";
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"queue:    {prediction.PredictedQueue} {prediction.Score[order[0]]:P0}, then {queues[order[1]]} {prediction.Score[order[1]]:P0}{review}"));
        Console.WriteLine($"entities: {(prediction.Entities.Length == 0 ? "(none)" : string.Join("; ", prediction.Entities))}");
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"similar:  [{similar.Queue}] {similar.Text} (cosine {TensorPrimitives.Dot(similar.Embedding, prediction.Embedding):F3})"));
        Console.WriteLine();
    }
    return 0;
}

// SDCA is multithreaded by default, and then its result depends on scheduling even with a seed.
static SdcaMaximumEntropyMulticlassTrainer.Options SingleThreaded() =>
    new() { NumberOfThreads = 1 };

static IDataView LoadTickets(MLContext ml, string path) =>
    ml.Data.LoadFromTextFile<Ticket>(path, separatorChar: '\t', hasHeader: true);

static long Count(IDataView data) => data.GetRowCount() ?? data.GetColumn<string>(nameof(Ticket.Text)).LongCount();

static int Usage()
{
    Console.Error.WriteLine("usage: TicketTriage download [--cache <dir>] | train [--cache <dir>] [--model <zip>] | " +
        "triage [--cache <dir>] [--model <zip>] [\"ticket text\" ...]");
    return 2;
}

internal sealed class Ticket
{
    [LoadColumn(0)] public string Queue { get; set; } = "";
    [LoadColumn(1)] public string Language { get; set; } = "";
    [LoadColumn(2)] public string Text { get; set; } = "";
}

internal sealed class EmbeddedTicket
{
    public string Queue { get; set; } = "";
    public string Text { get; set; } = "";
    [VectorType(384)] public float[] Embedding { get; set; } = [];
}

internal sealed class Extracted
{
    public string[] Entities { get; set; } = [];
}

internal sealed class Prediction
{
    public string PredictedQueue { get; set; } = "";
    public float[] Score { get; set; } = [];
    [VectorType(384)] public float[] Embedding { get; set; } = [];
    public string[] Entities { get; set; } = [];
}
