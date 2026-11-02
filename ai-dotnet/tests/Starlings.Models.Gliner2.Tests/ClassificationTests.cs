using System.Buffers.Binary;
using System.Globalization;
using System.Text;
using System.Text.Json;
using Starlings.TestSupport;
using Status = Starlings.Models.CapabilityStatus;

namespace Starlings.Models.Gliner2.Tests;

/// <summary>Weight-free tests of the classification head reader, the decision rule and argument checks. Run in CI.</summary>
public class ClassificationTests
{
    [Fact]
    public void CapabilityIsExperimentalOnCpuOnly()
    {
        var all = Gliner2Capabilities.Statuses;
        Assert.Equal(Status.Experimental, all["classification"]);
        Assert.Equal(Status.Unsupported, all["classification.cuda"]);
        Assert.Equal(Status.Unsupported, all["classification.multiple_tasks"]);
        Assert.Equal(Status.Unsupported, all["classification.label_descriptions"]);
        Assert.Equal(Status.Unsupported, all["classification.batch"]);
        Assert.Equal(Status.Experimental, Gliner2Capabilities.ProfileCapabilities[Gliner2Capabilities.StableProfile].Capabilities["classification"]);
        Assert.Equal(Status.Unsupported, Gliner2Capabilities.ProfileCapabilities[Gliner2Capabilities.StableCudaProfile].Capabilities["classification"]);
    }

    [Fact]
    public void ClassificationLockAddsTheCheckpointToTheEntityLock()
    {
        var files = Gliner2ModelAssets.ClassificationLock.Files;
        Assert.Equal(
            [Gliner2ModelAssets.ModelFile, Gliner2ModelAssets.TokenizerFile, Gliner2ModelAssets.ClassificationWeightsFile],
            files.Select(f => f.Path));
        var weights = files[2];
        Assert.Equal(Gliner2ModelAssets.ClassificationWeightsSize, weights.Size);
        Assert.Equal(Gliner2ModelAssets.ClassificationWeightsSha256, weights.Sha256);
        Assert.Equal(2, Gliner2ModelAssets.Lock.Files.Count);
    }

    [Fact]
    public void SingleLabelTakesTheSoftmaxArgmax()
    {
        var result = Gliner2TextClassifier.Decode("sentiment", ["positive", "negative", "neutral"], [1f, 3f, 2f], multiLabel: false, threshold: 0.5f);
        Assert.Equal("negative", result.Top.Label);
        Assert.Single(result.Selected);
        Assert.Equal(1.0, result.Scores.Sum(s => (double)s.Score), 5);
        double z = Math.Exp(1) + Math.Exp(3) + Math.Exp(2);
        Assert.Equal(Math.Exp(3) / z, result.Top.Score, 6);
        Assert.Equal(["positive", "negative", "neutral"], result.Scores.Select(s => s.Label));
    }

    [Fact]
    public void SingleLabelTiesGoToTheFirstLabel()
    {
        var result = Gliner2TextClassifier.Decode("t", ["a", "b"], [2f, 2f], multiLabel: false, threshold: 0.5f);
        Assert.Equal("a", result.Top.Label);
    }

    [Fact]
    public void MultiLabelSelectsEveryLabelAtOrAboveTheThresholdInLabelOrder()
    {
        var result = Gliner2TextClassifier.Decode("topics", ["a", "b", "c", "d"], [3f, -2f, 0f, 1f], multiLabel: true, threshold: 0.5f);
        Assert.Equal(["a", "c", "d"], result.Selected.Select(s => s.Label));
        Assert.Equal(0.5f, result.Scores[2].Score);
    }

    [Fact]
    public void MultiLabelFallsBackToTheBestLabelWhenNoneReachesTheThreshold()
    {
        var result = Gliner2TextClassifier.Decode("topics", ["a", "b", "c"], [-3f, -1f, -2f], multiLabel: true, threshold: 0.5f);
        Assert.Equal(["b"], result.Selected.Select(s => s.Label));
        Assert.True(result.Top.Score < 0.5f);
    }

    [Fact]
    public void ArgumentsAreChecked()
    {
        Assert.Throws<ArgumentException>(() => Gliner2TextClassifier.ValidateTask(" "));
        Assert.Throws<ArgumentException>(() => Gliner2TextClassifier.ValidateLabels(["only"]));
        Assert.Throws<ArgumentException>(() => Gliner2TextClassifier.ValidateLabels(["a", "a"]));
        Assert.Throws<ArgumentException>(() => Gliner2TextClassifier.ValidateLabels(["a", ""]));
        Assert.Equal(["B", "b"], Gliner2TextClassifier.ValidateLabels(["B", "b"]));
        Assert.Throws<ArgumentException>(() => Gliner2TextClassifier.Decode("t", ["a", "b"], [1f], false, 0.5f));
    }

    [Fact]
    public void HeadComputesTheMlpOfASyntheticCheckpoint()
    {
        var rng = new Random(7);
        var tensors = Gliner2ClassificationHead.Tensors
            .Select(t => (t.Name, t.Shape, Values: Enumerable.Range(0, t.Shape.Aggregate(1, (a, b) => a * b)).Select(_ => (float)(rng.NextDouble() - 0.5) * 0.1f).ToArray()))
            .ToArray();
        string path = WriteSafetensors(tensors, extra: true);
        try
        {
            var head = Gliner2ClassificationHead.FromSafetensors(path, expectedSha256: null);
            var x = Enumerable.Range(0, 2 * Gliner2ClassificationHead.HiddenSize).Select(_ => (float)(rng.NextDouble() - 0.5)).ToArray();
            var logits = head.Logits(x);
            Assert.Equal(2, logits.Length);
            for (int r = 0; r < 2; r++)
            {
                double logit = tensors[3].Values[0];
                for (int j = 0; j < Gliner2ClassificationHead.IntermediateSize; j++)
                {
                    double h = tensors[1].Values[j];
                    for (int k = 0; k < Gliner2ClassificationHead.HiddenSize; k++)
                    {
                        h += (double)tensors[0].Values[j * Gliner2ClassificationHead.HiddenSize + k] * x[r * Gliner2ClassificationHead.HiddenSize + k];
                    }

                    logit += tensors[2].Values[j] * Math.Max(0, h);
                }

                Assert.Equal(logit, logits[r], 4);
            }

            // The pinned digest belongs to the real head, not to these values.
            var ex = Assert.Throws<InvalidDataException>(() => Gliner2ClassificationHead.FromSafetensors(path));
            Assert.Contains(Gliner2ClassificationHead.HeadSha256, ex.Message, StringComparison.Ordinal);
            Assert.Throws<ArgumentException>(() => head.Logits(new float[10]));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void HeadRejectsAMissingTensorOrAnotherShape()
    {
        var shapes = Gliner2ClassificationHead.Tensors.Select(t => (t.Name, t.Shape, Values: new float[t.Shape.Aggregate(1, (a, b) => a * b)])).ToArray();
        string missing = WriteSafetensors(shapes[..3], extra: false);
        var wrong = shapes.ToArray();
        wrong[3] = (wrong[3].Name, [2], new float[2]);
        string reshaped = WriteSafetensors(wrong, extra: false);
        string garbage = Path.Combine(Path.GetTempPath(), $"gliner2-head-{Guid.NewGuid():N}.safetensors");
        File.WriteAllBytes(garbage, [1, 2, 3]);
        try
        {
            Assert.Contains("classifier.2.bias", Assert.Throws<InvalidDataException>(() => Gliner2ClassificationHead.FromSafetensors(missing, null)).Message, StringComparison.Ordinal);
            Assert.Contains("expected F32 [1]", Assert.Throws<InvalidDataException>(() => Gliner2ClassificationHead.FromSafetensors(reshaped, null)).Message, StringComparison.Ordinal);
            Assert.Throws<InvalidDataException>(() => Gliner2ClassificationHead.FromSafetensors(garbage, null));
        }
        finally
        {
            File.Delete(missing);
            File.Delete(reshaped);
            File.Delete(garbage);
        }
    }

    /// <summary>A minimal safetensors file (8-byte header length, JSON header, data), optionally with an unrelated tensor first.</summary>
    private static string WriteSafetensors((string Name, int[] Shape, float[] Values)[] tensors, bool extra)
    {
        var all = extra ? [("encoder.unrelated", new[] { 3 }, new float[] { 9f, 9f, 9f }), .. tensors] : tensors;
        var header = new Dictionary<string, object>();
        long offset = 0;
        foreach (var (name, shape, values) in all)
        {
            header[name] = new { dtype = "F32", shape, data_offsets = new[] { offset, offset + 4L * values.Length } };
            offset += 4L * values.Length;
        }

        byte[] json = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        string path = Path.Combine(Path.GetTempPath(), $"gliner2-head-{Guid.NewGuid():N}.safetensors");
        using var stream = File.Create(path);
        Span<byte> buffer = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(buffer, (ulong)json.Length);
        stream.Write(buffer);
        stream.Write(json);
        foreach (var (_, _, values) in all)
        {
            foreach (float v in values)
            {
                BinaryPrimitives.WriteSingleLittleEndian(buffer[..4], v);
                stream.Write(buffer[..4]);
            }
        }

        return path;
    }
}

/// <summary>
/// Capability <c>classification</c> on the CPU profile: the proposed classification cases
/// (<c>model-pipeline/proposed/gliner2-base-v1/classification</c>, run by the pinned GLiNER2) against this
/// profile's protected bounds for span logits (relative) and scores. Skips when the graph or <c>model.safetensors</c> is
/// absent (CI).
/// </summary>
public class ClassificationParityTests(Gliner2CpuFixture fixture, ITestOutputHelper output) : IClassFixture<Gliner2CpuFixture>
{
    /// <summary>Optional: the path of <c>model.safetensors</c> of <c>fastino/gliner2-base-v1</c> at the pinned revision.</summary>
    public const string WeightsEnvironmentVariable = "STARLING_GLINER2_SAFETENSORS";

    private static readonly Gliner2Tolerances Tolerances = Gliner2Tolerances.Load("onnx-fp32-cpu");

    [Fact]
    public void ClassificationMatchesTheReference()
    {
        var extractor = fixture.Extractor;
        string? weights = FindWeights();
        Assert.SkipWhen(weights is null, $"model.safetensors not found; set {WeightsEnvironmentVariable} or place it next to the graph's snapshot.");
        var classifier = new Gliner2TextClassifier(extractor, Gliner2ClassificationHead.FromSafetensors(weights!));
        string path = Path.Combine(Gliner2TestAssets.RepositoryRoot, "model-pipeline", "proposed", "gliner2-base-v1", "classification", "classification.jsonl");
        var rows = Gliner2TestAssets.ReadJsonLines(path);
        Assert.Equal(26, rows.Count);
        double maxRelative = 0, maxScore = 0;
        int decisionsExact = 0;
        foreach (var row in rows)
        {
            string id = row.GetProperty("id").GetString()!;
            string text = row.GetProperty("text").GetString()!;
            string task = row.GetProperty("task").GetString()!;
            string[] labels = row.GetProperty("labels").EnumerateArray().Select(e => e.GetString()!).ToArray();
            bool multi = row.GetProperty("multi_label").GetBoolean();
            float threshold = row.GetProperty("cls_threshold").GetSingle();
            var input = classifier.Prepare(text, task, labels);
            Assert.Equal(row.GetProperty("input_ids").EnumerateArray().Select(e => e.GetInt64()), input.InputIds.ToArray());
            Assert.Equal(row.GetProperty("schema_positions").EnumerateArray().Select(e => e.GetInt64()), input.SchemaIndex.ToArray());

            var result = classifier.Classify(text, task, labels, multi, threshold, TestContext.Current.CancellationToken);
            float[] refLogits = row.GetProperty("logits").EnumerateArray().Select(e => e.GetSingle()).ToArray();
            float[] refScores = row.GetProperty("probabilities").EnumerateArray().Select(e => e.GetSingle()).ToArray();
            for (int i = 0; i < labels.Length; i++)
            {
                maxRelative = Math.Max(maxRelative, Math.Abs(result.Scores[i].Logit - refLogits[i]) / Math.Max(1.0, Math.Abs(refLogits[i])));
                maxScore = Math.Max(maxScore, Math.Abs(result.Scores[i].Score - refScores[i]));
            }

            var expected = ReferenceSelection(row.GetProperty("output").GetProperty(task), multi);
            var actual = result.Selected.Select(s => s.Label).ToArray();
            if (expected.SequenceEqual(actual))
            {
                decisionsExact++;
            }
            else
            {
                // A different decision is allowed only where a reference score is within the score bound of the threshold.
                bool nearThreshold = multi && refScores.Any(s => Math.Abs(s - threshold) <= Tolerances.ScoresMaxAbs);
                Assert.True(nearThreshold, $"{id}: selected [{string.Join(", ", actual)}], reference [{string.Join(", ", expected)}]");
            }

            output.WriteLine(string.Create(CultureInfo.InvariantCulture, $"{id}: [{string.Join(", ", actual)}] top {result.Top.Score:G6}"));
        }

        output.WriteLine(string.Create(CultureInfo.InvariantCulture, $"cases {rows.Count}, decisions exact {decisionsExact}, logits max relative {maxRelative:E2} (bound {Tolerances.SpanLogitsRelative:E1}), scores max abs {maxScore:E2} (bound {Tolerances.ScoresMaxAbs:E1})"));
        Assert.True(maxRelative <= Tolerances.SpanLogitsRelative, $"logits relative {maxRelative:E2}");
        Assert.True(maxScore <= Tolerances.ScoresMaxAbs, $"scores {maxScore:E2}");
        Assert.Equal(rows.Count, decisionsExact);
    }

    [Fact]
    public void LongTextIsRejectedNotTruncated()
    {
        var extractor = fixture.Extractor;
        string? weights = FindWeights();
        Assert.SkipWhen(weights is null, $"model.safetensors not found; set {WeightsEnvironmentVariable}.");
        var classifier = new Gliner2TextClassifier(extractor, Gliner2ClassificationHead.FromSafetensors(weights!));
        string text = string.Join(' ', Enumerable.Repeat("word", 600));
        Assert.Throws<Gliner2InputTooLongException>(() => classifier.Classify(text, "topic", ["a", "b"], cancellationToken: TestContext.Current.CancellationToken));
    }

    private static string[] ReferenceSelection(JsonElement output, bool multi) =>
        multi
            ? output.EnumerateArray().Select(e => e.GetProperty("label").GetString()!).ToArray()
            : [output.GetProperty("label").GetString()!];

    private static string? FindWeights()
    {
        if (Environment.GetEnvironmentVariable(WeightsEnvironmentVariable) is { Length: > 0 } env && File.Exists(env))
        {
            return env;
        }

        var candidates = new List<string>
        {
            Path.Combine(Gliner2TestAssets.ModelsRoot, Gliner2ModelAssets.Repository, Gliner2ModelAssets.Revision, Gliner2ModelAssets.ClassificationWeightsFile),
        };
        if (Environment.GetEnvironmentVariable(Gliner2TestAssets.SnapshotEnvironmentVariable) is { Length: > 0 } snapshot)
        {
            candidates.Add(Path.Combine(snapshot, Gliner2ModelAssets.ClassificationWeightsFile));
        }

        return candidates.FirstOrDefault(File.Exists);
    }
}
