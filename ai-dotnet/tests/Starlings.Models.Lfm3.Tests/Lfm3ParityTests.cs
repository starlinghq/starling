using System.Text.Json;
using Starlings.Acquisition;
using Starlings.Generation;
using Starlings.Tokenizers;

namespace Starlings.Models.Lfm3.Tests;

/// <summary>
/// The .NET decode loop against <c>run_lfm3.py onnx</c> (the same fp32 graph on ONNX Runtime 1.30.0 in Python): the
/// greedy ids of every corpus case and the step-0 logits. Needs the fp32 snapshot (<see cref="SnapshotVariable"/>) and
/// the runner's output directory (<see cref="ReferenceVariable"/>); skips without them.
/// </summary>
public sealed class Lfm3ParityTests
{
    /// <summary>A directory holding <c>onnx/model.onnx</c>, its four data files and <c>tokenizer.json</c> of the pinned export.</summary>
    public const string SnapshotVariable = "STARLING_LFM3_SNAPSHOT";

    /// <summary>The <c>--out</c> directory of <c>run_lfm3.py onnx</c> (<c>onnx.jsonl</c>, <c>onnx-step0/</c>).</summary>
    public const string ReferenceVariable = "STARLING_LFM3_REFERENCE";

    private static readonly Lazy<Lfm3ChatModel?> _model = new(Load);

    [Fact]
    public void GreedyIdsAndStep0LogitsMatchOnnxRuntimeInPython()
    {
        string? reference = Environment.GetEnvironmentVariable(ReferenceVariable);
        Assert.SkipWhen(string.IsNullOrEmpty(reference) || !File.Exists(Path.Combine(reference, "onnx.jsonl")), $"Set {ReferenceVariable} to a run_lfm3.py onnx output directory.");
        Lfm3ChatModel? model = _model.Value;
        Assert.SkipWhen(model is null, $"Set {SnapshotVariable} to an LFM3-1.7B-ONNX fp32 snapshot.");

        Dictionary<string, JsonElement> template = Lfm3ChatTemplateTests.ReadCases();
        var failures = new List<string>();
        foreach (string line in File.ReadAllLines(Path.Combine(reference!, "onnx.jsonl")))
        {
            JsonElement row = JsonDocument.Parse(line).RootElement;
            string id = row.GetProperty("id").GetString()!;
            JsonElement tcase = template[id];
            int[] prompt = [.. tcase.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32())];
            int[] expected = [.. row.GetProperty("generated_ids").EnumerateArray().Select(e => e.GetInt32())];
            float[]? step0 = null;
            var options = new GenerationOptions { MaxNewTokens = tcase.GetProperty("max_new_tokens").GetInt32() };
            GenerationResult? result = null;
            foreach (var update in model!.GenerateStreamingCore(prompt, options, l => step0 = l, TestContext.Current.CancellationToken))
            {
                result = update.Result;
            }

            float[] expectedStep0 = ReadF32(Path.Combine(reference!, "onnx-step0", id + ".f32"));
            float maxAbs = 0;
            for (int i = 0; i < expectedStep0.Length; i++)
            {
                maxAbs = Math.Max(maxAbs, Math.Abs(expectedStep0[i] - step0![i]));
            }

            if (!result!.TokenIds.SequenceEqual(expected) || maxAbs > 1e-4f)
            {
                failures.Add($"{id}: ids {(result.TokenIds.SequenceEqual(expected) ? "equal" : "differ")}, step-0 max abs {maxAbs:g3}");
            }

            TestContext.Current.SendDiagnosticMessage($"{id}: {result.TokenIds.Length} ids, step-0 max abs {maxAbs:g3}, {result.FinishReason}");
        }

        Assert.Empty(failures);
    }

    [Fact]
    public void PromptIdsMatchTheReferenceTokenizer()
    {
        string? dir = Environment.GetEnvironmentVariable(SnapshotVariable);
        string path = Path.Combine(dir ?? string.Empty, Lfm3ModelAssets.TokenizerFile);
        Assert.SkipWhen(string.IsNullOrEmpty(dir) || !File.Exists(path), $"Set {SnapshotVariable} to an LFM3-1.7B-ONNX snapshot.");
        Tokenizer tokenizer = Tokenizer.FromFile(path);
        foreach (var (id, row) in Lfm3ChatTemplateTests.ReadCases())
        {
            IReadOnlyList<ChatTurn> turns = Lfm3ChatTemplateTests.Turns(row.GetProperty("messages"));
            bool enableThinking = row.GetProperty("enable_thinking").GetBoolean();
            bool preserveThinking = row.TryGetProperty("preserve_thinking", out var p) && p.GetBoolean();
            int[] expected = [.. row.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32())];
            string text = Lfm3ChatTemplate.Render(turns, enableThinking, preserveThinking);
            Assert.True(expected.SequenceEqual(tokenizer.Encode(text, addSpecialTokens: false).Ids.ToArray()), id);
        }
    }

    private static Lfm3ChatModel? Load()
    {
        string? dir = Environment.GetEnvironmentVariable(SnapshotVariable);
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir))
        {
            return null;
        }

        Lfm3ModelFiles files = Lfm3ModelAssets.OpenDirectory(dir, Lfm3Precision.Fp32, CacheVerification.Size);
        return Lfm3ChatModel.Create(files, new Lfm3Options { ContextLength = 4096 });
    }

    private static float[] ReadF32(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        var values = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }
}
