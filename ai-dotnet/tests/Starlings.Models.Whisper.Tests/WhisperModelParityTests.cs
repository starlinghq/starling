using Starlings.TestSupport;

namespace Starlings.Models.Whisper.Tests;

/// <summary>Encoder and decoder parity against ONNX Runtime in the reference (same graphs, so bit-exact).</summary>
/// <remarks>The model and golden fixtures live in the snapshot directory (not Git); tests skip when they are absent.</remarks>
public sealed class WhisperModelParityTests
{
    private static float[] ReadF32(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        var result = new float[bytes.Length / sizeof(float)];
        using var ms = new MemoryStream(bytes);
        using var reader = new BinaryReader(ms);
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = reader.ReadSingle();
        }

        return result;
    }

    private static float[] Golden(string name, out string reason)
    {
        var (path, r) = WhisperTestAssets.FindGoldenFile(name);
        reason = r;
        return path is null ? [] : ReadF32(path);
    }

    [Fact]
    public void EncoderMatchesReference()
    {
        var (files, modelReason) = WhisperTestAssets.FindModel();
        Assert.SkipWhen(files is null, modelReason);
        float[] mel = Golden("golden-mel-numpy.f32", out string goldenReason);
        Assert.SkipWhen(mel.Length == 0, goldenReason);
        float[] expected = Golden("golden-encoder-hidden.f32", out _);

        using var model = WhisperModel.Create(files!.EncoderPath, files.DecoderPath);
        float[] actual = model.Encode(mel, TestContext.Current.CancellationToken);

        Assert.Equal(expected.Length, actual.Length);
        float maxAbs = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(expected[i] - actual[i]));
        }

        Assert.True(maxAbs < 1e-5, $"encoder max abs diff {maxAbs}");
    }

    [Fact]
    public void DecoderMatchesReference()
    {
        var (files, modelReason) = WhisperTestAssets.FindModel();
        Assert.SkipWhen(files is null, modelReason);
        float[] hidden = Golden("golden-encoder-hidden.f32", out string goldenReason);
        Assert.SkipWhen(hidden.Length == 0, goldenReason);
        float[] expected = Golden("golden-decoder-logits.f32", out _);

        using var model = WhisperModel.Create(files!.EncoderPath, files.DecoderPath);
        long[] prompt = [50258, 50259, 50360, 50364];
        float[] actual = model.DecodeLogits(prompt, hidden, TestContext.Current.CancellationToken);

        Assert.Equal(expected.Length, actual.Length);
        float maxAbs = 0f;
        for (int i = 0; i < expected.Length; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(expected[i] - actual[i]));
        }

        Assert.True(maxAbs < 1e-5, $"decoder max abs diff {maxAbs}");
    }

    [Fact]
    public void GreedyDecodeMatchesReference()
    {
        var (files, modelReason) = WhisperTestAssets.FindModel();
        Assert.SkipWhen(files is null, modelReason);
        float[] mel = Golden("golden-mel-numpy.f32", out string goldenReason);
        Assert.SkipWhen(mel.Length == 0, goldenReason);

        using var model = WhisperModel.Create(files!.EncoderPath, files.DecoderPath);
        float[] hidden = model.Encode(mel, TestContext.Current.CancellationToken);
        int[] prompt = WhisperModel.Prompt("en");
        int[] actual = model.Generate(hidden, prompt, maxLength: 448, cancellationToken: TestContext.Current.CancellationToken);

        int[] expected = [50258, 50259, 50360, 50364, 1044, 291, 13, 50257];
        Assert.Equal(expected, actual);
    }

    [Fact]
    public void GenerateMatchesTheStepwiseFullContextLoop()
    {
        // Generate reuses one ids buffer and one growing logits buffer; the ids must equal a loop that allocates fresh
        // logits through DecodeLogits on every step. Speech-free input: a 440 Hz tone, so no golden fixture is needed.
        var (files, modelReason) = WhisperTestAssets.FindModel();
        Assert.SkipWhen(files is null, modelReason);
        var waveform = new float[WhisperMelSpectrogram.SampleRate * 2];
        for (int i = 0; i < waveform.Length; i++)
        {
            waveform[i] = 0.25f * MathF.Sin(2 * MathF.PI * 440 * i / WhisperMelSpectrogram.SampleRate);
        }

        var cancellationToken = TestContext.Current.CancellationToken;
        using var model = WhisperModel.Create(files!.EncoderPath, files.DecoderPath);
        float[] hidden = model.Encode(WhisperMelSpectrogram.ExtractChunk(waveform), cancellationToken);
        // The one-token prompt starts the logits buffer at 2 rows, so it grows during the decode.
        foreach (int[] prompt in new[] { WhisperModel.Prompt("en"), [WhisperModel.StartOfTranscript] })
        {
            foreach (int maxLength in new[] { 1, 3, 40 })
            {
                int[] expected = StepwiseGenerate(model, hidden, prompt, maxLength, cancellationToken);
                int[] actual = model.Generate(hidden, prompt, maxLength, cancellationToken);
                Assert.Equal(expected, actual);
            }
        }
    }

    [Fact]
    public void GenerateWithTimestampsMatchesTheStepwiseLoopAndSplitsIntoSegments()
    {
        var (files, modelReason) = WhisperTestAssets.FindModel();
        Assert.SkipWhen(files is null, modelReason);
        float[] mel = Golden("golden-mel-numpy.f32", out string goldenReason);
        Assert.SkipWhen(mel.Length == 0, goldenReason);

        var cancellationToken = TestContext.Current.CancellationToken;
        using var model = WhisperModel.Create(files!.EncoderPath, files.DecoderPath);
        float[] hidden = model.Encode(mel, cancellationToken);
        int[] prompt = WhisperModel.Prompt("en", noTimestamps: false);
        int[] actual = model.GenerateWithTimestamps(hidden, prompt, cancellationToken: cancellationToken);

        Assert.Equal(StepwiseGenerate(model, hidden, prompt, 448, cancellationToken, timestamps: true), actual);
        Assert.Equal(WhisperModel.TimestampBegin, actual[prompt.Length]);
        Assert.Equal(WhisperModel.EndOfText, actual[^1]);
        var segments = WhisperSegmenter.Split(actual[prompt.Length..^1], seek: 0, seekFrames: WhisperModel.NumFrames, out _);
        Assert.NotEmpty(segments);
        Assert.All(segments, segment => Assert.True(segment.Start <= segment.End));
    }

    private static int[] StepwiseGenerate(WhisperModel model, float[] hidden, int[] prompt, int maxLength, CancellationToken cancellationToken, bool timestamps = false)
    {
        var ids = new List<int>(prompt);
        for (int step = 0; step < maxLength && ids.Count < WhisperModel.MaxTargetPositions; step++)
        {
            float[] logits = model.DecodeLogits(ids.Select(id => (long)id).ToArray(), hidden, cancellationToken);
            var last = logits.AsSpan((ids.Count - 1) * WhisperModel.VocabSize, WhisperModel.VocabSize);
            WhisperModel.ApplySuppression(last, step == 0);
            if (timestamps)
            {
                WhisperModel.ApplyTimestampRules(last, ids.Skip(prompt.Length).Select(id => (long)id).ToArray());
            }

            ids.Add(WhisperModel.ArgMax(last));
            if (ids[^1] == WhisperModel.EndOfText)
            {
                break;
            }
        }

        return ids.ToArray();
    }
}
