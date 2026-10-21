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
}
