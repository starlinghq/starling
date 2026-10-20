namespace Starlings.Models.Whisper.Tests;

/// <summary>Encoder and decoder parity against ONNX Runtime in the reference (same graphs, so bit-exact).</summary>
public sealed class WhisperModelParityTests
{
    private const string Golden = "/home/mark/Projects/starling/.models/snapshots/whisper-large-v3-turbo";
    private const string EncoderModel = Golden + "/onnx/encoder_model_fp16.onnx";
    private const string DecoderModel = Golden + "/onnx/decoder_model_fp16.onnx";

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

    [Fact]
    public void EncoderMatchesReference()
    {
        using var model = WhisperModel.Create(EncoderModel, DecoderModel);
        float[] mel = ReadF32($"{Golden}/golden-mel-numpy.f32");
        float[] expected = ReadF32($"{Golden}/golden-encoder-hidden.f32");

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
        using var model = WhisperModel.Create(EncoderModel, DecoderModel);
        float[] hidden = ReadF32($"{Golden}/golden-encoder-hidden.f32");
        float[] expected = ReadF32($"{Golden}/golden-decoder-logits.f32");

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
        using var model = WhisperModel.Create(EncoderModel, DecoderModel);
        float[] mel = ReadF32($"{Golden}/golden-mel-numpy.f32");
        float[] hidden = model.Encode(mel, TestContext.Current.CancellationToken);

        int[] prompt = WhisperModel.Prompt("en");
        int[] actual = model.Generate(hidden, prompt, maxLength: 448, cancellationToken: TestContext.Current.CancellationToken);

        int[] expected = [50258, 50259, 50360, 50364, 1044, 291, 13, 50257];
        Assert.Equal(expected, actual);
    }
}
