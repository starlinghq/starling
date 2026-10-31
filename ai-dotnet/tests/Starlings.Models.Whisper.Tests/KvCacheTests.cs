using Starlings.TestSupport;

namespace Starlings.Models.Whisper.Tests;

/// <summary>
/// The KV-cache decoder loop: its lock entry and options (weight-free), and against the full-context loop on the same
/// graphs (needs the snapshot; skips without it).
/// </summary>
public sealed class KvCacheTests
{
    [Fact]
    public void TheLockPinsTheKvCacheGraphAtTheSameRevision()
    {
        var file = Assert.Single(WhisperModelAssets.Lock.Files, entry => entry.Path == WhisperModelAssets.DecoderWithPastFile);

        Assert.Equal(317_742_004, file.Size);
        Assert.Equal(WhisperModelAssets.DecoderWithPastSha256, file.Sha256);
        Assert.Equal("76ba4fb597cca51a329a1820e49282e26248b36b", WhisperModelAssets.Revision);
    }

    [Fact]
    public void KvCacheModeWithoutTheGraphIsRefusedBeforeAnyGraphLoads()
    {
        var options = new WhisperOptions { DecoderMode = WhisperDecoderMode.KvCache };

        Assert.Throws<ArgumentException>(() => WhisperModel.Create("missing-encoder.onnx", "missing-decoder.onnx", options));
        Assert.Throws<ArgumentException>(() => WhisperModel.Create("missing-encoder.onnx", "missing-decoder.onnx", null, options));
    }

    [Fact]
    public void TheThreePathTranscriberRefusesKvCacheBeforeReadingTheTokenizer()
    {
        // The tokenizer path does not exist: the mode check must come first.
        var ex = Assert.Throws<ArgumentException>(() => WhisperTranscriber.Create(
            "missing-encoder.onnx", "missing-decoder.onnx", "missing-tokenizer.json", new WhisperOptions { DecoderMode = WhisperDecoderMode.KvCache }));

        Assert.Equal("options", ex.ParamName);
    }

    [Fact]
    public void BothLoopsAgreeAtOneTokenAndRefuseZero()
    {
        var (files, reason) = WhisperTestAssets.FindModel();
        Assert.SkipWhen(files is null, reason);
        var cancellationToken = TestContext.Current.CancellationToken;
        using var model = WhisperModel.Create(files!.EncoderPath, files.DecoderPath, files.DecoderWithPastPath);
        float[] hidden = model.Encode(WhisperMelSpectrogram.ExtractChunk(new float[16_000]), cancellationToken);
        int[] prompt = WhisperModel.Prompt("en");
        foreach (var mode in new[] { WhisperDecoderMode.FullContext, WhisperDecoderMode.KvCache })
        {
            int[] ids = model.GenerateRecordingLogits(hidden, prompt, mode, timestamps: false, [], maxLength: 1, cancellationToken);
            Assert.Equal(prompt.Length + 1, ids.Length);
            Assert.Throws<ArgumentOutOfRangeException>(
                () => model.GenerateRecordingLogits(hidden, prompt, mode, timestamps: false, [], maxLength: 0, cancellationToken));
        }

        Assert.Equal(
            model.GenerateRecordingLogits(hidden, prompt, WhisperDecoderMode.FullContext, false, [], 1, cancellationToken),
            model.GenerateRecordingLogits(hidden, prompt, WhisperDecoderMode.KvCache, false, [], 1, cancellationToken));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.Generate(hidden, prompt, maxLength: 0, cancellationToken));
    }

    [Fact]
    public void TheKvCacheLoopEqualsTheFullContextLoop()
    {
        var (files, reason) = WhisperTestAssets.FindModel();
        Assert.SkipWhen(files is null, reason);
        var (melPath, melReason) = WhisperTestAssets.FindGoldenFile("golden-mel-numpy.f32");
        Assert.SkipWhen(melPath is null, melReason);
        string withPast = files!.DecoderWithPastPath!;
        byte[] bytes = File.ReadAllBytes(melPath!);
        float[] mel = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, mel, 0, bytes.Length);

        var cancellationToken = TestContext.Current.CancellationToken;
        using var model = WhisperModel.Create(files.EncoderPath, files.DecoderPath, withPast);
        Assert.Equal(WhisperDecoderMode.KvCache, model.DecoderMode);
        float[] hidden = model.Encode(mel, cancellationToken);
        foreach (bool timestamps in new[] { false, true })
        {
            int[] prompt = WhisperModel.Prompt("en", noTimestamps: !timestamps);
            var cachedLogits = new List<float[]>();
            var fullLogits = new List<float[]>();
            int[] cached = model.GenerateRecordingLogits(hidden, prompt, WhisperDecoderMode.KvCache, timestamps, cachedLogits, cancellationToken: cancellationToken);
            int[] full = model.GenerateRecordingLogits(hidden, prompt, WhisperDecoderMode.FullContext, timestamps, fullLogits, cancellationToken: cancellationToken);

            Assert.Equal(full, cached);
            Assert.Equal(timestamps ? model.GenerateWithTimestamps(hidden, prompt, cancellationToken: cancellationToken) : model.Generate(hidden, prompt, cancellationToken: cancellationToken), cached);
            Assert.Equal(fullLogits.Count, cachedLogits.Count);
            float maxAbs = 0;
            for (int step = 0; step < fullLogits.Count; step++)
            {
                for (int i = 0; i < WhisperModel.VocabSize; i++)
                {
                    maxAbs = Math.Max(maxAbs, Math.Abs(fullLogits[step][i] - cachedLogits[step][i]));
                }
            }

            // The first step is the full-context graph on both paths; later steps differ by fp16 summation order only.
            Assert.Equal(fullLogits[0], cachedLogits[0]);
            Assert.InRange(maxAbs, 0f, 0.02f);
        }
    }
}
