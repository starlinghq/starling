using Microsoft.ML.OnnxRuntime;
using Starlings.Models.E5.Browser;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.E5.Browser.Tests;

/// <summary>
/// The browser encoder's own code (input checks, chunk plan, padding, pooling) on the real graph, with native ONNX
/// Runtime standing in for onnxruntime-web: it must equal <see cref="E5Encoder"/> bit for bit, so any browser
/// difference is the backend's alone (measured in docs/evidence/browser-e5-package-2026-10-23/). Skips without the model.
/// </summary>
public sealed class NativeParityTests : IClassFixture<NativeParityTests.Model>
{
    private readonly Model _model;

    public NativeParityTests(Model model) => _model = model;

    public sealed class Model : E5ModelFixture
    {
        public Model()
            : base(new OnnxSessionOptions())
        {
        }
    }

    /// <summary>The e5 graph on a native session, as an <see cref="IE5GraphRunner"/>.</summary>
    private sealed class NativeRunner(OnnxModelSession session) : IE5GraphRunner
    {
        public int Dimension => 384;

        public bool HasTokenTypeIds => true;

        public Task RunAsync(long[] ids, long[] mask, long[]? types, int rows, int width, Memory<float> hidden, CancellationToken cancellationToken)
        {
            int cells = rows * width;
            long[] shape = [rows, width];
            var memory = OrtMemoryInfo.DefaultInstance;
            using var idsValue = OrtValue.CreateTensorValueFromMemory(memory, ids.AsMemory(0, cells), shape);
            using var maskValue = OrtValue.CreateTensorValueFromMemory(memory, mask.AsMemory(0, cells), shape);
            using var typesValue = OrtValue.CreateTensorValueFromMemory(memory, types!.AsMemory(0, cells), shape);
            using var hiddenValue = OrtValue.CreateTensorValueFromMemory(memory, hidden, [rows, width, Dimension]);
            session.Run(
                [E5Encoder.InputIdsName, E5Encoder.AttentionMaskName, E5Encoder.TokenTypeIdsName],
                [idsValue, maskValue, typesValue],
                [E5Encoder.LastHiddenStateName],
                [hiddenValue],
                cancellationToken);
            return Task.CompletedTask;
        }

        public void Dispose()
        {
        }
    }

    [Fact]
    public async Task SingleRowsEqualE5EncoderBitForBit()
    {
        var native = _model.Encoder;
        using var browser = new E5BrowserEncoder(new NativeRunner(native.Session), native.Options);
        foreach (var c in E5Fixtures.Cases)
        {
            var input = new EncoderInput(c.InputIds);
            Assert.Equal(native.Embed(input, TestContext.Current.CancellationToken), await browser.EmbedAsync(input, TestContext.Current.CancellationToken));
        }
    }

    [Fact]
    public async Task PaddedBatchesEqualE5EncoderBitForBit()
    {
        var native = _model.Encoder;
        using var browser = new E5BrowserEncoder(new NativeRunner(native.Session), native.Options with { MaxBatchSize = 5 });
        var inputs = E5Fixtures.Cases.Select(c => new EncoderInput(c.InputIds)).ToArray();
        using var chunked = E5Encoder.Create(_model.ModelPath!, native.Options with { MaxBatchSize = 5 });
        float[][] expected = chunked.EmbedBatch(inputs, TestContext.Current.CancellationToken);
        float[][] actual = await browser.EmbedBatchAsync(inputs, TestContext.Current.CancellationToken);
        for (int i = 0; i < inputs.Length; i++)
        {
            Assert.Equal(expected[i], actual[i]);
        }
    }

    [Fact]
    public void TheBytesOverloadOfTheTokenizerEqualsTheFileOverload()
    {
        var (files, reason) = E5TestAssets.FindFiles();
        Assert.SkipWhen(files is null, reason);
        byte[] bytes = File.ReadAllBytes(files!.TokenizerPath);
        var fromFile = E5Tokenizer.FromFile(files.TokenizerPath, E5ModelAssets.TokenizerSha256);
        var fromBytes = E5Tokenizer.FromUtf8Json(bytes, E5ModelAssets.TokenizerSha256);
        foreach (var text in E5Fixtures.Texts.Values)
        {
            Assert.Equal(fromFile.Tokenize(text).InputIds.ToArray(), fromBytes.Tokenize(text).InputIds.ToArray());
        }

        bytes[^2] ^= 1;
        Assert.Throws<InvalidDataException>(() => E5Tokenizer.FromUtf8Json(bytes, E5ModelAssets.TokenizerSha256));
    }
}
