using Starlings.Models.E5.Browser;
using Starlings.Runtime.OnnxWeb;

namespace Starlings.Models.E5.Browser.Tests;

public sealed class SignatureAndParsingTests
{
    // What onnxruntime-web 1.30.0 reports for the pinned e5 graph (sessionMetadata in starlings-onnxweb.js).
    private const string E5Metadata = """
        {"inputs":[{"name":"input_ids","type":"int64","shape":[-1,-1]},{"name":"attention_mask","type":"int64","shape":[-1,-1]},{"name":"token_type_ids","type":"int64","shape":[-1,-1]}],
         "outputs":[{"name":"last_hidden_state","type":"float32","shape":[-1,-1,384]}]}
        """;

    [Fact]
    public void ParsesSessionMetadata()
    {
        var (inputs, outputs) = OnnxWebTensorInfo.ParseMetadata(E5Metadata);
        Assert.Equal(["input_ids", "attention_mask", "token_type_ids"], inputs.Select(i => i.Name));
        Assert.All(inputs, i => Assert.Equal("int64", i.ElementType));
        Assert.Equal([-1L, -1L, 384L], outputs[0].Dimensions!);
    }

    [Fact]
    public void NonTensorValuesParseWithoutTypeOrShape()
    {
        var (inputs, _) = OnnxWebTensorInfo.ParseMetadata("""{"inputs":[{"name":"x","type":null,"shape":null}],"outputs":[]}""");
        Assert.Null(inputs[0].ElementType);
        Assert.Null(inputs[0].Dimensions);
    }

    [Fact]
    public void TheE5SignatureIsAccepted()
    {
        var (inputs, outputs) = OnnxWebTensorInfo.ParseMetadata(E5Metadata);
        Assert.Equal((384, true), E5GraphSignature.Validate(inputs, outputs));
        Assert.Equal((384, false), E5GraphSignature.Validate([.. inputs.Take(2)], outputs));
    }

    public static TheoryData<string> BadSignatures => new()
    {
        """{"inputs":[{"name":"input_ids","type":"int64","shape":[-1,-1]}],"outputs":[{"name":"last_hidden_state","type":"float32","shape":[-1,-1,384]}]}""",
        """{"inputs":[{"name":"input_ids","type":"int32","shape":[-1,-1]},{"name":"attention_mask","type":"int64","shape":[-1,-1]}],"outputs":[{"name":"last_hidden_state","type":"float32","shape":[-1,-1,384]}]}""",
        """{"inputs":[{"name":"input_ids","type":"int64","shape":[-1,-1]},{"name":"attention_mask","type":"int64","shape":[-1,-1]},{"name":"position_ids","type":"int64","shape":[-1,-1]}],"outputs":[{"name":"last_hidden_state","type":"float32","shape":[-1,-1,384]}]}""",
        """{"inputs":[{"name":"input_ids","type":"int64","shape":[-1,-1]},{"name":"attention_mask","type":"int64","shape":[-1,-1]}],"outputs":[{"name":"last_hidden_state","type":"float32","shape":[-1,-1,-1]}]}""",
        """{"inputs":[{"name":"input_ids","type":"int64","shape":[-1,-1]},{"name":"attention_mask","type":"int64","shape":[-1,-1]}],"outputs":[{"name":"sentence_embedding","type":"float32","shape":[-1,384]}]}""",
        """{"inputs":[{"name":"input_ids","type":"int64","shape":[-1,-1]},{"name":"attention_mask","type":"int64","shape":[-1,-1]}],"outputs":[{"name":"last_hidden_state","type":"float16","shape":[-1,-1,384]}]}""",
    };

    [Theory]
    [MemberData(nameof(BadSignatures))]
    public void OtherSignaturesAreRejected(string metadata)
    {
        var (inputs, outputs) = OnnxWebTensorInfo.ParseMetadata(metadata);
        Assert.Throws<NotSupportedException>(() => E5GraphSignature.Validate(inputs, outputs));
    }

    [Fact]
    public void ParsesRuntimeInfo()
    {
        var info = OnnxWebRuntimeInfo.Parse("""
            {"ort_version":"1.30.0","bundle":"wasm","bundle_url":"http://x/","cross_origin_isolated":true,"hardware_concurrency":12,
             "num_threads":4,"proxy":false,"webgpu_exposed":true,"secure_context":true}
            """);
        Assert.Equal(new OnnxWebRuntimeInfo("1.30.0", "wasm", true, 12, 4, false, true, true), info);
    }

    [Theory]
    [InlineData("ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665", true)]
    [InlineData("CA456C06B3A9505DDFD9131408916DD79290368331E7D76BB621F1CBA6BC8665", true)]
    [InlineData("ca456c06", false)]
    [InlineData("za456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665", false)]
    public void RecognizesSha256Hex(string value, bool expected) => Assert.Equal(expected, OnnxWebAssets.IsSha256Hex(value));

    [Fact]
    public void InputShapeMustMatchItsLength()
    {
        var input = OnnxWebInput.FromInt64("input_ids", new long[6], 2, 3);
        Assert.Equal("int64", input.ElementType);
        Assert.Equal(48, input.Bytes.Length);
        Assert.Equal(16, OnnxWebInput.FromFloat32("x", new float[4], 4).Bytes.Length);
        Assert.Throws<ArgumentException>(() => OnnxWebInput.FromInt64("input_ids", new long[5], 2, 3));
        Assert.Throws<ArgumentOutOfRangeException>(() => OnnxWebInput.FromInt64("input_ids", Array.Empty<long>(), -1));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(65)]
    public void RuntimeOptionsRejectThreadCountsOutOfRange(int threads) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new OnnxWebRuntimeOptions { NumThreads = threads }.Validate());

    [Fact]
    public void RuntimeOptionsCompareByValue()
    {
        Assert.Equal(new OnnxWebRuntimeOptions(), new OnnxWebRuntimeOptions { Bundle = OnnxWebBundle.Wasm });
        Assert.NotEqual(new OnnxWebRuntimeOptions(), new OnnxWebRuntimeOptions { NumThreads = 2 });
    }

    [Fact]
    public void TheBundlePathsPointAtThePackagedFiles()
    {
        Assert.Equal("_content/Starlings.Runtime.OnnxWeb/ort/ort.wasm.min.mjs", OnnxWebBundle.Wasm.ModuleUrl);
        Assert.False(OnnxWebBundle.Wasm.SupportsWebGpu);
        var path = Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "src", "Starlings.Runtime.OnnxWeb", "wwwroot", "starlings-onnxweb.js");
        Assert.True(File.Exists(path), path);
    }

    [Fact]
    public void PinnedGraphsNameTheirProfilesAndDigests()
    {
        Assert.Equal(E5ModelAssets.ModelSha256, E5BrowserAssets.Fp32.Sha256);
        Assert.Equal("multilingual-e5-small/onnx-fp32-wasm", E5BrowserAssets.For(E5BrowserVariant.Fp32).ProfileWasm);
        Assert.Equal("multilingual-e5-small/onnx-int8embed-webgpu", E5BrowserAssets.For(E5BrowserVariant.Int8Embed).ProfileWebGpu);
        Assert.Null(E5BrowserAssets.Int8Embed.DefaultUrl);
        Assert.Equal(
            "https://huggingface.co/intfloat/multilingual-e5-small/resolve/614241f622f53c4eeff9890bdc4f31cfecc418b3/onnx/model.onnx",
            E5BrowserAssets.Fp32.DefaultUrl);
        foreach (var profile in new[] { "onnx-fp32-wasm", "onnx-fp32-webgpu", "onnx-int8embed-wasm", "onnx-int8embed-webgpu" })
        {
            string tolerances = Path.Combine(TestSupport.E5TestAssets.RepositoryRoot, "model-pipeline", "protected", "multilingual-e5-small", profile, "tolerances.yaml");
            Assert.Contains($"profile: multilingual-e5-small/{profile}", File.ReadAllText(tolerances), StringComparison.Ordinal);
        }
    }
}
