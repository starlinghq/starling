using System.Text.Json;
using Starlings.Generation.OnnxGenAI;
using Starlings.TestSupport;

namespace Starlings.Generation.OnnxGenAI.Tests;

public sealed class StopSequenceMatcherTests
{
    [Fact]
    public void PassesTextThroughWithoutStops()
    {
        var m = new StopSequenceMatcher([]);
        Assert.Equal(("abc", (string?)null), m.Push("abc"));
        Assert.Equal(string.Empty, m.Flush());
    }

    [Fact]
    public void HoldsBackAPossibleStartAndMatchesAcrossPieces()
    {
        var m = new StopSequenceMatcher(["STOP"]);
        Assert.Equal(("ab", (string?)null), m.Push("abST"));
        Assert.Equal((string.Empty, (string?)null), m.Push("O"));
        Assert.Equal((string.Empty, (string?)"STOP"), m.Push("Pc"));
    }

    [Fact]
    public void ReleasesHeldTextWhenItCannotMatch()
    {
        var m = new StopSequenceMatcher(["STOP"]);
        Assert.Equal(("x", (string?)null), m.Push("xST"));
        Assert.Equal(("STx", (string?)null), m.Push("x"));
        Assert.Equal(("y", (string?)null), m.Push("yS"));
        Assert.Equal("S", m.Flush());
    }

    [Fact]
    public void CutsBeforeTheEarliestMatchAndDropsTheRest()
    {
        var m = new StopSequenceMatcher(["END", "ND"]);
        Assert.Equal(("ab", "END"), m.Push("abENDxx"));
        var n = new StopSequenceMatcher(["b", "abc"]);
        Assert.Equal((string.Empty, (string?)"abc"), n.Push("abc"));
    }

    [Fact]
    public void LongerStopWinsAtTheSameStart()
    {
        var m = new StopSequenceMatcher(["ab", "abc"]);
        Assert.Equal(("x", "abc"), m.Push("xabc"));
    }
}

public sealed class OptionsAndLoadTests
{
    [Fact]
    public void ModelOptionsOverlaysAreRecordedJson()
    {
        var o = new GenAIModelOptions { IntraOpThreads = 6, PadTokenId = 248319, ConfigOverlays = ["{\"search\":{\"top_k\":1}}"] };
        o.Validate();
        var overlays = o.BuildOverlays();
        Assert.Equal(3, overlays.Count);
        Assert.Equal(6, JsonDocument.Parse(overlays[0]).RootElement.GetProperty("model").GetProperty("decoder").GetProperty("session_options").GetProperty("intra_op_num_threads").GetInt32());
        Assert.Equal(248319, JsonDocument.Parse(overlays[1]).RootElement.GetProperty("model").GetProperty("pad_token_id").GetInt32());
    }

    [Fact]
    public void UnsetIntraOpThreadsUsesTheContainerAwareDefault()
    {
        // Constrained process (CPU quota or affinity): the decoder gets Environment.ProcessorCount threads instead of
        // one per physical core of the host. Unconstrained: no overlay, the GenAI default.
        var overlays = new GenAIModelOptions().BuildOverlays();
        if (Starlings.Threading.IntraOpThreadDefaults.ForProcess() is int threads)
        {
            Assert.Single(overlays);
            Assert.Equal(threads, JsonDocument.Parse(overlays[0]).RootElement.GetProperty("model").GetProperty("decoder").GetProperty("session_options").GetProperty("intra_op_num_threads").GetInt32());
        }
        else
        {
            Assert.Empty(overlays);
        }
    }

    [Fact]
    public void NativeCallsAreSerializedByDefaultOnCudaOnly()
    {
        Assert.False(new GenAIModelOptions().EffectiveSerializeNativeCalls);
        Assert.True(new GenAIModelOptions { Provider = GenAIExecutionProvider.Cuda }.EffectiveSerializeNativeCalls);
        Assert.False(new GenAIModelOptions { Provider = GenAIExecutionProvider.Cuda, SerializeNativeCalls = false }.EffectiveSerializeNativeCalls);
        Assert.True(new GenAIModelOptions { SerializeNativeCalls = true }.EffectiveSerializeNativeCalls);
    }

    [Fact]
    public void InvalidModelOptionsAreRejected()
    {
        Assert.Throws<ArgumentException>(() => new GenAIModelOptions { UseTf32 = true }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new GenAIModelOptions { IntraOpThreads = 0 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new GenAIModelOptions { CudaDeviceId = -1 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new GenAIModelOptions { Provider = (GenAIExecutionProvider)7 }.Validate());
        new GenAIModelOptions { Provider = GenAIExecutionProvider.Cuda, UseTf32 = true }.Validate();
    }

    [Fact]
    public void MissingDirectoryOrConfigFailsBeforeNativeWork()
    {
        string dir = Directory.CreateTempSubdirectory("df-genai-").FullName;
        try
        {
            Assert.Throws<FileNotFoundException>(() => GenAIModel.Load(dir));
            File.WriteAllText(Path.Combine(dir, "genai_config.json"), "{\"model\":{}}");
            Assert.Throws<GenAIModelException>(() => GenAIModel.Load(dir));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void NativeRuntimeLoadsAndRejectsADirectoryWithoutTheGraph()
    {
        // A valid genai_config.json (the fp32x one) without model.onnx: the CPU native package loads, GenAI rejects
        // the directory, and the adapter reports GenAIModelException (not a crash, not a fallback).
        string dir = Directory.CreateTempSubdirectory("df-genai-").FullName;
        try
        {
            File.WriteAllText(Path.Combine(dir, "genai_config.json"), MinimalConfig);
            var ex = Assert.Throws<GenAIModelException>(() => GenAIModel.Load(dir));
            Assert.Contains(dir, ex.Message, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void CudaWithTheCpuNativePackageFailsWithoutFallback()
    {
        string dir = Directory.CreateTempSubdirectory("df-genai-").FullName;
        try
        {
            File.WriteAllText(Path.Combine(dir, "genai_config.json"), MinimalConfig);
            var ex = Record.Exception(() => GenAIModel.Load(dir, new GenAIModelOptions { Provider = GenAIExecutionProvider.Cuda }));
            Assert.True(ex is GenAIProviderUnavailableException or GenAIModelException, ex?.ToString());
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void VersionsComeFromTheDependencyManifest()
    {
        Assert.Equal("0.17.1", GenAIRuntimeInfo.PinnedVersion);
        Assert.Equal("0.17.1", GenAIRuntimeInfo.ManagedVersion);
        Assert.Equal("Microsoft.ML.OnnxRuntimeGenAI/0.17.1", GenAIRuntimeInfo.NativePackage);
        GenAIRuntimeInfo.CheckVersions();
    }

    private const string MinimalConfig = """
        {
          "model": {
            "bos_token_id": 1,
            "context_length": 262144,
            "decoder": {
              "session_options": { "log_id": "onnxruntime-genai", "provider_options": [] },
              "filename": "model.onnx",
              "head_size": 256,
              "hidden_size": 1024,
              "inputs": {
                "input_ids": "input_ids", "attention_mask": "attention_mask", "position_ids": "position_ids",
                "past_key_names": "past_key_values.%d.key", "past_value_names": "past_key_values.%d.value",
                "past_conv_names": "past.%d.conv", "past_recurrent_names": "past.%d.recurrent"
              },
              "outputs": {
                "logits": "logits", "present_key_names": "present.%d.key", "present_value_names": "present.%d.value",
                "present_conv_names": "present.%d.conv", "present_recurrent_names": "present.%d.recurrent"
              },
              "num_attention_heads": 8,
              "num_hidden_layers": 24,
              "num_key_value_heads": 2
            },
            "eos_token_id": [248046, 248044],
            "pad_token_id": 248044,
            "type": "qwen3_5_text",
            "vocab_size": 248320
          },
          "search": { "do_sample": false, "max_length": 262144, "num_beams": 1 }
        }
        """;
}

public sealed class IncrementalDetokenizerTests
{
    [Fact]
    public void PushAndFlushEqualOneDecode()
    {
        var (tok, reason) = Qwen35TestAssets.Tokenizer;
        Assert.SkipWhen(tok is null, reason);
        string text = "Grüße 你好 " + char.ConvertFromUtf32(0x1F44B) + " <think>x</think> done";
        int[] ids = [.. tok!.Encode(text, addSpecialTokens: false).Ids];
        var d = new IncrementalDetokenizer(tok);
        string streamed = string.Concat(ids.Select(d.Push)) + d.Flush();
        Assert.Equal(tok.Decode(ids, false), streamed);
        Assert.Equal(text, streamed);
        Assert.Equal(0, d.Count);
    }
}
