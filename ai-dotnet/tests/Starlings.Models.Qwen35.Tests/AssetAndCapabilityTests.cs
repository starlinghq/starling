using Starlings.Acquisition;
using Starlings.Generation.OnnxGenAI;
using Starlings.Models.Qwen35;
using Starlings.TestSupport;

namespace Starlings.Models.Qwen35.Tests;

public sealed class AssetAndCapabilityTests
{
    [Theory]
    [InlineData(Qwen35Variant.Fp32x, "ortgenai-fp32x", 3_077_177_344L)]
    [InlineData(Qwen35Variant.Fp16x, "ortgenai-fp16x", 1_572_143_104L)]
    public void LocksPinTheBuildAndTheSharedFiles(Qwen35Variant variant, string dir, long dataSize)
    {
        var l = Qwen35ModelAssets.GetLock(variant);
        Assert.Equal("Qwen/Qwen3.5-0.8B", l.Repository);
        Assert.Equal("2fc06364715b967f1860aea9cf38778875588b17", l.Revision);
        Assert.Equal(
            ["chat_template.jinja", "transformers-5.18.0/tokenizer.json", $"{dir}/genai_config.json", $"{dir}/model.onnx", $"{dir}/model.onnx.data"],
            l.Files.Select(f => f.Path));
        Assert.Equal(dataSize, l.GetFile($"{dir}/model.onnx.data").Size);
        Assert.Equal(Qwen35ChatTemplate.TemplateSha256, l.GetFile("chat_template.jinja").Sha256);
        Assert.Equal(Qwen35ModelAssets.TokenizerSha256, l.GetFile(Qwen35ModelAssets.TokenizerFile).Sha256);

        // Every file from the hosting repository, pinned by commit, at its snapshot path.
        Assert.All(l.Files, f => Assert.Equal(
            $"https://huggingface.co/Starlinghq/qwen3.5-0.8b-ortgenai/resolve/95a690c1774f144957cb7b76986a9126e8b1bb5f/{f.Path}",
            f.Source!.AbsoluteUri));
        Assert.False(Qwen35ModelAssets.HostingIsPrivate);
        // Round trip through the JSON lock format.
        var parsed = ModelAssetLock.Parse(l.ToJson());
        Assert.Equal(l.Files.Select(f => (f.Path, f.Size, f.Sha256, f.Source)), parsed.Files.Select(f => (f.Path, f.Size, f.Sha256, f.Source)));
    }

    [Fact]
    public void LocksAgreeWithTheRegistryManifests()
    {
        foreach (var (variant, profile) in new[] { (Qwen35Variant.Fp32x, "ortgenai-fp32x-cpu"), (Qwen35Variant.Fp32x, "ortgenai-fp32x-cuda"), (Qwen35Variant.Fp16x, "ortgenai-fp16x-cuda") })
        {
            string manifest = File.ReadAllText(Path.Combine(Qwen35TestAssets.ManifestDirectory, profile, "manifest.yaml"));
            foreach (var f in Qwen35ModelAssets.GetLock(variant).Files)
            {
                Assert.Contains($"path: {f.Path}", manifest, StringComparison.Ordinal);
                Assert.Contains(f.Sha256, manifest, StringComparison.Ordinal);
                Assert.Contains($"size: {f.Size}", manifest, StringComparison.Ordinal);
            }
        }
    }

    [Fact]
    public void OpenDirectoryReportsMissingFiles()
    {
        string dir = Directory.CreateTempSubdirectory("df-qwen35-").FullName;
        try
        {
            var ex = Assert.Throws<ModelAssetMissingException>(() => Qwen35ModelAssets.OpenDirectory(dir, Qwen35Variant.Fp32x));
            Assert.Contains("ortgenai-fp32x/model.onnx.data", ex.MissingPaths);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void Fp16xOnCpuIsRejectedBeforeLoading()
    {
        var files = new Qwen35ModelFiles(Qwen35Variant.Fp16x, "/nonexistent", "/nonexistent/tokenizer.json", "/nonexistent/chat_template.jinja", null!);
        Assert.Throws<NotSupportedException>(() => Qwen35ChatModel.Load(files));
    }

    [Fact]
    public void ATemplateOtherThanThePortedOneIsRejected()
    {
        string dir = Directory.CreateTempSubdirectory("df-qwen35-").FullName;
        try
        {
            string template = Path.Combine(dir, "chat_template.jinja");
            File.WriteAllText(template, "{{ messages }}");
            var files = new Qwen35ModelFiles(Qwen35Variant.Fp32x, dir, Path.Combine(dir, "tokenizer.json"), template, null!);
            Assert.Throws<InvalidDataException>(() => Qwen35ChatModel.Load(files));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void EveryCapabilityHasAnExplicitStatus()
    {
        var all = Qwen35Capabilities.All;
        foreach (string key in new[] { "streaming", "cancellation.between_steps", "cancellation.inside_a_step", "batch", "tool_calling", "sampling.seeded", "template.owned_renderer.text", "template.tools", "extensions_ai.chat_client", "extensions_ai.tools", "mlnet", "aot", "persistence", "training", "quantized", "backend_tokenizer" })
        {
            Assert.True(all.ContainsKey(key), key);
        }

        Assert.All(all.Values, v => Assert.True(Enum.IsDefined(v)));
        Assert.Equal(Qwen35CapabilityStatus.Experimental, all["tool_calling"]);
        Assert.Equal(Qwen35CapabilityStatus.Experimental, all["template.tools"]);
        Assert.Equal(Qwen35CapabilityStatus.Experimental, all["template.tool_messages"]);
        Assert.Equal(Qwen35CapabilityStatus.Unsupported, all["extensions_ai.tool_mode.required"]);
        Assert.Equal(Qwen35CapabilityStatus.Unsupported, all["structured_output"]);
        Assert.Equal(Qwen35CapabilityStatus.Unsupported, all["batch"]);
        Assert.Equal(Qwen35CapabilityStatus.Unsupported, all["cancellation.inside_a_step"]);
    }

    /// <summary>
    /// Batch independence (plan 10.1) does not apply because batching is not offered (capability "batch" unsupported):
    /// no public entry point of the model or the generation backend takes several prompts or conversations, and no
    /// option asks GenAI for a batch or several return sequences. One request is one sequence; concurrency is tested
    /// separately (ConcurrentRequestsAreIndependent).
    /// </summary>
    [Fact]
    public void BatchIsNotOfferedByAnyPublicEntryPoint()
    {
        Assert.Equal(Qwen35CapabilityStatus.Unsupported, Qwen35Capabilities.All["batch"]);
        var types = typeof(Qwen35ChatModel).Assembly.GetExportedTypes().Concat(typeof(TextGenerator).Assembly.GetExportedTypes()).ToList();
        foreach (var type in types)
        {
            foreach (var member in type.GetMembers(System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Static))
            {
                Assert.False(member.Name.Contains("Batch", StringComparison.OrdinalIgnoreCase) || member.Name.Contains("ReturnSequences", StringComparison.OrdinalIgnoreCase), $"{type.Name}.{member.Name}");
            }

            foreach (var method in type.GetMethods().Cast<System.Reflection.MethodBase>().Concat(type.GetConstructors()))
            {
                foreach (var p in method.GetParameters())
                {
                    Assert.False(IsCollectionOfPrompts(p.ParameterType), $"{type.Name}.{method.Name}({p.Name}: {p.ParameterType})");
                }
            }
        }

        static bool IsCollectionOfPrompts(Type t)
        {
            var element = t.IsArray ? t.GetElementType() : t.GetInterfaces().Append(t)
                .FirstOrDefault(i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(IEnumerable<>))?.GetGenericArguments()[0];
            if (element is null || element == typeof(char))
            {
                return false;
            }

            bool sequenceOfIds = element.IsArray && element.GetElementType() == typeof(int) ||
                element.GetInterfaces().Append(element).Any(i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(IEnumerable<>) && i.GetGenericArguments()[0] == typeof(int));
            bool conversation = element.GetInterfaces().Append(element).Any(i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(IEnumerable<>) && i.GetGenericArguments()[0] == typeof(Qwen35Message));
            return sequenceOfIds || conversation || element == typeof(Qwen35Prompt);
        }
    }

    [Fact]
    public void PadOverlayIsOutsideTheTokenizerVocabulary()
    {
        var (tokenizer, reason) = Qwen35TestAssets.Tokenizer;
        Assert.SkipWhen(tokenizer is null, reason);
        Assert.Null(tokenizer!.IdToToken(Qwen35ChatModel.PadTokenIdOverlay));
        Assert.Equal(Qwen35ChatModel.EndOfTextTokenId, tokenizer.TokenToId("<|endoftext|>"));
        Assert.Equal(Qwen35ChatModel.ImEndTokenId, tokenizer.TokenToId("<|im_end|>"));
        // A user typing <|endoftext|> produces the configured pad id 248044, which GenAI would mask without the overlay.
        Assert.Contains(Qwen35ChatModel.EndOfTextTokenId, tokenizer.Encode("a <|endoftext|> b", addSpecialTokens: false).Ids);
    }

    [Fact]
    public void ChatOptionsMapOntoGenerationOptions()
    {
        var o = new Qwen35ChatOptions { MaxNewTokens = 9, StopSequences = ["x"], AdditionalStopTokenIds = [5], Sampling = new SamplingOptions { Seed = 2 }, EnableThinking = true };
        var g = o.ToGenerationOptions();
        Assert.Equal(9, g.MaxNewTokens);
        Assert.Equal(["x"], g.StopSequences);
        Assert.Equal([5], g.AdditionalStopTokenIds);
        Assert.Equal(2, g.Sampling!.Seed);
    }
}
