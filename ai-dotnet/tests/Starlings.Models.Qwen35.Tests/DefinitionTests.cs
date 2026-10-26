using System.Reflection;
using System.Text.Json;
using Starlings.Acquisition;
using Starlings.Generation;
using Starlings.Models.Qwen35;
using Starlings.TestSupport;

namespace Starlings.Models.Qwen35.Tests;

/// <summary>
/// The model definition package (Starlings.Models.Qwen35) without a runtime: its dependency boundary, the checkpoint's
/// profile and lock, and the backend-neutral seams (IChatTemplate, IToolCallParser), which must give the same text and
/// calls as the typed API.
/// </summary>
public sealed class DefinitionTests
{
    public static TheoryData<string> RenderCaseIds() => [.. Qwen35TemplateCorpus.Load().Where(c => c.Owned == "render").Select(c => c.Id)];

    [Fact]
    public void DefinitionAssemblyReferencesNoOnnxRuntime()
    {
        AssemblyName[] references = typeof(Qwen35ChatTemplate).Assembly.GetReferencedAssemblies();
        Assert.DoesNotContain(references, r => r.Name!.StartsWith("Microsoft.ML.OnnxRuntime", StringComparison.Ordinal));
        Assert.DoesNotContain(references, r => r.Name == "Starlings.Generation.OnnxGenAI" || r.Name == "Starlings.Runtime.Onnx");
        Assert.Contains(references, r => r.Name == "Starlings.Generation.Abstractions");

        // The runtime types live in the ONNX Runtime GenAI package, in the same namespace.
        Assert.Equal("Starlings.Models.Qwen35.OnnxGenAI", typeof(Qwen35ChatModel).Assembly.GetName().Name);
        Assert.Equal(typeof(Qwen35ChatTemplate).Namespace, typeof(Qwen35ChatModel).Namespace);
    }

    [Fact]
    public void ProfileAgreesWithTheRuntimeConstants()
    {
        ModelProfile p = Qwen35Checkpoint.Profile;
        Assert.Equal("qwen3.5-0.8b", p.Id);
        Assert.Equal(Qwen35ModelAssets.Repository, p.Repository);
        Assert.Equal(Qwen35ModelAssets.Revision, p.Revision);
        Assert.Equal(Qwen35ModelAssets.TokenizerProfile, p.TokenizerProfile);
        Assert.Equal(Qwen35ModelAssets.TokenizerSha256, p.TokenizerSha256);
        Assert.Equal(Qwen35ChatTemplate.TemplateSha256, p.ChatTemplateSha256);
        Assert.Equal([Qwen35ChatModel.EndOfTextTokenId, Qwen35ChatModel.ImEndTokenId], p.EosTokenIds);
        Assert.Equal((248320, 1024, 24, 256, 262144), (p.VocabularySize, p.HiddenSize, p.Layers, p.HeadSize, p.ContextLength));
        Assert.Equal("Apache-2.0", p.License);
        Assert.True(Qwen35ChatModel.PadTokenIdOverlay < p.VocabularySize);
    }

    [Fact]
    public void CheckpointLockEqualsThePackagedJsonAndTheRegistry()
    {
        ModelAssetLock l = Qwen35Checkpoint.Lock;
        Assert.Equal((Qwen35Checkpoint.Repository, Qwen35Checkpoint.Revision), (l.Repository, l.Revision));
        Assert.All(l.Files, f => Assert.Null(f.Source));
        Assert.Equal(
            ["chat_template.jinja", "config.json", "model.safetensors-00001-of-00001.safetensors", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"],
            l.Files.Select(f => f.Path));

        string path = Path.Combine(Qwen35TestAssets.RepositoryRoot, "ai-dotnet", "src", "Starlings.Models.Qwen35", "locks", Qwen35Checkpoint.LockFileName);
        string json = File.ReadAllText(path);
        using (var doc = JsonDocument.Parse(json))
        {
            Assert.Equal(ModelAssetLock.SchemaVersion, doc.RootElement.GetProperty("schema_version").GetInt32());
        }

        ModelAssetLock file = ModelAssetLock.Load(path);
        Assert.Equal((l.Name, l.Repository, l.Revision), (file.Name, file.Repository, file.Revision));
        Assert.Equal(l.Files.Select(f => (f.Path, f.Size, f.Sha256, f.Source)), file.Files.Select(f => (f.Path, f.Size, f.Sha256, f.Source)));

        // The Hub files the registry manifests also pin carry the same digests and sizes.
        foreach (string profile in new[] { "ortgenai-fp32x-cpu", "ortgenai-fp32x-cuda", "ortgenai-fp16x-cuda" })
        {
            string manifest = File.ReadAllText(Path.Combine(Qwen35TestAssets.ManifestDirectory, profile, "manifest.yaml"));
            foreach (string pinned in new[] { "chat_template.jinja", "config.json", "model.safetensors-00001-of-00001.safetensors", "model.safetensors.index.json" })
            {
                ModelAssetFile f = l.GetFile(pinned);
                Assert.Contains($"path: {f.Path}", manifest, StringComparison.Ordinal);
                Assert.Contains(f.Sha256, manifest, StringComparison.Ordinal);
                Assert.Contains($"size: {f.Size}", manifest, StringComparison.Ordinal);
            }
        }
    }

    [Theory]
    [MemberData(nameof(RenderCaseIds))]
    public void ChatTemplateSeamRendersTheTypedText(string id)
    {
        var c = Qwen35TemplateCorpus.Get(id);
        Qwen35Message[] messages = c.ToTypedMessages().Messages!;
        ChatTurn[] turns = [.. messages.Select(m => m.ToChatTurn())];
        ToolDefinition[] tools = [.. c.TypedTools.Select(t => t.ToDefinition())];
        IChatTemplate template = Qwen35ChatTemplate.Instance;
        Assert.Equal(Qwen35ChatTemplate.TemplateSha256, template.TemplateSha256);
        Assert.True(template.SupportsThinking);
        Assert.Equal(c.Rendered, template.Render(turns, tools, c.EnableThinking));
        Assert.Equal(messages, turns.Select(Qwen35Message.FromChatTurn), new MessageComparer());
    }

    [Fact]
    public void ChatTemplateSeamRejectsAsTheTypedRenderer()
    {
        // An empty conversation is an ArgumentException under IChatTemplate's contract; the typed renderer throws its own type.
        Assert.Throws<ArgumentException>(() => Qwen35ChatTemplate.Instance.Render([]));
        Assert.Throws<Qwen35TemplateException>(() => Qwen35ChatTemplate.Render([]));
        Assert.Throws<Qwen35TemplateException>(() => Qwen35ChatTemplate.Instance.Render([ChatTurn.User("a"), ChatTurn.System("late")]));
        Assert.Throws<ArgumentException>(() => Qwen35ChatTemplate.Instance.Render([ChatTurn.User("a")], [ToolDefinition.Function("bad<name")]));
    }

    [Fact]
    public void ToolCallParserSeamGivesTheTypedParts()
    {
        var schema = JsonDocument.Parse("""{"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}""").RootElement;
        ToolDefinition[] tools = [ToolDefinition.Function("get_weather", "Weather.", schema)];
        const string completion = "Let me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";

        Qwen35ParsedOutput typed = Qwen35ToolCallParser.Parse(completion, [.. tools.Select(Qwen35Tool.FromDefinition)], enableThinking: false);
        ParsedCompletion whole = ParsedCompletion.Parse(Qwen35ToolCallParser.Create(tools, enableThinking: false), completion);
        Assert.Equal(Describe(typed.ToParsedCompletion()), Describe(whole));
        Assert.Equal("Let me check.", whole.Text);
        ToolCall call = Assert.Single(whole.ToolCalls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("""{"city":"Paris","days":3}""", call.Arguments.GetRawText());

        // Streamed one character at a time through the interface: the same parts once adjacent text is merged.
        IToolCallParser streaming = Qwen35ToolCallParser.Create(tools, enableThinking: false);
        var parts = new List<CompletionPart>();
        foreach (char ch in completion)
        {
            parts.AddRange(streaming.Append(ch.ToString()));
        }

        parts.AddRange(streaming.Complete());
        Assert.Equal(Describe(whole), Describe(new ParsedCompletion(parts)));
    }

    private static string Describe(ParsedCompletion parsed) =>
        string.Join("|", Merge(parsed.Parts).Select(p => p.Text ?? $"call:{p.ToolCall!.Name}:{p.ToolCall.Arguments.GetRawText()}"));

    private static List<CompletionPart> Merge(IEnumerable<CompletionPart> parts)
    {
        var merged = new List<CompletionPart>();
        foreach (var p in parts)
        {
            if (p.Text is { } t && merged.Count > 0 && merged[^1].Text is { } prev)
            {
                merged[^1] = CompletionPart.FromText(prev + t);
            }
            else if (p.Text is not { Length: 0 })
            {
                merged.Add(p);
            }
        }

        return merged;
    }

    private sealed class MessageComparer : IEqualityComparer<Qwen35Message>
    {
        public bool Equals(Qwen35Message? x, Qwen35Message? y) =>
            x!.Role == y!.Role && x.Content == y.Content && x.ReasoningContent == y.ReasoningContent &&
            (x.ToolCalls ?? []).Select(c => (c.Name, c.Arguments.ValueKind == JsonValueKind.Undefined ? null : c.Arguments.GetRawText()))
                .SequenceEqual((y.ToolCalls ?? []).Select(c => (c.Name, c.Arguments.ValueKind == JsonValueKind.Undefined ? null : c.Arguments.GetRawText())));

        public int GetHashCode(Qwen35Message obj) => HashCode.Combine(obj.Role, obj.Content);
    }
}
