using System.Reflection;
using System.Text.Json;
using Starlings.Generation;
using Starlings.Models.QwenChat;

namespace Starlings.Models.QwenChat.Tests;

/// <summary>
/// The model definition package (Starlings.Models.QwenChat) without a runtime: its dependency boundary, the checkpoints
/// the bundle profiles delegate to, and the backend-neutral seams, which must give the same text and calls as the typed API.
/// </summary>
public sealed class DefinitionTests
{
    public static TheoryData<string, string> RenderCases()
    {
        var data = new TheoryData<string, string>();
        foreach (string model in new[] { TemplateCorpus.Qwen3, TemplateCorpus.Qwen25 })
        {
            foreach (var c in TemplateCorpus.Load(model).Where(c => c.Owned == "render"))
            {
                data.Add(model, c.Id);
            }
        }

        return data;
    }

    [Fact]
    public void DefinitionAssemblyReferencesNoOnnxRuntime()
    {
        AssemblyName[] references = typeof(QwenChatTemplate).Assembly.GetReferencedAssemblies();
        Assert.DoesNotContain(references, r => r.Name!.StartsWith("Microsoft.ML.OnnxRuntime", StringComparison.Ordinal));
        Assert.DoesNotContain(references, r => r.Name is "Starlings.Generation.OnnxGenAI" or "Starlings.Runtime.Onnx" or "Starlings.Acquisition");
        Assert.Equal("Starlings.Models.QwenChat.OnnxGenAI", typeof(QwenChatModel).Assembly.GetName().Name);
        Assert.Equal(typeof(QwenChatTemplate).Namespace, typeof(QwenChatModel).Namespace);
    }

    [Fact]
    public void ProfilesDelegateToTheirCheckpoints()
    {
        foreach (var (profile, checkpoint) in new[] { (QwenChatProfile.Qwen3, QwenChatCheckpoint.Qwen3), (QwenChatProfile.Qwen25, QwenChatCheckpoint.Qwen25) })
        {
            Assert.Same(checkpoint, profile.Checkpoint);
            QwenChatCheckpoint converted = profile;
            Assert.Same(checkpoint, converted);
            Assert.Equal(
                (checkpoint.Repository, checkpoint.Revision, checkpoint.ModelType, checkpoint.TokenizerSha256, checkpoint.TemplateSha256),
                (profile.Repository, profile.Revision, profile.ModelType, profile.TokenizerSha256, profile.TemplateSha256));
            Assert.Equal(checkpoint.EosTokenIds, profile.EosTokenIds);
            Assert.Equal(
                (checkpoint.HiddenSize, checkpoint.Layers, checkpoint.AttentionHeads, checkpoint.KeyValueHeads, checkpoint.HeadSize, checkpoint.ContextLength),
                (profile.HiddenSize, profile.Layers, profile.AttentionHeads, profile.KeyValueHeads, profile.HeadSize, profile.ContextLength));
            Assert.StartsWith(checkpoint.Id + "/", profile.Id, StringComparison.Ordinal);
            Assert.Same(checkpoint, QwenChatCheckpoint.FromId(checkpoint.Id));

            ModelProfile p = checkpoint.Profile;
            Assert.Equal((checkpoint.Id, checkpoint.Repository, checkpoint.Revision), (p.Id, p.Repository, p.Revision));
            Assert.Null(p.TokenizerProfile);
            Assert.Equal((checkpoint.TokenizerSha256, checkpoint.TemplateSha256), (p.TokenizerSha256, p.ChatTemplateSha256));
            Assert.Equal([151643, 151645], p.EosTokenIds);
            Assert.Equal((151936, checkpoint.HiddenSize, checkpoint.Layers, checkpoint.HeadSize, checkpoint.ContextLength),
                (p.VocabularySize, p.HiddenSize, p.Layers, p.HeadSize, p.ContextLength));
        }

        Assert.Equal((1024, 28, 16, 8, 128, 40960), (QwenChatCheckpoint.Qwen3.HiddenSize, QwenChatCheckpoint.Qwen3.Layers,
            QwenChatCheckpoint.Qwen3.AttentionHeads, QwenChatCheckpoint.Qwen3.KeyValueHeads, QwenChatCheckpoint.Qwen3.HeadSize, QwenChatCheckpoint.Qwen3.ContextLength));
        Assert.Equal((896, 24, 14, 2, 64, 32768), (QwenChatCheckpoint.Qwen25.HiddenSize, QwenChatCheckpoint.Qwen25.Layers,
            QwenChatCheckpoint.Qwen25.AttentionHeads, QwenChatCheckpoint.Qwen25.KeyValueHeads, QwenChatCheckpoint.Qwen25.HeadSize, QwenChatCheckpoint.Qwen25.ContextLength));
        Assert.Throws<NotSupportedException>(() => QwenChatCheckpoint.FromId("qwen3-0.6b/ortgenai-fp32-cpu"));
    }

    [Theory]
    [MemberData(nameof(RenderCases))]
    public void ChatTemplateSeamRendersTheTypedText(string model, string id)
    {
        TemplateCorpusCase c = TemplateCorpus.Get(model, id);
        QwenChatProfile profile = TemplateCorpus.Profile(model);
        QwenChatMessage[] messages = c.ToTypedMessages(profile).Messages!;
        IChatTemplate template = profile.Checkpoint.ChatTemplate;
        Assert.Same(template, QwenChatTemplate.For(profile.Checkpoint));
        Assert.Equal(profile.TemplateSha256, template.TemplateSha256);
        Assert.False(template.SupportsThinking);
        ToolDefinition[]? tools = c.TypedTools?.Select(t => t.ToDefinition()).ToArray();
        Assert.Equal(c.Rendered, template.Render([.. messages.Select(m => m.ToChatTurn())], tools));
    }

    [Fact]
    public void ChatTemplateSeamRejectsThinking()
    {
        Assert.Throws<NotSupportedException>(() => QwenChatCheckpoint.Qwen3.ChatTemplate.Render([ChatTurn.User("a")], enableThinking: true));
        Assert.Throws<ArgumentException>(() => QwenChatCheckpoint.Qwen25.ChatTemplate.Render([ChatTurn.User("a")], [ToolDefinition.Function("bad\"name")]));
    }

    [Fact]
    public void ToolCallParserSeamGivesTheTypedParts()
    {
        const string completion = "Checking.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 3}}\n</tool_call>";
        QwenChatParsedOutput typed = QwenChatToolCallParser.Parse(completion);
        ParsedCompletion neutral = ParsedCompletion.Parse(new QwenChatToolCallParser(), completion);
        Assert.Equal(typed.Text, neutral.Text);
        ToolCall call = Assert.Single(neutral.ToolCalls);
        QwenChatToolCall typedCall = Assert.Single(typed.ToolCalls);
        Assert.Equal((typedCall.Name, typedCall.Arguments.GetRawText()), (call.Name, call.Arguments.GetRawText()));
        Assert.Equal(typed.ToParsedCompletion().Parts.Count, neutral.Parts.Count);
        Assert.Equal(QwenChatMessage.FromChatTurn(neutral.ToAssistantTurn()).ToolCalls!.Single().Name, typed.ToAssistantMessage().ToolCalls!.Single().Name);

        var json = JsonDocument.Parse("""{"type": "function", "function": {"name": "f"}}""").RootElement;
        Assert.Equal(QwenChatTool.FromJson(json).Json.GetRawText(), QwenChatTool.FromDefinition(ToolDefinition.FromJson(json)).Json.GetRawText());
    }
}
