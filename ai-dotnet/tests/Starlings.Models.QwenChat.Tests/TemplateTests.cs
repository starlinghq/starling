using Starlings.Models.QwenChat;

namespace Starlings.Models.QwenChat.Tests;

public sealed class TemplateTests
{
    [Fact]
    public void Qwen25InsertsDefaultSystemAndPreservesWhitespace()
    {
        string text = QwenChatTemplate.Render(QwenChatProfile.Qwen25, [new(QwenChatRole.User, "  hi\n ")]);
        Assert.Equal("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n" +
            "<|im_start|>user\n  hi\n <|im_end|>\n<|im_start|>assistant\n", text);
    }

    [Fact]
    public void Qwen25PreservesExplicitSystem()
    {
        string text = QwenChatTemplate.Render(QwenChatProfile.Qwen25,
            [new(QwenChatRole.System, "Be brief."), new(QwenChatRole.User, "Hi")]);
        Assert.Equal("<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", text);
    }

    [Fact]
    public void Qwen3DoesNotInsertSystemAndExplicitlyDisablesThinking()
    {
        string text = QwenChatTemplate.Render(QwenChatProfile.Qwen3, [new(QwenChatRole.User, "Hi")]);
        Assert.Equal("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", text);
    }

    [Fact]
    public void Qwen3DropsEarlierAssistantReasoning()
    {
        string text = QwenChatTemplate.Render(QwenChatProfile.Qwen3,
            [new(QwenChatRole.User, "First"), new(QwenChatRole.Assistant, "<think>\nold\n</think>\n\nAnswer"),
             new(QwenChatRole.User, "Next")]);
        Assert.Contains("<|im_start|>assistant\nAnswer<|im_end|>", text, StringComparison.Ordinal);
        Assert.DoesNotContain("old", text, StringComparison.Ordinal);
    }

    [Fact]
    public void Qwen3KeepsExplicitAssistantReasoningAfterLatestUser()
    {
        string text = QwenChatTemplate.Render(QwenChatProfile.Qwen3,
            [new(QwenChatRole.User, "First"), new(QwenChatRole.Assistant, "Answer", "\nreason\n")]);
        Assert.Contains("<|im_start|>assistant\n<think>\nreason\n</think>\n\nAnswer<|im_end|>", text, StringComparison.Ordinal);
    }

    [Fact]
    public void RejectsUnsupportedMessageRolesAndReasoning()
    {
        Assert.Throws<NotSupportedException>(() => QwenChatTemplate.Render(QwenChatProfile.Qwen3,
            [new((QwenChatRole)99, "tool")]));
        Assert.DoesNotContain("reason", QwenChatTemplate.Render(QwenChatProfile.Qwen25,
            [new(QwenChatRole.Assistant, "Answer", "reason")]), StringComparison.Ordinal);
        Assert.Contains("<|im_start|>system\nlate<|im_end|>", QwenChatTemplate.Render(QwenChatProfile.Qwen3,
            [new(QwenChatRole.User, "A"), new(QwenChatRole.System, "late")]), StringComparison.Ordinal);
    }

    [Fact]
    public void UnknownProfileIsRejected()
    {
        Assert.Throws<NotSupportedException>(() => QwenChatProfile.FromId("qwen3.5-0.8b/ortgenai-fp32x"));
    }
}
