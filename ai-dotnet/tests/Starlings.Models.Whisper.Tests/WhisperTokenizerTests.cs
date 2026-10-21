using Starlings.TestSupport;

namespace Starlings.Models.Whisper.Tests;

/// <summary>Tokenizer decode through the owned whisper-bpe profile.</summary>
/// <remarks>The tokenizer lives in the snapshot directory (not Git); the test skips when it is absent.</remarks>
public sealed class WhisperTokenizerTests
{
    [Fact]
    public void DecodeSkipsSpecialTokens()
    {
        var (path, reason) = WhisperTestAssets.FindGoldenFile("tokenizer.json");
        Assert.SkipWhen(path is null, reason);

        var tokenizer = WhisperTokenizer.FromFile(path!);
        // Encode of "Hello world" with the whisper-bpe profile: [<|startoftranscript|>, <|notimestamps|>, Hello, world, <|endoftext|>].
        string text = tokenizer.Decode([50258, 50364, 15947, 1002, 50257]);
        Assert.Equal("Hello world", text);
    }
}
