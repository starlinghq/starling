namespace Starlings.Models.Whisper.Tests;

/// <summary>Tokenizer decode through the owned whisper-bpe profile.</summary>
public sealed class WhisperTokenizerTests
{
    private const string Golden = "/home/mark/Projects/starling/.models/snapshots/whisper-large-v3-turbo";

    [Fact(Skip = "Requires Starlings.Tokenizers 0.3.2 (whisper-bpe use_regex); starling pins 0.3.1.")]
    public void DecodeSkipsSpecialTokens()
    {
        var tokenizer = WhisperTokenizer.FromFile($"{Golden}/tokenizer.json");
        // Encode of "Hello world" with the whisper-bpe profile: [<|startoftranscript|>, <|notimestamps|>, Hello, world, <|endoftext|>].
        string text = tokenizer.Decode([50258, 50364, 15947, 1002, 50257]);
        Assert.Equal("Hello world", text);
    }
}
