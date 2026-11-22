namespace Starlings.Models.Whisper.Tests;

/// <summary>
/// Language detection (<see cref="WhisperModel.DetectLanguageToken"/>, <see cref="WhisperModel.IdToLanguage"/>): the
/// language-token range is contiguous and matches the model's language map, and the detection step masks every
/// non-language token before the argmax. Weight-free; runs in CI.
/// </summary>
public sealed class DetectLanguageTests
{
    [Fact]
    public void TheLanguageRangeIsTheModelsLanguageTokens()
    {
        Assert.Equal(WhisperModel.English, WhisperModel.LanguageIdFirst);
        Assert.Equal(WhisperModel.LanguageIdLast - WhisperModel.LanguageIdFirst + 1, WhisperModel.LanguageToId.Count);
        Assert.Equal(100, WhisperModel.LanguageToId.Count);

        foreach (var (code, id) in WhisperModel.LanguageToId)
        {
            Assert.InRange(id, WhisperModel.LanguageIdFirst, WhisperModel.LanguageIdLast);
        }
    }

    [Fact]
    public void IdToLanguageIsTheInverseOfLanguageToId()
    {
        Assert.Equal(WhisperModel.LanguageToId.Count, WhisperModel.IdToLanguage.Count);

        foreach (var (code, id) in WhisperModel.LanguageToId)
        {
            Assert.Equal(code, WhisperModel.IdToLanguage[id]);
        }
    }

    [Fact]
    public void DetectLanguageTokenMasksEveryNonLanguageToken()
    {
        // A non-language token dominates the raw logits; the detector must still return the best language token.
        var logits = new float[WhisperModel.VocabSize];
        logits[0] = 999f;                                   // a token far outside the language range
        logits[WhisperModel.LanguageIdFirst] = 5f;          // <|en|>
        logits[WhisperModel.LanguageToId["de"]] = 7f;       // <|de|>
        logits[WhisperModel.LanguageIdLast] = 6f;           // <|yue|>

        Assert.Equal(WhisperModel.LanguageToId["de"], WhisperModel.DetectLanguageToken(logits));
    }

    [Fact]
    public void DetectLanguageTokenReturnsTheMaxLanguageToken()
    {
        var logits = new float[WhisperModel.VocabSize];
        logits[WhisperModel.LanguageToId["ja"]] = 3f;
        logits[WhisperModel.LanguageToId["fr"]] = 9f;

        Assert.Equal(WhisperModel.LanguageToId["fr"], WhisperModel.DetectLanguageToken(logits));
    }

    [Fact]
    public void DetectLanguageTokenRejectsAWrongLength()
    {
        Assert.Throws<ArgumentException>(() => WhisperModel.DetectLanguageToken(new float[WhisperModel.VocabSize - 1]));
    }

    [Fact]
    public void DetectLanguageTokenReturnsAValidCode()
    {
        var logits = new float[WhisperModel.VocabSize];
        logits[WhisperModel.LanguageToId["en"]] = 1f;

        int id = WhisperModel.DetectLanguageToken(logits);

        Assert.True(WhisperModel.IdToLanguage.ContainsKey(id));
        Assert.Equal("en", WhisperModel.IdToLanguage[id]);
    }
}
