namespace Starlings.Models.Whisper.Tests;

/// <summary>
/// The timestamp rules (transformers <c>WhisperTimeStampLogitsProcessor</c>) and the segment split
/// (<c>_retrieve_segment</c>) on synthetic logits and tokens. Weight-free; runs in CI.
/// </summary>
public sealed class TimestampRulesTests
{
    private const int Begin = WhisperModel.TimestampBegin;

    private static float[] Logits(float text = 0f, float timestamps = 0f)
    {
        var logits = new float[WhisperModel.VocabSize];
        logits.AsSpan(0, Begin).Fill(text);
        logits.AsSpan(Begin).Fill(timestamps);
        return logits;
    }

    private static bool Masked(float[] logits, int id) => float.IsNegativeInfinity(logits[id]);

    [Fact]
    public void TheFirstTokenIsATimestampOfAtMostOneSecond()
    {
        var logits = Logits(text: 5f);
        WhisperModel.ApplyTimestampRules(logits, []);

        Assert.True(Masked(logits, 440));
        Assert.True(Masked(logits, WhisperModel.EndOfText));
        Assert.False(Masked(logits, Begin));
        Assert.False(Masked(logits, Begin + WhisperModel.MaxInitialTimestampIndex));
        Assert.True(Masked(logits, Begin + WhisperModel.MaxInitialTimestampIndex + 1));
        Assert.Equal(Begin, WhisperModel.ArgMax(logits));
    }

    [Fact]
    public void NoTimestampsIsAlwaysMasked()
    {
        var logits = Logits();
        logits[WhisperModel.NoTimestamps] = 100f;
        WhisperModel.ApplyTimestampRules(logits, [Begin, 440]);

        Assert.True(Masked(logits, WhisperModel.NoTimestamps));
    }

    [Fact]
    public void AfterAPairOnlyTextMayFollow()
    {
        // <|0.00|> text <|1.00|><|1.00|>: the pair is closed, so no timestamp may follow.
        var logits = Logits(text: -10f, timestamps: 10f);
        WhisperModel.ApplyTimestampRules(logits, [Begin, 440, Begin + 50, Begin + 50]);

        Assert.True(Masked(logits, Begin + 60));
        Assert.False(Masked(logits, 440));
        Assert.False(Masked(logits, WhisperModel.EndOfText));
    }

    [Fact]
    public void AfterOneTimestampFollowingTextNoTextMayFollow()
    {
        var logits = Logits(text: 10f);
        WhisperModel.ApplyTimestampRules(logits, [Begin, 440, Begin + 50]);

        Assert.True(Masked(logits, 440));
        Assert.True(Masked(logits, WhisperModel.EndOfText - 1));
        // End of text and the special tokens between it and the timestamps stay open.
        Assert.False(Masked(logits, WhisperModel.EndOfText));
        // The closing timestamp may repeat the last one, but not go back.
        Assert.False(Masked(logits, Begin + 50));
        Assert.True(Masked(logits, Begin + 49));
    }

    [Fact]
    public void TimestampsNeverDecreaseAfterText()
    {
        // <|0.00|> text <|1.00|><|1.00|> text: the next timestamp must be later than 1.00.
        var logits = Logits();
        WhisperModel.ApplyTimestampRules(logits, [Begin, 440, Begin + 50, Begin + 50, 1029]);

        Assert.True(Masked(logits, Begin + 50));
        Assert.False(Masked(logits, Begin + 51));
    }

    [Fact]
    public void WhenTimestampsOutweighEveryTextTokenOnlyATimestampMayFollow()
    {
        // 1501 timestamps at 0 against one text token at 5: logsumexp over the timestamps is about log(1501) = 7.3 > 5.
        var logits = Logits(text: float.NegativeInfinity);
        logits[440] = 5f;
        WhisperModel.ApplyTimestampRules(logits, [Begin, 1029]);

        Assert.True(Masked(logits, 440));
        Assert.False(Masked(logits, Begin + 10));
    }

    [Fact]
    public void ATextTokenThatOutweighsAllTimestampsStaysOpen()
    {
        var logits = Logits(text: float.NegativeInfinity);
        logits[440] = 8f;
        WhisperModel.ApplyTimestampRules(logits, [Begin, 1029]);

        Assert.False(Masked(logits, 440));
        Assert.Equal(440, WhisperModel.ArgMax(logits));
    }

    [Fact]
    public void ASingleClosedSegmentCoversTheWindow()
    {
        int[] tokens = [Begin, 440, 1029, Begin + 264];
        var segments = WhisperSegmenter.Split(tokens, seek: 0, seekFrames: 3000, out int advance);

        var segment = Assert.Single(segments);
        Assert.Equal((0.0, 264 * 0.02), (segment.Start, segment.End));
        Assert.Equal(tokens, segment.Ids);
        Assert.Equal(3000, advance);
    }

    [Fact]
    public void PairsSplitSegmentsAndAnOpenEndSeeksToTheLastClosedTimestamp()
    {
        // <|0.00|> a <|4.20|><|5.26|> b <|9.68|><|9.68|> c (no closing timestamp): two segments, then seek to 9.68 s.
        int[] tokens = [Begin, 440, Begin + 210, Begin + 263, 1029, Begin + 484, Begin + 484, 13];
        var segments = WhisperSegmenter.Split(tokens, seek: 0, seekFrames: 3000, out int advance);

        Assert.Equal(2, segments.Count);
        Assert.Equal((0.0, 210 * 0.02), (segments[0].Start, segments[0].End));
        Assert.Equal([Begin, 440, Begin + 210], segments[0].Ids);
        Assert.Equal((263 * 0.02, 484 * 0.02), (segments[1].Start, segments[1].End));
        Assert.Equal([Begin + 263, 1029, Begin + 484, Begin + 484], segments[1].Ids);
        Assert.Equal(484 * 2, advance);
    }

    [Fact]
    public void ALaterWindowIsOffsetBySeek()
    {
        int[] tokens = [Begin, 440, Begin + 100];
        var segment = Assert.Single(WhisperSegmenter.Split(tokens, seek: 968, seekFrames: 2032, out int advance));

        Assert.Equal(968 * 0.02 / 2, segment.Start);
        Assert.Equal(968 * 0.02 / 2 + 100 * 0.02, segment.End);
        Assert.Equal(2032, advance);
    }

    [Theory]
    [InlineData(3000, 1500)]
    [InlineData(30, 14)]
    [InlineData(60, 29)]
    [InlineData(58, 29)]
    public void AWindowWithoutAClosedSegmentEndsWhereTransformersComputesItInFloat32(int seekFrames, int position)
    {
        // No timestamp after the first: the end is the window's end, int(frames * 0.01 / 0.02) in float32 as torch does.
        var segment = Assert.Single(WhisperSegmenter.Split([Begin, 440, 1029], seek: 0, seekFrames, out int advance));

        Assert.Equal(position * 0.02, segment.End);
        Assert.Equal(seekFrames, advance);
    }
}
