namespace Starlings.Models.Whisper.Tests;

/// <summary>30 s chunking of the public audio path: shorter clips are zero-padded like WhisperFeatureExtractor, longer ones refused.</summary>
public sealed class ExtractChunkTests
{
    private static float[] Tone(int samples)
    {
        var waveform = new float[samples];
        for (int i = 0; i < samples; i++)
        {
            waveform[i] = 0.25f * MathF.Sin(2 * MathF.PI * 440 * i / WhisperMelSpectrogram.SampleRate);
        }

        return waveform;
    }

    [Fact]
    public void ShortClipEqualsTheExplicitlyZeroPaddedChunk()
    {
        float[] clip = Tone(16_000);
        var padded = new float[WhisperMelSpectrogram.MaxSamples];
        clip.CopyTo(padded, 0);

        float[] actual = WhisperMelSpectrogram.ExtractChunk(clip);

        Assert.Equal(WhisperMelSpectrogram.NumMelBins * WhisperMelSpectrogram.NumFrames, actual.Length);
        Assert.Equal(WhisperMelSpectrogram.Extract(padded), actual);
    }

    [Fact]
    public void FullChunkIsUnchanged()
    {
        float[] clip = Tone(WhisperMelSpectrogram.MaxSamples);

        Assert.Equal(WhisperMelSpectrogram.Extract(clip), WhisperMelSpectrogram.ExtractChunk(clip));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(WhisperMelSpectrogram.MaxSamples + 1)]
    public void EmptyOrLongerThanThirtySecondsIsRefused(int samples)
    {
        Assert.Throws<ArgumentException>(() => WhisperMelSpectrogram.ExtractChunk(new float[samples]));
    }
}
