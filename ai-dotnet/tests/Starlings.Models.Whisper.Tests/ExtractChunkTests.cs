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
    public void SkippingSilentFramesIsBitIdenticalToTheFullDft()
    {
        // A short clip (mostly zero padding) and a clip with exact digital silence in the middle. In both the clamp to
        // max - 8 floors the silent frames, so two more cases keep their value visible: an all-zero waveform, and a very
        // quiet clip whose loudest log-mel value lies above -10 but less than 8 above it, so the floor stays below -10.
        var shortClip = new double[WhisperMelSpectrogram.MaxSamples];
        Array.Copy(Tone(26_239).Select(value => (double)value).ToArray(), shortClip, 26_239);
        var gapped = Tone(WhisperMelSpectrogram.MaxSamples).Select(value => (double)value).ToArray();
        Array.Clear(gapped, 100_000, 64_000);
        var silence = new double[WhisperMelSpectrogram.MaxSamples];
        var quiet = shortClip.Select(value => value * 4e-5).ToArray();

        foreach (double[] waveform in new[] { shortClip, gapped, silence, quiet })
        {
            Assert.Equal(
                WhisperMelSpectrogram.Extract(waveform, skipSilentFrames: false),
                WhisperMelSpectrogram.Extract(waveform, skipSilentFrames: true));
        }

        // The quiet clip's padding frames really are unclamped: they hold (-10 + 4) / 4, not the floor.
        float[] quietMel = WhisperMelSpectrogram.Extract(quiet, skipSilentFrames: true);
        Assert.Equal(-1.5f, quietMel[WhisperMelSpectrogram.NumFrames - 1]);
        Assert.InRange(quietMel.Max(), -1.4f, 0.5f);
    }

    [Theory]
    [InlineData(400_000)]
    [InlineData(WhisperMelSpectrogram.MaxSamples + 1)]
    public void ExtractRequiresExactlyThirtySeconds(int samples)
    {
        Assert.Throws<ArgumentException>(() => WhisperMelSpectrogram.Extract(new double[samples]));
        Assert.Throws<ArgumentException>(() => WhisperMelSpectrogram.Extract(new float[samples]));
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
