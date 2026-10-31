namespace Starlings.Models.Whisper.Tests;

/// <summary>
/// The long-form spectrogram (<see cref="WhisperMelSpectrogram.ExtractLongForm"/>): one STFT over the whole clip,
/// samples / 160 frames, the clamp to max - 8 over the whole clip. Weight-free; runs in CI.
/// </summary>
public sealed class LongFormMelTests
{
    private static float[] Tone(int samples, float amplitude = 0.25f)
    {
        var waveform = new float[samples];
        for (int i = 0; i < samples; i++)
        {
            waveform[i] = amplitude * MathF.Sin(2 * MathF.PI * 440 * i / WhisperMelSpectrogram.SampleRate);
        }

        return waveform;
    }

    [Theory]
    [InlineData(WhisperMelSpectrogram.MaxSamples + 1, 3000)]
    [InlineData(WhisperMelSpectrogram.MaxSamples + 159, 3000)]
    [InlineData(WhisperMelSpectrogram.MaxSamples + 160, 3001)]
    [InlineData(551_840, 3449)]
    public void FramesAreSamplesOverTheHop(int samples, int frames)
    {
        float[] mel = WhisperMelSpectrogram.ExtractLongForm(Tone(samples), out int actual, TestContext.Current.CancellationToken);

        Assert.Equal(frames, actual);
        Assert.Equal(WhisperMelSpectrogram.NumMelBins * frames, mel.Length);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(WhisperMelSpectrogram.MaxSamples)]
    public void ThirtySecondsOrLessIsRefused(int samples)
    {
        Assert.Throws<ArgumentException>(() => WhisperMelSpectrogram.ExtractLongForm(new float[samples], out _, TestContext.Current.CancellationToken));
    }

    [Fact]
    public void TheFirstWindowMatchesTheChunkAndTheClampIsOverTheWholeClip()
    {
        // 30 s of tone, then 10 s of digital silence. Frames whose 400-sample window lies inside the first 30 s equal the
        // single-chunk spectrogram (the loudest value is in both); every frame of the silence holds the whole clip's floor.
        const int extra = 160_000;
        float[] tone = Tone(WhisperMelSpectrogram.MaxSamples);
        var clip = new float[WhisperMelSpectrogram.MaxSamples + extra];
        tone.CopyTo(clip, 0);

        float[] longMel = WhisperMelSpectrogram.ExtractLongForm(clip, out int frames, TestContext.Current.CancellationToken);
        float[] chunk = WhisperMelSpectrogram.Extract(tone);

        Assert.Equal(WhisperMelSpectrogram.NumFrames + extra / WhisperMelSpectrogram.HopLength, frames);
        float floor = longMel.Min();
        for (int m = 0; m < WhisperMelSpectrogram.NumMelBins; m++)
        {
            for (int t = 0; t <= 2998; t++)
            {
                Assert.Equal(chunk[m * WhisperMelSpectrogram.NumFrames + t], longMel[m * frames + t]);
            }

            for (int t = 3002; t < frames; t++)
            {
                Assert.Equal(floor, longMel[m * frames + t]);
            }
        }
    }

    [Fact]
    public void AQuietTailIsClampedAgainstTheLoudStart()
    {
        // The max - 8 clamp is taken over the whole clip, not per window: a tail 10^5 times quieter than the start sits
        // below the start's floor, so it is clamped there, while on its own it would keep its shape.
        const int extra = 160_000;
        var clip = new float[WhisperMelSpectrogram.MaxSamples + extra];
        Tone(WhisperMelSpectrogram.MaxSamples).CopyTo(clip, 0);
        Tone(extra, 0.25e-5f).CopyTo(clip, WhisperMelSpectrogram.MaxSamples);

        float[] mel = WhisperMelSpectrogram.ExtractLongForm(clip, out int frames, TestContext.Current.CancellationToken);
        float floor = mel.Min();

        Assert.All(Enumerable.Range(3002, frames - 3002), t => Assert.Equal(floor, mel[t]));
    }

    [Fact]
    public void ACancelledTokenStopsTheSpectrogram()
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();

        Assert.Throws<OperationCanceledException>(
            () => WhisperMelSpectrogram.ExtractLongForm(Tone(WhisperMelSpectrogram.MaxSamples * 2), out _, cancellation.Token));
    }

    [Theory]
    [InlineData(WhisperMelSpectrogram.MaxSamples, false)]
    [InlineData(WhisperMelSpectrogram.MaxSamples + 1, false)]
    [InlineData(WhisperMelSpectrogram.MaxSamples + 159, false)]
    [InlineData(WhisperMelSpectrogram.MaxSamples + 160, true)]
    public void LongFormStartsPastOneWindowOfFrames(int samples, bool longForm)
    {
        // transformers: is_shortform = frames <= 3000, and samples / 160 frames stay 3000 up to 480,159 samples.
        Assert.Equal(longForm, WhisperTranscriber.IsLongForm(samples));
    }
}
