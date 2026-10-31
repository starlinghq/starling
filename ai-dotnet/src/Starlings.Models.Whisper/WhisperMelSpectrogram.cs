// Copyright (c) 2026 StarlingHq. Licensed under the MIT License or the Apache License, Version 2.0, at your option.

namespace Starlings.Models.Whisper;

/// <summary>
/// The Whisper log-mel spectrogram, restated from transformers <c>WhisperFeatureExtractor</c>
/// (numpy path) and <c>audio_utils.spectrogram</c> so the .NET output matches the reference
/// bit for bit. Computes in double (the reference promotes to float64 before the FFT) and
/// returns float32.
/// </summary>
/// <remarks>
/// Fixed geometry for large-v3/turbo: sample rate 16,000, n_fft 400, hop 160, 128 mel bins,
/// 30 s (480,000 samples) -> 3000 frames. The encoder expects [128, 3000]; a long-form spectrogram
/// (<see cref="ExtractLongForm"/>) has samples / 160 frames and is decoded in 3000-frame windows.
/// </remarks>
public static class WhisperMelSpectrogram
{
    /// <summary>Audio sample rate in Hz.</summary>
    public const int SampleRate = 16_000;
    /// <summary>STFT window length in samples.</summary>
    public const int Nfft = 400;
    /// <summary>STFT hop length in samples.</summary>
    public const int HopLength = 160;
    /// <summary>Number of mel filters.</summary>
    public const int NumMelBins = 128;
    /// <summary>Number of one-sided FFT bins (n_fft / 2 + 1).</summary>
    public const int NumFrequencyBins = Nfft / 2 + 1;   // 201
    /// <summary>Number of frames for a 30 s clip (480000 samples at hop 160).</summary>
    public const int NumFrames = 3000;                  // 480000 samples at hop 160
    /// <summary>Number of samples in one 30 s chunk (the feature extractor's <c>n_samples</c>).</summary>
    public const int MaxSamples = 480_000;

    // The reference's power floor: log10(max(1e-10, power)).
    private const double PowerFloor = 1e-10;

    // A frame whose windowed samples are all zero has zero power in every bin, so its value is exactly this.
    private static readonly double SilentLogMel = Math.Log10(PowerFloor);

    // A naive DFT, not an FFT: it reproduces the reference's float64 spectrogram bit for bit, which an FFT's different
    // summation order would not.
    private static readonly (double Cos, double Sin)[][] Twiddles = BuildTwiddles();
    private static readonly double[] HannWindow = ComputeHannWindow();
    private static readonly double[,] MelFilters = ComputeMelFilters();

    /// <summary>
    /// Log-mel spectrogram of one float32 16 kHz clip of up to 30 s, shape [128, 3000]: the clip is zero-padded on the
    /// right to <see cref="MaxSamples"/>, as <c>WhisperFeatureExtractor</c> pads it. Longer clips are refused rather than
    /// silently truncated; <see cref="ExtractLongForm"/> computes their spectrogram.
    /// </summary>
    /// <exception cref="ArgumentException">The clip is empty or longer than 30 s.</exception>
    public static float[] ExtractChunk(float[] waveform)
    {
        ArgumentNullException.ThrowIfNull(waveform);
        if (waveform.Length is 0 or > MaxSamples)
        {
            throw new ArgumentException(
                $"Audio must hold 1 to {MaxSamples} samples (30 s at 16 kHz), got {waveform.Length}; use ExtractLongForm for longer clips.",
                nameof(waveform));
        }

        var padded = new double[MaxSamples];
        for (int i = 0; i < waveform.Length; i++)
        {
            padded[i] = waveform[i];
        }

        return Extract(padded);
    }

    /// <summary>
    /// Log-mel spectrogram of one float32 16 kHz clip longer than 30 s, over the whole clip as transformers
    /// <c>WhisperFeatureExtractor(audio, truncation=False, padding="longest")</c> computes it for long-form generation:
    /// no padding, one STFT over every sample, the clamp to max - 8 taken over the whole clip. Shape [128, frames] with
    /// <paramref name="frames"/> = samples / 160 (rounded down); long-form decoding slices it into 3000-frame windows.
    /// </summary>
    /// <remarks>
    /// Cost: the naive DFT (kept because it reproduces the reference's float64 spectrogram bit for bit, which an FFT's
    /// different summation order would not) does about 160,000 multiply-adds per frame on one thread: measured at about
    /// 12 s per 10 minutes of audio (about 70 s per hour) on the qualification machine. It allocates about 18 bytes per
    /// sample (the reflect-padded float64 copy, the float64 log-mel and the float32 result: about 1 GB per hour of
    /// audio) on top of the caller's 4-byte samples; the float32 result (3.2 bytes per sample) stays alive for the decode.
    /// <paramref name="cancellationToken"/> is observed every 1024 frames (about 10 s of audio).
    /// </remarks>
    /// <exception cref="ArgumentException">The clip holds 30 s or less (use <see cref="ExtractChunk"/>).</exception>
    /// <exception cref="OperationCanceledException">The token was cancelled during the spectrogram.</exception>
    public static float[] ExtractLongForm(float[] waveform, out int frames, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(waveform);
        if (waveform.Length <= MaxSamples)
        {
            throw new ArgumentException(
                $"Long-form audio must hold more than {MaxSamples} samples (30 s at 16 kHz), got {waveform.Length}; use ExtractChunk.",
                nameof(waveform));
        }

        frames = waveform.Length / HopLength;
        // Reflect-pad straight from the float32 input: no intermediate float64 copy of the clip.
        return Extract(ReflectPad(waveform), HannWindow, MelFilters, skipSilentFrames: true, cancellationToken);
    }

    /// <summary>Log-mel spectrogram of one 16 kHz waveform (exactly 480,000 samples), shape [128, 3000].</summary>
    /// <exception cref="ArgumentException">The waveform is not exactly 480,000 samples; use <see cref="ExtractChunk"/> for shorter clips.</exception>
    public static float[] Extract(double[] waveform) => Extract(waveform, skipSilentFrames: true);

    /// <summary>As <see cref="Extract(double[])"/>; <paramref name="skipSilentFrames"/> false computes every frame's DFT (tests).</summary>
    internal static float[] Extract(double[] waveform, bool skipSilentFrames)
    {
        ArgumentNullException.ThrowIfNull(waveform);
        if (waveform.Length != MaxSamples)
        {
            throw new ArgumentException(
                $"Waveform must hold exactly {MaxSamples} samples, got {waveform.Length}; ExtractChunk pads shorter clips.",
                nameof(waveform));
        }

        return Extract(ReflectPad(waveform), HannWindow, MelFilters, skipSilentFrames, CancellationToken.None);
    }

    /// <summary>Log-mel spectrogram of one float32 16 kHz waveform (exactly 480,000 samples), shape [128, 3000].</summary>
    /// <exception cref="ArgumentException">The waveform is not exactly 480,000 samples; use <see cref="ExtractChunk"/> for shorter clips.</exception>
    public static float[] Extract(float[] waveform)
    {
        ArgumentNullException.ThrowIfNull(waveform);
        var d = new double[waveform.Length];
        for (int i = 0; i < waveform.Length; i++)
        {
            d[i] = waveform[i];
        }

        return Extract(d);
    }

    /// <summary>Slaney triangular mel filterbank, shape [numFreqBins, numMelBins] (201, 128), Slaney area-normalized.</summary>
    public static double[,] ComputeMelFilters(double minFreq = 0.0, double maxFreq = 8000.0)
    {
        double melMin = HertzToMelSlaney(minFreq);
        double melMax = HertzToMelSlaney(maxFreq);

        double[] filterFreqs = new double[NumMelBins + 2];
        for (int i = 0; i < NumMelBins + 2; i++)
        {
            double mel = melMin + (melMax - melMin) * i / (NumMelBins + 1);
            filterFreqs[i] = MelToHertzSlaney(mel);
        }

        double[] fftFreqs = new double[NumFrequencyBins];
        for (int j = 0; j < NumFrequencyBins; j++)
        {
            fftFreqs[j] = (SampleRate / 2.0) * j / (NumFrequencyBins - 1);
        }

        double[] filterDiff = new double[NumMelBins + 1];
        for (int i = 0; i < NumMelBins + 1; i++)
        {
            filterDiff[i] = filterFreqs[i + 1] - filterFreqs[i];
        }

        var melFilters = new double[NumFrequencyBins, NumMelBins];
        for (int j = 0; j < NumFrequencyBins; j++)
        {
            for (int m = 0; m < NumMelBins; m++)
            {
                double down = -(filterFreqs[m] - fftFreqs[j]) / filterDiff[m];
                double up = (filterFreqs[m + 2] - fftFreqs[j]) / filterDiff[m + 1];
                melFilters[j, m] = Math.Max(0.0, Math.Min(down, up));
            }
        }

        // Slaney norm: enorm[m] = 2 / (filterFreqs[m + 2] - filterFreqs[m]).
        for (int m = 0; m < NumMelBins; m++)
        {
            double enorm = 2.0 / (filterFreqs[m + 2] - filterFreqs[m]);
            for (int j = 0; j < NumFrequencyBins; j++)
            {
                melFilters[j, m] *= enorm;
            }
        }

        return melFilters;
    }

    /// <summary>Periodic Hann window of length <see cref="Nfft"/> (400): 0.5 - 0.5*cos(2*pi*n/Nfft).</summary>
    public static double[] ComputeHannWindow()
    {
        var w = new double[Nfft];
        for (int n = 0; n < Nfft; n++)
        {
            w[n] = 0.5 - 0.5 * Math.Cos(2.0 * Math.PI * n / Nfft);
        }

        return w;
    }

    private static double HertzToMelSlaney(double freq)
    {
        double mels = 3.0 * freq / 200.0;
        if (freq >= 1000.0)
        {
            mels = 15.0 + (Math.Log(freq / 1000.0) * 27.0) / Math.Log(6.4);
        }

        return mels;
    }

    private static double MelToHertzSlaney(double mel)
    {
        double freq = 200.0 * mel / 3.0;
        if (mel >= 15.0)
        {
            freq = 1000.0 * Math.Exp((Math.Log(6.4) * (mel - 15.0)) / 27.0);
        }

        return freq;
    }

    /// <summary>Center padding by n_fft / 2 on each side, reflect mode (numpy <c>np.pad(mode="reflect")</c>), in float64.</summary>
    private static double[] ReflectPad(double[] waveform)
    {
        int n = waveform.Length, pad = Nfft / 2;
        double[] padded = new double[n + 2 * pad];
        Array.Copy(waveform, 0, padded, pad, n);
        for (int i = 0; i < pad; i++)
        {
            padded[i] = waveform[pad - i];
            padded[n + pad + i] = waveform[n - 2 - i];
        }

        return padded;
    }

    /// <summary>As <see cref="ReflectPad(double[])"/> from float32 samples (each widened exactly to float64).</summary>
    private static double[] ReflectPad(float[] waveform)
    {
        int n = waveform.Length, pad = Nfft / 2;
        double[] padded = new double[n + 2 * pad];
        for (int i = 0; i < n; i++)
        {
            padded[pad + i] = waveform[i];
        }

        for (int i = 0; i < pad; i++)
        {
            padded[i] = waveform[pad - i];
            padded[n + pad + i] = waveform[n - 2 - i];
        }

        return padded;
    }

    private static float[] Extract(double[] padded, double[] window, double[,] melFilters, bool skipSilentFrames, CancellationToken cancellationToken)
    {
        int frameCount = 1 + (padded.Length - Nfft) / HopLength;   // samples / 160 + 1, then the last is dropped
        int frames = frameCount - 1;

        // power spectrogram [frameCount, 201], then mel [128, frameCount].
        var mel = new double[NumMelBins * frameCount];
        double[] buf = new double[Nfft];
        double[] power = new double[NumFrequencyBins];

        // A frame whose windowed samples are all zero (the zero padding of a short clip, or exact digital silence) has zero
        // power in every bin, so its value is SilentLogMel exactly; skipping its DFT changes no bit of the output.
        for (int t = 0; t < frameCount; t++)
        {
            if ((t & 1023) == 0)
            {
                cancellationToken.ThrowIfCancellationRequested();
            }

            int offset = t * HopLength;
            bool silent = skipSilentFrames;
            for (int j = 0; j < Nfft; j++)
            {
                buf[j] = padded[offset + j] * window[j];
                silent = silent && buf[j] == 0.0;
            }

            if (silent)
            {
                for (int m = 0; m < NumMelBins; m++)
                {
                    mel[m * frameCount + t] = SilentLogMel;
                }

                continue;
            }

            for (int k = 0; k < NumFrequencyBins; k++)
            {
                double re = 0.0, im = 0.0;
                (double Cos, double Sin)[] tw = Twiddles[k];
                for (int j = 0; j < Nfft; j++)
                {
                    re += buf[j] * tw[j].Cos;
                    im += buf[j] * tw[j].Sin;
                }

                power[k] = re * re + im * im;
            }

            for (int m = 0; m < NumMelBins; m++)
            {
                double acc = 0.0;
                for (int k = 0; k < NumFrequencyBins; k++)
                {
                    acc += melFilters[k, m] * power[k];
                }

                mel[m * frameCount + t] = Math.Log10(Math.Max(PowerFloor, acc));
            }
        }

        // drop the last frame ([:, :-1]), clamp to max - 8, then (x + 4) / 4.
        double max = double.NegativeInfinity;
        for (int m = 0; m < NumMelBins; m++)
        {
            for (int t = 0; t < frames; t++)
            {
                double v = mel[m * frameCount + t];
                if (v > max)
                {
                    max = v;
                }
            }
        }

        double floor = max - 8.0;
        float[] result = new float[NumMelBins * frames];
        for (int m = 0; m < NumMelBins; m++)
        {
            for (int t = 0; t < frames; t++)
            {
                double v = mel[m * frameCount + t];
                if (v < floor)
                {
                    v = floor;
                }

                result[m * frames + t] = (float)((v + 4.0) / 4.0);
            }
        }

        return result;
    }

    private static (double Cos, double Sin)[][] BuildTwiddles()
    {
        var tw = new (double, double)[NumFrequencyBins][];
        for (int k = 0; k < NumFrequencyBins; k++)
        {
            tw[k] = new (double, double)[Nfft];
            for (int j = 0; j < Nfft; j++)
            {
                double angle = -2.0 * Math.PI * j * k / Nfft;
                tw[k][j] = (Math.Cos(angle), Math.Sin(angle));
            }
        }

        return tw;
    }
}
