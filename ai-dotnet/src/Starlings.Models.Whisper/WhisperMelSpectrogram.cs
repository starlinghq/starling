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
/// 30 s (480,000 samples) -> 3000 frames. The encoder expects [128, 3000].
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

    private static readonly (double Cos, double Sin)[][] Twiddles = BuildTwiddles();

    /// <summary>Log-mel spectrogram of one 16 kHz waveform (480,000 samples), shape [128, 3000].</summary>
    public static float[] Extract(double[] waveform) => Extract(waveform, ComputeHannWindow(), ComputeMelFilters());

    /// <summary>Log-mel spectrogram of one float32 16 kHz waveform (480,000 samples), shape [128, 3000].</summary>
    public static float[] Extract(float[] waveform)
    {
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

    private static float[] Extract(double[] waveform, double[] window, double[,] melFilters)
    {
        int n = waveform.Length;
        int pad = Nfft / 2;

        // center pad, reflect (numpy np.pad mode="reflect").
        double[] padded = new double[n + 2 * pad];
        Array.Copy(waveform, 0, padded, pad, n);
        for (int i = 0; i < pad; i++)
        {
            padded[i] = waveform[pad - i];
            padded[n + pad + i] = waveform[n - 2 - i];
        }

        int frameCount = 1 + (padded.Length - Nfft) / HopLength;   // 3001, then last dropped

        // power spectrogram [frameCount, 201], then mel [128, frameCount].
        var mel = new double[NumMelBins * frameCount];
        double[] buf = new double[Nfft];
        double[] power = new double[NumFrequencyBins];

        for (int t = 0; t < frameCount; t++)
        {
            int offset = t * HopLength;
            for (int j = 0; j < Nfft; j++)
            {
                buf[j] = padded[offset + j] * window[j];
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

                mel[m * frameCount + t] = Math.Log10(Math.Max(1e-10, acc));
            }
        }

        // drop the last frame ([:, :-1]), clamp to max - 8, then (x + 4) / 4.
        double max = double.NegativeInfinity;
        for (int m = 0; m < NumMelBins; m++)
        {
            for (int t = 0; t < NumFrames; t++)
            {
                double v = mel[m * frameCount + t];
                if (v > max)
                {
                    max = v;
                }
            }
        }

        double floor = max - 8.0;
        float[] result = new float[NumMelBins * NumFrames];
        for (int m = 0; m < NumMelBins; m++)
        {
            for (int t = 0; t < NumFrames; t++)
            {
                double v = mel[m * frameCount + t];
                if (v < floor)
                {
                    v = floor;
                }

                result[m * NumFrames + t] = (float)((v + 4.0) / 4.0);
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
