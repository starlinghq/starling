using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.EmbeddingGemma2.Tests;

/// <summary>
/// One single-request user of the default multimodal embedder: output digest, request latency, managed allocation per
/// request and peak resident memory (docs/evidence/eg2-image-loaded-soak/). Public API only, so the same test runs on an
/// earlier commit for a before/after comparison. Reports; asserts only that the run completes.
/// </summary>
[Collection(ProcessMemoryTestGroup.Name)]
public class RequestMeasurementTests(ITestOutputHelper output)
{
    [Fact]
    public void SingleRequestUserDigestLatencyAndPeakMemory()
    {
        var (files, reason) = EmbeddingGemma2ImageTestAssets.FindImageModel();
        Assert.SkipWhen(files is null, reason);
        var ct = TestContext.Current.CancellationToken;
        var inputs = MeasurementInputs.Create();
        Settle();
        using var embedder = EmbeddingGemma2MultimodalEmbedder.Create(files!, new EmbeddingGemma2MultimodalOptions
        {
            Session = new OnnxSessionOptions { ComputeModelSha256 = false },
            VerifyModelSha256 = false,
        });
        Settle();
        double loadedMb = Status("VmRSS:");
        bool reset = ResetPeak();

        using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        const int Passes = 3;
        var ms = new double[inputs.Length, Passes];
        long allocatedFirst = 0, allocatedLater = 0;
        for (int pass = 0; pass < Passes; pass++)
        {
            long before = GC.GetTotalAllocatedBytes(precise: true);
            for (int i = 0; i < inputs.Length; i++)
            {
                long t = Stopwatch.GetTimestamp();
                var e = embedder.Embed(inputs[i], ct);
                ms[i, pass] = Stopwatch.GetElapsedTime(t).TotalMilliseconds;
                if (pass == 0)
                {
                    sha.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(e.AsSpan()));
                }
            }

            long allocated = GC.GetTotalAllocatedBytes(precise: true) - before;
            if (pass == 0)
            {
                allocatedFirst = allocated;
            }
            else
            {
                allocatedLater += allocated / (Passes - 1);
            }
        }

        double peakMb = Status("VmHWM:");
        Settle();
        double afterMb = Status("VmRSS:");
        output.WriteLine($"embedding digest {Convert.ToHexStringLower(sha.GetHashAndReset())}");
        output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"RSS loaded {loadedMb:F0} MiB, peak during requests {peakMb:F0} MiB (peak reset after load: {reset}), after requests and GC {afterMb:F0} MiB"));
        output.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"managed allocation per pass of {inputs.Length} requests: first {allocatedFirst / 1048576.0:F1} MiB, later {allocatedLater / 1048576.0:F1} MiB"));
        for (int i = 0; i < inputs.Length; i++)
        {
            var row = Enumerable.Range(1, Passes - 1).Select(p => ms[i, p]).Order().ToArray();
            output.WriteLine(string.Create(CultureInfo.InvariantCulture, $"input {i}: first {ms[i, 0]:F0} ms, later {string.Join(" ", row.Select(x => x.ToString("F0", CultureInfo.InvariantCulture)))} ms"));
        }
    }

    private static void Settle()
    {
        for (int i = 0; i < 2; i++)
        {
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: true);
            GC.WaitForPendingFinalizers();
        }
    }

    /// <summary>Resets the process's peak resident memory (Linux: <c>5</c> to <c>/proc/self/clear_refs</c>).</summary>
    private static bool ResetPeak()
    {
        try
        {
            File.WriteAllText("/proc/self/clear_refs", "5");
            return true;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    private static double Status(string field)
    {
        foreach (string line in File.ReadLines("/proc/self/status"))
        {
            if (line.StartsWith(field, StringComparison.Ordinal))
            {
                return double.Parse(line[field.Length..].Trim().Split(' ')[0], CultureInfo.InvariantCulture) / 1024.0;
            }
        }

        return double.NaN;
    }
}

/// <summary>
/// Inputs of the measurements: images that resize up, strongly down, only horizontally and not at all at the default
/// budget; an image with text; two images in one input; and two videos.
/// </summary>
internal static class MeasurementInputs
{
    public static EmbeddingGemma2MultimodalInput[] Create()
    {
        var (sameW, sameH) = EmbeddingGemma2ImageProcessor.GetTargetSize(768, 768, EmbeddingGemma2ImageProcessor.DefaultMaxSoftTokens);
        (int W, int H) horizontalOnly = (0, 0);
        for (int h = 200; h < 1000 && horizontalOnly.W == 0; h++)
        {
            for (int w = 200; w < 1000; w += 7)
            {
                var (tw, th) = EmbeddingGemma2ImageProcessor.GetTargetSize(w, h, EmbeddingGemma2ImageProcessor.DefaultMaxSoftTokens);
                if (th == h && tw != w)
                {
                    horizontalOnly = (w, h);
                    break;
                }
            }
        }

        if (horizontalOnly.W == 0)
        {
            throw new InvalidOperationException("no size resizes horizontally only");
        }

        return
        [
            EmbeddingGemma2MultimodalInput.FromImage(Image(150, 110, 1)),
            EmbeddingGemma2MultimodalInput.FromImage(Image(2400, 1600, 2)),
            EmbeddingGemma2MultimodalInput.FromImage(Image(horizontalOnly.W, horizontalOnly.H, 3)),
            EmbeddingGemma2MultimodalInput.FromImage(Image(sameW, sameH, 4)),
            EmbeddingGemma2MultimodalInput.Create("a picture of", Image(300, 500, 5)),
            EmbeddingGemma2MultimodalInput.Create(Image(97, 233, 6), "and", Image(640, 480, 7)),
            EmbeddingGemma2MultimodalInput.FromVideo(Clip(0, 2)),
            EmbeddingGemma2MultimodalInput.FromVideo(Clip(1, 3)),
        ];
    }

    private static EmbeddingGemma2Image Image(int width, int height, int seed)
    {
        var rgb = new byte[width * height * 3];
        var random = new Random(seed);
        random.NextBytes(rgb);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int o = ((y * width) + x) * 3;
                rgb[o] = (byte)((rgb[o] >> 2) + (x * 191 / width));
                rgb[o + 1] = (byte)((rgb[o + 1] >> 2) + (y * 191 / height));
            }
        }

        return EmbeddingGemma2Image.FromRgb24(width, height, rgb);
    }

    /// <summary>A synthetic decoded clip (gradients and a moving square) at 4 frames per second.</summary>
    private static EmbeddingGemma2Video Clip(int input, int seconds)
    {
        const int fps = 4;
        int width = 96 + (input * 37 % 160);
        int height = 80 + (input * 53 % 144);
        var frames = new EmbeddingGemma2Image[fps * seconds];
        for (int index = 0; index < frames.Length; index++)
        {
            var rgb = new byte[width * height * 3];
            int side = Math.Min(width, height) / 3;
            int x0 = index * 7 % (width - side);
            int y0 = index * 5 % (height - side);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int o = ((y * width) + x) * 3;
                    bool square = x >= x0 && x < x0 + side && y >= y0 && y < y0 + side;
                    rgb[o] = square ? (byte)250 : (byte)((x * 255 / width) ^ (input * 17));
                    rgb[o + 1] = square ? (byte)250 : (byte)(y * 255 / height);
                    rgb[o + 2] = (byte)(((x / (4 + input)) + (y / 8) + index) % 2 == 0 ? 200 : 40);
                }
            }

            frames[index] = EmbeddingGemma2Image.FromRgb24(width, height, rgb);
        }

        return EmbeddingGemma2Video.FromDecodedFrames(frames, fps, seconds);
    }
}
