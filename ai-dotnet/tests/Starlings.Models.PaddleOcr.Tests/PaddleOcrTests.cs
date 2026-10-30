using Starlings.Acquisition;
using Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp;

namespace Starlings.Models.PaddleOcr.Tests;

/// <summary>Locates the pinned snapshot (<c>.models/snapshots/ppocrv5-mobile</c> with <c>det/</c> and <c>rec/</c>) when present.</summary>
internal static class PaddleOcrTestAssets
{
    public const string SnapshotEnvironmentVariable = "STARLING_PPOCR_SNAPSHOT";

    public static string RepositoryRoot { get; } = FindRepositoryRoot();

    public static (string? Dir, string Reason) FindSnapshot()
    {
        string? configured = Environment.GetEnvironmentVariable(SnapshotEnvironmentVariable);
        string dir = string.IsNullOrEmpty(configured) ? Path.Combine(RepositoryRoot, ".models", "snapshots", "ppocrv5-mobile") : configured;
        return Directory.Exists(Path.Combine(dir, "det")) && Directory.Exists(Path.Combine(dir, "rec"))
            ? (dir, string.Empty)
            : (null, $"{dir} has no det/ and rec/ snapshots. Set {SnapshotEnvironmentVariable}.");
    }

    public static (PaddleOcrModelFiles? Files, string Reason) FindModel()
    {
        var (dir, reason) = FindSnapshot();
        if (dir is null)
        {
            return (null, reason);
        }

        try
        {
            return (PaddleOcrModelAssets.OpenDirectories(Path.Combine(dir, "det"), Path.Combine(dir, "rec"), CacheVerification.Sha256), string.Empty);
        }
        catch (ModelAssetMissingException ex)
        {
            return (null, ex.Message);
        }
    }

    private static string FindRepositoryRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "model-pipeline")) && Directory.Exists(Path.Combine(dir.FullName, "ai-dotnet")))
            {
                return dir.FullName;
            }
        }

        throw new InvalidOperationException("Repository root not found above " + AppContext.BaseDirectory);
    }
}

public sealed class ConfigurationTests
{
    // The pinned detector inference.yml (PaddlePaddle/PP-OCRv5_mobile_det_onnx at e6f4fa85, Apache-2.0), verbatim.
    internal const string DetectorYaml = """
        Global:
          model_name: PP-OCRv5_mobile_det
        Hpi:
          backend_configs:
            paddle_infer:
              trt_dynamic_shapes: &id001
                x:
                - - 1
                  - 3
                  - 32
                  - 32
                - - 1
                  - 3
                  - 736
                  - 736
                - - 1
                  - 3
                  - 4000
                  - 4000
            tensorrt:
              dynamic_shapes: *id001
        PreProcess:
          transform_ops:
          - DecodeImage:
              channel_first: false
              img_mode: BGR
          - DetLabelEncode: null
          - DetResizeForTest:
              resize_long: 960
          - NormalizeImage:
              mean:
              - 0.485
              - 0.456
              - 0.406
              order: hwc
              scale: 1./255.
              std:
              - 0.229
              - 0.224
              - 0.225
          - ToCHWImage: null
          - KeepKeys:
              keep_keys:
              - image
              - shape
              - polys
              - ignore_tags
        PostProcess:
          name: DBPostProcess
          thresh: 0.3
          box_thresh: 0.6
          max_candidates: 1000
          unclip_ratio: 1.5

        """;

    [Fact]
    public void EmbeddedDetectorConfigurationIsThePinnedFile()
    {
        var bytes = System.Text.Encoding.UTF8.GetBytes(DetectorYaml);
        Assert.Equal(PaddleOcrModelAssets.DetectorConfigSha256, Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(bytes)));
    }

    [Theory]
    [InlineData("resize_long: 960", "resize_long: 736", "detector PreProcess")]
    [InlineData("img_mode: BGR", "img_mode: RGB", "detector PreProcess")]
    [InlineData("- 0.485", "- 0.5", "detector PreProcess")]
    [InlineData("scale: 1./255.", "scale: 0.00392", "detector PreProcess")]
    [InlineData("unclip_ratio: 1.5", "unclip_ratio: 2.0", "detector PostProcess")]
    [InlineData("box_thresh: 0.6", "box_thresh: 0.5", "detector PostProcess")]
    [InlineData("  max_candidates: 1000\n", "  max_candidates: 1000\n  use_dilation: true\n", "detector PostProcess")]
    [InlineData("model_name: PP-OCRv5_mobile_det", "model_name: PP-OCRv5_server_det", "Global.model_name")]
    [InlineData("PostProcess:", "Extra: 1\nPostProcess:", "top-level keys")]
    public void DetectorNearMissesAreRefused(string from, string to, string field)
    {
        string changed = DetectorYaml.Replace(from, to, StringComparison.Ordinal);
        Assert.NotEqual(DetectorYaml, changed);
        var error = Assert.Throws<PaddleOcrConfigurationException>(() => PaddleOcrConfiguration.Parse(changed, string.Empty));
        Assert.Contains(field, error.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("a:\n\tb: 1\n", "tabs")]
    [InlineData("a: \"quoted\"\n", "unsupported scalar")]
    [InlineData("a: [1, 2]\n", "unsupported scalar")]
    [InlineData("a: 1\na: 2\n", "duplicate key")]
    [InlineData("a: 'open\n", "unterminated")]
    public void YamlOutsideTheSubsetIsRefused(string yaml, string message)
    {
        var error = Assert.Throws<PaddleOcrConfigurationException>(() => MiniYaml.Parse(yaml));
        Assert.Contains(message, error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void YamlSubsetParsesQuotedPlainAndNestedForms()
    {
        var parsed = MiniYaml.Parse("k:\n- 'a'''\n- 　\n- - 1\n  - 2\n- x: null\n  y: true\n");
        Assert.Equal("{k:[a',　,[1,2],{x:null,y:true}]}", MiniYaml.Canonical(parsed));
    }

    [Fact]
    public void PinnedRecognizerDictionaryLoads()
    {
        var (dir, reason) = PaddleOcrTestAssets.FindSnapshot();
        Assert.SkipWhen(dir is null, reason);
        var configuration = PaddleOcrConfiguration.Load(Path.Combine(dir!, "det", "inference.yml"), Path.Combine(dir!, "rec", "inference.yml"));
        Assert.Equal(PaddleOcrContract.Classes, configuration.Characters.Count);
        Assert.Equal("blank", configuration.Characters[0]);
        Assert.Equal(" ", configuration.Characters[^1]);
        Assert.Equal("\u3000", configuration.Characters[1]);
        Assert.Contains("🇩🇪", configuration.Characters);
    }

    [Theory]
    [InlineData("  - '!'\n", "  - '?'\n", "SHA-256")]
    [InlineData("  - 一\n", "", "size")]
    [InlineData("      - 320\n  - KeepKeys", "      - 640\n  - KeepKeys", "recognizer PreProcess")]
    [InlineData("  name: CTCLabelDecode", "  name: AttnLabelDecode", "PostProcess.name")]
    public void RecognizerNearMissesAreRefused(string from, string to, string field)
    {
        var (dir, reason) = PaddleOcrTestAssets.FindSnapshot();
        Assert.SkipWhen(dir is null, reason);
        string recognizer = File.ReadAllText(Path.Combine(dir!, "rec", "inference.yml"));
        int at = recognizer.IndexOf(from, StringComparison.Ordinal);
        Assert.True(at >= 0, from);
        string changed = string.Concat(recognizer.AsSpan(0, at), to, recognizer.AsSpan(at + from.Length));
        var error = Assert.Throws<PaddleOcrConfigurationException>(() => PaddleOcrConfiguration.Parse(DetectorYaml, changed));
        Assert.Contains(field, error.Message, StringComparison.Ordinal);
    }
}

public sealed class ProcessingTests
{
    private static readonly string[] Characters = ["blank", "a", "b", "文", " "];

    [Fact]
    public void CtcDecodeCollapsesRepeatsAndDropsBlanks()
    {
        // steps: a, a, blank, a, b, b, 文, space
        int[] best = [1, 1, 0, 1, 2, 2, 3, 4];
        var probabilities = new float[best.Length * Characters.Length];
        float[] max = [0.9f, 0.8f, 0.99f, 0.7f, 0.6f, 0.95f, 0.5f, 0.75f];
        for (int t = 0; t < best.Length; t++)
        {
            for (int c = 0; c < Characters.Length; c++)
            {
                probabilities[t * Characters.Length + c] = c == best[t] ? max[t] : (1 - max[t]) / 4;
            }
        }

        var (text, confidence) = PaddleOcrProcessing.Decode(probabilities, best.Length, Characters);
        Assert.Equal("aab文 ", text);
        Assert.Equal((0.9f + 0.7f + 0.6f + 0.5f + 0.75f) / 5, confidence, 6);
    }

    [Fact]
    public void CtcDecodeOfOnlyBlanksIsEmptyWithZeroConfidence()
    {
        var probabilities = new float[3 * Characters.Length];
        for (int t = 0; t < 3; t++)
        {
            probabilities[t * Characters.Length] = 1;
        }

        Assert.Equal((string.Empty, 0f), PaddleOcrProcessing.Decode(probabilities, 3, Characters));
        Assert.Throws<ArgumentException>(() => PaddleOcrProcessing.Decode(probabilities, 4, Characters));
    }

    [Fact]
    public void FirstMaximumWinsTies()
    {
        float[] probabilities = [0.1f, 0.4f, 0.4f, 0.05f, 0.05f];
        Assert.Equal("a", PaddleOcrProcessing.Decode(probabilities, 1, Characters).Text);
    }

    [Theory]
    [InlineData(300, 820, 288, 832)] // longer side below 960: only rounded to multiples of 32
    [InlineData(420, 70, 416, 64)]
    [InlineData(1200, 3000, 384, 960)] // longer side limited to 960
    [InlineData(10, 10, 32, 32)] // padded to 32 x 32 first
    [InlineData(16, 200, 32, 192)]
    public void DetectorInputSizeFollowsTheContract(int height, int width, int expectedHeight, int expectedWidth)
    {
        var image = PaddleOcrImage.FromBgr24(width, height, new byte[width * height * 3]);
        var input = PaddleOcrProcessing.Detector(image);
        Assert.Equal((expectedHeight, expectedWidth, height, width), (input.Height, input.Width, input.SourceHeight, input.SourceWidth));
        Assert.Equal(3 * expectedHeight * expectedWidth, input.Tensor.Length);
    }

    [Fact]
    public void NormalizationUsesBgrOrderAndTheYmlConstants()
    {
        var image = PaddleOcrImage.FromRgb24(32, 32, Enumerable.Repeat(new byte[] { 255, 0, 0 }, 32 * 32).SelectMany(p => p).ToArray());
        var input = PaddleOcrProcessing.Detector(image);
        int plane = 32 * 32;
        // Channel 0 is blue (0), normalized with the first mean/std pair; channel 2 is red (255).
        Assert.Equal((float)(-0.485 / 0.229), input.Tensor[0], 6);
        Assert.Equal((float)(255 * (1.0 / 255.0 / 0.225)) + (float)(-0.406 / 0.225), input.Tensor[2 * plane], 5);
    }

    [Fact]
    public void RecognizerInputHasHeight48AndPadsToAtLeast320()
    {
        var crop = new byte[20 * 100 * 3];
        Array.Fill(crop, (byte)255);
        var (tensor, width) = PaddleOcrProcessing.Recognizer(crop, 20, 100);
        Assert.Equal(320, width);
        Assert.Equal(3 * 48 * 320, tensor.Length);
        Assert.Equal(1f, tensor[0]); // white maps to 1
        Assert.Equal(1f, tensor[239]); // resized width ceil(48 * 5) = 240
        Assert.Equal(0f, tensor[240]); // padding
        var (_, wide) = PaddleOcrProcessing.Recognizer(new byte[10 * 1000 * 3], 10, 1000);
        Assert.Equal(3200, wide);
    }

    [Fact]
    public void BoxesSortTopToBottomThenLeftToRightWithinARow()
    {
        DetectedBox Box(int x, int y) => new([x, y, x + 10, y, x + 10, y + 5, x, y + 5], 0.9);
        var sorted = PaddleOcrProcessing.Sort([Box(50, 12), Box(10, 40), Box(0, 15), Box(100, 10)]);
        Assert.Equal([(0, 15), (50, 12), (100, 10), (10, 40)], sorted.Select(b => (b.Points[0], b.Points[1])));
    }

    [Fact]
    public void BlankMapGivesNoBoxesAndASquareGivesItsUnclippedBox()
    {
        Assert.Empty(PaddleOcrProcessing.PostProcess(new float[64 * 64], 64, 64, 64, 64));
        var map = new float[64 * 64];
        for (int y = 20; y < 30; y++)
        {
            for (int x = 10; x < 50; x++)
            {
                map[y * 64 + x] = 0.95f;
            }
        }

        var box = Assert.Single(PaddleOcrProcessing.PostProcess(map, 64, 64, 128, 128));
        Assert.Equal(0.95, box.Score, 6);
        // The 40 x 10 region grows by area * 1.5 / perimeter = 5.4 on every side, then scales by 2.
        var (left, top, right, bottom) = TextQuad.From(box.Points).Bounds;
        Assert.InRange(left, 8, 10);
        Assert.InRange(top, 28, 30);
        Assert.InRange(right, 108, 110);
        Assert.InRange(bottom, 68, 70);
    }

    [Theory]
    [InlineData(32768, 1)]
    [InlineData(1, 32768)]
    [InlineData(40000, 1000)]
    [InlineData(1, 70000)]
    [InlineData(8193, 8193)] // under the side limit, over 64 Mpx
    public void ImagesBeyondTheInt16CornerRangeOrPixelBudgetAreRefused(int width, int height)
    {
        // The buffer is never read: the size is refused first.
        var error = Assert.Throws<PaddleOcrImageTooLargeException>(() => PaddleOcrImage.FromBgr24(width, height, []));
        Assert.Equal((width, height), (error.Width, error.Height));
        Assert.Throws<PaddleOcrImageTooLargeException>(() => PaddleOcrImage.FromRgba32(width, height, []));
    }

    [Fact]
    public void TheLongestAcceptedSideKeepsBoxCornersInRange()
    {
        // 32,767 x 1 is accepted; its corners fit int16, the type upstream stores them in.
        var image = PaddleOcrImage.FromBgr24(PaddleOcrContract.MaxSide, 1, new byte[PaddleOcrContract.MaxSide * 3]);
        Assert.Equal(PaddleOcrContract.MaxSide, image.Width);
        var input = PaddleOcrProcessing.Detector(image);
        Assert.Equal(960, input.Width);
    }

    [Fact]
    public void ImageFactoriesConvertToBgr()
    {
        Assert.Equal(new byte[] { 3, 2, 1 }, PaddleOcrImage.FromRgb24(1, 1, [1, 2, 3]).Bgr);
        Assert.Equal(new byte[] { 3, 2, 1 }, PaddleOcrImage.FromBgr24(1, 1, [3, 2, 1]).Bgr);
        Assert.Equal(new byte[] { 3, 2, 1, 7, 6, 5 }, PaddleOcrImage.FromRgba32(1, 2, [1, 2, 3, 0, 9, 9, 9, 9, 5, 6, 7, 255], 8).Bgr);
        Assert.Throws<ArgumentException>(() => PaddleOcrImage.FromRgb24(2, 2, new byte[5]));
        Assert.Throws<ArgumentOutOfRangeException>(() => PaddleOcrImage.FromRgb24(0, 2, []));
        Assert.IsNotType<PaddleOcrImageTooLargeException>(Record.Exception(() => PaddleOcrImage.FromRgb24(0, 2, [])));
    }
}

public sealed class CapabilityAndAssetTests
{
    [Fact]
    public void EveryCapabilityHasAnExplicitStatusAndNoneIsSupported()
    {
        Assert.Equal(CapabilityStatus.Experimental, PaddleOcrCapabilities.All["ocr"]);
        Assert.Equal(CapabilityStatus.Unsupported, PaddleOcrCapabilities.All["textline_orientation"]);
        Assert.Equal(CapabilityStatus.Untested, PaddleOcrCapabilities.All["languages.ja"]);
        Assert.DoesNotContain(CapabilityStatus.Supported, PaddleOcrCapabilities.All.Values);
        Assert.Equal("ppocrv5-mobile/onnx-fp32-cpu", PaddleOcrCapabilities.CpuProfile);
    }

    [Fact]
    public void LocksPinTheOfficialExportsByFullCommit()
    {
        Assert.Matches("^[0-9a-f]{40}$", PaddleOcrModelAssets.DetectorRevision);
        Assert.Matches("^[0-9a-f]{40}$", PaddleOcrModelAssets.RecognizerRevision);
        Assert.Equal(["inference.onnx", "inference.yml"], PaddleOcrModelAssets.DetectorLock.Files.Select(f => f.Path));
        Assert.Equal(["inference.onnx", "inference.yml"], PaddleOcrModelAssets.RecognizerLock.Files.Select(f => f.Path));
    }
}

public sealed class EngineTests
{
    private static string CorpusImage(string name) =>
        Path.Combine(PaddleOcrTestAssets.RepositoryRoot, "model-pipeline", "reference", "corpus", "ocr-v1", name);

    [Fact]
    public void RecognizesTheCorpusTextWithThePinnedModels()
    {
        var (files, reason) = PaddleOcrTestAssets.FindModel();
        Assert.SkipWhen(files is null, reason);
        using var engine = PaddleOcrEngine.Create(files!);
        foreach (var (name, expected) in new[]
        {
            ("invoice.png", new[] { "INVOICE #4821", "Widget A x3", "$1,299.00", "Service fee", "$45.50", "Total due", "$1,344.50" }),
            ("chinese.png", ["文字识别测试", "今天的天气很好，我们去公园散步。", "价格：128元，数量：3件"]),
            ("vertical.png", ["Sideways label 90"]),
            ("blank.png", []),
        })
        {
            var decoded = EmbeddingGemma2ImageDecoder.DecodeFile(CorpusImage(name));
            var lines = engine.Recognize(PaddleOcrImage.FromRgb24(decoded.Width, decoded.Height, decoded.Pixels.Span), TestContext.Current.CancellationToken);
            Assert.Equal(expected, lines.Select(line => line.Text));
            Assert.All(lines, line => Assert.InRange(line.Confidence, 0.9f, 1f));
        }
    }

    [Fact]
    public void ACancelledRequestRunsNothing()
    {
        var (files, reason) = PaddleOcrTestAssets.FindModel();
        Assert.SkipWhen(files is null, reason);
        using var engine = PaddleOcrEngine.Create(files!);
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        Assert.Throws<OperationCanceledException>(() => engine.Recognize(PaddleOcrImage.FromBgr24(4, 4, new byte[48]), cancelled.Token));
        engine.Dispose();
        Assert.Throws<ObjectDisposedException>(() => engine.Recognize(PaddleOcrImage.FromBgr24(4, 4, new byte[48]), TestContext.Current.CancellationToken));
    }
}
