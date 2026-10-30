using Microsoft.ML.OnnxRuntime;
using Starlings.Runtime.Onnx;

namespace Starlings.Models.PaddleOcr;

/// <summary>
/// PP-OCRv5 mobile OCR over ONNX Runtime: the DB detector finds text lines, each line is cropped upright and read by
/// the CTC recognizer. Lines come back in PaddleOCR's reading order (top to bottom, then left to right within a row).
/// </summary>
/// <remarks>
/// <para>The pipeline is paddlex 3.7.2's with the pinned configurations: the detector input's longer side is limited
/// to 960 pixels, DB threshold 0.3, box threshold 0.6, unclip ratio 1.5, quadrilateral boxes; recognition one crop at a
/// time (height 48), greedy CTC decoding. No text-line or document orientation model runs, so text rotated beyond what
/// the tall-crop rule fixes (a crop at least 1.5 times taller than wide is turned 90 degrees counter-clockwise) is
/// read as it stands.</para>
/// <para>A single instance is not proven safe for concurrent calls (<see cref="PaddleOcrCapabilities"/>,
/// <c>concurrent_calls</c>).</para>
/// </remarks>
public sealed class PaddleOcrEngine : IDisposable
{
    private readonly OnnxModelSession _detector;
    private readonly OnnxModelSession _recognizer;
    private int _disposed;

    private PaddleOcrEngine(OnnxModelSession detector, OnnxModelSession recognizer, PaddleOcrConfiguration configuration)
    {
        _detector = detector;
        _recognizer = recognizer;
        Configuration = configuration;
    }

    /// <summary>The checked configuration, including the CTC label map.</summary>
    public PaddleOcrConfiguration Configuration { get; }

    /// <summary>
    /// Loads verified files (<see cref="PaddleOcrModelAssets.OpenLocal"/> or
    /// <see cref="PaddleOcrModelAssets.OpenDirectories"/>). Both configurations are checked against the contract before
    /// any graph loads; graph digests the snapshots already verified are passed to the sessions.
    /// </summary>
    /// <exception cref="PaddleOcrConfigurationException">A configuration differs from the contract.</exception>
    public static PaddleOcrEngine Create(PaddleOcrModelFiles files, PaddleOcrOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(files);
        options ??= new PaddleOcrOptions();
        options = options with
        {
            DetectorSession = options.DetectorSession with { KnownModelSha256 = options.DetectorSession.KnownModelSha256 ?? files.DetectorSha256 },
            RecognizerSession = options.RecognizerSession with { KnownModelSha256 = options.RecognizerSession.KnownModelSha256 ?? files.RecognizerSha256 },
        };
        var configuration = PaddleOcrConfiguration.Load(files.DetectorConfigPath, files.RecognizerConfigPath);
        var detector = OnnxModelSession.Create(files.DetectorPath, options.DetectorSession);
        try
        {
            var recognizer = OnnxModelSession.Create(files.RecognizerPath, options.RecognizerSession);
            try
            {
                CheckSignature(detector, "detector");
                CheckSignature(recognizer, "recognizer");
                return new PaddleOcrEngine(detector, recognizer, configuration);
            }
            catch
            {
                recognizer.Dispose();
                throw;
            }
        }
        catch
        {
            detector.Dispose();
            throw;
        }
    }

    /// <summary>Detects and reads the text lines of an image.</summary>
    /// <exception cref="OperationCanceledException">The token was cancelled.</exception>
    public IReadOnlyList<TextLine> Recognize(PaddleOcrImage image, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(image);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        cancellationToken.ThrowIfCancellationRequested();
        var input = PaddleOcrProcessing.Detector(image);
        var probability = RunDetector(input.Tensor, input.Height, input.Width, cancellationToken);
        var boxes = PaddleOcrProcessing.Sort(PaddleOcrProcessing.PostProcess(probability, input.Height, input.Width, input.SourceHeight, input.SourceWidth));
        var lines = new List<TextLine>(boxes.Count);
        foreach (var box in boxes)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (PaddleOcrProcessing.Crop(image, box.Points) is not var (pixels, height, width))
            {
                continue;
            }

            var (tensor, tensorWidth) = PaddleOcrProcessing.Recognizer(pixels, height, width);
            var (probabilities, steps) = RunRecognizer(tensor, tensorWidth, cancellationToken);
            var (text, confidence) = PaddleOcrProcessing.Decode(probabilities, steps, Configuration.Characters);
            lines.Add(new TextLine(TextQuad.From(box.Points), text, confidence, (float)box.Score));
        }

        return lines;
    }

    /// <summary>Runs the detector on a <c>[1, 3, height, width]</c> input; returns the <c>[height, width]</c> probability map.</summary>
    internal float[] RunDetector(float[] tensor, int height, int width, CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        var output = new float[height * width];
        using var inputValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.AsMemory(), [1, 3, height, width]);
        using var outputValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, output.AsMemory(), [1, 1, height, width]);
        _detector.Run([PaddleOcrContract.InputName], [inputValue], [PaddleOcrContract.OutputName], [outputValue], cancellationToken);
        return output;
    }

    /// <summary>Runs the recognizer on a <c>[1, 3, 48, width]</c> input; returns the <c>[steps, classes]</c> probabilities.</summary>
    internal (float[] Probabilities, int Steps) RunRecognizer(float[] tensor, int width, CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        using var inputValue = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, tensor.AsMemory(), [1, 3, PaddleOcrContract.RecognizerHeight, width]);
        using var outputs = _recognizer.Run([PaddleOcrContract.InputName], [inputValue], [PaddleOcrContract.OutputName], cancellationToken);
        var output = outputs[0];
        var shape = output.GetTensorTypeAndShape().Shape;
        if (shape.Length != 3 || shape[0] != 1 || shape[2] != PaddleOcrContract.Classes)
        {
            throw new InvalidDataException($"The recognizer returned shape [{string.Join(", ", shape)}], expected [1, steps, {PaddleOcrContract.Classes}].");
        }

        return (output.GetTensorDataAsSpan<float>().ToArray(), (int)shape[1]);
    }

    private static void CheckSignature(OnnxModelSession session, string name)
    {
        if (session.Inputs.Count != 1 || session.Inputs[0].Name != PaddleOcrContract.InputName
            || session.Outputs.Count != 1 || session.Outputs[0].Name != PaddleOcrContract.OutputName)
        {
            throw new PaddleOcrConfigurationException($"The {name} graph must have the single input 'x' and the single output 'fetch_name_0'.");
        }
    }

    /// <summary>Disposes both sessions. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _recognizer.Dispose();
            _detector.Dispose();
        }
    }
}
