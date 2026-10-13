using Starlings.Models.E5;
using Starlings.Runtime.Onnx;

namespace Starlings.TestSupport;

/// <summary>
/// One encoder over the real e5 model per test class, loaded with the manifest SHA-256 enforced. When the model is
/// absent the fixture holds a skip reason instead and the tests skip, so CI without the model stays green.
/// </summary>
public abstract class E5ModelFixture : IDisposable
{
    private readonly E5Encoder? _encoder;

    protected E5ModelFixture(OnnxSessionOptions sessionOptions, Func<string?>? gate = null)
    {
        string? gateReason = gate?.Invoke();
        if (gateReason is not null)
        {
            SkipReason = gateReason;
            return;
        }

        var (path, reason) = E5TestAssets.FindModel();
        if (path is null)
        {
            SkipReason = reason;
            return;
        }

        ModelPath = path;
        _encoder = E5Encoder.Create(path, new E5EncoderOptions
        {
            Session = sessionOptions with { ExpectedModelSha256 = E5TestAssets.ModelSha256, CapturePlacement = true },
        });
    }

    /// <summary>Why the real-model tests skip, or null when the model is loaded.</summary>
    public string? SkipReason { get; }

    public string? ModelPath { get; }

    /// <summary>The encoder; skips the calling test when the model is unavailable.</summary>
    public E5Encoder Encoder
    {
        get
        {
            Assert.SkipWhen(_encoder is null, SkipReason ?? "model unavailable");
            return _encoder!;
        }
    }

    public void Dispose()
    {
        _encoder?.Dispose();
        GC.SuppressFinalize(this);
    }
}
