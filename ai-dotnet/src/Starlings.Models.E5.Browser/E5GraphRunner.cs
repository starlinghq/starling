using System.Runtime.Versioning;
using Starlings.Runtime.Onnx;
using Starlings.Runtime.OnnxWeb;

namespace Starlings.Models.E5.Browser;

/// <summary>
/// The one call the browser encoder makes: run the e5 graph on packed <c>[rows, width]</c> int64 inputs and write
/// <c>last_hidden_state</c> (<c>rows x width x dimension</c> floats). The seam between the encoder (testable on any
/// runtime) and onnxruntime-web (browser only).
/// </summary>
internal interface IE5GraphRunner : IDisposable
{
    int Dimension { get; }

    bool HasTokenTypeIds { get; }

    Task RunAsync(long[] ids, long[] mask, long[]? types, int rows, int width, Memory<float> hidden, CancellationToken cancellationToken);
}

/// <summary>
/// The graph checks of <see cref="E5Encoder"/>'s constructor, on onnxruntime-web's metadata: the same code, after
/// mapping onnxruntime-web's element type names (<c>int64</c>, <c>float32</c>) to ONNX Runtime's (<c>Int64</c>,
/// <c>Float</c>).
/// </summary>
internal static class E5GraphSignature
{
    internal static (int Dimension, bool HasTokenTypeIds) Validate(IReadOnlyList<OnnxWebTensorInfo> inputs, IReadOnlyList<OnnxWebTensorInfo> outputs) =>
        E5Encoder.ValidateSignature([.. inputs.Select(ToOnnx)], [.. outputs.Select(ToOnnx)]);

    private static OnnxTensorInfo ToOnnx(OnnxWebTensorInfo info) => new(
        info.Name,
        info.ElementType switch
        {
            "int64" => "Int64",
            "float32" => "Float",
            null => "NonTensor",
            var other => other,
        },
        info.Dimensions ?? [],
        [.. (info.Dimensions ?? []).Select(d => d < 0 ? "?" : string.Empty)]);
}

/// <summary>The e5 graph on an <see cref="OnnxWebSession"/>.</summary>
[SupportedOSPlatform("browser")]
internal sealed class OnnxWebE5GraphRunner : IE5GraphRunner
{
    private static readonly string[] OutputName = [E5Encoder.LastHiddenStateName];
    private readonly OnnxWebSession _session;
    private readonly bool _ownsSession;

    internal OnnxWebE5GraphRunner(OnnxWebSession session, bool ownsSession)
    {
        _session = session;
        _ownsSession = ownsSession;
        (Dimension, HasTokenTypeIds) = E5GraphSignature.Validate(session.Inputs, session.Outputs);
    }

    public int Dimension { get; }

    public bool HasTokenTypeIds { get; }

    internal OnnxWebSession Session => _session;

    public async Task RunAsync(long[] ids, long[] mask, long[]? types, int rows, int width, Memory<float> hidden, CancellationToken cancellationToken)
    {
        int cells = rows * width;
        OnnxWebInput[] inputs = types is null
            ?
            [
                OnnxWebInput.FromInt64(E5Encoder.InputIdsName, ids.AsMemory(0, cells), rows, width),
                OnnxWebInput.FromInt64(E5Encoder.AttentionMaskName, mask.AsMemory(0, cells), rows, width),
            ]
            :
            [
                OnnxWebInput.FromInt64(E5Encoder.InputIdsName, ids.AsMemory(0, cells), rows, width),
                OnnxWebInput.FromInt64(E5Encoder.AttentionMaskName, mask.AsMemory(0, cells), rows, width),
                OnnxWebInput.FromInt64(E5Encoder.TokenTypeIdsName, types.AsMemory(0, cells), rows, width),
            ];
        await _session.RunAsync(inputs, OutputName[0], hidden, cancellationToken).ConfigureAwait(false);
    }

    public void Dispose()
    {
        if (_ownsSession)
        {
            _session.Dispose();
        }
    }
}
