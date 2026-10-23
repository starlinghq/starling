using System.Runtime.Versioning;
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

/// <summary>The graph checks of <see cref="E5Encoder"/>'s constructor, on onnxruntime-web's metadata.</summary>
internal static class E5GraphSignature
{
    internal static (int Dimension, bool HasTokenTypeIds) Validate(IReadOnlyList<OnnxWebTensorInfo> inputs, IReadOnlyList<OnnxWebTensorInfo> outputs)
    {
        var byName = inputs.ToDictionary(i => i.Name, StringComparer.Ordinal);
        foreach (var name in new[] { E5Encoder.InputIdsName, E5Encoder.AttentionMaskName })
        {
            if (!byName.TryGetValue(name, out var info) || info.ElementType != "int64" || info.Dimensions is not { Count: 2 })
            {
                throw new NotSupportedException($"Graph input '{name}' (int64, rank 2) is missing or has another type; this is not the e5 encoder signature.");
            }
        }

        bool hasTypes = byName.TryGetValue(E5Encoder.TokenTypeIdsName, out var types);
        if (hasTypes && (types!.ElementType != "int64" || types.Dimensions is not { Count: 2 }))
        {
            throw new NotSupportedException($"Graph input '{E5Encoder.TokenTypeIdsName}' must be int64 of rank 2.");
        }

        var unknown = byName.Keys.Except([E5Encoder.InputIdsName, E5Encoder.AttentionMaskName, E5Encoder.TokenTypeIdsName], StringComparer.Ordinal).ToList();
        if (unknown.Count > 0)
        {
            throw new NotSupportedException($"Graph has inputs this encoder does not feed: {string.Join(", ", unknown)}.");
        }

        var output = outputs.FirstOrDefault(o => o.Name == E5Encoder.LastHiddenStateName);
        if (output is null || output.ElementType != "float32" || output.Dimensions is not { Count: 3 } || output.Dimensions[2] <= 0)
        {
            throw new NotSupportedException($"Graph output '{E5Encoder.LastHiddenStateName}' (float32, [batch, sequence, hidden] with fixed hidden size) is missing.");
        }

        return ((int)output.Dimensions[2], hasTypes);
    }
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
