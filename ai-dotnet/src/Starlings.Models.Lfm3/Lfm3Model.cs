using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Starlings.Runtime.Onnx;

namespace Starlings.Models.Lfm3;

/// <summary>
/// The step graph of <c>LiquidAI/LFM3-1.7B-ONNX</c> on ONNX Runtime. One graph serves the prefill and every decode
/// step: it takes <c>input_ids</c>, <c>attention_mask</c>, <c>position_ids</c>, <c>logits_to_keep</c> and the caches of
/// the previous call (<c>past_conv.N</c> for the 13 convolution layers, <c>past_key_values.N.key</c> and <c>.value</c>
/// for the 11 attention layers), and returns <c>logits</c> and the updated caches. Each request gets its own
/// <see cref="Lfm3Sequence"/>; the weights are shared. Thread-safe.
/// </summary>
public sealed class Lfm3Model : IDisposable
{
    internal const string InputIdsName = "input_ids";
    internal const string AttentionMaskName = "attention_mask";
    internal const string PositionIdsName = "position_ids";
    internal const string LogitsToKeepName = "logits_to_keep";
    internal const string LogitsName = "logits";

    private readonly OnnxModelSession _session;
    private int _disposed;

    private Lfm3Model(OnnxModelSession session, Lfm3Precision precision, int contextLength)
    {
        _session = session;
        Precision = precision;
        ContextLength = contextLength;
        var pastNames = new List<string>();
        var presentNames = new List<string>();
        var shapes = new List<long[]>();
        for (int layer = 0; layer < Lfm3Checkpoint.Layers; layer++)
        {
            if (Lfm3Checkpoint.LayerTypes[layer] == "conv")
            {
                pastNames.Add($"past_conv.{layer}");
                presentNames.Add($"present_conv.{layer}");
                shapes.Add([1, Lfm3Checkpoint.HiddenSize, Lfm3Checkpoint.ConvCacheLength]);
            }
            else
            {
                foreach (string part in (string[])["key", "value"])
                {
                    pastNames.Add($"past_key_values.{layer}.{part}");
                    presentNames.Add($"present.{layer}.{part}");
                    shapes.Add([1, Lfm3Checkpoint.KeyValueHeads, 0, Lfm3Checkpoint.HeadSize]);
                }
            }
        }

        PastNames = [.. pastNames];
        InitialPastShapes = [.. shapes];
        InputNames = [InputIdsName, AttentionMaskName, PositionIdsName, LogitsToKeepName, .. PastNames];
        OutputNames = [LogitsName, .. presentNames];
        CheckGraph();
    }

    /// <summary>The graph's precision.</summary>
    public Lfm3Precision Precision { get; }

    /// <summary>Largest prompt plus completion length a sequence accepts (<see cref="Lfm3Options.ContextLength"/>).</summary>
    public int ContextLength { get; }

    /// <summary>Session diagnostics (provider, model digest, inputs and outputs).</summary>
    public OnnxSessionDiagnostics Diagnostics => _session.Diagnostics;

    internal string[] PastNames { get; }

    internal long[][] InitialPastShapes { get; }

    internal string[] InputNames { get; }

    internal string[] OutputNames { get; }

    internal OnnxModelSession Session => _session;

    /// <summary>
    /// Loads the graph of verified files (<see cref="Lfm3ModelAssets.OpenLocal"/> or
    /// <see cref="Lfm3ModelAssets.OpenDirectory"/>). When the snapshot hashed the graph, the session takes that digest
    /// as <see cref="OnnxSessionOptions.KnownModelSha256"/> (unless the options set one) instead of reading it again.
    /// </summary>
    public static Lfm3Model Create(Lfm3ModelFiles files, Lfm3Options? options = null)
    {
        ArgumentNullException.ThrowIfNull(files);
        options ??= new Lfm3Options();
        options = options with { Session = options.Session with { KnownModelSha256 = options.Session.KnownModelSha256 ?? files.ModelSha256 } };
        return Create(files.ModelPath, files.Precision, options);
    }

    /// <summary>Loads an LFM3-1.7B step graph from <paramref name="modelPath"/> (its external data files next to it).</summary>
    /// <exception cref="InvalidDataException">The graph's inputs or outputs are not those of the LFM3-1.7B export at <paramref name="precision"/>.</exception>
    public static Lfm3Model Create(string modelPath, Lfm3Precision precision, Lfm3Options? options = null)
    {
        options ??= new Lfm3Options();
        options.Validate();
        var session = OnnxModelSession.Create(modelPath, options.Session);
        try
        {
            return new Lfm3Model(session, precision, options.ContextLength);
        }
        catch
        {
            session.Dispose();
            throw;
        }
    }

    /// <summary>Starts a request: empty caches, no tokens.</summary>
    public Lfm3Sequence CreateSequence()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return new Lfm3Sequence(this);
    }

    /// <summary>Disposes the session. Idempotent; sequences still open fail on their next call.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _session.Dispose();
        }
    }

    internal TensorElementType CacheElementType => Precision == Lfm3Precision.Fp16 ? TensorElementType.Float16 : TensorElementType.Float;

    private void CheckGraph()
    {
        var inputs = _session.Inputs.Select(i => i.Name).ToHashSet(StringComparer.Ordinal);
        var outputs = _session.Outputs.Select(o => o.Name).ToHashSet(StringComparer.Ordinal);
        string[] missingInputs = [.. InputNames.Where(n => !inputs.Contains(n))];
        string[] missingOutputs = [.. OutputNames.Where(n => !outputs.Contains(n))];
        if (missingInputs.Length > 0 || missingOutputs.Length > 0 || inputs.Count != InputNames.Length || outputs.Count != OutputNames.Length)
        {
            throw new InvalidDataException(
                $"The graph is not the LFM3-1.7B step graph: {inputs.Count} inputs and {outputs.Count} outputs " +
                $"(expected {InputNames.Length} and {OutputNames.Length}); missing inputs [{string.Join(", ", missingInputs)}], " +
                $"missing outputs [{string.Join(", ", missingOutputs)}].");
        }
    }
}
