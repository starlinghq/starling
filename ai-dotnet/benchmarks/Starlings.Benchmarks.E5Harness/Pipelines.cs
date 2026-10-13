using System.Buffers;
using System.Diagnostics;
using System.Globalization;
using System.Text.Json.Nodes;
using Starlings.Models.E5;
using Starlings.Runtime.Onnx;
using Microsoft.ML.OnnxRuntime;

namespace Starlings.Benchmarks;

/// <summary>Session settings shared by every implementation and recorded in the run block.</summary>
internal sealed record PipelineSettings(OnnxExecutionProvider Provider, int? IntraOpThreads, int? InterOpThreads, int CudaDeviceId, bool HashModelAtLoad)
{
    public bool IsCuda => Provider == OnnxExecutionProvider.Cuda;

    public JsonObject ToJson(IReadOnlyDictionary<string, string> providerOptions) => new()
    {
        ["provider"] = IsCuda ? "cuda" : "cpu",
        ["intra_op_num_threads"] = IntraOpThreads,
        ["inter_op_num_threads"] = InterOpThreads,
        ["execution_mode"] = "sequential",
        ["graph_optimization_level"] = "all",
        ["provider_options"] = IsCuda
            ? new JsonObject { ["CUDAExecutionProvider"] = new JsonObject(providerOptions.Select(kv => KeyValuePair.Create(kv.Key, (JsonNode?)kv.Value))) }
            : new JsonObject(),
        ["pad_id"] = 1,
        ["output_location"] = "host",
        ["max_batch_size"] = 32,
        ["model_sha256_check_at_load"] = HashModelAtLoad,
    };
}

/// <summary>
/// One way of turning a request (fixture cases) into embeddings, with per-stage timing. Stage times are written into
/// a caller-provided span in the order of <see cref="StageNames"/>.
/// </summary>
internal interface IEmbedPipeline : IDisposable
{
    string Impl { get; }

    double TokenizerLoadMs { get; }

    double SessionCreateMs { get; }

    string? ModelSha256 { get; }

    IReadOnlyList<string> SessionProviders { get; }

    JsonObject Settings { get; }

    string[] StageNames(bool text);

    float[][] Embed(BenchCase[] request, bool text, Span<double> stages);
}

/// <summary>
/// The product path: <see cref="E5Tokenizer"/> then <see cref="E5Encoder.EmbedBatch(IReadOnlyList{EncoderInput}, CancellationToken)"/>
/// (public API only). Stages: <c>tokenize</c> (text path) and <c>encode</c> (padding, ORT run, pooling inside the library).
/// </summary>
internal sealed class LibraryPipeline : IEmbedPipeline
{
    private static readonly string[] s_textStages = ["tokenize", "encode"];
    private static readonly string[] s_idStages = ["encode"];
    private readonly E5Tokenizer _tokenizer;
    private readonly E5Encoder _encoder;
    private readonly PipelineSettings _settings;

    public LibraryPipeline(E5ModelFiles files, PipelineSettings settings)
    {
        _settings = settings;
        long t0 = Stopwatch.GetTimestamp();
        _tokenizer = E5Tokenizer.FromFile(files.TokenizerPath);
        TokenizerLoadMs = MonotonicClock.ElapsedMs(t0);
        long t1 = Stopwatch.GetTimestamp();
        _encoder = E5Encoder.Create(files.ModelPath, new E5EncoderOptions
        {
            Session = new OnnxSessionOptions
            {
                Provider = settings.Provider,
                IntraOpNumThreads = settings.IntraOpThreads,
                InterOpNumThreads = settings.InterOpThreads,
                CudaDeviceId = settings.CudaDeviceId,
                ComputeModelSha256 = settings.HashModelAtLoad,
            },
        });
        SessionCreateMs = MonotonicClock.ElapsedMs(t1);
    }

    public string Impl => "library";

    public double TokenizerLoadMs { get; }

    public double SessionCreateMs { get; }

    public string? ModelSha256 => _encoder.Session.Diagnostics.ModelSha256;

    public IReadOnlyList<string> SessionProviders => _encoder.Session.Diagnostics.SessionProviders;

    public JsonObject Settings => _settings.ToJson(_encoder.Session.Diagnostics.ProviderOptions);

    public string[] StageNames(bool text) => text ? s_textStages : s_idStages;

    public float[][] Embed(BenchCase[] request, bool text, Span<double> stages)
    {
        var inputs = new EncoderInput[request.Length];
        long t = Stopwatch.GetTimestamp();
        int s = 0;
        if (text)
        {
            for (int i = 0; i < request.Length; i++)
            {
                inputs[i] = _tokenizer.Tokenize(request[i].Text);
            }

            stages[s++] = MonotonicClock.ElapsedMs(t);
            t = Stopwatch.GetTimestamp();
        }
        else
        {
            for (int i = 0; i < request.Length; i++)
            {
                inputs[i] = new EncoderInput(request[i].Ids);
            }
        }

        var result = _encoder.EmbedBatch(inputs);
        stages[s] = MonotonicClock.ElapsedMs(t);
        return result;
    }

    public void Dispose() => _encoder.Dispose();
}

/// <summary>
/// Direct C# execution of the identical backend workflow, without the Starling runtime and encoder layers: a raw
/// <see cref="InferenceSession"/> with the same session settings, the same tokenizer, the same right padding into
/// pooled buffers bound as <see cref="OrtValue"/>s, a preallocated host output and the same double-accumulation mean
/// pooling. The plan's adapter-overhead budget (section 11) is measured against this path.
/// Stages: <c>tokenize</c> (text path), <c>prep</c>, <c>run</c>, <c>pool</c>.
/// </summary>
internal sealed class DirectPipeline : IEmbedPipeline
{
    private const int Dimension = 384;
    private static readonly string[] s_textStages = ["tokenize", "prep", "run", "pool"];
    private static readonly string[] s_idStages = ["prep", "run", "pool"];
    private static readonly string[] s_inputNames = [E5Encoder.InputIdsName, E5Encoder.AttentionMaskName, E5Encoder.TokenTypeIdsName];
    private static readonly string[] s_outputNames = [E5Encoder.LastHiddenStateName];
    private readonly E5Tokenizer _tokenizer;
    private readonly SessionOptions _sessionOptions;
    private readonly OrtCUDAProviderOptions? _cudaOptions;
    private readonly InferenceSession _session;
    private readonly RunOptions _runOptions = new();
    private readonly PipelineSettings _settings;
    private readonly Dictionary<string, string> _providerOptions = new(StringComparer.Ordinal);

    public DirectPipeline(E5ModelFiles files, PipelineSettings settings)
    {
        _settings = settings;
        long t0 = Stopwatch.GetTimestamp();
        _tokenizer = E5Tokenizer.FromFile(files.TokenizerPath);
        TokenizerLoadMs = MonotonicClock.ElapsedMs(t0);
        long t1 = Stopwatch.GetTimestamp();
        if (settings.HashModelAtLoad)
        {
            ModelSha256 = BenchEnvironment.Sha256File(files.ModelPath);
        }

        _sessionOptions = new SessionOptions
        {
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        };
        if (settings.IntraOpThreads is int intra)
        {
            _sessionOptions.IntraOpNumThreads = intra;
        }

        if (settings.InterOpThreads is int inter)
        {
            _sessionOptions.InterOpNumThreads = inter;
        }

        if (settings.IsCuda)
        {
            _cudaOptions = new OrtCUDAProviderOptions();
            _cudaOptions.UpdateOptions(new Dictionary<string, string>
            {
                ["device_id"] = settings.CudaDeviceId.ToString(CultureInfo.InvariantCulture),
                ["use_tf32"] = "0",
            });
            _sessionOptions.AppendExecutionProvider_CUDA(_cudaOptions);
            foreach (var pair in _cudaOptions.GetOptions().Split(';', StringSplitOptions.RemoveEmptyEntries))
            {
                var kv = pair.Split('=', 2);
                _providerOptions[kv[0]] = kv.Length > 1 ? kv[1] : string.Empty;
            }
        }

        _session = new InferenceSession(files.ModelPath, _sessionOptions);
        SessionCreateMs = MonotonicClock.ElapsedMs(t1);
    }

    public string Impl => "direct";

    public double TokenizerLoadMs { get; }

    public double SessionCreateMs { get; }

    public string? ModelSha256 { get; }

    public IReadOnlyList<string> SessionProviders => _settings.IsCuda
        ? [OnnxRuntimeInfo.CudaProviderName, OnnxRuntimeInfo.CpuProviderName]
        : [OnnxRuntimeInfo.CpuProviderName];

    public JsonObject Settings => _settings.ToJson(_providerOptions);

    public string[] StageNames(bool text) => text ? s_textStages : s_idStages;

    public float[][] Embed(BenchCase[] request, bool text, Span<double> stages)
    {
        int count = request.Length;
        var rows = new long[count][];
        long t = Stopwatch.GetTimestamp();
        int s = 0;
        if (text)
        {
            for (int i = 0; i < count; i++)
            {
                rows[i] = _tokenizer.Tokenize(request[i].Text).InputIds.ToArray();
            }

            stages[s++] = MonotonicClock.ElapsedMs(t);
            t = Stopwatch.GetTimestamp();
        }
        else
        {
            for (int i = 0; i < count; i++)
            {
                rows[i] = request[i].Ids;
            }
        }

        int width = 0;
        foreach (var row in rows)
        {
            width = Math.Max(width, row.Length);
        }

        int cells = count * width;
        long[] ids = ArrayPool<long>.Shared.Rent(cells);
        long[] mask = ArrayPool<long>.Shared.Rent(cells);
        long[] types = ArrayPool<long>.Shared.Rent(cells);
        float[] hidden = ArrayPool<float>.Shared.Rent(cells * Dimension);
        try
        {
            for (int r = 0; r < count; r++)
            {
                var idRow = ids.AsSpan(r * width, width);
                var maskRow = mask.AsSpan(r * width, width);
                rows[r].CopyTo(idRow);
                idRow[rows[r].Length..].Fill(1);
                maskRow[..rows[r].Length].Fill(1);
                maskRow[rows[r].Length..].Clear();
            }

            Array.Clear(types, 0, cells);
            long[] shape = [count, width];
            var memory = OrtMemoryInfo.DefaultInstance;
            using var idsValue = OrtValue.CreateTensorValueFromMemory(memory, ids.AsMemory(0, cells), shape);
            using var maskValue = OrtValue.CreateTensorValueFromMemory(memory, mask.AsMemory(0, cells), shape);
            using var typesValue = OrtValue.CreateTensorValueFromMemory(memory, types.AsMemory(0, cells), shape);
            using var hiddenValue = OrtValue.CreateTensorValueFromMemory(memory, hidden.AsMemory(0, cells * Dimension), [count, width, Dimension]);
            stages[s++] = MonotonicClock.ElapsedMs(t);
            t = Stopwatch.GetTimestamp();

            _session.Run(_runOptions, s_inputNames, [idsValue, maskValue, typesValue], s_outputNames, [hiddenValue]);
            stages[s++] = MonotonicClock.ElapsedMs(t);
            t = Stopwatch.GetTimestamp();

            var result = new float[count][];
            Span<double> sum = stackalloc double[Dimension];
            for (int r = 0; r < count; r++)
            {
                sum.Clear();
                int n = rows[r].Length;
                var block = hidden.AsSpan(r * width * Dimension, width * Dimension);
                for (int p = 0; p < n; p++)
                {
                    var v = block.Slice(p * Dimension, Dimension);
                    for (int d = 0; d < Dimension; d++)
                    {
                        sum[d] += v[d];
                    }
                }

                double norm = 0;
                for (int d = 0; d < Dimension; d++)
                {
                    sum[d] /= Math.Max(n, 1e-9);
                    norm += sum[d] * sum[d];
                }

                norm = Math.Max(Math.Sqrt(norm), 1e-12);
                var output = new float[Dimension];
                for (int d = 0; d < Dimension; d++)
                {
                    output[d] = (float)(sum[d] / norm);
                }

                result[r] = output;
            }

            stages[s] = MonotonicClock.ElapsedMs(t);
            return result;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(hidden);
            ArrayPool<long>.Shared.Return(types);
            ArrayPool<long>.Shared.Return(mask);
            ArrayPool<long>.Shared.Return(ids);
        }
    }

    public void Dispose()
    {
        _session.Dispose();
        _runOptions.Dispose();
        _cudaOptions?.Dispose();
        _sessionOptions.Dispose();
    }
}
