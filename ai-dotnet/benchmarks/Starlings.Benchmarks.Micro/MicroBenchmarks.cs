using System.Globalization;
using BenchmarkDotNet.Attributes;
using Starlings.Models.E5;
using Starlings.Runtime.Onnx;
using Starlings.Tokenizers;

namespace Starlings.Benchmarks;

/// <summary>Settings read from the environment, so BenchmarkDotNet's child processes see the same values.</summary>
internal static class MicroSettings
{
    /// <summary>Set to 1 to add CUDA cases (needs the .Cuda build and an explicit <c>CUDA_VISIBLE_DEVICES</c>).</summary>
    public const string CudaVariable = "STARLING_BENCH_CUDA";

    /// <summary><c>intra_op_num_threads</c> for e5 sessions; default 6 (physical cores of the reference workstation).</summary>
    public const string ThreadsVariable = "STARLING_BENCH_THREADS";

    public static int Threads =>
        int.TryParse(Environment.GetEnvironmentVariable(ThreadsVariable), NumberStyles.Integer, CultureInfo.InvariantCulture, out int t) && t > 0 ? t : 6;

    public static IEnumerable<string> Providers()
    {
        yield return "cpu";
        if (Environment.GetEnvironmentVariable(CudaVariable) == "1")
        {
            if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES")))
            {
                throw new InvalidOperationException("CUDA benchmarks need an explicit device: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n>.");
            }

            yield return "cuda";
        }
    }

    public static E5ModelFiles Files => BenchFixtures.Files(Environment.GetEnvironmentVariable("STARLING_E5_SNAPSHOT"));
}

/// <summary>
/// The owned tokenizer on the e5 fixture texts: the 44 short single-text cases one at a time, the 512-token
/// <c>long-english</c> case, and the 44 short cases as one batch call. Truncation at 512, no padding (the e5 profile).
/// </summary>
[MemoryDiagnoser]
public class TokenizerBenchmarks
{
    private Tokenizer _tokenizer = null!;
    private E5Tokenizer _e5 = null!;
    private string[] _short = [];
    private string _long = string.Empty;

    [GlobalSetup]
    public void Setup()
    {
        var files = MicroSettings.Files;
        _tokenizer = Tokenizer.FromUtf8Json(File.ReadAllBytes(files.TokenizerPath)).WithPadding(null).WithTruncation(new TruncationOptions(512));
        _e5 = E5Tokenizer.FromFile(files.TokenizerPath);
        _short = BenchFixtures.Ordered.Where(c => c.Ids.Length < 512).Select(c => c.Text).ToArray();
        _long = BenchFixtures.Cases["long-english"].Text;
    }

    /// <summary>One <c>Encode</c> per short text; reported per text.</summary>
    [Benchmark(OperationsPerInvoke = 44)]
    public int EncodeShortEach()
    {
        int total = 0;
        foreach (string text in _short)
        {
            total += _tokenizer.Encode(text).Count;
        }

        return total;
    }

    /// <summary>One <c>Encode</c> of the 512-token text.</summary>
    [Benchmark]
    public int EncodeLong() => _tokenizer.Encode(_long).Count;

    /// <summary>One <c>EncodeBatch</c> of the 44 short texts.</summary>
    [Benchmark]
    public int EncodeBatchShort() => _tokenizer.EncodeBatch(_short).Count;

    /// <summary><see cref="E5Tokenizer.Tokenize"/> (encode plus the int64 id and mask copies) per short text.</summary>
    [Benchmark(OperationsPerInvoke = 44)]
    public int E5TokenizeShortEach()
    {
        int total = 0;
        foreach (string text in _short)
        {
            total += _e5.Tokenize(text).Length;
        }

        return total;
    }
}

/// <summary>
/// e5 embeddings through the public API, per workload of the shared workload file: one invocation embeds every request
/// of the workload once (so short-b1 reports the mean over the 44 short cases). From ids
/// (<see cref="E5Encoder.EmbedBatch(IReadOnlyList{EncoderInput}, CancellationToken)"/>) and from text
/// (<see cref="E5TextEmbedder.EmbedBatch"/>).
/// </summary>
[MemoryDiagnoser]
public class E5Benchmarks
{
    private E5Encoder _encoder = null!;
    private E5TextEmbedder _embedder = null!;
    private EncoderInput[][] _ids = [];
    private string[][] _texts = [];

    [ParamsSource(nameof(ProviderValues))]
    public string Provider { get; set; } = "cpu";

    [Params("short-b1", "short-b8", "long512-b1")]
    public string Workload { get; set; } = "short-b1";

    public static IEnumerable<string> ProviderValues() => MicroSettings.Providers();

    [GlobalSetup]
    public void Setup()
    {
        var files = MicroSettings.Files;
        _encoder = E5Encoder.Create(files.ModelPath, new E5EncoderOptions
        {
            Session = new OnnxSessionOptions
            {
                Provider = Provider == "cuda" ? OnnxExecutionProvider.Cuda : OnnxExecutionProvider.Cpu,
                IntraOpNumThreads = MicroSettings.Threads,
                InterOpNumThreads = 1,
                ExpectedModelSha256 = E5ModelAssets.ModelSha256,
            },
        });
        _embedder = new E5TextEmbedder(E5Tokenizer.FromFile(files.TokenizerPath, E5ModelAssets.TokenizerSha256), _encoder);
        var workload = BenchWorkloadSpec.Load().Workloads.Single(w => w.Name == Workload);
        _ids = workload.Requests.Select(r => r.Select(c => new EncoderInput(c.Ids)).ToArray()).ToArray();
        _texts = workload.Requests.Select(r => r.Select(c => c.Text).ToArray()).ToArray();
    }

    [GlobalCleanup]
    public void Cleanup() => _encoder.Dispose();

    /// <summary>All requests of the workload, from ids. Reported per invocation (all requests).</summary>
    [Benchmark]
    public int EmbedFromIds()
    {
        int rows = 0;
        foreach (var request in _ids)
        {
            rows += _encoder.EmbedBatch(request).Length;
        }

        return rows;
    }

    /// <summary>All requests of the workload, from text (owned tokenizer, then the encoder).</summary>
    [Benchmark]
    public int EmbedFromText()
    {
        int rows = 0;
        foreach (var request in _texts)
        {
            rows += _embedder.EmbedBatch(request).Length;
        }

        return rows;
    }
}
