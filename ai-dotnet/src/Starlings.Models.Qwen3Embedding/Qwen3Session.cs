using Starlings.Acquisition;
using Starlings.Batching;
using Starlings.Runtime.Onnx;

namespace Starlings.Models.Qwen3Embedding;

/// <summary>
/// The verified released directory, the tokenizer and one ONNX Runtime CPU session, shared by the embedder and the
/// reranker. A chunk of rows is right-padded with <see cref="Qwen3EmbeddingProfile.PadTokenId"/> and run once; each row
/// is read at its last attended position (<c>sum(mask) - 1</c>).
/// </summary>
internal sealed class Qwen3Session : IDisposable
{
    private readonly EncoderCore _core;
    private readonly Qwen3EmbeddingOptions _options;
    private int _disposed;

    public Qwen3Session(Qwen3EmbeddingProfile profile, string directory, Qwen3EmbeddingOptions? options)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentException.ThrowIfNullOrEmpty(directory);
        options ??= new Qwen3EmbeddingOptions();
        options.Validate();
        Profile = profile;
        _options = options;
        ModelSnapshot files = profile.OpenDirectory(directory, options.Verification);
        Tokenization = Qwen3Tokenization.Open(profile, files.GetFilePath(Qwen3EmbeddingProfile.TokenizerFile),
            verifyDigest: options.Verification != CacheVerification.Sha256);

        // From the path, so ONNX Runtime finds model.onnx.data next to the graph.
        Session = OnnxModelSession.Create(files.GetFilePath(Qwen3EmbeddingProfile.GraphFile), new OnnxSessionOptions
        {
            IntraOpNumThreads = options.Threads,
            InterOpNumThreads = 1,
            ExpectedModelSha256 = profile.RuntimeLock.GetFile(Qwen3EmbeddingProfile.GraphFile).Sha256,
            KnownModelSha256 = files.GetVerifiedSha256(Qwen3EmbeddingProfile.GraphFile),
        });
        try
        {
            OutputName = ValidateSignature(Session.Inputs, Session.Outputs, profile.Task);
            _core = new EncoderCore(Session, "input_ids", "attention_mask", tokenTypeIdsName: null, OutputName);
        }
        catch
        {
            Session.Dispose();
            throw;
        }
    }

    public Qwen3EmbeddingProfile Profile { get; }

    public Qwen3Tokenization Tokenization { get; }

    public OnnxModelSession Session { get; }

    public string OutputName { get; }

    /// <summary>
    /// Runs the rows in chunks (<see cref="Qwen3EmbeddingOptions.MaxBatchSize"/> rows and
    /// <see cref="Qwen3EmbeddingOptions.MaxBatchTokens"/> padded cells at most, input order) and returns per row the
    /// L2-normalized last-token hidden state (embedding) or the score <c>yes - no</c> as a one-element array (reranker).
    /// </summary>
    public float[][] Predict(IReadOnlyList<Qwen3Input> inputs, CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(inputs);
        var lengths = new int[inputs.Count];
        for (int i = 0; i < lengths.Length; i++)
        {
            Tokenization.Validate(inputs[i] ?? throw new ArgumentException($"Input {i} is null.", nameof(inputs)));
            lengths[i] = inputs[i].Length;
        }

        var plan = BatchPlan.Create(lengths, _options.MaxBatchSize, _options.MaxBatchTokens, sortByLength: false);
        var results = new float[inputs.Count][];
        foreach (var chunk in plan.Chunks)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int[] rows = plan.Rows(chunk).ToArray();
            RunChunk(inputs, rows, chunk.Width, results, cancellationToken);
        }

        return results;
    }

    private void RunChunk(IReadOnlyList<Qwen3Input> inputs, int[] rows, int width, float[][] results, CancellationToken cancellationToken)
    {
        bool embedding = Profile.Task == Qwen3EmbeddingTask.Embedding;
        int perToken = embedding ? Qwen3EmbeddingProfile.HiddenSize : 2;
        _core.RunChunk(inputs, rows, width, static input => new EncoderRow(input.InputIds, input.AttentionMask, default),
            Qwen3EmbeddingProfile.PadTokenId, [rows.Length, width, perToken], (output, mask) =>
            {
                for (int r = 0; r < rows.Length; r++)
                {
                    int last = LastAttended(mask.Slice(r * width, width));
                    var state = output.Slice(((r * width) + last) * perToken, perToken);
                    results[rows[r]] = embedding ? Normalize(state) : [Score(state)];
                }
            }, cancellationToken);
    }

    /// <summary><c>sum(mask) - 1</c>: with right padding, the last attended position.</summary>
    internal static int LastAttended(ReadOnlySpan<long> mask)
    {
        long sum = 0;
        foreach (long m in mask)
        {
            sum += m;
        }

        return sum > 0 ? (int)sum - 1 : throw new InvalidDataException("A row has no attended token.");
    }

    /// <summary>L2 normalization (double accumulation, divided by <c>max(norm, 1e-12)</c> as torch's F.normalize).</summary>
    internal static float[] Normalize(ReadOnlySpan<float> state)
    {
        double sum = 0;
        foreach (float v in state)
        {
            if (!float.IsFinite(v))
            {
                throw new InvalidDataException("The graph returned a NaN or infinite hidden state.");
            }

            sum += (double)v * v;
        }

        double norm = Math.Max(Math.Sqrt(sum), 1e-12);
        var result = new float[state.Length];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = (float)(state[i] / norm);
        }

        return result;
    }

    /// <summary>The reranker score from the two-row head <c>[no, yes]</c>: <c>yes - no</c>.</summary>
    internal static float Score(ReadOnlySpan<float> noYes)
    {
        float score = noYes[1] - noYes[0];
        return float.IsFinite(score) ? score : throw new InvalidDataException("The graph returned a NaN or infinite logit.");
    }

    internal static string ValidateSignature(IReadOnlyList<OnnxTensorInfo> inputs, IReadOnlyList<OnnxTensorInfo> outputs, Qwen3EmbeddingTask task)
    {
        string[] names = ["input_ids", "attention_mask"];
        if (inputs.Count != 2 || inputs.Any(i => !names.Contains(i.Name, StringComparer.Ordinal) || i.ElementType != "Int64" || i.Dimensions.Count != 2)
            || inputs[0].Name == inputs[1].Name)
        {
            throw new NotSupportedException("The graph must take exactly input_ids and attention_mask (int64, rank 2).");
        }

        (string name, long last) = task == Qwen3EmbeddingTask.Embedding ? ("last_hidden_state", Qwen3EmbeddingProfile.HiddenSize) : ("yes_no_logits", 2L);
        var output = outputs.SingleOrDefault(o => o.Name == name);
        if (output is null || output.ElementType != "Float" || output.Dimensions.Count != 3 || output.Dimensions[^1] != last)
        {
            throw new NotSupportedException($"The graph output {name} must be float [batch, sequence, {last}].");
        }

        return name;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            Session.Dispose();
        }
    }
}
