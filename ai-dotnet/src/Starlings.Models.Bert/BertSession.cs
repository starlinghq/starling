using Starlings.Acquisition;
using Starlings.Batching;
using Starlings.Runtime.Onnx;
using Starlings.Tokenizers;
using Microsoft.ML.OnnxRuntime;

namespace Starlings.Models.Bert;

internal sealed class BertSession : IDisposable
{
    private readonly Tokenizer _tokenizer;
    private readonly int _maxBatchSize;
    private readonly int? _maxBatchTokens;
    private readonly bool _sortByLength;
    private int _disposed;

    public BertSession(BertModelProfile profile, string snapshot, int threads, int maxBatchSize, int? maxBatchTokens = null, bool sortByLength = false)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentOutOfRangeException.ThrowIfLessThan(threads, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(threads, 64);
        ValidateBatching(maxBatchSize, maxBatchTokens);
        Profile = profile;
        _maxBatchSize = maxBatchSize;
        _maxBatchTokens = maxBatchTokens;
        _sortByLength = sortByLength;
        var files = ModelAssetStore.VerifyDirectory(profile.RuntimeLock, snapshot, CacheVerification.Sha256);
        // The profile requires truncation "longest_first" (BertModelProfile.Parse); stated here rather than left to the
        // library's default, which is the same strategy and direction.
        _tokenizer = Tokenizer.FromFile(files.GetFilePath(BertModelProfile.TokenizerFile))
            .WithPadding(null).WithTruncation(new TruncationOptions(profile.MaxTokens, TruncationStrategy.LongestFirst, TruncationDirection.Right));
        Session = OnnxModelSession.Create(files.GetFilePath(BertModelProfile.GraphFile), new OnnxSessionOptions
        {
            IntraOpNumThreads = threads,
            InterOpNumThreads = 1,
            ExpectedModelSha256 = profile.RuntimeLock.GetFile(BertModelProfile.GraphFile).Sha256,
        });
        try
        {
            ValidateSignature(Session.Inputs, Session.Outputs, profile.Task, profile.HiddenSize,
                profile.OutputName, profile.TokenTypeIds);
        }
        catch
        {
            Session.Dispose();
            throw;
        }
    }

    public BertModelProfile Profile { get; }
    public OnnxModelSession Session { get; }

    public BertInput Tokenize(string text, string? pair = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(text);
        var encoding = pair is null ? _tokenizer.Encode(text) : _tokenizer.Encode(text, pair);
        return new BertInput(encoding.Ids.Select(i => (long)i).ToArray(),
            encoding.AttentionMask.Select(i => (long)i).ToArray(), encoding.TypeIds.Select(i => (long)i).ToArray());
    }

    public float[][] Predict(IReadOnlyList<BertInput> inputs, CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(inputs);
        foreach (var input in inputs)
        {
            ValidateInput(input, Profile.MaxTokens, Profile.VocabularySize, Profile.PadTokenId,
                Profile.ClsTokenId, Profile.SepTokenId);
        }

        return RunPlanned(inputs, Plan(inputs, _maxBatchSize, _maxBatchTokens, _sortByLength), (chunk, width) =>
        {
            float[] raw = Run(chunk, cancellationToken);
            var rows = new float[chunk.Length][];
            for (int row = 0; row < chunk.Length; row++)
            {
                rows[row] = Profile.Task == BertTask.Reranking ? [raw[row]]
                    : Pool(raw.AsSpan(row * width * Profile.HiddenSize, width * Profile.HiddenSize),
                        PadMask(chunk[row], width), Profile.Pooling, Profile.HiddenSize, Profile.Normalize);
            }

            return rows;
        }, cancellationToken);
    }

    /// <summary>
    /// Runs <paramref name="plan"/>'s chunks in order with <paramref name="runChunk"/> (the chunk's inputs in call order
    /// and its padded width; one result per input) and returns the results in input order. Cancellation is checked
    /// before each chunk.
    /// </summary>
    internal static T[] RunPlanned<T>(IReadOnlyList<BertInput> inputs, BatchPlan plan, Func<BertInput[], int, T[]> runChunk, CancellationToken cancellationToken)
    {
        var results = new T[inputs.Count];
        foreach (var batch in plan.Chunks)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var rows = plan.Rows(batch);
            var chunk = new BertInput[rows.Length];
            for (int r = 0; r < rows.Length; r++)
            {
                chunk[r] = inputs[rows[r]];
            }

            var output = runChunk(chunk, batch.Width);
            for (int r = 0; r < rows.Length; r++)
            {
                results[rows[r]] = output[r];
            }
        }

        return results;
    }

    /// <summary>
    /// The chunk plan of a batch: at most <paramref name="maxRows"/> rows and, when set, <paramref name="maxCells"/>
    /// padded cells per call, over rows in ascending length order when <paramref name="sortByLength"/> is set.
    /// </summary>
    internal static BatchPlan Plan(IReadOnlyList<BertInput> inputs, int maxRows, int? maxCells, bool sortByLength)
    {
        var lengths = new int[inputs.Count];
        for (int i = 0; i < lengths.Length; i++)
        {
            lengths[i] = inputs[i].Length;
        }

        return BatchPlan.Create(lengths, maxRows, maxCells, sortByLength);
    }

    internal static void ValidateBatching(int maxBatchSize, int? maxBatchTokens)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxBatchSize, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(maxBatchSize, 128);
        if (maxBatchTokens is { } tokens)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(tokens, 1, nameof(maxBatchTokens));
        }
    }

    private static long[] PadMask(BertInput input, int width)
    {
        var mask = new long[width];
        input.AttentionMask.Span.CopyTo(mask);
        return mask;
    }

    private float[] Run(BertInput[] inputs, CancellationToken cancellationToken)
    {
        int count = inputs.Length;
        int width = inputs.Max(i => i.Length);
        int cells = checked(count * width);
        var ids = new long[cells];
        var mask = new long[cells];
        var types = new long[cells];
        for (int i = 0; i < count; i++)
        {
            var input = inputs[i];
            input.InputIds.Span.CopyTo(ids.AsSpan(i * width, input.Length));
            input.AttentionMask.Span.CopyTo(mask.AsSpan(i * width, input.Length));
            input.TokenTypeIds.Span.CopyTo(types.AsSpan(i * width, input.Length));
        }

        long[] shape = [count, width];
        using var idsValue = OrtValue.CreateTensorValueFromMemory(ids, shape);
        using var maskValue = OrtValue.CreateTensorValueFromMemory(mask, shape);
        using var typesValue = OrtValue.CreateTensorValueFromMemory(types, shape);
        int length = Profile.Task == BertTask.Embeddings ? checked(cells * Profile.HiddenSize) : count;
        var values = new float[length];
        long[] outputShape = Profile.Task == BertTask.Embeddings ? [count, width, Profile.HiddenSize] : [count, 1];
        using var output = OrtValue.CreateTensorValueFromMemory(values, outputShape);
        if (Profile.TokenTypeIds)
        {
            Session.Run(["input_ids", "attention_mask", "token_type_ids"], [idsValue, maskValue, typesValue],
                [Profile.OutputName], [output], cancellationToken);
        }
        else
        {
            Session.Run(["input_ids", "attention_mask"], [idsValue, maskValue],
                [Profile.OutputName], [output], cancellationToken);
        }
        if (values.Any(v => !float.IsFinite(v)))
        {
            throw new InvalidDataException("BERT graph returned a NaN or infinite value.");
        }

        return values;
    }

    internal static void ValidateSignature(IReadOnlyList<OnnxTensorInfo> inputs, IReadOnlyList<OnnxTensorInfo> outputs,
        BertTask task, int hiddenSize, string outputName, bool tokenTypeIds)
    {
        string[] names = tokenTypeIds ? ["input_ids", "attention_mask", "token_type_ids"] : ["input_ids", "attention_mask"];
        if (inputs.Count != names.Length || inputs.Select(i => i.Name).Distinct(StringComparer.Ordinal).Count() != names.Length
            || inputs.Any(i => !names.Contains(i.Name, StringComparer.Ordinal)
            || i.ElementType != "Int64" || i.Dimensions.Count != 2))
        {
            throw new NotSupportedException($"BERT graph requires exactly {string.Join(", ", names)} (int64 rank 2).");
        }

        var output = outputs.SingleOrDefault(o => o.Name == outputName);
        int rank = task == BertTask.Embeddings ? 3 : 2;
        int last = task == BertTask.Embeddings ? hiddenSize : 1;
        if (output is null || output.ElementType != "Float" || output.Dimensions.Count != rank || output.Dimensions[^1] != last)
        {
            throw new NotSupportedException($"BERT graph output {outputName} must be float rank {rank} with final dimension {last}.");
        }
    }

    internal static void ValidateInput(BertInput input, int maxTokens, int vocabularySize, int padTokenId,
        int clsTokenId, int sepTokenId)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (input.Length < 2 || input.Length > maxTokens || input.AttentionMask.Length != input.Length
            || input.TokenTypeIds.Length != input.Length)
        {
            throw new ArgumentException("BERT ids, mask, and type ids require equal lengths within the declared token limit.", nameof(input));
        }

        bool padding = false;
        int visible = 0;
        for (int i = 0; i < input.Length; i++)
        {
            long id = input.InputIds.Span[i], mask = input.AttentionMask.Span[i], type = input.TokenTypeIds.Span[i];
            if (id < 0 || id >= vocabularySize || mask is not (0 or 1) || type is not (0 or 1)
                || (mask == 1 && padding) || (mask == 0 && (id != padTokenId || type != 0)))
            {
                throw new ArgumentException("Invalid BERT token id, segment id, mask, or right padding.", nameof(input));
            }

            padding |= mask == 0;
            visible += (int)mask;
        }

        if (visible < 2 || input.InputIds.Span[0] != clsTokenId || input.InputIds.Span[visible - 1] != sepTokenId)
        {
            throw new ArgumentException("BERT rows must contain CLS first and SEP as the last visible token.", nameof(input));
        }
    }

    internal static float[] Pool(ReadOnlySpan<float> hidden, ReadOnlySpan<long> mask, string pooling, int width, string normalize)
    {
        if (hidden.Length != mask.Length * width || mask.IsEmpty || pooling is not ("mean" or "cls"))
        {
            throw new ArgumentException("Invalid BERT pooling shape or mode.", nameof(hidden));
        }

        Span<double> values = stackalloc double[width];
        values.Clear();
        if (pooling == "cls")
        {
            for (int d = 0; d < width; d++)
            {
                values[d] = hidden[d];
            }
        }
        else
        {
            long count = 0;
            for (int token = 0; token < mask.Length; token++)
            {
                if (mask[token] == 0)
                {
                    continue;
                }

                count++;
                for (int d = 0; d < width; d++)
                {
                    values[d] += hidden[token * width + d];
                }
            }

            if (count == 0)
            {
                throw new ArgumentException("Mean pooling needs a visible token.", nameof(mask));
            }

            for (int d = 0; d < width; d++)
            {
                values[d] /= count;
            }
        }

        var result = new float[width];
        if (normalize == "l2")
        {
            double norm = 0;
            for (int d = 0; d < width; d++)
            {
                norm += values[d] * values[d];
            }

            norm = Math.Max(Math.Sqrt(norm), 1e-12);
            for (int d = 0; d < width; d++)
            {
                result[d] = (float)(values[d] / norm);
            }
        }
        else
        {
            for (int d = 0; d < width; d++)
            {
                result[d] = (float)values[d];
            }
        }

        for (int d = 0; d < width; d++)
        {
            if (!float.IsFinite(result[d]))
            {
                throw new InvalidDataException("Pooling returned a non-finite value.");
            }
        }

        return result;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            Session.Dispose();
        }
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
}
