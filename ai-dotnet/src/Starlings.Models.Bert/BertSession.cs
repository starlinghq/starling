using Starlings.Acquisition;
using Starlings.Batching;
using Starlings.Runtime.Onnx;
using Starlings.Tokenizers;

namespace Starlings.Models.Bert;

internal sealed class BertSession : IDisposable
{
    private readonly Tokenizer _tokenizer;
    private readonly int _maxBatchSize;
    private readonly int? _maxBatchTokens;
    private readonly bool _sortByLength;
    private readonly EncoderCore _core;
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
        _tokenizer = Tokenizer.FromFile(files.GetFilePath(BertModelProfile.TokenizerFile))
            .WithPadding(null).WithTruncation(new TruncationOptions(profile.MaxTokens));
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
            _core = new EncoderCore(Session, "input_ids", "attention_mask", profile.TokenTypeIds ? "token_type_ids" : null, profile.OutputName);
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

        return RunPlanned(inputs, Plan(inputs, _maxBatchSize, _maxBatchTokens, _sortByLength), (chunk, width) => Run(chunk, width, cancellationToken), cancellationToken);
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

    /// <summary>
    /// Runs one chunk (its inputs in call order, right-padded to <paramref name="width"/>) and returns one result per
    /// input: the pooled embedding, or the reranker's logit as a one-element array. Padded positions get id 0, mask 0
    /// and type 0, as before the move to <see cref="EncoderCore"/>, whatever the profile's pad id (they are masked).
    /// </summary>
    private float[][] Run(BertInput[] inputs, int width, CancellationToken cancellationToken)
    {
        int count = inputs.Length;
        bool embeddings = Profile.Task == BertTask.Embeddings;
        long[] outputShape = embeddings ? [count, width, Profile.HiddenSize] : [count, 1];
        Span<int> rows = count <= 256 ? stackalloc int[count] : new int[count];
        for (int r = 0; r < count; r++)
        {
            rows[r] = r;
        }

        var results = new float[count][];
        _core.RunChunk(inputs, rows, width, ReadRow, padId: 0, outputShape, (output, mask) =>
        {
            foreach (float value in output)
            {
                if (!float.IsFinite(value))
                {
                    throw new InvalidDataException("BERT graph returned a NaN or infinite value.");
                }
            }

            for (int row = 0; row < count; row++)
            {
                results[row] = embeddings
                    ? Pool(output.Slice(row * width * Profile.HiddenSize, width * Profile.HiddenSize),
                        mask.Slice(row * width, width), Profile.Pooling, Profile.HiddenSize, Profile.Normalize)
                    : [output[row]];
            }
        }, cancellationToken);
        return results;
    }

    /// <summary>A row as <see cref="EncoderCore"/> packs it: ids, mask and segment ids, all of the row's length.</summary>
    private static EncoderRow ReadRow(BertInput input) => new(input.InputIds, input.AttentionMask, input.TokenTypeIds);

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

    /// <summary>
    /// Mean (over mask 1) or CLS pooling of one row, optionally L2-normalized: <see cref="EncoderCore.Pool"/>, with the
    /// BERT route's checks (shape and mode, a visible token for mean pooling, a finite result).
    /// </summary>
    internal static float[] Pool(ReadOnlySpan<float> hidden, ReadOnlySpan<long> mask, string pooling, int width, string normalize)
    {
        if (hidden.Length != mask.Length * width || mask.IsEmpty || pooling is not ("mean" or "cls"))
        {
            throw new ArgumentException("Invalid BERT pooling shape or mode.", nameof(hidden));
        }

        var mode = pooling == "cls" ? EncoderPooling.Cls : EncoderPooling.Mean;
        if (mode == EncoderPooling.Mean && !mask.ContainsAnyExcept(0L))
        {
            throw new ArgumentException("Mean pooling needs a visible token.", nameof(mask));
        }

        var result = new float[width];
        EncoderCore.Pool(hidden, mask, width, width, mode, normalize == "l2", result);
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
