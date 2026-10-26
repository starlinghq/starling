// Shared by Starlings.Models.E5.MLNet and Starlings.Models.EmbeddingGemma2.MLNet (compiled into each, internal):
// the ML.NET transform over any ITextEmbedder. Moved from the two packages' copies, which differed only in names.
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Starlings.Models.MLNet;

/// <summary>Embeds texts for the transform; the single place inference happens.</summary>
internal sealed class TextEmbeddingInference(ITextEmbedder embedder, Action<int>? observer)
{
    public float[][] Embed(IReadOnlyList<string> texts, int[] tokenCounts)
    {
        observer?.Invoke(texts.Count);
        return embedder.EmbedBatch(texts, tokenCounts);
    }
}

/// <summary>
/// Prepends a fixed prefix to every text, then embeds with <paramref name="inner"/> (e5's <c>"query: "</c> and
/// <c>"passage: "</c>). With an empty prefix the texts pass through unchanged.
/// </summary>
internal sealed class PrefixedTextEmbedder(ITextEmbedder inner, string prefix) : ITextEmbedder
{
    public int Dimension => inner.Dimension;

    public int MaxSequenceLength => inner.MaxSequenceLength;

    public float[][] EmbedBatch(IReadOnlyList<string> texts, Span<int> tokenCounts, CancellationToken cancellationToken = default)
    {
        if (prefix.Length > 0)
        {
            var prefixed = new string[texts.Count];
            for (int i = 0; i < prefixed.Length; i++)
            {
                prefixed[i] = prefix + texts[i];
            }

            texts = prefixed;
        }

        return inner.EmbedBatch(texts, tokenCounts, cancellationToken);
    }

    public int CountTokens(string text) => inner.CountTokens(prefix + text);

    public int[] CountTokens(IReadOnlyList<string> texts) => inner.CountTokens(texts.Select(text => prefix + text).ToArray());

    public TokenLimitCheck Validate(string text) => inner.Validate(prefix + text);
}

/// <summary>The embedding of the row a cursor or row is on, computed at most once per row.</summary>
internal interface ITextEmbeddingCurrentRow
{
    float[] Vector();

    int TokenCount();
}

/// <summary>
/// Row-wise: the first getter call on a row reads the text and embeds it; later calls on the same row (same position
/// and same text) return the cached result. The text comparison also protects row mappers whose input position does
/// not advance (prediction engines).
/// </summary>
internal sealed class RowWiseState(TextEmbeddingInference inference, DataViewRow input, ValueGetter<ReadOnlyMemory<char>> text) : ITextEmbeddingCurrentRow
{
    private readonly int[] _count = new int[1];
    private long _position = -1;
    private string? _text;
    private float[] _vector = [];
    private ReadOnlyMemory<char> _buffer;

    public float[] Vector()
    {
        Ensure();
        return _vector;
    }

    public int TokenCount()
    {
        Ensure();
        return _count[0];
    }

    private void Ensure()
    {
        long position = input.Position;
        text(ref _buffer);
        if (_text is not null && position == _position && _buffer.Span.SequenceEqual(_text))
        {
            return;
        }

        string value = _buffer.ToString();
        _vector = inference.Embed([value], _count)[0];
        _text = value;
        _position = position;
    }
}

/// <summary>Getter dispatch shared by the cursors and the mapper row.</summary>
internal static class TextEmbeddingGetters
{
    public static ValueGetter<TValue> Create<TValue>(TextEmbeddingBindings bindings, bool[] active, DataViewRow input, ITextEmbeddingCurrentRow? current, DataViewSchema.Column column)
    {
        int index = bindings.CheckColumn(column);
        if (!active[index])
        {
            throw new InvalidOperationException($"Column '{column.Name}' is not active in this cursor; request it in columnsNeeded.");
        }

        if (index < bindings.InputCount)
        {
            return input.GetGetter<TValue>(bindings.InputSchema[index]);
        }

        if (index == bindings.EmbeddingIndex)
        {
            if (typeof(TValue) != typeof(VBuffer<float>))
            {
                throw new InvalidOperationException($"Column '{column.Name}' is {column.Type}; request a getter of VBuffer<Single>, not {typeof(TValue).Name}.");
            }

            ValueGetter<VBuffer<float>> getter = (ref VBuffer<float> destination) =>
            {
                float[] vector = current!.Vector();
                var editor = VBufferEditor.Create(ref destination, vector.Length);
                vector.CopyTo(editor.Values);
                destination = editor.Commit();
            };
            return (ValueGetter<TValue>)(Delegate)getter;
        }

        if (typeof(TValue) != typeof(int))
        {
            throw new InvalidOperationException($"Column '{column.Name}' is {column.Type}; request a getter of Int32, not {typeof(TValue).Name}.");
        }

        ValueGetter<int> counter = (ref int destination) => destination = current!.TokenCount();
        return (ValueGetter<TValue>)(Delegate)counter;
    }
}

/// <summary>The lazily transformed view. Nothing is computed until a cursor's output getter is called.</summary>
internal sealed class TextEmbeddingDataView : IDataView
{
    private readonly IDataView _source;
    private readonly TextEmbeddingBindings _bindings;
    private readonly TextEmbeddingInference _inference;
    private readonly int _batchSize;

    public TextEmbeddingDataView(IDataView source, TextEmbeddingBindings bindings, TextEmbeddingInference inference, int batchSize)
    {
        _source = source;
        _bindings = bindings;
        _inference = inference;
        _batchSize = batchSize;
    }

    public bool CanShuffle => _source.CanShuffle;

    public DataViewSchema Schema => _bindings.OutputSchema;

    public long? GetRowCount() => _source.GetRowCount();

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
    {
        var active = _bindings.Active(columnsNeeded);
        var input = _source.GetRowCursor(_bindings.InputColumnsNeeded(active), rand);
        try
        {
            if (UsesLookAhead(active, rand))
            {
                return new TextEmbeddingBatchedCursor(_bindings, active, input, _source, _inference, _batchSize);
            }

            return new TextEmbeddingRowCursor(_bindings, active, input, _inference);
        }
        catch
        {
            input.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Row-wise: one wrapped cursor per input cursor, embedding concurrently on the shared (thread-safe) encoder.
    /// With look-ahead batching the order of the input cursors cannot be paired, so a single batched cursor is
    /// returned (the contract allows fewer cursors than requested).
    /// </summary>
    public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
    {
        var active = _bindings.Active(columnsNeeded);
        if (UsesLookAhead(active, rand))
        {
            return [GetRowCursor(_bindings.OutputSchema.Where(c => active[c.Index]), rand)];
        }

        var inputs = _source.GetRowCursorSet(_bindings.InputColumnsNeeded(active), n, rand);
        var cursors = new DataViewRowCursor[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            cursors[i] = new TextEmbeddingRowCursor(_bindings, active, inputs[i], _inference);
        }

        return cursors;
    }

    private bool UsesLookAhead(bool[] active, Random? rand) => _batchSize > 1 && rand is null && _bindings.AnyOutputActive(active);
}

/// <summary>Row-wise cursor: pass-through getters from the input cursor, one inference per row on first use.</summary>
internal sealed class TextEmbeddingRowCursor : DataViewRowCursor
{
    private readonly TextEmbeddingBindings _bindings;
    private readonly bool[] _active;
    private readonly DataViewRowCursor _input;
    private readonly RowWiseState? _current;

    public TextEmbeddingRowCursor(TextEmbeddingBindings bindings, bool[] active, DataViewRowCursor input, TextEmbeddingInference inference)
    {
        _bindings = bindings;
        _active = active;
        _input = input;
        if (bindings.AnyOutputActive(active))
        {
            _current = new RowWiseState(inference, input, input.GetGetter<ReadOnlyMemory<char>>(bindings.TextColumn));
        }
    }

    public override DataViewSchema Schema => _bindings.OutputSchema;

    public override long Position => _input.Position;

    public override long Batch => _input.Batch;

    public override ValueGetter<DataViewRowId> GetIdGetter() => _input.GetIdGetter();

    public override bool IsColumnActive(DataViewSchema.Column column) => _active[_bindings.CheckColumn(column)];

    public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column) =>
        TextEmbeddingGetters.Create<TValue>(_bindings, _active, _input, _current, column);

    public override bool MoveNext() => _input.MoveNext();

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _input.Dispose();
        }

        base.Dispose(disposing);
    }
}

/// <summary>
/// Look-ahead batching on a sequential, unshuffled cursor. A second input cursor, over the text column only, runs
/// ahead; when the main cursor's first output getter is called on a row outside the current batch, the next
/// <c>batchSize</c> texts are read from the look-ahead cursor and embedded with one call. Each row's id is compared
/// with the id the look-ahead cursor saw for that row; a mismatch (an input whose cursors do not return rows in the
/// same order) throws instead of returning another row's vector. Upstream transforms of the text column run twice.
/// </summary>
internal sealed class TextEmbeddingBatchedCursor : DataViewRowCursor, ITextEmbeddingCurrentRow
{
    private readonly TextEmbeddingBindings _bindings;
    private readonly bool[] _active;
    private readonly DataViewRowCursor _input;
    private readonly DataViewRowCursor _ahead;
    private readonly TextEmbeddingInference _inference;
    private readonly int _batchSize;
    private readonly ValueGetter<DataViewRowId> _inputId;
    private readonly ValueGetter<DataViewRowId> _aheadId;
    private readonly ValueGetter<ReadOnlyMemory<char>> _aheadText;
    private readonly DataViewRowId[] _ids;
    private readonly string[] _texts;
    private long _row = -1;
    private long _aheadRow = -1;
    private long _batchStart;
    private int _batchCount;
    private long _verifiedRow = -1;
    private float[][] _vectors = [];
    private int[] _counts = [];

    public TextEmbeddingBatchedCursor(TextEmbeddingBindings bindings, bool[] active, DataViewRowCursor input, IDataView source, TextEmbeddingInference inference, int batchSize)
    {
        _bindings = bindings;
        _active = active;
        _input = input;
        _inference = inference;
        _batchSize = batchSize;
        _ids = new DataViewRowId[batchSize];
        _texts = new string[batchSize];
        _inputId = input.GetIdGetter();
        _ahead = source.GetRowCursor([bindings.TextColumn], rand: null);
        _aheadId = _ahead.GetIdGetter();
        _aheadText = _ahead.GetGetter<ReadOnlyMemory<char>>(bindings.TextColumn);
    }

    public override DataViewSchema Schema => _bindings.OutputSchema;

    public override long Position => _input.Position;

    public override long Batch => _input.Batch;

    public override ValueGetter<DataViewRowId> GetIdGetter() => _inputId;

    public override bool IsColumnActive(DataViewSchema.Column column) => _active[_bindings.CheckColumn(column)];

    public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column) =>
        TextEmbeddingGetters.Create<TValue>(_bindings, _active, _input, this, column);

    public override bool MoveNext()
    {
        if (!_input.MoveNext())
        {
            return false;
        }

        _row++;
        return true;
    }

    // Ensure first: it may replace the arrays (an element access would read the field before the call).
    public float[] Vector()
    {
        int offset = Ensure();
        return _vectors[offset];
    }

    public int TokenCount()
    {
        int offset = Ensure();
        return _counts[offset];
    }

    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _ahead.Dispose();
            _input.Dispose();
        }

        base.Dispose(disposing);
    }

    private int Ensure()
    {
        if (_row < 0)
        {
            throw new InvalidOperationException("The cursor is not on a row; call MoveNext first.");
        }

        if (_row < _batchStart || _row >= _batchStart + _batchCount)
        {
            Fill();
        }

        int offset = (int)(_row - _batchStart);
        if (_verifiedRow != _row)
        {
            DataViewRowId id = default;
            _inputId(ref id);
            if (!id.Equals(_ids[offset]))
            {
                throw new InvalidOperationException($"Look-ahead batching: row {_row} has id {id} in the main cursor but {_ids[offset]} in the look-ahead cursor. The input view does not return rows in the same order to two cursors; use BatchSize = 1.");
            }

            _verifiedRow = _row;
        }

        return offset;
    }

    private void Fill()
    {
        while (_aheadRow < _row)
        {
            if (!_ahead.MoveNext())
            {
                throw new InvalidOperationException($"Look-ahead batching: the look-ahead cursor ended before row {_row}. The input view does not return the same rows to two cursors; use BatchSize = 1.");
            }

            _aheadRow++;
        }

        int count = 0;
        ReadOnlyMemory<char> text = default;
        while (true)
        {
            _aheadText(ref text);
            _texts[count] = text.ToString();
            _aheadId(ref _ids[count]);
            count++;
            if (count == _batchSize || !_ahead.MoveNext())
            {
                break;
            }

            _aheadRow++;
        }

        var batch = count == _batchSize ? _texts : _texts[..count];
        _counts = new int[count];
        _vectors = _inference.Embed(batch, _counts);
        _batchStart = _row;
        _batchCount = count;
    }
}
