using System.Collections.Concurrent;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Starlings.Models.EmbeddingGemma2.MLNet.Tests;

/// <summary>
/// Stand-in tokenizer: each character becomes its code point plus 10 between <c>&lt;bos&gt;</c> (2) and
/// <c>&lt;eos&gt;</c> (1). Counts calls, so tests can tell how many rows were embedded.
/// </summary>
internal sealed class CountingTokenizer : IEmbeddingGemma2Tokenizer
{
    private int _calls;

    public ConcurrentQueue<string> Seen { get; } = new();

    public int Calls => Volatile.Read(ref _calls);

    public EmbeddingGemma2Input Tokenize(string text)
    {
        Interlocked.Increment(ref _calls);
        Seen.Enqueue(text);
        return new EmbeddingGemma2Input(Ids(text));
    }

    public static long[] Ids(string text)
    {
        var ids = new long[text.Length + 2];
        ids[0] = 2;
        for (int i = 0; i < text.Length; i++)
        {
            ids[i + 1] = text[i] + 10;
        }

        ids[^1] = 1;
        return ids;
    }
}

/// <summary>
/// An in-memory view with columns <c>Text</c> (text) and <c>Index</c> (int). Unlike ML.NET's enumerable views it
/// splits rows across a real cursor set (each cursor takes the next row from a shared counter), so parallel cursors
/// can be tested. Row ids are the row indices. With <see cref="ReverseEverySecondCursor"/>, every second cursor
/// returns the rows in reverse order (an input that violates the same-order assumption of look-ahead batching).
/// </summary>
internal sealed class TestRowsView : IDataView
{
    private readonly string[] _texts;
    private int _cursorsCreated;

    public TestRowsView(IEnumerable<string> texts)
    {
        _texts = texts.ToArray();
        var builder = new DataViewSchema.Builder();
        builder.AddColumn("Text", TextDataViewType.Instance);
        builder.AddColumn("Index", NumberDataViewType.Int32);
        Schema = builder.ToSchema();
    }

    public bool ReverseEverySecondCursor { get; init; }

    public bool CanShuffle => true;

    public DataViewSchema Schema { get; }

    public int TextReads => Volatile.Read(ref _textReads);

    private int _textReads;

    public long? GetRowCount() => _texts.Length;

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
    {
        int number = Interlocked.Increment(ref _cursorsCreated);
        bool reverse = ReverseEverySecondCursor && number % 2 == 0;
        int next = 0;
        return new Cursor(this, columnsNeeded, () =>
        {
            int k = next++;
            return k < _texts.Length ? (reverse ? _texts.Length - 1 - k : k) : -1;
        });
    }

    public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
    {
        var columns = columnsNeeded.ToList();
        int shared = -1;
        Func<int> take = () =>
        {
            int k = Interlocked.Increment(ref shared);
            return k < _texts.Length ? k : -1;
        };
        return Enumerable.Range(0, Math.Max(1, n)).Select(_ => (DataViewRowCursor)new Cursor(this, columns, take)).ToArray();
    }

    private sealed class Cursor(TestRowsView view, IEnumerable<DataViewSchema.Column> columnsNeeded, Func<int> take) : DataViewRowCursor
    {
        private readonly HashSet<int> _active = columnsNeeded.Select(c => c.Index).ToHashSet();
        private long _position = -1;
        private int _row = -1;

        public override DataViewSchema Schema => view.Schema;

        public override long Position => _position;

        public override long Batch => 0;

        public override ValueGetter<DataViewRowId> GetIdGetter() => (ref DataViewRowId id) => id = new DataViewRowId((ulong)_row, 0);

        public override bool IsColumnActive(DataViewSchema.Column column) => _active.Contains(column.Index);

        public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
        {
            if (!_active.Contains(column.Index))
            {
                throw new InvalidOperationException($"Column {column.Name} is not active.");
            }

            if (column.Index == 0)
            {
                ValueGetter<ReadOnlyMemory<char>> text = (ref ReadOnlyMemory<char> value) =>
                {
                    Interlocked.Increment(ref view._textReads);
                    value = view._texts[_row].AsMemory();
                };
                return (ValueGetter<TValue>)(Delegate)text;
            }

            ValueGetter<int> index = (ref int value) => value = _row;
            return (ValueGetter<TValue>)(Delegate)index;
        }

        public override bool MoveNext()
        {
            int row = take();
            if (row < 0)
            {
                _row = -1;
                return false;
            }

            _row = row;
            _position++;
            return true;
        }
    }
}

public sealed class TextRow
{
    public string? Text { get; set; }

    public int Id { get; set; }
}

public sealed class EmbeddedRow
{
    public string? Text { get; set; }

    public int Id { get; set; }

    [VectorType(768)]
    public float[] Embedding { get; set; } = [];

    public int TokenCount { get; set; }
}
