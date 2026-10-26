// Shared by Starlings.Models.E5.MLNet and Starlings.Models.EmbeddingGemma2.MLNet (compiled into each, internal):
// the ML.NET transform over any ITextEmbedder. Moved from the two packages' copies, which differed only in names.
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Starlings.Models.MLNet;

/// <summary>Row-to-row mapping (prediction engines): always row-wise, one inference per row and text.</summary>
internal sealed class TextEmbeddingRowToRowMapper(TextEmbeddingBindings bindings, TextEmbeddingInference inference) : IRowToRowMapper
{
    public DataViewSchema InputSchema => bindings.InputSchema;

    public DataViewSchema OutputSchema => bindings.OutputSchema;

    public IEnumerable<DataViewSchema.Column> GetDependencies(IEnumerable<DataViewSchema.Column> dependingColumns) =>
        bindings.InputColumnsNeeded(bindings.Active(dependingColumns));

    public DataViewRow GetRow(DataViewRow input, IEnumerable<DataViewSchema.Column> activeColumns)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (input.Schema != bindings.InputSchema)
        {
            throw new ArgumentException("The row's schema is not the schema this mapper was created for.", nameof(input));
        }

        return new TextEmbeddingMappedRow(bindings, bindings.Active(activeColumns), input, inference);
    }

    private sealed class TextEmbeddingMappedRow : DataViewRow
    {
        private readonly TextEmbeddingBindings _bindings;
        private readonly bool[] _active;
        private readonly DataViewRow _input;
        private readonly RowWiseState? _current;

        public TextEmbeddingMappedRow(TextEmbeddingBindings bindings, bool[] active, DataViewRow input, TextEmbeddingInference inference)
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
    }
}
