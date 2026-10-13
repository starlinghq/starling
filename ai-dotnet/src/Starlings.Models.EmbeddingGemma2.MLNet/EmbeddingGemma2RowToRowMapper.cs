using Microsoft.ML;
using Microsoft.ML.Data;

namespace Starlings.Models.EmbeddingGemma2.MLNet;

/// <summary>Row-to-row mapping (prediction engines): always row-wise, one inference per row and text.</summary>
internal sealed class EmbeddingGemma2RowToRowMapper(EmbeddingGemma2Bindings bindings, EmbeddingGemma2Inference inference) : IRowToRowMapper
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

        return new EmbeddingGemma2MappedRow(bindings, bindings.Active(activeColumns), input, inference);
    }

    private sealed class EmbeddingGemma2MappedRow : DataViewRow
    {
        private readonly EmbeddingGemma2Bindings _bindings;
        private readonly bool[] _active;
        private readonly DataViewRow _input;
        private readonly RowWiseState? _current;

        public EmbeddingGemma2MappedRow(EmbeddingGemma2Bindings bindings, bool[] active, DataViewRow input, EmbeddingGemma2Inference inference)
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
            EmbeddingGemma2Getters.Create<TValue>(_bindings, _active, _input, _current, column);
    }
}
