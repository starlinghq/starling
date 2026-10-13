using Microsoft.ML;
using Microsoft.ML.Data;

namespace Starlings.Models.EmbeddingGemma2.MLNet;

/// <summary>
/// Maps an input schema to the output schema: every input column at its own index (including hidden ones), then the
/// embedding column, then the optional token-count column. An output name equal to an input name hides the input
/// column, as in every ML.NET transform.
/// </summary>
internal sealed class EmbeddingGemma2Bindings
{
    private EmbeddingGemma2Bindings(DataViewSchema input, DataViewSchema output, int textIndex, int tokenCountIndex, int dimension)
    {
        InputSchema = input;
        OutputSchema = output;
        TextIndex = textIndex;
        TokenCountIndex = tokenCountIndex;
        Dimension = dimension;
    }

    public DataViewSchema InputSchema { get; }

    public DataViewSchema OutputSchema { get; }

    public int InputCount => InputSchema.Count;

    /// <summary>Index of the text column in <see cref="InputSchema"/>.</summary>
    public int TextIndex { get; }

    public int EmbeddingIndex => InputSchema.Count;

    /// <summary>Index of the token-count column in <see cref="OutputSchema"/>, or -1.</summary>
    public int TokenCountIndex { get; }

    public int Dimension { get; }

    public DataViewSchema.Column TextColumn => InputSchema[TextIndex];

    public static EmbeddingGemma2Bindings Create(DataViewSchema input, EmbeddingGemma2ColumnNames names, int dimension)
    {
        ArgumentNullException.ThrowIfNull(input);
        var text = input.GetColumnOrNull(names.Input)
            ?? throw new ArgumentOutOfRangeException(nameof(input), $"Input column '{names.Input}' not found in the schema; the EmbeddingGemma 2 transform needs a text column of type {TextDataViewType.Instance}.");
        if (text.Type is not TextDataViewType)
        {
            throw new ArgumentOutOfRangeException(nameof(input), $"Schema mismatch for input column '{names.Input}': expected {TextDataViewType.Instance} (a scalar text column, ReadOnlyMemory<char> or string), got {text.Type}.");
        }

        var builder = new DataViewSchema.Builder();
        builder.AddColumns(input);
        builder.AddColumn(names.Output, new VectorDataViewType(NumberDataViewType.Single, dimension));
        if (names.TokenCount is not null)
        {
            builder.AddColumn(names.TokenCount, NumberDataViewType.Int32);
        }

        return new EmbeddingGemma2Bindings(input, builder.ToSchema(), text.Index, names.TokenCount is null ? -1 : input.Count + 1, dimension);
    }

    /// <summary>The SchemaShape counterpart of <see cref="Create"/>, for <c>IEstimator.GetOutputSchema</c>.</summary>
    public static SchemaShape CreateShape(SchemaShape input, EmbeddingGemma2ColumnNames names, int dimension)
    {
        ArgumentNullException.ThrowIfNull(input);
        var found = input.Where(c => string.Equals(c.Name, names.Input, StringComparison.Ordinal)).ToList();
        if (found.Count == 0)
        {
            throw new ArgumentOutOfRangeException(nameof(input), $"Input column '{names.Input}' not found in the schema; the EmbeddingGemma 2 transform needs a text column of type {TextDataViewType.Instance}.");
        }

        var text = found[0];
        if (text.Kind != SchemaShape.Column.VectorKind.Scalar || text.ItemType is not TextDataViewType)
        {
            string actual = text.Kind == SchemaShape.Column.VectorKind.Scalar ? $"{text.ItemType}" : $"{text.Kind} of {text.ItemType}";
            throw new ArgumentOutOfRangeException(nameof(input), $"Schema mismatch for input column '{names.Input}': expected {TextDataViewType.Instance} (a scalar text column), got {actual}.");
        }

        var columns = input.ToDictionary(c => c.Name, StringComparer.Ordinal);
        foreach (var column in OutputShapeColumns(names, dimension))
        {
            columns[column.Name] = column;
        }

        return new SchemaShape(columns.Values);
    }

    /// <summary>
    /// The new columns as <see cref="SchemaShape.Column"/>s. That type has no public constructor and
    /// <c>SchemaShape.Create(DataViewSchema)</c> is internal, so the columns come from a public ML.NET estimator that
    /// adds typed columns and reads none: a never-fitted custom mapping from an empty row type, whose output schema
    /// definition is renamed and retyped to the EmbeddingGemma 2 columns.
    /// </summary>
    private static SchemaShape OutputShapeColumns(EmbeddingGemma2ColumnNames names, int dimension)
    {
        var definition = SchemaDefinition.Create(typeof(ShapeRow), SchemaDefinition.Direction.Write);
        definition[nameof(ShapeRow.Embedding)].ColumnType = new VectorDataViewType(NumberDataViewType.Single, dimension);
        definition[nameof(ShapeRow.Embedding)].ColumnName = names.Output;
        if (names.TokenCount is null)
        {
            definition.RemoveAt(1);
        }
        else
        {
            definition[nameof(ShapeRow.TokenCount)].ColumnName = names.TokenCount;
        }

        var estimator = new MLContext().Transforms.CustomMapping<EmptyRow, ShapeRow>(static (_, _) => { }, contractName: null, outputSchemaDefinition: definition);
        return estimator.GetOutputSchema(new SchemaShape([]));
    }

    /// <summary>Which output columns are active, by output index.</summary>
    public bool[] Active(IEnumerable<DataViewSchema.Column> columnsNeeded)
    {
        ArgumentNullException.ThrowIfNull(columnsNeeded);
        var active = new bool[OutputSchema.Count];
        foreach (var column in columnsNeeded)
        {
            active[CheckColumn(column)] = true;
        }

        return active;
    }

    public bool AnyOutputActive(bool[] active) =>
        active[EmbeddingIndex] || (TokenCountIndex >= 0 && active[TokenCountIndex]);

    /// <summary>The input columns an output selection needs: the active pass-through columns, plus the text column if an output is active.</summary>
    public List<DataViewSchema.Column> InputColumnsNeeded(bool[] active)
    {
        var needed = new List<DataViewSchema.Column>();
        bool anyOutput = AnyOutputActive(active);
        for (int i = 0; i < InputCount; i++)
        {
            if (active[i] || (anyOutput && i == TextIndex))
            {
                needed.Add(InputSchema[i]);
            }
        }

        return needed;
    }

    public int CheckColumn(DataViewSchema.Column column)
    {
        if (column.Index < 0 || column.Index >= OutputSchema.Count || !string.Equals(OutputSchema[column.Index].Name, column.Name, StringComparison.Ordinal))
        {
            throw new ArgumentOutOfRangeException(nameof(column), $"Column '{column.Name}' (index {column.Index}) is not a column of this EmbeddingGemma 2 transform's output schema.");
        }

        return column.Index;
    }
}

/// <summary>Input, output and optional token-count column names, validated.</summary>
internal sealed record EmbeddingGemma2ColumnNames(string Output, string Input, string? TokenCount)
{
    public static EmbeddingGemma2ColumnNames Create(string outputColumnName, string? inputColumnName, string? tokenCountColumnName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputColumnName);
        inputColumnName ??= outputColumnName;
        ArgumentException.ThrowIfNullOrWhiteSpace(inputColumnName);
        if (tokenCountColumnName is not null && string.Equals(tokenCountColumnName, outputColumnName, StringComparison.Ordinal))
        {
            throw new ArgumentException($"The token-count column and the embedding column are both named '{outputColumnName}'.", nameof(tokenCountColumnName));
        }

        return new EmbeddingGemma2ColumnNames(outputColumnName, inputColumnName, tokenCountColumnName);
    }
}

#pragma warning disable CA1812, CA1819 // Instantiated and read only by ML.NET's schema reflection, never by this code.
internal sealed class EmptyRow
{
}

internal sealed class ShapeRow
{
    public float[] Embedding { get; set; } = [];

    public int TokenCount { get; set; }
}
#pragma warning restore CA1812, CA1819
