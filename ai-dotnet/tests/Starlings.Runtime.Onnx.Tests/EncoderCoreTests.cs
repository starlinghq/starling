namespace Starlings.Runtime.Onnx.Tests;

/// <summary>
/// The packing and pooling the text encoders share (<see cref="EncoderCore"/>), against hand-computed values.
/// Weight-free; the model packages' parity tests cover the same code on the real graphs.
/// </summary>
public class EncoderCoreTests
{
    private static EncoderRow Row((long[] Ids, long[]? Mask, long[]? Types) input) =>
        new(input.Ids, input.Mask ?? default(ReadOnlyMemory<long>), input.Types ?? default(ReadOnlyMemory<long>));

    [Fact]
    public void PackingRightPadsIdsMaskAndTypesInRowOrder()
    {
        (long[] Ids, long[]? Mask, long[]? Types)[] inputs =
        [
            ([5, 6], null, null),
            ([7, 8, 9], [1, 1, 0], [0, 1, 1]),
            ([4], null, null),
        ];
        var ids = new long[6];
        var mask = new long[6];
        var types = new long[6];
        Array.Fill(types, 99);

        EncoderCore.PackRightPadded(inputs, [1, 0], width: 3, padId: 1, Row, ids, mask, types);

        Assert.Equal([7, 8, 9, 5, 6, 1], ids);
        Assert.Equal([1, 1, 0, 1, 1, 0], mask);
        Assert.Equal([0, 1, 1, 0, 0, 0], types);
    }

    [Fact]
    public void PackingWithoutTypesLeavesTheTypeBufferUnread()
    {
        (long[] Ids, long[]? Mask, long[]? Types)[] inputs = [([3], null, null)];
        var ids = new long[2];
        var mask = new long[2];

        EncoderCore.PackRightPadded(inputs, [0], width: 2, padId: 0, Row, ids, mask, tokenTypes: default);

        Assert.Equal([3, 0], ids);
        Assert.Equal([1, 0], mask);
    }

    [Fact]
    public void MeanPoolingAveragesAttendedPositionsInDoubleAndNormalizes()
    {
        // Two positions of width 2 attended, one masked: mean (2, 4) / 2 ... then L2.
        float[] hidden = [1, 3, 3, 5, 100, 100];
        long[] mask = [1, 1, 0];
        var destination = new float[2];

        int attended = EncoderCore.Pool(hidden, mask, width: 2, outputDimension: 2, EncoderPooling.Mean, normalize: true, destination);

        double x = 2, y = 4, norm = Math.Sqrt((x * x) + (y * y));
        Assert.Equal(2, attended);
        Assert.Equal([(float)(x / norm), (float)(y / norm)], destination);
    }

    [Fact]
    public void ClsPoolingTakesTheFirstPositionAndIgnoresTheMask()
    {
        float[] hidden = [3, 4, 10, 10];
        var normalized = new float[2];
        var raw = new float[2];

        EncoderCore.Pool(hidden, [0, 1], width: 2, outputDimension: 2, EncoderPooling.Cls, normalize: true, normalized);
        EncoderCore.Pool(hidden, [0, 1], width: 2, outputDimension: 2, EncoderPooling.Cls, normalize: false, raw);

        Assert.Equal([0.6f, 0.8f], normalized);
        Assert.Equal([3f, 4f], raw);
    }

    [Fact]
    public void ASmallerOutputDimensionNormalizesTheLeadingValues()
    {
        float[] hidden = [3, 4, 12, 3, 4, 12];
        var full = new float[3];
        var leading = new float[2];

        EncoderCore.Pool(hidden, [1, 1], width: 3, outputDimension: 3, EncoderPooling.Mean, normalize: true, full);
        EncoderCore.Pool(hidden, [1, 1], width: 3, outputDimension: 2, EncoderPooling.Mean, normalize: true, leading);

        Assert.Equal([3f / 13, 4f / 13, 12f / 13], full);
        Assert.Equal([0.6f, 0.8f], leading);
    }

    [Fact]
    public void AnAllZeroMaskPoolsToZerosAndReportsNoAttendedPosition()
    {
        var destination = new float[2];

        int attended = EncoderCore.Pool([1, 2], [0], width: 2, outputDimension: 2, EncoderPooling.Mean, normalize: true, destination);

        Assert.Equal(0, attended);
        Assert.Equal([0f, 0f], destination);
    }
}
