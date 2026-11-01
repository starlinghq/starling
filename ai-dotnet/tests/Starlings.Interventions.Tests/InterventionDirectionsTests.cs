using static Starlings.Interventions.Tests.TestData;

namespace Starlings.Interventions.Tests;

public sealed class InterventionDirectionsTests
{
    [Fact]
    public void DecodesLittleEndianFloat32RowMajor()
    {
        // [[1, -2], [0.5, 0]] as little-endian binary32.
        byte[] bytes =
        [
            0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0xC0,
            0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00,
        ];
        InterventionDirections q = InterventionDirections.Decode(bytes, rank: 2, hiddenSize: 2);
        Assert.Equal([1f, -2f], q.Row(0).ToArray());
        Assert.Equal([0.5f, 0f], q.Row(1).ToArray());
        Assert.Equal(bytes, q.Encode());
    }

    [Theory]
    [InlineData(15)]
    [InlineData(17)]
    [InlineData(0)]
    public void RejectsWrongByteLength(int length)
    {
        Assert.Throws<InterventionContractException>(() => InterventionDirections.Decode(new byte[length], rank: 2, hiddenSize: 2));
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void RejectsNonFiniteValues(float bad)
    {
        float[] values = [1f, 0f, 0f, bad];
        Assert.Throws<InterventionContractException>(() => InterventionDirections.Create(2, 2, values));
        byte[] bytes = new byte[16];
        Buffer.BlockCopy(values, 0, bytes, 0, 16);
        Assert.Throws<InterventionContractException>(() => InterventionDirections.Decode(bytes, 2, 2));
    }

    [Fact]
    public void RejectsInvalidShapes()
    {
        Assert.Throws<InterventionContractException>(() => InterventionDirections.Create(3, 2, new float[6]));
        Assert.Throws<InterventionContractException>(() => InterventionDirections.Create(0, 2, []));
        Assert.Throws<InterventionContractException>(() => InterventionDirections.Create(1, 2, new float[3]));
    }

    [Fact]
    public void OrthonormalityDiagnostic()
    {
        InterventionDirections q = InterventionDirections.Create(3, 16, Orthonormal(3, 16, seed: 7));
        Assert.True(q.MaxOrthonormalityError() < 1e-6, q.MaxOrthonormalityError().ToString(System.Globalization.CultureInfo.InvariantCulture));

        // Not orthonormal, but the contract does not reject it: the diagnostic reports it.
        InterventionDirections scaled = InterventionDirections.Create(1, 2, [2f, 0f]);
        Assert.Equal(3.0, scaled.MaxOrthonormalityError(), 12);
    }

    [Fact]
    public void LoadsAndVerifiesThroughTheProfileFile()
    {
        using TempDir dir = new();
        byte[] layer1 = InterventionDirections.Create(2, 8, Orthonormal(2, 8, seed: 1)).Encode();
        byte[] layer3 = InterventionDirections.Create(1, 8, Orthonormal(1, 8, seed: 3)).Encode();
        dir.Write("directions/layer1.f32", layer1);
        dir.Write("directions/layer3.f32", layer3);
        InterventionProfile profile = new(
            "synthetic-demo",
            BaseSha,
            InterventionOperatorKind.DirectionalProjection,
            [
                Site(1, rank: 2, scales: [1f, 0.5f], sha: InterventionHashes.ComputeSha256(layer1)),
                Site(3, sha: InterventionHashes.ComputeSha256(layer3)),
            ]);
        byte[] json = profile.Serialize();
        string path = dir.Write("profile.json", json);

        InterventionProfileFile file = InterventionProfileFile.Load(path);
        Assert.Equal(InterventionHashes.ComputeSha256(json), file.Sha256);
        file.VerifyDirections();
        InterventionDirections q = file.LoadDirections(file.Profile.Sites[0]);
        Assert.Equal(2, q.Rank);
        Assert.Equal(layer1, q.Encode());

        // One flipped bit: the hash check fails before the values are used.
        layer3[5] ^= 0x01;
        dir.Write("directions/layer3.f32", layer3);
        Assert.Throws<InterventionMismatchException>(file.VerifyDirections);

        // Truncated: the size check fails first.
        dir.Write("directions/layer3.f32", layer3[..^4]);
        Assert.Throws<InterventionContractException>(file.VerifyDirections);

        File.Delete(Path.Combine(dir.Path, "directions", "layer3.f32"));
        Assert.Throws<FileNotFoundException>(file.VerifyDirections);
    }

    [Fact]
    public void ProfileHashIsOverStoredBytes()
    {
        using TempDir dir = new();
        // A valid but non-canonical document: the file hash is of these bytes, not of the re-serialization.
        string compact = Text(Profile().Serialize()).Replace("\n", "", StringComparison.Ordinal).Replace("  ", "", StringComparison.Ordinal);
        string path = dir.Write("profile.json", Utf8(compact));
        InterventionProfileFile file = InterventionProfileFile.Load(path);
        Assert.Equal(InterventionHashes.ComputeSha256(Utf8(compact)), file.Sha256);
        Assert.NotEqual(InterventionHashes.ComputeSha256(file.Profile.Serialize()), file.Sha256);
    }

    [Fact]
    public void DirectionsPathCannotLeaveTheProfileDirectory()
    {
        Assert.Throws<InterventionContractException>(() => Internal.StrictJson.Resolve("/tmp/x", "../y"));
        Assert.Throws<InterventionContractException>(() => Internal.StrictJson.Resolve("/tmp/x", "/etc/passwd"));
        Assert.EndsWith(Path.Combine("x", "a", "b.f32"), Internal.StrictJson.Resolve("/tmp/x", "a/b.f32"), StringComparison.Ordinal);
    }
}
