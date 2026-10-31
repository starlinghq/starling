using static Starlings.Interventions.Tests.TestData;

namespace Starlings.Interventions.Tests;

public sealed class InterventionProfileTests
{
    [Fact]
    public void SerializesToTheDocumentedCanonicalForm()
    {
        Assert.Equal(CanonicalProfile, Text(Profile().Serialize()));
    }

    [Fact]
    public void ParseThenSerializeIsByteIdentical()
    {
        byte[] canonical = Utf8(CanonicalProfile);
        InterventionProfile profile = InterventionProfile.Parse(canonical);
        Assert.Equal(canonical, profile.Serialize());
        Assert.Equal("synthetic-demo", profile.Id);
        Assert.Equal([1, 3], profile.Sites.Select(s => s.Layer));
        Assert.Equal([1f, 0.5f], profile.Sites[0].Scales);
        Assert.Equal(2 * 8 * 4, profile.Sites[0].DirectionsByteLength);
    }

    [Fact]
    public void PermutedKeysAndSitesSerializeToTheSameBytes()
    {
        const string permuted = """
            {"sites":[
              {"scales":[0.1],"rank":1,"layer":3,"hidden_size":8,"target":"residual_post","directions_sha256":"1111111111111111111111111111111111111111111111111111111111111111","dtype":"float32","directions_file":"directions/layer3.f32"},
              {"dtype":"float32","directions_file":"directions/layer1.f32","directions_sha256":"1111111111111111111111111111111111111111111111111111111111111111","scales":[1.0,5e-1],"layer":1,"target":"residual_post","hidden_size":8,"rank":2}],
             "operator":"directional_projection","base_model_sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
             "description":"Synthetic random directions for tests","id":"synthetic-demo","schema_version":1}
            """;
        Assert.Equal(CanonicalProfile, Text(InterventionProfile.Parse(Utf8(permuted)).Serialize()));
    }

    [Theory]
    [InlineData(0.1f)]
    [InlineData(1f / 3)]
    [InlineData(float.Epsilon)]
    [InlineData(float.MaxValue)]
    [InlineData(float.MinValue)]
    [InlineData(-0.0f)]
    [InlineData(1.17549435e-38f)]
    [InlineData(16777217f)]
    [InlineData(123456.789f)]
    public void ScalesRoundTripBitExactly(float value)
    {
        InterventionProfile profile = new("p", BaseSha, InterventionOperatorKind.DirectionalProjection, [Site(0, scales: [value])]);
        byte[] json = profile.Serialize();
        float parsed = InterventionProfile.Parse(json).Sites[0].Scales[0];
        Assert.Equal(BitConverter.SingleToInt32Bits(value), BitConverter.SingleToInt32Bits(parsed));
        Assert.Equal(json, InterventionProfile.Parse(json).Serialize());
    }

    [Fact]
    public void ScaleTextIsShortestRoundTrip()
    {
        InterventionProfile profile = new("p", BaseSha, InterventionOperatorKind.DirectionalProjection, [Site(0, rank: 3, scales: [0.1f, 1f / 3, 2f])]);
        string text = Text(profile.Serialize());
        Assert.Contains("\n        0.1,\n", text, StringComparison.Ordinal);
        Assert.Contains("\n        0.33333334,\n", text, StringComparison.Ordinal);
        Assert.Contains("\n        2\n", text, StringComparison.Ordinal);
    }

    public static TheoryData<string, string, string> Invalid => new()
    {
        { "unknown top-level property", """ "extra": 1, """, "extra" },
        { "duplicate property", """ "id": "again", """, "id" },
        { "schema version 2", """ "schema_version": 2, """, "schema_version" },
        { "schema version as string", """ "schema_version": "1", """, "schema_version" },
        { "schema version as float", """ "schema_version": 1.0, """, "schema_version" },
    };

    [Theory]
    [MemberData(nameof(Invalid))]
    public void RejectsInvalidPrefixedProperty(string because, string prefix, string path)
    {
        // The prefix goes in front of the canonical document's properties, so later duplicates are detected too.
        string json = "{" + prefix + CanonicalProfile.TrimStart()[1..];
        InterventionContractException e = Assert.Throws<InterventionContractException>(() => InterventionProfile.Parse(Utf8(json)));
        Assert.True(e.JsonPath == path || e.Message.StartsWith(path, StringComparison.Ordinal), $"{because}: {e.Message}");
    }

    public static TheoryData<string, string, string> Replaced => new()
    {
        { "\"directional_projection\"", "\"steering_vector\"", "operator" },
        { "\"target\": \"residual_post\",\n      \"hidden_size\": 8,\n      \"rank\": 2", "\"target\": \"attention_out\",\n      \"hidden_size\": 8,\n      \"rank\": 2", "sites[0].target" },
        { "\"dtype\": \"float32\",\n      \"directions_file\": \"directions/layer1.f32\"", "\"dtype\": \"float16\",\n      \"directions_file\": \"directions/layer1.f32\"", "sites[0].dtype" },
        { "\"layer\": 1,", "\"layer\": 1.0,", "sites[0].layer" },
        { "\"layer\": 1,", "\"layer\": -1,", "sites[0].layer" },
        { "\"layer\": 1,", "\"layer\": \"1\",", "sites[0].layer" },
        { "\"rank\": 2,", "\"rank\": 0,", "sites[0].rank" },
        { "\"rank\": 2,", "\"rank\": 2,\n      \"bias\": 0,", "sites[0].bias" },
        { "\"rank\": 2,", "\"rank\": 9,", "sites[0].rank" },
        { "\"hidden_size\": 8,\n      \"rank\": 2", "\"hidden_size\": 0,\n      \"rank\": 2", "sites[0].hidden_size" },
        { "1,\n        0.5", "1", "sites[0].scales" },
        { "1,\n        0.5", "1,\n        0.5,\n        2", "sites[0].scales" },
        { "1,\n        0.5", "1,\n        \"x\"", "sites[0].scales[1]" },
        { "1,\n        0.5", "1,\n        1e39", "sites[0].scales[1]" },
        { "\"base_model_sha256\": \"0123456789abcdef", "\"base_model_sha256\": \"0123456789ABCDEF", "base_model_sha256" },
        { "0123456789abcdef\",\n  \"operator\"", "0123456789abcde\",\n  \"operator\"", "base_model_sha256" },
        { "\"id\": \"synthetic-demo\"", "\"id\": \"Synthetic Demo\"", "id" },
        { "\"id\": \"synthetic-demo\"", "\"id\": \"\"", "id" },
        { "directions/layer1.f32", "../layer1.f32", "sites[0].directions_file" },
        { "directions/layer1.f32", "/abs/layer1.f32", "sites[0].directions_file" },
        { "directions/layer1.f32", "directions\\\\layer1.f32", "sites[0].directions_file" },
        { "directions/layer1.f32", "c:/layer1.f32", "sites[0].directions_file" },
        { "directions/layer1.f32", "directions//layer1.f32", "sites[0].directions_file" },
        { "directions/layer1.f32", "./layer1.f32", "sites[0].directions_file" },
        { "\"layer\": 3,", "\"layer\": 1,", "sites" },
        { "\"operator\": \"directional_projection\",\n", "", "operator" },
    };

    [Theory]
    [MemberData(nameof(Replaced))]
    public void RejectsInvalidValue(string find, string replace, string path)
    {
        Assert.Contains(find, CanonicalProfile, StringComparison.Ordinal);
        string json = CanonicalProfile.Replace(find, replace, StringComparison.Ordinal);
        InterventionContractException e = Assert.Throws<InterventionContractException>(() => InterventionProfile.Parse(Utf8(json)));
        Assert.Equal(path, e.JsonPath);
    }

    [Theory]
    [InlineData("[]")]
    [InlineData("")]
    [InlineData("{\"schema_version\": 1,}")]
    [InlineData("// comment\n{}")]
    [InlineData("{\"schema_version\": NaN}")]
    [InlineData("{}")]
    [InlineData("{\"schema_version\": 1, \"id\": \"p\", \"base_model_sha256\": \"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\", \"operator\": \"directional_projection\", \"sites\": []}")]
    public void RejectsMalformedDocuments(string json)
    {
        Assert.Throws<InterventionContractException>(() => InterventionProfile.Parse(Utf8(json)));
    }

    [Fact]
    public void RejectsByteOrderMark()
    {
        byte[] json = [0xEF, 0xBB, 0xBF, .. Utf8(CanonicalProfile)];
        Assert.Throws<InterventionContractException>(() => InterventionProfile.Parse(json));
    }

    [Fact]
    public void ConstructorValidatesLikeTheParser()
    {
        Assert.Throws<InterventionContractException>(() => Site(0, scales: [float.NaN]));
        Assert.Throws<InterventionContractException>(() => Site(0, scales: [float.PositiveInfinity]));
        Assert.Throws<InterventionContractException>(() => Site(0, rank: 2, scales: [1f]));
        Assert.Throws<InterventionContractException>(() => Site(0, sha: "x"));
        Assert.Throws<InterventionContractException>(() => new InterventionProfile("p", BaseSha, InterventionOperatorKind.DirectionalProjection, [Site(2), Site(2)]));
        Assert.Throws<InterventionContractException>(() => new InterventionProfile("p", BaseSha, InterventionOperatorKind.DirectionalProjection, []));
        Assert.Throws<InterventionContractException>(() => new InterventionProfile("p", BaseSha, (InterventionOperatorKind)7, [Site(0)]));
        Assert.Throws<InterventionContractException>(() => new InterventionSite(0, (InterventionTarget)5, 8, 1, InterventionDType.Fp32, "d.f32", DirSha, [1f]));
    }

    [Fact]
    public void VerifiesTheBaseGraphBeforeAnythingLoads()
    {
        InterventionProfile profile = Profile();
        profile.VerifyBaseModel(BaseSha);
        Assert.True(profile.AppliesTo(BaseSha));
        Assert.False(profile.AppliesTo(OtherSha));
        Assert.Throws<InterventionMismatchException>(() => profile.VerifyBaseModel(OtherSha));
        Assert.Throws<ArgumentException>(() => profile.VerifyBaseModel(BaseSha.ToUpperInvariant()));
    }

    [Fact]
    public void VerifiesTheBaseGraphFile()
    {
        using TempDir dir = new();
        byte[] graph = Utf8("not really onnx, just bytes with an identity");
        string path = dir.Write("model.onnx", graph);
        InterventionProfile profile = new("p", InterventionHashes.ComputeSha256(graph), InterventionOperatorKind.DirectionalProjection, [Site(0)]);
        profile.VerifyBaseModelFile(path);
        File.AppendAllText(path, "!");
        Assert.Throws<InterventionMismatchException>(() => profile.VerifyBaseModelFile(path));
    }

    [Fact]
    public void FindsSites()
    {
        InterventionProfile profile = Profile();
        Assert.Equal(2, profile.FindSite(1, InterventionTarget.ResidualPost)!.Rank);
        Assert.Null(profile.FindSite(2, InterventionTarget.ResidualPost));
    }

    [Fact]
    public void SelectionValidatesAndMatches()
    {
        InterventionSelection selection = new("synthetic-demo", 0.5f);
        selection.EnsureMatches(Profile());
        Assert.Equal(new InterventionSelection("synthetic-demo", 0.5f), selection);
        Assert.Throws<InterventionMismatchException>(() => new InterventionSelection("other", 1f).EnsureMatches(Profile()));
        Assert.Throws<InterventionContractException>(() => new InterventionSelection("synthetic-demo", float.NaN));
        Assert.Throws<InterventionContractException>(() => new InterventionSelection("synthetic-demo", float.NegativeInfinity));
        Assert.Throws<InterventionContractException>(() => new InterventionSelection("Bad Id", 1f));
        _ = new InterventionSelection("synthetic-demo", 0f);
        _ = new InterventionSelection("synthetic-demo", -2f);
    }
}
