using System.Text.Json;
using System.Text.Json.Nodes;

namespace Starlings.Models.Bert.Tests;

/// <summary>
/// The route contract table (model-pipeline/routes/bert-onnx-cpu-v1/contracts.json) is the one source of the
/// per-repository contracts for this package and for the Python allow-list (onboarding_bert.py).
/// </summary>
public sealed class ContractTableTests
{
    private static readonly string ContractFile = Path.Combine(FindRepositoryRoot(), "model-pipeline", "routes",
        "bert-onnx-cpu-v1", "contracts.json");

    [Fact]
    public void EmbeddedTableIsTheContractFileByteForByte()
    {
        Assert.Equal(File.ReadAllText(ContractFile), BertModelProfile.EmbeddedContractsJson());
    }

    [Fact]
    public void SupportedRepositoriesFollowTheTable()
    {
        var expected = JsonNode.Parse(File.ReadAllText(ContractFile))!["contracts"]!.AsObject().Select(entry => entry.Key);
        Assert.Equal(expected, BertModelProfile.SupportedRepositories);
        Assert.Contains(Contracts.Values, spec => spec.Task == BertTask.Embeddings);
        Assert.Contains(Contracts.Values, spec => spec.Task == BertTask.Reranking);
    }

    [Fact]
    public void EveryQueryPromptIsTheBgeInstruction()
    {
        var withQuery = Contracts.Values.Where(spec => spec.Prompts.ContainsKey("query")).ToList();
        Assert.NotEmpty(withQuery);
        Assert.All(withQuery, spec => Assert.Equal(BertModelProfile.BgeQueryInstruction, spec.Prompts["query"]));
    }

    [Theory]
    [InlineData("task", "classification")]
    [InlineData("pooling", "max")]
    [InlineData("normalize", "l1")]
    [InlineData("tokenizer_profile", "gpt2-bpe")]
    [InlineData("output_name", "")]
    [InlineData("max_tokens", 0)]
    [InlineData("pad_token_id", 30522)]
    [InlineData("cls_token_id", -1)]
    [InlineData("hidden_size", true)]
    [InlineData("token_type_ids", 1)]
    [InlineData("score_activation_source", "config")]
    [InlineData("unknown_field", 1)]
    public void RejectsAnEmbeddingsContractThatBreaksARule(string key, object value)
    {
        var table = Table();
        table["contracts"]![First(BertTask.Embeddings)]![key] = JsonValue.Create(value);
        Assert.Throws<InvalidDataException>(() => BertModelProfile.ParseContracts(table.ToJsonString()));
    }

    [Theory]
    [InlineData("pooling", "cls")]
    [InlineData("normalize", "l2")]
    [InlineData("score_activation_source", "sigmoid")]
    public void RejectsARerankingContractThatBreaksARule(string key, string value)
    {
        var table = Table();
        table["contracts"]![First(BertTask.Reranking)]![key] = value;
        Assert.Throws<InvalidDataException>(() => BertModelProfile.ParseContracts(table.ToJsonString()));
    }

    [Fact]
    public void RejectsMissingFieldsPromptsWithoutNoneAndBadKeys()
    {
        string repository = BertModelProfile.SupportedRepositories[0];
        var missing = Table();
        missing["contracts"]![repository]!.AsObject().Remove("output_name");
        Assert.Throws<InvalidDataException>(() => BertModelProfile.ParseContracts(missing.ToJsonString()));

        var prompts = Table();
        prompts["contracts"]![repository]!["prompts"] = new JsonObject { ["query"] = "q: " };
        Assert.Throws<InvalidDataException>(() => BertModelProfile.ParseContracts(prompts.ToJsonString()));

        var key = Table();
        var contracts = key["contracts"]!.AsObject();
        var entry = contracts[repository]!.DeepClone();
        contracts.Add("not-a-repository", entry);
        Assert.Throws<InvalidDataException>(() => BertModelProfile.ParseContracts(key.ToJsonString()));

        var format = Table();
        format["format_version"] = 2;
        Assert.Throws<InvalidDataException>(() => BertModelProfile.ParseContracts(format.ToJsonString()));
    }

    [Fact]
    public void RejectsDuplicateRepositories()
    {
        string json = BertModelProfile.EmbeddedContractsJson();
        string repository = BertModelProfile.SupportedRepositories[0];
        string key = JsonSerializer.Serialize(repository) + ": ";
        int at = json.IndexOf(key + "{", StringComparison.Ordinal);
        string duplicated = json.Insert(at, key + JsonNode.Parse(json)!["contracts"]![repository]!.ToJsonString() + ", ");
        Assert.Throws<InvalidDataException>(() => BertModelProfile.ParseContracts(duplicated));
    }

    [Fact]
    public void RejectsMalformedJson()
    {
        Assert.Throws<InvalidDataException>(() => BertModelProfile.ParseContracts("{"));
        Assert.Throws<InvalidDataException>(() => BertModelProfile.ParseContracts("[]"));
    }

    private static IReadOnlyDictionary<string, BertModelProfile.BertSpec> Contracts =>
        BertModelProfile.ParseContracts(BertModelProfile.EmbeddedContractsJson());

    /// <summary>The first repository of the table with the task (the tests name no model).</summary>
    private static string First(BertTask task) => Contracts.First(entry => entry.Value.Task == task).Key;

    private static JsonNode Table() => JsonNode.Parse(BertModelProfile.EmbeddedContractsJson(),
        documentOptions: new JsonDocumentOptions { AllowDuplicateProperties = false })!;

    private static string FindRepositoryRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "model-pipeline")) && Directory.Exists(Path.Combine(dir.FullName, "ai-dotnet")))
            {
                return dir.FullName;
            }
        }

        throw new DirectoryNotFoundException("Repository root not found above the test binaries.");
    }
}
