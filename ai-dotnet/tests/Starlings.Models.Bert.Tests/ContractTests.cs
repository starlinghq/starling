using System.Text.Json;
using Starlings.Runtime.Onnx;

namespace Starlings.Models.Bert.Tests;

public sealed class ContractTests
{
    [Theory]
    [InlineData("sentence-transformers/all-MiniLM-L6-v2", 256, "mean", BertTask.Embeddings, 384)]
    [InlineData("BAAI/bge-small-en-v1.5", 512, "cls", BertTask.Embeddings, 384)]
    [InlineData("cross-encoder/ms-marco-MiniLM-L6-v2", 512, "none", BertTask.Reranking, 384)]
    [InlineData("nomic-ai/nomic-embed-text-v1.5", 8192, "mean", BertTask.Embeddings, 768)]
    public void ParsesOnlyDeclaredTaskContracts(string repository, int maxTokens, string pooling, BertTask task, int hiddenSize)
    {
        var profile = BertModelProfile.Parse(ProfileJson(repository));
        Assert.Equal(task, profile.Task);
        Assert.Equal(maxTokens, profile.MaxTokens);
        Assert.Equal(pooling, profile.Pooling);
        Assert.Equal(hiddenSize, profile.HiddenSize);
        Assert.Equal(2, profile.RuntimeLock.Files.Count);
        Assert.Equal(new string('a', 40), profile.RuntimeLock.Revision);
        Assert.Equal(repository == "BAAI/bge-small-en-v1.5" ? 2 : 1, profile.Prompts.Count);
    }

    [Theory]
    [InlineData("max_tokens", 2048)]
    [InlineData("hidden_size", 768)]
    [InlineData("pad_token_id", 1)]
    [InlineData("vocabulary_size", 250037)]
    public void RejectsIncompatibleNumericContract(string key, int value)
    {
        var document = ProfileData("sentence-transformers/all-MiniLM-L6-v2");
        ((Dictionary<string, object>)document["behavior"])[key] = value;
        Assert.Throws<NotSupportedException>(() => BertModelProfile.Parse(JsonSerializer.Serialize(document)));
    }

    [Fact]
    public void FactoriesDoNotConfuseScoresWithVectors()
    {
        var embedding = BertModelProfile.Parse(ProfileJson("sentence-transformers/all-MiniLM-L6-v2"));
        var scoring = BertModelProfile.Parse(ProfileJson("cross-encoder/ms-marco-MiniLM-L6-v2"));
        Assert.Throws<ArgumentException>(() => BertReranker.Create(embedding, "does-not-exist"));
        Assert.Throws<ArgumentException>(() => BertTextEmbedder.Create(scoring, "does-not-exist"));
    }

    [Fact]
    public void MaskedMeanAndClsRemainDifferentAndIgnorePaddedRows()
    {
        var hidden = new float[3 * 384];
        hidden[0] = 3;
        hidden[384 + 1] = 4;
        hidden.AsSpan(768).Fill(9999);
        var mean = BertSession.Pool(hidden, new long[] { 1, 1, 0 }, "mean", 384, "l2");
        var cls = BertSession.Pool(hidden, new long[] { 1, 1, 0 }, "cls", 384, "l2");
        Assert.InRange(Math.Abs(mean[0] - 0.6), 0, 1e-7);
        Assert.InRange(Math.Abs(mean[1] - 0.8), 0, 1e-7);
        Assert.Equal(1, cls[0]);
        Assert.Equal(0, cls[1]);
        Assert.All(mean.Skip(2), value => Assert.Equal(0, value));
    }

    [Fact]
    public void NonFinitePoolingFailsRatherThanReturningPlausibleScores()
    {
        var hidden = new float[2 * 384];
        hidden[0] = float.NaN;
        Assert.Throws<InvalidDataException>(() => BertSession.Pool(hidden, new long[] { 1, 1 }, "mean", 384, "l2"));
    }

    [Fact]
    public void PairSegmentIdsAreAcceptedButMaskGapsAndLeftPaddingFail()
    {
        BertSession.ValidateInput(new BertInput(new long[] { 101, 200, 102, 201, 102 },
            new long[] { 1, 1, 1, 1, 1 }, new long[] { 0, 0, 0, 1, 1 }), 512);
        Assert.Throws<ArgumentException>(() => BertSession.ValidateInput(new BertInput(
            new long[] { 101, 0, 102 }, new long[] { 1, 0, 1 }, new long[] { 0, 0, 0 }), 512));
        Assert.Throws<ArgumentException>(() => BertSession.ValidateInput(new BertInput(
            new long[] { 0, 101, 102 }, new long[] { 0, 1, 1 }, new long[] { 0, 0, 0 }), 512));
    }

    [Fact]
    public void SignatureRequiresPairSegmentIdsAndFixedTaskOutputWidth()
    {
        OnnxTensorInfo Input(string name) => new(name, "Int64", [-1, -1], ["batch", "sequence"]);
        OnnxTensorInfo[] inputs = [Input("input_ids"), Input("attention_mask"), Input("token_type_ids")];
        OnnxTensorInfo[] embeddings = [new("last_hidden_state", "Float", [-1, -1, 384], ["batch", "sequence", ""])];
        BertSession.ValidateSignature(inputs, embeddings, BertTask.Embeddings, 384);
        Assert.Throws<NotSupportedException>(() => BertSession.ValidateSignature(inputs[..2], embeddings, BertTask.Embeddings, 384));
        Assert.Throws<NotSupportedException>(() => BertSession.ValidateSignature(inputs, embeddings, BertTask.Reranking, 384));
        Assert.Throws<NotSupportedException>(() => BertSession.ValidateSignature(
            [inputs[0], inputs[0], inputs[2]], embeddings, BertTask.Embeddings, 384));
    }

    private static string ProfileJson(string repository) => JsonSerializer.Serialize(ProfileData(repository));

    private static Dictionary<string, object> ProfileData(string repository)
    {
        bool reranking = repository.StartsWith("cross-encoder/", StringComparison.Ordinal);
        bool bge = repository == "BAAI/bge-small-en-v1.5";
        bool nomic = repository == "nomic-ai/nomic-embed-text-v1.5";
        var behavior = new Dictionary<string, object>
        {
            ["architecture"] = "bert", ["graph_file"] = "onnx/model.onnx", ["tokenizer_file"] = "tokenizer.json",
            ["max_tokens"] = nomic ? 8192 : reranking || bge ? 512 : 256, ["padding_side"] = "right", ["pad_token_id"] = 0,
            ["vocabulary_size"] = 30522, ["hidden_size"] = nomic ? 768 : 384, ["truncation"] = "longest_first",
            ["tokenizer_profile"] = "bert-wordpiece-uncased", ["pooling"] = reranking ? "none" : bge ? "cls" : "mean",
            ["normalize"] = reranking || nomic ? "none" : "l2", ["output_name"] = reranking ? "logits" : "last_hidden_state",
            ["prompts"] = bge ? new Dictionary<string, string> { ["none"] = "", ["query"] = BertModelProfile.BgeQueryInstruction }
                : new Dictionary<string, string> { ["none"] = "" },
        };
        if (reranking)
        {
            behavior["score_activation"] = "identity";
        }

        return new Dictionary<string, object>
        {
            ["format_version"] = 1, ["id"] = "test-bert", ["task"] = reranking ? "reranking" : "embeddings",
            ["origin"] = new { repo = repository, revision = new string('a', 40), files = new[]
            {
                new { path = "onnx/model.onnx", size = 1, sha256 = new string('1', 64) },
                new { path = "tokenizer.json", size = 1, sha256 = new string('2', 64) },
            } },
            ["runtime"] = new { backend = "onnx", version = "1.30.0", provider = "CPUExecutionProvider", precision = "fp32" },
            ["behavior"] = behavior,
        };
    }
}
