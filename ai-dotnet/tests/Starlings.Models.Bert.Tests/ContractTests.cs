using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Starlings.Acquisition;
using Starlings.Runtime.Onnx;

namespace Starlings.Models.Bert.Tests;

public sealed class ContractTests
{
    [Theory]
    [InlineData("sentence-transformers/all-MiniLM-L6-v2", 256, "mean", BertTask.Embeddings, 384)]
    [InlineData("BAAI/bge-small-en-v1.5", 512, "cls", BertTask.Embeddings, 384)]
    [InlineData("BAAI/bge-base-en-v1.5", 512, "cls", BertTask.Embeddings, 768)]
    [InlineData("BAAI/bge-m3", 8192, "cls", BertTask.Embeddings, 1024)]
    [InlineData("cross-encoder/ms-marco-MiniLM-L6-v2", 512, "none", BertTask.Reranking, 384)]
    [InlineData("Starlinghq/bge-reranker-v2-m3-onnx", 8192, "none", BertTask.Reranking, 1024)]
    [InlineData("nomic-ai/nomic-embed-text-v1.5", 8192, "mean", BertTask.Embeddings, 768)]
    [InlineData("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 128, "mean", BertTask.Embeddings, 384)]
    public void ParsesOnlyDeclaredTaskContracts(string repository, int maxTokens, string pooling, BertTask task, int hiddenSize)
    {
        var profile = BertModelProfile.Parse(ProfileJson(repository));
        Assert.Equal(task, profile.Task);
        Assert.Equal(maxTokens, profile.MaxTokens);
        Assert.Equal(pooling, profile.Pooling);
        Assert.Equal(hiddenSize, profile.HiddenSize);
        Assert.Equal(2, profile.RuntimeLock.Files.Count);
        Assert.Equal(new string('a', 40), profile.RuntimeLock.Revision);
        Assert.Equal(repository is "BAAI/bge-small-en-v1.5" or "BAAI/bge-base-en-v1.5" ? 2 : 1, profile.Prompts.Count);
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
    public void XlmrRerankerContractIsExactAndUsesNoSegmentIds()
    {
        var profile = BertModelProfile.Parse(ProfileJson("Starlinghq/bge-reranker-v2-m3-onnx"));
        Assert.Equal(("logits", "xlmr-unigram", false), (profile.OutputName, profile.TokenizerProfile, profile.TokenTypeIds));
        Assert.Equal((250002, 0, 2, 1), (profile.VocabularySize, profile.ClsTokenId, profile.SepTokenId, profile.PadTokenId));
        Assert.Equal("none", profile.Normalize);
    }

    [Theory]
    [InlineData("pooling", "cls")]
    [InlineData("tokenizer_profile", "bert-wordpiece-uncased")]
    [InlineData("output_name", "token_embeddings")]
    [InlineData("score_activation", "sigmoid")]
    [InlineData("max_tokens", 512)]
    [InlineData("hidden_size", 384)]
    [InlineData("pad_token_id", 0)]
    public void RejectsXlmrRerankerNearMisses(string key, object value)
    {
        var document = ProfileData("Starlinghq/bge-reranker-v2-m3-onnx");
        ((Dictionary<string, object>)document["behavior"])[key] = value;
        Assert.Throws<NotSupportedException>(() => BertModelProfile.Parse(JsonSerializer.Serialize(document)));
    }

    [Fact]
    public void XlmrRerankerRefusesEmbeddingsTask()
    {
        var document = ProfileData("Starlinghq/bge-reranker-v2-m3-onnx");
        document["task"] = "embeddings";
        Assert.Throws<NotSupportedException>(() => BertModelProfile.Parse(JsonSerializer.Serialize(document)));
    }

    [Fact]
    public void MultilingualMiniLmPairsXlmrTokenizationWithSegmentIds()
    {
        var profile = BertModelProfile.Parse(ProfileJson("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"));
        Assert.Equal(("last_hidden_state", "xlmr-unigram", true), (profile.OutputName, profile.TokenizerProfile, profile.TokenTypeIds));
        Assert.Equal((250002, 0, 2, 1), (profile.VocabularySize, profile.ClsTokenId, profile.SepTokenId, profile.PadTokenId));
        Assert.Equal(("mean", "none", 128, 384), (profile.Pooling, profile.Normalize, profile.MaxTokens, profile.HiddenSize));
    }

    [Theory]
    [InlineData("pooling", "cls")]
    [InlineData("normalize", "l2")]
    [InlineData("tokenizer_profile", "bert-wordpiece-uncased")]
    [InlineData("max_tokens", 512)]
    [InlineData("pad_token_id", 0)]
    [InlineData("vocabulary_size", 250037)]
    public void RejectsMultilingualMiniLmNearMisses(string key, object value)
    {
        var document = ProfileData("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2");
        ((Dictionary<string, object>)document["behavior"])[key] = value;
        Assert.Throws<NotSupportedException>(() => BertModelProfile.Parse(JsonSerializer.Serialize(document)));
    }

    [Theory]
    [InlineData("BAAI/bge-m3", true)]
    [InlineData("Starlinghq/bge-reranker-v2-m3-onnx", true)]
    [InlineData("sentence-transformers/all-MiniLM-L6-v2", false)]
    public void RuntimeLockCarriesExternalGraphDataExactlyWhenPinned(string repository, bool externalData)
    {
        var profile = BertModelProfile.Parse(ProfileJson(repository, externalData));
        string[] expected = externalData
            ? [BertModelProfile.GraphFile, BertModelProfile.GraphDataFile, BertModelProfile.TokenizerFile]
            : [BertModelProfile.GraphFile, BertModelProfile.TokenizerFile];
        Assert.Equal(expected, profile.RuntimeLock.Files.Select(file => file.Path));
        Assert.DoesNotContain(profile.RuntimeLock.Files, file => file.Path == "model.safetensors");
    }

    [Fact]
    public void RuntimeLockAloneVerifiesADirectoryWithOnlyItsFiles()
    {
        string directory = Path.Combine(Path.GetTempPath(), "bert-runtime-lock-" + Guid.NewGuid().ToString("N"));
        try
        {
            var contents = new Dictionary<string, byte[]>
            {
                [BertModelProfile.GraphFile] = Encoding.UTF8.GetBytes("graph"),
                [BertModelProfile.GraphDataFile] = Encoding.UTF8.GetBytes("external weights"),
                [BertModelProfile.TokenizerFile] = Encoding.UTF8.GetBytes("{}"),
                ["model.safetensors"] = Encoding.UTF8.GetBytes("source-only weights, never written"),
            };
            var files = contents.Select(entry => new
            {
                path = entry.Key, size = entry.Value.LongLength, sha256 = Convert.ToHexStringLower(SHA256.HashData(entry.Value)),
            }).ToArray();
            var document = ProfileData("Starlinghq/bge-reranker-v2-m3-onnx");
            document["origin"] = new { repo = "Starlinghq/bge-reranker-v2-m3-onnx", revision = new string('a', 40), files };
            var profile = BertModelProfile.Parse(JsonSerializer.Serialize(document));
            foreach (var file in profile.RuntimeLock.Files)
            {
                string path = Path.Combine(directory, file.Path);
                Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                File.WriteAllBytes(path, contents[file.Path]);
            }

            var snapshot = ModelAssetStore.VerifyDirectory(profile.RuntimeLock, directory);
            Assert.True(File.Exists(snapshot.GetFilePath(BertModelProfile.GraphDataFile)));
            Assert.Throws<ModelAssetMissingException>(() => ModelAssetStore.VerifyDirectory(profile.SourceLock, directory));
            File.Delete(Path.Combine(directory, BertModelProfile.GraphDataFile));
            Assert.Throws<ModelAssetMissingException>(() => ModelAssetStore.VerifyDirectory(profile.RuntimeLock, directory));
        }
        finally
        {
            if (Directory.Exists(directory))
            {
                Directory.Delete(directory, recursive: true);
            }
        }
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
            new long[] { 1, 1, 1, 1, 1 }, new long[] { 0, 0, 0, 1, 1 }), 512, 30522, 0, 101, 102);
        Assert.Throws<ArgumentException>(() => BertSession.ValidateInput(new BertInput(
            new long[] { 101, 0, 102 }, new long[] { 1, 0, 1 }, new long[] { 0, 0, 0 }), 512, 30522, 0, 101, 102));
        Assert.Throws<ArgumentException>(() => BertSession.ValidateInput(new BertInput(
            new long[] { 0, 101, 102 }, new long[] { 0, 1, 1 }, new long[] { 0, 0, 0 }), 512, 30522, 0, 101, 102));
    }

    [Fact]
    public void XlmrPairLayoutIsAcceptedWithZeroSegmentIds()
    {
        // <s> A </s></s> B </s>, right padded with <pad> = 1.
        BertSession.ValidateInput(new BertInput(new long[] { 0, 2367, 2, 2, 1274, 2, 1 },
            new long[] { 1, 1, 1, 1, 1, 1, 0 }, new long[7]), 8192, 250002, 1, 0, 2);
        Assert.Throws<ArgumentException>(() => BertSession.ValidateInput(new BertInput(new long[] { 0, 2367, 2, 2, 1274, 2, 0 },
            new long[] { 1, 1, 1, 1, 1, 1, 0 }, new long[7]), 8192, 250002, 1, 0, 2));
    }

    [Fact]
    public void RerankerSignatureWithoutSegmentIdsNeedsOneLogit()
    {
        OnnxTensorInfo Input(string name) => new(name, "Int64", [-1, -1], ["batch", "sequence"]);
        OnnxTensorInfo[] inputs = [Input("input_ids"), Input("attention_mask")];
        OnnxTensorInfo[] logits = [new("logits", "Float", [-1, 1], ["batch", ""])];
        BertSession.ValidateSignature(inputs, logits, BertTask.Reranking, 1024, "logits", false);
        Assert.Throws<NotSupportedException>(() => BertSession.ValidateSignature(
            [.. inputs, Input("token_type_ids")], logits, BertTask.Reranking, 1024, "logits", false));
        Assert.Throws<NotSupportedException>(() => BertSession.ValidateSignature(
            inputs, [new("logits", "Float", [-1, 2], ["batch", ""])], BertTask.Reranking, 1024, "logits", false));
    }

    [Fact]
    public void SignatureRequiresPairSegmentIdsAndFixedTaskOutputWidth()
    {
        OnnxTensorInfo Input(string name) => new(name, "Int64", [-1, -1], ["batch", "sequence"]);
        OnnxTensorInfo[] inputs = [Input("input_ids"), Input("attention_mask"), Input("token_type_ids")];
        OnnxTensorInfo[] embeddings = [new("last_hidden_state", "Float", [-1, -1, 384], ["batch", "sequence", ""])];
        BertSession.ValidateSignature(inputs, embeddings, BertTask.Embeddings, 384, "last_hidden_state", true);
        Assert.Throws<NotSupportedException>(() => BertSession.ValidateSignature(inputs[..2], embeddings, BertTask.Embeddings, 384, "last_hidden_state", true));
        Assert.Throws<NotSupportedException>(() => BertSession.ValidateSignature(inputs, embeddings, BertTask.Reranking, 384, "logits", true));
        Assert.Throws<NotSupportedException>(() => BertSession.ValidateSignature(
            [inputs[0], inputs[0], inputs[2]], embeddings, BertTask.Embeddings, 384, "last_hidden_state", true));
    }

    private static string ProfileJson(string repository, bool externalData = false) =>
        JsonSerializer.Serialize(ProfileData(repository, externalData));

    /// <summary>A schema-valid candidate for the repository, its behavior taken from the route contract table.</summary>
    private static Dictionary<string, object> ProfileData(string repository, bool externalData = false)
    {
        var spec = BertModelProfile.ParseContracts(BertModelProfile.EmbeddedContractsJson())[repository];
        bool reranking = spec.Task == BertTask.Reranking;
        var behavior = new Dictionary<string, object>
        {
            ["architecture"] = "bert", ["graph_file"] = "onnx/model.onnx", ["tokenizer_file"] = "tokenizer.json",
            ["max_tokens"] = spec.MaxTokens, ["padding_side"] = "right", ["pad_token_id"] = spec.PadTokenId,
            ["vocabulary_size"] = spec.VocabularySize, ["hidden_size"] = spec.HiddenSize,
            ["truncation"] = "longest_first", ["tokenizer_profile"] = spec.TokenizerProfile,
            ["pooling"] = spec.Pooling, ["normalize"] = spec.Normalize, ["output_name"] = spec.OutputName,
            ["prompts"] = new Dictionary<string, string>(spec.Prompts),
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
                new { path = "model.safetensors", size = 1, sha256 = new string('0', 64) },
                new { path = "onnx/model.onnx", size = 1, sha256 = new string('1', 64) },
                new { path = "onnx/model.onnx_data", size = 1, sha256 = new string('3', 64) },
                new { path = "tokenizer.json", size = 1, sha256 = new string('2', 64) },
            }.Where(file => externalData || file.path != "onnx/model.onnx_data").ToArray() },
            ["runtime"] = new { backend = "onnx", version = "1.30.0", provider = "CPUExecutionProvider", precision = "fp32" },
            ["behavior"] = behavior,
        };
    }
}
