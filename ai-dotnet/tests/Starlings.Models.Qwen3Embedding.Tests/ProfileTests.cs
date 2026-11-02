using Starlings.Runtime.Onnx;

namespace Starlings.Models.Qwen3Embedding.Tests;

public sealed class ProfileTests
{
    [Fact]
    public void BuiltInProfilesArePinned()
    {
        var embedding = Qwen3EmbeddingProfile.Embedding06B;
        Assert.Equal("qwen3-embedding-0.6b", embedding.Id);
        Assert.Equal(Qwen3EmbeddingTask.Embedding, embedding.Task);
        Assert.Equal("97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", embedding.Revision);
        Assert.Equal("qwen3-bpe-embedding", embedding.TokenizerProfile);
        Assert.Equal(8192, embedding.MaxTokens);

        var reranker = Qwen3EmbeddingProfile.Reranker06B;
        Assert.Equal("qwen3-reranker-0.6b", reranker.Id);
        Assert.Equal(Qwen3EmbeddingTask.Reranking, reranker.Task);
        Assert.Equal("e61197ed45024b0ed8a2d74b80b4d909f1255473", reranker.Revision);
        Assert.Equal("qwen3-bpe-chat", reranker.TokenizerProfile);
        Assert.Empty(reranker.Prompts);

        foreach (var profile in Qwen3EmbeddingProfile.BuiltIn)
        {
            Assert.Same(profile, Qwen3EmbeddingProfile.FromId(profile.Id));
            Assert.Equal(profile.Repository, profile.RuntimeLock.Repository);
            Assert.Equal(profile.Revision, profile.RuntimeLock.Revision);
            Assert.True(profile.RuntimeLock.GetFile(Qwen3EmbeddingProfile.GraphDataFile).Size > 2_000_000_000L);
            Assert.Equal(6, profile.RuntimeLock.Files.Count);
        }

        Assert.Equal("aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
            reranker.RuntimeLock.GetFile(Qwen3EmbeddingProfile.TokenizerFile).Sha256);
        Assert.Throws<NotSupportedException>(() => Qwen3EmbeddingProfile.FromId("qwen3-embedding-4b"));
    }

    [Fact]
    public void PromptsAreTheUpstreamTextVerbatim()
    {
        var prompts = Qwen3EmbeddingProfile.Embedding06B.Prompts;
        Assert.Equal(["document", "none", "query"], prompts.Keys.Order(StringComparer.Ordinal));
        Assert.Equal(string.Empty, prompts["none"]);
        Assert.Equal(string.Empty, prompts["document"]);
        Assert.Equal("Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:", prompts["query"]);
        Assert.Equal("Instruct: Find code\nQuery:sort a list", Qwen3TextEmbedder.FormatQuery("Find code", "sort a list"));
    }

    [Fact]
    public void CapabilitiesAreExplicit()
    {
        Assert.Equal(CapabilityStatus.Experimental, Qwen3EmbeddingCapabilities.Embeddings);
        Assert.Equal(CapabilityStatus.Experimental, Qwen3EmbeddingCapabilities.Reranking);
        Assert.Equal(CapabilityStatus.Experimental, Qwen3EmbeddingCapabilities.Batching);
        Assert.Equal(CapabilityStatus.Untested, Qwen3EmbeddingCapabilities.MatryoshkaDimensions);
        Assert.Equal(CapabilityStatus.Untested, Qwen3EmbeddingCapabilities.Cuda);
        Assert.Equal(CapabilityStatus.Unsupported, Qwen3EmbeddingCapabilities.Download);
        Assert.Equal(CapabilityStatus.Unsupported, Qwen3EmbeddingCapabilities.ReducedPrecision);
        Assert.Equal(CapabilityStatus.Unsupported, Qwen3EmbeddingCapabilities.LargerCheckpoints);
        Assert.Equal(CapabilityStatus.Unsupported, Qwen3EmbeddingCapabilities.BeyondMaxTokens);
    }

    [Fact]
    public void OpenDirectoryRejectsAnIncompleteDirectory()
    {
        string dir = Directory.CreateTempSubdirectory("qwen3-empty-").FullName;
        try
        {
            Assert.ThrowsAny<Exception>(() => Qwen3EmbeddingProfile.Embedding06B.OpenDirectory(dir));
            Assert.ThrowsAny<Exception>(() => Qwen3TextEmbedder.Create(dir));
            Assert.ThrowsAny<Exception>(() => Qwen3Reranker.Create(dir));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void CreateRejectsTheOtherTask()
    {
        Assert.Throws<ArgumentException>(() => Qwen3TextEmbedder.Create("unused", profile: Qwen3EmbeddingProfile.Reranker06B));
        Assert.Throws<ArgumentException>(() => Qwen3Reranker.Create("unused", profile: Qwen3EmbeddingProfile.Embedding06B));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Qwen3EmbeddingOptions { MaxBatchSize = 0 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new Qwen3EmbeddingOptions { Threads = 65 }.Validate());
        Assert.Throws<ArgumentOutOfRangeException>(() => new Qwen3EmbeddingOptions { MaxBatchTokens = 0 }.Validate());
    }

    [Fact]
    public void PoolingAndScoreHelpers()
    {
        Assert.Equal(2, Qwen3Session.LastAttended([1, 1, 1, 0, 0]));
        Assert.Equal(0, Qwen3Session.LastAttended([1]));
        Assert.Throws<InvalidDataException>(() => Qwen3Session.LastAttended([0, 0]));

        float[] unit = Qwen3Session.Normalize([3f, 4f]);
        Assert.Equal(0.6f, unit[0], 6);
        Assert.Equal(0.8f, unit[1], 6);
        Assert.Equal([0f, 0f], Qwen3Session.Normalize([0f, 0f]));
        Assert.Throws<InvalidDataException>(() => Qwen3Session.Normalize([float.NaN]));

        Assert.Equal(1.5f, Qwen3Session.Score([-0.5f, 1f]));
        Assert.Throws<InvalidDataException>(() => Qwen3Session.Score([float.PositiveInfinity, 1f]));
        Assert.Equal(0.5f, Qwen3Reranker.Sigmoid(0f));
        Assert.True(Qwen3Reranker.Sigmoid(7.6f) > 0.999f);
    }

    [Fact]
    public void SignatureChecks()
    {
        OnnxTensorInfo Int(string name) => new(name, "Int64", [-1, -1], ["batch", "sequence"]);
        OnnxTensorInfo Float(string name, long last) => new(name, "Float", [-1, -1, last], ["batch", "sequence", string.Empty]);
        OnnxTensorInfo[] inputs = [Int("input_ids"), Int("attention_mask")];

        Assert.Equal("last_hidden_state", Qwen3Session.ValidateSignature(inputs, [Float("last_hidden_state", 1024)], Qwen3EmbeddingTask.Embedding));
        Assert.Equal("yes_no_logits", Qwen3Session.ValidateSignature(inputs, [Float("yes_no_logits", 2), Float("last_hidden_state", 1024)], Qwen3EmbeddingTask.Reranking));
        Assert.Throws<NotSupportedException>(() => Qwen3Session.ValidateSignature(inputs, [Float("last_hidden_state", 1024)], Qwen3EmbeddingTask.Reranking));
        Assert.Throws<NotSupportedException>(() => Qwen3Session.ValidateSignature(inputs, [Float("last_hidden_state", 896)], Qwen3EmbeddingTask.Embedding));
        Assert.Throws<NotSupportedException>(() => Qwen3Session.ValidateSignature([.. inputs, Int("position_ids")], [Float("last_hidden_state", 1024)], Qwen3EmbeddingTask.Embedding));
    }
}
