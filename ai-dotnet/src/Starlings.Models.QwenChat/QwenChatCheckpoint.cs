using System.Security.Cryptography;
using Starlings.Acquisition;
using Starlings.Generation;
using Starlings.Tokenizers;

namespace Starlings.Models.QwenChat;

/// <summary>
/// An admitted original Qwen checkpoint, independent of a runtime: its Hub repository and commit, tokenizer and template
/// identities, end-of-sequence ids and decoder shape (from the pinned <c>config.json</c>). The ONNX Runtime GenAI bundles
/// built from these checkpoints are <c>QwenChatProfile</c> in Starlings.Models.QwenChat.OnnxGenAI; the checkpoint's own Hub
/// files are locked by <see cref="Lock"/>.
/// </summary>
public sealed class QwenChatCheckpoint
{
    private QwenChatCheckpoint(string id, string repository, string revision, string modelType, string tokenizerSha256,
        string templateSha256, int[] eosTokenIds, int hiddenSize, int layers, int attentionHeads, int keyValueHeads,
        int headSize, int contextLength, ModelAssetFile[] checkpointFiles)
    {
        Id = id;
        Repository = repository;
        Revision = revision;
        ModelType = modelType;
        TokenizerSha256 = tokenizerSha256;
        TemplateSha256 = templateSha256;
        EosTokenIds = Array.AsReadOnly(eosTokenIds);
        HiddenSize = hiddenSize;
        Layers = layers;
        AttentionHeads = attentionHeads;
        KeyValueHeads = keyValueHeads;
        HeadSize = headSize;
        ContextLength = contextLength;
        Profile = new ModelProfile
        {
            Id = id,
            Repository = repository,
            Revision = revision,
            TokenizerProfile = null,
            TokenizerSha256 = tokenizerSha256,
            ChatTemplateSha256 = templateSha256,
            EosTokenIds = [.. eosTokenIds.Order()],
            VocabularySize = VocabularySize,
            HiddenSize = hiddenSize,
            Layers = layers,
            HeadSize = headSize,
            ContextLength = contextLength,
            License = "Apache-2.0",
        };
        LockFileName = $"{id}.checkpoint.lock.json";
        Lock = new ModelAssetLock($"{id}-checkpoint", repository, revision, checkpointFiles);
    }

    /// <summary>Rows of the embedding and the logits of both checkpoints.</summary>
    public const int VocabularySize = 151936;

    /// <summary><c>&lt;|endoftext|&gt;</c>.</summary>
    public const int EndOfTextTokenId = 151643;

    /// <summary><c>&lt;|im_start|&gt;</c>.</summary>
    public const int ImStartTokenId = 151644;

    /// <summary><c>&lt;|im_end|&gt;</c>.</summary>
    public const int ImEndTokenId = 151645;

    /// <summary>Pinned Qwen3 original (rendered with <c>enable_thinking=false</c>).</summary>
    public static QwenChatCheckpoint Qwen3 { get; } = new("qwen3-0.6b", "Qwen/Qwen3-0.6B",
        "c1899de289a04d12100db370d81485cdf75e47ca", "qwen3", "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8", [151643, 151645],
        hiddenSize: 1024, layers: 28, attentionHeads: 16, keyValueHeads: 8, headSize: 128, contextLength: 40960,
        checkpointFiles:
        [
            new ModelAssetFile("config.json", 726, "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd"),
            new ModelAssetFile("generation_config.json", 239, "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2"),
            new ModelAssetFile("merges.txt", 1_671_853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
            new ModelAssetFile("model.safetensors", 1_503_300_328, "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"),
            new ModelAssetFile("tokenizer.json", 11_422_654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
            new ModelAssetFile("tokenizer_config.json", 9_732, "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"),
            new ModelAssetFile("vocab.json", 2_776_833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
        ]);

    /// <summary>Pinned Qwen2.5 original with upstream default system insertion.</summary>
    public static QwenChatCheckpoint Qwen25 { get; } = new("qwen2.5-0.5b-instruct", "Qwen/Qwen2.5-0.5B-Instruct",
        "7ae557604adf67be50417f59c2c2f167def9a775", "qwen2", "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f", [151643, 151645],
        hiddenSize: 896, layers: 24, attentionHeads: 14, keyValueHeads: 2, headSize: 64, contextLength: 32768,
        checkpointFiles:
        [
            new ModelAssetFile("config.json", 659, "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45"),
            new ModelAssetFile("generation_config.json", 242, "e558847a8b4402616f1273797b015104dc266fe4b520056fca88823ba8f8ebe6"),
            new ModelAssetFile("merges.txt", 1_671_839, "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
            new ModelAssetFile("model.safetensors", 988_097_824, "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"),
            new ModelAssetFile("tokenizer.json", 7_031_645, "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
            new ModelAssetFile("tokenizer_config.json", 7_305, "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583"),
            new ModelAssetFile("vocab.json", 2_776_833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
        ]);

    /// <summary>Model id as the registry names it (<c>qwen3-0.6b</c>, <c>qwen2.5-0.5b-instruct</c>).</summary>
    public string Id { get; }

    /// <summary>Original Hub repository.</summary>
    public string Repository { get; }

    /// <summary>Immutable original revision.</summary>
    public string Revision { get; }

    /// <summary>The original configuration's model type (<c>qwen3</c>, <c>qwen2</c>).</summary>
    public string ModelType { get; }

    /// <summary>Pinned original tokenizer digest.</summary>
    public string TokenizerSha256 { get; }

    /// <summary>Digest of the UTF-8 upstream template string.</summary>
    public string TemplateSha256 { get; }

    /// <summary>EOS token ids from the pinned generation configuration.</summary>
    public IReadOnlyList<int> EosTokenIds { get; }

    /// <summary>Original hidden-state width.</summary>
    public int HiddenSize { get; }

    /// <summary>Original number of decoder layers.</summary>
    public int Layers { get; }

    /// <summary>Original number of attention heads.</summary>
    public int AttentionHeads { get; }

    /// <summary>Original number of key/value heads.</summary>
    public int KeyValueHeads { get; }

    /// <summary>Original attention head width; Qwen3 declares it independently of hidden size.</summary>
    public int HeadSize { get; }

    /// <summary>Original maximum prompt plus completion context.</summary>
    public int ContextLength { get; }

    /// <summary>The backend-neutral identity and shape (no named tokenizer profile: the checkpoint's own <c>tokenizer.json</c>).</summary>
    public ModelProfile Profile { get; }

    /// <summary>
    /// File name of <see cref="Lock"/> as a JSON lock (<c>schema_version</c> 2, no per-file <c>url</c>: every file comes
    /// from the Hub repository), shipped in this package under <c>locks/</c> for tools in other languages
    /// (<c>qwen3-0.6b.checkpoint.lock.json</c>, <c>qwen2.5-0.5b-instruct.checkpoint.lock.json</c>);
    /// <see cref="ModelAssetLock.Load"/> reads it back as an equal lock.
    /// </summary>
    public string LockFileName { get; }

    /// <summary>
    /// The asset lock of the Hub checkpoint's files, downloaded from <see cref="Repository"/> at <see cref="Revision"/>:
    /// <c>config.json</c>, <c>generation_config.json</c>, the safetensors weights, <c>tokenizer.json</c>,
    /// <c>tokenizer_config.json</c>, <c>vocab.json</c> and <c>merges.txt</c> (the chat template is inside
    /// <c>tokenizer_config.json</c>). These are the files of the pipeline's candidate contract
    /// (<c>model-pipeline/candidates/qwen3/profile.json</c>, <c>qwen25/profile.json</c>, <c>origin.files</c>) without the
    /// card and license; on 2026-11-01 every size and the Hub's LFS digest of every LFS file agreed with the Hub at the
    /// pinned revision, and the SHA-256 of every other file was recomputed from the downloaded bytes. This package does
    /// not download it; pass it to
    /// <see cref="ModelAssetStore.DownloadAsync(ModelAssetLock, IProgress{ModelAssetProgress}?, CancellationToken)"/> to
    /// fetch the checkpoint explicitly (about 1.5 GB for Qwen3, 1.0 GB for Qwen2.5).
    /// </summary>
    public ModelAssetLock Lock { get; }

    /// <summary>The owned renderer of this checkpoint's template behind <see cref="IChatTemplate"/> (<see cref="QwenChatTemplate.For"/>).</summary>
    public IChatTemplate ChatTemplate => QwenChatTemplate.For(this);

    /// <summary>Verify and open the admitted original tokenizer without loading native model weights.</summary>
    /// <exception cref="InvalidDataException">The bytes or the special token ids differ from the admitted original.</exception>
    public Tokenizer OpenTokenizer(string path)
    {
        using FileStream stream = File.OpenRead(path);
        if (Convert.ToHexStringLower(SHA256.HashData(stream)) != TokenizerSha256)
        {
            throw new InvalidDataException("Tokenizer bytes differ from the admitted original.");
        }

        Tokenizer tokenizer = Tokenizer.FromFile(path);
        if (tokenizer.TokenToId("<|im_start|>") != ImStartTokenId || tokenizer.TokenToId("<|im_end|>") != ImEndTokenId ||
            tokenizer.TokenToId("<|endoftext|>") != EndOfTextTokenId || tokenizer.IdToToken(VocabularySize - 1) is not null)
        {
            throw new InvalidDataException("Original tokenizer identity or unused padding row differs.");
        }

        return tokenizer;
    }

    /// <summary>Resolve an admitted checkpoint id, rejecting others.</summary>
    /// <exception cref="NotSupportedException">Not an admitted checkpoint.</exception>
    public static QwenChatCheckpoint FromId(string id) => id switch
    {
        "qwen3-0.6b" => Qwen3,
        "qwen2.5-0.5b-instruct" => Qwen25,
        _ => throw new NotSupportedException($"Unsupported Qwen chat checkpoint: {id}"),
    };
}
