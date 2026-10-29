using System.Security.Cryptography;
using Starlings.Generation;
using Starlings.Tokenizers;

namespace Starlings.Models.QwenChat;

/// <summary>
/// An admitted original Qwen checkpoint, independent of a runtime: its Hub repository and commit, tokenizer and template
/// identities, end-of-sequence ids and decoder shape (from the pinned <c>config.json</c>). The ONNX Runtime GenAI bundles
/// built from these checkpoints are <c>QwenChatProfile</c> in Starlings.Models.QwenChat.OnnxGenAI.
/// </summary>
public sealed class QwenChatCheckpoint
{
    private QwenChatCheckpoint(string id, string repository, string revision, string modelType, string tokenizerSha256,
        string templateSha256, int[] eosTokenIds, int hiddenSize, int layers, int attentionHeads, int keyValueHeads,
        int headSize, int contextLength)
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
        hiddenSize: 1024, layers: 28, attentionHeads: 16, keyValueHeads: 8, headSize: 128, contextLength: 40960);

    /// <summary>Pinned Qwen2.5 original with upstream default system insertion.</summary>
    public static QwenChatCheckpoint Qwen25 { get; } = new("qwen2.5-0.5b-instruct", "Qwen/Qwen2.5-0.5B-Instruct",
        "7ae557604adf67be50417f59c2c2f167def9a775", "qwen2", "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f", [151643, 151645],
        hiddenSize: 896, layers: 24, attentionHeads: 14, keyValueHeads: 2, headSize: 64, contextLength: 32768);

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
