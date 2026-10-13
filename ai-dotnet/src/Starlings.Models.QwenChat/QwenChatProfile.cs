using System.Security.Cryptography;
using Starlings.Tokenizers;

namespace Starlings.Models.QwenChat;

/// <summary>An immutable admitted original model identity. Graph digests are verified against its proposed bundle lock.</summary>
public sealed class QwenChatProfile
{
    private QwenChatProfile(string id, string repository, string revision, string modelType, string tokenizerSha256,
        string templateSha256, params int[] eosTokenIds)
    {
        Id = id;
        Repository = repository;
        Revision = revision;
        ModelType = modelType;
        TokenizerSha256 = tokenizerSha256;
        TemplateSha256 = templateSha256;
        EosTokenIds = Array.AsReadOnly(eosTokenIds);
    }

    /// <summary>Pinned Qwen3 original, with enable_thinking=false.</summary>
    public static QwenChatProfile Qwen3 { get; } = new("qwen3-0.6b/ortgenai-fp32-cpu", "Qwen/Qwen3-0.6B",
        "c1899de289a04d12100db370d81485cdf75e47ca", "qwen3", "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8", 151643, 151645);

    /// <summary>Pinned Qwen2.5 original with upstream default system insertion.</summary>
    public static QwenChatProfile Qwen25 { get; } = new("qwen2.5-0.5b-instruct/ortgenai-fp32-cpu", "Qwen/Qwen2.5-0.5B-Instruct",
        "7ae557604adf67be50417f59c2c2f167def9a775", "qwen2", "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f", 151643, 151645);

    /// <summary>Portable candidate profile id.</summary>
    public string Id { get; }
    /// <summary>Original Hub repository.</summary>
    public string Repository { get; }
    /// <summary>Immutable original revision.</summary>
    public string Revision { get; }
    /// <summary>Required GenAI model type.</summary>
    public string ModelType { get; }
    /// <summary>Pinned original tokenizer digest.</summary>
    public string TokenizerSha256 { get; }
    /// <summary>Digest of the UTF-8 upstream template string.</summary>
    public string TemplateSha256 { get; }
    /// <summary>EOS token ids from the pinned generation configuration.</summary>
    public IReadOnlyList<int> EosTokenIds { get; }
    /// <summary>Original hidden-state width.</summary>
    public int HiddenSize => ModelType == "qwen3" ? 1024 : 896;
    /// <summary>Original number of decoder layers.</summary>
    public int Layers => ModelType == "qwen3" ? 28 : 24;
    /// <summary>Original number of attention heads.</summary>
    public int AttentionHeads => ModelType == "qwen3" ? 16 : 14;
    /// <summary>Original number of key/value heads.</summary>
    public int KeyValueHeads => ModelType == "qwen3" ? 8 : 2;
    /// <summary>Original attention head width; Qwen3 declares it independently of hidden size.</summary>
    public int HeadSize => ModelType == "qwen3" ? 128 : 64;
    /// <summary>Original maximum prompt plus completion context.</summary>
    public int ContextLength => ModelType == "qwen3" ? 40960 : 32768;

    /// <summary>Verify and open the admitted original tokenizer without loading native model weights.</summary>
    public Tokenizer OpenTokenizer(string path)
    {
        using FileStream stream = File.OpenRead(path);
        if (Convert.ToHexStringLower(SHA256.HashData(stream)) != TokenizerSha256)
        {
            throw new InvalidDataException("Tokenizer bytes differ from the admitted original.");
        }

        Tokenizer tokenizer = Tokenizer.FromFile(path);
        if (tokenizer.TokenToId("<|im_start|>") != 151644 || tokenizer.TokenToId("<|im_end|>") != 151645 ||
            tokenizer.TokenToId("<|endoftext|>") != 151643 || tokenizer.IdToToken(151935) is not null)
        {
            throw new InvalidDataException("Original tokenizer identity or unused padding row differs.");
        }

        return tokenizer;
    }

    /// <summary>Resolve an admitted candidate id, rejecting other profiles.</summary>
    public static QwenChatProfile FromId(string id) => id switch
    {
        "qwen3-0.6b/ortgenai-fp32-cpu" => Qwen3,
        "qwen2.5-0.5b-instruct/ortgenai-fp32-cpu" => Qwen25,
        _ => throw new NotSupportedException($"Unsupported Qwen chat profile: {id}"),
    };
}
