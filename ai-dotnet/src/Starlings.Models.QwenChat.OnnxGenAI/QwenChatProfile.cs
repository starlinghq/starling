using Starlings.Tokenizers;

namespace Starlings.Models.QwenChat;

/// <summary>One file of a profile's bundle lock.</summary>
/// <param name="Path">File name in the bundle directory.</param>
/// <param name="Sha256">Lower-case hex SHA-256 of the file.</param>
/// <param name="Size">Size in bytes.</param>
public sealed record QwenChatBundleFile(string Path, string Sha256, long Size);

/// <summary>
/// An admitted original checkpoint (<see cref="Checkpoint"/>) with its ONNX Runtime GenAI bundle lock: the six files the
/// pinned builder produces from the pinned original (rebuilt byte-identically; registry manifest
/// <c>execution.graph.assets</c>). Converts implicitly to its <see cref="QwenChatCheckpoint"/>, so
/// <c>QwenChatTemplate.Render(profile, ...)</c> renders the profile's template.
/// </summary>
public sealed class QwenChatProfile
{
    private QwenChatProfile(string id, QwenChatCheckpoint checkpoint, QwenChatBundleFile[] bundleFiles)
    {
        Id = id;
        Checkpoint = checkpoint;
        BundleFiles = Array.AsReadOnly(bundleFiles);
    }

    /// <summary>Pinned Qwen3 original, with enable_thinking=false.</summary>
    public static QwenChatProfile Qwen3 { get; } = new("qwen3-0.6b/ortgenai-fp32-cpu", QwenChatCheckpoint.Qwen3,
        [
            new("genai_config.json", "0af5816836cdd4414bc38239443ec2fe1f554ca44daf282b48f07e3768e264e4", 1648),
            new("model.onnx", "b9fdea5216f1642d7926c64f77875c43a43b320531db841d8b3984b79a754ae8", 248657),
            new("model.onnx.data", "2f800ce53e4c864ff5c96e99f869dc4ca0973f0c64c1f1860eea302093b3d7c4", 2405171200),
            new("tokenizer.json", "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4", 11422654),
            new("tokenizer_config.json", "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101", 9732),
            new("chat_template.jinja", "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8", 4168),
        ]);

    /// <summary>Pinned Qwen2.5 original with upstream default system insertion.</summary>
    public static QwenChatProfile Qwen25 { get; } = new("qwen2.5-0.5b-instruct/ortgenai-fp32-cpu", QwenChatCheckpoint.Qwen25,
        [
            new("genai_config.json", "a023918fb6dfdf680b0267bdb4ead00083a954194d8178d8fb1af7a0d9adbe63", 1581),
            new("model.onnx", "716a2db391784191cc1323417e1ff5bec682673b7105b8ddc4011e7a9f2c4a88", 147574),
            new("model.onnx.data", "4269c6a8f794c9db3f1304c18ead66ca8aa6eb67e57b8bbe1b815cdbdecd7f99", 1984561152),
            new("tokenizer.json", "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539", 7031645),
            new("tokenizer_config.json", "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583", 7305),
            new("chat_template.jinja", "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f", 2507),
        ]);

    /// <summary>The bundle lock: every file <see cref="QwenChatModel.Load"/> requires, with its pinned digest and size.</summary>
    public IReadOnlyList<QwenChatBundleFile> BundleFiles { get; }

    /// <summary>Portable candidate profile id.</summary>
    public string Id { get; }

    /// <summary>The admitted original checkpoint the bundle is built from.</summary>
    public QwenChatCheckpoint Checkpoint { get; }

    /// <summary>Original Hub repository.</summary>
    public string Repository => Checkpoint.Repository;
    /// <summary>Immutable original revision.</summary>
    public string Revision => Checkpoint.Revision;
    /// <summary>Required GenAI model type (the original configuration's model type).</summary>
    public string ModelType => Checkpoint.ModelType;
    /// <summary>Pinned original tokenizer digest.</summary>
    public string TokenizerSha256 => Checkpoint.TokenizerSha256;
    /// <summary>Digest of the UTF-8 upstream template string.</summary>
    public string TemplateSha256 => Checkpoint.TemplateSha256;
    /// <summary>EOS token ids from the pinned generation configuration.</summary>
    public IReadOnlyList<int> EosTokenIds => Checkpoint.EosTokenIds;
    /// <summary>Original hidden-state width.</summary>
    public int HiddenSize => Checkpoint.HiddenSize;
    /// <summary>Original number of decoder layers.</summary>
    public int Layers => Checkpoint.Layers;
    /// <summary>Original number of attention heads.</summary>
    public int AttentionHeads => Checkpoint.AttentionHeads;
    /// <summary>Original number of key/value heads.</summary>
    public int KeyValueHeads => Checkpoint.KeyValueHeads;
    /// <summary>Original attention head width; Qwen3 declares it independently of hidden size.</summary>
    public int HeadSize => Checkpoint.HeadSize;
    /// <summary>Original maximum prompt plus completion context.</summary>
    public int ContextLength => Checkpoint.ContextLength;

    /// <summary>Verify and open the admitted original tokenizer without loading native model weights (<see cref="QwenChatCheckpoint.OpenTokenizer"/>).</summary>
    public Tokenizer OpenTokenizer(string path) => Checkpoint.OpenTokenizer(path);

    /// <summary>The profile's checkpoint (<see cref="Checkpoint"/>).</summary>
    public static implicit operator QwenChatCheckpoint(QwenChatProfile profile)
    {
        ArgumentNullException.ThrowIfNull(profile);
        return profile.Checkpoint;
    }

    /// <summary>Resolve an admitted candidate id, rejecting other profiles.</summary>
    public static QwenChatProfile FromId(string id) => id switch
    {
        "qwen3-0.6b/ortgenai-fp32-cpu" => Qwen3,
        "qwen2.5-0.5b-instruct/ortgenai-fp32-cpu" => Qwen25,
        _ => throw new NotSupportedException($"Unsupported Qwen chat profile: {id}"),
    };
}
