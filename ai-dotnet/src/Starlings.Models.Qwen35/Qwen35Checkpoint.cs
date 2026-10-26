using Starlings.Acquisition;
using Starlings.Generation;

namespace Starlings.Models.Qwen35;

/// <summary>
/// The pinned Hub checkpoint <c>Qwen/Qwen3.5-0.8B</c>, independent of a runtime: its repository and commit, the
/// special token ids the template and the owned tokenizer use, its <see cref="ModelProfile"/>, and the asset lock of
/// the checkpoint's own files (<see cref="Lock"/>). The ONNX Runtime GenAI bundle built from it is locked separately
/// (<c>Qwen35ModelAssets</c> in Starlings.Models.Qwen35.OnnxGenAI).
/// </summary>
public static class Qwen35Checkpoint
{
    /// <summary>Hub repository of the checkpoint.</summary>
    public const string Repository = "Qwen/Qwen3.5-0.8B";

    /// <summary>Pinned commit of <see cref="Repository"/>.</summary>
    public const string Revision = "2fc06364715b967f1860aea9cf38778875588b17";

    /// <summary>Owned tokenizer profile (Starlings.Tokenizers).</summary>
    public const string TokenizerProfile = "qwen35-bpe-chat";

    /// <summary>
    /// SHA-256 of the owned tokenizer's <c>tokenizer.json</c> (the <c>qwen35-bpe-chat</c> profile asset: the transformers
    /// 5.18.0 runtime tokenizer of the checkpoint, which differs from the Hub file <see cref="HubTokenizerSha256"/>).
    /// </summary>
    public const string TokenizerSha256 = "a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f";

    /// <summary>SHA-256 of the Hub repository's own <c>tokenizer.json</c> at <see cref="Revision"/> (in <see cref="Lock"/>).</summary>
    public const string HubTokenizerSha256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42";

    /// <summary><c>&lt;|endoftext|&gt;</c>.</summary>
    public const int EndOfTextTokenId = 248044;

    /// <summary><c>&lt;|im_start|&gt;</c>: starts a turn.</summary>
    public const int ImStartTokenId = 248045;

    /// <summary><c>&lt;|im_end|&gt;</c>: ends an assistant turn.</summary>
    public const int ImEndTokenId = 248046;

    /// <summary>Rows of the embedding and the logits; the tokenizer's ids end at 248,076.</summary>
    public const int VocabularySize = 248320;

    /// <summary>
    /// The checkpoint's identity and decoder shape (text model of <c>config.json</c> at <see cref="Revision"/>: hidden
    /// size 1,024, 24 layers of which every fourth is full attention, head size 256, 262,144 positions). End-of-sequence
    /// ids are <see cref="EndOfTextTokenId"/> and <see cref="ImEndTokenId"/>, as in the built bundle's
    /// <c>genai_config.json</c>.
    /// </summary>
    public static ModelProfile Profile { get; } = new()
    {
        Id = "qwen3.5-0.8b",
        Repository = Repository,
        Revision = Revision,
        TokenizerProfile = TokenizerProfile,
        TokenizerSha256 = TokenizerSha256,
        ChatTemplateSha256 = Qwen35ChatTemplate.TemplateSha256,
        EosTokenIds = [EndOfTextTokenId, ImEndTokenId],
        VocabularySize = VocabularySize,
        HiddenSize = 1024,
        Layers = 24,
        HeadSize = 256,
        ContextLength = 262144,
        License = "Apache-2.0",
    };

    /// <summary>
    /// File name of <see cref="Lock"/> as a JSON lock (<c>schema_version</c> 2, no per-file <c>url</c>: every file comes
    /// from the Hub repository), shipped in this package under <c>locks/</c> for tools in other languages;
    /// <see cref="ModelAssetLock.Load"/> reads it back as an equal lock.
    /// </summary>
    public const string LockFileName = "qwen3.5-0.8b.checkpoint.lock.json";

    /// <summary>
    /// The asset lock of the Hub checkpoint's files a text runtime reads: <c>config.json</c>, the safetensors weights and
    /// their index, <c>tokenizer.json</c>, <c>tokenizer_config.json</c> and <c>chat_template.jinja</c>, downloaded from
    /// <see cref="Repository"/> at <see cref="Revision"/>. Sizes and digests were read from the Hub on 2026-10-26: the
    /// Hub's LFS SHA-256 for the safetensors file, and the SHA-256 of the downloaded bytes for every other file (the Hub's
    /// LFS digest of <c>tokenizer.json</c> agrees). This package does not download it; pass it to
    /// <see cref="ModelAssetStore.DownloadAsync(ModelAssetLock, IProgress{ModelAssetProgress}?, CancellationToken)"/> to fetch the checkpoint explicitly (about 1.76 GB).
    /// </summary>
    public static ModelAssetLock Lock { get; } = new("qwen3.5-0.8b-checkpoint", Repository, Revision,
    [
        new ModelAssetFile("chat_template.jinja", 7_755, Qwen35ChatTemplate.TemplateSha256),
        new ModelAssetFile("config.json", 2_907, "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204"),
        new ModelAssetFile("model.safetensors-00001-of-00001.safetensors", 1_746_942_600, "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"),
        new ModelAssetFile("model.safetensors.index.json", 50_900, "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e"),
        new ModelAssetFile("tokenizer.json", 12_807_982, HubTokenizerSha256),
        new ModelAssetFile("tokenizer_config.json", 16_709, "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c"),
    ]);
}
