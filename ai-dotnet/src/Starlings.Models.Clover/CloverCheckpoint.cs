using Starlings.Acquisition;
using Starlings.Generation;

namespace Starlings.Models.Clover;

/// <summary>
/// The pinned checkpoint <c>Diffuspace/Clover-35B-A3B-Preview</c>, independent of a runtime: its repository and commit,
/// the special token ids the template, the owned tokenizer and the block-diffusion decoder use, the decoder shape of
/// <c>config.json</c>, its <see cref="ModelProfile"/> and the asset lock of its Hub files.
/// </summary>
/// <remarks>
/// Clover is a block-diffusion fine-tune of <c>Qwen/Qwen3.6-35B-A3B</c>: a hybrid Gated DeltaNet and full-attention
/// mixture-of-experts decoder (40 layers, 256 routed experts, 8 per token, one shared expert) with added bidirectional
/// GDN parameters. It generates a block of <see cref="BlockSize"/> masked positions (<see cref="MaskTokenId"/>) at a
/// time and fills the confident ones in parallel. No runtime in this repository runs it (<see cref="CloverCapabilities"/>).
/// </remarks>
public static class CloverCheckpoint
{
    /// <summary>Hub repository of the checkpoint.</summary>
    public const string Repository = "Diffuspace/Clover-35B-A3B-Preview";

    /// <summary>Pinned commit of <see cref="Repository"/>.</summary>
    public const string Revision = "4b0402b65e523d0183a14c431955438714cb16ff";

    /// <summary>Owned tokenizer profile (Starlings.Tokenizers, starlinghq/tokenizers#20).</summary>
    public const string TokenizerProfile = "clover-bpe-chat";

    /// <summary>
    /// SHA-256 of the checkpoint's <c>tokenizer.json</c> (the <c>clover-bpe-chat</c> profile asset; transformers 5.19.0
    /// runs this file as is).
    /// </summary>
    public const string TokenizerSha256 = "7c62ba7d8bb13291c94459896cd27537858beb58a88a037b09a150e213420988";

    /// <summary>
    /// <c>&lt;|mask|&gt;</c>: the diffusion mask (<c>mask_token_id</c> of <c>config.json</c>). It takes the vocabulary id
    /// of a former regular token, inside the base model's id range.
    /// </summary>
    public const int MaskTokenId = 151669;

    /// <summary><c>&lt;|endoftext|&gt;</c>: <c>pad_token_id</c> and <c>bos_token_id</c> of the generation config, and an end-of-sequence id.</summary>
    public const int EndOfTextTokenId = 248044;

    /// <summary><c>&lt;|im_start|&gt;</c>: starts a turn.</summary>
    public const int ImStartTokenId = 248045;

    /// <summary><c>&lt;|im_end|&gt;</c>: ends an assistant turn; an end-of-sequence id.</summary>
    public const int ImEndTokenId = 248046;

    /// <summary><c>&lt;think&gt;</c> (non-special added token).</summary>
    public const int ThinkTokenId = 248068;

    /// <summary><c>&lt;/think&gt;</c>: <c>dllm_think_end_token_id</c> of the generation config.</summary>
    public const int ThinkEndTokenId = 248069;

    /// <summary>Rows of the embedding and the logits; the tokenizer's ids end at 248,076.</summary>
    public const int VocabularySize = 248320;

    /// <summary>Hidden size.</summary>
    public const int HiddenSize = 2048;

    /// <summary>Decoder layers: 30 Gated DeltaNet (<c>linear_attention</c>) and 10 full attention, every fourth.</summary>
    public const int Layers = 40;

    /// <summary>Head width of the full-attention layers.</summary>
    public const int HeadSize = 256;

    /// <summary>Routed experts per layer.</summary>
    public const int Experts = 256;

    /// <summary>Routed experts per token.</summary>
    public const int ExpertsPerToken = 8;

    /// <summary>Positions denoised together (<c>block_size</c> of <c>config.json</c>).</summary>
    public const int BlockSize = 32;

    /// <summary>The checkpoint's context, prompt and completion together (<c>max_position_embeddings</c>).</summary>
    public const int ContextLength = 131072;

    /// <summary>
    /// The checkpoint's identity and decoder shape. The end-of-sequence ids are <see cref="EndOfTextTokenId"/> and
    /// <see cref="ImEndTokenId"/>, the <c>eos_token_id</c> list of the generation config.
    /// </summary>
    public static ModelProfile Profile { get; } = new()
    {
        Id = "clover-35b-a3b-preview",
        Repository = Repository,
        Revision = Revision,
        TokenizerProfile = TokenizerProfile,
        TokenizerSha256 = TokenizerSha256,
        ChatTemplateSha256 = CloverChatTemplate.TemplateSha256,
        EosTokenIds = [EndOfTextTokenId, ImEndTokenId],
        VocabularySize = VocabularySize,
        HiddenSize = HiddenSize,
        Layers = Layers,
        HeadSize = HeadSize,
        ContextLength = ContextLength,
        License = "Apache-2.0",
    };

    /// <summary>
    /// The asset lock of the Hub checkpoint's model files: <c>config.json</c>, <c>generation_config.json</c>, the 26
    /// bf16 safetensors shards and their index, the two bidirectional GDN parameter files, <c>tokenizer.json</c>,
    /// <c>tokenizer_config.json</c> and <c>chat_template.jinja</c>, at <see cref="Revision"/> (about 69.3 GB). Sizes and
    /// digests were read from the Hub on 2026-10-28: the Hub's LFS SHA-256 for the safetensors files and
    /// <c>tokenizer.json</c>, and the SHA-256 of the downloaded bytes for every other file. The checkpoint's Python
    /// modeling code is not locked: nothing here runs it. This package does not download it; pass it to
    /// <see cref="ModelAssetStore.DownloadAsync(ModelAssetLock, IProgress{ModelAssetProgress}?, CancellationToken)"/> to
    /// fetch the checkpoint explicitly.
    /// </summary>
    public static ModelAssetLock Lock { get; } = new("clover-35b-a3b-preview-checkpoint", Repository, Revision,
    [
        new ModelAssetFile("bd_gdn_conv_right_weight.safetensors", 1_478_024, "6b27471aa645628c403ce1cba40105528f60aa9bd5b4e873b5ffaa2074ad71d4"),
        new ModelAssetFile("bd_gdn_softmax_scalars.safetensors", 15_584, "a8165aee2433e9b795bcc572a80c53fdfa1585a1b84534af4c20bbdb1332037a"),
        new ModelAssetFile("chat_template.jinja", 7_912, CloverChatTemplate.TemplateSha256),
        new ModelAssetFile("config.json", 2_575, "e07639edf658af63dbbb52b6eb488d72d1a2f194f15638539b02e3e94b76ecf5"),
        new ModelAssetFile("generation_config.json", 588, "4e5bac9359cb9f9ddefefb302e529c8351c919c4b27bb8f139157246cab58335"),
        new ModelAssetFile("model-00001-of-00026.safetensors", 3_835_692_216, "c340eef575e465c061812a527d427f25b1cd09e850b453a936083b3d769f1751"),
        new ModelAssetFile("model-00002-of-00026.safetensors", 552_234_728, "2a24ff37d42bf33c8cefd58d96d8f502f98a3da0889d086f5eaa1648ced03610"),
        new ModelAssetFile("model-00003-of-00026.safetensors", 3_357_897_832, "f7d0ec384ebe97ad56b6f3468cb15bb4d55281327ddbfdbc7c359c58e2038bcd"),
        new ModelAssetFile("model-00004-of-00026.safetensors", 3_370_808_144, "0b88421141e09e9cec2316914bf3e5e73d62a0c03a2d4486c7c09932649ff6a0"),
        new ModelAssetFile("model-00005-of-00026.safetensors", 3_357_897_832, "a065d9424ab410e2ddc5e8d0d4f417282e3993d5ea612898ea64c04632c30d67"),
        new ModelAssetFile("model-00006-of-00026.safetensors", 3_959_424_664, "e61750b3be8c7934608df1bf4b9a9e15fdb5ae0c624899321ce1d7887323b23e"),
        new ModelAssetFile("model-00007-of-00026.safetensors", 1_096_787_600, "92a452f10d75a41d2e5eaaba46858dcecfec400ba8f853122460e31c1f825085"),
        new ModelAssetFile("model-00008-of-00026.safetensors", 3_946_841_768, "226e7fb60fbdfe85f3671ebf471f503930b721e6ff06b62c50d460ff2f5e5ca2"),
        new ModelAssetFile("model-00009-of-00026.safetensors", 1_096_460_264, "73ff108e69d42fc9d2353067209546b7b25833796b7e3e1d1290a995d98e4807"),
        new ModelAssetFile("model-00010-of-00026.safetensors", 3_946_841_752, "71b8223e3f135d9f0215f07ba2ffa376d8177ac0913d119e531728813f6e2d21"),
        new ModelAssetFile("model-00011-of-00026.safetensors", 1_096_460_168, "e2a6b92e994b28b7b9125f443dce256668b140dcd71f3c65cb03ac1e31552d44"),
        new ModelAssetFile("model-00012-of-00026.safetensors", 3_409_970_832, "221c34e566494b044813aa4b4de82350fa8ee1601323d2d52224a7db73979180"),
        new ModelAssetFile("model-00013-of-00026.safetensors", 1_633_331_080, "752a860ad1593e9d273d215ab7aa194ce95c35cea75f5d60498142d07a725ccf"),
        new ModelAssetFile("model-00014-of-00026.safetensors", 3_422_553_648, "4351cbf05ee65e8c0b8a4316324423e2bb9874fd98b062f9ad02bd1cd34c5319"),
        new ModelAssetFile("model-00015-of-00026.safetensors", 1_633_658_576, "b6a1f0d82bb6145632e369a1de85adde3446cd721b5cced28b4a037f41d34fe6"),
        new ModelAssetFile("model-00016-of-00026.safetensors", 3_946_841_880, "10ada7f1814b1ef6da75e913e946af05b135bef1cc5c8e874c11e03f238f76e7"),
        new ModelAssetFile("model-00017-of-00026.safetensors", 1_096_460_032, "e598e5845299541f4c38fa0bb80e20e2c39c525e9676880e3b2477e9740df939"),
        new ModelAssetFile("model-00018-of-00026.safetensors", 3_946_841_752, "cdaa56291ae23785a489f9e9fbd063fe25e9b1a7e228d787b910a2e3559ba3eb"),
        new ModelAssetFile("model-00019-of-00026.safetensors", 1_096_460_224, "66550744d97ffa0b9f8f0cad1b66a7a1426ed0b99404ae0731e51e622ca705d1"),
        new ModelAssetFile("model-00020-of-00026.safetensors", 3_409_970_832, "4d9a696b1efc80ef80554cbfb09629e9355e8082bf7eaae7053b67a9eb9556d1"),
        new ModelAssetFile("model-00021-of-00026.safetensors", 1_633_331_152, "96cf4f331f51936dac903f9e8726c2d4125b7fda97bebd2bee34eefa319a2ceb"),
        new ModelAssetFile("model-00022-of-00026.safetensors", 3_370_808_176, "d5e11497960fe856c241662ed6e1bb0bda234baf42b8b7f09421c6bfe35a63e3"),
        new ModelAssetFile("model-00023-of-00026.safetensors", 3_357_897_864, "d9cf73995d153079fdec0e87f83c711914457331139f11f06fa312b15f439308"),
        new ModelAssetFile("model-00024-of-00026.safetensors", 3_370_808_176, "adcf9389c87d8e95d76cad82d35100317d164331cce2fe9ddecf3d5c68f372f9"),
        new ModelAssetFile("model-00025-of-00026.safetensors", 2_759_146_000, "52990ce8792a4cda271a9aad4135ff4f5707c9b91c3051afe1d7ffae071be593"),
        new ModelAssetFile("model-00026-of-00026.safetensors", 1_615_874_872, "7fb786f9db4227edd863e0cc00f777f454bff460671cda3dd63f6faec2f79592"),
        new ModelAssetFile("model.safetensors.index.json", 70_668, "6270f95c35739b2c40db0026e8683fb84544c258fa637cd88a3b4887a79fd063"),
        new ModelAssetFile("tokenizer.json", 19_989_133, TokenizerSha256),
        new ModelAssetFile("tokenizer_config.json", 6_742, "3399c8d380151f9c1bf562257c0063665b74def51735f4d0b01086e63266950e"),
    ]);
}
