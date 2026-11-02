using Starlings.Acquisition;

namespace Starlings.Models.Kev;

/// <summary>A Kev 1.0 checkpoint this package has a contract for.</summary>
public enum KevModel
{
    /// <summary><c>jaredpalmer/kev-0.8b</c> on <c>Qwen/Qwen3.5-0.8B-Base</c> (profiles <c>kev-0.8b/onnx-fp32-cpu</c> and <c>onnx-fp32-cuda</c>).</summary>
    Kev08B,

    /// <summary><c>jaredpalmer/kev-4b</c> on <c>Qwen/Qwen3.5-4B-Base</c> (profiles <c>kev-4b/onnx-fp32-cpu</c> and <c>onnx-fp32-cuda</c>).</summary>
    Kev4B,
}

/// <summary>
/// The behavior contract of one Kev profile: pinned sources, backbone shape, pointer head and limits. Every value is
/// checked against the files at load time; anything else is refused. The pipeline's candidate contract
/// (<c>model-pipeline/candidates/kev-*/profile.json</c>) holds the same values.
/// </summary>
public sealed record KevModelProfile
{
    /// <summary>Delimiter <c>&lt;|fim_prefix|&gt;</c>: opens the state.</summary>
    public const int StateTokenId = 248060;

    /// <summary>Delimiter <c>&lt;|fim_middle|&gt;</c>: opens a question's instruction.</summary>
    public const int QuestionTokenId = 248061;

    /// <summary>Delimiter <c>&lt;|box_start|&gt;</c>: opens an option.</summary>
    public const int OptionTokenId = 248049;

    /// <summary>Delimiter <c>&lt;|box_end|&gt;</c>: closes an option; its hidden state is the option's key.</summary>
    public const int OptionEndTokenId = 248050;

    /// <summary>Delimiter <c>&lt;|fim_suffix|&gt;</c>: ends a question; its hidden state is the question's query.</summary>
    public const int DecideTokenId = 248062;

    /// <summary>Kev's pad token <c>&lt;|endoftext|&gt;</c> (never attended to).</summary>
    public const int PadTokenId = 248044;

    /// <summary>Pointer head projection width.</summary>
    public const int HeadDim = 256;

    /// <summary>
    /// Validated context (model cards, "Long documents"): states of at most 8,192 tokens, the <c>&lt;state&gt;</c> token
    /// included. Upstream serves up to 65,536 with accuracy loss above 8,192; this package refuses longer states.
    /// </summary>
    public const int MaxStateTokens = 8192;

    /// <summary>
    /// Row limit of upstream's 8K serving context (<c>SERVE_MAX_BRANCH_8K</c>): the state plus one question's branch.
    /// </summary>
    public const int MaxRowTokens = 8192;

    /// <summary>Most options per question (TypeSafe System One API, <c>kev.api.MAX_OPTIONS</c>).</summary>
    public const int MaxOptions = 255;

    /// <summary>Kev source commit (tag <c>kev-1.0</c>) whose encoding, head and answer mapping this package follows.</summary>
    public const string SourceRevision = "6b719c3c3f367295f6ef336f4f751cf5ff970abc";

    /// <summary>Tokenizer profile of <c>Starlings.Tokenizers</c> (the Kev repositories' <c>tokenizer.json</c>, identical for both sizes).</summary>
    public const string TokenizerProfile = "qwen2-bpe-kev";

    /// <summary>The Kev-0.8B contract.</summary>
    public static KevModelProfile Kev08B { get; } = new()
    {
        Model = KevModel.Kev08B,
        Id = "kev-0.8b/onnx-fp32-cpu",
        CudaId = "kev-0.8b/onnx-fp32-cuda",
        Repository = "jaredpalmer/kev-0.8b",
        Revision = "9a45d25eb2ab761841196625383fa1dff0e56c1e",
        BaseRepository = "Qwen/Qwen3.5-0.8B-Base",
        BaseRevision = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
        HeadPtSha256 = "f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6",
        AdapterSha256 = "9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a",
        HiddenSize = 1024,
        Layers = 24,
        KeyValueHeads = 2,
        LinearKeyHeads = 16,
        LinearValueHeads = 16,
        Temperature = 2.3510958125672174,
        CheckpointLock = new("kev-0.8b-checkpoint", "jaredpalmer/kev-0.8b", "9a45d25eb2ab761841196625383fa1dff0e56c1e",
        [
            new ModelAssetFile("adapter_config.json", 1_273, "748acb2cda88454cb1ba69d745ba336f3fcb5486eac349e90960c8b8d8d3e854"),
            new ModelAssetFile("adapter_model.safetensors", 43_338_624, "9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a"),
            new ModelAssetFile("head.pt", 2_103_999, "f400bd12802b2b105ae45d6b03774a158a3db4fccff42413734ddca2e5c920b6"),
            new ModelAssetFile("provenance.json", 4_221, "fce039fbda7de19451c9b5d3065aee2c5e063e1562a8bc18c72e297fbd15e9d8"),
            new ModelAssetFile("tokenizer.json", 19_989_325, "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
            new ModelAssetFile("tokenizer_config.json", 1_128, "8671bed7c852ce9e661be94f179a7b4ffd091c2a65aea0363e5501c20318ee45"),
        ]),
        BaseCheckpointLock = new("qwen3.5-0.8b-base-checkpoint", "Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
        [
            new ModelAssetFile("config.json", 2_907, "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204"),
            new ModelAssetFile("merges.txt", 3_353_259, "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"),
            new ModelAssetFile("model.safetensors-00001-of-00001.safetensors", 1_746_942_600, "c2b1e5a17d9c1e27685d92ed9b382911ebb99955ecd89052d1721241adfbab6c"),
            new ModelAssetFile("model.safetensors.index.json", 50_900, "ce9a885efdf27d3664fdef5d512ad365216f1074051ef840c7cd8e5431495d0a"),
            new ModelAssetFile("preprocessor_config.json", 390, "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"),
            new ModelAssetFile("tokenizer.json", 12_807_196, "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927"),
            new ModelAssetFile("tokenizer_config.json", 16_712, "e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de"),
            new ModelAssetFile("video_preprocessor_config.json", 386, "d039cd7d88b3502a99edd455496b75b44ecf7dd3b3669748bafac37e6cecc085"),
            new ModelAssetFile("vocab.json", 6_722_759, "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"),
        ]),
    };

    /// <summary>The Kev-4B contract.</summary>
    public static KevModelProfile Kev4B { get; } = new()
    {
        Model = KevModel.Kev4B,
        Id = "kev-4b/onnx-fp32-cpu",
        CudaId = "kev-4b/onnx-fp32-cuda",
        Repository = "jaredpalmer/kev-4b",
        Revision = "139fdd94f1b6a6ad80cc15e08fcb99cac885a101",
        BaseRepository = "Qwen/Qwen3.5-4B-Base",
        BaseRevision = "1001bb4d826a52d1f399e183466143f4da7b741b",
        HeadPtSha256 = "dd633435998ecc751ac538717a3742e32149500fabf7d7276287dbf0693f347c",
        AdapterSha256 = "90e817356246e7f18bfa7ca3d31794cd4fbeb3332a66a84cb51d9ceae925f2b2",
        HiddenSize = 2560,
        Layers = 32,
        KeyValueHeads = 4,
        LinearKeyHeads = 16,
        LinearValueHeads = 32,
        Temperature = 2.406050072164233,
        CheckpointLock = new("kev-4b-checkpoint", "jaredpalmer/kev-4b", "139fdd94f1b6a6ad80cc15e08fcb99cac885a101",
        [
            new ModelAssetFile("adapter_config.json", 1_271, "8a05dfd6c5e7f61a62e6db5d8de8093a8dbbd105aa7ee7b5e7b73f6249f45617"),
            new ModelAssetFile("adapter_model.safetensors", 129_924_032, "90e817356246e7f18bfa7ca3d31794cd4fbeb3332a66a84cb51d9ceae925f2b2"),
            new ModelAssetFile("head.pt", 5_249_791, "dd633435998ecc751ac538717a3742e32149500fabf7d7276287dbf0693f347c"),
            new ModelAssetFile("provenance.json", 4_265, "7aa999a629f25d93713950d6fb39d0b4c37fe8a9f43d2673a79c4c455d42b26a"),
            new ModelAssetFile("tokenizer.json", 19_989_325, "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
            new ModelAssetFile("tokenizer_config.json", 1_128, "8671bed7c852ce9e661be94f179a7b4ffd091c2a65aea0363e5501c20318ee45"),
        ]),
        BaseCheckpointLock = new("qwen3.5-4b-base-checkpoint", "Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b",
        [
            new ModelAssetFile("config.json", 3_161, "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"),
            new ModelAssetFile("merges.txt", 3_353_259, "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"),
            new ModelAssetFile("model.safetensors-00001-of-00002.safetensors", 5_329_398_712, "df547074dce70532a0493e5433152bd17a65efb89088cfabc2e7e2371a93d712"),
            new ModelAssetFile("model.safetensors-00002-of-00002.safetensors", 3_990_429_344, "590fbaac095dd31db886c322d9d2f7df47777966391acf306ddddc3e4e3a15ef"),
            new ModelAssetFile("model.safetensors.index.json", 76_196, "eae340074abb0a5f31a6621f7ae8e8248a7c1790df04a722c4e4b70c2a6d1dbb"),
            new ModelAssetFile("preprocessor_config.json", 390, "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"),
            new ModelAssetFile("tokenizer.json", 12_807_196, "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927"),
            new ModelAssetFile("tokenizer_config.json", 16_713, "3891e840d7dc5fca0af33d3a25083a735e36fe06214e3f707024820cb6b9f89c"),
            new ModelAssetFile("video_preprocessor_config.json", 386, "d039cd7d88b3502a99edd455496b75b44ecf7dd3b3669748bafac37e6cecc085"),
            new ModelAssetFile("vocab.json", 6_722_759, "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"),
        ]),
    };

    /// <summary>Both profiles.</summary>
    public static IReadOnlyList<KevModelProfile> All { get; } = [Kev08B, Kev4B];

    /// <summary>The profile of a model.</summary>
    public static KevModelProfile For(KevModel model) => model switch
    {
        KevModel.Kev08B => Kev08B,
        KevModel.Kev4B => Kev4B,
        _ => throw new ArgumentOutOfRangeException(nameof(model), model, "Unknown Kev model."),
    };

    /// <summary>The profile of a Hub repository, or null when this package has no contract for it.</summary>
    public static KevModelProfile? ForRepository(string repository) =>
        All.FirstOrDefault(p => string.Equals(p.Repository, repository, StringComparison.Ordinal));

    /// <summary>The checkpoint.</summary>
    public required KevModel Model { get; init; }

    /// <summary>Registry profile id on the CPU provider.</summary>
    public required string Id { get; init; }

    /// <summary>Registry profile id on the CUDA provider (the same bundle and lock, <c>use_tf32=0</c>).</summary>
    public required string CudaId { get; init; }

    /// <summary>Hub repository of the adapter, head and tokenizer.</summary>
    public required string Repository { get; init; }

    /// <summary>Pinned commit of <see cref="Repository"/> (the released weights).</summary>
    public required string Revision { get; init; }

    /// <summary>Hub repository of the frozen base the adapter is merged into.</summary>
    public required string BaseRepository { get; init; }

    /// <summary>Pinned commit of <see cref="BaseRepository"/>.</summary>
    public required string BaseRevision { get; init; }

    /// <summary>SHA-256 of the source <c>head.pt</c> (recorded in <c>head.json</c> by the merge).</summary>
    public required string HeadPtSha256 { get; init; }

    /// <summary>SHA-256 of the source <c>adapter_model.safetensors</c> (recorded in <c>head.json</c> by the merge).</summary>
    public required string AdapterSha256 { get; init; }

    /// <summary>Backbone hidden size (the pointer head's input).</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Decoder layers; every fourth is full attention, the others Gated DeltaNet.</summary>
    public required int Layers { get; init; }

    /// <summary>Key/value heads of the full-attention layers.</summary>
    public required int KeyValueHeads { get; init; }

    /// <summary>Key heads of the Gated DeltaNet layers.</summary>
    public required int LinearKeyHeads { get; init; }

    /// <summary>Value heads of the Gated DeltaNet layers (the recurrent state's head dimension).</summary>
    public required int LinearValueHeads { get; init; }

    /// <summary>Calibration temperature stored in <c>head.pt</c>.</summary>
    public required double Temperature { get; init; }

    /// <summary>
    /// The asset lock of the Kev Hub checkpoint's files, downloaded from <see cref="Repository"/> at
    /// <see cref="Revision"/>: <c>adapter_config.json</c>, <c>adapter_model.safetensors</c> (<see cref="AdapterSha256"/>),
    /// <c>head.pt</c> (<see cref="HeadPtSha256"/>), <c>provenance.json</c>, <c>tokenizer.json</c> and
    /// <c>tokenizer_config.json</c>: the pipeline candidate contract's <c>origin.files</c>
    /// (<c>model-pipeline/candidates/kev-*/profile.json</c>) without the card. On 2026-11-01 every size and every LFS
    /// digest agreed with the Hub at the pinned revision, and the SHA-256 of every other file was recomputed from the
    /// downloaded bytes. Shipped in this package as <c>locks/</c><see cref="CheckpointLockFileName"/> (JSON lock
    /// <c>schema_version</c> 2). Nothing downloads it; pass it to
    /// <see cref="ModelAssetStore.DownloadAsync(ModelAssetLock, IProgress{ModelAssetProgress}?, CancellationToken)"/>
    /// to fetch the files explicitly.
    /// </summary>
    public required ModelAssetLock CheckpointLock { get; init; }

    /// <summary>
    /// The asset lock of the frozen base checkpoint's files, downloaded from <see cref="BaseRepository"/> at
    /// <see cref="BaseRevision"/>: the configuration, the safetensors weights and their index, the tokenizer files and
    /// the preprocessor configurations (the candidate contract's <c>origin.base.files</c> without the card and license,
    /// checked against the Hub as <see cref="CheckpointLock"/> was). The merge reads these with
    /// <see cref="CheckpointLock"/> to build the ONNX bundle (about 1.8 GB of base files for Kev-0.8B, 9.3 GB for
    /// Kev-4B). Shipped in this package as <c>locks/</c><see cref="BaseCheckpointLockFileName"/>.
    /// </summary>
    public required ModelAssetLock BaseCheckpointLock { get; init; }

    /// <summary>File name of <see cref="CheckpointLock"/> as a JSON lock in this package (<c>kev-0.8b.checkpoint.lock.json</c>, <c>kev-4b.checkpoint.lock.json</c>).</summary>
    public string CheckpointLockFileName => LockFileName(CheckpointLock);

    /// <summary>File name of <see cref="BaseCheckpointLock"/> as a JSON lock in this package (<c>qwen3.5-0.8b-base.checkpoint.lock.json</c>, <c>qwen3.5-4b-base.checkpoint.lock.json</c>).</summary>
    public string BaseCheckpointLockFileName => LockFileName(BaseCheckpointLock);

    // "<id>-checkpoint" is written as "<id>.checkpoint.lock.json", as Qwen35Checkpoint.LockFileName.
    private static string LockFileName(ModelAssetLock l) => $"{l.Name[..^"-checkpoint".Length]}.checkpoint.lock.json";

    /// <summary>Whether decoder layer <paramref name="layer"/> is a full-attention layer.</summary>
    public static bool IsFullAttention(int layer) => layer % 4 == 3;
}
