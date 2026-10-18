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

    /// <summary>Whether decoder layer <paramref name="layer"/> is a full-attention layer.</summary>
    public static bool IsFullAttention(int layer) => layer % 4 == 3;
}
