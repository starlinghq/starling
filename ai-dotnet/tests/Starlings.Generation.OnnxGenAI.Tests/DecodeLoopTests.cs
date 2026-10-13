using System.Text;
using Starlings.Generation.OnnxGenAI;
using Starlings.TestSupport;
using Starlings.Tokenizers;

namespace Starlings.Generation.OnnxGenAI.Tests;

/// <summary>
/// The owned decode loop against scripted sequences with the real qwen35-bpe-chat tokenizer (no model weights): stop
/// tokens, the max-token boundary, stop sequences, streaming reassembly, UTF-8 safety, cancellation and validation.
/// </summary>
public sealed class DecodeLoopTests
{
    private const int ImEnd = 248046;

    private static CancellationToken Ct => TestContext.Current.CancellationToken;
    private const int EndOfText = 248044;
    private static readonly int[] Eos = [ImEnd, EndOfText];
    private static readonly int[] Prompt = [248045, 846, 198];

    private static Tokenizer RequireTokenizer()
    {
        var (tokenizer, reason) = Qwen35TestAssets.Tokenizer;
        Assert.SkipWhen(tokenizer is null, reason);
        return tokenizer!;
    }

    private static (TextGenerator Generator, List<FakeSequence> Sequences) Create(
        Tokenizer tokenizer, IReadOnlyList<int> script, int contextLength = 1000, int? pad = null, Action<int>? onStep = null, Action<FakeSequence>? onCreate = null)
    {
        var sequences = new List<FakeSequence>();
        var generator = new TextGenerator(
            o =>
            {
                var s = new FakeSequence(o, script, Eos, onStep);
                onCreate?.Invoke(s);
                sequences.Add(s);
                return s;
            },
            tokenizer,
            Eos,
            contextLength,
            pad);
        return (generator, sequences);
    }

    private static int[] Ids(Tokenizer tokenizer, string text) => [.. tokenizer.Encode(text, addSpecialTokens: false).Ids];

    [Fact]
    public void StopTokenEndsGenerationAndIsNotPartOfTheCompletion()
    {
        var tok = RequireTokenizer();
        int[] content = Ids(tok, "The capital of France is Paris.");
        var (gen, seqs) = Create(tok, [.. content, ImEnd, 11, 12]);
        var r = gen.Generate(Prompt, new GenerationOptions { MaxNewTokens = 64 }, Ct);
        Assert.Equal(GenerationFinishReason.Stop, r.FinishReason);
        Assert.Equal(ImEnd, r.StopTokenId);
        Assert.Null(r.StopSequence);
        Assert.Equal(content, r.TokenIds);
        Assert.Equal("The capital of France is Paris.", r.Text);
        Assert.Equal(new GenerationUsage(Prompt.Length, content.Length), r.Usage);
        Assert.Equal(content.Length + 1, seqs[0].Steps);
        Assert.Equal(Prompt.Length + 64, seqs[0].Options.MaxLength);
        Assert.Null(seqs[0].Options.Sampling);
        Assert.True(seqs[0].Disposed);
    }

    [Fact]
    public void EndOfTextIsAlsoAStopToken()
    {
        var tok = RequireTokenizer();
        var (gen, _) = Create(tok, [.. Ids(tok, "Hi"), EndOfText]);
        var r = gen.Generate(Prompt, null, Ct);
        Assert.Equal(EndOfText, r.StopTokenId);
        Assert.Equal("Hi", r.Text);
    }

    [Theory]
    [InlineData(1, "Length", 1, 1)]
    [InlineData(2, "Length", 2, 2)]
    [InlineData(3, "Length", 3, 3)] // the stop token would need a fourth step
    [InlineData(4, "Stop", 3, 4)]
    [InlineData(10, "Stop", 3, 4)]
    public void MaxNewTokensBoundsCompletionTokensAndNeverRunsAnExtraStep(int max, string finish, int completion, int steps)
    {
        var tok = RequireTokenizer();
        int[] content = Ids(tok, " one two three")[..3];
        var (gen, seqs) = Create(tok, [.. content, ImEnd]);
        var r = gen.Generate(Prompt, new GenerationOptions { MaxNewTokens = max }, Ct);
        Assert.Equal(finish, r.FinishReason.ToString());
        Assert.Equal(completion, r.Usage.CompletionTokens);
        Assert.Equal(content[..completion], r.TokenIds);
        Assert.Equal(steps, seqs[0].Steps);
    }

    [Fact]
    public void StopSequenceCutsTheTextAndNeverStreamsPartOfIt()
    {
        var tok = RequireTokenizer();
        int[] content = Ids(tok, "Answer: 42\nObservation: done");
        var (gen, _) = Create(tok, [.. content, ImEnd]);
        var pieces = new List<string>();
        GenerationResult? result = null;
        foreach (var u in gen.GenerateStreaming(Prompt, new GenerationOptions { StopSequences = ["\nObservation:"] }, Ct))
        {
            pieces.Add(u.Text);
            result ??= u.Result;
        }

        Assert.Equal(GenerationFinishReason.StopSequence, result!.FinishReason);
        Assert.Equal("\nObservation:", result.StopSequence);
        Assert.Null(result.StopTokenId);
        Assert.Equal("Answer: 42", result.Text);
        Assert.Equal(result.Text, string.Concat(pieces));
        Assert.DoesNotContain(pieces, p => p.Contains('\n', StringComparison.Ordinal));
        // Tokens up to the one that completed the match are kept and counted.
        Assert.Equal(result.TokenIds.Length, result.Usage.CompletionTokens);
        Assert.StartsWith("Answer: 42\nObservation:", tok.Decode(result.TokenIds.AsSpan(), false), StringComparison.Ordinal);
    }

    [Fact]
    public void EarliestStopSequenceWinsAndAnAbsentOneChangesNothing()
    {
        var tok = RequireTokenizer();
        int[] content = Ids(tok, "alpha beta gamma delta");
        var (gen, _) = Create(tok, [.. content, ImEnd]);
        var r = gen.Generate(Prompt, new GenerationOptions { StopSequences = ["delta", " beta"] }, Ct);
        Assert.Equal("alpha", r.Text);
        Assert.Equal(" beta", r.StopSequence);

        var (plain, _) = Create(tok, [.. content, ImEnd]);
        var a = plain.Generate(Prompt, new GenerationOptions { StopSequences = ["zz-never"] }, Ct);
        Assert.Equal("alpha beta gamma delta", a.Text);
        Assert.Equal(GenerationFinishReason.Stop, a.FinishReason);
    }

    [Fact]
    public void HeldBackStopPrefixIsReleasedWhenGenerationEnds()
    {
        var tok = RequireTokenizer();
        int[] content = Ids(tok, "result: <|ob");
        var (gen, _) = Create(tok, [.. content, ImEnd]);
        var updates = gen.GenerateStreaming(Prompt, new GenerationOptions { StopSequences = ["<|observation|>"] }, Ct).ToList();
        var result = updates[^1].Result!;
        Assert.Equal("result: <|ob", result.Text);
        Assert.Equal(GenerationFinishReason.Stop, result.FinishReason);
        Assert.Equal(result.Text, string.Concat(updates.Select(u => u.Text)));
    }

    [Fact]
    public void AdditionalStopTokenIdsPerRequest()
    {
        var tok = RequireTokenizer();
        int[] content = Ids(tok, "a b c d");
        var (gen, _) = Create(tok, [.. content, ImEnd]);
        var r = gen.Generate(Prompt, new GenerationOptions { AdditionalStopTokenIds = [content[2]] }, Ct);
        Assert.Equal(GenerationFinishReason.Stop, r.FinishReason);
        Assert.Equal(content[2], r.StopTokenId);
        Assert.Equal(content[..2], r.TokenIds);
    }

    [Fact]
    public void MultiByteCharacterSplitAcrossByteTokensIsEmittedWhole()
    {
        var tok = RequireTokenizer();
        // U+1F600 is F0 9F 98 80 in UTF-8: four single-byte tokens, then a space and a word.
        int[] bytes = [.. new byte[] { 0xF0, 0x9F, 0x98, 0x80 }.Select(b => ByteToken(tok, b))];
        int[] script = [.. bytes, .. Ids(tok, " ok"), ImEnd];
        var (gen, _) = Create(tok, script);
        var updates = gen.GenerateStreaming(Prompt, null, Ct).ToList();
        string emoji = char.ConvertFromUtf32(0x1F600);
        Assert.Equal(emoji, updates[0].Text);
        Assert.Equal(bytes, updates[0].TokenIds);
        Assert.All(updates, u => Assert.DoesNotContain((char)0xFFFD, u.Text));
        Assert.Equal(emoji + " ok", updates[^1].Result!.Text);
    }

    [Fact]
    public void IncompleteUtf8AtTheEndIsFlushedAsReplacementCharacter()
    {
        var tok = RequireTokenizer();
        int[] script = [.. Ids(tok, "x"), ByteToken(tok, 0xF0), ByteToken(tok, 0x9F), ImEnd];
        var (gen, _) = Create(tok, script);
        var updates = gen.GenerateStreaming(Prompt, null, Ct).ToList();
        var result = updates[^1].Result!;
        Assert.Equal(tok.Decode(result.TokenIds.AsSpan(), false), result.Text);
        Assert.EndsWith(((char)0xFFFD).ToString(), result.Text, StringComparison.Ordinal);
        Assert.Equal(result.Text, string.Concat(updates.Select(u => u.Text)));
    }

    [Fact]
    public void StreamingReassemblesTheCompletionOnRandomByteLevelSequences()
    {
        var tok = RequireTokenizer();
        var rng = new Random(20261010);
        int[] byteTokens = [.. Enumerable.Range(0, 256).Select(b => ByteToken(tok, (byte)b))];
        for (int trial = 0; trial < 300; trial++)
        {
            int n = rng.Next(1, 40);
            var script = new List<int>();
            for (int i = 0; i < n; i++)
            {
                int kind = rng.Next(10);
                script.Add(kind < 4 ? byteTokens[rng.Next(256)] : kind < 9 ? rng.Next(0, 248044) : rng.Next(248044, 248077));
            }

            script.RemoveAll(Eos.Contains);
            if (script.Count == 0)
            {
                continue;
            }

            script.Add(ImEnd);
            var (gen, _) = Create(tok, script);
            var updates = gen.GenerateStreaming(Prompt, null, Ct).ToList();
            var result = updates[^1].Result!;
            Assert.Equal(tok.Decode(result.TokenIds.AsSpan(), false), result.Text);
            Assert.Equal(result.Text, string.Concat(updates.Select(u => u.Text)));
            Assert.Equal(result.TokenIds, updates.SelectMany(u => u.TokenIds));
            Assert.All(updates.Take(updates.Count - 1), u => Assert.NotEmpty(u.Text));
            Assert.All(updates, u => Assert.False(u.Text.Length > 0 && char.IsHighSurrogate(u.Text[^1])));
        }
    }

    [Fact]
    public void CancellationBetweenStepsThrowsAndReleasesTheSequence()
    {
        var tok = RequireTokenizer();
        using var cts = new CancellationTokenSource();
        int[] content = Ids(tok, " one two three four five six seven eight");
        var (gen, seqs) = Create(tok, [.. content, ImEnd], onStep: step =>
        {
            if (step == 3)
            {
                cts.Cancel();
            }
        });
        var seen = new List<GenerationUpdate>();
        Assert.Throws<OperationCanceledException>(() =>
        {
            foreach (var u in gen.GenerateStreaming(Prompt, null, cts.Token))
            {
                seen.Add(u);
            }
        });
        Assert.Equal(3, seqs[0].Steps);
        Assert.True(seqs[0].Disposed);
        Assert.DoesNotContain(seen, u => u.IsFinal);
    }

    [Fact]
    public void AlreadyCancelledTokenThrowsBeforeThePrefill()
    {
        var tok = RequireTokenizer();
        var (gen, seqs) = Create(tok, [1, ImEnd]);
        using var cts = new CancellationTokenSource();
        cts.Cancel();
        Assert.Throws<OperationCanceledException>(() => gen.Generate(Prompt, null, cts.Token));
        Assert.Empty(seqs);
    }

    [Fact]
    public void DisposingTheEnumeratorEarlyReleasesTheSequence()
    {
        var tok = RequireTokenizer();
        var (gen, seqs) = Create(tok, [.. Ids(tok, " a b c d e f"), ImEnd]);
        using (var e = gen.GenerateStreaming(Prompt, null, Ct).GetEnumerator())
        {
            Assert.True(e.MoveNext());
        }

        Assert.True(seqs[0].Disposed);
    }

    [Fact]
    public void ArgumentsAreValidatedBeforeAnyWork()
    {
        var tok = RequireTokenizer();
        var (gen, seqs) = Create(tok, [1, ImEnd], contextLength: 10, pad: 248319);
        Assert.Throws<ArgumentException>(() => gen.GenerateStreaming([], null, Ct));
        Assert.Throws<ArgumentException>(() => gen.GenerateStreaming([1, 248319, 2], null, Ct));
        var ex = Assert.Throws<GenerationContextLengthException>(() => gen.GenerateStreaming(Prompt, new GenerationOptions { MaxNewTokens = 8 }, Ct));
        Assert.Equal(10, ex.ContextLength);
        Assert.Throws<ArgumentOutOfRangeException>(() => gen.GenerateStreaming(Prompt, new GenerationOptions { MaxNewTokens = 0 }, Ct));
        Assert.Throws<ArgumentException>(() => gen.GenerateStreaming(Prompt, new GenerationOptions { MaxNewTokens = 2, StopSequences = [""] }, Ct));
        Assert.Throws<ArgumentOutOfRangeException>(() => gen.GenerateStreaming(Prompt, new GenerationOptions { MaxNewTokens = 2, Sampling = new SamplingOptions { Seed = 1, Temperature = 0 } }, Ct));
        Assert.Throws<ArgumentOutOfRangeException>(() => gen.GenerateStreaming(Prompt, new GenerationOptions { MaxNewTokens = 2, Sampling = new SamplingOptions { Seed = 1, TopP = 1.5 } }, Ct));
        Assert.Throws<ArgumentOutOfRangeException>(() => gen.GenerateStreaming(Prompt, new GenerationOptions { MaxNewTokens = 2, Sampling = new SamplingOptions { Seed = 1, TopK = 0 } }, Ct));
        Assert.Throws<ArgumentOutOfRangeException>(() => gen.GenerateStreaming(Prompt, new GenerationOptions { MaxNewTokens = 2, Sampling = new SamplingOptions { Seed = -1 } }, Ct));
        Assert.Empty(seqs);
    }

    [Fact]
    public void PrefillIsChunkedAndCancellableBetweenChunks()
    {
        var tok = RequireTokenizer();
        int[] prompt = [.. Enumerable.Range(100, 23)];
        var (gen, seqs) = Create(tok, [1, ImEnd]);
        gen.Generate(prompt, new GenerationOptions { MaxNewTokens = 4, PrefillChunkTokens = 10 }, Ct);
        Assert.Equal([10, 10, 3], seqs[0].Appends);
        gen.Generate(prompt, new GenerationOptions { MaxNewTokens = 4 }, Ct);
        Assert.Equal([23], seqs[1].Appends);
        Assert.Throws<ArgumentOutOfRangeException>(() => gen.GenerateStreaming(prompt, new GenerationOptions { PrefillChunkTokens = 0 }, Ct));

        // Cancelled during the first chunk: the next chunk boundary throws and the sequence is released.
        using var cts = new CancellationTokenSource();
        var (cancelled, cseqs) = Create(tok, [1, ImEnd], onCreate: s => s.OnAppend = n => cts.Cancel());
        Assert.Throws<OperationCanceledException>(() => cancelled.Generate(prompt, new GenerationOptions { MaxNewTokens = 4, PrefillChunkTokens = 10 }, cts.Token));
        Assert.Equal([10], cseqs[0].Appends);
        Assert.True(cseqs[0].Disposed);
    }

    [Fact]
    public void SamplingSettingsReachTheSequence()
    {
        var tok = RequireTokenizer();
        var (gen, seqs) = Create(tok, [1, ImEnd]);
        var sampling = new SamplingOptions { Seed = 3, Temperature = 0.7, TopK = 20, TopP = 0.8 };
        gen.Generate(Prompt, new GenerationOptions { MaxNewTokens = 5, Sampling = sampling }, Ct);
        Assert.Equal(sampling, seqs[0].Options.Sampling);
    }

    /// <summary>The byte-level token of one byte (GPT-2 bytes-to-unicode mapping).</summary>
    internal static int ByteToken(Tokenizer tokenizer, byte b)
    {
        var printable = new List<int>();
        printable.AddRange(Enumerable.Range('!', '~' - '!' + 1));
        printable.AddRange(Enumerable.Range(0xA1, 0xAC - 0xA1 + 1));
        printable.AddRange(Enumerable.Range(0xAE, 0xFF - 0xAE + 1));
        int code;
        if (printable.Contains(b))
        {
            code = b;
        }
        else
        {
            int n = 0;
            for (int i = 0; i < b; i++)
            {
                if (!printable.Contains(i))
                {
                    n++;
                }
            }

            code = 256 + n;
        }

        return tokenizer.TokenToId(new string((char)code, 1)) ?? throw new InvalidOperationException($"no byte token for {b:X2}");
    }

    internal static string Utf8(params byte[] bytes) => Encoding.UTF8.GetString(bytes);
}
