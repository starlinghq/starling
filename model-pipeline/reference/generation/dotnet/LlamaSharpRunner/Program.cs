// M0 generation-backend check, lane B: llama.cpp through LLamaSharp 0.27.0.
//
// Modes:
//   parity     feed the reference prompt ids, greedy argmax on raw logits, write results.jsonl + step0/*.f32;
//              also render with LLamaTemplate and tokenize with the GGUF tokenizer (template/tokenizer boundary).
//   semantics  streaming, cancellation, KV/state reuse, sequence isolation; writes semantics.json.
//   bench      exploratory warm timings (prefill, TTFT, decode); writes bench.json.
//
// Usage: LlamaSharpRunner --mode parity|semantics|bench --model <gguf> --prompts <jsonl> --reference <ref dir>
//        --out <dir> --device cpu|cuda [--threads N] [--native-dir <dir with libllama.so>] [--samples N]
using System.Diagnostics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using GenCheck;
using LLama;
using LLama.Common;
using LLama.Native;
using LLama.Sampling;

var opts = ParseArgs(args);
var mode = opts["mode"];
var device = opts.GetValueOrDefault("device", "cpu");
var threads = int.Parse(opts.GetValueOrDefault("threads", "6"));
var outDir = opts["out"];
Directory.CreateDirectory(outDir);

var nativeLog = new List<string>();
if (opts.TryGetValue("native-dir", out var nativeDir))
{
    NativeLibraryConfig.All.WithLibrary(Path.Combine(nativeDir, "libllama.so"), Path.Combine(nativeDir, "libmtmd.so"));
}
else
{
    NativeLibraryConfig.All.WithCuda(device == "cuda").WithAutoFallback(false);
}
NativeLibraryConfig.All.WithLogCallback((level, message) =>
{
    if (level >= LLamaLogLevel.Warning || message.Contains("CUDA") || message.Contains("loaded"))
    {
        lock (nativeLog)
        {
            nativeLog.Add($"{level}: {message.TrimEnd()}");
        }
    }
});

var mp = new ModelParams(opts["model"])
{
    ContextSize = 4096,
    GpuLayerCount = device == "cuda" ? 99 : 0,
    Threads = threads,
    BatchThreads = threads,
    BatchSize = 2048,
    UBatchSize = 512,
    SeqMax = 4,
};
using var weights = LLamaWeights.LoadFromFile(mp);
using var context = weights.CreateContext(mp);
var vocab = weights.Vocab;
var nVocab = vocab.Count;

var env = Shared.EnvironmentInfo();
env["runner"] = "model-pipeline/reference/generation/dotnet/LlamaSharpRunner";
env["llamasharp"] = typeof(LLamaWeights).Assembly.GetName().Version?.ToString();
env["llamasharp_informational"] = typeof(LLamaWeights).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
env["device"] = device;
env["threads"] = threads;
env["n_gpu_layers"] = mp.GpuLayerCount;
env["context_size"] = mp.ContextSize;
env["batch_size"] = mp.BatchSize;
env["ubatch_size"] = mp.UBatchSize;
env["model"] = opts["model"];
env["model_sha256"] = Sha256(opts["model"]);
env["native_libraries"] = LoadedNativeLibs();
env["n_vocab"] = nVocab;
env["vocab"] = VocabInfo();
env["system_info"] = System.Runtime.InteropServices.Marshal.PtrToStringUTF8(NativeApi.llama_print_system_info());

switch (mode)
{
    case "parity":
        Parity();
        break;
    case "semantics":
        await Semantics();
        break;
    case "bench":
        Bench();
        break;
    default:
        throw new ArgumentException($"unknown mode {mode}");
}
lock (nativeLog)
{
    env["native_log_warnings"] = new JsonArray(nativeLog.Distinct().Take(60).Select(s => (JsonNode?)s).ToArray());
}
File.WriteAllText(Path.Combine(outDir, mode == "parity" ? "environment.json" : $"environment-{mode}.json"),
    env.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
return 0;

// ---------------------------------------------------------------- parity

void Parity()
{
    var cases = Shared.LoadCases(opts["prompts"], opts["reference"]);
    using var results = new StreamWriter(Path.Combine(outDir, "results.jsonl"), false, new UTF8Encoding(false)) { NewLine = "\n", AutoFlush = true };
    var tokenizerEqual = 0;
    var tokenizerRows = new JsonArray();
    foreach (var c in cases)
    {
        var sw = Stopwatch.StartNew();
        var (output, finish, step0) = Greedy(context, c.RefPromptIds, c.MaxNewTokens, LLamaSeqId.Zero, clear: true);
        sw.Stop();
        Shared.WriteStep0(outDir, c.Id, Shared.LogSoftmax(step0));

        // Template and tokenizer boundary.
        string? rendered = null;
        int[]? backendIds = null;
        string? templateError = null;
        try
        {
            var tpl = new LLamaTemplate(weights) { AddAssistant = true };
            foreach (var m in c.Messages)
            {
                tpl.Add((string)m!["role"]!, (string)m["content"]!);
            }
            rendered = Encoding.UTF8.GetString(tpl.Apply());
            backendIds = context.Tokenize(rendered, addBos: true, special: true).Select(t => (int)t).ToArray();
        }
        catch (Exception e)
        {
            templateError = $"{e.GetType().Name}: {e.Message}";
        }
        var refTok = context.Tokenize(c.RefRendered, addBos: true, special: true).Select(t => (int)t).ToArray();
        var eq = refTok.SequenceEqual(c.RefPromptIds);
        if (eq)
        {
            tokenizerEqual++;
        }
        tokenizerRows.Add(new JsonObject { ["id"] = c.Id, ["equal"] = eq, ["len_ref"] = c.RefPromptIds.Length, ["len_llama"] = refTok.Length });

        var row = new JsonObject
        {
            ["id"] = c.Id,
            ["backend"] = $"llamasharp-{device}-{Path.GetFileNameWithoutExtension(opts["model"])}",
            ["prompt_ids"] = Shared.ToJson(c.RefPromptIds),
            ["output_ids"] = Shared.ToJson(output),
            ["finish_reason"] = finish,
            ["rendered"] = rendered,
            ["backend_prompt_ids"] = backendIds is null ? null : Shared.ToJson(backendIds),
            ["template_error"] = templateError,
            ["template_inputs_ignored"] = (c.Tools is not null ? "tools " : "") + (c.TemplateKwargs is not null ? "chat_template_kwargs" : ""),
            ["llama_tokenize_reference_rendered_equal"] = eq,
            ["output_text"] = context.DeTokenize(output.Select(i => (LLamaToken)i).ToList()),
            ["elapsed_s"] = sw.Elapsed.TotalSeconds,
        };
        results.WriteLine(row.ToJsonString(Shared.Compact));
        Console.WriteLine($"{c.Id,-20} gen {output.Count,4} {finish,-6} {sw.Elapsed.TotalSeconds:F2}s tokenizer-equal {eq}");
    }
    env["tokenizer_parity_on_reference_rendering"] = $"{tokenizerEqual}/{cases.Count}";
    env["tokenizer_rows"] = tokenizerRows;
}

// Greedy loop on raw logits. Prompt is decoded in chunks of BatchSize with logits only for the last token.
(List<int> Output, string Finish, float[] Step0) Greedy(LLamaContext ctx, int[] prompt, int maxNew, LLamaSeqId seq, bool clear, int startPos = 0)
{
    if (clear)
    {
        ctx.NativeHandle.MemoryClear();
    }
    var batch = new LLamaBatch();
    var pos = startPos;
    var logitIndex = Prefill(ctx, batch, prompt, seq, ref pos);
    var output = new List<int>();
    float[] step0 = [];
    var finish = "length";
    for (var step = 0; step < maxNew; step++)
    {
        var logits = ctx.NativeHandle.GetLogitsIth(logitIndex);
        if (step == 0)
        {
            step0 = logits.ToArray();
        }
        var next = Shared.ArgMax(logits);
        output.Add(next);
        if (((LLamaToken)next).IsEndOfGeneration(vocab))
        {
            finish = "eos";
            break;
        }
        if (step == maxNew - 1)
        {
            break;
        }
        batch.Clear();
        logitIndex = batch.Add((LLamaToken)next, pos++, seq, true);
        Check(ctx.NativeHandle.Decode(batch));
    }
    return (output, finish, step0);
}

int Prefill(LLamaContext ctx, LLamaBatch batch, int[] prompt, LLamaSeqId seq, ref int pos)
{
    var chunk = (int)ctx.BatchSize;
    var logitIndex = -1;
    for (var i = 0; i < prompt.Length; i += chunk)
    {
        batch.Clear();
        var end = Math.Min(prompt.Length, i + chunk);
        for (var j = i; j < end; j++)
        {
            var idx = batch.Add((LLamaToken)prompt[j], pos++, seq, j == prompt.Length - 1);
            if (j == prompt.Length - 1)
            {
                logitIndex = idx;
            }
        }
        Check(ctx.NativeHandle.Decode(batch));
    }
    return logitIndex;
}

static void Check(DecodeResult r)
{
    if (r != DecodeResult.Ok)
    {
        throw new InvalidOperationException($"llama_decode returned {r}");
    }
}

// ---------------------------------------------------------------- semantics

async Task Semantics()
{
    var cases = Shared.LoadCases(opts["prompts"], opts["reference"]).ToDictionary(c => c.Id);
    var res = new JsonObject();

    // 1. Streaming: StreamingTokenDecoder fed one token at a time against a whole-sequence detokenization.
    var streaming = new JsonArray();
    foreach (var id in new[] { "zh", "emoji", "ja", "hi", "th", "code-py" })
    {
        var c = cases[id];
        var dec = new StreamingTokenDecoder(context);
        var sb = new StringBuilder();
        var ids = c.RefOutputIds.Where(t => !((LLamaToken)t).IsEndOfGeneration(vocab)).ToArray();
        foreach (var t in ids)
        {
            dec.Add(t);
            sb.Append(dec.Read());
        }
        var whole = context.DeTokenize(ids.Select(t => (LLamaToken)t).ToList());
        var bytes = new List<byte>();
        foreach (var t in ids)
        {
            var buf = new byte[256];
            var n = weights.NativeHandle.TokenToSpan((LLamaToken)t, buf);
            bytes.AddRange(buf.Take((int)n));
        }
        var utf8 = new UTF8Encoding(false, false).GetString(bytes.ToArray());
        streaming.Add(new JsonObject { ["id"] = id, ["streamed_equals_detokenize"] = sb.ToString() == whole, ["detokenize_equals_utf8_of_pieces"] = whole == utf8, ["chars"] = whole.Length });
    }
    res["streaming_decoder"] = streaming;

    // 1b. StatelessExecutor streaming with greedy sampling, prompt = reference rendering (template off).
    var ctxParams = mp;
    var exec = new StatelessExecutor(weights, ctxParams) { ApplyTemplate = false };
    {
        var c = cases["en-list"];
        var pieces = new List<string>();
        await foreach (var p in exec.InferAsync(c.RefRendered, new InferenceParams { MaxTokens = 64, SamplingPipeline = new GreedySamplingPipeline() }))
        {
            pieces.Add(p);
        }
        var refText = context.DeTokenize(c.RefOutputIds.Take(64).Where(t => !((LLamaToken)t).IsEndOfGeneration(vocab)).Select(t => (LLamaToken)t).ToList());
        res["executor_streaming"] = new JsonObject
        {
            ["id"] = c.Id,
            ["pieces"] = pieces.Count,
            ["concatenated_equals_reference_text_first_64"] = string.Concat(pieces) == refText,
            ["concatenated_prefix"] = string.Concat(pieces)[..Math.Min(80, string.Concat(pieces).Length)],
        };
    }

    // 2. Cancellation during decode: cancel after 10 yielded pieces.
    {
        var c = cases["en-list"];
        using var cts = new CancellationTokenSource();
        var n = 0;
        var swc = new Stopwatch();
        string outcome = "completed";
        try
        {
            await foreach (var _ in exec.InferAsync(c.RefRendered, new InferenceParams { MaxTokens = 128, SamplingPipeline = new GreedySamplingPipeline() }, cts.Token))
            {
                n++;
                if (n == 10)
                {
                    cts.Cancel();
                    swc.Start();
                }
            }
        }
        catch (OperationCanceledException e)
        {
            outcome = e.GetType().Name;
        }
        swc.Stop();
        res["cancel_during_decode"] = new JsonObject { ["pieces_before_cancel"] = 10, ["pieces_total"] = n, ["outcome"] = outcome, ["ms_cancel_to_end"] = swc.Elapsed.TotalMilliseconds };
    }
    // 2b. Cancellation during prefill: long prompt, token cancelled 20 ms after start.
    {
        var c = cases["long-license"];
        using var cts = new CancellationTokenSource();
        var sw = Stopwatch.StartNew();
        var n = 0;
        string outcome = "completed";
        double firstPieceMs = -1;
        cts.CancelAfter(20);
        try
        {
            await foreach (var _ in exec.InferAsync(c.RefRendered, new InferenceParams { MaxTokens = 128, SamplingPipeline = new GreedySamplingPipeline() }, cts.Token))
            {
                if (n++ == 0)
                {
                    firstPieceMs = sw.Elapsed.TotalMilliseconds;
                }
            }
        }
        catch (OperationCanceledException e)
        {
            outcome = e.GetType().Name;
        }
        res["cancel_during_prefill"] = new JsonObject { ["cancel_after_ms"] = 20, ["pieces"] = n, ["outcome"] = outcome, ["ms_to_end"] = sw.Elapsed.TotalMilliseconds, ["first_piece_ms"] = firstPieceMs };
    }

    // 3. Memory: partial removal on the hybrid (attention + recurrent) memory.
    var getMem = typeof(SafeLLamaContextHandle).GetMethod("llama_get_memory", BindingFlags.NonPublic | BindingFlags.Static)!;
    IntPtr Mem() => (IntPtr)getMem.Invoke(null, [context.NativeHandle])!;
    {
        var c = cases["en-math"];
        var (fresh, _, _) = Greedy(context, c.RefPromptIds, 24, LLamaSeqId.Zero, clear: true);
        // Prefill full prompt, then try to remove the last 5 positions and re-decode them.
        context.NativeHandle.MemoryClear();
        var b = new LLamaBatch();
        var pos = 0;
        Prefill(context, b, c.RefPromptIds, LLamaSeqId.Zero, ref pos);
        var cut = c.RefPromptIds.Length - 5;
        var partial = NativeApi.llama_memory_seq_rm(Mem(), LLamaSeqId.Zero, cut, -1);
        var maxPosAfter = context.NativeHandle.MemorySequenceMaxPosition(LLamaSeqId.Zero);
        var tail = new JsonObject { ["seq_rm_partial_returned"] = partial, ["max_pos_after"] = (int)maxPosAfter, ["memory_can_shift"] = context.NativeHandle.MemoryCanShift };
        if (partial)
        {
            var p2 = cut;
            var idx = Prefill(context, b, c.RefPromptIds[cut..], LLamaSeqId.Zero, ref p2);
            tail["next_token_after_redecode_equals_fresh"] = Shared.ArgMax(context.NativeHandle.GetLogitsIth(idx)) == fresh[0];
        }
        var whole = NativeApi.llama_memory_seq_rm(Mem(), LLamaSeqId.Zero, -1, -1);
        tail["seq_rm_whole_returned"] = whole;
        res["memory_seq_rm"] = tail;
    }

    // 4. Prefix reuse through state save/load of a sequence, and through seq_cp.
    {
        var c = cases["multi-turn"];
        var (fresh, _, _) = Greedy(context, c.RefPromptIds, 32, LLamaSeqId.Zero, clear: true);
        // Prefill all but the last token, snapshot, clear, restore, then continue with the last token.
        context.NativeHandle.MemoryClear();
        var b = new LLamaBatch();
        var pos = 0;
        var prefix = c.RefPromptIds[..^1];
        Prefill(context, b, prefix, LLamaSeqId.Zero, ref pos);
        var size = context.NativeHandle.GetStateSize(LLamaSeqId.Zero);
        var buf = new byte[size];
        nuint written;
        unsafe
        {
            fixed (byte* p = buf)
            {
                written = context.NativeHandle.GetState(p, size, LLamaSeqId.Zero);
            }
        }
        context.NativeHandle.MemoryClear();
        nuint read;
        unsafe
        {
            fixed (byte* p = buf)
            {
                read = context.NativeHandle.SetState(p, written, LLamaSeqId.Zero);
            }
        }
        var (restored, _, _) = Greedy(context, [c.RefPromptIds[^1]], 32, LLamaSeqId.Zero, clear: false, startPos: prefix.Length);
        res["seq_state_save_load"] = new JsonObject { ["state_bytes"] = (long)written, ["set_state_read"] = (long)read, ["continuation_equals_fresh"] = restored.SequenceEqual(fresh), ["fresh_len"] = fresh.Count };

        // seq_cp: prefill on seq 0, copy to seq 1, continue on seq 1 only.
        context.NativeHandle.MemoryClear();
        pos = 0;
        Prefill(context, b, prefix, LLamaSeqId.Zero, ref pos);
        string cpOutcome;
        try
        {
            context.NativeHandle.MemorySequenceCopy(LLamaSeqId.Zero, (LLamaSeqId)1, -1, -1);
            var (cp, _, _) = Greedy(context, [c.RefPromptIds[^1]], 32, (LLamaSeqId)1, clear: false, startPos: prefix.Length);
            cpOutcome = cp.SequenceEqual(fresh) ? "equal" : $"differs at {CommonPrefix(cp, fresh)}";
        }
        catch (Exception e)
        {
            cpOutcome = $"{e.GetType().Name}: {e.Message}";
        }
        res["seq_cp_prefix_reuse"] = cpOutcome;
    }

    // 5. Two conversations interleaved in one batch (seq 0 and seq 1): outputs must equal separate runs.
    {
        var a = cases["en-short"];
        var bcase = cases["zh"];
        var (fa, _, _) = Greedy(context, a.RefPromptIds, 32, LLamaSeqId.Zero, clear: true);
        var (fb, _, _) = Greedy(context, bcase.RefPromptIds, 32, LLamaSeqId.Zero, clear: true);
        context.NativeHandle.MemoryClear();
        var batch = new LLamaBatch();
        int ia = -1, ib = -1;
        for (var i = 0; i < a.RefPromptIds.Length; i++)
        {
            var k = batch.Add((LLamaToken)a.RefPromptIds[i], i, LLamaSeqId.Zero, i == a.RefPromptIds.Length - 1);
            if (i == a.RefPromptIds.Length - 1)
            {
                ia = k;
            }
        }
        for (var i = 0; i < bcase.RefPromptIds.Length; i++)
        {
            var k = batch.Add((LLamaToken)bcase.RefPromptIds[i], i, (LLamaSeqId)1, i == bcase.RefPromptIds.Length - 1);
            if (i == bcase.RefPromptIds.Length - 1)
            {
                ib = k;
            }
        }
        string outcome;
        try
        {
            Check(context.NativeHandle.Decode(batch));
            var oa = new List<int>();
            var ob = new List<int>();
            int pa = a.RefPromptIds.Length, pb = bcase.RefPromptIds.Length;
            bool doneA = false, doneB = false;
            for (var step = 0; step < 32 && !(doneA && doneB); step++)
            {
                batch.Clear();
                if (!doneA)
                {
                    var t = Shared.ArgMax(context.NativeHandle.GetLogitsIth(ia));
                    oa.Add(t);
                    doneA = ((LLamaToken)t).IsEndOfGeneration(vocab) || oa.Count == 32;
                    if (!doneA)
                    {
                        ia = batch.Add((LLamaToken)t, pa++, LLamaSeqId.Zero, true);
                    }
                }
                if (!doneB)
                {
                    var t = Shared.ArgMax(context.NativeHandle.GetLogitsIth(ib));
                    ob.Add(t);
                    doneB = ((LLamaToken)t).IsEndOfGeneration(vocab) || ob.Count == 32;
                    if (!doneB)
                    {
                        ib = batch.Add((LLamaToken)t, pb++, (LLamaSeqId)1, true);
                    }
                }
                if (batch.TokenCount > 0)
                {
                    Check(context.NativeHandle.Decode(batch));
                }
            }
            outcome = $"seq0 {(oa.SequenceEqual(fa) ? "equal" : $"differs at {CommonPrefix(oa, fa)}")}, seq1 {(ob.SequenceEqual(fb) ? "equal" : $"differs at {CommonPrefix(ob, fb)}")}";
        }
        catch (Exception e)
        {
            outcome = $"{e.GetType().Name}: {e.Message}";
        }
        res["two_sequences_one_batch"] = outcome;
    }

    File.WriteAllText(Path.Combine(outDir, "semantics.json"), res.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine(res.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
}

static int CommonPrefix(List<int> a, List<int> b)
{
    var n = 0;
    while (n < a.Count && n < b.Count && a[n] == b[n])
    {
        n++;
    }
    return n;
}

// ---------------------------------------------------------------- bench

void Bench()
{
    var cases = Shared.LoadCases(opts["prompts"], opts["reference"]).ToDictionary(c => c.Id);
    var samples = int.Parse(opts.GetValueOrDefault("samples", "5"));
    var res = new JsonObject { ["loadavg_start"] = File.ReadAllText("/proc/loadavg").Trim(), ["samples"] = samples, ["warmup"] = 1, ["decode_tokens"] = 128 };
    foreach (var id in new[] { "en-short", "long-license" })
    {
        var prompt = cases[id].RefPromptIds;
        var prefillMs = new List<double>();
        var ttftMs = new List<double>();
        var decodeTokS = new List<double>();
        for (var s = -1; s < samples; s++)
        {
            context.NativeHandle.MemoryClear();
            var b = new LLamaBatch();
            var pos = 0;
            var t0 = Stopwatch.GetTimestamp();
            var idx = Prefill(context, b, prompt, LLamaSeqId.Zero, ref pos);
            var logits = context.NativeHandle.GetLogitsIth(idx);
            var t1 = Stopwatch.GetTimestamp();
            var next = Shared.ArgMax(logits);
            var t2 = Stopwatch.GetTimestamp();
            // Fixed-length decode: 128 tokens, EOG ignored so every sample does the same work.
            for (var i = 0; i < 128; i++)
            {
                b.Clear();
                var k = b.Add((LLamaToken)next, pos++, LLamaSeqId.Zero, true);
                Check(context.NativeHandle.Decode(b));
                next = Shared.ArgMax(context.NativeHandle.GetLogitsIth(k));
            }
            var t3 = Stopwatch.GetTimestamp();
            if (s >= 0)
            {
                prefillMs.Add(Shared.Ms(t1 - t0));
                ttftMs.Add(Shared.Ms(t2 - t0));
                decodeTokS.Add(128 / (Shared.Ms(t3 - t2) / 1000));
            }
        }
        res[id] = new JsonObject
        {
            ["prompt_tokens"] = prompt.Length,
            ["prefill_ms"] = Shared.Stats(prefillMs),
            ["prefill_tok_s"] = Shared.Stats(prefillMs.Select(m => prompt.Length / (m / 1000)).ToList()),
            ["ttft_ms"] = Shared.Stats(ttftMs),
            ["decode_tok_s"] = Shared.Stats(decodeTokS),
        };
    }
    res["loadavg_end"] = File.ReadAllText("/proc/loadavg").Trim();
    File.WriteAllText(Path.Combine(outDir, "bench.json"), res.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine(res.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
}

// ---------------------------------------------------------------- helpers

JsonObject VocabInfo()
{
    var eog = new JsonArray();
    for (var t = 0; t < nVocab; t++)
    {
        if (((LLamaToken)t).IsEndOfGeneration(vocab))
        {
            eog.Add(t);
        }
    }
    string? Meta(string key)
    {
        var v = weights.NativeHandle.MetadataValueByKey(key);
        return v is null ? null : Encoding.UTF8.GetString(v.Value.Span).TrimEnd('\0');
    }
    return new JsonObject
    {
        ["type"] = vocab.Type.ToString(),
        ["bos"] = (int?)vocab.BOS,
        ["eos"] = (int?)vocab.EOS,
        ["eot"] = (int?)vocab.EOT,
        ["pad"] = (int?)vocab.Pad,
        ["should_add_bos"] = vocab.ShouldAddBOS,
        ["should_add_eos"] = vocab.ShouldAddEOS,
        ["eog_ids"] = eog,
        ["is_eog_248046_im_end"] = ((LLamaToken)248046).IsEndOfGeneration(vocab),
        ["is_eog_248044_endoftext"] = ((LLamaToken)248044).IsEndOfGeneration(vocab),
        ["gguf_tokenizer_pre"] = Meta("tokenizer.ggml.pre"),
        ["gguf_add_bos"] = Meta("tokenizer.ggml.add_bos_token"),
        ["gguf_chat_template_len"] = Meta("tokenizer.chat_template")?.Length,
        ["gguf_general_architecture"] = Meta("general.architecture"),
    };
}

static JsonArray LoadedNativeLibs()
{
    var arr = new JsonArray();
    foreach (var line in File.ReadAllLines("/proc/self/maps"))
    {
        var i = line.IndexOf('/');
        if (i < 0)
        {
            continue;
        }
        var path = line[i..];
        var name = Path.GetFileName(path);
        if ((name.StartsWith("libllama") || name.StartsWith("libggml") || name.StartsWith("libcudart") || name.StartsWith("libcublas")) && !arr.Any(n => (string)n!["path"]! == path))
        {
            arr.Add(new JsonObject { ["path"] = path, ["sha256"] = Sha256(path) });
        }
    }
    return arr;
}

static string Sha256(string path)
{
    using var f = File.OpenRead(path);
    return Convert.ToHexStringLower(SHA256.HashData(f));
}

static Dictionary<string, string> ParseArgs(string[] a)
{
    var d = new Dictionary<string, string>();
    for (var i = 0; i < a.Length; i += 2)
    {
        d[a[i].TrimStart('-')] = a[i + 1];
    }
    return d;
}
