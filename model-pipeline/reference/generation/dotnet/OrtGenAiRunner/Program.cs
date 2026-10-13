// M0 generation-backend check: ONNX Runtime GenAI from C#.
// Modes: parity (greedy on reference prompt ids, step-0 logits, template and tokenizer boundary),
// semantics (streaming, cancellation, rewind, prefix reuse, concurrent generators, IChatClient),
// bench (exploratory warm prefill/decode timings).
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using GenCheck;
using Microsoft.Extensions.AI;
using Microsoft.ML.OnnxRuntimeGenAI;

var opts = Args.Parse(args);
var cases = Shared.LoadCases(opts.Prompts, opts.Reference);
Directory.CreateDirectory(opts.Out);

using var config = new Config(opts.Model);
if (opts.Device == "cuda")
{
    config.ClearProviders();
    config.AppendProvider("cuda");
    foreach (var kv in opts.ProviderOptions)
    {
        var parts = kv.Split('=', 2);
        config.SetProviderOption("cuda", parts[0], parts[1]);
    }
}
else
{
    config.ClearProviders();
}
if (opts.Threads > 0)
{
    config.Overlay("{\"model\":{\"decoder\":{\"session_options\":{\"intra_op_num_threads\":" + opts.Threads + "}}}}");
}
foreach (var ov in opts.Overlays)
{
    config.Overlay(ov);
}
var loadSw = Stopwatch.StartNew();
using var model = new Model(config);
var loadMs = loadSw.Elapsed.TotalMilliseconds;
using var tokenizer = new Tokenizer(model);
var eos = tokenizer.GetEosTokenIds().ToArray();
var env = Shared.EnvironmentInfo();
env["backend"] = "onnxruntime-genai";
env["package"] = opts.Device == "cuda" ? "Microsoft.ML.OnnxRuntimeGenAI.Cuda 0.17.1" : "Microsoft.ML.OnnxRuntimeGenAI 0.17.1";
env["model_dir"] = opts.Model;
env["model_type"] = model.GetModelType();
env["device"] = opts.Device;
env["threads"] = opts.Threads;
env["overlays"] = new JsonArray(opts.Overlays.Select(x => (JsonNode?)JsonValue.Create(x)).ToArray());
env["provider_options"] = new JsonArray(opts.ProviderOptions.Select(x => (JsonNode?)JsonValue.Create(x)).ToArray());
env["model_load_ms"] = loadMs;
env["eos_token_ids"] = Shared.ToJson(eos);
env["bos_token_id"] = tokenizer.GetBosTokenId();
env["pad_token_id"] = tokenizer.GetPadTokenId();
env["files"] = new JsonObject(Directory.GetFiles(opts.Model).Where(f => !f.EndsWith(".data", StringComparison.Ordinal) || opts.HashData)
    .OrderBy(f => f, StringComparer.Ordinal)
    .Select(f => KeyValuePair.Create(Path.GetFileName(f), (JsonNode?)JsonValue.Create(Sha256(f)))));
env["native_libs"] = NativeLibs();

if (opts.Modes.Contains("parity"))
{
    Parity();
}
if (opts.Modes.Contains("semantics"))
{
    Semantics();
}
if (opts.Modes.Contains("bench"))
{
    Bench();
}
File.WriteAllText(Path.Combine(opts.Out, "environment.json"), env.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
return 0;

GeneratorParams NewParams(int maxLength, int minLength = 0)
{
    var p = new GeneratorParams(model);
    p.SetSearchOption("max_length", maxLength);
    p.SetSearchOption("do_sample", false);
    if (minLength > 0)
    {
        p.SetSearchOption("min_length", minLength);
    }
    return p;
}

float[] LastLogits(Generator g)
{
    using var t = g.GetOutput("logits");
    var shape = t.Shape();
    var vocab = (int)shape[^1];
    var total = (int)t.NumElements();
    var row = new float[vocab];
    if (t.Type() == ElementType.float32)
    {
        t.GetData<float>().Slice(total - vocab, vocab).CopyTo(row);
    }
    else if (t.Type() == ElementType.float16)
    {
        var src = t.GetData<Half>().Slice(total - vocab, vocab);
        for (var i = 0; i < vocab; i++)
        {
            row[i] = (float)src[i];
        }
    }
    else
    {
        throw new NotSupportedException($"logits type {t.Type()}");
    }
    return row;
}

static string MessagesJson(PromptCase c) => c.Messages.ToJsonString();

void Parity()
{
    var rows = new StringBuilder();
    var templateNotes = new JsonArray();
    foreach (var c in cases)
    {
        using var p = NewParams(c.RefPromptIds.Length + c.MaxNewTokens);
        using var g = new Generator(model, p);
        var sw = Stopwatch.StartNew();
        g.AppendTokens(c.RefPromptIds);
        var step0 = LastLogits(g);
        var lp0 = Shared.LogSoftmax(step0);
        Shared.WriteStep0(opts.Out, c.Id, lp0);
        var outIds = new List<int>();
        while (!g.IsDone() && outIds.Count < c.MaxNewTokens)
        {
            g.GenerateNextToken();
            outIds.Add(g.GetNextTokens()[0]);
            if (eos.Contains(outIds[^1]))
            {
                break;
            }
        }
        var elapsed = sw.Elapsed.TotalSeconds;
        var finish = outIds.Count > 0 && eos.Contains(outIds[^1]) ? "eos" : "length";

        // Template and tokenizer boundary.
        string? rendered = null;
        string? renderError = null;
        try
        {
            rendered = tokenizer.ApplyChatTemplate(null!, MessagesJson(c), c.Tools?.ToJsonString()!, true);
        }
        catch (Exception e)
        {
            renderError = e.GetType().Name + ": " + e.Message;
        }
        int[]? backendIds = null;
        if (rendered is not null)
        {
            using var s = tokenizer.Encode(rendered);
            backendIds = s[0].ToArray();
        }
        using var refEnc = tokenizer.Encode(c.RefRendered);
        var tokOnRefText = refEnc[0].ToArray();
        var fullDecode = tokenizer.Decode(outIds.ToArray());
        using var stream = tokenizer.CreateStream();
        var streamed = new StringBuilder();
        foreach (var id in outIds)
        {
            streamed.Append(stream.Decode(id));
        }

        var row = new JsonObject
        {
            ["id"] = c.Id,
            ["backend"] = $"ortgenai-{opts.Device}-{Path.GetFileName(opts.Model.TrimEnd('/'))}",
            ["prompt_ids"] = Shared.ToJson(c.RefPromptIds),
            ["output_ids"] = Shared.ToJson(outIds),
            ["finish_reason"] = finish,
            ["step0_argmax"] = Shared.ArgMax(step0),
            ["rendered"] = rendered,
            ["render_error"] = renderError,
            ["backend_prompt_ids"] = backendIds is null ? null : Shared.ToJson(backendIds),
            ["tokenizer_on_ref_text_equal"] = tokOnRefText.SequenceEqual(c.RefPromptIds),
            ["output_text"] = fullDecode,
            ["stream_equals_decode"] = streamed.ToString() == fullDecode,
            ["elapsed_s"] = elapsed,
        };
        rows.Append(row.ToJsonString(Shared.Compact)).Append('\n');
        Console.WriteLine($"{c.Id,-20} prompt {c.RefPromptIds.Length,5} gen {outIds.Count,4} {finish,-6} " +
            $"match {(outIds.SequenceEqual(c.RefOutputIds) ? "exact" : "DIFF")} render {(rendered == c.RefRendered ? "eq" : "NE")} " +
            $"tok {(tokOnRefText.SequenceEqual(c.RefPromptIds) ? "eq" : "NE")} {elapsed:F2}s");
    }
    File.WriteAllText(Path.Combine(opts.Out, "results.jsonl"), rows.ToString());
}

void Semantics()
{
    var res = new JsonObject();
    var c = cases.First(x => x.Id == "multi-turn");
    var shortCase = cases.First(x => x.Id == "en-list");

    // 1. Cancellation: the loop is caller-driven; cancel after 10 tokens and time the exit.
    {
        using var cts = new CancellationTokenSource();
        using var p = NewParams(shortCase.RefPromptIds.Length + 128);
        using var g = new Generator(model, p);
        g.AppendTokens(shortCase.RefPromptIds);
        var n = 0;
        long cancelAt = 0, exitAt = 0;
        try
        {
            while (!g.IsDone())
            {
                cts.Token.ThrowIfCancellationRequested();
                g.GenerateNextToken();
                n++;
                if (n == 10)
                {
                    cts.Cancel();
                    cancelAt = Stopwatch.GetTimestamp();
                }
            }
        }
        catch (OperationCanceledException)
        {
            exitAt = Stopwatch.GetTimestamp();
        }
        res["manual_loop_cancel"] = new JsonObject { ["tokens_before_cancel"] = n, ["cancel_to_exit_ms"] = Shared.Ms(exitAt - cancelAt) };
    }

    // 2. RewindTo on the hybrid (recurrent-state) model.
    {
        using var p = NewParams(shortCase.RefPromptIds.Length + 64);
        using var g = new Generator(model, p);
        g.AppendTokens(shortCase.RefPromptIds);
        for (var i = 0; i < 8; i++)
        {
            g.GenerateNextToken();
        }
        var first = g.GetSequence(0).ToArray();
        var r = new JsonObject();
        foreach (var target in new[] { (ulong)(shortCase.RefPromptIds.Length + 4), 0UL })
        {
            try
            {
                g.RewindTo(target);
                r[$"rewind_to_{target}"] = $"ok, token count {g.TokenCount()}";
                if (target > 0)
                {
                    for (var i = 0; i < 4; i++)
                    {
                        g.GenerateNextToken();
                    }
                    var again = g.GetSequence(0).ToArray();
                    r["regenerated_equal_after_partial_rewind"] = again.SequenceEqual(first);
                }
            }
            catch (Exception e)
            {
                r[$"rewind_to_{target}"] = e.GetType().Name + ": " + e.Message;
            }
        }
        res["rewind"] = r;
    }

    // 3. Prefix reuse: continue the same generator with the next turn's ids versus a fresh generator.
    {
        // Turn 1 is the first user message; the reference rendering of the full conversation gives turn 2.
        var full = c.RefPromptIds;
        using var enc1 = tokenizer.Encode("<|im_start|>user\nMy name is Ana and I like hiking.<|im_end|>\n<|im_start|>assistant\n");
        var turn1 = enc1[0].ToArray();
        var r = new JsonObject { ["turn1_is_prefix_of_full"] = full.Take(turn1.Length).SequenceEqual(turn1) };
        if (full.Take(turn1.Length).SequenceEqual(turn1))
        {
            using var p = NewParams(full.Length + 32);
            using var g = new Generator(model, p);
            g.AppendTokens(turn1);
            g.AppendTokens(full.AsSpan(turn1.Length));
            var a = new List<int>();
            for (var i = 0; i < 16 && !g.IsDone(); i++)
            {
                g.GenerateNextToken();
                a.Add(g.GetNextTokens()[0]);
            }
            using var p2 = NewParams(full.Length + 32);
            using var g2 = new Generator(model, p2);
            g2.AppendTokens(full);
            var b = new List<int>();
            for (var i = 0; i < 16 && !g2.IsDone(); i++)
            {
                g2.GenerateNextToken();
                b.Add(g2.GetNextTokens()[0]);
            }
            r["incremental_append_equals_full_prompt"] = a.SequenceEqual(b);
            r["first16_equal_reference"] = a.SequenceEqual(c.RefOutputIds.Take(a.Count));
        }
        res["prefix_reuse"] = r;
    }

    // 4. Two generators interleaved token by token (state isolation).
    {
        var c1 = cases.First(x => x.Id == "zh");
        var c2 = cases.First(x => x.Id == "code-py");
        using var p1 = NewParams(c1.RefPromptIds.Length + 32);
        using var p2 = NewParams(c2.RefPromptIds.Length + 32);
        using var g1 = new Generator(model, p1);
        using var g2 = new Generator(model, p2);
        g1.AppendTokens(c1.RefPromptIds);
        g2.AppendTokens(c2.RefPromptIds);
        var o1 = new List<int>();
        var o2 = new List<int>();
        for (var i = 0; i < 32; i++)
        {
            if (!g1.IsDone())
            {
                g1.GenerateNextToken();
                o1.Add(g1.GetNextTokens()[0]);
            }
            if (!g2.IsDone())
            {
                g2.GenerateNextToken();
                o2.Add(g2.GetNextTokens()[0]);
            }
        }
        res["interleaved_generators"] = new JsonObject
        {
            ["a_equals_reference_prefix"] = o1.SequenceEqual(c1.RefOutputIds.Take(o1.Count)),
            ["b_equals_reference_prefix"] = o2.SequenceEqual(c2.RefOutputIds.Take(o2.Count)),
            ["a_len"] = o1.Count,
            ["b_len"] = o2.Count,
        };
    }

    // 5. Concurrent generators on two threads sharing one Model.
    {
        var picks = new[] { "zh", "code-py", "en-math", "ja" }.Select(id => cases.First(x => x.Id == id)).ToArray();
        var outs = new List<int>[picks.Length];
        Parallel.For(0, picks.Length, new ParallelOptions { MaxDegreeOfParallelism = 2 }, i =>
        {
            using var p = NewParams(picks[i].RefPromptIds.Length + 32);
            using var g = new Generator(model, p);
            g.AppendTokens(picks[i].RefPromptIds);
            var o = new List<int>();
            for (var k = 0; k < 32 && !g.IsDone(); k++)
            {
                g.GenerateNextToken();
                o.Add(g.GetNextTokens()[0]);
            }
            outs[i] = o;
        });
        res["parallel_generators_equal_reference_prefix"] = new JsonArray(picks.Select((pc, i) =>
            (JsonNode?)JsonValue.Create(outs[i].SequenceEqual(pc.RefOutputIds.Take(outs[i].Count)))).ToArray());
    }

    // 6. Bundled Microsoft.Extensions.AI IChatClient: rendering, finish reason, usage, cancellation.
    {
        var r = new JsonObject();
        using var client = new OnnxRuntimeGenAIChatClient(model, ownsModel: false);
        var msgs = new List<ChatMessage> { new(ChatRole.User, "What is the capital of France? Answer in one sentence.") };
        var text = new StringBuilder();
        var pieces = new JsonArray();
        ChatFinishReason? finish = null;
        UsageDetails? usage = null;
        foreach (var u in client.GetStreamingResponseAsync(msgs, new ChatOptions { MaxOutputTokens = 64 }).ToBlockingEnumerable())
        {
            text.Append(u.Text);
            if (u.Text.Length > 0)
            {
                pieces.Add(u.Text);
            }
            finish ??= u.FinishReason;
            foreach (var uc in u.Contents.OfType<UsageContent>())
            {
                usage = uc.Details;
            }
        }
        var en = cases.First(x => x.Id == "en-short");
        using var refText = tokenizer.Encode(text.ToString());
        r["text"] = text.ToString();
        r["updates"] = pieces;
        r["finish_reason"] = finish?.Value;
        r["input_tokens"] = usage?.InputTokenCount;
        r["output_tokens"] = usage?.OutputTokenCount;
        r["reference_text"] = tokenizer.Decode(en.RefOutputIds.Where(t => !eos.Contains(t)).ToArray());
        using var cts = new CancellationTokenSource();
        var n = 0;
        long cancelAt = 0;
        try
        {
            foreach (var u in client.GetStreamingResponseAsync(new List<ChatMessage> { new(ChatRole.User, "Count from 1 to 100.") },
                new ChatOptions { MaxOutputTokens = 256 }, cts.Token).ToBlockingEnumerable())
            {
                if (++n == 5)
                {
                    cancelAt = Stopwatch.GetTimestamp();
                    cts.Cancel();
                }
            }
            r["cancel"] = $"not cancelled after {n} updates";
        }
        catch (OperationCanceledException e)
        {
            r["cancel"] = $"{e.GetType().Name} after {n} updates, {Shared.Ms(Stopwatch.GetTimestamp() - cancelAt):F1} ms after Cancel()";
        }
        res["ichatclient"] = r;
    }
    File.WriteAllText(Path.Combine(opts.Out, "semantics.json"), res.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine(res.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
}

void Bench()
{
    var res = new JsonObject { ["loadavg_start"] = File.ReadAllText("/proc/loadavg").Trim() };
    foreach (var id in new[] { "long-license", "en-list" })
    {
        var c = cases.First(x => x.Id == id);
        const int decodeTokens = 128;
        var prefillMs = new List<double>();
        var ttftMs = new List<double>();
        var decodeTps = new List<double>();
        for (var s = 0; s < opts.Samples + 1; s++)
        {
            using var p = NewParams(c.RefPromptIds.Length + decodeTokens, c.RefPromptIds.Length + decodeTokens);
            using var g = new Generator(model, p);
            var t0 = Stopwatch.GetTimestamp();
            g.AppendTokens(c.RefPromptIds);
            var t1 = Stopwatch.GetTimestamp();
            g.GenerateNextToken();
            _ = g.GetNextTokens()[0];
            var t2 = Stopwatch.GetTimestamp();
            var n = 1;
            while (!g.IsDone() && n < decodeTokens)
            {
                g.GenerateNextToken();
                _ = g.GetNextTokens()[0];
                n++;
            }
            var t3 = Stopwatch.GetTimestamp();
            if (s == 0)
            {
                continue; // warm-up
            }
            prefillMs.Add(Shared.Ms(t1 - t0));
            ttftMs.Add(Shared.Ms(t2 - t0));
            decodeTps.Add((n - 1) / (Shared.Ms(t3 - t2) / 1000.0));
        }
        res[id] = new JsonObject
        {
            ["prompt_tokens"] = c.RefPromptIds.Length,
            ["decode_tokens"] = decodeTokens,
            ["prefill_ms"] = Shared.Stats(prefillMs),
            ["prefill_tok_per_s"] = Shared.Stats(prefillMs.Select(m => c.RefPromptIds.Length / (m / 1000.0)).ToList()),
            ["ttft_ms"] = Shared.Stats(ttftMs),
            ["decode_tok_per_s"] = Shared.Stats(decodeTps),
        };
    }
    res["loadavg_end"] = File.ReadAllText("/proc/loadavg").Trim();
    File.WriteAllText(Path.Combine(opts.Out, "bench.json"), res.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine(res.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
}

static string Sha256(string path)
{
    using var f = File.OpenRead(path);
    return Convert.ToHexStringLower(SHA256.HashData(f));
}

static JsonObject NativeLibs()
{
    var o = new JsonObject();
    var dir = AppContext.BaseDirectory;
    foreach (var f in Directory.GetFiles(dir, "*.so*", SearchOption.AllDirectories).OrderBy(x => x, StringComparer.Ordinal))
    {
        o[Path.GetRelativePath(dir, f)] = Sha256(f);
    }
    return o;
}

internal sealed record Args(string Model, string Device, string Prompts, string Reference, string Out, int Threads, HashSet<string> Modes, int Samples, bool HashData, string[] ProviderOptions, string[] Overlays)
{
    public static Args Parse(string[] a)
    {
        string Get(string k, string? d = null)
        {
            var i = Array.IndexOf(a, k);
            return i >= 0 ? a[i + 1] : d ?? throw new ArgumentException($"missing {k}");
        }
        return new Args(Get("--model"), Get("--device", "cpu"), Get("--prompts"), Get("--reference"), Get("--out"),
            int.Parse(Get("--threads", "0")), Get("--modes", "parity").Split(',').ToHashSet(), int.Parse(Get("--samples", "5")),
            a.Contains("--hash-data"),
            a.Select((x, i) => (x, i)).Where(t => t.x == "--provider-option").Select(t => a[t.i + 1]).ToArray(),
            a.Select((x, i) => (x, i)).Where(t => t.x == "--overlay").Select(t => a[t.i + 1]).ToArray());
    }
}
