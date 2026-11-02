// Pilot LLM fidelity runner (P-016). Applies no thresholds; writes nothing under model-pipeline/protected/.
//
//   dotnet run --project ai-dotnet/tools/Starlings.Qwen35.Fidelity -c Release -- \
//     --bundle <snapshot dir> --variant fp32x|fp16x|int4 --device cpu|cuda \
//     --prompts model-pipeline/reference/generation/prompts-v1.jsonl --reference <dir> --out <dir> [--modes parity,semantics] [--threads N]
//
// --reference is a directory holding results.jsonl from run_reference.py or generation.jsonl (the proposed fixture).
// parity:    per prompt, the owned template rendering and owned-tokenizer ids (compared with the reference rendering
//            and ids), then greedy generation from the reference prompt ids through Qwen35ChatModel's decode loop,
//            streamed: results.jsonl (compare.py format: output ids including the stop token, finish reason) and
//            step0/<id>.f32 (log-softmax after the prefill), plus streaming reassembly and usage checks per prompt.
// semantics: max-token boundary, stop sequences, additional stop ids, cached vs uncached logits, cancellation and
//            reuse, concurrent and interleaved requests, seeded sampling, the text path with enable_thinking:
//            semantics.json. Development prompts only (the checks pick prompts by id).
// fixture-semantics: the three checks of `semantics` that feed G2 (cached vs uncached, chunked prefill, concurrency) on
//            any generation fixture, the cases chosen from the fixture by a fixed rule (FixtureSemantics below), for
//            example the release prompts: semantics.json, with a `selection` section naming the rule and the cases.
// longcontext: per prompt, greedy decoding on the native sequence with chunked prefill (--prefill-chunk, default 512),
//            the chosen-token log-probability at every step against the reference's (transformers' cached decoding),
//            and cached vs uncached logits at a few steps (a fresh chunked prefill of the same prefix): longcontext.json.
// Score with: python model-pipeline/reference/generation/compare.py --reference <ref> --candidate <out>.
// semantics, stress and rawstress use fixed prompts of prompts-v1.jsonl (Cases.ModePrompts); with a prompt set that
// lacks them (for example a release prompt set) the tool stops before loading the model, with exit code 2.
// fixture-semantics needs a generation fixture (step_margin per case), at least 6 cases, the longest prompt above 64
// tokens and the second longest above 7; otherwise, and with semantics in the same run, the tool stops with exit 2.
// Exit codes: 0 done; 2 usage or input error; 70 an exception escaped (ai-dotnet/tools/Shared/ToolMain.cs).
//
// Task quality (G3): `-- generation --bundle <dir> --prompts <jsonl> --out <dir>` runs the chat path (owned template,
// greedy) over a prompts file with no reference; see GenerationDump.cs (input of `model-pipeline evaluate --g3
// generation`).
using System.Collections.Concurrent;
using System.Collections.Immutable;
using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Starlings.Acquisition;
using Starlings.Generation;
using Starlings.Generation.OnnxGenAI;
using Starlings.Models.Qwen35;

return ToolMain.Run(ToolBody);

int ToolBody()
{
    if (args.Length >= 1 && args[0] == "generation")
    {
        return GenerationDump.Run(args[1..]);
    }

    var opt = Options.Parse(args);
    if (opt is null)
    {
        Console.Error.WriteLine("usage: --bundle <dir> --variant fp32x|fp16x|int4 --device cpu|cuda --prompts <jsonl> --reference <dir> --out <dir> [--modes parity,semantics] [--threads N]");
        return 2;
    }

    Directory.CreateDirectory(opt.Out);
    var cases = Cases.Load(opt.Prompts, opt.Reference);
    if (Cases.MissingForModes(cases, opt.Modes) is { } missingForModes)
    {
        // Checked before the model loads: semantics and stress run on fixed prompts of prompts-v1.jsonl.
        Console.Error.WriteLine($"error: {missingForModes} are not in {opt.Prompts}; use --prompts model-pipeline/reference/generation/prompts-v1.jsonl for these modes, or --modes parity");
        return 2;
    }

    if (opt.Modes.Contains("fixture-semantics") && (opt.Modes.Contains("semantics")
        ? "--modes semantics and fixture-semantics both write semantics.json; pick one"
        : Cases.FixtureSemanticsProblem(cases)) is { } fixtureProblem)
    {
        Console.Error.WriteLine($"error: {fixtureProblem}");
        return 2;
    }

    if (opt.Modes.SetEquals(["rawstress"]))
    {
        // The plain GenAI reproducer loads its own model; nothing else may hold the weights in this process.
        Directory.CreateDirectory(opt.Out);
        RawStress();
        return 0;
    }

    var clock = Stopwatch.StartNew();
    var files = Qwen35ModelAssets.OpenDirectory(opt.Bundle, opt.Variant, CacheVerification.Size);
    using var model = Qwen35ChatModel.Load(files, new Qwen35ModelOptions
    {
        Provider = opt.Device == "cuda" ? GenAIExecutionProvider.Cuda : GenAIExecutionProvider.Cpu,
        IntraOpThreads = opt.Threads > 0 ? opt.Threads : null,
        SerializeNativeCalls = opt.Serialize,
    });
    double loadMs = clock.Elapsed.TotalMilliseconds;
    string backend = $"df-qwen35-{opt.Device}-{Qwen35ModelAssets.VariantDirectory(opt.Variant)}";
    var env = new JsonObject
    {
        ["runner"] = "ai-dotnet/tools/Starlings.Qwen35.Fidelity",
        ["backend"] = backend,
        ["profile"] = model.Info.Profile,
        ["checkpoint"] = model.Info.Checkpoint,
        ["genai_version"] = model.Info.GenAIVersion,
        ["tokenizer"] = model.Info.Tokenizer,
        ["chat_template"] = model.Info.ChatTemplate,
        ["config_overlays"] = new JsonArray(model.Info.ConfigOverlays.Select(o => (JsonNode?)o).ToArray()),
        ["provider_options"] = new JsonObject(model.Info.ProviderOptions.Select(kv => KeyValuePair.Create(kv.Key, (JsonNode?)kv.Value))),
        ["stop_token_ids"] = new JsonArray(model.Info.StopTokenIds.Select(i => (JsonNode?)i).ToArray()),
        ["threads"] = opt.Threads,
        ["prefill_chunk_tokens"] = opt.PrefillChunk,
        ["serializes_native_calls"] = model.Info.SerializesNativeCalls,
        ["model_load_ms"] = loadMs,
        ["prompts_sha256"] = Sha256(opt.Prompts),
        ["reference"] = Path.GetFullPath(opt.Reference),
        ["dotnet"] = Environment.Version.ToString(),
        ["os"] = System.Runtime.InteropServices.RuntimeInformation.OSDescription,
        ["cpu_count"] = Environment.ProcessorCount,
        ["loadavg_start"] = File.Exists("/proc/loadavg") ? File.ReadAllText("/proc/loadavg").Trim() : null,
        ["cuda_visible_devices"] = Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES"),
        ["source_revision"] = Environment.GetEnvironmentVariable("DF_SOURCE_REVISION"),
    };

    if (opt.Modes.Contains("parity"))
    {
        Parity();
    }

    if (opt.Modes.Contains("semantics"))
    {
        Semantics();
    }

    if (opt.Modes.Contains("fixture-semantics"))
    {
        FixtureSemantics();
    }

    if (opt.Modes.Contains("stress"))
    {
        Stress();
    }

    if (opt.Modes.Contains("longcontext"))
    {
        LongContext();
    }


    env["loadavg_end"] = File.Exists("/proc/loadavg") ? File.ReadAllText("/proc/loadavg").Trim() : null;
    File.WriteAllText(Path.Combine(opt.Out, "environment.json"), env.ToJsonString(Json.Indented) + "\n");
    return 0;

    void Parity()
    {
        Directory.CreateDirectory(Path.Combine(opt.Out, "step0"));
        var rows = new StringBuilder();
        foreach (var c in cases)
        {
            string? rendered = null;
            string? renderError = null;
            int[]? ownIds = null;
            try
            {
                var prompt = model.CreatePrompt(c.Messages, c.EnableThinking);
                rendered = prompt.Text;
                ownIds = [.. prompt.TokenIds];
            }
            catch (Exception e) when (e is Qwen35TemplateException or NotSupportedException)
            {
                renderError = e.GetType().Name + ": " + e.Message;
            }

            float[]? step0 = null;
            var text = new StringBuilder();
            var streamedIds = new List<int>();
            int updates = 0;
            GenerationResult? result = null;
            foreach (var u in model.Generator.GenerateStreamingCore(c.RefPromptIds, new GenerationOptions { MaxNewTokens = c.MaxNewTokens, PrefillChunkTokens = opt.PrefillChunk }, l => step0 = l, CancellationToken.None))
            {
                updates++;
                text.Append(u.Text);
                streamedIds.AddRange(u.TokenIds);
                result ??= u.Result;
            }

            var lp0 = LogSoftmax(step0!);
            WriteF32(Path.Combine(opt.Out, "step0", c.Id + ".f32"), lp0);
            int[] outputIds = result!.StopTokenId is int stop ? [.. result.TokenIds, stop] : [.. result.TokenIds];
            string referenceText = model.Tokenizer.Decode(c.RefOutputIds.Where(i => !model.Info.StopTokenIds.Contains(i)).ToArray(), false);
            var row = new JsonObject
            {
                ["id"] = c.Id,
                ["backend"] = backend,
                ["prompt_ids"] = Ints(c.RefPromptIds),
                ["output_ids"] = Ints(outputIds),
                ["finish_reason"] = result.FinishReason == GenerationFinishReason.Stop ? "eos" : "length",
                ["finish_reason_api"] = result.FinishReason.ToString(),
                ["step0_argmax"] = ArgMax(lp0),
                ["rendered"] = rendered,
                ["render_error"] = renderError,
                ["backend_prompt_ids"] = ownIds is null ? null : Ints(ownIds),
                ["output_text"] = result.Text,
                ["stream_updates"] = updates,
                ["stream_text_equals_result"] = text.ToString() == result.Text,
                ["stream_ids_equal_result"] = streamedIds.SequenceEqual(result.TokenIds),
                ["stream_text_equals_decode"] = text.ToString() == model.Tokenizer.Decode(result.TokenIds.AsSpan(), false),
                ["text_equals_reference_text"] = result.Text == referenceText,
                ["usage_prompt_tokens"] = result.Usage.PromptTokens,
                ["usage_completion_tokens"] = result.Usage.CompletionTokens,
                ["usage_consistent"] = result.Usage.PromptTokens == c.RefPromptIds.Length && result.Usage.CompletionTokens == result.TokenIds.Length,
                ["prefill_ms"] = result.Timings.Prefill.TotalMilliseconds,
                ["decode_ms"] = result.Timings.Decode.TotalMilliseconds,
            };
            rows.Append(row.ToJsonString(Json.Compact)).Append('\n');
            bool exact = outputIds.SequenceEqual(c.RefOutputIds);
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"{c.Id,-20} prompt {c.RefPromptIds.Length,5} gen {outputIds.Length,4} {row["finish_reason"],-6} {(exact ? "exact" : "DIFF ")} " +
                $"render {(rendered is null ? "unsupported" : rendered == c.RefRendered ? "eq" : "NE")} " +
                $"ids {(ownIds is null ? "-" : ownIds.SequenceEqual(c.RefPromptIds) ? "eq" : "NE")} stream {((bool)row["stream_text_equals_decode"]! ? "ok" : "BAD")} " +
                $"{result.Timings.Prefill.TotalSeconds:F2}s+{result.Timings.Decode.TotalSeconds:F2}s"));
        }

        File.WriteAllText(Path.Combine(opt.Out, "results.jsonl"), rows.ToString());
    }

    void Semantics()
    {
        var res = new JsonObject();
        Case Get(string id) => cases.First(c => c.Id == id);
        GenerationResult Run(Case c, GenerationOptions o, CancellationToken ct = default) => model.Generator.Generate(c.RefPromptIds, o, ct);
        bool PrefixOfRef(Case c, ImmutableArray<int> ids) => ids.SequenceEqual(c.RefOutputIds.Take(ids.Length));

        // 1. Max-token boundary on a prompt that ends with a stop token: n content tokens then <|im_end|>.
        {
            var c = Get("en-short");
            int content = c.RefOutputIds.Length - 1;
            var arr = new JsonArray();
            foreach (int max in new[] { 1, content - 1, content, content + 1 })
            {
                var r = Run(c, new GenerationOptions { MaxNewTokens = max });
                arr.Add(new JsonObject
                {
                    ["max_new_tokens"] = max,
                    ["completion_tokens"] = r.Usage.CompletionTokens,
                    ["finish"] = r.FinishReason.ToString(),
                    ["ids_prefix_of_reference"] = PrefixOfRef(c, r.TokenIds),
                    ["expected_finish"] = max > content ? "Stop" : "Length",
                    ["ok"] = PrefixOfRef(c, r.TokenIds) && r.Usage.CompletionTokens == Math.Min(max, content) &&
                        r.FinishReason.ToString() == (max > content ? "Stop" : "Length"),
                });
            }

            res["max_token_boundary"] = new JsonObject { ["prompt"] = c.Id, ["reference_content_tokens"] = content, ["runs"] = arr };
        }

        // 2. Stop sequences: a multi-token substring of the reference text, and a string that never occurs.
        {
            var c = Get("en-short");
            string full = model.Tokenizer.Decode(c.RefOutputIds.AsSpan(0, c.RefOutputIds.Length - 1), false);
            string stop = model.Tokenizer.Decode(c.RefOutputIds.AsSpan(2, 2), false);
            var pieces = new List<string>();
            GenerationResult? r = null;
            foreach (var u in model.Generator.GenerateStreaming(c.RefPromptIds, new GenerationOptions { MaxNewTokens = 64, StopSequences = [stop] }))
            {
                pieces.Add(u.Text);
                r ??= u.Result;
            }

            var none = Run(c, new GenerationOptions { MaxNewTokens = 64, StopSequences = ["\u0001never\u0001"] });
            var plain = Run(c, new GenerationOptions { MaxNewTokens = 64 });
            res["stop_sequences"] = new JsonObject
            {
                ["prompt"] = c.Id,
                ["stop"] = stop,
                ["text"] = r!.Text,
                ["finish"] = r.FinishReason.ToString(),
                ["text_is_reference_before_stop"] = r.Text == full[..full.IndexOf(stop, StringComparison.Ordinal)],
                ["no_piece_contains_stop"] = pieces.All(p => !p.Contains(stop, StringComparison.Ordinal)),
                ["stream_concat_equals_text"] = string.Concat(pieces) == r.Text,
                ["completion_tokens"] = r.Usage.CompletionTokens,
                ["absent_stop_equals_plain"] = none.Text == plain.Text && none.TokenIds.SequenceEqual(plain.TokenIds) && none.FinishReason == plain.FinishReason,
            };
        }

        // 3. Additional stop token id: the reference's 4th generated id ends generation before it.
        {
            var c = Get("en-list");
            int stopId = c.RefOutputIds[3];
            int first = Array.IndexOf(c.RefOutputIds, stopId);
            var r = Run(c, new GenerationOptions { MaxNewTokens = 64, AdditionalStopTokenIds = [stopId] });
            res["additional_stop_token"] = new JsonObject
            {
                ["prompt"] = c.Id,
                ["stop_id"] = stopId,
                ["finish"] = r.FinishReason.ToString(),
                ["stop_token_id"] = r.StopTokenId,
                ["completion_tokens"] = r.Usage.CompletionTokens,
                ["ok"] = r.FinishReason == GenerationFinishReason.Stop && r.StopTokenId == stopId && r.TokenIds.SequenceEqual(c.RefOutputIds[..first]),
            };
        }

        // 4. Cached vs uncached: logits during cached decoding against a fresh prefill of the same prefix.
        res["cached_vs_uncached"] = CachedVsUncached(Get("en-list"));

        // 5. Cancellation between decode steps, then reuse of the model.
        {
            var c = Get("en-list");
            using var cts = new CancellationTokenSource();
            int seen = 0;
            long cancelAt = 0;
            string outcome;
            try
            {
                foreach (var u in model.Generator.GenerateStreaming(c.RefPromptIds, new GenerationOptions { MaxNewTokens = 128 }, cts.Token))
                {
                    if (++seen == 10)
                    {
                        cancelAt = Stopwatch.GetTimestamp();
                        cts.Cancel();
                    }
                }

                outcome = $"not cancelled after {seen} updates";
            }
            catch (OperationCanceledException e)
            {
                outcome = $"{e.GetType().Name} after {seen} updates";
            }

            double latencyMs = cancelAt == 0 ? double.NaN : Stopwatch.GetElapsedTime(cancelAt).TotalMilliseconds;
            using var pre = new CancellationTokenSource();
            pre.Cancel();
            string preOutcome;
            try
            {
                Run(c, new GenerationOptions { MaxNewTokens = 8 }, pre.Token);
                preOutcome = "not cancelled";
            }
            catch (OperationCanceledException)
            {
                preOutcome = "OperationCanceledException before prefill";
            }

            var reuse = Run(Get("en-short"), new GenerationOptions { MaxNewTokens = 64 });
            res["cancellation"] = new JsonObject
            {
                ["outcome"] = outcome,
                ["cancel_to_exception_ms"] = latencyMs,
                ["already_cancelled_token"] = preOutcome,
                ["reuse_after_cancel_equals_reference"] = reuse.TokenIds.Append(reuse.StopTokenId ?? -1).SequenceEqual(Get("en-short").RefOutputIds),
            };
        }

        // 6. Concurrent requests on one model (two threads) and two interleaved streams.
        res["concurrency"] = Concurrency(["zh", "code-py", "en-math", "ja"], Get("zh"), Get("code-cs"));

        // 7. Seeded sampling: reproducible with the same seed; top_k = 1 equals greedy.
        {
            var c = Get("en-list");
            var s1 = new SamplingOptions { Seed = 7, Temperature = 0.7, TopK = 20, TopP = 0.8 };
            var r1 = Run(c, new GenerationOptions { MaxNewTokens = 32, Sampling = s1 });
            var r2 = Run(c, new GenerationOptions { MaxNewTokens = 32, Sampling = s1 });
            var r3 = Run(c, new GenerationOptions { MaxNewTokens = 32, Sampling = s1 with { Seed = 8 } });
            var k1 = Run(c, new GenerationOptions { MaxNewTokens = 32, Sampling = new SamplingOptions { Seed = 1, TopK = 1 } });
            res["sampling"] = new JsonObject
            {
                ["same_seed_identical"] = r1.TokenIds.SequenceEqual(r2.TokenIds),
                ["other_seed_differs"] = !r1.TokenIds.SequenceEqual(r3.TokenIds),
                ["top_k_1_equals_greedy_reference_prefix"] = PrefixOfRef(c, k1.TokenIds) && k1.TokenIds.Length == 32,
                ["sample_text_seed7"] = r1.Text,
            };
        }

        // 8. Text path: messages through the owned template and tokenizer, enable_thinking on, and a multi-turn prompt.
        {
            var o = new JsonObject();
            foreach (string id in new[] { "thinking-on", "multi-turn", "special-endoftext" })
            {
                var c = Get(id);
                var r = model.Generate(c.Messages, new Qwen35ChatOptions { MaxNewTokens = 24, EnableThinking = c.EnableThinking });
                o[id] = PrefixOfRef(c, r.TokenIds) && r.Usage.PromptTokens == c.RefPromptIds.Length;
            }

            res["text_path_prefix_equal"] = o;
        }

        // 9. Chunked prefill against a one-call prefill: step-0 logits and the first 32 tokens.
        res["chunked_prefill"] = ChunkedPrefill([("long-license", 128), ("long-license", 100), ("multi-turn", 7)]);

        File.WriteAllText(Path.Combine(opt.Out, "semantics.json"), res.ToJsonString(Json.Indented) + "\n");
        Console.WriteLine(res.ToJsonString(Json.Indented));
    }

    // Cached vs uncached: logits during cached decoding (12 steps) against a fresh prefill of the same prefix.
    JsonObject CachedVsUncached(Case cachedCase)
    {
        var c = cachedCase;
        const int steps = 12;
        var gen = new List<int>();
        var cached = new List<float[]>();
        using (var seq = model.Backend.CreateSequence(new GenAISequenceOptions { MaxLength = c.RefPromptIds.Length + steps + 1 }))
        {
            seq.AppendTokens(c.RefPromptIds);
            var l0 = new float[seq.VocabularySize];
            seq.CopyLastLogits(l0);
            cached.Add(l0);
            for (int k = 0; k < steps; k++)
            {
                gen.Add(seq.GenerateNextToken());
                var l = new float[seq.VocabularySize];
                seq.CopyLastLogits(l);
                cached.Add(l);
            }
        }

        // After GenerateNextToken GenAI reports either the logits that selected the token or those of the next
        // position, depending on when it runs the model; the matching uncached prefix is found by argmax.
        var perStep = new JsonArray();
        double worst = 0;
        int tokensEqual = 0;
        for (int k = 1; k <= steps; k++)
        {
            int[] prefixA = [.. c.RefPromptIds, .. gen.Take(k)];
            int[] prefixB = [.. c.RefPromptIds, .. gen.Take(k - 1)];
            var ua = Prefill(prefixA);
            var ub = Prefill(prefixB);
            var lc = LogSoftmax(cached[k]);
            double da = MaxAbs(lc, LogSoftmax(ua));
            double db = MaxAbs(lc, LogSoftmax(ub));
            bool aligned = da <= db;
            double d = Math.Min(da, db);
            worst = Math.Max(worst, d);
            if (gen[k - 1] == ArgMax(LogSoftmax(Prefill(prefixB))))
            {
                tokensEqual++;
            }

            perStep.Add(new JsonObject { ["step"] = k, ["max_abs_logprob"] = d, ["aligned_with"] = aligned ? "prefix incl. token k" : "prefix before token k" });
        }

        return new JsonObject
        {
            ["prompt"] = c.Id,
            ["steps"] = steps,
            ["tokens_equal_uncached_argmax"] = tokensEqual,
            ["max_abs_logprob"] = worst,
            ["generated_equals_reference_prefix"] = gen.SequenceEqual(c.RefOutputIds.Take(steps)),
            ["per_step"] = perStep,
        };
    }

    // Concurrent requests on one model (two threads, 24 tokens, prefix of the reference) and two interleaved streams.
    JsonObject Concurrency(string[] parallelIds, Case interleavedA, Case interleavedB)
    {
        Case Get(string id) => cases.First(c => c.Id == id);
        bool PrefixOfRef(Case c, ImmutableArray<int> ids) => ids.SequenceEqual(c.RefOutputIds.Take(ids.Length));
        GenerationResult Run(Case c, GenerationOptions o) => model.Generator.Generate(c.RefPromptIds, o, default);
        string[] ids = parallelIds;
        var outs = new ConcurrentDictionary<string, bool>();
        Parallel.ForEach(ids, new ParallelOptions { MaxDegreeOfParallelism = 2 }, id =>
        {
            var c = Get(id);
            var r = Run(c, new GenerationOptions { MaxNewTokens = 24 });
            outs[id] = PrefixOfRef(c, r.TokenIds);
        });

        var a = interleavedA;
        var b = interleavedB;
        using var ea = model.Generator.GenerateStreaming(a.RefPromptIds, new GenerationOptions { MaxNewTokens = 24 }).GetEnumerator();
        using var eb = model.Generator.GenerateStreaming(b.RefPromptIds, new GenerationOptions { MaxNewTokens = 24 }).GetEnumerator();
        var ia = new List<int>();
        var ib = new List<int>();
        bool ma = true, mb = true;
        while (ma || mb)
        {
            if (ma && (ma = ea.MoveNext()))
            {
                ia.AddRange(ea.Current.TokenIds);
            }

            if (mb && (mb = eb.MoveNext()))
            {
                ib.AddRange(eb.Current.TokenIds);
            }
        }

        return new JsonObject
        {
            ["parallel_two_threads_prefix_equal"] = new JsonObject(outs.Select(kv => KeyValuePair.Create(kv.Key, (JsonNode?)kv.Value))),
            ["interleaved_streams_prefix_equal"] = new JsonObject
            {
                [a.Id] = ia.SequenceEqual(a.RefOutputIds.Take(ia.Count)) && ia.Count > 0,
                [b.Id] = ib.SequenceEqual(b.RefOutputIds.Take(ib.Count)) && ib.Count > 0,
            },
        };
    }

    // Chunked prefill against a one-call prefill: step-0 logits and the first 32 tokens, per (case, chunk size).
    JsonObject ChunkedPrefill((string Id, int Chunk)[] runs)
    {
        Case Get(string id) => cases.First(c => c.Id == id);
        bool PrefixOfRef(Case c, ImmutableArray<int> ids) => ids.SequenceEqual(c.RefOutputIds.Take(ids.Length));
        var o = new JsonObject();
        foreach (var (id, chunk) in runs)
        {
            var c = Get(id);
            float[]? whole = null, chunked = null;
            GenerationResult? a = null;
            string? wholeError = null;
            try
            {
                a = model.Generator.GenerateStreamingCore(c.RefPromptIds, new GenerationOptions { MaxNewTokens = 32 }, l => whole = l, CancellationToken.None).Last().Result!;
            }
            catch (GenAIModelException e)
            {
                // A one-call prefill can exhaust device memory (logits for every prompt position).
                wholeError = e.Message.Split('\n')[0];
            }

            var b = model.Generator.GenerateStreamingCore(c.RefPromptIds, new GenerationOptions { MaxNewTokens = 32, PrefillChunkTokens = chunk }, l => chunked = l, CancellationToken.None).Last().Result!;
            o[$"{id}/{chunk}"] = new JsonObject
            {
                ["whole_prefill_error"] = wholeError,
                ["tokens_equal_whole"] = a is null ? null : a.TokenIds.SequenceEqual(b.TokenIds),
                ["equal_reference_prefix"] = PrefixOfRef(c, b.TokenIds),
                ["step0_max_abs_logprob_vs_whole"] = whole is null ? null : MaxAbs(LogSoftmax(whole), LogSoftmax(chunked!)),
                ["prefill_ms_whole"] = a?.Timings.Prefill.TotalMilliseconds,
                ["prefill_ms_chunked"] = b.Timings.Prefill.TotalMilliseconds,
            };
        }

        return o;
    }

    // The runtime-semantics checks that feed G2 (cached vs uncached, chunked prefill, concurrency) on any generation
    // fixture, with the cases chosen from the fixture by a fixed rule instead of by development-prompt id. The checks
    // compare with the reference's greedy prefix, so the cases are those whose reference is furthest from a near tie (the
    // near-tie rule belongs to the greedy-ids check, not to these): cached vs uncached on the case with the largest
    // smallest top-2 margin over its first 12 steps; concurrency on the six cases with the largest smallest margin over
    // their first 24 steps (four in parallel on two threads, two interleaved); chunked prefill on the longest prompt
    // (chunks of 64 and 32 tokens) and the second longest (chunks of 7). Ties go to the earlier case in the fixture.
    void FixtureSemantics()
    {
        double MinMargin(Case c, int steps) => c.RefStepMargin!.Take(steps).DefaultIfEmpty(double.PositiveInfinity).Min();
        List<Case> Rank(Func<Case, double> key) => [.. cases.Select((c, i) => (c, i)).OrderByDescending(x => key(x.c)).ThenBy(x => x.i).Select(x => x.c)];
        var byCached = Rank(c => MinMargin(c, 12));
        var byConcurrency = Rank(c => MinMargin(c, 24));
        var byLength = Rank(c => c.RefPromptIds.Length);
        var cached = byCached[0];
        var parallel = byConcurrency.Take(4).ToArray();
        var (ia, ib) = (byConcurrency[4], byConcurrency[5]);
        (string Id, int Chunk)[] chunkRuns = [(byLength[0].Id, 64), (byLength[0].Id, 32), (byLength[1].Id, 7)];
        JsonNode Margin(Case c, int steps, string key) => new JsonObject { ["case"] = c.Id, [key] = MinMargin(c, steps) };
        var res = new JsonObject
        {
            ["selection"] = new JsonObject
            {
                ["rule"] = "cached_vs_uncached: largest min reference step_margin over steps 0-11; concurrency: six largest min step_margin over steps 0-23 (4 parallel on 2 threads, 2 interleaved); chunked_prefill: longest prompt (chunks 64, 32) and second longest (chunk 7); ties to fixture order",
                ["cached_vs_uncached"] = Margin(cached, 12, "min_margin_12_steps"),
                ["concurrency_parallel"] = new JsonArray([.. parallel.Select(c => Margin(c, 24, "min_margin_24_steps"))]),
                ["concurrency_interleaved"] = new JsonArray([.. new[] { ia, ib }.Select(c => Margin(c, 24, "min_margin_24_steps"))]),
                ["chunked_prefill"] = new JsonArray([.. chunkRuns.Select(r => (JsonNode)new JsonObject { ["case"] = r.Id, ["chunk"] = r.Chunk, ["prompt_tokens"] = cases.First(c => c.Id == r.Id).RefPromptIds.Length })]),
            },
            ["cached_vs_uncached"] = CachedVsUncached(cached),
            ["concurrency"] = Concurrency([.. parallel.Select(c => c.Id)], ia, ib),
            ["chunked_prefill"] = ChunkedPrefill(chunkRuns),
        };
        File.WriteAllText(Path.Combine(opt.Out, "semantics.json"), res.ToJsonString(Json.Indented) + "\n");
        Console.WriteLine(res.ToJsonString(Json.Indented));
    }

    // Concurrency stress: repeated rounds of requests on parallel threads sharing the model, each compared with the
    // reference; then the same requests sequentially. Records the first divergence and the reference margin there.
    void Stress()
    {
        string[] ids = ["zh", "code-py", "en-math", "ja", "en-list", "th"];
        const int rounds = 8;
        const int tokens = 32;
        var res = new JsonObject();
        foreach (int threads in new[] { 1, 2, 4 })
        {
            var arr = new JsonArray();
            int mismatches = 0;
            for (int round = 0; round < rounds; round++)
            {
                var outs = new ConcurrentDictionary<string, int[]>();
                Parallel.ForEach(ids, new ParallelOptions { MaxDegreeOfParallelism = threads }, id =>
                {
                    var c = cases.First(x => x.Id == id);
                    outs[id] = [.. model.Generator.Generate(c.RefPromptIds, new GenerationOptions { MaxNewTokens = tokens }).TokenIds];
                });
                foreach (string id in ids)
                {
                    var c = cases.First(x => x.Id == id);
                    int[] o = outs[id];
                    int n = 0;
                    while (n < o.Length && n < c.RefOutputIds.Length && o[n] == c.RefOutputIds[n])
                    {
                        n++;
                    }

                    if (n < o.Length)
                    {
                        mismatches++;
                        arr.Add(new JsonObject { ["round"] = round, ["id"] = id, ["first_divergence"] = n, ["generated"] = o.Length });
                    }
                }
            }

            res[$"threads_{threads}"] = new JsonObject { ["rounds"] = rounds, ["requests"] = rounds * ids.Length, ["mismatching_requests"] = mismatches, ["mismatches"] = arr };
            Console.WriteLine($"stress threads {threads}: {mismatches} of {rounds * ids.Length} requests differ from the reference prefix");
        }

        File.WriteAllText(Path.Combine(opt.Out, "stress.json"), res.ToJsonString(Json.Indented) + "\n");
    }

    // Minimal reproducer without any Starling code on the generation path: the plain ONNX Runtime GenAI C# API
    // (Config, Model, GeneratorParams, Generator), one Generator per request, greedy, on parallel threads.
    void RawStress()
    {
        string[] ids = ["zh", "code-py", "en-math", "ja", "en-list", "th"];
        const int rounds = 8;
        const int tokens = 32;
        string dir = Path.Combine(opt.Bundle, Qwen35ModelAssets.VariantDirectory(opt.Variant));
        using var config = new Microsoft.ML.OnnxRuntimeGenAI.Config(dir);
        config.ClearProviders();
        if (opt.Device == "cuda")
        {
            config.AppendProvider("cuda");
            config.SetProviderOption("cuda", "use_tf32", "0");
        }

        config.Overlay("{\"model\":{\"pad_token_id\":248319}}");
        using var raw = new Microsoft.ML.OnnxRuntimeGenAI.Model(config);
        var res = new JsonObject();
        foreach (int threads in new[] { 1, 2 })
        {
            int mismatches = 0;
            var arr = new JsonArray();
            for (int round = 0; round < rounds; round++)
            {
                var outs = new ConcurrentDictionary<string, List<int>>();
                Parallel.ForEach(ids, new ParallelOptions { MaxDegreeOfParallelism = threads }, id =>
                {
                    var c = cases.First(x => x.Id == id);
                    using var p = new Microsoft.ML.OnnxRuntimeGenAI.GeneratorParams(raw);
                    p.SetSearchOption("max_length", c.RefPromptIds.Length + tokens);
                    p.SetSearchOption("do_sample", false);
                    using var g = new Microsoft.ML.OnnxRuntimeGenAI.Generator(raw, p);
                    g.AppendTokens(c.RefPromptIds);
                    var o = new List<int>();
                    while (!g.IsDone() && o.Count < tokens)
                    {
                        g.GenerateNextToken();
                        o.Add(g.GetNextTokens()[0]);
                    }

                    outs[id] = o;
                });
                foreach (string id in ids)
                {
                    var c = cases.First(x => x.Id == id);
                    var o = outs[id];
                    int n = 0;
                    while (n < o.Count && n < c.RefOutputIds.Length && o[n] == c.RefOutputIds[n])
                    {
                        n++;
                    }

                    if (n < o.Count)
                    {
                        mismatches++;
                        arr.Add(new JsonObject { ["round"] = round, ["id"] = id, ["first_divergence"] = n });
                    }
                }
            }

            res[$"threads_{threads}"] = new JsonObject { ["requests"] = rounds * ids.Length, ["mismatching_requests"] = mismatches, ["mismatches"] = arr };
            Console.WriteLine($"raw GenAI stress threads {threads}: {mismatches} of {rounds * ids.Length} requests differ from the reference prefix");
        }

        File.WriteAllText(Path.Combine(opt.Out, "rawstress.json"), res.ToJsonString(Json.Indented) + "\n");
    }

    void LongContext()
    {
        int chunk = opt.PrefillChunk ?? 512;
        Directory.CreateDirectory(Path.Combine(opt.Out, "step0"));
        string refPath = File.Exists(Path.Combine(opt.Reference, "generation.jsonl")) ? Path.Combine(opt.Reference, "generation.jsonl") : Path.Combine(opt.Reference, "results.jsonl");
        var refRows = File.ReadAllText(refPath, Encoding.UTF8).Split('\n').Where(l => l.Trim().Length > 0)
            .Select(l => JsonNode.Parse(l)!.AsObject()).ToDictionary(r => (string)r["id"]!);
        var rows = new JsonArray();
        foreach (var c in cases)
        {
            var refRow = refRows[c.Id];
            double[] refLp = [.. refRow["step_logprob"]!.AsArray().Select(x => (double)x!)];
            double[] refMargin = [.. refRow["step_margin"]!.AsArray().Select(x => (double)x!)];
            int steps = c.MaxNewTokens;
            int[] checkpoints = [.. new[] { 2, steps / 2, steps }.Where(k => k >= 2).Distinct()];
            var gen = new List<int>();
            var chosenLp = new List<double>();
            var cachedAt = new Dictionary<int, float[]>();
            float[] step0;
            var clock = Stopwatch.StartNew();
            double prefillMs;
            using (var seq = model.Backend.CreateSequence(new GenAISequenceOptions { MaxLength = c.RefPromptIds.Length + steps }))
            {
                for (int i = 0; i < c.RefPromptIds.Length; i += chunk)
                {
                    seq.AppendTokens(c.RefPromptIds.AsSpan(i, Math.Min(chunk, c.RefPromptIds.Length - i)));
                }

                prefillMs = clock.Elapsed.TotalMilliseconds;
                var l0 = new float[seq.VocabularySize];
                seq.CopyLastLogits(l0);
                step0 = LogSoftmax(l0);
                WriteF32(Path.Combine(opt.Out, "step0", c.Id + ".f32"), step0);
                for (int k = 1; k <= steps && !seq.IsDone; k++)
                {
                    int t = seq.GenerateNextToken();
                    gen.Add(t);

                    // After GenerateNextToken, GenAI reports the logits that selected t (P-016 evidence, section 4).
                    var l = new float[seq.VocabularySize];
                    seq.CopyLastLogits(l);
                    var lp = LogSoftmax(l);
                    chosenLp.Add(lp[t]);
                    if (checkpoints.Contains(k))
                    {
                        cachedAt[k] = lp;
                    }

                    if (model.Info.StopTokenIds.Contains(t))
                    {
                        break;
                    }
                }
            }

            double decodeMs = clock.Elapsed.TotalMilliseconds - prefillMs;
            int n = 0;
            while (n < gen.Count && n < c.RefOutputIds.Length && gen[n] == c.RefOutputIds[n])
            {
                n++;
            }

            // Chosen-token log-probability against the reference over the common prefix.
            double lpWorst = 0;
            for (int k = 0; k < n; k++)
            {
                lpWorst = Math.Max(lpWorst, Math.Abs(chosenLp[k] - refLp[k]));
            }

            var cache = new JsonArray();
            double cacheWorst = 0;
            foreach (int k in checkpoints.Where(k => k <= gen.Count))
            {
                int[] prefix = [.. c.RefPromptIds, .. gen.Take(k - 1)];
                var t0 = Stopwatch.GetTimestamp();
                var unc = LogSoftmax(PrefillChunked(prefix, chunk));
                double d = MaxAbs(cachedAt[k], unc);
                cacheWorst = Math.Max(cacheWorst, d);
                cache.Add(new JsonObject
                {
                    ["step"] = k,
                    ["prefix_tokens"] = prefix.Length,
                    ["max_abs_logprob"] = d,
                    ["uncached_argmax_equals_token"] = ArgMax(unc) == gen[k - 1],
                    ["uncached_chosen_logprob_minus_reference"] = k - 1 < n ? unc[gen[k - 1]] - refLp[k - 1] : null,
                    ["uncached_prefill_ms"] = Stopwatch.GetElapsedTime(t0).TotalMilliseconds,
                });
            }

            int[] refOut = c.RefOutputIds;
            var row = new JsonObject
            {
                ["id"] = c.Id,
                ["prompt_tokens"] = c.RefPromptIds.Length,
                ["prefill_chunk_tokens"] = chunk,
                ["generated"] = gen.Count,
                ["reference_generated"] = refOut.Length,
                ["exact"] = gen.SequenceEqual(refOut),
                ["common_prefix"] = n,
                ["first_divergence_reference_margin"] = n < Math.Min(gen.Count, refOut.Length) ? refMargin[n] : null,
                ["min_reference_margin"] = refMargin.Take(Math.Max(n, 1)).Min(),
                ["step0_argmax"] = ArgMax(step0),
                ["step0_argmax_equals_reference"] = ArgMax(step0) == refOut[0],
                ["chosen_logprob_max_abs_vs_reference"] = lpWorst,
                ["cached_vs_uncached_max_abs_logprob"] = cacheWorst,
                ["cached_vs_uncached"] = cache,
                ["output_ids"] = Ints(gen),
                ["output_text"] = model.Tokenizer.Decode(gen.Where(i => !model.Info.StopTokenIds.Contains(i)).ToArray(), false),
                ["prefill_ms"] = prefillMs,
                ["decode_ms"] = decodeMs,
            };
            rows.Add(row);
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"{c.Id,-14} prompt {c.RefPromptIds.Length,6} gen {gen.Count,3} {((bool)row["exact"]! ? "exact" : $"DIFF@{n}")} " +
                $"step0 {((bool)row["step0_argmax_equals_reference"]! ? "eq" : "NE")} chosen-lp {lpWorst:E2} cache {cacheWorst:E2} " +
                $"prefill {prefillMs / 1000:F1}s decode {decodeMs / 1000:F1}s"));
        }

        File.WriteAllText(Path.Combine(opt.Out, "longcontext.json"), rows.ToJsonString(Json.Indented) + "\n");
    }

    float[] PrefillChunked(int[] ids, int chunk)
    {
        using var seq = model.Backend.CreateSequence(new GenAISequenceOptions { MaxLength = ids.Length + 1 });
        for (int i = 0; i < ids.Length; i += chunk)
        {
            seq.AppendTokens(ids.AsSpan(i, Math.Min(chunk, ids.Length - i)));
        }

        var l = new float[seq.VocabularySize];
        seq.CopyLastLogits(l);
        return l;
    }

    float[] Prefill(int[] ids)
    {
        using var seq = model.Backend.CreateSequence(new GenAISequenceOptions { MaxLength = ids.Length + 1 });
        seq.AppendTokens(ids);
        var l = new float[seq.VocabularySize];
        seq.CopyLastLogits(l);
        return l;
    }

    static float[] LogSoftmax(float[] logits)
    {
        double max = double.NegativeInfinity;
        foreach (float v in logits)
        {
            max = Math.Max(max, v);
        }

        double sum = 0;
        foreach (float v in logits)
        {
            sum += Math.Exp(v - max);
        }

        double lse = max + Math.Log(sum);
        var r = new float[logits.Length];
        for (int i = 0; i < r.Length; i++)
        {
            r[i] = (float)(logits[i] - lse);
        }

        return r;
    }

    static double MaxAbs(float[] a, float[] b)
    {
        double m = 0;
        for (int i = 0; i < a.Length; i++)
        {
            m = Math.Max(m, Math.Abs((double)a[i] - b[i]));
        }

        return m;
    }

    static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
        {
            if (v[i] > v[best])
            {
                best = i;
            }
        }

        return best;
    }

    static void WriteF32(string path, float[] v)
    {
        var bytes = new byte[v.Length * 4];
        Buffer.BlockCopy(v, 0, bytes, 0, bytes.Length);
        if (!BitConverter.IsLittleEndian)
        {
            throw new PlatformNotSupportedException("little-endian only");
        }

        File.WriteAllBytes(path, bytes);
    }

    static JsonArray Ints(IEnumerable<int> ids) => new(ids.Select(i => (JsonNode?)i).ToArray());

    static string Sha256(string path)
    {
        using var s = File.OpenRead(path);
        return Convert.ToHexStringLower(SHA256.HashData(s));
    }
}

internal sealed record Options(string Bundle, Qwen35Variant Variant, string Device, string Prompts, string Reference, string Out, HashSet<string> Modes, int Threads, int? PrefillChunk, bool? Serialize)
{
    public static Options? Parse(string[] a)
    {
        string? Get(string key, string? fallback = null)
        {
            int i = Array.IndexOf(a, key);
            return i >= 0 && i + 1 < a.Length ? a[i + 1] : fallback;
        }

        string? bundle = Get("--bundle"), prompts = Get("--prompts"), reference = Get("--reference"), output = Get("--out");
        if (bundle is null || prompts is null || reference is null || output is null)
        {
            return null;
        }

        var variant = Get("--variant", "fp32x") switch
        {
            "fp16x" => Qwen35Variant.Fp16x,
            "int4" => Qwen35Variant.Int4,
            _ => Qwen35Variant.Fp32x,
        };
        return new Options(bundle, variant, Get("--device", "cpu")!, prompts, reference, output,
            [.. Get("--modes", "parity,semantics")!.Split(',')], int.Parse(Get("--threads", "0")!, CultureInfo.InvariantCulture),
            Get("--prefill-chunk") is string pc ? int.Parse(pc, CultureInfo.InvariantCulture) : null,
            Get("--serialize") is string ser ? bool.Parse(ser) : null);
    }
}

internal sealed record Case(string Id, IReadOnlyList<Qwen35Message> Messages, bool EnableThinking, bool HasTools, int MaxNewTokens, string RefRendered, int[] RefPromptIds, int[] RefOutputIds, double[]? RefStepMargin = null);

internal static class Cases
{
    public static List<Case> Load(string promptsPath, string referenceDir)
    {
        string refPath = File.Exists(Path.Combine(referenceDir, "generation.jsonl"))
            ? Path.Combine(referenceDir, "generation.jsonl")
            : Path.Combine(referenceDir, "results.jsonl");
        var refs = ReadJsonl(refPath).ToDictionary(r => (string)r["id"]!);
        var list = new List<Case>();
        foreach (var p in ReadJsonl(promptsPath))
        {
            string id = (string)p["id"]!;
            var r = refs.TryGetValue(id, out var found) ? found : throw new InvalidDataException($"prompt '{id}' of {promptsPath} has no reference row in {refPath}");
            var messages = p["messages"]!.AsArray().Select(m =>
            {
                string role = (string)m!["role"]!;
                var parsed = role switch
                {
                    "system" => Qwen35Role.System,
                    "user" => Qwen35Role.User,
                    "assistant" => Qwen35Role.Assistant,
                    _ => throw new NotSupportedException($"role {role}"),
                };
                return new Qwen35Message(parsed, (string)m["content"]!, (string?)m["reasoning_content"]);
            }).ToList();
            var kwargs = p["chat_template_kwargs"]?.AsObject();
            bool thinking = kwargs?["enable_thinking"] is JsonNode t && (bool)t;
            list.Add(new Case(id, p["tools"] is null ? messages : new ToolsUnsupported(messages), thinking, p["tools"] is not null, (int)p["max_new_tokens"]!,
                (string)r["rendered"]!, [.. r["prompt_ids"]!.AsArray().Select(x => (int)x!)], [.. r["output_ids"]!.AsArray().Select(x => (int)x!)],
                r["step_margin"] is JsonArray margins ? [.. margins.Select(x => (double)x!)] : null));
        }

        return list;
    }

    /// <summary>The fixed prompts (of prompts-v1.jsonl) that the semantics, stress and rawstress modes use.</summary>
    public static readonly IReadOnlyDictionary<string, string[]> ModePrompts = new Dictionary<string, string[]>(StringComparer.Ordinal)
    {
        ["semantics"] = ["en-short", "en-list", "zh", "code-py", "en-math", "ja", "code-cs", "thinking-on", "multi-turn", "special-endoftext", "long-license"],
        ["stress"] = ["zh", "code-py", "en-math", "ja", "en-list", "th"],
        ["rawstress"] = ["zh", "code-py", "en-math", "ja", "en-list", "th"],
    };

    /// <summary>Why --modes fixture-semantics cannot run on these cases (see FixtureSemantics), or null.</summary>
    public static string? FixtureSemanticsProblem(IReadOnlyList<Case> cases)
    {
        if (cases.Any(c => c.RefStepMargin is null))
        {
            return "--modes fixture-semantics needs a generation fixture (generation.jsonl with step_margin) as --reference";
        }

        int[] lengths = [.. cases.Select(c => c.RefPromptIds.Length).OrderDescending()];
        return cases.Count < 6 || lengths[0] <= 64 || lengths[1] <= 7
            ? "--modes fixture-semantics needs at least 6 cases, the longest prompt above 64 tokens and the second longest above 7"
            : null;
    }

    /// <summary>The prompts the requested modes need but the prompt set lacks, described for an error message; null if none.</summary>
    public static string? MissingForModes(IReadOnlyList<Case> cases, IReadOnlySet<string> modes)
    {
        var have = cases.Select(c => c.Id).ToHashSet(StringComparer.Ordinal);
        var missing = ModePrompts
            .Where(m => modes.Contains(m.Key))
            .Select(m => (Mode: m.Key, Ids: m.Value.Where(id => !have.Contains(id)).ToArray()))
            .Where(m => m.Ids.Length > 0)
            .Select(m => $"prompts {string.Join(", ", m.Ids)} (mode {m.Mode})")
            .ToList();
        return missing.Count == 0 ? null : string.Join("; ", missing);
    }

    private static IEnumerable<JsonObject> ReadJsonl(string path) =>
        File.ReadAllText(path, Encoding.UTF8).Split('\n').Where(l => l.Trim().Length > 0).Select(l => JsonNode.Parse(l)!.AsObject());
}

/// <summary>A conversation with a tool definition: the owned renderer has no tools input, so rendering is refused.</summary>
internal sealed class ToolsUnsupported(List<Qwen35Message> inner) : IReadOnlyList<Qwen35Message>
{
    public Qwen35Message this[int index] => throw new NotSupportedException("Tool definitions are not supported by the owned template renderer.");

    public int Count => inner.Count;

    public IEnumerator<Qwen35Message> GetEnumerator() => throw new NotSupportedException("Tool definitions are not supported by the owned template renderer.");

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
}

internal static class Json
{
    public static readonly JsonSerializerOptions Indented = new() { WriteIndented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
    public static readonly JsonSerializerOptions Compact = new() { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
}
