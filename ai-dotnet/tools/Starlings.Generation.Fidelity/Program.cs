using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Starlings.Generation;
using Starlings.Generation.OnnxGenAI;
using Starlings.Models.QwenChat;
using Starlings.Tokenizers;

return ToolMain.Run(ToolBody);

int ToolBody()
{
    var arguments = new Dictionary<string, string>(StringComparer.Ordinal);
    for (int index = 0; index < args.Length; index += 2)
    {
        if (index + 1 >= args.Length || !arguments.TryAdd(args[index], args[index + 1]))
        {
            throw new ArgumentException("Expected unique --candidate/--bundle/--corpus/--out/--threads argument pairs.");
        }
    }

    using JsonDocument candidate = JsonDocument.Parse(File.ReadAllText(arguments["--candidate"]));
    QwenChatProfile profile = QwenChatProfile.FromId(candidate.RootElement.GetProperty("id").GetString()!);
    if (candidate.RootElement.GetProperty("origin").GetProperty("revision").GetString() != profile.Revision)
    {
        throw new InvalidDataException("Wrong original revision.");
    }

    int threads = arguments.TryGetValue("--threads", out string? value) ? int.Parse(value, System.Globalization.CultureInfo.InvariantCulture) : 2;
    string output = Path.GetFullPath(arguments["--out"]);
    string bundle = Path.GetFullPath(arguments["--bundle"]);
    if (output.StartsWith(bundle + Path.DirectorySeparatorChar, StringComparison.Ordinal) || output == bundle)
    {
        throw new InvalidDataException("Output overlaps model bundle.");
    }

    Directory.CreateDirectory(output);
    bool templateOnly = arguments.TryGetValue("--templates-only", out string? templates) && templates == "true";
    using QwenChatModel? model = templateOnly ? null : QwenChatModel.Load(profile, bundle, new QwenChatModelOptions { IntraOpThreads = threads });
    Tokenizer tokenizer = model?.Tokenizer ?? profile.OpenTokenizer(Path.Combine(bundle, "tokenizer.json"));
    using var writer = new StreamWriter(Path.Combine(output, templateOnly ? "templates.jsonl" : "results.jsonl"), false, new UTF8Encoding(false));
    var names = new HashSet<string>(StringComparer.Ordinal);
    int rowIndex = 0;
    foreach (string line in File.ReadLines(arguments["--corpus"]))
    {
        using JsonDocument document = JsonDocument.Parse(line);
        JsonElement row = document.RootElement;
        string id = row.GetProperty("id").GetString() ?? throw new InvalidDataException("Missing corpus id.");
        if (!names.Add(id) || string.IsNullOrWhiteSpace(id))
        {
            throw new InvalidDataException("Duplicate or empty corpus id.");
        }

        if (row.TryGetProperty("enable_thinking", out JsonElement thinking) && thinking.GetBoolean())
        {
            throw new NotSupportedException("This experimental Qwen3 profile explicitly disables thinking.");
        }

        var messages = new List<QwenChatMessage>();
        foreach (JsonElement message in row.GetProperty("messages").EnumerateArray())
        {
            if (message.EnumerateObject().Any(property => property.Name is not ("role" or "content" or "reasoning_content")))
            {
                throw new NotSupportedException("Tools and multimodal input are unsupported.");
            }

            QwenChatRole role = message.GetProperty("role").GetString() switch
            {
                "system" => QwenChatRole.System,
                "user" => QwenChatRole.User,
                "assistant" => QwenChatRole.Assistant,
                _ => throw new NotSupportedException("Unsupported message role."),
            };
            messages.Add(new QwenChatMessage(role, message.GetProperty("content").GetString()!,
                message.TryGetProperty("reasoning_content", out JsonElement reasoning) ? reasoning.GetString() : null));
        }

        int maximum = row.TryGetProperty("max_new_tokens", out JsonElement max) ? max.GetInt32() :
            candidate.RootElement.GetProperty("behavior").GetProperty("max_new_tokens").GetInt32();
        var options = new QwenChatOptions { MaxNewTokens = maximum };
        string promptText = QwenChatTemplate.Render(profile, messages);
        var prompt = new QwenChatPrompt(promptText, tokenizer.Encode(promptText, addSpecialTokens: false).Ids);
        if (model is null)
        {
            writer.WriteLine(JsonSerializer.Serialize(new { id, profile = profile.Id, prompt_text = prompt.Text, prompt_ids = prompt.TokenIds }));
            writer.Flush();
            continue;
        }
        float[] logits = new float[151936];
        using (GenAISequence sequence = model.CreateSequence(new GenAISequenceOptions { MaxLength = prompt.TokenIds.Length + maximum }))
        {
            sequence.AppendTokens(prompt.TokenIds.AsSpan());
            sequence.CopyLastLogits(logits);
        }

        string logitsName = $"logits-{rowIndex++:D4}.f32";
        string logitsPath = Path.Combine(output, logitsName);
        using (var binary = new BinaryWriter(File.Create(logitsPath)))
        {
            foreach (float logit in logits)
            {
                binary.Write(logit);
            }
        }

        GenerationResult generated = model.Generate(prompt, options);
        var updates = model.GenerateStreaming(prompt, options).ToList();
        GenerationResult? streamed = updates.LastOrDefault()?.Result;
        GenerationResult chunked = model.Generate(prompt, options with { PrefillChunkTokens = 7 });
        model.Generate([new(QwenChatRole.User, "Another independent request.")], new QwenChatOptions { MaxNewTokens = 1 });
        GenerationResult repeated = model.Generate(prompt, options);
        bool cancellation = false;
        using (var canceled = new CancellationTokenSource())
        {
            canceled.Cancel();
            try
            {
                model.Generate(prompt, options, canceled.Token);
            }
            catch (OperationCanceledException)
            {
                cancellation = true;
            }
        }

        bool? cancellationBetweenSteps = null;
        using (var midRequest = new CancellationTokenSource())
        using (IEnumerator<GenerationUpdate> canceledStream = model.GenerateStreaming(prompt, options, midRequest.Token).GetEnumerator())
        {
            if (canceledStream.MoveNext() && !canceledStream.Current.IsFinal)
            {
                midRequest.Cancel();
                cancellationBetweenSteps = false;
                try
                {
                    canceledStream.MoveNext();
                }
                catch (OperationCanceledException)
                {
                    cancellationBetweenSteps = true;
                }
            }
        }

        int firstToken = generated.TokenIds.Length > 0 ? generated.TokenIds[0] : generated.StopTokenId ?? -1;
        GenerationResult stopped = model.Generate(prompt, options with { AdditionalStopTokenIds = [firstToken] });
        bool tokenStop = stopped.FinishReason == GenerationFinishReason.Stop && stopped.StopTokenId == firstToken &&
            stopped.TokenIds.IsEmpty && stopped.Text.Length == 0 && stopped.Usage.CompletionTokens == 0;
        bool stringStop = false;
        if (generated.Text.Length != 0)
        {
            string stopString = generated.Text.EnumerateRunes().First().ToString();
            GenerationResult stringStopped = model.Generate(prompt, options with { StopSequences = [stopString] });
            stringStop = stringStopped.FinishReason == GenerationFinishReason.StopSequence &&
                stringStopped.StopSequence == stopString && stringStopped.Text.Length == 0 && stringStopped.Usage.CompletionTokens > 0;
        }

        var checks = new Dictionary<string, bool>(StringComparer.Ordinal)
        {
            ["cache"] = Same(generated, chunked),
            ["isolation"] = Same(generated, repeated),
            ["stream"] = streamed is not null && Same(generated, streamed) && updates.Count(update => update.IsFinal) == 1 &&
                string.Concat(updates.Select(update => update.Text)) == generated.Text &&
                updates.SelectMany(update => update.TokenIds).SequenceEqual(generated.TokenIds),
            ["usage"] = generated.Usage.PromptTokens == prompt.TokenIds.Length && generated.Usage.CompletionTokens == generated.TokenIds.Length,
            ["cancellation"] = cancellation,
            ["token_stop"] = tokenStop,
            ["string_stop"] = stringStop,
        };
        var result = new
        {
            id,
            profile = profile.Id,
            revision = profile.Revision,
            backend = "dotnet",
            runtime = new { backend = "dotnet", version = model.Info.GenAIVersion, precision = "fp32", provider = "CPUExecutionProvider" },
            prompt_text = prompt.Text,
            prompt_ids = prompt.TokenIds,
            output_ids = generated.TokenIds,
            output_text = generated.Text,
            finish_reason = generated.FinishReason == GenerationFinishReason.Stop ? "stop" : "length",
            stop_token_id = generated.StopTokenId,
            usage = new { prompt_tokens = generated.Usage.PromptTokens, completion_tokens = generated.Usage.CompletionTokens },
            initial_logits = new
            {
                path = logitsName, dtype = "float32-le", shape = new[] { 1, 151936 },
                sha256 = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(logitsPath))),
            },
            checks,
            cancellation_between_steps = cancellationBetweenSteps,
        };
        writer.WriteLine(JsonSerializer.Serialize(result));
        writer.Flush();
        Console.WriteLine($"{id}: prompt={prompt.TokenIds.Length} completion={generated.TokenIds.Length} stop={generated.StopTokenId}");
    }

    if (names.Count == 0)
    {
        throw new InvalidDataException("Empty corpus.");
    }

    if (model is not null)
    {
        File.WriteAllText(Path.Combine(output, "environment.json"), JsonSerializer.Serialize(model.Info));
    }

    return 0;

    static bool Same(GenerationResult first, GenerationResult second) =>
        first.Text == second.Text && first.TokenIds.SequenceEqual(second.TokenIds) && first.FinishReason == second.FinishReason &&
        first.StopTokenId == second.StopTokenId && first.Usage == second.Usage;
}
