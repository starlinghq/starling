// Streaming console chat with Qwen/Qwen3.5-0.8B from the Starlings packages.
//
//   dotnet run -- download [--cache <dir>] [--variant fp32x|fp16x]           explicit acquisition (network)
//   dotnet run -- chat [--cache <dir> | --snapshot <dir>] [--variant fp32x|fp16x] [--cuda] [--thinking] [--max-new <n>] [--chat-client]
//                                                                             offline: interactive streaming chat
//   dotnet run -- ask  [same options] "question"                             offline: one streamed answer, then exit
//
// Without --cache the files go to $STARLING_MODELS or the user cache directory. --snapshot verifies an existing directory
// in the snapshot layout (chat_template.jinja, transformers-5.18.0/tokenizer.json, ortgenai-<variant>/) instead.
// Generation never downloads anything. In chat, Ctrl+C cancels the answer being generated (at the next decode step);
// an empty line or /exit quits, /reset clears the conversation. --chat-client streams through the Microsoft.Extensions.AI
// IChatClient adapter (Qwen35ChatClient) instead of the typed API; the tokens are the same.
using System.Globalization;
using System.Reflection;
using System.Text;
using Starlings.Acquisition;
using Starlings.Generation.OnnxGenAI;
using Starlings.Models.Qwen35;
using Starlings.Models.Qwen35.Extensions.AI;
using Microsoft.Extensions.AI;

// A failure ends with a one-line message and exit code 1, not an unhandled-exception abort.
try
{
    return await ToolBody();
}
catch (Exception ex) when (ex is not (OutOfMemoryException or StackOverflowException))
{
    Console.Error.WriteLine($"error: {ex.Message}");
    return 1;
}

async Task<int> ToolBody()
{
    if (args.Length == 0 || args[0] is not ("download" or "chat" or "ask"))
    {
        Console.Error.WriteLine("usage: QwenChat download [--cache <dir>] [--variant fp32x|fp16x] | chat|ask [--cache <dir> | --snapshot <dir>] [--variant fp32x|fp16x] [--cuda] [--thinking] [--max-new <n>] [--chat-client] [question]");
        return 2;
    }

    string? cache = null;
    string? snapshot = null;
    var variant = Qwen35Variant.Fp32x;
    bool cuda = false;
    bool thinking = false;
    bool useChatClient = false;
    int maxNew = 512;
    var rest = new List<string>();
    for (int i = 1; i < args.Length; i++)
    {
        switch (args[i])
        {
            case "--cache" when i + 1 < args.Length:
                cache = args[++i];
                break;
            case "--snapshot" when i + 1 < args.Length:
                snapshot = args[++i];
                break;
            case "--variant" when i + 1 < args.Length:
                variant = args[++i] == "fp16x" ? Qwen35Variant.Fp16x : Qwen35Variant.Fp32x;
                break;
            case "--cuda":
                cuda = true;
                break;
            case "--thinking":
                thinking = true;
                break;
            case "--chat-client":
                useChatClient = true;
                break;
            case "--max-new" when i + 1 < args.Length:
                maxNew = int.Parse(args[++i], CultureInfo.InvariantCulture);
                break;
            default:
                rest.Add(args[i]);
                break;
        }
    }

    var acquisition = new ModelAcquisitionOptions { CacheDirectory = cache };

    if (args[0] == "download")
    {
        // 1. Explicit acquisition of the pinned lock: every file (chat_template.jinja, the runtime tokenizer.json and the built
        //    graph) from the Hugging Face hosting repository at its pinned commit. The only step that uses the network.
        var assetLock = Qwen35ModelAssets.GetLock(variant);
        Console.WriteLine($"lock:  {assetLock}");
        Console.WriteLine($"cache: {acquisition.ResolveCacheDirectory()}");
        Console.WriteLine($"from:  {Qwen35ModelAssets.HostingBaseUrl}");

        var progress = new FileProgress();
        Qwen35ModelFiles downloaded = await Qwen35ModelAssets.DownloadAsync(variant, acquisition, progress);
        foreach (ModelAssetFile file in assetLock.Files)
        {
            string how = progress.Downloaded.Contains(file.Path) ? "downloaded" : "already cached";
            Console.WriteLine($"{file.Path}: {how}, {file.Size} bytes, sha256 {file.Sha256} verified");
        }

        Console.WriteLine($"snapshot: {downloaded.Snapshot.Directory}");
        return 0;
    }

    // 2. Offline load: verifies the files (size and SHA-256) and never downloads. Loading the 3 GB fp32x graph takes a while.
    Qwen35ModelFiles files = snapshot is null ? Qwen35ModelAssets.OpenLocal(variant, acquisition) : Qwen35ModelAssets.OpenDirectory(snapshot, variant);
    using var model = Qwen35ChatModel.Load(files, new Qwen35ModelOptions { Provider = cuda ? GenAIExecutionProvider.Cuda : GenAIExecutionProvider.Cpu });
    Console.WriteLine($"model: {model.Info.Profile} ({model.Info.Checkpoint}) on {model.Info.Provider}, ONNX Runtime GenAI {model.Info.GenAIVersion}");
    Console.WriteLine($"tokenizer: {model.Info.Tokenizer}");
    Console.WriteLine($"packages: {Version<Qwen35ChatModel>()}, {Version<GenAIModel>()}, {Version<Starlings.Tokenizers.Tokenizer>()}, {Version<Qwen35ChatClient>()}; " +
        $"GenAI managed {GenAIRuntimeInfo.ManagedVersion ?? "unknown"}, native {GenAIRuntimeInfo.NativePackage ?? "unknown"}");
    var options = new Qwen35ChatOptions { MaxNewTokens = maxNew, EnableThinking = thinking };
    // The same model behind Microsoft.Extensions.AI's IChatClient; the client wraps the model and does not own it.
    using IChatClient chatClient = new Qwen35ChatClient(model, new Qwen35ChatClientOptions { Defaults = options });

    if (args[0] == "ask")
    {
        string question = rest.Count > 0 ? string.Join(' ', rest) : "What is the capital of France? Answer in one sentence.";
        Console.WriteLine($"user: {question}");
        Console.Write("assistant: ");
        Qwen35Message[] ask = [Qwen35Message.User(question)];
        var reply = useChatClient ? await StreamChatClient(ask, CancellationToken.None) : Stream(ask, CancellationToken.None);
        return reply is null ? 1 : 0;
    }

    // 3. Interactive chat. Each turn sends the whole conversation (the template re-renders it); requests share no state.
    var history = new List<Qwen35Message>();
    CancellationTokenSource? current = null;
    Console.CancelKeyPress += (_, e) =>
    {
        if (current is { IsCancellationRequested: false } cts)
        {
            e.Cancel = true;
            cts.Cancel();
        }
    };
    Console.WriteLine("Type a message (empty line or /exit quits, /reset clears the conversation, Ctrl+C stops an answer).");
    while (true)
    {
        Console.Write("\nyou> ");
        string? line = Console.ReadLine();
        if (string.IsNullOrWhiteSpace(line) || line.Trim() == "/exit")
        {
            return 0;
        }

        if (line.Trim() == "/reset")
        {
            history.Clear();
            Console.WriteLine("(conversation cleared)");
            continue;
        }

        history.Add(Qwen35Message.User(line));
        using (current = new CancellationTokenSource())
        {
            Console.Write("qwen> ");
            string? reply = useChatClient ? await StreamChatClient(history, current.Token) : Stream(history, current.Token);
            if (reply is null)
            {
                history.RemoveAt(history.Count - 1);
            }
            else
            {
                history.Add(Qwen35Message.Assistant(reply));
            }
        }

        current = null;
    }

    // Streams one answer to the console; returns the full text, or null when cancelled.
    string? Stream(IReadOnlyList<Qwen35Message> messages, CancellationToken cancellationToken)
    {
        try
        {
            foreach (GenerationUpdate update in model.GenerateStreaming(messages, options, cancellationToken))
            {
                Console.Write(update.Text);
                if (update.Result is { } done)
                {
                    Console.WriteLine();
                    Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                        $"[{done.FinishReason}; {done.Usage.PromptTokens} prompt + {done.Usage.CompletionTokens} completion tokens; " +
                        $"prefill {done.Timings.Prefill.TotalMilliseconds:F0} ms, " +
                        $"{(done.Usage.CompletionTokens > 0 ? done.Usage.CompletionTokens / done.Timings.Decode.TotalSeconds : 0):F1} tokens/s]"));
                    return done.Text;
                }
            }
        }
        catch (OperationCanceledException)
        {
            Console.WriteLine("\n[cancelled]");
        }

        return null;
    }

    // The same through IChatClient: text deltas, then the finish reason and usage on the last update.
    async Task<string?> StreamChatClient(IReadOnlyList<Qwen35Message> messages, CancellationToken cancellationToken)
    {
        var conversation = messages.Select(m => new ChatMessage(
            m.Role switch { Qwen35Role.System => ChatRole.System, Qwen35Role.User => ChatRole.User, _ => ChatRole.Assistant },
            m.Content)).ToList();
        var text = new StringBuilder();
        try
        {
            await foreach (ChatResponseUpdate update in chatClient.GetStreamingResponseAsync(conversation, cancellationToken: cancellationToken))
            {
                Console.Write(update.Text);
                text.Append(update.Text);
                if (update.FinishReason is { } finish)
                {
                    UsageDetails usage = update.Contents.OfType<UsageContent>().Single().Details;
                    Console.WriteLine();
                    Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                        $"[{(finish == ChatFinishReason.Stop ? "Stop" : "Length")}; {usage.InputTokenCount} prompt + {usage.OutputTokenCount} completion tokens; IChatClient]"));
                    return text.ToString();
                }
            }
        }
        catch (OperationCanceledException)
        {
            Console.WriteLine("\n[cancelled]");
        }

        return null;
    }
}

// Package name and version of the assembly that defines T (the diagnostics line).
static string Version<T>()
{
    AssemblyName name = typeof(T).Assembly.GetName();
    string? informational = typeof(T).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
    return $"{name.Name} {informational?.Split('+')[0]}";
}

internal sealed class FileProgress : IProgress<ModelAssetProgress>
{
    public HashSet<string> Downloaded { get; } = [];

    public void Report(ModelAssetProgress value)
    {
        if (value.Stage == ModelAssetStage.Downloading)
        {
            Downloaded.Add(value.Path);
        }
    }
}
