using System.Collections.Immutable;
using Starlings.Acquisition;
using Starlings.Generation.OnnxGenAI;
using Starlings.TestSupport;

namespace Starlings.Models.Qwen35.Tests;

/// <summary>
/// Gate G4 operational tests (plan 10.2) for <c>qwen3.5-0.8b</c> fp32x on CPU that the other classes do not cover:
/// failed initialization (template, configuration, graph), malformed locks and snapshots, unsupported provider and
/// variant, queue saturation, timed-out requests with their cancellation latency, and disposal during an in-flight
/// request. Cancellation between steps then reuse and concurrent requests are in <see cref="CpuModelTests"/>; the
/// one-hour soak is ai-dotnet/tools/Starlings.Soak.Tool. Tests that need the bundle or the tokenizer skip without them.
/// </summary>
public sealed class OperationalTests(Qwen35CpuFixture fixture, ITestOutputHelper output) : IClassFixture<Qwen35CpuFixture>, IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("starlings-qwen35-g4-").FullName;

    private static CancellationToken Ct => TestContext.Current.CancellationToken;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private Qwen35ChatModel Model
    {
        get
        {
            Assert.SkipWhen(fixture.Model is null, fixture.Reason);
            return fixture.Model!;
        }
    }

    private static Qwen35ModelFiles Files()
    {
        var (files, reason) = Qwen35TestAssets.FindModel(Qwen35Variant.Fp32x);
        Assert.SkipWhen(files is null, reason);
        return files!;
    }

    private string Write(string name, string text)
    {
        string path = Path.Combine(_dir, name);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, text);
        return path;
    }

    [Fact]
    public void UnsupportedVariantAndCorruptTemplateFailBeforeAnyNativeWork()
    {
        string template = Write("chat_template.jinja", "{{ messages }}");
        var files = new Qwen35ModelFiles(Qwen35Variant.Fp32x, _dir, Path.Combine(_dir, "tokenizer.json"), template, null!);
        Assert.Throws<NotSupportedException>(() => Qwen35ChatModel.Load(files with { Variant = Qwen35Variant.Fp16x }));
        var ex = Assert.Throws<InvalidDataException>(() => Qwen35ChatModel.Load(files));
        Assert.Contains(Qwen35ChatTemplate.TemplateSha256, ex.Message, StringComparison.Ordinal);
        Assert.Throws<FileNotFoundException>(() => Qwen35ChatModel.Load(files with { ChatTemplatePath = Path.Combine(_dir, "absent.jinja") }));
    }

    [Fact]
    public void MalformedLocksAndSnapshotsAreRejected()
    {
        foreach (var variant in new[] { Qwen35Variant.Fp32x, Qwen35Variant.Fp16x })
        {
            OperationalChecks.MalformedLocksAreRejected(Qwen35ModelAssets.GetLock(variant), output);
            OperationalChecks.WrongSnapshotsAreRejected(Qwen35ModelAssets.GetLock(variant), dir => Qwen35ModelAssets.OpenDirectory(dir, variant, CacheVerification.Size));
        }
    }

    [Fact]
    public void MalformedConfigurationOrMissingGraphFailsTyped()
    {
        var files = Files();
        string config = File.ReadAllText(Path.Combine(files.ModelDirectory, "genai_config.json"));

        // genai_config.json that is not JSON.
        string notJson = Path.Combine(_dir, "not-json");
        Write("not-json/genai_config.json", config[..(config.Length / 3)]);
        output.WriteLine("truncated config: " + Assert.Throws<GenAIModelException>(() => Qwen35ChatModel.Load(files with { ModelDirectory = notJson })).Message);

        // A valid configuration without the graph next to it.
        string noGraph = Path.Combine(_dir, "no-graph");
        Write("no-graph/genai_config.json", config);
        output.WriteLine("missing graph: " + Assert.Throws<GenAIModelException>(() => Qwen35ChatModel.Load(files with { ModelDirectory = noGraph })).Message);

        // No configuration at all.
        string empty = Directory.CreateDirectory(Path.Combine(_dir, "empty")).FullName;
        Assert.Throws<FileNotFoundException>(() => Qwen35ChatModel.Load(files with { ModelDirectory = empty }));

        // A tokenizer that is not the qwen35-bpe-chat asset.
        var tokenizerError = Record.Exception(() => Qwen35ChatModel.Load(files with { TokenizerPath = Write("tokenizer.json", "{}") }));
        output.WriteLine($"bad tokenizer: {tokenizerError?.GetType().FullName}: {tokenizerError?.Message}");
        Assert.True(tokenizerError is Starlings.Tokenizers.TokenizerFormatException or InvalidDataException, tokenizerError?.ToString() ?? "a bad tokenizer.json loaded");
    }

    [Fact]
    public void UnsupportedProvidersAreRejectedNotIgnored()
    {
        var files = Files();
        Assert.Throws<ArgumentOutOfRangeException>(() => Qwen35ChatModel.Load(files, new Qwen35ModelOptions { Provider = (GenAIExecutionProvider)7 }));

        // This project loads the CPU native package: CUDA fails the load; there is no silent CPU fallback.
        var ex = Record.Exception(() => Qwen35ChatModel.Load(files, new Qwen35ModelOptions { Provider = GenAIExecutionProvider.Cuda }).Dispose());
        output.WriteLine($"{ex?.GetType().Name}: {ex?.Message}");
        Assert.True(ex is GenAIProviderUnavailableException or GenAIModelException, ex?.ToString() ?? "CUDA load succeeded on the CPU native package");
    }

    [Fact]
    public void SaturatedRequestsAllCompleteWithTheirSingleRequestIds()
    {
        var model = Model;
        string[] prompts = ["Name a colour.", "Say hello in French.", "What is 2 + 3?", "Name a planet."];
        OperationalChecks.SaturatedCallersComplete(
            prompts.Length,
            (i, ct) => model.Generate([Qwen35Message.User(prompts[i])], new Qwen35ChatOptions { MaxNewTokens = 6 }, ct).TokenIds,
            (expected, actual, i) => Assert.Equal(expected, actual),
            output,
            callers: Math.Max(8, Environment.ProcessorCount),
            callsPerCaller: 1,
            timeout: TimeSpan.FromMinutes(10));
    }

    [Fact]
    public void TimedOutRequestsReportCancellationLatencyThenReuse()
    {
        var model = Model;
        var c = Qwen35TestAssets.LoadCases().Single(c => c.Id == "en-short");
        var longPrompt = Qwen35Prompt.FromTokenIds(Qwen35TestAssets.LoadCases().Single(c => c.Id == "en-list").PromptIds);
        OperationalChecks.TimedOutRequestsThenReuse(
            ct =>
            {
                foreach (var update in model.GenerateStreaming(longPrompt, new Qwen35ChatOptions { MaxNewTokens = 512 }, ct))
                {
                    _ = update;
                }
            },
            () => Qwen35ModelChecks.GreedyFromReferenceIds(model, c),
            TimeSpan.FromMilliseconds(400),
            TimeSpan.FromSeconds(5),
            output);
    }

    [Fact]
    public void DisposalDuringAnInFlightRequestLetsItFinishThenRejectsNewOnes()
    {
        var files = Files();
        var model = Qwen35ChatModel.Load(files);
        var c = Qwen35TestAssets.LoadCases().Single(c => c.Id == "en-short");
        var prompt = Qwen35Prompt.FromTokenIds(c.PromptIds);
        var results = new List<ImmutableArray<int>>();
        OperationalChecks.DisposeDuringInFlightWork(
            () =>
            {
                var r = model.Generate(prompt, new Qwen35ChatOptions { MaxNewTokens = 32 }, Ct);
                lock (results)
                {
                    results.Add(r.TokenIds);
                }
            },
            model.Dispose,
            () => model.Generate(prompt, new Qwen35ChatOptions { MaxNewTokens = 2 }, Ct),
            output,
            settle: TimeSpan.FromMilliseconds(400));

        // Requests that finished, including the one in flight at disposal, are complete and correct.
        Assert.All(results, ids => Assert.Equal(c.ContentIds.Take(32), ids));
    }
}
