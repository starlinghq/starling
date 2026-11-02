// Clean-consumer check of Starlings.Models.Bert for the release-candidate drill (tools/release-drill-ai.sh).
// The script copies this file into a temporary directory outside the repository and builds it against the packed
// packages from a local folder feed. It runs offline (no model download): it proves the built-in profiles survive
// packing, so the package is usable without the repository's candidate files, and that Create verifies the snapshot
// before anything else (inference never downloads).
//
//   dotnet BertConsumer.dll <empty dir>
//
// Exit code 0 only if every check passes.
using Starlings.Acquisition;
using Starlings.Models.Bert;

try
{
    return Run();
}
catch (Exception ex) when (ex is not (OutOfMemoryException or StackOverflowException))
{
    Console.Error.WriteLine($"error: {ex}");
    return 70;
}

int Run()
{
    if (args.Length != 1)
    {
        Console.Error.WriteLine("usage: <empty dir>");
        return 2;
    }

    int failures = 0;
    void Check(bool ok, string what)
    {
        Console.WriteLine($"{(ok ? "ok:  " : "FAIL:")} {what}");
        if (!ok)
        {
            failures++;
        }
    }

    // Built-in name -> (profile id, task, runtime files).
    var expected = new Dictionary<string, (string Id, BertTask Task, int RuntimeFiles)>(StringComparer.Ordinal)
    {
        ["bge-base"] = ("bge-base-en-v1.5/onnx-fp32-cpu", BertTask.Embeddings, 2),
        ["bge-m3"] = ("bge-m3/onnx-fp32-cpu", BertTask.Embeddings, 3),
        ["bge-reranker-v2-m3"] = ("bge-reranker-v2-m3/onnx-fp32-cpu", BertTask.Reranking, 3),
        ["bge-small"] = ("bge-small-en-v1.5/onnx-fp32-cpu", BertTask.Embeddings, 2),
        ["minilm"] = ("all-minilm-l6-v2/onnx-fp32-cpu", BertTask.Embeddings, 2),
        ["ms-marco"] = ("ms-marco-minilm-l6-v2/onnx-fp32-cpu", BertTask.Reranking, 2),
        ["nomic-embed"] = ("nomic-embed-text-v1.5/onnx-fp32-cpu", BertTask.Embeddings, 2),
    };
    Check(BertModelProfile.BuiltInIds.SequenceEqual(expected.Keys.Order(StringComparer.Ordinal)),
        $"BuiltInIds: {string.Join(", ", BertModelProfile.BuiltInIds)}");
    foreach (var (name, want) in expected)
    {
        var profile = BertModelProfile.BuiltIn(name);
        Check(profile.Id == want.Id && profile.Task == want.Task && profile.RuntimeLock.Files.Count == want.RuntimeFiles
                && profile.RuntimeLock.Revision.Length == 40,
            $"BuiltIn(\"{name}\"): {profile.Id}, {profile.Task}, {profile.RuntimeLock.Repository}@{profile.RuntimeLock.Revision}, "
            + $"runtime files {string.Join(" ", profile.RuntimeLock.Files.Select(f => f.Path))}");
    }

    var reranker = BertModelProfile.BuiltIn("bge-reranker-v2-m3");
    Check(reranker.RuntimeLock.Repository == "Starlinghq/bge-reranker-v2-m3-onnx"
            && reranker.RuntimeLock.Revision == "047d4622e93d347828457e8a13bccd997fb6a969",
        "bge-reranker-v2-m3 pins Starlinghq/bge-reranker-v2-m3-onnx at 047d4622e93d347828457e8a13bccd997fb6a969");

    try
    {
        BertModelProfile.BuiltIn("kev-0.8b");
        Check(false, "BuiltIn rejects a name outside BuiltInIds");
    }
    catch (ArgumentException)
    {
        Check(true, "BuiltIn rejects a name outside BuiltInIds");
    }

    // An empty snapshot directory: Create must refuse before loading any graph or downloading anything.
    try
    {
        using var embedder = BertTextEmbedder.Create(BertModelProfile.BuiltIn("minilm"), args[0]);
        Check(false, "Create on an empty directory throws ModelAssetMissingException");
    }
    catch (ModelAssetMissingException)
    {
        Check(true, "Create on an empty directory throws ModelAssetMissingException (verification first, no download)");
    }

    Console.WriteLine(failures == 0 ? "RESULT: PASS" : $"RESULT: FAIL ({failures})");
    return failures == 0 ? 0 : 1;
}
