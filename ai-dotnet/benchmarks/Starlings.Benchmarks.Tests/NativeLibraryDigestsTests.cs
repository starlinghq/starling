using System.Text.Json.Nodes;

namespace Starlings.Benchmarks.Tests;

public sealed class NativeLibraryDigestsTests
{
    private const string Maps = """
        55d0c0000000-55d0c0001000 r--p 00000000 103:02 1 /usr/share/dotnet/dotnet
        7f0000000000-7f0000100000 r--p 00000000 103:02 2 /opt/app/runtimes/linux-x64/native/libonnxruntime.so
        7f0000100000-7f0000900000 r-xp 00100000 103:02 2 /opt/app/runtimes/linux-x64/native/libonnxruntime.so
        7f0000900000-7f0000a00000 r--p 00000000 103:02 3 /opt/app/runtimes/linux-x64/native/libonnxruntime-genai.so
        7f0000a00000-7f0000b00000 r--p 00000000 103:02 4 /venv/site-packages/onnxruntime/capi/libonnxruntime.so.1.30.0
        7f0000b00000-7f0000c00000 r--p 00000000 103:02 5 /venv/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-313-x86_64-linux-gnu.so
        7f0000c00000-7f0000d00000 rw-s 00000000 103:02 6 /home/u/.cache/Microsoft/DeveloperTools/.onnxruntime/onnxruntime-genai.db-shm
        7f0000d00000-7f0000e00000 r--p 00000000 103:02 7 /usr/lib/libcudart.so.13
        7f0000e00000-7f0000f00000 r--p 00000000 103:02 8 /opt/my app/libonnxruntime_providers_shared.so
        7f0000f00000-7f0001000000 rw-p 00000000 00:00 0
        7ffc00000000-7ffc00021000 rw-p 00000000 00:00 0 [stack]
        """;

    [Fact]
    public void PathsAreTheOnnxRuntimeAndGenAiSharedObjectsOnly()
    {
        Assert.Equal(
            [
                "/opt/app/runtimes/linux-x64/native/libonnxruntime-genai.so",
                "/opt/app/runtimes/linux-x64/native/libonnxruntime.so",
                "/opt/my app/libonnxruntime_providers_shared.so",
                "/venv/site-packages/onnxruntime/capi/libonnxruntime.so.1.30.0",
                "/venv/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-313-x86_64-linux-gnu.so",
            ],
            NativeLibraryDigests.Paths(Maps));
    }

    [Theory]
    [InlineData("/a/libonnxruntime.so", true)]
    [InlineData("/a/libonnxruntime.so.1.30.0", true)]
    [InlineData("/a/libonnxruntime-genai.so", true)]
    [InlineData("/a/libonnxruntime_providers_cuda.so", true)]
    [InlineData("/a/onnxruntime-genai.db-shm", false)]
    [InlineData("/a/libcudart.so.13", false)]
    [InlineData("/a/Microsoft.ML.OnnxRuntime.dll", false)]
    public void RecognizesOnnxRuntimeLibraryNames(string path, bool expected) =>
        Assert.Equal(expected, NativeLibraryDigests.IsOnnxRuntimeLibrary(path));

    [Fact]
    public void EveryLibraryCarriesItsDigest()
    {
        var json = NativeLibraryDigests.ToJson(["/x/libonnxruntime.so", "/x/libonnxruntime-genai.so"], p => p.EndsWith("genai.so", StringComparison.Ordinal) ? null : "ab");

        Assert.Equal(2, json.Count);
        Assert.Equal("/x/libonnxruntime.so", json[0]!["path"]!.GetValue<string>());
        Assert.Equal("ab", json[0]!["sha256"]!.GetValue<string>());
        Assert.Null(json[1]!["sha256"]);
    }

    [Fact]
    public void CollectDigestsTheFilesThisProcessMapped()
    {
        JsonNode value = NativeLibraryDigests.Collect();
        if (!OperatingSystem.IsLinux())
        {
            Assert.Equal(NativeLibraryDigests.Unavailable, value.GetValue<string>());
            return;
        }

        // This test process maps no ONNX Runtime library; the value is still the array form.
        Assert.IsType<JsonArray>(value);
    }

    [Fact]
    public void DigestIsTheFileSha256()
    {
        string dir = Directory.CreateTempSubdirectory("starlings-native-").FullName;
        try
        {
            string path = Path.Combine(dir, "libonnxruntime.so");
            File.WriteAllBytes(path, "abc"u8.ToArray());
            var maps = $"7f0000000000-7f0000001000 r--p 00000000 103:02 9 {path}\n";
            var json = NativeLibraryDigests.ToJson(NativeLibraryDigests.Paths(maps), NativeLibraryDigests.Sha256OrNull);

            Assert.Equal("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", json.Single()!["sha256"]!.GetValue<string>());
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
