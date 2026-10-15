using System.Diagnostics;

namespace Starlings.Soak.Tool.Tests;

/// <summary>
/// The shared entry-point wrapper (ai-dotnet/tools/Shared/ToolMain.cs, linked into every tool and harness): an exception
/// that escapes a tool's body ends the process with "error:" and exit code 70 instead of an abort and a core dump.
/// </summary>
public sealed class ToolMainTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("starlings-toolmain-tests-").FullName;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    [Fact]
    public void TheBodysExitCodePassesThroughUnchanged()
    {
        var stderr = new StringWriter();
        Assert.Equal(0, ToolMain.Run(() => 0, stderr));
        Assert.Equal(3, ToolMain.Run(() => 3, stderr));
        Assert.Empty(stderr.ToString());
    }

    [Fact]
    public void AnExceptionGivesTheDocumentedExitCodeAndTheFullExceptionOnStderr()
    {
        var stderr = new StringWriter();
        int exit = ToolMain.Run(() => throw new InvalidOperationException("typed failure", new FormatException("inner cause")), stderr);

        Assert.Equal(ToolMain.UnhandledErrorExitCode, exit);
        Assert.Equal(70, exit);
        string text = stderr.ToString();
        Assert.StartsWith("error: System.InvalidOperationException: typed failure", text, StringComparison.Ordinal);
        Assert.Contains("inner cause", text, StringComparison.Ordinal);
        Assert.Contains(" at ", text, StringComparison.Ordinal); // the stack trace
    }

    [Fact]
    public async Task AnAsyncBodyBehavesTheSame()
    {
        var stderr = new StringWriter();
        Assert.Equal(4, await ToolMain.RunAsync(() => Task.FromResult(4), stderr));
        int exit = await ToolMain.RunAsync(async () =>
        {
            await Task.Yield();
            throw new IOException("disk gone");
        }, stderr);
        Assert.Equal(ToolMain.UnhandledErrorExitCode, exit);
        Assert.StartsWith("error: System.IO.IOException: disk gone", stderr.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task OutOfMemoryIsRethrownNotReported()
    {
        var stderr = new StringWriter();
#pragma warning disable CA2201 // Reserved exception types: thrown here only to stand in for the runtime's own.
        Assert.Throws<OutOfMemoryException>(() => ToolMain.Run(() => throw new OutOfMemoryException(), stderr));
        await Assert.ThrowsAsync<OutOfMemoryException>(() => ToolMain.RunAsync(() => throw new OutOfMemoryException(), stderr));
        Assert.Empty(stderr.ToString());
        Assert.True(ToolMain.IsFatal(new StackOverflowException()));
#pragma warning restore CA2201
        Assert.False(ToolMain.IsFatal(new ArgumentException("not fatal")));
    }

    [Fact]
    public void ATypedErrorInARealToolProcessExitsCleanlyWithoutASignal()
    {
        // Reanalysing a malformed raw file throws a JSON reader exception inside the tool; before the wrapper it escaped Main,
        // and .NET aborted the process (SIGABRT, exit status 134 from a shell, and a core dump).
        string raw = Path.Combine(_dir, "broken.json");
        File.WriteAllText(raw, "{ this is not json");

        var (exit, stderr) = RunSoakTool("reanalyze", raw);

        Assert.Equal(ToolMain.UnhandledErrorExitCode, exit);
        Assert.StartsWith("error: System.Text.Json.", stderr, StringComparison.Ordinal); // JsonReaderException, with its stack
        Assert.Contains(" at ", stderr, StringComparison.Ordinal);
        Assert.DoesNotContain("Unhandled exception", stderr, StringComparison.Ordinal);
        // .NET reports a child killed by a signal as 128 + the signal number (SIGABRT: 134); 70 is a normal exit.
        Assert.False(exit > 128, $"killed by signal {exit - 128}");
    }

    [Fact]
    public void AUsageErrorKeepsItsOwnExitCode()
    {
        var (exit, _) = RunSoakTool();
        Assert.Equal(2, exit);
    }

    private static (int ExitCode, string Stderr) RunSoakTool(params string[] args)
    {
        string dll = typeof(SoakRunner).Assembly.Location;
        string host = Environment.GetEnvironmentVariable("DOTNET_HOST_PATH") is { Length: > 0 } h ? h : "dotnet";
        var info = new ProcessStartInfo(host)
        {
            UseShellExecute = false,
            RedirectStandardError = true,
            RedirectStandardOutput = true,
        };
        info.ArgumentList.Add(dll);
        foreach (string a in args)
        {
            info.ArgumentList.Add(a);
        }

        using var process = Process.Start(info)!;
        var stdout = process.StandardOutput.ReadToEndAsync();
        string stderr = process.StandardError.ReadToEnd();
        Assert.True(process.WaitForExit(TimeSpan.FromMinutes(2)), "the tool did not exit");
        _ = stdout.Result;
        return (process.ExitCode, stderr);
    }
}
