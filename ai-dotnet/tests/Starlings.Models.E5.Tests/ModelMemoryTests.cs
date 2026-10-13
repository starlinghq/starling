using System.Diagnostics;
using Starlings.Runtime.Onnx;
using Starlings.TestSupport;

namespace Starlings.Models.E5.Tests;

/// <summary>
/// Tests that measure process-wide memory. They run in a collection with parallelization disabled: xunit runs such a
/// collection on its own after the parallel ones, so other classes (model fixtures, tokenizer loads) cannot move the
/// process's private memory during the measurement.
/// </summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class ProcessMemoryTestGroup
{
    public const string Name = "process memory (not parallel)";
}

/// <summary>Profile <c>multilingual-e5-small/onnx-fp32-cpu</c>: repeated load and dispose of the real model.</summary>
[Collection(ProcessMemoryTestGroup.Name)]
public class ModelMemoryTests(ITestOutputHelper output)
{
    [Fact]
    public void RepeatedLoadAndDisposeReleasesTheModel()
    {
        var (path, reason) = E5TestAssets.FindModel();
        Assert.SkipWhen(path is null, reason);
        var options = new E5EncoderOptions { Session = new OnnxSessionOptions { ComputeModelSha256 = false } };
        var input = new EncoderInput(E5Fixtures.Cases[2].InputIds);
        var samples = new List<double>();
        for (int i = 0; i < 6; i++)
        {
            using (var encoder = E5Encoder.Create(path!, options))
            {
                encoder.Embed(input, TestContext.Current.CancellationToken);
            }

            GC.Collect();
            GC.WaitForPendingFinalizers();
            using var process = Process.GetCurrentProcess();
            samples.Add(process.PrivateMemorySize64 / 1048576.0);
        }

        output.WriteLine("private memory (MB) after each load/embed/dispose cycle: " + string.Join(", ", samples.Select(s => s.ToString("F0", System.Globalization.CultureInfo.InvariantCulture))));

        // Each load maps a 470 MB model; a leaked session would add hundreds of MB per cycle.
        Assert.True(samples[^1] - samples[1] < 200, "private memory grew across load/dispose cycles");
    }
}
