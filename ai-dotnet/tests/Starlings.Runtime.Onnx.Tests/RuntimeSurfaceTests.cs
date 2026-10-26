using Microsoft.ML.OnnxRuntime;
using Starlings.TestSupport;

namespace Starlings.Runtime.Onnx.Tests;

/// <summary>
/// Provider probe, typed CUDA options, logging, execution mode, graph optimization, I/O binding, warmup and the
/// exception base. Everything here runs on the CPU native package; the CUDA paths are checked only up to what a CPU-only
/// build can show (validation, option strings, the clear refusal).
/// </summary>
public class RuntimeSurfaceTests
{
    private static readonly float[] X = [1, 2, 3, 4, 5, 6, 7, 8];
    private static readonly float[] Y = [10, 20, 30, 40, 50, 60, 70, 80];
    private static readonly float[] Z = [11, 22, 33, 44, 55, 66, 77, 88];

    [Fact]
    public void PackageExceptionsShareOneBase()
    {
        Assert.True(typeof(OnnxSessionException).IsAssignableFrom(typeof(OnnxRuntimeUnavailableException)));
        Assert.True(typeof(OnnxSessionException).IsAssignableFrom(typeof(OnnxProviderUnavailableException)));
        Assert.True(typeof(OnnxSessionException).IsAssignableFrom(typeof(OnnxModelIntegrityException)));

        OnnxSessionException caught = Assert.ThrowsAny<OnnxSessionException>(
            () => OnnxModelSession.Create(TestModels.Add, new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cuda }));
        Assert.IsType<OnnxProviderUnavailableException>(caught);
    }

    [Fact]
    public void ProviderProbeReportsCpuAndRefusesCudaOnTheCpuPackage()
    {
        Assert.True(OnnxRuntimeInfo.TryInitializeProvider(OnnxExecutionProvider.Cpu, 0, out string? cpuProblem));
        Assert.Null(cpuProblem);

        // This test project references the CPU native package, which has no CUDA execution provider.
        Assert.False(OnnxRuntimeInfo.TryInitializeProvider(OnnxExecutionProvider.Cuda, 0, out string? cudaProblem));
        Assert.Contains("Microsoft.ML.OnnxRuntime.Gpu.Linux", cudaProblem, StringComparison.Ordinal);

        var withOptions = new OnnxSessionOptions
        {
            Provider = OnnxExecutionProvider.Cuda,
            Cuda = new OnnxCudaProviderOptions { GpuMemoryLimitBytes = 1L << 30, EnableCudaGraph = true },
        };
        Assert.False(OnnxRuntimeInfo.TryInitializeProvider(withOptions, out string? problem));
        Assert.Equal(cudaProblem, problem);
    }

    [Fact]
    public void ProviderProbeRejectsInvalidArguments()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => OnnxRuntimeInfo.TryInitializeProvider(OnnxExecutionProvider.Cuda, -1, out _));
        Assert.Throws<ArgumentOutOfRangeException>(() => OnnxRuntimeInfo.TryInitializeProvider((OnnxExecutionProvider)7, 0, out _));
        var invalid = new OnnxSessionOptions { Provider = OnnxExecutionProvider.Cuda, Cuda = new OnnxCudaProviderOptions { GpuMemoryLimitBytes = 0 } };
        Assert.Throws<ArgumentOutOfRangeException>(() => OnnxRuntimeInfo.TryInitializeProvider(invalid, out _));
        Assert.Throws<ArgumentNullException>(() => OnnxRuntimeInfo.TryInitializeProvider(null!, out _));
    }

    [Fact]
    public void CudaOptionsMapToOnnxRuntimeOptionStrings()
    {
        Assert.Equal(
            new Dictionary<string, string> { ["device_id"] = "0", ["use_tf32"] = "0" },
            OnnxCudaProviderOptions.ToProviderOptions(0, null));

        var options = new OnnxCudaProviderOptions
        {
            GpuMemoryLimitBytes = 2_147_483_648,
            ArenaExtendStrategy = OnnxArenaExtendStrategy.SameAsRequested,
            CudnnConvolutionAlgorithmSearch = OnnxCudnnConvolutionAlgorithmSearch.Heuristic,
            EnableCudaGraph = true,
        };
        Assert.Equal(
            new Dictionary<string, string>
            {
                ["device_id"] = "3",
                ["use_tf32"] = "0",
                ["gpu_mem_limit"] = "2147483648",
                ["arena_extend_strategy"] = "kSameAsRequested",
                ["cudnn_conv_algo_search"] = "HEURISTIC",
                ["enable_cuda_graph"] = "1",
            },
            OnnxCudaProviderOptions.ToProviderOptions(3, options));

        var other = OnnxCudaProviderOptions.ToProviderOptions(0, new OnnxCudaProviderOptions
        {
            ArenaExtendStrategy = OnnxArenaExtendStrategy.NextPowerOfTwo,
            CudnnConvolutionAlgorithmSearch = OnnxCudnnConvolutionAlgorithmSearch.Default,
            EnableCudaGraph = false,
        });
        Assert.Equal("kNextPowerOfTwo", other["arena_extend_strategy"]);
        Assert.Equal("DEFAULT", other["cudnn_conv_algo_search"]);
        Assert.Equal("0", other["enable_cuda_graph"]);
        Assert.Equal("EXHAUSTIVE", OnnxCudaProviderOptions.ToProviderOptions(0, new OnnxCudaProviderOptions
        {
            CudnnConvolutionAlgorithmSearch = OnnxCudnnConvolutionAlgorithmSearch.Exhaustive,
        })["cudnn_conv_algo_search"]);
    }

    [Theory]
    [InlineData(0L, null, null)]
    [InlineData(-5L, null, null)]
    [InlineData(null, 9, null)]
    [InlineData(null, null, 9)]
    public void InvalidCudaOptionsAreRejectedOnCpuBuilds(long? limit, int? arena, int? search)
    {
        var options = new OnnxSessionOptions
        {
            // Validated even when the provider is CPU, so a configuration error shows on a machine without a GPU.
            Cuda = new OnnxCudaProviderOptions
            {
                GpuMemoryLimitBytes = limit,
                ArenaExtendStrategy = (OnnxArenaExtendStrategy?)arena,
                CudnnConvolutionAlgorithmSearch = (OnnxCudnnConvolutionAlgorithmSearch?)search,
            },
        };
        Assert.Throws<ArgumentOutOfRangeException>(() => OnnxModelSession.Create(TestModels.Add, options));
    }

    [Fact]
    public void CudaOptionsAreIgnoredForCpuSessions()
    {
        var options = new OnnxSessionOptions { Cuda = new OnnxCudaProviderOptions { GpuMemoryLimitBytes = 1024, EnableCudaGraph = true } };
        using var session = OnnxModelSession.Create(TestModels.Add, options);
        Assert.Empty(session.Diagnostics.ProviderOptions);
        AssertAdds(session);
    }

    [Fact]
    public void LoggingSeveritiesMapBothWays()
    {
        foreach (var severity in Enum.GetValues<OnnxLogSeverity>())
        {
            Assert.Equal(severity, OnnxLogging.FromOrt(OnnxLogging.ToOrt(severity)));
        }

        Assert.Equal(OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR, OnnxLogging.ToOrt(OnnxLogSeverity.Error));
        Assert.Equal(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE, OnnxLogging.ToOrt(OnnxLogSeverity.Verbose));
    }

    [Fact]
    public void LoggingAdapterDeliversMessagesAndDropsSinkExceptions()
    {
        var received = new List<OnnxRuntimeLogMessage>();
        var function = OnnxLogging.Adapt(received.Add);
        function(IntPtr.Zero, OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, "onnxruntime", "session-a", "graph.cc:12", "a message");
        var message = Assert.Single(received);
        Assert.Equal(new OnnxRuntimeLogMessage(OnnxLogSeverity.Warning, "onnxruntime", "session-a", "graph.cc:12", "a message"), message);

        var throwing = OnnxLogging.Adapt(_ => throw new InvalidOperationException("sink failure"));
        throwing(IntPtr.Zero, OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR, "c", "id", "loc", "m");
    }

    [Fact]
    public void LoggingCannotBeConfiguredOnceTheEnvironmentExists()
    {
        // Any earlier call creates the process-wide environment; this one makes sure of it.
        _ = OnnxRuntimeInfo.NativeVersion;
        var ex = Assert.Throws<InvalidOperationException>(() => OnnxRuntimeInfo.ConfigureLogging(_ => { }));
        Assert.Contains("ConfigureLogging", ex.Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentNullException>(() => OnnxRuntimeInfo.ConfigureLogging(null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => OnnxRuntimeInfo.ConfigureLogging(_ => { }, (OnnxLogSeverity)42));
    }

    [Fact]
    public void SessionReportsLoggingExecutionModeAndOptimizationLevel()
    {
        using (var defaults = OnnxModelSession.Create(TestModels.Add))
        {
            var d = defaults.Diagnostics;
            Assert.Equal(OnnxExecutionMode.Sequential, d.ExecutionMode);
            Assert.Equal("ORT_ENABLE_ALL", d.GraphOptimizationLevel);
            Assert.Equal(OnnxLogSeverity.Error, d.LogSeverity);
            Assert.Null(d.LogId);
            Assert.Null(d.OptimizedModelFilePath);
        }

        var options = new OnnxSessionOptions
        {
            LogId = "runtime-surface",
            LogSeverity = OnnxLogSeverity.Fatal,
            GraphOptimizationLevel = OnnxGraphOptimizationLevel.None,
        };
        using var session = OnnxModelSession.Create(TestModels.Add, options);
        Assert.Equal("runtime-surface", session.Diagnostics.LogId);
        Assert.Equal(OnnxLogSeverity.Fatal, session.Diagnostics.LogSeverity);
        Assert.Equal("ORT_DISABLE_ALL", session.Diagnostics.GraphOptimizationLevel);
        AssertAdds(session);
    }

    [Theory]
    [InlineData(OnnxGraphOptimizationLevel.Basic, "ORT_ENABLE_BASIC")]
    [InlineData(OnnxGraphOptimizationLevel.Extended, "ORT_ENABLE_EXTENDED")]
    [InlineData(OnnxGraphOptimizationLevel.Layout, "ORT_ENABLE_LAYOUT")]
    public void EveryOptimizationLevelReachesOnnxRuntime(OnnxGraphOptimizationLevel level, string expected)
    {
        using var session = OnnxModelSession.Create(TestModels.Add, new OnnxSessionOptions { GraphOptimizationLevel = level });
        Assert.Equal(expected, session.Diagnostics.GraphOptimizationLevel);
        AssertAdds(session);
    }

    [Fact]
    public void ParallelExecutionUsesInterOpThreadsAndGivesTheSameOutputs()
    {
        var options = new OnnxSessionOptions { ExecutionMode = OnnxExecutionMode.Parallel, InterOpNumThreads = 2 };
        using var session = OnnxModelSession.Create(TestModels.Add, options);
        Assert.Equal(OnnxExecutionMode.Parallel, session.Diagnostics.ExecutionMode);
        Assert.Equal(2, session.Diagnostics.InterOpNumThreads);
        AssertAdds(session);
    }

    [Fact]
    public void InvalidNewOptionsAreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => OnnxModelSession.Create(TestModels.Add, new OnnxSessionOptions { ExecutionMode = (OnnxExecutionMode)5 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => OnnxModelSession.Create(TestModels.Add, new OnnxSessionOptions { GraphOptimizationLevel = (OnnxGraphOptimizationLevel)9 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => OnnxModelSession.Create(TestModels.Add, new OnnxSessionOptions { LogSeverity = (OnnxLogSeverity)9 }));
        Assert.Throws<ArgumentException>(() => OnnxModelSession.Create(TestModels.Add, new OnnxSessionOptions { OptimizedModelFilePath = string.Empty }));
    }

    [Fact]
    public void OptimizedModelIsWrittenAndLoadsAgain()
    {
        string directory = Path.Combine(Path.GetTempPath(), $"starlings-optimized-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            string model = Path.Combine(directory, "add.onnx");
            string optimized = Path.Combine(directory, "add.optimized.onnx");
            File.WriteAllBytes(model, TestModels.Add);

            using (var session = OnnxModelSession.Create(model, new OnnxSessionOptions { OptimizedModelFilePath = optimized }))
            {
                Assert.Equal(optimized, session.Diagnostics.OptimizedModelFilePath);
            }

            Assert.True(new FileInfo(optimized).Length > 0);
            using (var reloaded = OnnxModelSession.Create(optimized, new OnnxSessionOptions { GraphOptimizationLevel = OnnxGraphOptimizationLevel.None }))
            {
                AssertAdds(reloaded);
            }

            Assert.Throws<ArgumentException>(() => OnnxModelSession.Create(model, new OnnxSessionOptions { OptimizedModelFilePath = model }));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void DictionaryRunMatchesTheListRun()
    {
        using var session = OnnxModelSession.Create(TestModels.Add);
        using var xv = OrtValue.CreateTensorValueFromMemory(X, [2, 4]);
        using var yv = OrtValue.CreateTensorValueFromMemory(Y, [2, 4]);
        var inputs = new Dictionary<string, OrtValue> { ["y"] = yv, ["x"] = xv };

        using (var all = session.Run(inputs, cancellationToken: TestContext.Current.CancellationToken))
        {
            Assert.Equal(Z, Assert.Single(all).GetTensorDataAsSpan<float>().ToArray());
        }

        using (var named = session.Run(inputs, ["z"], TestContext.Current.CancellationToken))
        {
            Assert.Equal(Z, Assert.Single(named).GetTensorDataAsSpan<float>().ToArray());
        }

        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        Assert.Throws<OperationCanceledException>(() => session.Run(inputs, null, cancelled.Token));
    }

    [Fact]
    public void IoBindingRunsWithBoundInputsAndDeviceOutputs()
    {
        using var session = OnnxModelSession.Create(TestModels.Add);
        using var xv = OrtValue.CreateTensorValueFromMemory(X, [2, 4]);
        using var yv = OrtValue.CreateTensorValueFromMemory(Y, [2, 4]);
        using var memory = session.CreateDeviceMemoryInfo();
        Assert.Equal("Cpu", memory.Name);

        using var binding = session.CreateIoBinding();
        binding.BindInput("x", xv);
        binding.BindInput("y", yv);
        binding.BindOutputToDevice("z", memory);

        for (int i = 0; i < 2; i++)
        {
            session.RunWithBinding(binding, TestContext.Current.CancellationToken);
            using var outputs = binding.GetOutputValues();
            Assert.Equal(Z, Assert.Single(outputs).GetTensorDataAsSpan<float>().ToArray());
        }

        // Preallocated output bound in place.
        float[] z = new float[8];
        using var zv = OrtValue.CreateTensorValueFromMemory(z, [2, 4]);
        binding.ClearBoundOutputs();
        binding.BindOutput("z", zv);
        session.RunWithBinding(binding, TestContext.Current.CancellationToken);
        Assert.Equal(Z, z);

        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        Assert.Throws<OperationCanceledException>(() => session.RunWithBinding(binding, cancelled.Token));
    }

    [Fact]
    public void IoBindingCallsFailAfterDispose()
    {
        var session = OnnxModelSession.Create(TestModels.Add);
        var binding = session.CreateIoBinding();
        binding.Dispose();
        session.Dispose();
        Assert.Throws<ObjectDisposedException>(() => session.CreateIoBinding());
        Assert.Throws<ObjectDisposedException>(() => session.RunWithBinding(binding, TestContext.Current.CancellationToken));
        Assert.Throws<ObjectDisposedException>(() => session.WarmupWithZeroInputs(cancellationToken: TestContext.Current.CancellationToken));
    }

    [Fact]
    public void WarmupRunsAndReportsEachDuration()
    {
        using var session = OnnxModelSession.Create(TestModels.Add);
        using var xv = OrtValue.CreateTensorValueFromMemory(X, [2, 4]);
        using var yv = OrtValue.CreateTensorValueFromMemory(Y, [2, 4]);

        var durations = session.Warmup(new Dictionary<string, OrtValue> { ["x"] = xv, ["y"] = yv }, 3, TestContext.Current.CancellationToken);

        Assert.Equal(3, durations.Count);
        Assert.All(durations, d => Assert.True(d >= TimeSpan.Zero));
        Assert.Throws<ArgumentOutOfRangeException>(() => session.Warmup(new Dictionary<string, OrtValue>(), 0, TestContext.Current.CancellationToken));
    }

    [Fact]
    public void WarmupWithZeroInputsBuildsTensorsFromTheSignature()
    {
        using var session = OnnxModelSession.Create(TestModels.FakeE5);
        var durations = session.WarmupWithZeroInputs(
            new Dictionary<string, long> { ["batch_size"] = 3, ["sequence_length"] = 7 },
            2,
            TestContext.Current.CancellationToken);
        Assert.Equal(2, durations.Count);

        using var add = OnnxModelSession.Create(TestModels.Add);
        Assert.Single(add.WarmupWithZeroInputs(cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentOutOfRangeException>(() => add.WarmupWithZeroInputs(new Dictionary<string, long> { ["batch"] = 0 }, cancellationToken: TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentOutOfRangeException>(() => add.WarmupWithZeroInputs(runs: 0, cancellationToken: TestContext.Current.CancellationToken));
    }

    private static void AssertAdds(OnnxModelSession session)
    {
        using var xv = OrtValue.CreateTensorValueFromMemory(X, [2, 4]);
        using var yv = OrtValue.CreateTensorValueFromMemory(Y, [2, 4]);
        using var outputs = session.Run(["x", "y"], [xv, yv], ["z"], TestContext.Current.CancellationToken);
        Assert.Equal(Z, outputs[0].GetTensorDataAsSpan<float>().ToArray());
    }
}
