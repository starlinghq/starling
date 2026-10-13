using BenchmarkDotNet.Running;

// BenchmarkDotNet micro-benchmarks (P-008). Examples:
//   dotnet run -c Release --project ai-dotnet/benchmarks/Starlings.Benchmarks.Micro -- --filter '*Tokenizer*'
//   dotnet run -c Release --project ai-dotnet/benchmarks/Starlings.Benchmarks.Micro -- --filter '*E5*'
// The e5 benchmarks need the model (STARLING_E5_SNAPSHOT or the acquisition cache); see ai-dotnet/benchmarks/README.md.
BenchmarkSwitcher.FromAssembly(typeof(Starlings.Benchmarks.TokenizerBenchmarks).Assembly).Run(args);
