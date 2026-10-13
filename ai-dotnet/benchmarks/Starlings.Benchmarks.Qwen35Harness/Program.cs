using Starlings.Benchmarks;

// Qwen3.5 generation harness (G5, plan section 11). See docs/benchmarks/PROTOCOL.md and ai-dotnet/benchmarks/README.md.
return HarnessHost.Run(args, new Qwen35Driver());
