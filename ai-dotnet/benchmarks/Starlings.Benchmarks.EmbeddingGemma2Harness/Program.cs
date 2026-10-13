using Starlings.Benchmarks;

// EmbeddingGemma 2 text and image harness (G5, plan section 11). See docs/benchmarks/PROTOCOL.md and ai-dotnet/benchmarks/README.md.
return HarnessHost.Run(args, new EmbeddingGemma2Driver());
