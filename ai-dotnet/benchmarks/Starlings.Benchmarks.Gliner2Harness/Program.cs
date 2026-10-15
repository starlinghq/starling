using Starlings.Benchmarks;

// GLiNER2 entity-extraction harness (G5, plan section 11). See docs/benchmarks/PROTOCOL.md and ai-dotnet/benchmarks/README.md.
// 'dump-inputs --out <file>' writes the model inputs Prepare builds for every workload request (the Python
// same-backend runner, model-pipeline/reference/bench/bench_gliner2_matched.py, times the inference stage on them).
return ToolMain.Run(() => args.Length > 0 && args[0] == "dump-inputs" ? Gliner2InputDump.Run(args[1..]) : HarnessHost.Run(args, new Gliner2Driver()));
