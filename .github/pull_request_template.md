## Summary

<!-- What changed and why. Link the issue (P-0xx if from the backlog). -->

## Evidence

<!-- Tests, conformance or parity reports, benchmark output. Name the artifact IDs if a model bundle is involved. -->

## Checklist

- [ ] Checks for the touched directories pass locally.
- [ ] No protected fixture, tolerance, or release evaluation set changed. If one did, the PR carries the `protected-change` label, explains why, and carries only that change.
- [ ] No library under `ai-dotnet/src/` references Microsoft.ML.Tokenizers or another tokenizer, directly or transitively.
- [ ] Changed package references come with the updated `packages.lock.json` files.
- [ ] Unsupported features are listed as unsupported, not omitted.
- [ ] No model weights or large assets committed.
