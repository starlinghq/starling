# embeddinggemma-2 `onnx-fp32-cpu` on the P-030 release sets

Run on 2026-10-14 at `10c016c` (main after #26). The text release set is `embeddinggemma-2/release/onnx-fp32-release` (232 holdout rows, each embedded without a prompt and with the query and document prompts: 696 cases). The evaluator verified the protected evidence against Git. Nothing under `protected/` was changed.

| Gate | Set | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- | --- |
| G1 | tokenizer holdout `release/onnx-fp32-release/tokenizer.jsonl` (232 rows) | **passed** | ids and attention mask exact 232/232; tokens, decode and offsets also exact (measured; `offsets_unit: none`) | [`g1-tokenizer-holdout/report.md`](g1-tokenizer-holdout/report.md) |
| G2 | text-encoder release set | **not-run**: the accepted set has no committed `embeddings.jsonl` | Measured on a regenerated reference, not gate evidence (below): stage 1 ids exact 696/696; .NET from ids and from text embedding max abs 5.1e-7 (5e-6), min cosine 0.999999999996 (0.9999999); C# vs Python ORT 1.7e-7 (1e-6); Python ORT token embeddings per case 4.22e-6 relative (1e-4); every padded batch within 2.0e-7 of single (5e-6). | [`g2-measured-not-gate/report.md`](g2-measured-not-gate/report.md) |
| G3 | MIRACL dev, 8 languages, 200 queries, 2,778 passages | **passed** | nDCG@10 **0.8325** (minimum 0.826), MRR@10 **0.8136** (minimum 0.808); recall@10 0.9889. The PyTorch reference scores 0.8329 / 0.8142. The budget names the data+metadata digest (#26). | [`g3-retrieval/report.md`](g3-retrieval/report.md) |

## G2: the reference embeddings are not in the accepted set

`SOURCE.md` of the set says `embeddings.jsonl` (696 rows, 7.0 MB, SHA-256 `c2ffe5e45be11904b7ad0277758c0d8bafc48c6c65dcc87ff932ccbaf574285c`) was left out of Git because of its size. It names the open decision: "commit it with the accepted set ... or keep it outside Git pinned by digest". The evaluator reads fixtures only from `model-pipeline/protected/`, so without that file G2 cannot be evaluated on the release set. I did not build a path for an outside-Git primary reference. That would settle the maintainer's open decision.

What I measured instead:

- I regenerated the reference with the command in `SOURCE.md`, using the proposal's runner `run_embeddinggemma2.py` at `e50cb3d`. That version adds `--batch-ids`, which is not on main. The environment came from `embeddinggemma2/requirements.txt` (torch 2.14.1+cpu, transformers 5.19.0, sentence-transformers 6.1.0, tokenizers 0.23.3), with 4 threads. `tokenizer.jsonl`, `tokenizer-batch.json` and `tokenizer-implementations.json` came out byte-identical to the protected files. **`embeddings.jsonl` did not: `ac0a3a9b64c6c241df721727ac48e7abe9acd1f7b926185c8d487e79def4b47c` against the recorded `c2ffe5e4…`.** The proposal ran with 6 threads, and `reference.json` puts the effect of the thread count at up to 1.5e-7.
- The .NET stages and Python stage 2 then ran against a scratch copy of the release directory, with the regenerated `embeddings.jsonl`, outside the repository. The evaluator ran with `--protected-base none`, so its gates are `not-run` by construction. The numbers in the table come from that run.

## Commands

```sh
REL=model-pipeline/protected/embeddinggemma-2/release/onnx-fp32-release
MAN=model-pipeline/registry/embeddinggemma-2/onnx-fp32-cpu/manifest.yaml
TOOL="dotnet ai-dotnet/tools/Starlings.Fidelity.Tool/bin/Release/net10.0/Starlings.Fidelity.Tool.dll"
$TOOL tokenizer --model embeddinggemma2 --tokenizer <snapshot>/tokenizer.json --fixtures $REL/tokenizer.jsonl --out <run>/g1-holdout
model-pipeline evaluate $MAN --tokenizer-fixtures $REL/tokenizer.jsonl --results <run>/g1-holdout --out <report>
# measured G2 (scratch copy <S> of the release dir with the regenerated embeddings.jsonl)
$TOOL embeddinggemma2 --out <run>/stages --fixtures <S>/$REL --threads 4
<eg2 venv>/bin/python model-pipeline/reference/embeddinggemma2/stage_embeddinggemma2_export.py --model-dir <snapshot> \
  --graph <snapshot>/onnx/embeddinggemma2-text-fp32.onnx --fixtures <S>/$REL --profile embeddinggemma-2/onnx-fp32-cpu --out <run>/stages --intra-op-threads 4
model-pipeline evaluate <S>/model-pipeline/registry/embeddinggemma-2/onnx-fp32-cpu/manifest.yaml --repo-root <S> --protected-base none \
  --fixtures $REL --results <run>/stages --out <report>
```

Digests (outside Git): G1 candidate `e47f7a52d094f1aff9ba38d565be104125aa962ecaf85ce6ba5194f388a157a4`; measured stage results `tokenizer` `4ebf954c…`, `export` `754be496…`, `dotnet` `73ce4b84…`. G3 embeddings (2,978 rows) `2458870f4d1649eb1191193b62919866cbf386e6bbd9c426138da341cfe9ff65`. Wall clock (shared machine, not benchmarks): G3 producer 1,682 s, reference regeneration 602 s, .NET stages 537 s, Python stage 2 466 s, G1 producer 1.5 s.
