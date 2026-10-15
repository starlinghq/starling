# embeddinggemma-2 `onnx-fp32-cpu` on the P-030 release sets

Run on 2026-10-14 at `10c016c` (main after #26). The text release set is `embeddinggemma-2/release/onnx-fp32-release` (232 holdout rows, each embedded without a prompt and with the query and document prompts: 696 cases). The evaluator verified the protected evidence against Git. Nothing under `protected/` was changed.

| Gate | Set | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- | --- |
| G1 | tokenizer holdout `release/onnx-fp32-release/tokenizer.jsonl` (232 rows) | **passed** | ids and attention mask exact 232/232; tokens, decode and offsets also exact (measured; `offsets_unit: none`) | [`g1-tokenizer-holdout/report.md`](g1-tokenizer-holdout/report.md) |
| G2 | text-encoder release set (696 cases) | **passed** (rerun, below) | stage 1 ids exact 696/696; .NET embedding from ids and from text max abs **5.1e-7** (5e-6), min cosine 0.999999999996 (0.9999999); Python ORT embedding 5.1e-7 (5e-6); C# vs Python ORT 1.7e-7 (1e-6); token embeddings per case 4.22e-6 relative (1e-4); padded batches within 2.0e-7 of single (5e-6). The first run was not-run (no reference; measured only, [`g2-measured-not-gate/report.md`](g2-measured-not-gate/report.md)) | [`g2-release/report.md`](g2-release/report.md) |
| G3 | MIRACL dev, 8 languages, 200 queries, 2,778 passages | **passed** | nDCG@10 **0.8325** (minimum 0.826), MRR@10 **0.8136** (minimum 0.808); recall@10 0.9889. The PyTorch reference scores 0.8329 / 0.8142. The budget names the data+metadata digest (#26). | [`g3-retrieval/report.md`](g3-retrieval/report.md) |

## G2 rerun: the outside-Git reference bound by its digest

#43 recorded the digest of `embeddings.jsonl` in the protected `reference.json` (`embeddings.sha256`, `committed: false`), and #45 added `model-pipeline evaluate --reference-embeddings <file>`. The evaluator accepts the file only if its SHA-256 equals that digest. It reads `reference.json` as a Git-verified protected fixture, and it refuses a path inside the repository outside `model-pipeline/protected/`.

- **Reference.** I regenerated it with the command in `SOURCE.md`, using the runner pinned in the set's `environment.json` (`run_embeddinggemma2.py` at `4b90bc9a0e6d16160be27484583ec17c861e915d`, with `model_pipeline` from the same commit). It ran in the `embeddinggemma2/requirements.txt` environment: Python 3.11.16, torch 2.14.1+cpu, transformers 5.19.0, sentence-transformers 6.1.0, tokenizers 0.23.3 and numpy 2.4.6. It used the full `google/embeddinggemma-2@914f7f89` checkpoint at the recorded `default_threads` (`OMP_NUM_THREADS=6`), and it wrote to `~/.cache/starling-release-refs/embeddinggemma-2/`, outside the repository. The output `embeddings.jsonl` has 696 rows, SHA-256 **`c2ffe5e45be11904b7ad0277758c0d8bafc48c6c65dcc87ff932ccbaf574285c`**, equal to `reference.json` `embeddings.sha256`. `tokenizer.jsonl`, `tokenizer-batch.json` and `tokenizer-implementations.json` came out byte-identical to the protected files. The regenerated `reference.json` and `environment.json` differ, as expected, because they hold timings and machine details. Wall clock: 352 s.
- **Producers.** They ran at `303d7ca` (main after #45) and took their fixtures from a directory outside the repository holding copies of the protected `tokenizer.jsonl` and `tokenizer-batch.json` plus the regenerated `embeddings.jsonl`. Both producers read the case ids from that file. Runs: `Starlings.Fidelity.Tool embeddinggemma2 --threads 6` (205 s), then `stage_embeddinggemma2_export.py --intra-op-threads 6` (228 s).
- **Evaluation.** I evaluated against the real protected set with the real manifest and `--protected-base auto`, so the protected evidence is `verified`. **Exit 0, first divergence none, G1 passed, G2 passed.** Candidate stage results (outside Git): `tokenizer` `7b6f0db0…`, `export` `fdf0dc5c…`, `dotnet` `d54d1f6c…` (full digests in `g2-release/report.json`).

```sh
model-pipeline evaluate model-pipeline/registry/embeddinggemma-2/onnx-fp32-cpu/manifest.yaml \
  --fixtures model-pipeline/protected/embeddinggemma-2/release/onnx-fp32-release \
  --reference-embeddings ~/.cache/starling-release-refs/embeddinggemma-2/embeddings.jsonl \
  --results <run>/stages --out <report> --require-gates G1,G2
```

## First run: the reference embeddings are not in the accepted set

`SOURCE.md` of the set says `embeddings.jsonl` (696 rows, 7.0 MB, SHA-256 `c2ffe5e45be11904b7ad0277758c0d8bafc48c6c65dcc87ff932ccbaf574285c`) was left out of Git because of its size. It names the open decision: "commit it with the accepted set ... or keep it outside Git pinned by digest". The evaluator reads fixtures only from `model-pipeline/protected/`, so without that file G2 cannot be evaluated on the release set. I did not build a path for an outside-Git primary reference. That would settle the maintainer's open decision.

What I measured instead:

- I regenerated the reference with the command in `SOURCE.md`, using the proposal's runner `run_embeddinggemma2.py` at `e50cb3d` (it adds `--batch-ids`, which is not on main) in an environment from `embeddinggemma2/requirements.txt` (torch 2.14.1+cpu, transformers 5.19.0, sentence-transformers 6.1.0, tokenizers 0.23.3). At 4 threads (the limit while the soak measurements ran), `tokenizer.jsonl`, `tokenizer-batch.json` and `tokenizer-implementations.json` came out byte-identical to the protected files, but `embeddings.jsonl` did not (`ac0a3a9b…`). At 6 threads, the proposal's `torch_threads`, after the soaks finished, **`embeddings.jsonl` is byte-identical to the digest `SOURCE.md` records: `c2ffe5e45be11904b7ad0277758c0d8bafc48c6c65dcc87ff932ccbaf574285c`.** The accepted reference is therefore reproducible from the committed sources, which bears on the maintainer's open decision 2 (commit the file, or pin it by digest outside Git).
- The .NET stages and Python stage 2 then ran on a scratch copy of the release directory outside the repository, using the regenerated `embeddings.jsonl` with the pinned digest. The evaluator ran with `--protected-base none`, so its gates are `not-run` by construction. The numbers in the table come from that run, which reports no failure. Recording G2 as passed needs the file in `model-pipeline/protected/` (a `protected-change`) or a maintainer rule for digest-pinned fixtures kept outside Git.

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

Digests (outside Git): G1 candidate `e47f7a52d094f1aff9ba38d565be104125aa962ecaf85ce6ba5194f388a157a4`; measured stage results `tokenizer` `4ebf954c…`, `export` `754be496…`, `dotnet` `73ce4b84…`. G3 embeddings (2,978 rows) `2458870f4d1649eb1191193b62919866cbf386e6bbd9c426138da341cfe9ff65`. Wall clock (shared machine, not benchmarks): G3 producer 1,682 s, reference regeneration 602 s (4 threads), .NET stages 537 s, Python stage 2 466 s, G1 producer 1.5 s.
