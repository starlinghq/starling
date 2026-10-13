# Executable model onboarding pipeline

## Objective and scope

Deliver a tested, recipe-driven CLI that takes the existing pinned E5 CPU profile from asset verification through reference execution, Python ONNX execution, .NET fidelity, and a review bundle. Demonstrate a successful real-model run, a verified resume, and failure/invalidation behavior. Open a draft PR with evidence and passing required checks.

Work is isolated on `codex/model-onboarding-pipeline` in `/home/mark/Projects/AI-pipeline`, based on `90754a28b18986e32073c4e23a71e5e077484e03`. Existing worktrees, Qwen tokenizer PRs, EmbeddingGemma development, protected fixtures, and tolerances are not implementation targets. Use existing model snapshots read-only; create environments, build outputs, and run artifacts in this worktree or a dedicated pipeline artifacts directory. CPU tests use bounded thread counts; no GPU jobs are needed.

This PR delivers the orchestration foundation and the first working recipe. Plugins provide the extension boundary for subsequent models. It does not claim that a new architecture, generation evaluator, or a new tokenizer can be qualified without its missing implementation. Existing E5 APIs and model code are reused rather than refactored while other model work is active. Qualification reports distinguish the automated G1/G2 result from a full release decision; this pipeline does not publish or silently promote support status.

## Implementation plan and ownership

1. **Recipe validation and capability inspection (small-model agent).** Add a versioned strict recipe schema that references the existing registry manifest instead of copying source hashes. Implement local snapshot inspection of configuration, tokenizer component/parameter requirements, and sentence-transformers modules. Report observed facts, unknowns, unsupported features, and the declared plugin route. Inspection is informational; it never executes upstream code or grants support. Add focused malformed-input and capability-gap tests.
2. **Resumable stage engine (workhorse-model agent).** Implement stage dependency validation, deterministic input identities, output digest verification, atomic run records, per-stage logs and timings, bounded subprocess execution, and failure records. Resume only completed stages with matching inputs and intact outputs. Preserve failures and invalidate dependent stages when an input or output changes. Prevent concurrent writers to one run directory. Test interruption, changed inputs, corruption, missing evidence, and failure recovery with small subprocess fixtures.
3. **E5 plugin and recipe (workhorse-model agent).** Wrap existing pinned reference/export/fidelity tools. Verify asset size and SHA-256 against the manifest; write a resolved lock. Generate fresh reference outputs only into run artifacts, never protected paths. Run the Python ONNX stage and C# tokenizer/embedding stage, then invoke the protected evaluator requiring G1/G2. Record environment/toolchain identity. Expose explicit missing-dependency and unsupported-profile failures. Add an optional clean-consumer release-drill stage and metadata/report generation without automatic release claims.
4. **CLI integration and review bundle (primary agent).** Add `inspect`, `plan`, `run`, and `propose` commands. Provide explicit snapshot, reference Python, output directory, resume, thread, timeout, and distribution-check controls. Produce a portable review summary, structured failure task, and exact provenance. Document plugin extension, fixture adoption, and proposed versus qualified status. Ensure existing CLI commands remain compatible.
5. **Real-target validation (primary agent).** Use the existing immutable E5 snapshot and a dedicated Python environment. Execute fresh reference, export, .NET, and protected fidelity stages. Run again with resume and verify skipped computation and output identities. Corrupt an artifact in a separate test run and prove invalidation/recovery. Exercise asset mismatch and unsupported recipe failures. Run the clean-consumer drill from committed code with the real model. Save small evidence summaries and commands in the PR; keep weights and large tensors outside Git.
6. **Independent review and integration (subagent, then primary agent).** Review cache correctness, protected-evidence boundaries, output paths, command execution, status claims, and failure behavior. Fix findings and run the repository Python checks, manifest validation, applicable .NET tests, and tokenizer independence checks. Recheck upstream main for conflicts with ongoing work. Commit only this PR's files, push the isolated branch, create a draft PR, and verify GitHub checks.

## Module interfaces for parallel implementation

All agents work only in the isolated worktree and own disjoint files. They do not commit, push, edit the CLI, change protected evidence, or start expensive inference. The primary agent integrates and runs real workloads.

- `onboarding_recipe.py`: `load_recipe(path: Path, repo_root: Path) -> dict`; `inspect_snapshot(snapshot: Path) -> dict`. Raise `RecipeError` for invalid inputs. Recipe keys: `recipe_version: 1`, `id`, `plugin: e5-onnx-cpu-v1`, `manifest` (repository-relative path), `reference_corpus` (repository-relative path), `reference_environment` (repository-relative requirements file). Reject unknown keys, traversal, and unsupported plugins. Additional plugins require code and their own recipe validation/qualification contract.
- `onboarding_runner.py`: `Stage(name, dependencies, inputs, parameters, execute)`. `inputs` maps logical names to existing file/directory paths; `parameters` is JSON-compatible; `execute(output_dir, dependency_output_dirs)` writes only its stage output and returns JSON-compatible metadata. `run_pipeline(stages, run_dir, *, provenance, resume=False) -> dict`; `plan_stages(stages) -> list[dict]`; `run_command(argv, *, cwd, log_path, env=None, timeout=...)` executes without a shell and raises on failure. Stage failures return a structured failed run; invalid plans raise `PipelineError`. Include stage timing, fingerprint, output digests, cache status, and failure details in `run.json`.
- `onboarding_e5.py`: `build_stages(recipe, *, repo_root, snapshot, reference_python, threads=2, timeout=1800, distribution=False) -> list[Stage]`; `describe_plan(recipe, *, repo_root) -> dict` for planning without dependencies or assets. Resolve manifest facts through existing loader. A final stage emits `review.json` and `review.md` with exact gate evidence and explicit incomplete release gates. Dependency directories come from the engine; no shared mutable stage outputs.
- Primary-owned files: `cli.py`, CLI integration tests, documentation, evidence summary, and final packaging of the work.

## Acceptance criteria

- Existing Python tests and CLI commands continue to pass.
- Strict recipes cannot inject shell commands or escape repository input paths. Output roots cannot overlap source snapshots or protected evidence.
- Resume detects changed inputs, environment/plugin identity changes, missing/corrupt output files, and failed stages. A cached check never turns missing required evidence into success.
- Fresh reference outputs remain proposals. Only the existing protected evaluator can provide G1/G2 gate results; required gates must pass for a successful qualification run.
- A real E5 CPU run produces complete tokenizer/export/.NET evidence and passed G1/G2 with unchanged protected inputs. The run identifies all source assets and relevant code/environment inputs.
- The clean-consumer drill uses a real model and committed candidate code; a skipped or synthetic model test cannot satisfy this criterion.
- An unsupported model yields an actionable capability report or failure task, never a support badge.
- No weights, large tensors, credentials, unrelated worktree changes, or new production inference dependencies enter the PR.
- The draft PR includes exact validation results, a real-run evidence summary, known scope limits, and passing applicable GitHub checks.

## Follow-on work enabled by this PR

After the orchestration contract is proven, add a second compatible encoder recipe and extract shared .NET components based on actual differences. Add generation and extraction plugins with their own evidence contracts; convert EmbeddingGemma inspection gaps into bounded feature tasks. Trusted qualification scheduling and upstream discovery can call the same CLI. These additions must not be represented as implemented by the initial E5 recipe.
