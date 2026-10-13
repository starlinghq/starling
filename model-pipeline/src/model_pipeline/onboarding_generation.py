"""Proposed Qwen generation onboarding with original/Python GenAI/owned .NET parity."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

from model_pipeline import fixtures
from model_pipeline.evaluate_generation_candidate import POLICY, evaluate
from model_pipeline.onboarding_candidate import guard_output, repository_input, write_json
from model_pipeline.onboarding_runner import PipelineError, Stage, run_command

PLUGIN = "qwen-genai-cpu-v1"
# Distribution is appended by onboarding_plugins from onboarding_distribution.
SHARED_DISTRIBUTION = True
REFERENCE = "model-pipeline/reference/qwen-chat/run_reference.py"
BUILDER = "model-pipeline/reference/qwen-chat/build_bundle.py"
BACKEND = "model-pipeline/reference/qwen-chat/run_genai.py"
FIDELITY = "ai-dotnet/tools/Starlings.Generation.Fidelity/Starlings.Generation.Fidelity.csproj"
PROFILES = {
    "Qwen/Qwen3-0.6B": {
        "id": "qwen3-0.6b/ortgenai-fp32-cpu",
        "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "architecture": "Qwen3ForCausalLM",
        "model_type": "qwen3",
        "eos_token_ids": [151643, 151645],
        "template": "qwen3-text-v1",
    },
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "id": "qwen2.5-0.5b-instruct/ortgenai-fp32-cpu",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "architecture": "Qwen2ForCausalLM",
        "model_type": "qwen2",
        "eos_token_ids": [151643, 151645],
        "template": "qwen25-text-v1",
    },
}
CONFIG_CONTRACT = {
    "qwen3": {
        "hidden_size": 1024,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 40960,
        "intermediate_size": 3072,
    },
    "qwen2": {
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "max_position_embeddings": 32768,
        "intermediate_size": 4864,
    },
}
EXPECTED_BEHAVIOR = {
    repository: {
        **{key: record[key] for key in ("architecture", "model_type", "eos_token_ids", "template")},
        "vocab_size": 151936,
        "im_start_token_id": 151644,
        "im_end_token_id": 151645,
        "endoftext_token_id": 151643,
        "pad_token_id_overlay": 151935,
        "pad_overlay_reason": "unused embedding row; preserve literal <|endoftext|> prompt attention",
        "enable_thinking_default": False,
        "max_new_tokens": 16,
        "cache": "fresh-request; cached-vs-uncached reference verified",
        "tools": "unsupported",
        "multimodal": "unsupported",
    }
    for repository, record in PROFILES.items()
}
EXPECTED_RUNTIME = {"backend": "ortgenai", "version": "0.17.1", "provider": "CPUExecutionProvider", "precision": "fp32"}
REQUIRED_EVIDENCE = [
    "prompt_text",
    "prompt_ids",
    "initial_logits",
    "output_ids",
    "finish_reason",
    "stream",
    "usage",
    "cancellation",
    "cache",
    "isolation",
]


def _profile(recipe: dict, root: Path) -> tuple[Path, dict]:
    from model_pipeline.onboarding_candidate import load_candidate

    if recipe.get("plugin") != PLUGIN or recipe.get("recipe_version") != 1:
        raise PipelineError(f"expected recipe_version 1 and plugin {PLUGIN}")
    path = repository_input(root, recipe["manifest"])
    candidate = load_candidate(path)
    try:
        repository = candidate["origin"]["repo"]
        expected = PROFILES[repository]
        if (
            candidate["format_version"] != 1
            or candidate["task"] != "generation"
            or candidate["id"] != expected["id"]
            or candidate["origin"]["revision"] != expected["revision"]
            or candidate["runtime"] != EXPECTED_RUNTIME
            or json.dumps(candidate["behavior"], sort_keys=True)
            != json.dumps(EXPECTED_BEHAVIOR[repository], sort_keys=True)
            or candidate["evaluation"]["status"] != "proposed"
            or any(candidate["evaluation"]["policy"].get(key) != value for key, value in POLICY.items())
            or candidate["evaluation"]["policy"].get("required_evidence") != REQUIRED_EVIDENCE
        ):
            raise PipelineError("unsupported generation contract: exact pinned text-only CPU profiles required")
    except (KeyError, TypeError) as exc:
        raise PipelineError(f"incomplete or unsupported generation candidate: {exc}") from exc
    return path, candidate


def describe_plan(recipe: dict, *, repo_root: Path) -> dict:
    _, candidate = _profile(recipe, repo_root.resolve())
    return {
        "plugin": PLUGIN,
        "recipe": recipe["id"],
        "profile": candidate["id"],
        "revision": candidate["origin"]["revision"],
        "asset_count": len(candidate["origin"]["files"]),
        "stages": ["assets", "environment", "reference", "bundle", "backend", "dotnet", "evaluate", "review"],
        "required_gates": ["development-parity"],
        "optional_stages": [],
        "reference_status": "proposed; separate independent adoption required",
        "release_status": "unapproved; protected G1/G2 not awarded",
        "dependencies": {
            "python_packages": [
                "torch",
                "transformers",
                "tokenizers",
                "numpy",
                "safetensors",
                "onnxruntime-genai",
                "onnxruntime",
                "onnx",
                "onnx-ir",
            ],
            "dotnet_sdk": "global.json",
        },
    }


def build_stages(
    recipe: dict,
    *,
    repo_root: Path,
    snapshot: Path,
    reference_python: Path,
    threads: int = 2,
    timeout: float = 1800,
    distribution: bool = False,
) -> list[Stage]:
    from model_pipeline.onboarding_candidate import candidate_stages, code_inputs, source_reader

    if distribution:
        raise PipelineError(
            "generation distribution is not implemented by this plugin; run the independent clean-consumer drill"
        )
    if type(threads) is not int or not 1 <= threads <= 2:
        raise PipelineError("Qwen CPU profiles permit one or two threads")
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or not 0 < timeout <= 86400:
        raise PipelineError("timeout must be finite, positive, and at most 86400 seconds")
    root, snapshot = repo_root.resolve(), snapshot.resolve()
    manifest_path, candidate = _profile(recipe, root)
    corpus = repository_input(root, recipe["reference_corpus"])
    _, stages, shared_inputs = candidate_stages(
        recipe,
        repo_root=root,
        snapshot=snapshot,
        reference_python=reference_python,
        threads=threads,
        timeout=timeout,
        source_files=(
            REFERENCE,
            BUILDER,
            BACKEND,
            "model-pipeline/reference/qwen-chat/common.py",
            "model-pipeline/src/model_pipeline/onboarding_generation.py",
            "model-pipeline/src/model_pipeline/evaluate_generation_candidate.py",
        ),
    )
    python = Path(os.path.abspath(reference_python))
    dotnet = shutil.which("dotnet") or "dotnet"
    env = {
        "PYTHONPATH": str(root / "model-pipeline/src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DOTNET_PROCESSOR_COUNT": str(threads),
        "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
    }

    def guard(out: Path) -> None:
        guard_output(
            out,
            (
                snapshot,
                root / "model-pipeline/protected",
                corpus,
                manifest_path,
            ),
            "stage output overlaps original source or protected evidence",
        )

    def python_run(out: Path, script: str, args: list[str]) -> dict:
        guard(out)
        return run_command(
            [str(python), str(root / script), "--candidate", str(manifest_path), *args, "--out", str(out)],
            cwd=root,
            log_path=out / "command.log",
            env=env,
            timeout=timeout,
        )

    def reference(out: Path, deps: dict) -> dict:
        return python_run(
            out, REFERENCE, ["--snapshot", str(snapshot), "--corpus", str(corpus), "--threads", str(threads)]
        )

    def bundle(out: Path, deps: dict) -> dict:
        return python_run(out, BUILDER, ["--snapshot", str(snapshot)])

    def backend(out: Path, deps: dict) -> dict:
        return python_run(
            out,
            BACKEND,
            [
                "--snapshot",
                str(snapshot),
                "--bundle",
                str(deps["bundle"]),
                "--corpus",
                str(corpus),
                "--threads",
                str(threads),
            ],
        )

    def dotnet_run(out: Path, deps: dict) -> dict:
        guard(out)
        # Build intermediates and the published tool (including native runtime libraries) are
        # transient: they must not be hashed into run.json, rehashed on resume, or proposed.
        with tempfile.TemporaryDirectory(prefix="dotnet-build-", dir=out) as build:
            published = Path(build) / "published"
            build_result = run_command(
                [
                    dotnet,
                    "publish",
                    str(root / FIDELITY),
                    "-c",
                    "Release",
                    "-m:1",
                    "--disable-build-servers",
                    "--artifacts-path",
                    build,
                    "--output",
                    str(published),
                    "-p:RestoreLockedMode=true",
                ],
                cwd=root,
                log_path=out / "build.log",
                env=env,
                timeout=timeout,
            )
            assembly = published / "Starlings.Generation.Fidelity.dll"
            if not assembly.is_file():
                raise PipelineError("generation fidelity publish produced no tool assembly")
            result = run_command(
                [
                    dotnet,
                    str(assembly),
                    "--candidate",
                    str(manifest_path),
                    "--bundle",
                    str(deps["bundle"]),
                    "--corpus",
                    str(corpus),
                    "--out",
                    str(out),
                    "--threads",
                    str(threads),
                ],
                cwd=root,
                log_path=out / "command.log",
                env=env,
                timeout=timeout,
            )
            return {**result, "build": build_result}

    def evaluation(out: Path, deps: dict) -> dict:
        guard(out)
        report = evaluate(candidate, corpus, deps["reference"], deps["backend"], deps["dotnet"])
        write_json(out / "report.json", report)
        if report["status"] != "passed":
            raise PipelineError("development generation parity failed: " + "; ".join(report["failures"]))
        return report

    def review(out: Path, deps: dict) -> dict:
        guard(out)
        report = json.loads((deps["evaluate"] / "report.json").read_text(encoding="utf-8"))
        if report["status"] != "passed":
            raise PipelineError("review requires complete passed development parity")
        shutil.copyfile(deps["evaluate"] / "report.json", out / "report.json")
        data = {
            "profile": candidate["id"],
            "task": candidate["task"],
            "development_parity": report,
            "policy": candidate["evaluation"]["policy"],
            "candidate_commit": fixtures.git_head(root),
            "source_lock": json.loads((deps["assets"] / "source-lock.json").read_text(encoding="utf-8")),
            "support_status": "experimental",
            "support_status_changed": False,
            "release_qualification": "unapproved",
            "protected_release_qualification": "unapproved",
            "reference_status": "proposed",
            "protected_fixture_adoption": False,
            "distribution": "not-run",
        }
        write_json(out / "review.json", data)
        (out / "review.md").write_text(
            f"# Candidate review: {candidate['id']}\n\n"
            "Development task parity passed. References/policies remain proposed. "
            "Protected G1/G2 and full release qualification are unapproved; distribution is not run.\n",
            encoding="utf-8",
        )
        return {"development_parity": "passed", "release_qualification": "unapproved"}

    def stage(name: str, dependencies: tuple[str, ...], execute, *, extra_inputs=None) -> Stage:
        return Stage(
            name,
            dependencies,
            {**shared_inputs, **(extra_inputs or {})},
            {"plugin": PLUGIN, "profile": candidate, "threads": threads, "timeout": timeout},
            execute,
        )

    dotnet_inputs = code_inputs(
        root,
        files=(
            "global.json",
            "Directory.Build.props",
            "ai-dotnet/Directory.Build.props",
            "ai-dotnet/Directory.Build.targets",
            "ai-dotnet/Directory.Packages.props",
            *(str(path.relative_to(root)) for path in (root / "Directory.Build.targets",) if path.is_file()),
        ),
        directories=(
            "ai-dotnet/src/Starlings.Models.QwenChat",
            "ai-dotnet/src/Starlings.Generation.OnnxGenAI",
            "ai-dotnet/tools/Starlings.Generation.Fidelity",
        ),
    )
    stages.extend(
        [
            stage("reference", ("assets", "environment"), reference, extra_inputs={"corpus": corpus}),
            stage("bundle", ("assets", "environment"), bundle),
            stage("backend", ("bundle", "environment"), backend, extra_inputs={"corpus": corpus}),
            stage("dotnet", ("bundle", "environment"), dotnet_run, extra_inputs={"corpus": corpus, **dotnet_inputs}),
            stage("evaluate", ("reference", "backend", "dotnet"), evaluation, extra_inputs={"corpus": corpus}),
            dataclasses.replace(
                stage("review", ("assets", "evaluate"), review),
                parameters={
                    "plugin": PLUGIN,
                    "profile": candidate,
                    "threads": threads,
                    "timeout": timeout,
                    "candidate_commit": fixtures.git_head(root),
                },
            ),
        ]
    )
    return [
        source_reader(stage, candidate, snapshot) if stage.name in {"reference", "bundle", "backend"} else stage
        for stage in stages
    ]
