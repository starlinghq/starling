"""Explicit BERT candidate execution and development parity; no protected qualification claims."""

from __future__ import annotations

import array
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

from model_pipeline import fixtures
from model_pipeline.onboarding_candidate import guard_output, write_json
from model_pipeline.onboarding_runner import PipelineError, Stage, run_command

PLUGIN = "bert-onnx-cpu-v1"
# Distribution is appended by onboarding_plugins from onboarding_distribution.
SHARED_DISTRIBUTION = True
RUNNER = "model-pipeline/reference/bert/run_bert.py"
TOOL = "ai-dotnet/tools/Starlings.Bert.Fidelity.Tool/Starlings.Bert.Fidelity.Tool.csproj"
FORMAT = "starlings.bert-candidate-results"
QUERY = "Represent this sentence for searching relevant passages: "
POLICY = {
    "token_ids": "exact",
    "attention_mask": "exact",
    "token_type_ids": "exact",
    "hidden_max_abs": 1e-4,
    "embedding_max_abs": 1e-5,
    "score_max_abs": 1e-4,
    "same_backend_max_abs": 1e-5,
    "padded_batch_max_abs": 1e-5,
    "non_finite": "fail",
    "required_paths": ["reference", "python-onnx", "dotnet-ids", "dotnet-text"],
}
CONTRACTS = {
    "sentence-transformers/all-MiniLM-L6-v2": ("embeddings", 256, "mean"),
    "BAAI/bge-small-en-v1.5": ("embeddings", 512, "cls"),
    "cross-encoder/ms-marco-MiniLM-L6-v2": ("reranking", 512, "none"),
}


def validate_candidate_behavior(candidate: dict) -> None:
    """Refuse schema-valid contracts that the fixed model implementation does not implement."""
    try:
        repo = candidate["origin"]["repo"]
        task, length, pooling = CONTRACTS[repo]
        expected = {
            "architecture": "bert",
            "graph_file": "onnx/model.onnx",
            "tokenizer_file": "tokenizer.json",
            "max_tokens": length,
            "padding_side": "right",
            "pad_token_id": 0,
            "vocabulary_size": 30522,
            "hidden_size": 384,
            "truncation": "longest_first",
            "tokenizer_profile": "bert-wordpiece-uncased",
            "pooling": pooling,
            "normalize": "l2" if task == "embeddings" else "none",
            "output_name": "last_hidden_state" if task == "embeddings" else "logits",
            "prompts": {"none": "", **({"query": QUERY} if repo == "BAAI/bge-small-en-v1.5" else {})},
        }
        if task == "reranking":
            expected["score_activation"] = "identity"
        if candidate["task"] != task or candidate["behavior"] != expected:
            raise PipelineError(
                f"unsupported BERT behavior: {repo} requires its exact {task}/{pooling}/{length} contract"
            )
        if candidate["runtime"] != {
            "backend": "onnx",
            "version": "1.30.0",
            "provider": "CPUExecutionProvider",
            "precision": "fp32",
        }:
            raise PipelineError("unsupported BERT runtime: only ONNX 1.30.0 CPU fp32 is implemented")
        if candidate["evaluation"]["status"] != "proposed" or candidate["evaluation"]["policy"] != POLICY:
            raise PipelineError("unsupported BERT evaluation policy: use the predeclared proposed development policy")
    except (KeyError, TypeError) as exc:
        raise PipelineError(f"unsupported or incomplete BERT contract: {exc}") from exc


def _document(path: Path, candidate: dict, digest: str, stage: str) -> tuple[dict, dict[str, dict]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"missing or malformed required {stage} evidence: {path}: {exc}") from exc
    if not isinstance(data, dict) or type(data.get("format_version")) is not int:
        raise PipelineError(f"{stage}: evidence must be an object with an integer format version")
    if (
        data.get("format"),
        data.get("format_version"),
        data.get("profile"),
        data.get("task"),
        data.get("candidate_sha256"),
        data.get("stage"),
    ) != (FORMAT, 1, candidate["id"], candidate["task"], digest, stage):
        raise PipelineError(f"{stage}: wrong format, profile, task, contract digest, or stage identity")
    expected = {
        entry["path"]: entry["sha256"]
        for entry in candidate["origin"]["files"]
        if entry["path"] in {"onnx/model.onnx", "tokenizer.json"}
    }
    try:
        assets = data["assets"]
        if len(assets) != 2 or {entry["path"]: entry["sha256"] for entry in assets} != expected or len(expected) != 2:
            raise PipelineError(f"{stage}: model/tokenizer asset identities differ from candidate")
        if data["environment"]["provider"] != "CPUExecutionProvider" or data["environment"]["onnxruntime"] != "1.30.0":
            raise PipelineError(f"{stage}: runtime/provider identity differs from candidate")
        records = data["cases"]
        if not isinstance(records, list) or len(records) < 2:
            raise PipelineError(f"{stage}: missing required case rows")
        cases = {}
        dimension = 384 if candidate["task"] == "embeddings" else 1
        for row in records:
            identifier = row["id"]
            if not isinstance(identifier, str) or not identifier or identifier in cases:
                raise PipelineError(f"{stage}: duplicate or invalid case id {identifier!r}")
            _vector(row["output"], dimension, f"{stage}/{identifier}/output")
            if stage == "dotnet":
                _vector(row["text_output"], dimension, f"{stage}/{identifier}/text_output")
            if candidate["task"] == "reranking":
                _vector(row["logits"], 1, f"{stage}/{identifier}/logits")
                _vector([row["score"]], 1, f"{stage}/{identifier}/score")
                if row["logits"] != row["output"] or row["score"] != row["output"][0]:
                    raise PipelineError(f"{stage}/{identifier}: identity score activation differs from raw logits")
                if stage == "dotnet" and row["text_score"] != row["text_output"][0]:
                    raise PipelineError(f"{stage}/{identifier}: text score activation differs from raw logits")
            ids, mask, types = (row[key] for key in ("input_ids", "attention_mask", "token_type_ids"))
            if (
                not all(isinstance(value, list) for value in (ids, mask, types))
                or not 2 <= len(ids) <= candidate["behavior"]["max_tokens"]
                or len(mask) != len(ids)
                or len(types) != len(ids)
                or any(type(value) is not int or not 0 <= value < 30522 for value in ids)
                or any(type(value) is not int or value != 1 for value in mask)
                or any(type(value) is not int or value not in (0, 1) for value in types)
                or ids[0] != 101
                or ids[-1] != 102
            ):
                raise PipelineError(f"{stage}/{identifier}: invalid unpadded token inputs")
            cases[identifier] = row
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError(f"{stage}: incomplete or malformed required evidence: {exc}") from exc
    return data, cases


def _vector(values: list, length: int, where: str) -> None:
    if not isinstance(values, list) or len(values) != length:
        raise PipelineError(f"{where}: wrong output shape; expected {length} values")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise PipelineError(f"{where}: non-finite or nonnumeric output")


def _difference(actual: list, expected: list, bound: float, where: str) -> dict:
    _vector(actual, len(expected), where)
    differences = [abs(left - right) for left, right in zip(actual, expected, strict=True)]
    maximum = max(differences, default=0.0)
    return {
        "id": where,
        "status": "passed" if maximum <= bound else "failed",
        "bound": bound,
        "max_abs": maximum,
        "mean_abs": math.fsum(differences) / len(differences) if differences else 0.0,
    }


def _hidden(path: Path, document: dict, cases: dict[str, dict]) -> dict[str, list[float]]:
    try:
        specification = document["hidden"]
        relative = Path(specification["file"])
        if relative.is_absolute() or len(relative.parts) != 1 or relative.name in {".", ".."}:
            raise PipelineError("unsafe hidden-state sidecar path")
        tensor_path = path.parent / relative
        if tensor_path.is_symlink() or fixtures.sha256_file(tensor_path) != specification["sha256"]:
            raise PipelineError("hidden-state tensor identity mismatch")
        values = array.array("f")
        blob = tensor_path.read_bytes()
        if len(blob) % 4:
            raise PipelineError("hidden-state tensor has an incomplete float32 value")
        values.frombytes(blob)
        if sys.byteorder != "little":
            values.byteswap()
        if any(not math.isfinite(value) for value in values):
            raise PipelineError("hidden-state tensor contains non-finite values")
        output, position = {}, 0
        for row in specification["cases"]:
            identifier = row["id"]
            if (
                identifier not in cases
                or identifier in output
                or row["shape"] != [len(cases[identifier]["input_ids"]), 384]
                or row["offset"] != position
            ):
                raise PipelineError("hidden-state cases have wrong ids, shape, offset, or duplicate rows")
            end = position + math.prod(row["shape"])
            if end > len(values):
                raise PipelineError("hidden-state tensor is truncated")
            output[identifier] = values[position:end].tolist()
            position = end
        if set(output) != set(cases) or position != len(values):
            raise PipelineError("hidden-state tensor has missing cases or unreferenced values")
        return output
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise PipelineError(f"missing or malformed required hidden-state evidence: {exc}") from exc


def evaluate_development(
    candidate: dict, candidate_path: Path, reference_path: Path, onnx_path: Path, dotnet_path: Path
) -> dict:
    """Check complete proposed evidence without reading or promoting protected fixtures."""
    validate_candidate_behavior(candidate)
    digest = fixtures.sha256_file(candidate_path)
    reference, expected = _document(reference_path, candidate, digest, "reference")
    onnx, python_cases = _document(onnx_path, candidate, digest, "onnx")
    dotnet, dotnet_cases = _document(dotnet_path, candidate, digest, "dotnet")
    machines = [document.get("environment", {}).get("machine") for document in (reference, onnx, dotnet)]
    if any(not isinstance(machine, str) or not machine for machine in machines) or len(set(machines)) != 1:
        raise PipelineError("same-backend comparison requires all stages to name the same machine")
    reference_digest = fixtures.sha256_file(reference_path)
    if onnx.get("reference_sha256") != reference_digest or dotnet.get("reference_sha256") != reference_digest:
        raise PipelineError("backend evidence names a different original reference artifact")
    if set(expected) != set(python_cases) or set(expected) != set(dotnet_cases):
        raise PipelineError("backend evidence has missing or unknown required case rows")
    checks = []
    policy = candidate["evaluation"]["policy"]
    bound = policy["embedding_max_abs"] if candidate["task"] == "embeddings" else policy["score_max_abs"]
    for identifier, row in expected.items():
        for stage, cases in (("onnx", python_cases), ("dotnet", dotnet_cases)):
            for key in ("input_ids", "attention_mask", "token_type_ids"):
                if cases[identifier][key] != row[key]:
                    raise PipelineError(f"{stage}/{identifier}: {key} differs from proposed original reference")
            checks.append(
                _difference(cases[identifier]["output"], row["output"], bound, f"{stage}/{identifier}/vs_reference")
            )
        checks.append(
            _difference(
                dotnet_cases[identifier]["text_output"], row["output"], bound, f"dotnet-text/{identifier}/vs_reference"
            )
        )
        checks.append(
            _difference(
                dotnet_cases[identifier]["output"],
                python_cases[identifier]["output"],
                policy["same_backend_max_abs"],
                f"dotnet/{identifier}/vs_python",
            )
        )
        checks.append(
            _difference(
                dotnet_cases[identifier]["text_output"],
                python_cases[identifier]["output"],
                policy["same_backend_max_abs"],
                f"dotnet-text/{identifier}/vs_python",
            )
        )
        if candidate["task"] == "reranking":
            if "original_score" not in row:
                raise PipelineError(f"reference/{identifier}: independent public CrossEncoder score is missing")
            checks.append(
                _difference(
                    [row["original_score"]], row["output"], bound, f"reference/{identifier}/cross_encoder_public"
                )
            )
        else:
            for stage, values in (
                ("reference", row["output"]),
                ("onnx", python_cases[identifier]["output"]),
                ("dotnet", dotnet_cases[identifier]["output"]),
                ("dotnet-text", dotnet_cases[identifier]["text_output"]),
            ):
                checks.append(
                    _difference(
                        [math.sqrt(math.fsum(value * value for value in values))],
                        [1.0],
                        policy["embedding_max_abs"],
                        f"{stage}/{identifier}/l2_norm",
                    )
                )
    if candidate["task"] == "embeddings":
        tensors = [
            _hidden(path, document, cases)
            for path, document, cases in (
                (reference_path, reference, expected),
                (onnx_path, onnx, python_cases),
                (dotnet_path, dotnet, dotnet_cases),
            )
        ]
        for identifier in expected:
            checks.append(
                _difference(
                    tensors[1][identifier],
                    tensors[0][identifier],
                    policy["hidden_max_abs"],
                    f"onnx/{identifier}/hidden",
                )
            )
            checks.append(
                _difference(
                    tensors[2][identifier],
                    tensors[1][identifier],
                    policy["hidden_max_abs"],
                    f"dotnet/{identifier}/hidden_vs_python",
                )
            )
    for stage, document, cases in (
        ("reference", reference, expected),
        ("onnx", onnx, python_cases),
        ("dotnet", dotnet, dotnet_cases),
    ):
        batches = document.get("batches")
        required = {"ids", "text"} if stage == "dotnet" else {"ids"}
        if (
            not isinstance(batches, list)
            or any(not isinstance(batch, dict) for batch in batches)
            or {batch.get("path") for batch in batches} != required
            or len(batches) != len(required)
        ):
            raise PipelineError(f"{stage}: missing or duplicate required ids/text padded-batch evidence")
        for batch in batches:
            identifiers = batch.get("case_ids", [])
            lengths = batch.get("input_lengths", [])
            if (
                len(identifiers) != len(cases)
                or set(identifiers) != set(cases)
                or len(set(identifiers)) != len(identifiers)
            ):
                raise PipelineError(f"{stage}: batch has missing, unknown, or duplicate case rows")
            if lengths != [len(cases[identifier]["input_ids"]) for identifier in identifiers] or len(set(lengths)) < 2:
                raise PipelineError(f"{stage}: batch does not prove differently sized padded rows")
            if not isinstance(batch.get("output"), list) or len(batch["output"]) != len(identifiers):
                raise PipelineError(f"{stage}: batch output has wrong shape")
            chunks = batch.get("chunks")
            if not isinstance(chunks, list) or not chunks:
                raise PipelineError(f"{stage}: actual padded batch chunk evidence is missing")
            covered, padded_chunk = [], False
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    raise PipelineError(f"{stage}: malformed padded batch chunk")
                chunk_ids = chunk.get("case_ids", [])
                chunk_lengths = chunk.get("input_lengths", [])
                if (
                    not chunk_ids
                    or any(identifier not in cases for identifier in chunk_ids)
                    or chunk_lengths != [len(cases[identifier]["input_ids"]) for identifier in chunk_ids]
                    or chunk.get("width") != max(chunk_lengths)
                ):
                    raise PipelineError(f"{stage}: padded batch chunk has wrong identities, lengths, or width")
                covered.extend(chunk_ids)
                padded_chunk |= len(set(chunk_lengths)) > 1
            if sorted(covered) != sorted(identifiers) or not padded_chunk:
                raise PipelineError(
                    f"{stage}: chunks do not cover all cases with a real differently sized padded batch"
                )
            for identifier, output in zip(identifiers, batch["output"], strict=True):
                single = cases[identifier]["text_output" if batch["path"] == "text" else "output"]
                checks.append(
                    _difference(
                        output, single, policy["padded_batch_max_abs"], f"{stage}/{identifier}/batch_{batch['path']}"
                    )
                )
    if dotnet.get("checks", {}).get("pre_cancelled_request") is not True:
        raise PipelineError(".NET cancellation evidence is missing or failed")
    failed = [check["id"] for check in checks if check["status"] != "passed"]
    return {
        "format": "starlings.bert-development-parity",
        "format_version": 1,
        "profile": candidate["id"],
        "task": candidate["task"],
        "status": "failed" if failed else "passed",
        "failed_checks": failed,
        "checks": checks,
        "case_count": len(expected),
        "evaluation_status": "proposed",
        "policy": policy,
        "protected_release_qualification": {"status": "unapproved", "G1": "not-run", "G2": "not-run"},
        "inputs": {
            "candidate_sha256": digest,
            "reference_sha256": reference_digest,
            "onnx_sha256": fixtures.sha256_file(onnx_path),
            "dotnet_sha256": fixtures.sha256_file(dotnet_path),
        },
    }


def describe_plan(recipe: dict, *, repo_root: Path) -> dict:
    from model_pipeline.onboarding_candidate import load_candidate

    candidate = load_candidate(repo_root / recipe["manifest"])
    validate_candidate_behavior(candidate)
    return {
        "plugin": PLUGIN,
        "recipe": recipe["id"],
        "profile": candidate["id"],
        "task": candidate["task"],
        "stages": ["assets", "environment", "reference", "onnx", "dotnet", "evaluate", "review"],
        "reference_status": "proposed",
        "qualification_status": "unapproved",
        "policy": POLICY,
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
    if distribution:
        raise PipelineError("BERT distribution stage is not wired; use the root clean-consumer workflow")
    from model_pipeline.onboarding_candidate import candidate_stages, source_reader

    root, snapshot = repo_root.resolve(), snapshot.resolve()
    source_files = (
        "model-pipeline/src/model_pipeline/onboarding_bert.py",
        "model-pipeline/src/model_pipeline/onboarding_candidate.py",
        "model-pipeline/src/model_pipeline/onboarding_runner.py",
        "model-pipeline/src/model_pipeline/fixtures.py",
        RUNNER,
        "Directory.Build.props",
        "global.json",
        "ai-dotnet/Directory.Build.props",
        "ai-dotnet/Directory.Build.targets",
        "ai-dotnet/Directory.Packages.props",
        *(str(path.relative_to(root)) for path in (root / "Directory.Build.targets",) if path.is_file()),
    )
    source_directories = (
        "model-pipeline/schemas",
        "model-pipeline/reference/bert",
        "ai-dotnet/src/Starlings.Models.Bert",
        "ai-dotnet/tools/Starlings.Bert.Fidelity.Tool",
        "ai-dotnet/src/Starlings.Runtime.Onnx",
        "ai-dotnet/src/Starlings.Acquisition",
    )
    candidate, initial, shared_inputs = candidate_stages(
        recipe,
        repo_root=root,
        snapshot=snapshot,
        reference_python=reference_python,
        threads=threads,
        timeout=timeout,
        source_files=source_files,
        source_directories=source_directories,
    )
    validate_candidate_behavior(candidate)
    profile = root / recipe["manifest"]
    corpus = root / recipe["reference_corpus"]
    python = Path(os.path.abspath(reference_python))
    # candidate_stages already declares source_files/source_directories through code_inputs.
    code = shared_inputs
    parameters = {"plugin": PLUGIN, "candidate": candidate, "threads": threads, "timeout": timeout}
    env = {
        "PYTHONPATH": str(root / "model-pipeline/src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "DOTNET_PROCESSOR_COUNT": str(threads),
        "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
    }

    def guard(out: Path) -> None:
        guard_output(
            out,
            (snapshot, root / "model-pipeline/protected"),
            "candidate output overlaps source snapshot or protected evidence",
        )

    def stage(name: str, dependencies: tuple, callback, inputs: dict | None = None, **extra) -> Stage:
        return Stage(
            name, dependencies, {**code, "candidate": profile, **(inputs or {})}, {**parameters, **extra}, callback
        )

    def python_stage(out: Path, kind: str, extra: list[str]) -> dict:
        guard(out)
        return run_command(
            [
                str(python),
                str(root / RUNNER),
                "--profile",
                str(profile),
                "--snapshot",
                str(snapshot),
                "--stage",
                kind,
                "--out",
                str(out),
                "--threads",
                str(threads),
                *extra,
            ],
            cwd=root,
            log_path=out / "command.log",
            env=env,
            timeout=timeout,
        )

    def reference(out: Path, deps: dict) -> dict:
        return python_stage(out, "reference", ["--corpus", str(corpus)])

    def onnx(out: Path, deps: dict) -> dict:
        return python_stage(out, "onnx", ["--reference", str(deps["reference"] / "reference.json")])

    def dotnet(out: Path, deps: dict) -> dict:
        guard(out)
        with tempfile.TemporaryDirectory(prefix="dotnet-build-", dir=out) as build:
            dotnet_executable = shutil.which("dotnet") or "dotnet"
            published = Path(build) / "published"
            build_result = run_command(
                [
                    dotnet_executable,
                    "publish",
                    str(root / TOOL),
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
            assembly = published / "Starlings.Bert.Fidelity.Tool.dll"
            if not assembly.is_file():
                raise PipelineError("BERT fidelity publish succeeded without producing the required tool assembly")
            result = run_command(
                [
                    dotnet_executable,
                    str(assembly),
                    "--profile",
                    str(profile),
                    "--snapshot",
                    str(snapshot),
                    "--reference",
                    str(deps["reference"] / "reference.json"),
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
        report = evaluate_development(
            candidate,
            profile,
            deps["reference"] / "reference.json",
            deps["onnx"] / "onnx.json",
            deps["dotnet"] / "dotnet.json",
        )
        write_json(out / "report.json", report)
        (out / "report.md").write_text(
            f"# Development parity: {candidate['id']}\n\nStatus: **{report['status']}**; "
            f"{report['case_count']} cases. Original references and comparison policies are **proposed**.\n\n"
            "Protected G1/G2 release qualification is **unapproved**. No protected fixtures were adopted.\n",
            encoding="utf-8",
        )
        if report["status"] != "passed":
            raise PipelineError(
                f"BERT development parity failed: {', '.join(report['failed_checks'])}; see {out / 'report.json'}"
            )
        return {"status": report["status"], "qualification_status": "unapproved", "reference_status": "proposed"}

    def review(out: Path, deps: dict) -> dict:
        guard(out)
        report = json.loads((deps["evaluate"] / "report.json").read_text(encoding="utf-8"))
        if report["status"] != "passed":
            raise PipelineError("review requires complete passed development parity")
        for name in ("report.json", "report.md"):
            shutil.copyfile(deps["evaluate"] / name, out / name)
        document = {
            "profile": candidate["id"],
            "task": candidate["task"],
            "development_parity": report,
            "candidate_commit": fixtures.git_head(root),
            "source_lock": json.loads((deps["assets"] / "source-lock.json").read_text()),
            "support_status": "experimental",
            "release_qualification": "unapproved",
            "reference_status": "proposed",
            "protected_fixture_adoption": False,
            "distribution": "not-run",
        }
        write_json(out / "review.json", document)
        (out / "review.md").write_text(
            f"# Candidate review: {candidate['id']}\n\n"
            "Development task parity passed. References/policies remain proposed. "
            "Protected G1/G2 and full release qualification are unapproved; distribution is not run.\n",
            encoding="utf-8",
        )
        return {"development_parity": "passed", "release_qualification": "unapproved"}

    stages = [
        *initial,
        stage("reference", ("assets", "environment"), reference, {"corpus": corpus}),
        stage("onnx", ("reference", "assets", "environment"), onnx),
        stage("dotnet", ("reference", "assets", "environment"), dotnet),
        stage("evaluate", ("reference", "onnx", "dotnet"), evaluation),
        stage("review", ("assets", "reference", "evaluate"), review, candidate_commit=fixtures.git_head(root)),
    ]
    return [
        source_reader(item, candidate, snapshot) if item.name in {"reference", "onnx", "dotnet"} else item
        for item in stages
    ]
