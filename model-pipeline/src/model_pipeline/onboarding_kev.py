"""Kev decision models on ONNX Runtime CPU fp32 (route A of model-pipeline/admissions/kev.md).

Explicit allow-list: each pinned Kev checkpoint, its pinned base, the pinned Kev code and the backbone/head/tokenizer
contract are checked field by field, here and in the .NET library. Anything else is refused. Stages:

    assets -> environment -> reference -> build -> onnx -> dotnet -> evaluate -> review

The Kev checkpoint is `--snapshot`; its base is `--base-snapshot`, a snapshot directory in Hugging Face cache layout
(`<cache>/models--<org>--<name>/snapshots/<revision>`), because Kev loads the base and its tokenizer through
transformers from that cache (offline). Reference and fixture outputs are proposals; nothing here adopts protected
fixtures or changes tolerances.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path, PurePosixPath

from model_pipeline import fixtures
from model_pipeline.evaluate_decision import POLICY, evaluate, protected_policy, render_markdown
from model_pipeline.onboarding_candidate import guard_output, repository_input, write_json
from model_pipeline.onboarding_runner import PipelineError, Stage, run_command

PLUGIN = "kev-onnx-cpu-v1"
# Distribution is appended by onboarding_plugins from onboarding_distribution (and refused below).
SHARED_DISTRIBUTION = True
RUNNER = "model-pipeline/reference/kev/run_kev.py"
TOOL = "ai-dotnet/tools/Starlings.Kev.Fidelity.Tool/Starlings.Kev.Fidelity.Tool.csproj"
STAGES = ["assets", "environment", "reference", "build", "onnx", "dotnet", "evaluate", "review"]
CODE = {"repository": "https://github.com/jaredpalmer/kev", "revision": "6b719c3c3f367295f6ef336f4f751cf5ff970abc"}
RUNTIME = {"backend": "onnx", "version": "1.30.0", "provider": "CPUExecutionProvider", "precision": "fp32"}
DELIMITERS = {"state": 248060, "question": 248061, "option": 248049, "option_end": 248050, "decide": 248062}
DELIMITER_TOKENS = {
    "state": "<|fim_prefix|>",
    "question": "<|fim_middle|>",
    "option": "<|box_start|>",
    "option_end": "<|box_end|>",
    "decide": "<|fim_suffix|>",
}
PAD = ("<|endoftext|>", 248044)
CAPABILITIES = {
    "decision.noul": "experimental",
    "decision.choice": "experimental",
    "decision.score": "experimental",
    "calibrated_temperature": "experimental",
    "delimiter_escaping": "experimental",
    "question_isolation": "experimental",
    "state_cache_within_request": "experimental",
    "state_json_rendering": "experimental",
    "cuda": "untested",
    "date_facts": "unsupported",
    "state_over_8192_tokens": "unsupported",
    "truncation": "unsupported",
    "bf16": "unsupported",
    "option_isolation": "unsupported",
    "http_server": "unsupported",
    "training": "unsupported",
}
KEV_FILES = (
    "README.md",
    "adapter_config.json",
    "adapter_model.safetensors",
    "head.pt",
    "provenance.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
BASE_FILES = (
    "LICENSE",
    "README.md",
    "config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)
TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"


def _behavior(hidden: int, layers: int, kv_heads: int, linear_value_heads: int, temperature: float) -> dict:
    return {
        "architecture": "Qwen3_5ForConditionalGeneration",
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "full_attention_interval": 4,
        "num_key_value_heads": kv_heads,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": linear_value_heads,
        "head_dim": 256,
        "pointer_dim": 256,
        "temperature": temperature,
        "delimiters": dict(DELIMITERS),
        "pad_token_id": PAD[1],
        "max_state_tokens": 8192,
        "max_row_tokens": 8192,
        "tokenizer_file": "tokenizer.json",
        "capabilities": dict(CAPABILITIES),
    }


CONTRACTS = {
    "jaredpalmer/kev-0.8b": {
        "id": "kev-0.8b/onnx-fp32-cpu",
        "revision": "9a45d25eb2ab761841196625383fa1dff0e56c1e",
        "base": ("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"),
        "shards": ("model.safetensors-00001-of-00001.safetensors",),
        "behavior": _behavior(1024, 24, 2, 16, 2.3510958125672174),
        "protected_tolerances": "model-pipeline/protected/kev-0.8b/onnx-fp32/tolerances.yaml",
    },
    "jaredpalmer/kev-4b": {
        "id": "kev-4b/onnx-fp32-cpu",
        "revision": "139fdd94f1b6a6ad80cc15e08fcb99cac885a101",
        "base": ("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b"),
        "shards": (
            "model.safetensors-00001-of-00002.safetensors",
            "model.safetensors-00002-of-00002.safetensors",
        ),
        "behavior": _behavior(2560, 32, 4, 32, 2.406050072164233),
        # No binding tolerances yet: the Kev-0.8B bounds are proposed unchanged.
        "protected_tolerances": None,
    },
}


def _refuse(field: str, detail: str) -> None:
    raise PipelineError(f"unsupported Kev contract field {field}: {detail}")


def _same(field: str, actual, expected) -> None:
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True) or (
        isinstance(expected, float) and (type(actual) is not float or actual != expected)
    ):
        _refuse(field, f"expected {expected!r}, found {actual!r}")


def _files(field: str, records, expected: tuple[str, ...]) -> None:
    paths = [record.get("path") for record in records] if isinstance(records, list) else None
    if paths is None or sorted(paths) != sorted(expected) or len(set(paths)) != len(paths):
        _refuse(field, f"pinned files must be exactly {sorted(expected)}")


def validate_candidate_behavior(candidate: dict, repo_root: Path | None = None) -> dict:
    """Refuse every contract this route does not implement; returns the matching allow-list entry."""
    try:
        origin = candidate["origin"]
        repo = origin["repo"]
        if repo not in CONTRACTS:
            _refuse("origin.repo", f"{repo!r} is not an allow-listed Kev checkpoint")
        contract = CONTRACTS[repo]
        _same("format_version", candidate["format_version"], 1)
        _same("id", candidate["id"], contract["id"])
        _same("task", candidate["task"], "decision")
        _same("origin.revision", origin["revision"], contract["revision"])
        _files("origin.files", origin["files"], KEV_FILES)
        tokenizer = next(record for record in origin["files"] if record["path"] == "tokenizer.json")
        _same("origin.files[tokenizer.json].sha256", tokenizer["sha256"], TOKENIZER_SHA256)
        base = origin["base"]
        _same("origin.base.repo", base["repo"], contract["base"][0])
        _same("origin.base.revision", base["revision"], contract["base"][1])
        _files("origin.base.files", base["files"], BASE_FILES + contract["shards"])
        for record in base["files"]:
            path = PurePosixPath(record["path"])
            if path.is_absolute() or len(path.parts) != 1:
                _refuse("origin.base.files", f"unsafe path {record['path']!r}")
        _same("origin.code", origin["code"], CODE)
        extra = set(origin) - {"repo", "revision", "files", "base", "code"}
        if extra:
            _refuse("origin", f"unknown fields {sorted(extra)}")
        _same("runtime", candidate["runtime"], RUNTIME)
        behavior, expected = candidate["behavior"], contract["behavior"]
        if set(behavior) != set(expected):
            _refuse("behavior", f"fields must be exactly {sorted(expected)}")
        for key, value in expected.items():
            _same(f"behavior.{key}", behavior[key], value)
        _same("evaluation.status", candidate["evaluation"]["status"], "proposed")
        _same("evaluation.policy", candidate["evaluation"]["policy"], POLICY)
        if repo_root is not None and contract["protected_tolerances"]:
            path = repo_root / contract["protected_tolerances"]
            bound = protected_policy(path.read_text(encoding="utf-8"))
            for key, value in bound.items():
                _same(f"evaluation.policy.{key} (binding {contract['protected_tolerances']})", POLICY[key], value)
        return contract
    except (KeyError, TypeError, AttributeError, StopIteration) as exc:
        raise PipelineError(f"unsupported or incomplete Kev contract: {exc!r}") from exc


def _profile(recipe: dict, root: Path) -> tuple[Path, dict, dict]:
    from model_pipeline.onboarding_candidate import load_candidate

    if recipe.get("plugin") != PLUGIN or recipe.get("recipe_version") != 1:
        raise PipelineError(f"expected recipe_version 1 and plugin {PLUGIN}")
    path = repository_input(root, recipe["manifest"])
    candidate = load_candidate(path)
    return path, candidate, validate_candidate_behavior(candidate, root)


def describe_plan(recipe: dict, *, repo_root: Path) -> dict:
    root = repo_root.resolve()
    _, candidate, contract = _profile(recipe, root)
    origin = candidate["origin"]
    return {
        "plugin": PLUGIN,
        "recipe": recipe["id"],
        "profile": candidate["id"],
        "task": candidate["task"],
        "revision": origin["revision"],
        "base": {"repo": origin["base"]["repo"], "revision": origin["base"]["revision"]},
        "code": origin["code"],
        "asset_count": len(origin["files"]) + len(origin["base"]["files"]),
        "stages": list(STAGES),
        "inputs": {
            "snapshot": f"local {origin['repo']} checkpoint at {origin['revision']}",
            "base_snapshot": f"<hf cache>/models--{origin['base']['repo'].replace('/', '--')}/snapshots/"
            f"{origin['base']['revision']}",
            "reference_python": "reference/kev/requirements.txt plus Kev from a checkout at the pinned commit",
        },
        "policy": candidate["evaluation"]["policy"],
        "policy_source": contract["protected_tolerances"]
        or "proposed: the Kev-0.8B binding tolerances, unchanged; no binding tolerances exist for this profile",
        "capabilities": candidate["behavior"]["capabilities"],
        "reference_status": "proposed; separate independent adoption required",
        "qualification_status": "unapproved",
    }


def _base_cache(base_snapshot: Path, repo: str, revision: str) -> Path:
    """The Hugging Face cache root holding a pinned base snapshot, after checking its layout."""
    if (
        base_snapshot.name != revision
        or base_snapshot.parent.name != "snapshots"
        or base_snapshot.parent.parent.name != "models--" + repo.replace("/", "--")
    ):
        raise PipelineError(
            f"base snapshot must be <hf cache>/models--{repo.replace('/', '--')}/snapshots/{revision}: {base_snapshot}"
        )
    return base_snapshot.parents[2]


def verify_base(candidate: dict, base_snapshot: Path) -> list[dict]:
    """Pinned size and SHA-256 of every base file. Hugging Face cache symlinks may only resolve inside the cache root
    (the repository's blobs, or a cache-wide content store that deduplicates them)."""
    base = candidate["origin"]["base"]
    repository = base_snapshot.parents[2].resolve()
    records = []
    for record in base["files"]:
        path = base_snapshot / record["path"]
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise PipelineError(f"missing base source: {record['path']}: {exc}") from exc
        if not resolved.is_relative_to(repository) or not resolved.is_file():
            raise PipelineError(f"base source escapes its Hugging Face cache or is not a file: {record['path']}")
        size, digest = resolved.stat().st_size, fixtures.sha256_file(resolved)
        if (size, digest) != (record["size"], record["sha256"]):
            raise PipelineError(
                f"base asset mismatch: {record['path']}: expected {record['size']} bytes sha256 {record['sha256']}, "
                f"found {size} bytes sha256 {digest}"
            )
        records.append({**record, "verification": "pinned-size-and-sha256"})
    return records


def check_base_config(candidate: dict, base_snapshot: Path) -> dict:
    """The pinned base config and the pinned Kev tokenizer agree with the contract (no upstream code runs)."""
    behavior = candidate["behavior"]
    config = json.loads((base_snapshot / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config", {})
    if config.get("architectures") != [behavior["architecture"]]:
        raise PipelineError(f"base architecture {config.get('architectures')} differs from the contract")
    for key in (
        "hidden_size",
        "num_hidden_layers",
        "full_attention_interval",
        "num_key_value_heads",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "head_dim",
    ):
        if text.get(key) != behavior[key]:
            raise PipelineError(f"base text_config.{key} is {text.get(key)!r}, contract {behavior[key]!r}")
    interval, layers = behavior["full_attention_interval"], behavior["num_hidden_layers"]
    expected = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(layers)]
    if text.get("layer_types") != expected:
        raise PipelineError("base layer_types differ from the contract's full_attention_interval pattern")
    return {key: text[key] for key in ("hidden_size", "num_hidden_layers", "layer_types")}


def check_tokenizer(candidate: dict, snapshot: Path) -> dict:
    tokenizer = json.loads((snapshot / candidate["behavior"]["tokenizer_file"]).read_text(encoding="utf-8"))
    added = {token["content"]: token["id"] for token in tokenizer.get("added_tokens", [])}
    observed = {name: added.get(token) for name, token in DELIMITER_TOKENS.items()}
    if observed != candidate["behavior"]["delimiters"] or added.get(PAD[0]) != candidate["behavior"]["pad_token_id"]:
        raise PipelineError(f"pinned tokenizer delimiter/pad ids {observed}, {added.get(PAD[0])} differ from contract")
    return {"delimiters": observed, "pad_token_id": added[PAD[0]]}


def kev_checkout(reference_python: Path, root: Path, timeout: float) -> dict:
    """The Kev source the reference environment imports must be a clean checkout at the pinned commit."""
    found = subprocess.run(
        [str(reference_python), "-I", "-c", "import kev, pathlib; print(pathlib.Path(kev.__file__).resolve().parent)"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if found.returncode:
        raise PipelineError(f"reference Python cannot import kev: {found.stderr.strip()[-500:]}")
    package = Path(found.stdout.strip().splitlines()[-1])

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(package), *args], capture_output=True, text=True, timeout=timeout, check=False
        )
        if result.returncode:
            raise PipelineError(f"kev source at {package} is not a git checkout: {result.stderr.strip()}")
        return result.stdout.strip()

    head, top = git("rev-parse", "HEAD"), git("rev-parse", "--show-toplevel")
    if head != CODE["revision"]:
        raise PipelineError(f"kev checkout {top} is at {head}, contract pins {CODE['revision']}")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise PipelineError(f"kev checkout {top} has local modifications: {dirty[:500]}")
    return {"package": str(package), "checkout": top, "revision": head, "clean": True}


def build_stages(
    recipe: dict,
    *,
    repo_root: Path,
    snapshot: Path,
    reference_python: Path,
    base_snapshot: Path | None = None,
    threads: int = 2,
    timeout: float = 1800,
    distribution: bool = False,
) -> list[Stage]:
    from model_pipeline.onboarding_candidate import candidate_stages, code_inputs, verify_assets

    if distribution:
        raise PipelineError("Kev distribution is not implemented by this plugin")
    if base_snapshot is None:
        raise PipelineError("the Kev route needs --base-snapshot (the pinned base in Hugging Face cache layout)")
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or not 0 < timeout <= 86400:
        raise PipelineError("timeout must be finite, positive, and at most 86400 seconds")
    root, snapshot = repo_root.resolve(), snapshot.resolve()
    base_snapshot = Path(os.path.abspath(base_snapshot))
    manifest, candidate, contract = _profile(recipe, root)
    base = candidate["origin"]["base"]
    cache = _base_cache(base_snapshot, base["repo"], base["revision"])
    if not base_snapshot.is_dir():
        raise PipelineError(f"base snapshot is not a directory: {base_snapshot}")
    corpus = repository_input(root, recipe["reference_corpus"])
    protected = [root / contract["protected_tolerances"]] if contract["protected_tolerances"] else []
    _, initial, shared = candidate_stages(
        recipe,
        repo_root=root,
        snapshot=snapshot,
        reference_python=reference_python,
        threads=threads,
        timeout=timeout,
        source_files=(
            RUNNER,
            "model-pipeline/reference/kev/kev_merge.py",
            "model-pipeline/reference/kev/kev_ort.py",
            "model-pipeline/src/model_pipeline/onboarding_kev.py",
            "model-pipeline/src/model_pipeline/evaluate_decision.py",
            *([contract["protected_tolerances"]] if protected else []),
        ),
    )
    python = Path(os.path.abspath(reference_python))
    env = {
        "PYTHONPATH": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_CACHE": str(cache),
        "TORCH_FORCE_WEIGHTS_ONLY_LOAD": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "DOTNET_PROCESSOR_COUNT": str(threads),
        "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
    }
    parameters = {"plugin": PLUGIN, "candidate": candidate, "threads": threads, "timeout": timeout}

    def guard(out: Path) -> None:
        guard_output(
            out,
            (snapshot, base_snapshot, root / "model-pipeline/protected", corpus, manifest),
            "stage output overlaps original source or protected evidence",
        )

    def verified(stage: Stage, *, base_too: bool) -> Stage:
        """Recheck the pins of everything the stage reads before it runs."""
        original = stage.execute

        def checked(out: Path, deps: dict) -> dict:
            guard(out)
            lock = verify_assets(candidate, snapshot, out)
            records = verify_base(candidate, base_snapshot) if base_too else []
            result = original(out, deps)
            return {
                **result,
                **lock,
                "base_files_verified": len(records),
                "original_source_pins": "verified-before-read",
            }

        return replace(stage, execute=checked)

    assets_stage, environment_stage = initial
    original_assets, original_environment = assets_stage.execute, environment_stage.execute

    def assets(out: Path, deps: dict) -> dict:
        result = original_assets(out, deps)
        lock = json.loads((out / "source-lock.json").read_text(encoding="utf-8"))
        lock["base"] = {
            "repo": base["repo"],
            "revision": base["revision"],
            "assets": verify_base(candidate, base_snapshot),
        }
        lock["code"] = candidate["origin"]["code"]
        lock["contract_checks"] = {
            "base_config": check_base_config(candidate, base_snapshot),
            "tokenizer": check_tokenizer(candidate, snapshot),
        }
        write_json(out / "source-lock.json", lock)
        return {**result, "base_files_verified": len(lock["base"]["assets"])}

    def environment(out: Path, deps: dict) -> dict:
        result = original_environment(out, deps)
        checkout = kev_checkout(python, root, timeout)
        write_json(out / "kev-source.json", checkout)
        return {**result, "kev_source": checkout["revision"]}

    def runner(out: Path, stage: str, extra: list[str]) -> dict:
        guard(out)
        return run_command(
            [
                str(python),
                str(root / RUNNER),
                stage,
                "--profile",
                str(manifest),
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
        return runner(out, "reference", ["--snapshot", str(snapshot), "--corpus", str(corpus)])

    def build(out: Path, deps: dict) -> dict:
        return runner(out, "build", ["--snapshot", str(snapshot), "--base-snapshot", str(base_snapshot)])

    def onnx(out: Path, deps: dict) -> dict:
        return runner(
            out,
            "onnx",
            [
                "--reference",
                str(deps["reference"] / "reference.json"),
                "--bundle",
                str(deps["build"] / "bundle"),
                "--build",
                str(deps["build"] / "build.json"),
            ],
        )

    def dotnet(out: Path, deps: dict) -> dict:
        guard(out)
        project = root / TOOL
        if not project.is_file():
            raise PipelineError(f"Kev .NET fidelity tool is missing: {TOOL}")
        return run_command(
            [
                shutil.which("dotnet") or "dotnet",
                "run",
                "--project",
                str(project),
                "-c",
                "Release",
                "--",
                "--profile",
                str(manifest),
                "--snapshot",
                str(deps["build"] / "bundle"),
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

    def evaluation(out: Path, deps: dict) -> dict:
        guard(out)
        report = evaluate(
            candidate,
            manifest,
            deps["reference"] / "reference.json",
            deps["build"] / "build.json",
            deps["onnx"] / "onnx.json",
            deps["dotnet"] / "dotnet.json",
        )
        write_json(out / "report.json", report)
        (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
        if report["status"] != "passed":
            raise PipelineError(
                f"Kev development parity failed: {'; '.join(report['failures'])}; see {out / 'report.json'}"
            )
        return {"status": report["status"], "qualification_status": "unapproved", "reference_status": "proposed"}

    def review(out: Path, deps: dict) -> dict:
        guard(out)
        report = json.loads((deps["evaluate"] / "report.json").read_text(encoding="utf-8"))
        if report["status"] != "passed":
            raise PipelineError("review requires complete passed development parity")
        for name in ("report.json", "report.md"):
            shutil.copyfile(deps["evaluate"] / name, out / name)
        build_record = json.loads((deps["build"] / "build.json").read_text(encoding="utf-8"))
        document = review_document(candidate, contract, report, build_record, deps, root)
        write_json(out / "review.json", document)
        (out / "review.md").write_text(render_review(document), encoding="utf-8")
        return {"development_parity": "passed", "release_qualification": "unapproved"}

    def stage(name: str, dependencies: tuple[str, ...], execute, inputs: dict | None = None, **extra) -> Stage:
        return Stage(name, dependencies, {**shared, **(inputs or {})}, {**parameters, **extra}, execute)

    dotnet_directories = tuple(
        relative
        for relative in (
            "ai-dotnet/tools/Starlings.Kev.Fidelity.Tool",
            "ai-dotnet/src/Starlings.Models.Kev",
            "ai-dotnet/src/Starlings.Runtime.Onnx",
        )
        if (root / relative).is_dir()
    )
    dotnet_inputs = code_inputs(
        root,
        files=(
            "global.json",
            "Directory.Build.props",
            "ai-dotnet/Directory.Build.props",
            "ai-dotnet/Directory.Build.targets",
            "ai-dotnet/Directory.Packages.props",
        ),
        directories=dotnet_directories,
    )
    return [
        replace(assets_stage, inputs={**assets_stage.inputs, "base_snapshot": base_snapshot}, execute=assets),
        replace(environment_stage, execute=environment),
        verified(stage("reference", ("assets", "environment"), reference, {"corpus": corpus}), base_too=True),
        verified(stage("build", ("assets", "environment"), build), base_too=True),
        stage("onnx", ("reference", "build", "environment"), onnx),
        stage("dotnet", ("reference", "build", "environment"), dotnet, dotnet_inputs),
        stage("evaluate", ("reference", "build", "onnx", "dotnet"), evaluation),
        stage("review", ("assets", "build", "evaluate"), review, candidate_commit=fixtures.git_head(root)),
    ]


def review_document(candidate: dict, contract: dict, report: dict, build: dict, deps: dict, root: Path) -> dict:
    return {
        "profile": candidate["id"],
        "task": candidate["task"],
        "development_parity": report,
        "bundle": build["bundle"],
        "graph_ops": build["graph"]["ops"],
        "policy": candidate["evaluation"]["policy"],
        "policy_source": contract["protected_tolerances"] or "proposed (Kev-0.8B binding values, unchanged)",
        "capabilities": candidate["behavior"]["capabilities"],
        "candidate_commit": fixtures.git_head(root),
        "source_lock": json.loads((deps["assets"] / "source-lock.json").read_text(encoding="utf-8")),
        "support_status": "experimental",
        "release_qualification": "unapproved",
        "reference_status": "proposed",
        "protected_fixture_adoption": False,
        "distribution": "not-run",
    }


def render_review(document: dict) -> str:
    report = document["development_parity"]
    worst = {check["id"]: check for check in report["checks"]}
    lines = [
        f"# Candidate review: {document['profile']}",
        "",
        "Development decision parity passed. References are proposed; bounds come from "
        f"{document['policy_source']}. Protected G1/G2 and full release qualification are unapproved; "
        "distribution is not run.",
        "",
        "| Comparison | Max abs | Bound |",
        "| --- | ---: | ---: |",
        *(f"| {name} | {check['max_abs']:.3e} | {check['bound']:.1e} |" for name, check in worst.items()),
        "",
        "Capabilities: " + ", ".join(f"{name} {status}" for name, status in document["capabilities"].items()) + ".",
        "",
    ]
    return "\n".join(lines)
