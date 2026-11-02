"""Command-line integration for reviewed, recipe-driven onboarding."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from model_pipeline import protection


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _positive(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _overlap(left: Path, right: Path) -> bool:
    return left.is_relative_to(right) or right.is_relative_to(left)


def _output_path(path: Path, repo_root: Path, *, inputs: tuple[Path, ...] = ()) -> Path:
    """Keep output away from source and inputs, even through symlink aliases."""
    output = path.absolute()
    if any(part.is_symlink() for part in (output, *output.parents)):
        raise ValueError(f"output path must not contain symbolic links: {path}")
    output = output.resolve()
    root = repo_root.resolve()
    if output == root or root.is_relative_to(output):
        raise ValueError("output cannot contain the repository")
    if output.is_relative_to(root) and not any(
        output.is_relative_to(root / name) and output != root / name for name in ("artifacts", ".models")
    ):
        raise ValueError("outputs inside the repository must be below artifacts/ or .models/")
    for source in inputs:
        if _overlap(output, source.resolve()):
            raise ValueError(f"output overlaps an input: {source}")
    return output


def _load(args: argparse.Namespace) -> tuple[Path, dict]:
    from model_pipeline.onboarding_recipe import load_recipe

    recipe_path = Path(args.recipe).resolve()
    root = Path(args.repo_root).resolve() if args.repo_root else protection.find_repository_root(recipe_path.parent)
    return root, load_recipe(recipe_path, root)


def _inspect(args: argparse.Namespace) -> int:
    from model_pipeline.onboarding_recipe import inspect_snapshot

    _json(inspect_snapshot(Path(args.snapshot)))
    return 0


def _scaffold(args: argparse.Namespace) -> int:
    from model_pipeline.scaffold import scaffold

    root = Path(args.repo_root).resolve() if args.repo_root else protection.find_repository_root(Path.cwd())
    _json(scaffold(root, args.candidate, corpus=args.corpus, write=args.write, token_type_ids=args.token_type_ids))
    return 0


def _workflow(args: argparse.Namespace) -> int:
    from model_pipeline.onboarding_workflow import run_workflow

    path = Path(args.workflow).resolve()
    root = Path(args.repo_root).resolve() if args.repo_root else protection.find_repository_root(path.parent)
    result = run_workflow(
        path,
        Path(args.bindings).resolve(),
        root=root,
        output=Path(args.out),
        threads=args.threads,
        timeout=args.timeout,
        resume=args.resume,
        distribution=args.distribution,
    )
    _json(result)
    return 0 if result["status"] == "completed" else 1


def _plan(args: argparse.Namespace) -> int:
    from model_pipeline.onboarding_plugins import get_plugin, uses_shared_distribution

    root, recipe = _load(args)
    plugin = get_plugin(recipe)
    plan = plugin.describe_plan(recipe, repo_root=root)
    if uses_shared_distribution(plugin):
        # The plugin's own plan describes only its stages; the shared distribution stage is composed here.
        optional = list(plan.get("optional_stages", []))
        if "distribution" not in optional:
            optional.append("distribution")
        plan["optional_stages"] = optional
    _json(plan)
    return 0


def _run(args: argparse.Namespace) -> int:
    from model_pipeline.onboarding_plugins import build_stages
    from model_pipeline.onboarding_runner import run_pipeline

    root, recipe = _load(args)
    snapshot = Path(args.snapshot).resolve()
    output = _output_path(
        Path(args.out),
        root,
        inputs=(
            snapshot,
            *((Path(args.base_snapshot).absolute(),) if args.base_snapshot else ()),
            Path(args.recipe),
            Path(args.reference_python),
            *(root / recipe[key] for key in ("manifest", "reference_corpus", "reference_environment")),
        ),
    )
    # A second pinned source (an adapter's base checkpoint) only for routes that declare one.
    base_snapshot = {"base_snapshot": Path(args.base_snapshot).absolute()} if args.base_snapshot else {}
    stages = build_stages(
        recipe,
        recipe_path=Path(args.recipe),
        **base_snapshot,
        repo_root=root,
        snapshot=snapshot,
        reference_python=Path(args.reference_python).absolute(),
        threads=args.threads,
        timeout=args.timeout,
        distribution=args.distribution,
    )
    result = run_pipeline(
        stages,
        output,
        provenance={"recipe": recipe, "recipe_path": str(Path(args.recipe).resolve()), "repo_root": str(root)},
        resume=args.resume,
    )
    if result["status"] != "completed":
        failures = [
            {"stage": name, "failure": stage.get("failure"), "output_dir": stage.get("output_dir")}
            for name, stage in result["stages"].items()
            if stage["status"] == "failed"
        ]
        task = {
            "format": "starlings.onboarding-failure-task",
            "recipe": recipe,
            "run_record": str(output / "run.json"),
            "failures": failures,
            "repair_attempt_limit": 2,
            "constraints": [
                "Do not modify protected fixtures or tolerances to make implementation pass.",
                "Propose missing shared capabilities as reviewed code changes.",
                "Re-run the recipe and require its independent evaluator to pass.",
            ],
        }
        task_path = output / f"failure-task-{result['run_id']}.json"
        with task_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(task, indent=2) + "\n")
    _json(
        {
            "status": result["status"],
            "run_record": str(output / "run.json"),
            "stages": {
                name: {key: stage.get(key) for key in ("status", "cache_status", "failure") if key in stage}
                for name, stage in result["stages"].items()
            },
        }
    )
    return 0 if result["status"] == "completed" else 1


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"invalid artifact path: {value!r}")
    return path


def _propose(args: argparse.Namespace) -> int:
    """Export verified evidence; never modify the registry or publish a PR."""
    from model_pipeline.onboarding_runner import verify_run_outputs

    run_dir = Path(args.run_dir).resolve()
    record_path = run_dir / "run.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    verify_run_outputs(record, run_dir)
    root = Path(record["provenance"]["repo_root"])
    output = _output_path(Path(args.out), root, inputs=(run_dir,))
    if output.exists():
        raise ValueError(f"proposal output already exists: {output}")
    files: list[tuple[Path, Path]] = []
    for name, stage in record["stages"].items():
        _safe_relative(name)
        if stage.get("status") != "completed":
            raise ValueError(f"stage {name} is incomplete")
        source = Path(stage["output_dir"])
        if not source.resolve().is_relative_to(run_dir):
            raise ValueError(f"stage {name} output is outside the run")
        for relative, expected in stage["outputs"].items():
            if relative == "." and expected.get("type") == "directory":
                continue
            path = source / _safe_relative(relative)
            if any(part.is_symlink() for part in (path, *path.parents)):
                raise ValueError(f"artifact contains a symbolic link: {path}")
            if expected.get("type") == "directory":
                if not path.is_dir():
                    raise ValueError(f"missing artifact directory: {path}")
                continue
            if not path.is_file():
                raise ValueError(f"missing artifact: {path}")
            files.append((path, Path("stages") / name / relative))
    output.mkdir(parents=True)
    try:
        for source, relative in files:
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        shutil.copyfile(record_path, output / "run.json")
        (output / "README.md").write_text(
            "# Model onboarding review bundle\n\n"
            "This directory contains the verified outputs of a completed onboarding run. "
            "See `stages/review/review.md` for gate results and limitations. "
            "`run.json` preserves the original execution paths and input/output identities; "
            "stage artifacts are copied under `stages/<name>/`. "
            "Completion does not approve publication or promote a model's support status.\n",
            encoding="utf-8",
        )
    except BaseException:
        # A partially written proposal stays visibly incomplete and cannot overwrite a later export.
        (output / "INCOMPLETE").write_text("Export failed; this is not a complete review bundle.\n", encoding="utf-8")
        raise
    _json({"status": "proposed", "output": str(output), "files": len(files) + 2})
    return 0


def _dispatch(args: argparse.Namespace) -> int:
    from model_pipeline.onboarding_recipe import RecipeError
    from model_pipeline.onboarding_runner import PipelineError

    try:
        return args.onboarding_handler(args)
    except (RecipeError, PipelineError, protection.ProtectedEvidenceError, ValueError, OSError, KeyError) as exc:
        print(f"onboarding error: {exc}", file=sys.stderr)
        return 2


def register(subparsers: argparse._SubParsersAction) -> None:
    inspect = subparsers.add_parser("inspect", help="inspect local model metadata without executing upstream code")
    inspect.add_argument("snapshot", help="existing local model snapshot")
    inspect.set_defaults(func=_dispatch, onboarding_handler=_inspect)

    scaffold = subparsers.add_parser(
        "scaffold",
        help="generate a bert-onnx-cpu-v1 candidate's recipe, contract entry, fetch line, README row and test row",
        description="Derives the onboarding inputs of model-pipeline/candidates/<candidate>/profile.json: the recipe, "
        "the entry in routes/bert-onnx-cpu-v1/contracts.json (read by onboarding_bert.py and Starlings.Models.Bert), "
        "the tools/fetch-tokenizer-assets.sh line, the package README row, the contract-test row and a C# runtime "
        "lock. Prints them as JSON; --write applies them. Pieces already present are reported as unchanged. Changes "
        "no gate, tolerance, protected file or maturity. Exit codes: 0 done; 2 refused.",
    )
    scaffold.add_argument(
        "candidate", help="candidate directory name under model-pipeline/candidates/, or its profile.json"
    )
    scaffold.add_argument("--repo-root")
    scaffold.add_argument(
        "--corpus", help="repository-relative reference corpus (default: the task's five-model corpus)"
    )
    scaffold.add_argument("--write", action="store_true", help="write the generated pieces into the repository")
    scaffold.add_argument(
        "--token-type-ids",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="whether the graph takes token_type_ids (default: an existing table entry's value, else the tokenizer "
        "profile's: true for bert-wordpiece-uncased, false for xlmr-unigram)",
    )
    scaffold.set_defaults(func=_dispatch, onboarding_handler=_scaffold)

    plan = subparsers.add_parser("plan", help="validate an onboarding recipe and describe its stages")
    plan.add_argument("recipe")
    plan.add_argument("--repo-root")
    plan.set_defaults(func=_dispatch, onboarding_handler=_plan)

    run = subparsers.add_parser("run", help="execute or resume a recipe and produce declared fidelity evidence")
    run.add_argument("recipe")
    run.add_argument("--repo-root")
    run.add_argument("--snapshot", required=True, help="existing immutable local source snapshot")
    run.add_argument(
        "--base-snapshot", help="existing immutable base snapshot, for routes whose model is built on a pinned base"
    )
    run.add_argument(
        "--reference-python", required=True, help="Python interpreter with the recipe's reference dependencies"
    )
    run.add_argument("--out", required=True, help="dedicated run directory")
    run.add_argument("--threads", type=_positive, default=2)
    run.add_argument("--timeout", type=_positive, default=1800, help="per-subprocess limit in seconds")
    run.add_argument("--resume", action="store_true", help="reuse stages only after input and output verification")
    run.add_argument("--distribution", action="store_true", help="also run the real-model clean-consumer drill")
    run.set_defaults(func=_dispatch, onboarding_handler=_run)

    workflow = subparsers.add_parser("workflow", help="execute independent model recipes and aggregate their evidence")
    workflow.add_argument("workflow")
    workflow.add_argument("--bindings", required=True, help="JSON with local snapshot/reference Python paths per model")
    workflow.add_argument("--repo-root")
    workflow.add_argument("--out", required=True)
    workflow.add_argument("--threads", type=_positive, default=2)
    workflow.add_argument("--timeout", type=_positive, default=1800)
    workflow.add_argument("--resume", action="store_true")
    workflow.add_argument("--distribution", action="store_true")
    workflow.set_defaults(func=_dispatch, onboarding_handler=_workflow)

    propose = subparsers.add_parser("propose", help="export verified run artifacts into a new review directory")
    propose.add_argument("run_dir")
    propose.add_argument("--out", required=True)
    propose.set_defaults(func=_dispatch, onboarding_handler=_propose)
