"""Command-line entry point. Stages are added as backlog items land (plan section 8)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from model_pipeline import benchmarks
from model_pipeline.manifest import find_manifests, validate_file

# Options a caller might use to supply thresholds. They exist only to be refused with a clear reason.
REFUSED_OPTIONS = ("--tolerances", "--tolerance", "--threshold", "--thresholds", "--bound", "--bounds")


def _validate(args: argparse.Namespace) -> int:
    manifests = [m for root in args.paths for m in find_manifests(Path(root))]
    failed = 0
    for path in manifests:
        problems = validate_file(path)
        for problem in problems:
            print(problem, file=sys.stderr)
        failed += bool(problems)
    print(f"{len(manifests)} manifest(s) checked, {failed} invalid")
    return 1 if failed else 0


class _RefuseOverride(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        from model_pipeline.evaluate import EXIT_PROTECTED

        print(
            f"error: {option_string} is refused. The evaluator reads thresholds only from the profile's protected "
            "tolerances file (manifest qualification.tolerances under model-pipeline/protected/). Changing a bound "
            "is a separate pull request with maintainer review (plan 10.1).",
            file=sys.stderr,
        )
        parser.exit(EXIT_PROTECTED)


def _evaluate_g3(args: argparse.Namespace) -> int:
    from model_pipeline import evaluate, evaluate_task_quality, protection, results

    if not args.dataset:
        print("error: --g3 needs --dataset", file=sys.stderr)
        return evaluate.EXIT_INPUT
    try:
        evaluation = evaluate_task_quality.evaluate(
            Path(args.manifest),
            args.g3,
            Path(args.dataset),
            Path(args.results),
            baseline=Path(args.baseline) if args.baseline else None,
            repo_root=Path(args.repo_root) if args.repo_root else None,
            protected_base=args.protected_base,
        )
    except (protection.ProtectedEvidenceError, results.ToleranceOverrideError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return evaluate.EXIT_PROTECTED
    except (evaluate.EvaluationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return evaluate.EXIT_INPUT
    json_path, md_path = evaluate_task_quality.write_report(evaluation, Path(args.out))
    r = evaluation.report
    shown = {k: v for k, v in r["metrics"].items() if ".label." not in k}
    print(f"{r['profile']} {r['task']}: " + ", ".join(f"{k} {v:.4f}" for k, v in shown.items()))
    print(f"G3 {r['gates']['G3']['status']}: {r['gates']['G3']['reason']}")
    print(f"report: {json_path} {md_path}")
    return evaluation.exit_code


def _evaluate(args: argparse.Namespace) -> int:
    from model_pipeline import evaluate, protection, results

    if args.g3:
        return _evaluate_g3(args)
    if args.dataset or args.baseline:
        print("error: --dataset and --baseline apply to --g3 only", file=sys.stderr)
        return evaluate.EXIT_INPUT
    try:
        evaluation = evaluate.evaluate(
            Path(args.manifest),
            Path(args.results),
            repo_root=Path(args.repo_root) if args.repo_root else None,
            protected_base=args.protected_base,
            require_gates=tuple(g for g in args.require_gates.split(",") if g),
            reference_tensors=Path(args.reference_tensors) if args.reference_tensors else None,
        )
    except (protection.ProtectedEvidenceError, results.ToleranceOverrideError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return evaluate.EXIT_PROTECTED
    except (evaluate.EvaluationError, results.ResultsError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return evaluate.EXIT_INPUT
    json_path, md_path = evaluate.write_report(evaluation, Path(args.out))
    r = evaluation.report
    print(
        f"{r['profile']}: first divergence {r['first_divergence'] or 'none'}; "
        + ", ".join(f"{g} {v['status']}" for g, v in r["gates"].items())
        + f"; protected evidence {r['protected_evidence']['status']}"
    )
    for finding in r["findings"]:
        print(f"  {finding['stage']}: [{finding['taxonomy']}] {finding['finding']}")
    print(f"report: {json_path} {md_path}")
    return evaluation.exit_code


def _check_protected(args: argparse.Namespace) -> int:
    from model_pipeline import protection

    if args.files is not None:
        paths = [p for p in args.files.split("\n") if p]
    else:
        root = protection.find_repository_root(Path.cwd())
        paths = protection.changed_files(root, args.base, args.head)
    labels = json.loads(args.labels_json) if args.labels_json else []
    if labels and isinstance(labels[0], dict):
        labels = [label["name"] for label in labels]
    result = protection.check_changes(paths, labels)
    print(f"{len(paths)} changed file(s); protected: {len(result.protected)}; labels: {labels}")
    for problem in result.problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    return 0 if result.ok else 1


def _catalog(args: argparse.Namespace) -> int:
    from model_pipeline import catalog

    registry, out, repo_root = Path(args.registry), Path(args.out), Path(args.repo_root)
    try:
        files = catalog.generate(registry, out, repo_root)
    except catalog.CatalogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.check:
        stale = catalog.stale(files, out)
        for name in stale:
            print(f"STALE: {out / name}", file=sys.stderr)
        if stale:
            print("The catalog is stale. Regenerate it with: model-pipeline catalog", file=sys.stderr)
            return 1
        print(f"catalog up to date: {len(files)} file(s) in {out}")
        return 0
    catalog.write(files, out)
    print(f"catalog written: {', '.join(str(out / name) for name in files)}")
    return 0


def _bench_validate(args: argparse.Namespace) -> int:
    results, problems = benchmarks.load([Path(p) for p in args.paths])
    for problem in problems:
        print(problem, file=sys.stderr)
    print(f"{len(results.runs)} run(s), {len(results.series)} series, {len(problems)} problem(s)")
    return 1 if problems else 0


def _bench_summarize(args: argparse.Namespace) -> int:
    results, problems = benchmarks.load([Path(p) for p in args.paths])
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    summary = benchmarks.summarize(results, resamples=args.resamples, seed=args.seed)
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    sys.stdout.write(benchmarks.markdown(summary))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="model-pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    from model_pipeline.onboarding_cli import register

    register(sub)

    validate = sub.add_parser("validate", help="validate model manifests against the schema")
    validate.add_argument("paths", nargs="+", help="manifest files or directories to search")
    validate.set_defaults(func=_validate)

    ev = sub.add_parser(
        "evaluate",
        help="staged fidelity evaluation of candidate results against a profile's protected fixtures",
        description="Exit codes: 0 no failure; 1 a check failed; 2 bad input; 3 protected evidence changed, "
        "outside model-pipeline/protected/, or a threshold override was attempted; 4 a --require-gates gate "
        "did not pass, or the profile has no protected tolerances file yet (measured, never qualified).",
    )
    ev.add_argument("manifest", help="registry manifest.yaml of the profile")
    ev.add_argument(
        "--results",
        required=True,
        help="directory with *.results.json files, one per stage (task generation: the output directory of "
        "Starlings.Qwen35.Fidelity; --g3: the candidate's embeddings or predictions JSONL file)",
    )
    ev.add_argument("--out", required=True, help="directory for report.json and report.md")
    ev.add_argument(
        "--protected-base",
        default="auto",
        help="commit the protected files must equal: 'auto' (merge base with origin/main), a commit-ish, or "
        "'none' (unverified; gates cannot pass)",
    )
    ev.add_argument("--repo-root", help="repository root (default: found above the manifest)")
    ev.add_argument("--require-gates", default="", help="comma-separated gates that must pass, e.g. G1,G2")
    ev.add_argument(
        "--reference-tensors",
        help="tasks image-embedding and video-embedding: directory of the reference .npz tensors (outside Git; "
        "each array is checked against the protected tensors.json) for the soft-token and token-embedding checks; "
        "task generation: directory with the reference step0/<id>.f32 vectors (checked against the fixture digests) "
        "for the full-vocabulary step-0 bound",
    )
    ev.add_argument(
        "--g3",
        choices=("retrieval", "ner"),
        help="task-quality evaluation (G3) instead of staged fidelity: metrics on --dataset, compared with the "
        "budget in the profile's protected task-quality.yaml only (absent: no budget, not a gate result)",
    )
    ev.add_argument("--dataset", help="--g3 retrieval: dataset directory; --g3 ner: gold JSONL file")
    ev.add_argument(
        "--baseline",
        help="--g3 retrieval: embeddings JSONL of the fp32 reference profile, for the top-10 overlap metric",
    )
    for option in REFUSED_OPTIONS:
        ev.add_argument(option, nargs="?", action=_RefuseOverride, help=argparse.SUPPRESS)
    ev.set_defaults(func=_evaluate)

    cp = sub.add_parser(
        "check-protected",
        help="fail a pull request that changes protected evidence without the label, or together with code",
    )
    cp.add_argument("--base", default="origin/main")
    cp.add_argument("--head", default="HEAD")
    cp.add_argument("--labels-json", default=os.environ.get("PR_LABELS", ""), help="JSON list of label names")
    cp.add_argument("--files", help="newline-separated changed paths (default: git diff base...head)")
    cp.set_defaults(func=_check_protected)

    from model_pipeline.catalog import DEFAULT_OUT, DEFAULT_REGISTRY, REPO_ROOT

    cat = sub.add_parser(
        "catalog",
        help="generate the public catalog and compatibility matrix from the registry manifests (P-021)",
        description="Writes README.md, compatibility-matrix.md and catalog.json. Exit codes: 0 written or up to "
        "date; 1 --check found stale or missing output; 2 an invalid manifest or an empty registry.",
    )
    cat.add_argument("registry", nargs="?", default=str(DEFAULT_REGISTRY), help="registry directory")
    cat.add_argument("--out", default=str(DEFAULT_OUT), help="output directory (default: docs/catalog)")
    cat.add_argument("--repo-root", default=str(REPO_ROOT), help="repository root that evidence paths are relative to")
    cat.add_argument("--check", action="store_true", help="write nothing; exit 1 when the committed output is stale")
    cat.set_defaults(func=_catalog)

    bench = sub.add_parser("bench", help="raw benchmark results (starlings-bench-raw/1, P-008)")
    bench_sub = bench.add_subparsers(dest="bench_command", required=True)
    bench_validate = bench_sub.add_parser("validate", help="validate raw JSON lines files against the schema")
    bench_validate.add_argument("paths", nargs="+")
    bench_validate.set_defaults(func=_bench_validate)
    bench_summarize = bench_sub.add_parser("summarize", help="medians, p5-p95, bootstrap CIs, C# vs Python")
    bench_summarize.add_argument("paths", nargs="+")
    bench_summarize.add_argument("--resamples", type=int, default=1000, help="bootstrap resamples (default 1000)")
    bench_summarize.add_argument("--seed", type=int, default=1)
    bench_summarize.add_argument("--json", help="also write the summary as JSON to this path")
    bench_summarize.set_defaults(func=_bench_summarize)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
