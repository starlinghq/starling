"""Independent evaluator for decision-model (Kev) development parity.

Compares per-question option probabilities from the PyTorch reference, Python ONNX Runtime (cached state, uncached
rows, right-padded batches) and the .NET implementation (from reference token ids, cached and uncached, and from
record text) under the fields of the Kev tolerance file (model-pipeline/protected/kev-0.8b/onnx-fp32/tolerances.yaml):

- probability_max_abs: every backend path against the reference;
- cached_vs_uncached_probability_max_abs: cached-state rows against uncached rows, per backend;
- padded_batch_vs_single_probability_max_abs: Python ONNX padded batches against each row alone;
- nan_or_inf: fail on any non-finite or non-numeric probability;
- argmax_flips: 0 against the reference, except questions whose reference top-two margin is below the policy
  exception (each listed); a one-option question has margin 1.0 and cannot flip;
- accuracy: identical to the reference on labelled questions;
- csharp_vs_python probability_max_abs: .NET against Python ONNX Runtime on the same machine and provider;
- report only: mean and p99 of the per-question maximum absolute difference.

.NET token ids must equal the reference ids exactly (state ids and every row's ids, positions and readout offsets).
The evaluator never reads or writes protected fixtures.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from model_pipeline import fixtures
from model_pipeline.onboarding_runner import PipelineError

FORMAT = "starlings.kev-development-parity"
REQUIRED_PATHS = [
    "reference",
    "python-onnx-cached",
    "python-onnx-uncached",
    "python-onnx-padded-batch",
    "dotnet-ids-cached",
    "dotnet-ids-uncached",
    "dotnet-text",
]
POLICY = {
    "probability_max_abs": 2.0e-4,
    "cached_vs_uncached_probability_max_abs": 1.0e-5,
    "padded_batch_vs_single_probability_max_abs": 1.0e-5,
    "nan_or_inf": "fail",
    "report_only": ["mean_abs", "p99_abs"],
    "argmax_flips": 0,
    "argmax_flip_margin_exception": 1.0e-4,
    "accuracy": "identical",
    "csharp_vs_python_probability_max_abs": 1.0e-6,
    "token_ids": "exact",
    "required_paths": REQUIRED_PATHS,
}
# Paths compared with the reference (probability bound, flips and accuracy) and their source keys.
BACKEND_PATHS = {
    "python-onnx-cached": ("onnx", "output"),
    "python-onnx-uncached": ("onnx", "uncached_output"),
    "dotnet-ids-cached": ("dotnet", "output"),
    "dotnet-ids-uncached": ("dotnet", "uncached_output"),
    "dotnet-text": ("dotnet", "text_output"),
}
CACHED_VS_UNCACHED = {
    "python-onnx": ("python-onnx-cached", "python-onnx-uncached"),
    "dotnet": ("dotnet-ids-cached", "dotnet-ids-uncached"),
}
CSHARP_VS_PYTHON = {
    "dotnet-ids-cached": "python-onnx-cached",
    "dotnet-ids-uncached": "python-onnx-uncached",
    "dotnet-text": "python-onnx-cached",
}
REQUIRED_BUNDLE_DIGESTS = ("model.onnx", "model.onnx.data")


def protected_policy(text: str) -> dict:
    """Map the Kev tolerance file (YAML text) onto the candidate policy fields; refuses an unexpected layout."""
    import re

    import yaml

    data = yaml.safe_load(text)
    try:
        floats, discrete = data["float"], data["discrete"]
        margin = re.search(r"argmax_flips:\s*0\s*#.*top-two margin is below ([0-9.eE+-]+)", text)
        if margin is None:
            raise PipelineError("tolerance file does not state the argmax flip margin exception")
        return {
            "probability_max_abs": float(floats["probability_max_abs"]),
            "cached_vs_uncached_probability_max_abs": float(floats["cached_vs_uncached_probability_max_abs"]),
            "padded_batch_vs_single_probability_max_abs": float(floats["padded_batch_vs_single_probability_max_abs"]),
            "nan_or_inf": floats["nan_or_inf"],
            "report_only": list(floats["report_only"]),
            "argmax_flips": int(discrete["argmax_flips"]),
            "argmax_flip_margin_exception": float(margin[1]),
            "accuracy": discrete["accuracy"],
            "csharp_vs_python_probability_max_abs": float(
                data["csharp_vs_python_same_machine_and_provider"]["probability_max_abs"]
            ),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError(f"unexpected Kev tolerance file layout: {exc}") from exc


def _load(path: Path, stage: str) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PipelineError(f"missing or malformed {stage} evidence: {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list) and stage != "build":
        raise PipelineError(f"{stage}: evidence must be an object with a cases list")
    return data


def _finite_probs(values, options: int) -> tuple[list[float] | None, bool]:
    """(probabilities or None for a wrong shape, whether every value is a finite number)."""
    if not isinstance(values, list) or len(values) != options:
        return None, True
    numeric = all(type(value) in (int, float) for value in values)
    finite = numeric and all(math.isfinite(value) for value in values)
    return ([float(value) for value in values] if numeric else None), finite


def _stats(maxima: list[float]) -> dict:
    if not maxima:
        return {"n": 0, "max_abs": 0.0, "mean_abs": 0.0, "p99_abs": 0.0}
    ordered = sorted(maxima)
    rank = 0.99 * (len(ordered) - 1)
    low = math.floor(rank)
    high = min(low + 1, len(ordered) - 1)
    p99 = ordered[low] + (ordered[high] - ordered[low]) * (rank - low)
    return {"n": len(maxima), "max_abs": ordered[-1], "mean_abs": math.fsum(maxima) / len(maxima), "p99_abs": p99}


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def _margin(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    top = sorted(values, reverse=True)
    return top[0] - top[1]


def _compare(name: str, metric: str, bound: float, pairs: list[tuple[str, list[float], list[float]]]) -> dict:
    maxima = [max(abs(a - b) for a, b in zip(left, right, strict=True)) for _, left, right in pairs]
    stats = _stats(maxima)
    worst = pairs[maxima.index(stats["max_abs"])][0] if maxima else None
    return {
        "id": name,
        "metric": metric,
        "bound": bound,
        "status": "passed" if maxima and stats["max_abs"] <= bound else "failed",
        **stats,
        "worst_question": worst,
        "population": "per-question maximum absolute probability difference",
    }


def evaluate(
    candidate: dict, candidate_path: Path, reference_path: Path, build_path: Path, onnx_path: Path, dotnet_path: Path
) -> dict:
    policy = candidate["evaluation"]["policy"]
    if any(policy.get(key) != value for key, value in POLICY.items()):
        raise PipelineError("decision evaluation needs the predeclared Kev policy")
    digest = fixtures.sha256_file(candidate_path)
    reference_digest = fixtures.sha256_file(reference_path)
    reference = _load(reference_path, "reference")
    build = _load(build_path, "build")
    onnx = _load(onnx_path, "onnx")
    dotnet = _load(dotnet_path, "dotnet")
    failures: list[str] = []
    for stage, document in (("reference", reference), ("build", build), ("onnx", onnx), ("dotnet", dotnet)):
        if document.get("profile") != candidate["id"] or document.get("candidate_sha256") != digest:
            failures.append(f"{stage}: names another candidate or contract digest")
    for stage, document in (("onnx", onnx), ("dotnet", dotnet)):
        if document.get("reference_sha256") != reference_digest:
            failures.append(f"{stage}: names a different reference.json")
    try:
        bundle = {name: entry["sha256"] for name, entry in build["bundle"].items()}
    except (KeyError, TypeError, AttributeError) as exc:
        raise PipelineError(f"build: malformed bundle digests: {exc}") from exc
    for stage, document in (("onnx", onnx), ("dotnet", dotnet)):
        environment = document.get("environment")
        models = environment.get("model_sha256") if isinstance(environment, dict) else None
        if (
            not isinstance(models, dict)
            or any(name not in models for name in REQUIRED_BUNDLE_DIGESTS)
            or any(bundle.get(name) != value for name, value in models.items())
        ):
            failures.append(f"{stage}: model_sha256 does not name the built bundle")
    machines = [
        d.get("environment", {}).get("machine") if isinstance(d.get("environment"), dict) else None
        for d in (onnx, dotnet)
    ]
    if any(not isinstance(machine, str) or not machine for machine in machines) or machines[0] != machines[1]:
        failures.append("csharp_vs_python requires the Python and .NET runs on the same named machine")
    if dotnet.get("cancellation_verified") is not True:
        failures.append("dotnet: cancellation was not verified")

    def by_id(document: dict, stage: str) -> dict:
        rows = {}
        for case in document["cases"]:
            identifier = case.get("id") if isinstance(case, dict) else None
            if not isinstance(identifier, str) or not identifier or identifier in rows:
                raise PipelineError(f"{stage}: duplicate or invalid case id {identifier!r}")
            rows[identifier] = case
        return rows

    expected, onnx_cases, dotnet_cases = by_id(reference, "reference"), by_id(onnx, "onnx"), by_id(dotnet, "dotnet")
    if len(expected) < 2 or set(expected) != set(onnx_cases) or set(expected) != set(dotnet_cases):
        raise PipelineError("backend evidence has missing or unknown cases")

    outputs: dict[str, dict[str, list[float]]] = {path: {} for path in [*REQUIRED_PATHS]}
    batches: list[tuple[str, list[int]]] = []
    nonfinite: list[str] = []
    labels: dict[str, int] = {}
    token_mismatches: list[str] = []
    shape_errors: list[str] = []
    for identifier, case in expected.items():
        try:
            questions = case["record"]["questions"]
            rows = case["rows"]
            if len(questions) != len(rows) or len(case["probs"]) != len(rows):
                raise PipelineError(f"reference/{identifier}: questions, rows and probabilities differ in count")
            options = [len(question["options"]) for question in questions]
            if [len(row["opts"]) for row in rows] != options:
                raise PipelineError(f"reference/{identifier}: readout offsets do not match the options")
        except (KeyError, TypeError) as exc:
            raise PipelineError(f"reference/{identifier}: malformed case: {exc}") from exc
        sources = {
            "reference": case["probs"],
            **{
                path: (onnx_cases if stage == "onnx" else dotnet_cases)[identifier].get(key)
                for path, (stage, key) in BACKEND_PATHS.items()
            },
        }
        batch = onnx_cases[identifier].get("batch")
        if len(rows) > 1:
            if not isinstance(batch, dict):
                shape_errors.append(f"python-onnx-padded-batch/{identifier}: missing padded batch")
            else:
                sources["python-onnx-padded-batch"] = batch.get("output")
                lengths = batch.get("row_lengths")
                expected_lengths = [len(case["state_ids"]) + len(row["ids"]) for row in rows]
                if lengths != expected_lengths or batch.get("width") != max(expected_lengths):
                    shape_errors.append(f"python-onnx-padded-batch/{identifier}: wrong row lengths or width")
                else:
                    batches.append((identifier, lengths))
        for path, values in sources.items():
            if not isinstance(values, list) or len(values) != len(rows):
                shape_errors.append(f"{path}/{identifier}: wrong number of questions")
                continue
            for index, (probabilities, count) in enumerate(zip(values, options, strict=True)):
                key = f"{identifier}/{questions[index]['id']}"
                parsed, finite = _finite_probs(probabilities, count)
                if parsed is None:
                    shape_errors.append(f"{path}/{key}: wrong shape or non-numeric output")
                elif not finite:
                    nonfinite.append(f"{path}/{key}")
                else:
                    outputs[path][key] = parsed
        for question in questions:
            if question.get("label") is not None:
                labels[f"{identifier}/{question['id']}"] = question["label"]
        observed = dotnet_cases[identifier]
        if observed.get("state_ids") != case["state_ids"]:
            token_mismatches.append(f"{identifier}: state_ids")
        if observed.get("rows") != rows:
            token_mismatches.append(f"{identifier}: rows")
    if shape_errors:
        failures.extend(shape_errors)
    if not any(len(set(lengths)) > 1 for _, lengths in batches):
        failures.append("python-onnx-padded-batch: no batch of differently sized rows")

    questions_all = list(outputs["reference"])

    def pairs(left: str, right: str) -> list[tuple[str, list[float], list[float]]]:
        keys = [key for key in questions_all if key in outputs[left] and key in outputs[right]]
        return [(key, outputs[left][key], outputs[right][key]) for key in keys]

    checks = []
    for path in BACKEND_PATHS:
        checks.append(
            _compare(
                f"{path}/vs_reference", "probability_max_abs", policy["probability_max_abs"], pairs(path, "reference")
            )
        )
    for name, (cached, uncached) in CACHED_VS_UNCACHED.items():
        checks.append(
            _compare(
                f"{name}/cached_vs_uncached",
                "cached_vs_uncached_probability_max_abs",
                policy["cached_vs_uncached_probability_max_abs"],
                pairs(cached, uncached),
            )
        )
    checks.append(
        _compare(
            "python-onnx/padded_batch_vs_single",
            "padded_batch_vs_single_probability_max_abs",
            policy["padded_batch_vs_single_probability_max_abs"],
            pairs("python-onnx-padded-batch", "python-onnx-uncached"),
        )
    )
    for csharp_path, python_path in CSHARP_VS_PYTHON.items():
        checks.append(
            _compare(
                f"{csharp_path}/vs_{python_path}",
                "csharp_vs_python_probability_max_abs",
                policy["csharp_vs_python_probability_max_abs"],
                pairs(csharp_path, python_path),
            )
        )
    for check in checks:
        if check["status"] != "passed":
            failures.append(f"{check['id']}: {check['metric']} {check['max_abs']:.3g} > {check['bound']:.3g}")

    exception = policy["argmax_flip_margin_exception"]
    flips: dict[str, dict] = {}
    for path in [*BACKEND_PATHS, "python-onnx-padded-batch"]:
        listed, outside = [], []
        for key, actual, wanted in pairs(path, "reference"):
            if _argmax(actual) != _argmax(wanted):
                entry = {"question": key, "reference_margin": _margin(wanted)}
                (listed if entry["reference_margin"] < exception else outside).append(entry)
        flips[path] = {"outside_margin": outside, "listed_near_ties": listed}
        if len(outside) > policy["argmax_flips"]:
            failures.append(f"{path}: {len(outside)} argmax flips with reference margin >= {exception:g}")
    report_only_flips = {
        f"{csharp_path}/vs_{python_path}": [
            key for key, left, right in pairs(csharp_path, python_path) if _argmax(left) != _argmax(right)
        ]
        for csharp_path, python_path in CSHARP_VS_PYTHON.items()
    }

    accuracy: dict[str, dict] = {}
    if labels:
        for path in ["reference", *BACKEND_PATHS]:
            scored = [key for key in labels if key in outputs[path]]
            correct = sum(_argmax(outputs[path][key]) == labels[key] for key in scored)
            accuracy[path] = {
                "n": len(scored),
                "correct": correct,
                "accuracy": correct / len(scored) if scored else None,
            }
        for path in BACKEND_PATHS:
            if accuracy[path] != accuracy["reference"]:
                failures.append(f"{path}: accuracy differs from the reference")
        accuracy_status = (
            "identical" if not any(f.endswith("accuracy differs from the reference") for f in failures) else "differs"
        )
    else:
        accuracy_status = "no labelled questions"
    if nonfinite:
        failures.append(f"nan_or_inf: {len(nonfinite)} outputs are non-finite")
    if token_mismatches:
        failures.append(f"dotnet token ids differ from the reference: {', '.join(token_mismatches)}")
    return {
        "format": FORMAT,
        "format_version": 1,
        "profile": candidate["id"],
        "task": candidate["task"],
        "status": "failed" if failures else "passed",
        "failures": failures,
        "checks": checks,
        "argmax_flips": flips,
        "report_only": {"csharp_vs_python_argmax_flips": report_only_flips},
        "accuracy": {"status": accuracy_status, "labelled_questions": len(labels), "paths": accuracy},
        "nan_or_inf": nonfinite,
        "token_ids": {"status": "identical" if not token_mismatches else "differ", "mismatches": token_mismatches},
        "case_count": len(expected),
        "question_count": len(questions_all),
        "padded_batches": len(batches),
        "policy": policy,
        "evaluation_status": candidate["evaluation"]["status"],
        "protected_release_qualification": {"status": "unapproved", "G1": "not-run", "G2": "not-run"},
        "environments": {"onnx": onnx.get("environment"), "dotnet": dotnet.get("environment")},
        "inputs": {
            "candidate_sha256": digest,
            "reference_sha256": reference_digest,
            "build_sha256": fixtures.sha256_file(build_path),
            "onnx_sha256": fixtures.sha256_file(onnx_path),
            "dotnet_sha256": fixtures.sha256_file(dotnet_path),
            "bundle_sha256": bundle,
        },
    }


def render_markdown(report: dict) -> str:
    lines = [
        f"# Decision development parity: {report['profile']}",
        "",
        f"Status: **{report['status']}**; {report['case_count']} records, {report['question_count']} questions, "
        f"{report['padded_batches']} padded batches. Policy status: **{report['evaluation_status']}**.",
        "",
        "| Check | Bound | Max abs | Mean abs | p99 abs | Status |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for check in report["checks"]:
        lines.append(
            f"| {check['id']} | {check['bound']:.1e} | {check['max_abs']:.3e} | {check['mean_abs']:.3e} | "
            f"{check['p99_abs']:.3e} | {check['status']} |"
        )
    lines += ["", "Argmax flips against the reference (outside / listed near ties):", ""]
    for path, entry in report["argmax_flips"].items():
        listed = ", ".join(
            f"{item['question']} (margin {item['reference_margin']:.2e})" for item in entry["listed_near_ties"]
        )
        counts = f"{len(entry['outside_margin'])} / {len(entry['listed_near_ties'])}"
        lines.append(f"- {path}: {counts}{': ' + listed if listed else ''}")
    accuracy = report["accuracy"]
    lines += ["", f"Accuracy on {accuracy['labelled_questions']} labelled questions: {accuracy['status']}."]
    lines.append(f"Token ids (.NET against reference): {report['token_ids']['status']}.")
    lines.append(f"Non-finite outputs: {len(report['nan_or_inf'])}.")
    if report["failures"]:
        lines += ["", "Failures:", "", *(f"- {failure}" for failure in report["failures"])]
    lines += [
        "",
        "Mean and p99 are report-only, over per-question maximum absolute differences. "
        "Protected G1/G2 release qualification is **unapproved**; no protected fixtures were adopted.",
        "",
    ]
    return "\n".join(lines)
