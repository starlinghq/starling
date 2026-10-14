"""Raw benchmark results (starlings-bench-raw/1): validation and summaries (P-008, plan section 11).

Every harness writes JSON lines: one ``run`` record per process (artifacts, environment, settings, quality) and
``series`` records with raw per-sample timings. This module validates those files and computes the published
statistics from the raw samples:

- medians, p5 and p95 per series group, with sample counts;
- 95% bootstrap confidence intervals for the median, resampling runs (independent processes) and then samples
  within each run, so run-to-run variation widens the interval instead of being hidden by pooling;
- p99 only when a group has at least ``P99_MIN_SAMPLES`` samples, p5/p95 only with at least ``TAIL_MIN_SAMPLES``;
- C#-versus-Python ratios of medians with bootstrap intervals (the plan's headline comparison), and the adapter
  overhead budget (library versus direct C# at p95: no more than the larger of 1 ms or 10%);
- medians of per-sample metrics other than time (generation harnesses: time to first token, tokens per second).

Embedding harnesses rate each series by ``max_abs`` against the run's ``embedding_max_abs_tolerance``; the G5 model
harnesses (entities, generation) record their own verdict in ``quality.pass``.

Standard library only, so the summarizer runs without NumPy.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "benchmark-raw.schema.json"
SCHEMA_ID = "starlings-bench-raw/1"
P99_MIN_SAMPLES = 1000  # ~10 samples beyond the 99th percentile
TAIL_MIN_SAMPLES = 20  # at least one sample beyond p5/p95
OVERHEAD_BUDGET_MS = 1.0
OVERHEAD_BUDGET_FRACTION = 0.10


def _validators() -> dict[str, Draft202012Validator]:
    """One validator per record kind, so errors name the failing field instead of 'not valid under oneOf'."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return {
        kind: Draft202012Validator({"$defs": schema["$defs"], "$ref": f"#/$defs/{kind}"}) for kind in ("run", "series")
    }


@dataclass
class RawResults:
    runs: dict[str, dict] = field(default_factory=dict)
    series: list[dict] = field(default_factory=list)


def load(paths: list[Path]) -> tuple[RawResults, list[str]]:
    """Reads and validates raw files. Returns the records and a list of problems (empty when valid)."""
    validators = _validators()
    results = RawResults()
    problems: list[str] = []
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
            if not line.strip():
                continue
            where = f"{path}:{number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append(f"{where}: not JSON ({exc})")
                continue
            kind = record.get("kind") if isinstance(record, dict) else None
            if kind not in validators:
                problems.append(f"{where}: kind must be 'run' or 'series', got {kind!r}")
                continue
            errors = sorted(validators[kind].iter_errors(record), key=lambda e: list(e.absolute_path))
            if errors:
                problems.extend(f"{where}: {e.message}" for e in errors[:5])
                continue
            if record["kind"] == "run":
                results.runs[record["run_id"]] = record
            else:
                problems.extend(f"{where}: {p}" for p in _series_problems(record))
                results.series.append(record)
    for s in results.series:
        if s["run_id"] not in results.runs:
            problems.append(f"series {s['workload']} refers to unknown run {s['run_id']}")
    return results, problems


def _series_problems(s: dict) -> list[str]:
    problems = []
    if len(s["ms"]) != s["n"]:
        problems.append(f"n={s['n']} but {len(s['ms'])} samples")
    for name, values in s["stages_ms"].items():
        if len(values) != len(s["ms"]):
            problems.append(f"stage {name} has {len(values)} values for {len(s['ms'])} samples")
    for name, values in s.get("metrics", {}).items():
        if len(values) != len(s["ms"]):
            problems.append(f"metric {name} has {len(values)} values for {len(s['ms'])} samples")
    return problems


# --------------------------------------------------------------------------------------------- statistics


def quantile(values: list[float], q: float) -> float:
    """Linear interpolation between order statistics (NumPy's default method)."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def median(values: list[float]) -> float:
    return quantile(values, 0.5)


def tail(values: list[float], q: float, minimum: int) -> float | None:
    """The q-quantile, or None when there are too few samples to report it."""
    return quantile(values, q) if len(values) >= minimum else None


def bootstrap(
    runs: list[list[float]], stat, resamples: int = 2000, seed: int = 1, level: float = 0.95
) -> tuple[float, float]:
    """Hierarchical bootstrap interval of ``stat`` over runs (independent processes), then samples within runs."""
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        pooled: list[float] = []
        for _ in range(len(runs)):
            run = runs[rng.randrange(len(runs))]
            pooled.extend(run[rng.randrange(len(run))] for _ in range(len(run)))
        estimates.append(stat(pooled))
    alpha = (1 - level) / 2
    return quantile(estimates, alpha), quantile(estimates, 1 - alpha)


def bootstrap_pair(
    a: list[list[float]], b: list[list[float]], stat, combine, resamples: int = 2000, seed: int = 1
) -> tuple[float, float]:
    """Interval of ``combine(stat(a*), stat(b*))`` with independent hierarchical resamples of both sides."""
    rng = random.Random(seed)

    def draw(runs: list[list[float]]) -> list[float]:
        pooled: list[float] = []
        for _ in range(len(runs)):
            run = runs[rng.randrange(len(runs))]
            pooled.extend(run[rng.randrange(len(run))] for _ in range(len(run)))
        return pooled

    estimates = [combine(stat(draw(a)), stat(draw(b))) for _ in range(resamples)]
    return quantile(estimates, 0.025), quantile(estimates, 0.975)


# -------------------------------------------------------------------------------------------- summaries


def _key(run: dict, s: dict) -> tuple:
    return (
        run["settings"]["provider"],
        s["mode"],
        s["path"],
        s["workload"],
        s["concurrency"],
        run["harness"]["language"],
        run["harness"]["impl"],
        run["model"],
    )


def group(results: RawResults) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for s in results.series:
        groups[_key(results.runs[s["run_id"]], s)].append(s)
    return groups


def summarize_group(series: list[dict], resamples: int, seed: int) -> dict:
    runs = [s["ms"] for s in series]
    pooled = [x for r in runs for x in r]
    lo, hi = bootstrap(runs, median, resamples, seed)
    throughput = [s["n"] * s["batch"] / (s["wall_ms"] / 1000) for s in series]
    stages = defaultdict(list)
    metrics = defaultdict(list)
    for s in series:
        for name, values in s["stages_ms"].items():
            stages[name].extend(values)
        for name, values in s.get("metrics", {}).items():
            metrics[name].extend(values)
    max_abs = [s["quality"]["max_abs"] for s in series if "max_abs" in s["quality"]]
    summary = {
        "runs": len(runs),
        "n": len(pooled),
        "batch": series[0]["batch"],
        "median_ms": median(pooled),
        "median_ci95_ms": [lo, hi],
        "run_medians_ms": [median(r) for r in runs],
        "p5_ms": tail(pooled, 0.05, TAIL_MIN_SAMPLES),
        "p95_ms": tail(pooled, 0.95, TAIL_MIN_SAMPLES),
        "p99_ms": tail(pooled, 0.99, P99_MIN_SAMPLES),
        "items_per_s": median(throughput),
        "items_per_s_range": [min(throughput), max(throughput)],
        "stage_median_ms": {name: median(values) for name, values in stages.items()},
        "quality_max_abs": max(max_abs) if max_abs else None,
        "quality_finite": all(s["quality"].get("finite", True) for s in series),
    }
    if metrics:
        summary["metric_median"] = {name: median(values) for name, values in metrics.items()}
    return summary


def compare(
    groups: dict[tuple, list[dict]], resamples: int, seed: int, a_impl: str, b_impl: str, stat=median
) -> list[dict]:
    """Pairs every group of implementation ``a_impl`` with the same workload of ``b_impl``."""
    out = []
    for key, series in sorted(groups.items()):
        if key[6] != a_impl:
            continue
        other = next((v for k, v in groups.items() if k[:5] == key[:5] and k[6] == b_impl), None)
        if other is None:
            continue
        a = [s["ms"] for s in series]
        b = [s["ms"] for s in other]
        pa = [x for r in a for x in r]
        pb = [x for r in b for x in r]
        ratio = stat(pa) / stat(pb)
        delta = stat(pa) - stat(pb)
        out.append(
            {
                "provider": key[0],
                "mode": key[1],
                "path": key[2],
                "workload": key[3],
                "concurrency": key[4],
                "a": a_impl,
                "b": b_impl,
                "n_a": len(pa),
                "n_b": len(pb),
                "runs_a": len(a),
                "runs_b": len(b),
                "a_ms": stat(pa),
                "b_ms": stat(pb),
                "ratio": ratio,
                "ratio_ci95": list(bootstrap_pair(a, b, stat, lambda x, y: x / y, resamples, seed)),
                "delta_ms": delta,
                "delta_ci95_ms": list(bootstrap_pair(a, b, stat, lambda x, y: x - y, resamples, seed)),
            }
        )
    return out


def overhead_budget(row: dict) -> dict:
    """Plan 11: p95 adapter overhead (library minus direct) no greater than max(1 ms, 10% of direct)."""
    budget = max(OVERHEAD_BUDGET_MS, OVERHEAD_BUDGET_FRACTION * row["b_ms"])
    lo, hi = row["delta_ci95_ms"]
    verdict = "met" if hi <= budget else ("not met" if lo > budget else "inconclusive")
    return {**row, "budget_ms": budget, "verdict": verdict}


def split_by_quality(results: RawResults) -> tuple[list[dict], list[dict]]:
    """Separates series whose timings may be presented from those that failed quality.

    A series is excluded when its own outputs exceed its run's ``embedding_max_abs_tolerance`` or are not finite
    (embedding harnesses) or its own ``quality.pass`` is false (the other harnesses), or when its run's quality did
    not pass (the preliminary sweep or any other series of the same process failed).
    """
    valid, failed = [], []
    for s in results.series:
        run = results.runs[s["run_id"]]
        tolerance = run["quality"].get("embedding_max_abs_tolerance")
        if tolerance is not None:
            own = s["quality"]["max_abs"] <= tolerance and s["quality"]["finite"]
        else:
            own = bool(s["quality"].get("pass", False))
        if own and run["quality"]["pass"]:
            valid.append(s)
            continue
        failed.append(
            {
                "run_id": s["run_id"],
                "language": run["harness"]["language"],
                "impl": run["harness"]["impl"],
                "provider": run["settings"]["provider"],
                "mode": s["mode"],
                "path": s["path"],
                "workload": s["workload"],
                "concurrency": s["concurrency"],
                "max_abs": s["quality"].get("max_abs"),
                "finite": s["quality"].get("finite"),
                "tolerance": tolerance,
                "reason": "run quality failed"
                if own
                else ("series outside tolerance" if tolerance is not None else "series failed its own check"),
            }
        )
    return valid, failed


def summarize(results: RawResults, resamples: int = 2000, seed: int = 1) -> dict:
    valid, failures = split_by_quality(results)
    groups = group(RawResults(runs=results.runs, series=valid))
    p95 = lambda v: quantile(v, 0.95)  # noqa: E731
    overhead = [
        overhead_budget(r)
        for r in compare(groups, resamples, seed, "library", "direct", stat=p95)
        if r["n_a"] >= TAIL_MIN_SAMPLES and r["n_b"] >= TAIL_MIN_SAMPLES
    ]
    return {
        "quality_failures": failures,
        "groups": [
            {
                "provider": k[0],
                "mode": k[1],
                "path": k[2],
                "workload": k[3],
                "concurrency": k[4],
                "language": k[5],
                "impl": k[6],
                **summarize_group(v, resamples, seed),
            }
            for k, v in sorted(groups.items())
        ],
        "csharp_vs_python": compare(groups, resamples, seed, "library", "python-ort"),
        "direct_vs_python": compare(groups, resamples, seed, "direct", "python-ort"),
        "adapter_overhead_p95": overhead,
        "runs": [
            {
                "run_id": r["run_id"],
                "language": r["harness"]["language"],
                "impl": r["harness"]["impl"],
                "provider": r["settings"]["provider"],
                "mode": r["mode"],
                "threads": r["settings"]["intra_op_num_threads"],
                "started_utc": r["started_utc"],
                "git_commit": r["artifacts"]["git_commit"],
                "git_dirty": r["artifacts"]["git_dirty"],
                "model_sha256": r["artifacts"]["model_sha256"],
                "tokenizer_sha256": r["artifacts"]["tokenizer_sha256"],
                "packages": r["artifacts"]["packages"],
                "load_before": r["environment"]["load_before"],
                "load_after": r["environment"]["load_after"],
                "quality": r["quality"],
            }
            for r in sorted(results.runs.values(), key=lambda r: r["started_utc"])
        ],
    }


# ---------------------------------------------------------------------------------------------- Markdown


def _num(v: float | None, fmt: str) -> str:
    return "n/a" if v is None else format(v, fmt)


def _ms(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:,.0f}" if v >= 100 else (f"{v:.1f}" if v >= 10 else f"{v:.2f}")


def _gpu_load(load: dict) -> str:
    gpus = load.get("gpus") or []
    return ", ".join(f"GPU{g['index']} {g['util_pct']}% {g['mem_used_mib']} MiB" for g in gpus) or "n/a"


def markdown(summary: dict) -> str:
    lines = [
        f"Medians with 95% hierarchical bootstrap CIs (runs, then samples). p5/p95 need n >= {TAIL_MIN_SAMPLES}; "
        f"p99 needs n >= {P99_MIN_SAMPLES} and is otherwise not reported.",
        "",
    ]
    if summary["quality_failures"]:
        lines += [
            "### Quality failures: excluded from every timing table below",
            "",
            "| Harness | Provider | Mode | Path | Workload | Conc. | Max abs | Finite | Tolerance | Reason |",
            "| --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |",
        ]
        for f in summary["quality_failures"]:
            lines.append(
                f"| {f['language']}/{f['impl']} | {f['provider']} | {f['mode']} | {f['path']} | {f['workload']} "
                f"| {f['concurrency']} | {_num(f['max_abs'], '.2e')} | {f['finite']} | {_num(f['tolerance'], '.1e')} "
                f"| {f['reason']} |"
            )
        lines.append("")
    lines += [
        "### C# (library) vs Python, same backend",
        "",
        "| Provider | Mode | Path | Workload | Conc. | C# median ms | Python median ms | C#/Python | 95% CI | "
        "Delta ms (95% CI) | n C#/Py |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for r in summary["csharp_vs_python"]:
        lines.append(
            f"| {r['provider']} | {r['mode']} | {r['path']} | {r['workload']} | {r['concurrency']} | {_ms(r['a_ms'])} "
            f"| {_ms(r['b_ms'])} | {r['ratio']:.3f} | {r['ratio_ci95'][0]:.3f}–{r['ratio_ci95'][1]:.3f} "
            f"| {r['delta_ms']:+.2f} ({r['delta_ci95_ms'][0]:+.2f} to {r['delta_ci95_ms'][1]:+.2f}) "
            f"| {r['n_a']}/{r['n_b']} |"
        )
    if summary["adapter_overhead_p95"]:
        lines += [
            "",
            "### Adapter overhead budget (plan 11): library vs direct C#, p95",
            "",
            "| Provider | Path | Workload | Conc. | Library p95 | Direct p95 | Delta (95% CI) | Budget | Verdict |",
            "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |",
        ]
        for r in summary["adapter_overhead_p95"]:
            lines.append(
                f"| {r['provider']} | {r['path']} | {r['workload']} | {r['concurrency']} | {_ms(r['a_ms'])} "
                f"| {_ms(r['b_ms'])} | {r['delta_ms']:+.2f} ({r['delta_ci95_ms'][0]:+.2f} to "
                f"{r['delta_ci95_ms'][1]:+.2f}) | {r['budget_ms']:.2f} | {r['verdict']} |"
            )
    lines += [
        "",
        "### All series",
        "",
        "| Provider | Mode | Path | Workload | Conc. | Harness | Runs | n | Median ms (95% CI) | p5–p95 ms | p99 ms "
        "| Items/s | Stage medians ms | Quality max abs |",
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | --- | --- | ---: | --- | ---: |",
    ]
    for g in summary["groups"]:
        stages = ", ".join(f"{k} {_ms(v)}" for k, v in g["stage_median_ms"].items())
        tails = f"{_ms(g['p5_ms'])}–{_ms(g['p95_ms'])}" if g["p5_ms"] is not None else "n/a"
        harness = f"{g['language']}/{g['impl']}"
        lines.append(
            f"| {g['provider']} | {g['mode']} | {g['path']} | {g['workload']} | {g['concurrency']} | {harness} "
            f"| {g['runs']} | {g['n']} | {_ms(g['median_ms'])} ({_ms(g['median_ci95_ms'][0])}–"
            f"{_ms(g['median_ci95_ms'][1])}) | {tails} | {_ms(g['p99_ms'])} | {g['items_per_s']:,.1f} | {stages} "
            f"| {_num(g['quality_max_abs'], '.2e') if g['quality_max_abs'] is not None else 'pass'} |"
        )
    with_metrics = [g for g in summary["groups"] if "metric_median" in g]
    if with_metrics:
        names = list(dict.fromkeys(name for g in with_metrics for name in g["metric_median"]))
        lines += [
            "",
            "### Per-sample metrics (medians)",
            "",
            "| Provider | Mode | Path | Workload | Conc. | Harness | n | " + " | ".join(names) + " |",
            "| --- | --- | --- | --- | ---: | --- | ---: | " + " | ".join("---:" for _ in names) + " |",
        ]
        for g in with_metrics:
            values = " | ".join(_num(g["metric_median"].get(name), ",.2f") for name in names)
            lines.append(
                f"| {g['provider']} | {g['mode']} | {g['path']} | {g['workload']} | {g['concurrency']} "
                f"| {g['language']}/{g['impl']} | {g['n']} | {values} |"
            )
    lines += [
        "",
        "### Runs",
        "",
        "| Started (UTC) | Harness | Provider | Mode | Threads | Quality (ids/text max abs, pass) | Load before | "
        "GPUs before |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in summary["runs"]:
        q = r["quality"]
        if "ids_single" in q:
            qtext = f"{q['ids_single']['max_abs']:.2e} / {q['text_single']['max_abs']:.2e}, {q['pass']}"
        elif "max_abs" in q.get("cold_first_request", {}):
            qtext = f"{q['cold_first_request']['max_abs']:.2e} (first request), {q['pass']}"
        else:
            qtext = f"{q['cases']} cases, {q['pass']}"
        loadavg = r["load_before"]["loadavg"]
        lines.append(
            f"| {r['started_utc'][:19]} | {r['language']}/{r['impl']} | {r['provider']} | {r['mode']} "
            f"| {r['threads']} | {qtext} | {loadavg} | {_gpu_load(r['load_before'])} |"
        )
    return "\n".join(lines) + "\n"
