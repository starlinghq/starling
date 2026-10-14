"""G5 performance budgets: C# raw results against the Python same-backend runs (plan 10 and 11, M2).

Reads raw result files (`starlings-bench-raw/1`) from the C# harnesses and the Python matched runners and checks
them against `model-pipeline/protected/performance-budgets.yaml`, profile by profile. Every key of a profile is a
rule; a key this module does not implement is an error, so a budget can never be skipped silently. The rules:

- `warm_median_ratio_max`: per warm workload (path, workload, concurrency), median C# library total over median
  Python total at most the budget.
- `cold_start_ratio_max`: the same on the cold-start series (spawn to result).
- `adapter_overhead_max` (``"max(1 ms, 10%) of direct C#, on medians"``): per warm workload with a direct C# series,
  median library minus median direct at most max(1 ms, 10% of the direct median).
- `inference_ratio_max`: per warm workload, median C# `inference` stage over median Python `inference` stage.
- `prepare_plus_decode_share_of_inference_max` (C# only): per warm workload, the median of each sample's
  `prepare` + `decode` over the median `inference`; every workload within the budget.
- `preprocess_share_of_embed_max` (C# only, image path): per image row of the `preprocess-only` workload, the median
  `preprocess` of that row's samples over the median `embed` of the samples that embedded the same row alone (any
  other warm image workload whose request is that one row; samples map to requests as `i % requests`, concurrency
  1). The requests come from the workload file the run names, checked by its SHA-256. Every row within the budget.
- `decode_tokens_per_second_ratio_min`: per warm workload, median C# `decode_tokens_per_s` over the Python median at
  least the budget.
- `time_to_first_token_ratio_max`: per warm workload, median C# `ttft_ms` over the Python median.

Decisions use the point estimates (ratios of medians, pooled over runs). Each one is reported with its 95%
hierarchical bootstrap interval (runs, then samples); budget comments in the YAML (for example "the upper 95% bound
at most 1.15") are printed with the rule, not gated. A rule without data is `no data`, never a pass.

Verdict per profile: `pass` (every rule has data and passes), `fail` (a rule fails, a series failed quality, or the
sides are not comparable: different graph, workload, provider, thread count or runtime version), `incomplete` (a
rule has no data). Separately, `qualifying` lists what the run lacks to be G5 evidence under docs/benchmarks/
PROTOCOL.md (at least 3 warm runs and 10 cold processes per side, a clean checkout); G5 needs `pass` and qualifying.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import yaml

from model_pipeline import benchmarks

REPO = Path(__file__).resolve().parents[3]
BUDGETS = REPO / "model-pipeline" / "protected" / "performance-budgets.yaml"
ADAPTER_OVERHEAD = "max(1 ms, 10%) of direct C#, on medians"
MIN_WARM_RUNS = 3
MIN_COLD_PROCESSES = 10
DESCRIPTIVE_KEYS = {"comparison"}


# ------------------------------------------------------------------------------------------------ budgets


def load_budgets(path: Path = BUDGETS) -> tuple[dict, dict[tuple[str, str], str]]:
    """The budget profiles and the trailing comment of each ``profile.key`` line (reported, not gated)."""
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    comments: dict[tuple[str, str], str] = {}
    profile = None
    for line in text.split("\n"):
        top = re.match(r"^  (\S[^:]*):\s*$", line)
        if top:
            profile = top.group(1)
            continue
        entry = re.match(r"^    (\w+):[^#]*#\s*(.+)$", line)
        if entry and profile:
            comments[(profile, entry.group(1))] = entry.group(2).strip()
    return data["profiles"], comments


# ----------------------------------------------------------------------------------------------- profiles


def profile_of(run: dict, series: dict) -> str | None:
    """The budget profile a series measures, from the run's model and provider and the series' path."""
    model, provider = run["model"], run["settings"]["provider"]
    if model in ("multilingual-e5-small", "gliner2-base-v1"):
        return f"{model}/onnx-fp32-{provider}"
    if model == "embeddinggemma-2":
        kind = {"text": "onnx", "image": "image", "ids": "onnx"}.get(series["path"])
        return f"{model}/{kind}-fp32-{provider}" if kind else None
    if model == "qwen3.5-0.8b":
        variant = run["settings"].get("variant")
        if variant is None:
            found = re.search(r"ortgenai-(fp\d+x)", run["artifacts"].get("model_path") or "")
            variant = found.group(1) if found else None
        return f"{model}/ortgenai-{variant}-{provider}" if variant else None
    return None


def side_of(run: dict) -> str | None:
    harness = run["harness"]
    if harness["language"] == "python":
        return "python"
    return {"library": "csharp", "direct": "direct"}.get(harness["impl"])


def collect(results: benchmarks.RawResults, valid: list[dict]) -> dict[str, dict]:
    """Per profile: its runs by side and its valid series grouped by (side, mode, path, workload, concurrency)."""
    profiles: dict[str, dict] = defaultdict(lambda: {"runs": defaultdict(dict), "groups": defaultdict(list)})
    for s in results.series:
        run = results.runs[s["run_id"]]
        profile, side = profile_of(run, s), side_of(run)
        if profile is None or side is None:
            continue
        profiles[profile]["runs"][side][run["run_id"]] = run
    valid_ids = {id(s) for s in valid}
    for s in results.series:
        run = results.runs[s["run_id"]]
        profile, side = profile_of(run, s), side_of(run)
        if profile is None or side is None or id(s) not in valid_ids:
            continue
        key = (side, s["mode"], s["path"], s["workload"], s["concurrency"])
        profiles[profile]["groups"][key].append(s)
    return profiles


# ----------------------------------------------------------------------------------------------- values


def total(s: dict) -> list[float]:
    return s["ms"]


def stage(name: str):
    return lambda s: s["stages_ms"][name]


def metric(name: str):
    return lambda s: s["metrics"][name]


def _pooled(runs: list[list[float]]) -> list[float]:
    return [x for r in runs for x in r]


def ratio_check(a: list[list[float]], b: list[list[float]], resamples: int, seed: int) -> dict:
    """median(a) / median(b) over pooled runs, with its bootstrap interval."""
    value = benchmarks.median(_pooled(a)) / benchmarks.median(_pooled(b))
    ci = benchmarks.bootstrap_pair(a, b, benchmarks.median, lambda x, y: x / y, resamples, seed)
    return {"value": value, "ci95": list(ci), "runs": [len(a), len(b)], "n": [len(_pooled(a)), len(_pooled(b))]}


def _label(key: tuple) -> str:
    _, mode, path, workload, concurrency = key
    return f"{mode}/{path}/{workload}/c{concurrency}"


def pairs(groups: dict, a_side: str, b_side: str, mode: str, extract, required=None):
    """(label, a runs, b runs) for every workload of ``mode`` that either side measured; a side without the workload
    gives None (the check then has no data instead of being skipped)."""
    keys = sorted({k[1:] for k in groups if k[0] in (a_side, b_side) and k[1] == mode})
    for rest in keys:
        a, b = groups.get((a_side, *rest)), groups.get((b_side, *rest))
        if required and not all(required(s) for s in (a or []) + (b or [])):
            continue
        yield _label((None, *rest)), [extract(s) for s in a] if a else None, [extract(s) for s in b] if b else None


def has_stage(name: str):
    return lambda s: name in s["stages_ms"]


def has_metric(name: str):
    return lambda s: name in s.get("metrics", {})


# -------------------------------------------------------------------------------------------------- rules


def rule_ratio(groups, budget, upper, mode, extract, required, resamples, seed, a="csharp", b="python") -> list[dict]:
    checks = []
    for label, ra, rb in pairs(groups, a, b, mode, extract, required):
        if ra is None or rb is None:
            missing = "C# library" if ra is None else ("Python" if b == "python" else b)
            checks.append({"workload": label, "value": None, "pass": None, "missing": missing})
            continue
        c = ratio_check(ra, rb, resamples, seed)
        c.update({"workload": label, "pass": c["value"] <= budget if upper else c["value"] >= budget})
        checks.append(c)
    return checks


def rule_adapter_overhead(groups, budget: str, resamples: int, seed: int) -> list[dict]:
    if budget != ADAPTER_OVERHEAD:
        raise ValueError(f"adapter_overhead_max {budget!r} is not the implemented rule {ADAPTER_OVERHEAD!r}")
    checks = []
    for label, lib, direct in pairs(groups, "csharp", "direct", "warm", total):
        if lib is None or direct is None:
            continue  # the direct harness measures only some workloads (e5: the ids path)
        lib_m, direct_m = benchmarks.median(_pooled(lib)), benchmarks.median(_pooled(direct))
        limit = max(1.0, 0.10 * direct_m)
        ci = benchmarks.bootstrap_pair(lib, direct, benchmarks.median, lambda x, y: x - y, resamples, seed)
        checks.append(
            {
                "workload": label,
                "value": lib_m - direct_m,
                "limit_ms": limit,
                "ci95": list(ci),
                "pass": lib_m - direct_m <= limit,
            }
        )
    return checks


def share_check(numerator: list[list[float]], denominator: list[list[float]], resamples: int, seed: int) -> dict:
    return ratio_check(numerator, denominator, resamples, seed)


def rule_prepare_decode_share(groups, budget: float, resamples: int, seed: int) -> list[dict]:
    checks = []
    for key, series in sorted(groups.items()):
        if key[0] != "csharp" or key[1] != "warm":
            continue
        if not all({"prepare", "decode", "inference"} <= set(s["stages_ms"]) for s in series):
            continue
        num = [
            [p + d for p, d in zip(s["stages_ms"]["prepare"], s["stages_ms"]["decode"], strict=True)] for s in series
        ]
        c = share_check(num, [s["stages_ms"]["inference"] for s in series], resamples, seed)
        c.update({"workload": _label(key), "pass": c["value"] <= budget})
        checks.append(c)
    return checks


def _workload_requests(run: dict) -> dict[str, list[list[str]]]:
    """Workload name -> requests (row ids), from the run's workload file, checked against its SHA-256."""
    path = REPO / run["artifacts"]["workload_file"]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != run["artifacts"]["workload_sha256"]:
        raise ValueError(f"{path} is not the workload file of run {run['run_id']} (SHA-256 differs)")
    return {w["name"]: w.get("requests", []) for w in json.loads(data)["workloads"]}


def _by_request(values: list[float], requests: int) -> list[list[float]]:
    out: list[list[float]] = [[] for _ in range(requests)]
    for i, v in enumerate(values):
        out[i % requests].append(v)
    return out


def rule_preprocess_share(groups, runs: dict, budget: float, resamples: int, seed: int) -> list[dict]:
    """Per row of `preprocess-only`: median preprocess over median embed of the same row alone (see module doc)."""
    pre_key = next((k for k in groups if k[0] == "csharp" and k[1] == "warm" and k[3] == "preprocess-only"), None)
    if pre_key is None or pre_key[4] != 1:
        return []
    pre_series = groups[pre_key]
    requests = _workload_requests(runs[pre_series[0]["run_id"]])
    pre_rows = [r[0] for r in requests["preprocess-only"]]
    numer: dict[str, list[list[float]]] = defaultdict(list)
    for s in pre_series:
        for row, values in zip(pre_rows, _by_request(s["stages_ms"]["preprocess"], len(pre_rows)), strict=True):
            numer[row].append(values)
    denom: dict[str, list[list[float]]] = defaultdict(list)
    for key, series in groups.items():
        if key[0] != "csharp" or key[1] != "warm" or key[2] != "image" or key[3] == "preprocess-only" or key[4] != 1:
            continue
        reqs = requests.get(key[3], [])
        if not reqs:
            continue
        for s in series:
            if "embed" not in s["stages_ms"]:
                continue
            for req, values in zip(reqs, _by_request(s["stages_ms"]["embed"], len(reqs)), strict=True):
                if len(req) == 1 and values:
                    denom[req[0]].append(values)
    checks = []
    for row in pre_rows:
        if row not in denom or not all(numer[row]):
            checks.append({"workload": f"warm/image/preprocess-only:{row}", "value": None, "pass": None})
            continue
        c = share_check(numer[row], denom[row], resamples, seed)
        c.update({"workload": f"warm/image/preprocess-only:{row}", "pass": c["value"] <= budget})
        checks.append(c)
    return checks


def evaluate_rule(name: str, budget, profile: dict, resamples: int, seed: int) -> list[dict]:
    groups = profile["groups"]
    if name == "warm_median_ratio_max":
        return rule_ratio(groups, budget, True, "warm", total, None, resamples, seed)
    if name == "cold_start_ratio_max":
        return rule_ratio(groups, budget, True, "cold", total, None, resamples, seed)
    if name == "inference_ratio_max":
        return rule_ratio(groups, budget, True, "warm", stage("inference"), has_stage("inference"), resamples, seed)
    if name == "decode_tokens_per_second_ratio_min":
        m = "decode_tokens_per_s"
        return rule_ratio(groups, budget, False, "warm", metric(m), has_metric(m), resamples, seed)
    if name == "time_to_first_token_ratio_max":
        return rule_ratio(groups, budget, True, "warm", metric("ttft_ms"), has_metric("ttft_ms"), resamples, seed)
    if name == "adapter_overhead_max":
        return rule_adapter_overhead(groups, budget, resamples, seed)
    if name == "prepare_plus_decode_share_of_inference_max":
        return rule_prepare_decode_share(groups, budget, resamples, seed)
    if name == "preprocess_share_of_embed_max":
        runs = {rid: r for side in profile["runs"].values() for rid, r in side.items()}
        return rule_preprocess_share(groups, runs, budget, resamples, seed)
    raise ValueError(f"budget key {name!r} has no implemented rule; refusing to skip it")


# ------------------------------------------------------------------------------------------ comparability


def _versions(run: dict) -> dict:
    """Runtime versions that must match across sides when both record them."""
    settings, env = run["settings"], run["environment"]
    return {"ort": env.get("ort_version"), "genai": settings.get("genai_version")}


def comparability(profile: dict) -> list[str]:
    runs = [r for side in profile["runs"].values() for r in side.values()]
    problems = []
    for field, get in (
        ("model_sha256", lambda r: r["artifacts"]["model_sha256"]),
        ("workload_sha256", lambda r: r["artifacts"]["workload_sha256"]),
        ("intra_op_num_threads", lambda r: r["settings"]["intra_op_num_threads"]),
        ("provider", lambda r: r["settings"]["provider"]),
    ):
        values = {get(r) for r in runs}
        if len(values) > 1:
            problems.append(f"{field} differs across runs: {sorted(map(str, values))}")
    for name in ("ort", "genai"):
        values = {_versions(r)[name] for r in runs} - {None}
        if len(values) > 1:
            problems.append(f"{name} version differs across runs: {sorted(values)}")
    return problems


def qualifying(profile: dict, rules: dict) -> list[str]:
    problems = []
    for name, result in rules.items():
        for c in result["checks"]:
            runs = c.get("runs")
            if not runs:
                continue
            cold = c["workload"].startswith("cold/")
            sides = ("C#", "C#") if "share" in name else ("C#", "Python")
            for side, r, n in zip(sides, runs, c["n"], strict=True):
                if cold and n < MIN_COLD_PROCESSES:
                    problems.append(f"{name} {c['workload']}: {side} has {n} cold processes (< {MIN_COLD_PROCESSES})")
                if not cold and r < MIN_WARM_RUNS:
                    problems.append(f"{name} {c['workload']}: {side} has {r} warm run(s) (< {MIN_WARM_RUNS})")
    for side in profile["runs"].values():
        for run in side.values():
            if run["artifacts"]["git_dirty"] is not False:
                problems.append(
                    f"run {run['run_id']} ({run['harness']['name']}): git_dirty={run['artifacts']['git_dirty']}"
                )
    return sorted(set(problems))


# --------------------------------------------------------------------------------------------- evaluation


def evaluate(
    results: benchmarks.RawResults,
    budgets: dict,
    comments: dict | None = None,
    only: set[str] | None = None,
    resamples: int = 1000,
    seed: int = 1,
) -> list[dict]:
    """One verdict per budget profile that has data (or every profile in ``only``)."""
    comments = comments or {}
    valid, failures = benchmarks.split_by_quality(results)
    profiles = collect(results, valid)
    failed_by_profile: dict[str, list[dict]] = defaultdict(list)
    for f in failures:
        run = results.runs[f["run_id"]]
        profile = profile_of(run, {"path": f["path"]})
        if profile:
            failed_by_profile[profile].append(f)
    names = sorted(only) if only else sorted(set(profiles) | set(failed_by_profile))
    out = []
    for name in names:
        if name not in budgets:
            out.append({"profile": name, "verdict": "no budget", "rules": {}, "problems": [], "qualifying": []})
            continue
        profile = profiles.get(name)
        if profile is None:
            out.append({"profile": name, "verdict": "no data", "rules": {}, "problems": [], "qualifying": []})
            continue
        rules = {}
        for key, budget in budgets[name].items():
            if key in DESCRIPTIVE_KEYS:
                continue
            checks = evaluate_rule(key, budget, profile, resamples, seed)
            decided = [c["pass"] for c in checks if c["pass"] is not None]
            missing = any(c["pass"] is None for c in checks)
            status = (
                "no data" if not decided else ("fail" if not all(decided) else ("incomplete" if missing else "pass"))
            )
            rules[key] = {"budget": budget, "status": status, "checks": checks, "note": comments.get((name, key))}
        problems = comparability(profile)
        problems += [
            f"quality failed: {f['mode']}/{f['path']}/{f['workload']} ({f['reason']})" for f in failed_by_profile[name]
        ]
        statuses = {r["status"] for r in rules.values()}
        if problems or "fail" in statuses:
            verdict = "fail"
        elif statuses <= {"pass"} and statuses:
            verdict = "pass"
        else:
            verdict = "incomplete"
        out.append(
            {
                "profile": name,
                "comparison": budgets[name].get("comparison"),
                "verdict": verdict,
                "rules": rules,
                "problems": problems,
                "qualifying": qualifying(profile, rules),
            }
        )
    return out


def _fmt(v) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def report(verdicts: list[dict]) -> str:
    lines = []
    for v in verdicts:
        g5 = v["verdict"] == "pass" and not v["qualifying"]
        lines.append(f"## {v['profile']}: {v['verdict'].upper()}" + (" (G5 evidence)" if g5 else ""))
        if v.get("comparison"):
            lines.append(f"comparison: {v['comparison']}")
        for key, rule in v["rules"].items():
            note = f"  # {rule['note']}" if rule.get("note") else ""
            lines.append(f"- {key} = {rule['budget']}: {rule['status']}{note}")
            for c in rule["checks"]:
                ci = c.get("ci95")
                ci_text = f" [{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else ""
                extra = f" limit {_fmt(c['limit_ms'])} ms" if "limit_ms" in c else ""
                verdict = {True: "pass", False: "FAIL", None: "no data"}[c["pass"]]
                if c.get("missing"):
                    verdict += f" (no {c['missing']} series)"
                lines.append(f"    {c['workload']}: {_fmt(c['value'])}{ci_text}{extra} {verdict}")
            if not rule["checks"]:
                lines.append("    no matching series")
        for p in v["problems"]:
            lines.append(f"- problem: {p}")
        if v["verdict"] not in ("no data", "no budget"):
            if v["qualifying"] or v["verdict"] != "pass":
                lines.append(f"- not qualifying G5 evidence ({len(v['qualifying'])} run-protocol reason(s)):")
                lines.extend(f"    {q}" for q in v["qualifying"][:20])
            else:
                lines.append("- qualifying: yes (run counts and clean checkout)")
        lines.append("")
    return "\n".join(lines)
