"""Batching-sweep results (starlings-bench-sweep/1): validation and markdown summaries.

The ``sweep`` command of the e5, EmbeddingGemma 2 and GLiNER2 harnesses (ai-dotnet/benchmarks/README.md, "Batching
sweep") writes one ``run`` record per process, one ``cell`` record per (workload set, batch, concurrency) with raw
per-call timings, token and padding counts, memory and a quality check against single calls, and an ``end`` record.
This module validates those files (``model-pipeline/schemas/benchmark-sweep.schema.json``) and writes one markdown
table per model x provider x configuration (intra-op threads, the library's rows per run, the machine), plus an
intra-op thread comparison when the files hold several thread counts.

Cells whose quality check failed, and every cell of a run whose fixture sweep failed, are listed but never ranked.
Repeated cells (same configuration in several runs) are reported as the median over runs, with the run count.

Standard library only, apart from jsonschema for validation.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "benchmark-sweep.schema.json"
SCHEMA_ID = "starlings-bench-sweep/1"
KINDS = ("run", "cell", "end")


def _validators() -> dict[str, Draft202012Validator]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return {kind: Draft202012Validator({"$defs": schema["$defs"], "$ref": f"#/$defs/{kind}"}) for kind in KINDS}


@dataclass
class SweepResults:
    runs: dict[str, dict] = field(default_factory=dict)
    cells: list[dict] = field(default_factory=list)
    ends: dict[str, dict] = field(default_factory=dict)


def load(paths: list[Path]) -> tuple[SweepResults, list[str]]:
    """Reads and validates sweep files. Returns the records and the problems found (empty when valid)."""
    validators = _validators()
    results = SweepResults()
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
                problems.append(f"{where}: kind must be one of {', '.join(KINDS)}, got {kind!r}")
                continue
            errors = sorted(validators[kind].iter_errors(record), key=lambda e: list(e.absolute_path))
            if errors:
                problems.extend(f"{where}: {e.message}" for e in errors[:5])
                continue
            if kind == "run":
                results.runs[record["run_id"]] = record
            elif kind == "end":
                results.ends[record["run_id"]] = record
            else:
                n = record["n"]
                for name in ("ms", "call_real_tokens", "call_padded_cells"):
                    if len(record[name]) != n:
                        problems.append(f"{where}: {name} has {len(record[name])} values, n is {n}")
                for stage, values in record["stages_ms"].items():
                    if len(values) != n:
                        problems.append(f"{where}: stages_ms.{stage} has {len(values)} values, n is {n}")
                results.cells.append(record)
    for cell in results.cells:
        if cell["run_id"] not in results.runs:
            problems.append(
                f"cell {cell['set']}/b{cell['batch']}/c{cell['concurrency']}: run {cell['run_id']} has no run record"
            )
    return results, problems


def _machine(run: dict) -> str:
    env = run["environment"]
    if run["settings"]["provider"] == "cuda" and isinstance(env.get("gpu"), dict):
        return str(env["gpu"].get("name") or "GPU")
    return f"{env.get('cpu_model')} ({env.get('logical_cpus')} logical CPUs)"


def _config(run: dict) -> tuple:
    s = run["settings"]
    return (
        run["model"],
        s["provider"],
        s.get("intra_op_num_threads"),
        run["sweep"]["max_batch_size"],
        s.get("max_batch_tokens"),
        _machine(run),
    )


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize(results: SweepResults) -> dict:
    """Groups cells by configuration and (set, batch, concurrency); medians over runs."""
    preliminary_failed = {
        rid
        for rid, run in results.runs.items()
        if isinstance(run.get("quality"), dict) and not run["quality"].get("pass", False)
    }
    groups: dict[tuple, dict[tuple, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for cell in results.cells:
        run = results.runs.get(cell["run_id"])
        if run is None:  # reported by load()
            continue
        groups[_config(run)][(cell["set"], cell["batch"], cell["concurrency"])].append(cell)

    configs = []
    for config, cells in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        rows = []
        for (set_name, batch, concurrency), members in cells.items():
            valid = [c for c in members if c["quality"]["pass"] and c["run_id"] not in preliminary_failed]
            use = valid or members
            stages = defaultdict(list)
            for c in use:
                for stage, values in c["stages_ms"].items():
                    if values:
                        stages[stage].append(statistics.median(values))
            vram = [c["vram"]["peak_mib"] for c in use if c.get("vram") and c["vram"].get("peak_mib") is not None]
            p95 = [c["latency_p95_ms"] for c in use if c.get("latency_p95_ms") is not None]
            rows.append(
                {
                    "set": set_name,
                    "batch": batch,
                    "concurrency": concurrency,
                    "runs": len(members),
                    "valid": bool(valid),
                    "calls": sum(c["n"] for c in use),
                    "items_per_s": _median([c["items_per_s"] for c in use]),
                    "latency_p50_ms": _median([c["latency_p50_ms"] for c in use]),
                    "latency_p95_ms": _median(p95),
                    "padding_efficiency": _median([c["padding_efficiency"] for c in use]),
                    "stages_p50_ms": {k: _median(v) for k, v in stages.items()},
                    "peak_rss_mib": max(c["rss"]["peak_bytes"] for c in use) / 1048576,
                    "peak_vram_mib": max(vram) if vram else None,
                    "max_diff": max(c["quality"]["max_diff"] for c in members),
                    "mismatches": sum(c["quality"]["mismatches"] for c in members),
                    "tolerance": members[0]["quality"]["tolerance"],
                    "failed": [
                        c["run_id"] for c in members if not c["quality"]["pass"] or c["run_id"] in preliminary_failed
                    ],
                }
            )
        rows.sort(key=lambda r: (r["set"], r["batch"], r["concurrency"]))
        model, provider, threads, max_batch_size, max_batch_tokens, machine = config
        configs.append(
            {
                "model": model,
                "provider": provider,
                "intra_op_num_threads": threads,
                "max_batch_size": max_batch_size,
                "max_batch_tokens": max_batch_tokens,
                "machine": machine,
                "rows": rows,
            }
        )
    incomplete = sorted(set(results.runs) - set(results.ends))
    failed_runs = sorted(rid for rid, end in results.ends.items() if not end["pass"])
    skipped = sorted(
        f"{results.runs[rid]['model']}/{results.runs[rid]['settings']['provider']}/{cell}"
        for rid, end in results.ends.items()
        if rid in results.runs
        for cell in end.get("skipped_cells", [])
    )
    return {
        "runs": len(results.runs),
        "cells": len(results.cells),
        "incomplete_runs": incomplete,
        "skipped_cells": skipped,
        "failed_runs": failed_runs,
        "preliminary_failed_runs": sorted(preliminary_failed),
        "configs": configs,
        "threads": _thread_table(configs),
    }


def _thread_table(configs: list[dict]) -> list[dict]:
    """Items/s by intra-op thread count, per (model, provider, max batch size, machine, set, batch, concurrency)."""
    table: dict[tuple, dict] = defaultdict(dict)
    for cfg in configs:
        for row in cfg["rows"]:
            if not row["valid"]:
                continue
            key = (
                cfg["model"],
                cfg["provider"],
                cfg["max_batch_size"],
                cfg["machine"],
                row["set"],
                row["batch"],
                row["concurrency"],
            )
            table[key][cfg["intra_op_num_threads"]] = row["items_per_s"]
    out = []
    for key, by_threads in sorted(table.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        if len(by_threads) < 2:
            continue
        model, provider, mbs, machine, set_name, batch, concurrency = key
        out.append(
            {
                "model": model,
                "provider": provider,
                "max_batch_size": mbs,
                "machine": machine,
                "set": set_name,
                "batch": batch,
                "concurrency": concurrency,
                "items_per_s": {
                    str(k): v for k, v in sorted(by_threads.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))
                },
            }
        )
    return out


def _fmt(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


_DETAIL_HEADER = (
    "| batch | conc | runs | calls | items/s | p50 ms | p95 ms | padding eff | stage p50 ms | peak RSS MiB "
    "| peak VRAM MiB | max diff (tol) | mismatches | quality |"
)
_DETAIL_RULE = "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | --- |"
_LEGEND = (
    "Items/s = items completed / cell wall time (all callers). Latency is per call. Padding efficiency = real "
    "tokens / padded cells (rows x longest row per ORT run, as the library chunks). Quality: every output vs the "
    "same item run alone, worst difference against the profile's padded-batch tolerance."
)


def _heading(cfg: dict) -> str:
    parts = [
        cfg["model"],
        cfg["provider"],
        f"intra-op {cfg['intra_op_num_threads'] or 'ORT default'}",
        f"max_batch_size {cfg['max_batch_size']}",
    ]
    if cfg["max_batch_tokens"]:
        parts.append(f"max_batch_tokens {cfg['max_batch_tokens']}")
    parts.append(cfg["machine"])
    return "## " + " · ".join(str(p) for p in parts)


def _best_note(rows: list[dict]) -> str | None:
    valid = [r for r in rows if r["valid"]]
    if not valid:
        return None
    best = max(valid, key=lambda r: r["items_per_s"])
    note = f"Best: batch {best['batch']} at concurrency {best['concurrency']}, {_fmt(best['items_per_s'])} items/s"
    single = [r for r in valid if r["batch"] == 1]
    if single:
        top = max(single, key=lambda r: r["items_per_s"])
        ratio = best["items_per_s"] / top["items_per_s"]
        note += (
            f"; best single-item callers: concurrency {top['concurrency']}, {_fmt(top['items_per_s'])} items/s "
            f"(ratio {ratio:.2f})"
        )
    return note + "."


def _detail(r: dict) -> str:
    stages = ", ".join(f"{k} {_fmt(v, 2)}" for k, v in r["stages_p50_ms"].items())
    cells = [
        r["batch"],
        r["concurrency"],
        r["runs"],
        r["calls"],
        _fmt(r["items_per_s"]),
        _fmt(r["latency_p50_ms"], 2),
        _fmt(r["latency_p95_ms"], 2),
        _fmt(r["padding_efficiency"], 3),
        stages,
        _fmt(r["peak_rss_mib"], 0),
        _fmt(r["peak_vram_mib"], 0),
        f"{r['max_diff']:.2e} ({r['tolerance']:.0e})",
        r["mismatches"],
        "pass" if r["valid"] else "FAILED",
    ]
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _set_tables(set_name: str, rows: list[dict]) -> list[str]:
    concurrency = sorted({r["concurrency"] for r in rows})
    batches = sorted({r["batch"] for r in rows})
    lines = [f"### {set_name}: items/s (batch x concurrency)", ""]
    lines.append("| batch | " + " | ".join(f"c={c}" for c in concurrency) + " |")
    lines.append("| ---: |" + " ---: |" * len(concurrency))
    by = {(r["batch"], r["concurrency"]): r for r in rows}
    for b in batches:
        cells = []
        for c in concurrency:
            r = by.get((b, c))
            cells.append("" if r is None else _fmt(r["items_per_s"]) + ("" if r["valid"] else " FAILED"))
        lines.append(f"| {b} | " + " | ".join(cells) + " |")
    note = _best_note(rows)
    if note:
        lines += ["", note]
    lines += ["", _DETAIL_HEADER, _DETAIL_RULE]
    lines += [_detail(r) for r in rows]
    return lines + [""]


def _thread_order(t: str) -> tuple:
    return (t == "None", int(t) if t != "None" else 0)


def _thread_lines(threads: list[dict]) -> list[str]:
    counts = sorted({t for row in threads for t in row["items_per_s"]}, key=_thread_order)
    lines = ["## Intra-op threads (items/s)", ""]
    lines.append("| model | provider | set | batch | conc | " + " | ".join(f"t={t}" for t in counts) + " |")
    lines.append("| --- | --- | --- | ---: | ---: |" + " ---: |" * len(counts))
    for row in threads:
        values = " | ".join(_fmt(row["items_per_s"][t]) if t in row["items_per_s"] else "" for t in counts)
        key = f"| {row['model']} | {row['provider']} | {row['set']} | {row['batch']} | {row['concurrency']} |"
        lines.append(f"{key} {values} |")
    return lines + [""]


def markdown(summary: dict) -> str:
    lines = ["# Batching sweep summary", "", f"{summary['runs']} run(s), {summary['cells']} cell(s)."]
    if summary["incomplete_runs"]:
        lines.append(f"Incomplete runs (no end record): {', '.join(summary['incomplete_runs'])}.")
    if summary["skipped_cells"]:
        lines.append(f"Skipped by the in-flight guard (not measured): {', '.join(summary['skipped_cells'])}.")
    failed = sorted(set(summary["failed_runs"]) | set(summary["preliminary_failed_runs"]))
    if failed:
        lines.append(f"Runs with quality failures: {', '.join(failed)}; failed cells are marked and never ranked.")
    lines += ["", _LEGEND]
    for cfg in summary["configs"]:
        lines += ["", _heading(cfg), ""]
        for set_name in sorted({r["set"] for r in cfg["rows"]}):
            lines += _set_tables(set_name, [r for r in cfg["rows"] if r["set"] == set_name])
    if summary["threads"]:
        lines += _thread_lines(summary["threads"])
    return "\n".join(lines).rstrip() + "\n"
