"""Batching-sweep files (starlings-bench-sweep/1): validation and the markdown summary."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from model_pipeline import bench_sweep
from model_pipeline.cli import main

SHA = "0" * 64


def _run(run_id: str, threads: int | None = 4, provider: str = "cpu", preliminary: bool = True) -> dict:
    return {
        "schema": "starlings-bench-sweep/1",
        "kind": "run",
        "run_id": run_id,
        "started_utc": "2026-10-17T00:00:00Z",
        "mode": "sweep",
        "model": "multilingual-e5-small",
        "profile": f"onnx-fp32-{provider}",
        "harness": {"name": "Starlings.Benchmarks.E5Harness", "language": "csharp", "impl": "library", "version": "1"},
        "artifacts": {
            "model_sha256": SHA,
            "tokenizer_sha256": SHA,
            "workload_sha256": SHA,
            "fixture_sha256": {},
            "git_commit": "abc",
            "git_dirty": False,
            "packages": {},
        },
        "environment": {
            "cpu_model": "Test CPU",
            "logical_cpus": 12,
            "ram_bytes": 1,
            "ort_version": "1.30.0",
            "gpu": {"name": "NVIDIA GeForce RTX 3090"} if provider == "cuda" else None,
            "load_before": {"loadavg": [0.1, 0.1, 0.1], "gpus": None},
        },
        "settings": {"provider": provider, "intra_op_num_threads": threads, "max_batch_size": 64},
        "load_ms": {"session_create": 100.0},
        "sweep": {
            "sets": ["short"],
            "batches": [1, 8],
            "concurrency": [1, 4],
            "max_batch_size": "call (64)",
            "seed": 1,
            "pools": {},
            "quality_check": "embedding max abs vs single",
            "tolerance": 1e-5,
            "tolerance_file": "model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/tolerances.yaml",
        },
        "quality": {"pass": preliminary},
    }


def _cell(run_id: str, batch: int, concurrency: int, items_per_s: float, passed: bool = True, n: int = 3) -> dict:
    return {
        "schema": "starlings-bench-sweep/1",
        "kind": "cell",
        "run_id": run_id,
        "model": "multilingual-e5-small",
        "provider": "cpu",
        "set": "short",
        "batch": batch,
        "concurrency": concurrency,
        "intra_op_num_threads": 4,
        "max_batch_size": 64,
        "max_batch_tokens": None,
        "n": n,
        "items": n * batch,
        "wall_ms": 100.0,
        "items_per_s": items_per_s,
        "calls_per_s": items_per_s / batch,
        "latency_p50_ms": 5.0 * batch,
        "latency_p95_ms": None,
        "padding_efficiency": 1.0 if batch == 1 else 0.6,
        "ms": [5.0 * batch] * n,
        "offset_ms": [0.0] * n,
        "worker": [0] * n,
        "stages_ms": {"tokenize": [0.1] * n, "encode": [4.9 * batch] * n},
        "call_real_tokens": [10 * batch] * n,
        "call_padded_cells": [16 * batch] * n,
        "call_runs": [1] * n,
        "rss": {"before_bytes": 100 << 20, "peak_bytes": 200 << 20},
        "vram": None,
        "quality": {
            "check": "embedding max abs vs single",
            "items_checked": n * batch,
            "max_diff": 0.0 if passed else 1e-3,
            "tolerance": 1e-5,
            "finite": True,
            "mismatches": 0,
            "pass": passed,
        },
    }


def _end(run_id: str, passed: bool = True) -> dict:
    return {
        "schema": "starlings-bench-sweep/1",
        "kind": "end",
        "run_id": run_id,
        "cells": 4,
        "failed_cells": [],
        "pass": passed,
    }


def _write(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_valid_file_summarizes_per_configuration(tmp_path: Path) -> None:
    records = [
        _run("r1"),
        _cell("r1", 1, 1, 100.0),
        _cell("r1", 1, 4, 300.0),
        _cell("r1", 8, 1, 200.0),
        _cell("r1", 8, 4, 900.0, passed=False),
        _end("r1", passed=False),
    ]
    results, problems = bench_sweep.load([_write(tmp_path / "s.jsonl", records)])
    assert problems == []
    summary = bench_sweep.summarize(results)
    assert len(summary["configs"]) == 1
    text = bench_sweep.markdown(summary)
    assert "## multilingual-e5-small · cpu · baseline · intra-op 4 · max_batch_size call (64)" in text
    # The failed cell is marked and never ranked: best is batch 1 at concurrency 4.
    assert "900.0 FAILED" in text
    assert "Best: batch 1 at concurrency 4, 300.0 items/s" in text
    assert "r1" in summary["failed_runs"]


def test_thread_counts_get_their_own_table(tmp_path: Path) -> None:
    a = [_run("t2", threads=2), _cell("t2", 1, 1, 50.0), _end("t2")]
    b = [_run("t6", threads=6), _cell("t6", 1, 1, 120.0), _end("t6")]
    for c in b[1:2]:
        c["intra_op_num_threads"] = 6
    results, problems = bench_sweep.load([_write(tmp_path / "a.jsonl", a), _write(tmp_path / "b.jsonl", b)])
    assert problems == []
    summary = bench_sweep.summarize(results)
    assert len(summary["configs"]) == 2
    assert summary["threads"][0]["items_per_s"] == {"2": 50.0, "6": 120.0}
    assert "## Intra-op threads (items/s)" in bench_sweep.markdown(summary)


def test_repeated_cells_take_the_median_over_runs(tmp_path: Path) -> None:
    records = []
    for rid, value in (("x", 100.0), ("y", 140.0), ("z", 120.0)):
        records += [_run(rid), _cell(rid, 1, 1, value), _end(rid)]
    results, _ = bench_sweep.load([_write(tmp_path / "s.jsonl", records)])
    row = bench_sweep.summarize(results)["configs"][0]["rows"][0]
    assert row["runs"] == 3
    assert row["items_per_s"] == 120.0


def test_problems_are_reported(tmp_path: Path) -> None:
    broken = _cell("r1", 8, 1, 10.0)
    broken["call_padded_cells"] = [1]
    bad_kind = copy.deepcopy(broken)
    bad_kind["kind"] = "series"
    orphan = _cell("missing", 1, 1, 1.0)
    results, problems = bench_sweep.load([_write(tmp_path / "s.jsonl", [_run("r1"), broken, bad_kind, orphan])])
    assert any("call_padded_cells has 1 values" in p for p in problems)
    assert any("kind must be one of" in p for p in problems)
    assert any("has no run record" in p for p in problems)
    assert bench_sweep.summarize(results)["incomplete_runs"] == ["r1"]


def test_preliminary_failure_excludes_every_cell_of_the_run(tmp_path: Path) -> None:
    records = [_run("r1", preliminary=False), _cell("r1", 1, 1, 100.0), _end("r1", passed=False)]
    results, _ = bench_sweep.load([_write(tmp_path / "s.jsonl", records)])
    summary = bench_sweep.summarize(results)
    assert summary["configs"][0]["rows"][0]["valid"] is False
    assert "Best:" not in bench_sweep.markdown(summary)


def test_skipped_cells_are_listed(tmp_path: Path) -> None:
    end = _end("r1")
    end["skipped_cells"] = ["short/b64/c16"]
    results, problems = bench_sweep.load([_write(tmp_path / "s.jsonl", [_run("r1"), _cell("r1", 1, 1, 1.0), end])])
    assert problems == []
    text = bench_sweep.markdown(bench_sweep.summarize(results))
    assert "multilingual-e5-small/cpu/short/b64/c16" in text


def test_cli_writes_markdown_and_json(tmp_path: Path) -> None:
    path = _write(tmp_path / "s.jsonl", [_run("r1"), _cell("r1", 1, 1, 100.0), _end("r1")])
    out, js = tmp_path / "summary.md", tmp_path / "summary.json"
    assert main(["bench", "sweep", str(path), "--out", str(out), "--json", str(js)]) == 0
    assert out.read_text(encoding="utf-8").startswith("# Batching sweep summary")
    assert json.loads(js.read_text(encoding="utf-8"))["cells"] == 1
    bad = _write(tmp_path / "bad.jsonl", [{"kind": "cell"}])
    assert main(["bench", "sweep", str(bad)]) == 1


def test_batching_configurations_are_compared(tmp_path: Path) -> None:
    base = [_run("b"), _cell("b", 1, 4, 300.0), _cell("b", 1, 16, 320.0), _cell("b", 8, 4, 500.0), _end("b")]
    sorted_run = _run("s")
    sorted_run["sweep"]["config"] = "sorted"
    sorted_run["sweep"]["sort_by_length"] = True
    sorted_cells = [_cell("s", 8, 4, 650.0)]
    micro_run = _run("m")
    micro_run["sweep"]["config"] = "micro-w0-r8"
    micro_run["sweep"]["micro_batching"] = {"max_wait_ms": 0, "max_rows": 8, "max_tokens": None}
    micro_cells = [_cell("m", 1, 4, 280.0), _cell("m", 1, 16, 610.0)]
    for c in micro_cells:
        c["config"] = "micro-w0-r8"
        c["stages_ms"] = {"call": c["stages_ms"]["encode"]}
        c["micro_batching"] = {
            "batches": 10,
            "items": 60,
            "mean_rows": 6.0,
            "tokens": 600,
            "padded_cells": 900,
            "padding_efficiency": 0.667,
            "queue_ms_per_item": 1.5,
            "run_ms_per_batch": 9.0,
            "run_ms_total": 90.0,
        }
    files = [
        _write(tmp_path / "b.jsonl", base),
        _write(tmp_path / "s.jsonl", [sorted_run, *sorted_cells, _end("s")]),
        _write(tmp_path / "m.jsonl", [micro_run, *micro_cells, _end("m")]),
    ]
    results, problems = bench_sweep.load(files)
    assert problems == []
    summary = bench_sweep.summarize(results)
    assert {c["config"] for c in summary["configs"]} == {"baseline", "sorted", "micro-w0-r8"}
    [entry] = summary["comparison"]
    assert entry["configs"]["sorted"]["best_items_per_s"] == 650.0
    assert entry["configs"]["micro-w0-r8"]["b1_items_per_s"] == {"4": 280.0, "16": 610.0}
    assert entry["configs"]["baseline"]["b1_items_per_s"] == {"4": 300.0, "16": 320.0}
    text = bench_sweep.markdown(summary)
    assert "## Batching configurations (items/s)" in text
    assert "| micro-w0-r8 | 610.0 | (1, 16) |" in text
    assert "6.00 / 1.50" in text
