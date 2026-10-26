"""Stage results from an e5 browser conformance run (follow-up plan W6, browser profiles).

`browser/conformance/run_conformance.py` drives `Starlings.Models.E5.Browser` in headless Chromium and writes one JSON
file per run. This module converts such a file into the stage results format (`results.py`) so that the staged
evaluator (`evaluate.py`) checks the browser profiles against their protected tolerances like any other profile:

    tokenizer.results.json   stage 1: the e5 pipeline tokenizer in the browser (owned tokenizer, prefix handling and
                             truncation to 512 tokens), `run.e5_tokenizer`: ids and attention mask per fixture case
    dotnet.results.json      stage 3: the .NET library in the browser on onnxruntime-web. `run.embeddings` (one call per
                             fixture id sequence) as output `embedding`; `run.batch8` (padded batch of the first 8 short
                             cases) as batch `short-b8`; `run.generator_embeddings` (all fixture texts through
                             `E5BrowserEmbeddingGenerator` in one call, so padded batches) as batch
                             `generator-all-cases` with output `text_embedding`

Stage 2 (`export`) for a browser profile is Python ONNX Runtime (CPU) on the same graph, written by
`reference/encoder/stage_e5_browser_native.py`. The evaluator compares the browser stage with it for the
`browser_vs_native_same_graph` bound.

The browser run records no per-case ids for its embeddings: the conformance app reads them from the protected
`embeddings.jsonl` it is served, so the converter takes `input_ids` from that same file. The run does record the SHA-256
of the graph the loader verified (`run.load.model_sha256`); the converter refuses a run whose graph is not the
manifest's. `tokenizer.json` is declared with the manifest's digest, which the browser loader enforces
(`E5BrowserAssets`). Like every producer, the converter writes outputs only, never thresholds.

Provenance comes from the run, never from the converter: `producer.repository_commit` is the run's `source_commit`
(the commit the harness was built and run from), or `UNRECORDED_COMMIT` for runs that predate it, and the converting
HEAD is recorded separately as `producer.converted_at_commit`. `environment.browser` is the run's `browser` field (or
its older `browser_version`). When the run records `run.fixtures_sha256` (the digests of the fixture bytes the app
fetched), they must equal the `--fixtures` files; older runs are converted with `FIXTURES_NOT_RECORDED` in
`producer.fixtures_sha256`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from model_pipeline import fixtures, manifest, results

PRODUCER = "browser/conformance/run_conformance.py (E5Conformance) via model-pipeline browser-results"
TEXT_BATCH = "generator-all-cases"
ID_BATCH = "short-b8"
# Recorded when a run file predates the harness's `source_commit` field: the converter never substitutes its own HEAD.
UNRECORDED_COMMIT = "unrecorded (run predates source_commit)"
FIXTURES_NOT_RECORDED = "fixtures digest not recorded by the run"
FIXTURE_FILES = ("embeddings.jsonl", "tokenizer.jsonl")


class BrowserResultsError(ValueError):
    """The conformance run cannot be converted (failed run, wrong profile or graph, incomplete outputs)."""


def _browser(data: dict[str, Any]) -> str | None:
    """`chromium <version>` from the run's `browser` field, else from its older top-level `browser_version`."""
    browser = data.get("browser")
    if isinstance(browser, dict) and browser.get("version"):
        return f"{browser.get('name') or 'chromium'} {browser['version']}"
    if data.get("browser_version"):
        return f"chromium {data['browser_version']}"
    return None


def _fixtures_provenance(run_file: Path, run: dict[str, Any], fixtures_dir: Path) -> dict[str, str] | str:
    """The fixture digests the run recorded, checked against the --fixtures files; or the reason there are none."""
    recorded = run.get("fixtures_sha256")
    if recorded is None:
        return FIXTURES_NOT_RECORDED
    if not isinstance(recorded, dict):
        raise BrowserResultsError(f"{run_file}: run.fixtures_sha256 must map file names to SHA-256 digests")
    for name in FIXTURE_FILES:
        actual = fixtures.sha256_file(fixtures_dir / name)
        if recorded.get(name) != actual:
            raise BrowserResultsError(
                f"{run_file}: the run fetched {name} with SHA-256 {recorded.get(name)}; {fixtures_dir / name} is "
                f"{actual}, so the run was not made on these fixtures"
            )
    return {name: recorded[name] for name in FIXTURE_FILES}


def _file(man: dict[str, Any], path: str) -> dict[str, Any]:
    for f in man["origin"]["files"]:
        if f["path"] == path:
            return f
    raise BrowserResultsError(f"{man['id']}: {path} is not in origin.files")


def convert(
    run_file: Path,
    manifest_path: Path,
    fixtures_dir: Path,
    out_dir: Path,
    *,
    machine: str | None = None,
    converted_at_commit: str | None = None,
) -> dict[str, Path]:
    """Write `tokenizer.results.json` and `dotnet.results.json` for one conformance run; returns their paths."""
    data = json.loads(run_file.read_text(encoding="utf-8"))
    run = data.get("run") or {}
    if "error" in run:
        raise BrowserResultsError(f"{run_file}: the run failed: {str(run['error']).splitlines()[0]}")
    problems = manifest.validate_file(manifest_path)
    if problems:
        raise BrowserResultsError("manifest is invalid: " + "; ".join(str(p) for p in problems))
    man = manifest.load(manifest_path)
    if not manifest.is_browser(man):
        raise BrowserResultsError(
            f"{manifest_path}: not a browser profile (backend {manifest.BROWSER_BACKEND} expected)"
        )
    if run.get("profile") != man["id"]:
        raise BrowserResultsError(f"{run_file}: run reports profile {run.get('profile')!r}, manifest is {man['id']!r}")
    try:
        graph = manifest.graph_asset(man)
    except ValueError as exc:
        raise BrowserResultsError(str(exc)) from exc
    ran = (run.get("load") or {}).get("model_sha256")
    if ran != graph["sha256"]:
        raise BrowserResultsError(
            f"{run_file}: the browser loaded a graph with SHA-256 {ran}; the profile's graph {graph['path']} is "
            f"{graph['sha256']}"
        )
    tokenizer_file = _file(man, "tokenizer.json")

    fixtures_digest = _fixtures_provenance(run_file, run, fixtures_dir)
    reference = {r["id"]: r for r in fixtures.read_jsonl(fixtures_dir / "embeddings.jsonl")}
    for name in ("embeddings", "batch8", "generator_embeddings", "e5_tokenizer"):
        if name not in run:
            raise BrowserResultsError(f"{run_file}: run.{name} is missing (a summary file without outputs?)")
    unknown = sorted(set(run["embeddings"]) - set(reference))
    if unknown:
        raise BrowserResultsError(f"{run_file}: embeddings for cases not in {fixtures_dir}/embeddings.jsonl: {unknown}")

    ort_web = run.get("onnxruntime_web") or {}
    environment = {
        "machine": machine,
        "provider": run.get("provider"),
        "provider_options": {},
        "browser": _browser(data),
        "cross_origin_isolated": data.get("cross_origin_isolated"),
        "dotnet_runtime": run.get("runtime"),
        "build_label": data.get("label"),
        "onnxruntime_web": ort_web,
        "run_file": run_file.name,
        "run_file_sha256": fixtures.sha256_file(run_file),
    }
    for key in ("machine", "browser"):
        if environment[key] is None:
            del environment[key]  # not recorded
    source_commit = data.get("source_commit")
    producer = {
        "name": PRODUCER,
        # The commit the harness was built and run from, as the run records it; never the converter's HEAD.
        "repository_commit": source_commit if isinstance(source_commit, str) and source_commit else UNRECORDED_COMMIT,
        "converted_at_commit": converted_at_commit,
        "fixtures_sha256": fixtures_digest,
    }
    candidate = (
        f"Starlings.Models.E5.Browser on onnxruntime-web {ort_web.get('version')} ({run.get('provider')}, bundle "
        f"{ort_web.get('bundle')}, {ort_web.get('num_threads')} threads), build {data.get('label')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = results.new_results(
        profile=man["id"],
        stage="tokenizer",
        candidate=f"{candidate}: E5 tokenization (owned tokenizer, truncation to 512)",
        producer=producer,
        environment=environment,
        assets=[{"path": "tokenizer.json", "sha256": tokenizer_file["sha256"]}],
    )
    for row in run["e5_tokenizer"]:
        tok["cases"][row["id"]] = {"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]}

    dot = results.new_results(
        profile=man["id"],
        stage="dotnet",
        candidate=candidate,
        producer=producer,
        environment=environment,
        assets=[
            {"path": graph["path"], "sha256": graph["sha256"]},
            {"path": "tokenizer.json", "sha256": tokenizer_file["sha256"]},
        ],
    )
    for case_id, vector in run["embeddings"].items():
        dot["cases"][case_id] = {"input_ids": reference[case_id]["input_ids"], "embedding": list(vector)}
    dot["batches"][ID_BATCH] = {
        "output": "embedding",
        "case_ids": list(run["batch8"]),
        "values": [list(v) for v in run["batch8"].values()],
    }
    dot["batches"][TEXT_BATCH] = {
        "output": "text_embedding",
        "case_ids": list(run["generator_embeddings"]),
        "values": [list(v) for v in run["generator_embeddings"].values()],
    }

    paths = {"tokenizer": out_dir / "tokenizer.results.json", "dotnet": out_dir / "dotnet.results.json"}
    fixtures.write_json(paths["tokenizer"], tok)
    fixtures.write_json(paths["dotnet"], dot)
    for path in paths.values():
        results.load(path)  # schema and threshold-key validation
    return paths
