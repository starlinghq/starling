"""Stage results format version 3, task `generation` (backlog R-010): writers, converters and the reader.

Version 3 carries two stage files per run, next to a float32 sidecar:

    template.results.json   per case `rendered`, `render_error`, `prompt_ids` (the owned tokenizer's ids)        (G1)
    dotnet.results.json     per case `prompt_ids` (fed), `output_ids` (with the stop token), `finish_reason`,
                            `step0_top` (five best [token, log-probability]), `stream_text_equals_result`,
                            `stream_ids_equal_result`, `stream_text_equals_decode`, `usage`
                            {prompt_tokens, completion_tokens, consistent}; `sections` cached_vs_uncached,
                            chunked_prefill, concurrency (and selection); tensor `step0_logprobs`           (G2)
    step0.f32               the whole step-0 log-softmax vector per case (optional; without it only the five
                            best are checked and the full-vocabulary bound stays not-run)

Two converters write it:

* `convert_runner_directory`: the output directory of `ai-dotnet/tools/Starlings.Qwen35.Fidelity`
  (`results.jsonl`, `step0/<id>.f32`, `semantics.json`, `environment.json`).
* `convert_fixture`: the protected `starlings.generation-fixture` v1 file (`generation.jsonl`), as the reference side.
  It carries only what the fixture records (rendering, ids, finish reason, the top step-0 pairs, usage counts); the
  runner-observed streaming flags are not invented, so the streaming check of a converted fixture does not pass.

`load_run` reads a directory of version 3 files into the rows the generation checks use, so the checks (and the
protected evidence they bind) are the same for a runner directory and for version 3 files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from model_pipeline import results
from model_pipeline.evaluate_generation_candidate import read_float32, read_rows
from model_pipeline.onboarding_runner import PipelineError

TASK = "generation"
SIDECAR = "step0.f32"
TOP_KEPT = 5
STREAM_FLAGS = ("stream_text_equals_result", "stream_ids_equal_result", "stream_text_equals_decode")


class ConversionError(ValueError):
    """The source directory or file cannot be converted."""


def top_pairs(vector: list[float], count: int = TOP_KEPT) -> list[list[Any]]:
    """The `count` best [token, log-probability] pairs, best first, ties by the lower id."""
    order = sorted(range(len(vector)), key=lambda i: (-vector[i], i))
    return [[i, vector[i]] for i in order[:count]]


def _producer(env: dict | None, default: str) -> dict[str, Any]:
    env = env or {}
    return {"name": env.get("runner") or default, "repository_commit": env.get("source_revision")}


def _write(directory: Path, name: str, document: dict[str, Any]) -> Path:
    path = directory / name
    path.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"{path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConversionError(f"{path}: expected a JSON object")
    return data


def convert_runner_directory(runner_dir: Path, out_dir: Path, profile: str, vocab: int | None = None) -> list[Path]:
    """Write `template.results.json`, `dotnet.results.json` and `step0.f32` for the runner directory `runner_dir`.

    `vocab` is the vector length; by default it is read from the first step-0 file (4 bytes per value).
    """
    try:
        rows = read_rows(runner_dir / "results.jsonl")
    except PipelineError as exc:
        raise ConversionError(str(exc)) from exc
    semantics = _read_json(runner_dir / "semantics.json")
    env = _read_json(runner_dir / "environment.json")
    out_dir.mkdir(parents=True, exist_ok=True)

    environment = dict(env or {})
    environment.setdefault("provider_options", {})
    environment.pop("profile", None)

    template = results.new_results(
        profile=profile,
        stage="template",
        candidate="Starlings.Qwen35.Fidelity",
        producer=_producer(env, "Starlings.Qwen35.Fidelity"),
        environment=environment,
        assets=[],
        task=TASK,
    )
    dotnet = results.new_results(
        profile=profile,
        stage="dotnet",
        candidate="Starlings.Qwen35.Fidelity",
        producer=_producer(env, "Starlings.Qwen35.Fidelity"),
        environment=environment,
        assets=[],
        task=TASK,
    )
    vectors: list[tuple[str, list[int], list[float]]] = []
    for name, row in rows.items():
        template["cases"][name] = {
            "rendered": row.get("rendered"),
            "render_error": row.get("render_error"),
            "prompt_ids": row.get("backend_prompt_ids"),
        }
        case: dict[str, Any] = {
            "prompt_ids": row.get("prompt_ids"),
            "output_ids": row.get("output_ids"),
            "finish_reason": row.get("finish_reason"),
        }
        for flag in STREAM_FLAGS:
            if flag in row:
                case[flag] = row[flag]
        if any(k in row for k in ("usage_prompt_tokens", "usage_completion_tokens", "usage_consistent")):
            case["usage"] = {
                "prompt_tokens": row.get("usage_prompt_tokens"),
                "completion_tokens": row.get("usage_completion_tokens"),
                "consistent": row.get("usage_consistent"),
            }
        step0 = runner_dir / "step0" / f"{name}.f32"
        if step0.is_file():
            size = step0.stat().st_size
            count = vocab if vocab is not None else size // 4
            try:
                vector = list(read_float32(runner_dir, f"step0/{name}.f32", count))
            except PipelineError as exc:
                raise ConversionError(str(exc)) from exc
            vocab = count
            vectors.append((name, [count], vector))
            case["step0_top"] = top_pairs(vector)
        dotnet["cases"][name] = case
    if vectors:
        dotnet["tensors"][results.STEP0_TENSOR] = results.write_tensor(out_dir, SIDECAR, vectors)
    if semantics:
        dotnet["sections"] = {k: v for k, v in semantics.items() if isinstance(v, dict)}
    return [_write(out_dir, "template.results.json", template), _write(out_dir, "dotnet.results.json", dotnet)]


def convert_fixture(fixture_path: Path, out_dir: Path, profile: str) -> list[Path]:
    """Write the reference side of a `starlings.generation-fixture` v1 file as version 3 stage results."""
    try:
        cases = read_rows(fixture_path)
    except PipelineError as exc:
        raise ConversionError(str(exc)) from exc
    out_dir.mkdir(parents=True, exist_ok=True)
    producer = {"name": "starlings.generation-fixture", "repository_commit": None}
    environment: dict[str, Any] = {"provider_options": {}}
    template = results.new_results(
        profile=profile, stage="template", candidate="fixture-v1", producer=producer, environment=environment,
        assets=[], task=TASK,
    )  # fmt: skip
    dotnet = results.new_results(
        profile=profile, stage="dotnet", candidate="fixture-v1", producer=producer, environment=environment,
        assets=[], task=TASK,
    )  # fmt: skip
    for name, case in cases.items():
        rejected = bool(case.get("tools"))
        template["cases"][name] = {
            "rendered": None if rejected else case.get("rendered"),
            "render_error": "tools are unsupported by the owned renderer" if rejected else None,
            "prompt_ids": None if rejected else case.get("prompt_ids"),
        }
        output = case.get("output_ids")
        dotnet["cases"][name] = {
            "prompt_ids": case.get("prompt_ids"),
            "output_ids": output,
            "finish_reason": case.get("finish_reason"),
            "step0_top": [list(p) for p in (case.get("step0_top") or [])[:TOP_KEPT]],
            "usage": {
                "prompt_tokens": len(case.get("prompt_ids") or []),
                "completion_tokens": len(output or []) - (1 if case.get("finish_reason") == "eos" else 0),
            },
        }
    return [_write(out_dir, "template.results.json", template), _write(out_dir, "dotnet.results.json", dotnet)]


# Reader -------------------------------------------------------------------------------------------------------


def has_stage_results(directory: Path) -> bool:
    return directory.is_dir() and any(directory.glob(f"*{results.RESULTS_SUFFIX}"))


def load_stage_files(directory: Path) -> dict[str, results.StageResults]:
    """The `template` and `dotnet` files of `directory`; both are required."""
    try:
        stages = results.load_directory(directory)
    except results.ToleranceOverrideError:
        raise
    except results.ResultsError as exc:
        raise ConversionError(str(exc)) from exc
    for stage in stages.values():
        if stage.format_version != 3 or stage.task != TASK:
            raise ConversionError(f"{stage.path}: expected format version 3, task generation")
    missing = [s for s in results.GENERATION_STAGES if s not in stages]
    if missing:
        raise ConversionError(f"{directory}: generation results lack the stage file(s) {missing}")
    return stages


def merge_rows(stages: dict[str, results.StageResults]) -> dict[str, dict[str, Any]]:
    """Per-case rows in the shape the generation checks read (the field names of the runner's `results.jsonl`)."""
    rows: dict[str, dict[str, Any]] = {}
    for name, case in stages["template"].cases.items():
        rows.setdefault(name, {"id": name}).update(
            rendered=case.get("rendered"),
            render_error=case.get("render_error"),
            backend_prompt_ids=case.get("prompt_ids"),
        )
    for name, case in stages["dotnet"].cases.items():
        row = rows.setdefault(name, {"id": name})
        for key in ("prompt_ids", "output_ids", "finish_reason", "step0_top", *STREAM_FLAGS):
            if key in case:
                row[key] = case[key]
        usage = case.get("usage")
        if isinstance(usage, dict):
            row["usage_prompt_tokens"] = usage.get("prompt_tokens")
            row["usage_completion_tokens"] = usage.get("completion_tokens")
            row["usage_consistent"] = usage.get("consistent")
    return rows


def read_step0(stage: results.StageResults, vocab: int) -> dict[str, list[float]]:
    """The step-0 sidecar of the dotnet stage (checked against its digest); empty when the run wrote none."""
    if results.STEP0_TENSOR not in stage.tensor_specs:
        return {}
    try:
        tensor = stage.tensor(results.STEP0_TENSOR)
    except results.ResultsError as exc:
        raise ConversionError(str(exc)) from exc
    bad = [name for name, shape in tensor.shapes.items() if shape != [vocab]]
    if bad:
        raise ConversionError(f"{stage.path}: step-0 vectors of {bad} do not have shape [{vocab}]")
    return tensor.values


def describe_files(stages: dict[str, results.StageResults]) -> list[Path]:
    files = [stages[s].path for s in results.GENERATION_STAGES]
    spec = stages["dotnet"].tensor_specs.get(results.STEP0_TENSOR)
    if spec:
        files.append(stages["dotnet"].path.parent / spec["file"])
    return files
