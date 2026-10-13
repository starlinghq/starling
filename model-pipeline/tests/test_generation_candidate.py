from __future__ import annotations

import array
import hashlib
import json
from pathlib import Path

import pytest

from model_pipeline.evaluate_generation_candidate import CHECKS, POLICY, evaluate
from model_pipeline.onboarding_runner import PipelineError


def evidence(tmp_path: Path):
    candidate = {
        "id": "test-generation",
        "origin": {"revision": "a" * 40},
        "runtime": {"version": "0.17.1"},
        "behavior": {"vocab_size": 4, "eos_token_ids": [3], "max_new_tokens": 1},
        "evaluation": {"policy": POLICY},
    }
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"id": "one"}) + "\n")
    directories = []
    for backend in CHECKS:
        directory = tmp_path / backend
        directory.mkdir()
        raw = array.array("f", [1, 2, 3, 4]).tobytes()
        (directory / "initial.f32").write_bytes(raw)
        row = {
            "id": "one",
            "profile": candidate["id"],
            "revision": candidate["origin"]["revision"],
            "backend": backend,
            "runtime": {
                "backend": backend,
                "version": "5.18.0" if backend == "transformers" else "0.17.1",
                "precision": "fp32",
                "provider": "CPUExecutionProvider",
            },
            "prompt_text": "hello",
            "prompt_ids": [1],
            "output_ids": [2],
            "output_text": "world",
            "finish_reason": "length",
            "stop_token_id": None,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "checks": dict.fromkeys(CHECKS[backend], True),
            "initial_logits": {
                "path": "initial.f32",
                "dtype": "float32-le",
                "shape": [1, 4],
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
            "cancellation_between_steps": True,
        }
        (directory / "results.jsonl").write_text(json.dumps(row) + "\n")
        directories.append(directory)
    return candidate, corpus, directories


def mutate(directory: Path, change):
    path = directory / "results.jsonl"
    row = json.loads(path.read_text())
    change(row)
    path.write_text(json.dumps(row) + "\n")


def test_matching_development_evidence_is_not_release_qualification(tmp_path):
    candidate, corpus, directories = evidence(tmp_path)
    report = evaluate(candidate, corpus, *directories)
    assert report["status"] == "passed"
    assert report["reference_status"] == "proposed"
    assert report["protected_release_qualification"] == "unapproved"


@pytest.mark.parametrize(
    "key,value",
    [
        ("revision", "wrong"),
        ("prompt_ids", [True]),
        ("output_ids", [9]),
        ("usage", {}),
        ("checks", {}),
        ("finish_reason", "eos"),
    ],
)
def test_rejects_incomplete_or_wrong_evidence(tmp_path, key, value):
    candidate, corpus, directories = evidence(tmp_path)
    mutate(directories[2], lambda row: row.update({key: value}))
    with pytest.raises(PipelineError):
        evaluate(candidate, corpus, *directories)


@pytest.mark.parametrize("key", ["stop_token_id", "finish_reason"])
def test_missing_termination_keys_are_malformed_evidence(tmp_path, key):
    candidate, corpus, directories = evidence(tmp_path)
    mutate(directories[2], lambda row: row.pop(key))
    with pytest.raises(PipelineError, match="malformed evidence"):
        evaluate(candidate, corpus, *directories)


def test_rejects_duplicate_rows(tmp_path):
    candidate, corpus, directories = evidence(tmp_path)
    path = directories[1] / "results.jsonl"
    path.write_text(path.read_text() * 2)
    with pytest.raises(PipelineError, match="duplicate"):
        evaluate(candidate, corpus, *directories)


@pytest.mark.parametrize("values", [[1, 2, 3], [1, 2, 3, float("nan")], [1, 2, 3, float("inf")]])
def test_rejects_invalid_logits(tmp_path, values):
    candidate, corpus, directories = evidence(tmp_path)
    raw = array.array("f", values).tobytes()
    (directories[2] / "initial.f32").write_bytes(raw)
    mutate(directories[2], lambda row: row["initial_logits"].update({"sha256": hashlib.sha256(raw).hexdigest()}))
    with pytest.raises(PipelineError):
        evaluate(candidate, corpus, *directories)


def test_actual_discrete_divergence_fails(tmp_path):
    candidate, corpus, directories = evidence(tmp_path)
    mutate(directories[2], lambda row: row.update({"output_ids": [1]}))
    assert evaluate(candidate, corpus, *directories)["status"] == "failed"


def test_actual_numerical_divergence_fails(tmp_path):
    candidate, corpus, directories = evidence(tmp_path)
    raw = array.array("f", [1, 2, 3, 4.1]).tobytes()
    (directories[2] / "initial.f32").write_bytes(raw)
    mutate(directories[2], lambda row: row["initial_logits"].update({"sha256": hashlib.sha256(raw).hexdigest()}))
    assert evaluate(candidate, corpus, *directories)["status"] == "failed"


def test_semantic_failure_is_not_silently_ignored(tmp_path):
    candidate, corpus, directories = evidence(tmp_path)
    mutate(directories[2], lambda row: row["checks"].update({"isolation": False}))
    assert evaluate(candidate, corpus, *directories)["status"] == "failed"


def test_requires_actual_between_step_cancellation_case(tmp_path):
    candidate, corpus, directories = evidence(tmp_path)
    mutate(directories[2], lambda row: row.update({"cancellation_between_steps": None}))
    with pytest.raises(PipelineError, match="between decode"):
        evaluate(candidate, corpus, *directories)


def test_duplicate_tensor_paths_are_rejected(tmp_path):
    candidate, corpus, directories = evidence(tmp_path)
    corpus.write_text(corpus.read_text() + json.dumps({"id": "two"}) + "\n")
    for directory in directories:
        path = directory / "results.jsonl"
        row = json.loads(path.read_text())
        row["id"] = "two"
        path.write_text(path.read_text() + json.dumps(row) + "\n")
    with pytest.raises(PipelineError, match="duplicate tensor"):
        evaluate(candidate, corpus, *directories)


def test_jsonl_unicode_line_separators_are_text_not_record_boundaries(tmp_path):
    candidate, corpus, directories = evidence(tmp_path)
    for directory in directories:
        path = directory / "results.jsonl"
        row = json.loads(path.read_text())
        row["prompt_text"] = "hello\u2028world\u0085line\u2029end"
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n")
    assert evaluate(candidate, corpus, *directories)["status"] == "passed"


def test_rejects_wrong_runtime_identity(tmp_path):
    candidate, corpus, directories = evidence(tmp_path)
    mutate(directories[0], lambda row: row["runtime"].update({"version": "5.19.0"}))
    with pytest.raises(PipelineError, match="runtime identity"):
        evaluate(candidate, corpus, *directories)
