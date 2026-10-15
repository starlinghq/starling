"""Staged fidelity evaluator, task generation, on a synthetic profile in a throwaway Git repository."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import subprocess
from pathlib import Path

import pytest
import yaml

from model_pipeline import evaluate, fixtures
from model_pipeline.cli import main
from model_pipeline.manifest import load

SYNTHETIC = Path(__file__).parent / "data" / "synthetic-manifest.yaml"
PROFILE = "synthetic-llm/fp32-cpu"
PROTECTED = "model-pipeline/protected/synthetic-llm"
VOCAB = 8
TOLERANCES = {
    "profile": PROFILE,
    "provider_options": {},
    "discrete": {
        "template_rendering": "exact",
        "prompt_ids": "exact",
        "step0_argmax": "equal",
        "streaming_reassembly": "exact",
        "usage": "exact",
        "concurrent_requests": "equal_to_sequential",
        "greedy_ids": "exact_except_listed_margin_below_1e-3",
        "finish_reason": "equal",
    },
    "float": {
        "step0_logprob_max_abs": 5.0e-3,
        "step0_top5_logprob_max_abs": 2.0e-3,
        "cached_vs_uncached_logprob_max_abs": 2.0e-4,
        "chunked_vs_one_call_prefill_logprob_max_abs": 1.0e-4,
        "nan_or_inf": "fail",
    },
}


def logprobs(seed: int) -> list[float]:
    raw = [math.sin(seed * 7 + i) * 3 for i in range(VOCAB)]
    m = max(raw)
    z = math.log(sum(math.exp(v - m) for v in raw)) + m
    return [struct.unpack("<f", struct.pack("<f", v - z))[0] for v in raw]


def f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


CASES = [
    {
        "id": "a",
        "tools": None,
        "rendered": "<u>hi</u><a>",
        "prompt_ids": [1, 2, 3],
        "output_ids": [4, 5, 7],
        "finish_reason": "eos",
        "step_margin": [0.5, 0.0005, 2.0],
    },
    {
        "id": "b",
        "tools": None,
        "rendered": "<u>yo</u><a>",
        "prompt_ids": [1, 6, 3],
        "output_ids": [5, 5],
        "finish_reason": "length",
        "step_margin": [1.0, 0.2],
    },
    {
        "id": "t",
        "tools": [{"name": "f"}],
        "rendered": "<tools/><a>",
        "prompt_ids": [1, 0, 3],
        "output_ids": [7],
        "finish_reason": "eos",
        "step_margin": [3.0],
    },
]


def fixture_rows() -> list[dict]:
    rows = []
    for k, case in enumerate(CASES):
        lp = logprobs(k)
        top = sorted(range(VOCAB), key=lambda i: -lp[i])
        rows.append(
            {
                **case,
                "step0_top": [[i, lp[i]] for i in top[:6]],
                "step0_vocab": VOCAB,
                "step0_sha256": hashlib.sha256(f32(lp)).hexdigest(),
            }
        )
    return rows


def git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=root,
        capture_output=True,
        check=True,
    )


class Repo:
    def __init__(self, root: Path, tolerances: dict | None = None):
        self.root = root
        self.protected = root / PROTECTED
        (self.protected / "reference").mkdir(parents=True)
        (self.protected / "fp32-cpu").mkdir(parents=True)
        fixtures.write_jsonl(self.protected / "reference" / "generation.jsonl", fixture_rows())
        (self.protected / "fp32-cpu" / "tolerances.yaml").write_text(yaml.safe_dump(tolerances or TOLERANCES), "utf-8")
        m = load(SYNTHETIC)
        m["id"] = PROFILE
        m["behavior"].update(tasks=["generation"])
        m["qualification"]["fixtures"] = f"{PROTECTED}/reference"
        m["qualification"]["tolerances"] = f"{PROTECTED}/fp32-cpu/tolerances.yaml"
        self.manifest = root / "model-pipeline" / "registry" / PROFILE / "manifest.yaml"
        self.manifest.parent.mkdir(parents=True)
        self.manifest.write_text(yaml.safe_dump(m, sort_keys=False), encoding="utf-8")
        self.results = root / "run"
        self.tensors = root / "reference-step0"
        (self.results / "step0").mkdir(parents=True)
        (self.tensors / "step0").mkdir(parents=True)
        for k, case in enumerate(CASES):
            (self.tensors / "step0" / f"{case['id']}.f32").write_bytes(f32(logprobs(k)))
        git(root, "init", "-q")
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", "synthetic")

    def write_run(self, edit=None, noise: float = 0.0, semantics: dict | None = None, environment=True) -> None:
        rows = []
        for k, case in enumerate(CASES):
            row = {
                "id": case["id"],
                "backend": "test",
                "prompt_ids": case["prompt_ids"],
                "output_ids": list(case["output_ids"]),
                "finish_reason": case["finish_reason"],
                "rendered": None if case["tools"] else case["rendered"],
                "render_error": "NotSupportedException: tools" if case["tools"] else None,
                "backend_prompt_ids": None if case["tools"] else case["prompt_ids"],
                "stream_text_equals_result": True,
                "stream_ids_equal_result": True,
                "stream_text_equals_decode": True,
                "usage_prompt_tokens": len(case["prompt_ids"]),
                "usage_completion_tokens": len(case["output_ids"]) - (case["finish_reason"] == "eos"),
                "usage_consistent": True,
            }
            if edit:
                edit(row)
            rows.append(row)
            vec = [v + (noise if i == 0 else 0.0) for i, v in enumerate(logprobs(k))]
            (self.results / "step0" / f"{case['id']}.f32").write_bytes(f32(vec))
        fixtures.write_jsonl(self.results / "results.jsonl", rows)
        sem = (
            semantics
            if semantics is not None
            else {
                "cached_vs_uncached": {"max_abs_logprob": 3e-5, "steps": 12, "generated_equals_reference_prefix": True},
                "chunked_prefill": {
                    "a/7": {
                        "whole_prefill_error": None,
                        "tokens_equal_whole": True,
                        "step0_max_abs_logprob_vs_whole": 0.0,
                    }
                },
                "concurrency": {
                    "parallel_two_threads_prefix_equal": {"a": True, "b": True},
                    "interleaved_streams_prefix_equal": {"a": True},
                },
            }
        )
        (self.results / "semantics.json").write_text(json.dumps(sem), encoding="utf-8")
        if environment:
            env = {"runner": "test", "profile": PROFILE, "provider_options": {}}
            (self.results / "environment.json").write_text(json.dumps(env), encoding="utf-8")

    def run(self, tensors=True):
        return evaluate.evaluate(
            self.manifest,
            self.results,
            repo_root=self.root,
            protected_base="HEAD",
            reference_tensors=self.tensors if tensors else None,
        )


def status(ev, check_id):
    return next(c for c in ev.report["checks"] if c["id"] == check_id)


def test_identical_run_passes_both_gates(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run()
    ev = repo.run()
    assert ev.report["gates"]["G1"]["status"] == "passed", ev.report["gates"]
    assert ev.report["gates"]["G2"]["status"] == "passed", ev.report["gates"]
    assert ev.exit_code == 0
    assert "t" in ev.report["excluded_by_contract"]
    assert "Template stage" in evaluate.markdown(ev.report)


def test_rendering_difference_fails_g1(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run(edit=lambda r: r.update(rendered="<u>HI</u>") if r["id"] == "a" else None)
    ev = repo.run()
    assert status(ev, "tokenizer.template_rendering")["status"] == "failed"
    assert ev.report["gates"]["G1"]["status"] == "failed"
    assert ev.report["first_divergence"] == "tokenizer"
    assert ev.exit_code == 1


def test_tools_prompt_must_be_rejected(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run(edit=lambda r: r.update(rendered="<tools/><a>", render_error=None) if r["id"] == "t" else None)
    assert status(repo.run(), "tokenizer.template_rendering")["status"] == "failed"


def test_near_tie_divergence_is_listed_and_allowed(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run(edit=lambda r: r.update(output_ids=[4, 6, 7]) if r["id"] == "a" else None)
    ev = repo.run()
    check = status(ev, "dotnet-ids.greedy_ids")
    assert check["status"] == "passed"
    assert ev.report["greedy_divergences"] == [{"id": "a", "step": 1, "reference_margin": 0.0005}]


def test_divergence_at_large_margin_fails(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run(edit=lambda r: r.update(output_ids=[6, 5]) if r["id"] == "b" else None)
    ev = repo.run()
    assert status(ev, "dotnet-ids.greedy_ids")["status"] == "failed"
    assert ev.report["gates"]["G2"]["status"] == "failed"


def test_step0_bounds(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run(noise=1e-2)
    ev = repo.run()
    assert status(ev, "dotnet-ids.step0_logprob")["status"] == "failed"
    # Without the reference vectors a passing top-64 value is only a lower bound: not-run, G2 cannot pass.
    repo.write_run(noise=1e-4)
    ev = repo.run(tensors=False)
    assert status(ev, "dotnet-ids.step0_logprob")["status"] == "not-run"
    assert ev.report["gates"]["G2"]["status"] == "not-run"


def test_reference_tensor_digest_is_checked(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run()
    (repo.tensors / "step0" / "a.f32").write_bytes(f32([0.0] * VOCAB))
    with pytest.raises(evaluate.EvaluationError, match="SHA-256"):
        repo.run()


def test_missing_semantics_and_environment_leave_g2_open(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run(environment=False)
    (repo.results / "semantics.json").unlink()
    ev = repo.run()
    assert ev.report["gates"]["G1"]["status"] == "passed"
    assert ev.report["gates"]["G2"]["status"] == "not-run"


def test_semantics_failures(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run(
        semantics={
            "cached_vs_uncached": {"max_abs_logprob": 1e-3, "generated_equals_reference_prefix": True},
            "chunked_prefill": {"a/7": {"whole_prefill_error": "OOM"}},
            "concurrency": {"parallel_two_threads_prefix_equal": {"a": False}},
        }
    )
    ev = repo.run()
    assert status(ev, "dotnet-text.cached_vs_uncached")["status"] == "failed"
    assert status(ev, "dotnet-text.chunked_prefill")["status"] == "not-run"
    assert status(ev, "dotnet-text.concurrent_requests")["status"] == "failed"


def test_semantics_of_another_fixture_are_refused(tmp_path):
    # A semantics.json measured on other prompts (for example the development prompts during a release evaluation)
    # cannot stand in for the evaluated fixture.
    repo = Repo(tmp_path)
    for sem in (
        {"cached_vs_uncached": {"prompt": "en-list", "max_abs_logprob": 3e-5}},
        {"chunked_prefill": {"long-license/128": {"tokens_equal_whole": True, "step0_max_abs_logprob_vs_whole": 0.0}}},
        {"concurrency": {"interleaved_streams_prefix_equal": {"zh": True}}},
        {"selection": {"cached_vs_uncached": {"case": "en-list"}}},
    ):
        repo.write_run(semantics=sem)
        with pytest.raises(evaluate.EvaluationError, match="not in the fixture"):
            repo.run()


def test_fixture_semantics_with_selection_pass(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run(
        semantics={
            "selection": {
                "rule": "fixed rule",
                "cached_vs_uncached": {"case": "b", "min_margin_12_steps": 0.5},
                "concurrency_parallel": [{"case": "a"}, {"case": "b"}],
                "chunked_prefill": [{"case": "a", "chunk": 7, "prompt_tokens": 9}],
            },
            "cached_vs_uncached": {
                "prompt": "b",
                "max_abs_logprob": 3e-5,
                "steps": 12,
                "generated_equals_reference_prefix": True,
            },
            "chunked_prefill": {
                "a/7": {"whole_prefill_error": None, "tokens_equal_whole": True, "step0_max_abs_logprob_vs_whole": 0.0}
            },
            "concurrency": {
                "parallel_two_threads_prefix_equal": {"a": True, "b": True},
                "interleaved_streams_prefix_equal": {"a": True},
            },
        }
    )
    ev = repo.run()
    assert ev.report["gates"]["G2"]["status"] == "passed", ev.report["gates"]


def test_missing_case_fails(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run()
    lines = (repo.results / "results.jsonl").read_text(encoding="utf-8").splitlines()
    (repo.results / "results.jsonl").write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
    ev = repo.run()
    assert status(ev, "dotnet-ids.greedy_ids")["status"] == "failed"
    assert ev.report["gates"]["G2"]["status"] == "failed"


def test_results_may_not_carry_thresholds(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run(edit=lambda r: r.update(threshold=1.0))
    code = main(
        [
            "evaluate",
            str(repo.manifest),
            "--results",
            str(repo.results),
            "--out",
            str(tmp_path / "out"),
            "--repo-root",
            str(repo.root),
        ]
    )
    assert code == evaluate.EXIT_PROTECTED


def test_unimplemented_rule_leaves_gate_open(tmp_path):
    tol = json.loads(json.dumps(TOLERANCES))
    tol["discrete"]["greedy_ids"] = "something_else"
    repo = Repo(tmp_path, tol)
    repo.write_run()
    ev = repo.run()
    assert status(ev, "dotnet-ids.greedy_ids")["status"] == "not-run"
    assert ev.report["gates"]["G2"]["status"] == "not-run"


def test_cli_writes_report(tmp_path):
    repo = Repo(tmp_path)
    repo.write_run()
    out = tmp_path / "out"
    code = main(
        [
            "evaluate",
            str(repo.manifest),
            "--results",
            str(repo.results),
            "--out",
            str(out),
            "--repo-root",
            str(repo.root),
            "--protected-base",
            "HEAD",
            "--reference-tensors",
            str(repo.tensors),
            "--require-gates",
            "G1,G2",
        ]
    )
    assert code == 0
    assert json.loads((out / "report.json").read_text(encoding="utf-8"))["task"] == "generation"
