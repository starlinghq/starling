"""Staged fidelity evaluator, task `generation` (Qwen3.5 on ONNX Runtime GenAI; P-016 G1/G2, workstream R5).

Compares the output directory of the C# fidelity runner (`ai-dotnet/tools/Starlings.Qwen35.Fidelity`, modes
`parity,semantics`) with the protected generation fixture (`generation.jsonl`, format `starlings.generation-fixture`
version 1), using only the profile's protected tolerances:

    tokenizer    the template stage: per case the owned template rendering and the owned tokenizer's ids, exact
                 against the fixture `rendered` and `prompt_ids`; the case with tools must be rejected (the manifest
                 lists tools as unsupported)                                                               (G1)
    export       not applicable: the graph is the builder's output; no Python ONNX Runtime GenAI stage is bound
    dotnet-ids   greedy generation from the fixture prompt ids: greedy ids (near-tie rule reading the fixture
                 `step_margin`), finish reason, step-0 argmax, step-0 log-probabilities (top 5 from the fixture;
                 full vocabulary only with the reference vectors), NaN/Inf, streaming reassembly, usage, provider
                 options                                                                                   (G2)
    dotnet-text  runtime semantics from `semantics.json`: cached vs uncached logits, chunked vs one-call prefill,
                 concurrent and interleaved requests                                                       (G2)

Runner directory (`--results`): `results.jsonl` (one row per case, the format of
`model-pipeline/reference/generation/compare.py` plus the runner's checks), `step0/<id>.f32` (log-softmax after the
prefill, float32 little-endian, one per case), `semantics.json` and `environment.json`. Every fixture case is
required; a case missing from the results fails the check that needs it.

The full reference step-0 vectors are not in Git (the fixture keeps the top 64 per case and the SHA-256 of each full
vector). Pass `--reference-tensors <dir>` holding `step0/<id>.f32` from `run_reference.py`; each file is checked
against the fixture digest. Without them `float.step0_logprob_max_abs` is measured on the 64 reference ids only: a
value above the bound fails (the full maximum can only be larger), otherwise the check is not-run and G2 cannot pass.

Reuses the readers of `evaluate_generation_candidate.py` (rows, ids, float32 sidecars).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_pipeline import fixtures, results
from model_pipeline.evaluate import Check, EvaluationError, _bound, flatten, substage_status
from model_pipeline.evaluate_generation_candidate import read_float32, read_rows, valid_ids
from model_pipeline.onboarding_runner import PipelineError

TASK = "generation"
SUBSTAGES = ("tokenizer", "export", "dotnet-ids", "dotnet-text")
SUBSTAGE_TITLES = {
    "tokenizer": "Template stage: owned template rendering and owned tokenizer ids against the fixture",
    "export": "Export (not applicable: no Python ONNX Runtime GenAI stage is bound)",
    "dotnet-ids": ".NET greedy generation from the fixture prompt ids",
    "dotnet-text": ".NET runtime semantics (cached vs uncached, chunked prefill, concurrency)",
}
FIXTURE_FORMAT = "starlings.generation-fixture"

EXPECTED_KEYS: dict[str, Any] = {
    "profile": "<family>/<profile>, must equal the manifest id",
    "provider_options": "{} on CPU; {use_tf32: 0} on CUDA",
    "discrete.template_rendering": "exact",
    "discrete.prompt_ids": "exact",
    "discrete.step0_argmax": "equal",
    "discrete.streaming_reassembly": "exact",
    "discrete.usage": "exact",
    "discrete.concurrent_requests": "equal_to_sequential",
    "discrete.greedy_ids": (
        "exact | exact_except_listed_margin_below_<m> | equal_until_first_divergence_at_margin_below_<m>"
    ),
    "discrete.finish_reason": "equal | equal_when_ids_equal",
    "float.step0_logprob_max_abs": 5.0e-3,
    "float.step0_top5_logprob_max_abs": 2.0e-3,
    "float.cached_vs_uncached_logprob_max_abs": 2.0e-4,
    "float.chunked_vs_one_call_prefill_logprob_max_abs": 1.0e-4,
    "float.nan_or_inf": "fail",
}
KNOWN_KEYS = set(EXPECTED_KEYS) | {"reference"}

_GREEDY_RULE = re.compile(
    r"^(?:exact_except_listed_margin_below_|equal_until_first_divergence_at_margin_below_)([0-9.eE+-]+)$"
)


@dataclass
class Fixtures:
    directory: Path
    cases: dict[str, dict]
    files: list[Path]


def load_fixtures(directory: Path) -> Fixtures:
    path = directory / "generation.jsonl"
    try:
        cases = read_rows(path)
    except PipelineError as exc:
        raise EvaluationError(str(exc)) from exc
    for name, case in cases.items():
        missing = [
            k
            for k in (
                "rendered",
                "prompt_ids",
                "output_ids",
                "finish_reason",
                "step_margin",
                "step0_top",
                "step0_vocab",
            )
            if k not in case
        ]
        if missing:
            raise EvaluationError(f"{path}: case {name!r} lacks {missing}")
    files = [path]
    if (directory / "environment.json").is_file():
        files.append(directory / "environment.json")
    return Fixtures(directory, cases, files)


@dataclass
class RunnerResults:
    directory: Path
    rows: dict[str, dict]
    semantics: dict | None
    environment: dict | None
    files: list[Path]

    def describe(self, rel) -> dict[str, Any]:
        return {f.name: {"path": rel(f), "sha256": fixtures.sha256_file(f)} for f in self.files if f.is_file()} | {
            "runner": (self.environment or {}).get("runner"),
            "backend": (self.environment or {}).get("backend"),
            "source_revision": (self.environment or {}).get("source_revision"),
        }


def _json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"{path}: {exc}") from exc
    if not isinstance(data, dict):
        raise EvaluationError(f"{path}: expected a JSON object")
    return data


def load_results(directory: Path) -> RunnerResults:
    path = directory / "results.jsonl"
    if not path.is_file():
        raise EvaluationError(
            f"{directory}: no results.jsonl; the generation task reads the output directory of "
            "Starlings.Qwen35.Fidelity (--modes parity,semantics)"
        )
    try:
        rows = read_rows(path)
    except PipelineError as exc:
        raise EvaluationError(str(exc)) from exc
    semantics = _json(directory / "semantics.json")
    environment = _json(directory / "environment.json")
    for where, value in (("results.jsonl", list(rows.values())), ("semantics.json", semantics)):
        forbidden = results._find_forbidden(value)
        if forbidden:
            raise results.ToleranceOverrideError(
                f"{directory / where}: runner outputs carry outputs only; thresholds come from the profile's "
                f"protected tolerances file. Rejected keys: {', '.join(forbidden)}"
            )
    files = [path, directory / "semantics.json", directory / "environment.json"]
    return RunnerResults(directory, rows, semantics, environment, [f for f in files if f.is_file()])


def _same_profile(env: dict, profile_id: str) -> bool:
    """The runner's environment names the profile, or (Starlings.Qwen35.Fidelity) the variant without the device,
    `qwen3.5-0.8b/ortgenai-fp32x` for `...-fp32x-cpu`, with the device in its backend name `df-qwen35-cpu-...`."""
    profile = env.get("profile")
    if profile is None or profile == profile_id:
        return True
    device = profile_id.rsplit("-", 1)[-1]
    return profile_id == f"{profile}-{device}" and f"-{device}-" in str(env.get("backend", ""))


def _first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


class GenerationEvaluator:
    def __init__(
        self,
        man: dict[str, Any],
        fx: Fixtures,
        tol: dict[str, Any],
        run: RunnerResults,
        reference_tensors: Path | None = None,
    ):
        self.man = man
        self.fx = fx
        self.tol = flatten(tol)
        self.raw_tol = tol
        self.run = run
        self.reference_tensors = reference_tensors
        self.checks: list[Check] = []
        self.vocab = man["behavior"].get("vocab_size") or next(iter(fx.cases.values()))["step0_vocab"]
        self._step0: dict[str, list[float] | None] = {}
        self.divergences: list[dict[str, Any]] = []
        self.excluded: dict[str, str] = {}

    # helpers ---------------------------------------------------------------------------------------
    def _add(self, check: Check) -> Check:
        self.checks.append(check)
        return check

    def _key_check(self, cid: str, substage: str, title: str, key: str) -> Check:
        check = Check(cid, substage, title, key, self.tol.get(key))
        if key not in self.tol:
            check.detail = f"tolerance key {key} is not set in the protected tolerances file"
        return self._add(check)

    def _missing(self) -> list[str]:
        return sorted(set(self.fx.cases) - set(self.run.rows))

    def _candidate_step0(self, name: str) -> list[float] | None:
        if name not in self._step0:
            try:
                self._step0[name] = list(read_float32(self.run.directory, f"step0/{name}.f32", self.vocab))
            except PipelineError:
                self._step0[name] = None
        return self._step0[name]

    def _reference_step0(self, name: str) -> list[float] | None:
        if self.reference_tensors is None:
            return None
        case = self.fx.cases[name]
        try:
            return list(read_float32(self.reference_tensors, f"step0/{name}.f32", self.vocab, case.get("step0_sha256")))
        except PipelineError as exc:
            raise EvaluationError(f"reference step-0 vector of {name!r}: {exc}") from exc

    # run -------------------------------------------------------------------------------------------
    def run_checks(self) -> list[Check]:
        unknown = sorted(set(self.run.rows) - set(self.fx.cases))
        if unknown:
            raise EvaluationError(f"{self.run.directory}: results for cases not in the fixture: {unknown}")
        self._identity()
        self._template()
        self._generation()
        self._semantics()
        for key in sorted(set(self.tol) - KNOWN_KEYS - {k for k in self.tol if k.startswith("provider_options")}):
            substage = "tokenizer" if key in ("discrete.template_rendering", "discrete.prompt_ids") else "dotnet-ids"
            self._add(
                Check(
                    f"tolerances.{key}",
                    substage,
                    f"Tolerance key {key}",
                    key,
                    self.tol[key],
                    "not-run",
                    detail="the generation evaluator does not implement this key; its gate cannot pass",
                )
            )
        return self.checks

    def _identity(self) -> None:
        check = self._add(Check("dotnet-ids.identity", "dotnet-ids", "Runner identity and fed prompt ids"))
        env = self.run.environment
        problems = []
        if env is not None and not _same_profile(env, self.man["id"]):
            problems.append(
                f"environment.json profile {env.get('profile')!r} (backend {env.get('backend')!r}) is not "
                f"{self.man['id']!r}"
            )
        for name, row in self.run.rows.items():
            if row.get("prompt_ids") != self.fx.cases[name]["prompt_ids"]:
                problems.append(f"{name}: fed prompt ids differ from the fixture")
            if not valid_ids(row.get("output_ids"), self.vocab):
                problems.append(f"{name}: output_ids missing or out of vocabulary")
        if problems:
            check.status, check.detail = "invalid", "; ".join(problems[:8])
        else:
            check.status = "passed"
            check.metrics = {"cases": len(self.run.rows), "environment": env is not None}

    def _template(self) -> None:
        render = self._key_check(
            "tokenizer.template_rendering", "tokenizer", "Owned template rendering exact", "discrete.template_rendering"
        )
        ids = self._key_check("tokenizer.prompt_ids", "tokenizer", "Owned tokenizer ids exact", "discrete.prompt_ids")
        missing = self._missing()
        render_fail, ids_fail, compared = [], [], 0
        for name, case in self.fx.cases.items():
            row = self.run.rows.get(name)
            if row is None:
                continue
            if case.get("tools"):
                self.excluded[name] = "tools prompt: unsupported by the owned renderer (manifest behavior.unsupported)"
                if row.get("rendered") is not None or not row.get("render_error"):
                    render_fail.append(f"{name}: tools prompt was rendered instead of rejected")
                continue
            compared += 1
            if row.get("rendered") != case["rendered"]:
                render_fail.append(
                    f"{name}: rendering differs"
                    if row.get("rendered") is not None
                    else f"{name}: not rendered ({row.get('render_error')})"
                )
            if row.get("backend_prompt_ids") != case["prompt_ids"]:
                ids_fail.append(name)
        for check, failed, rule in ((render, render_fail, "exact"), (ids, ids_fail, "exact")):
            check.metrics = {"compared": compared, "failed": len(failed), "excluded_by_contract": sorted(self.excluded)}
            if check.tolerance_key not in self.tol:
                continue
            if check.bound != rule:
                check.detail = f"rule {check.bound!r} is not implemented"
                continue
            if failed or missing:
                check.status = "failed"
                check.detail = "; ".join(
                    [*(str(f) for f in failed[:8]), *([f"missing cases {missing}"] if missing else [])]
                )
            else:
                check.status = "passed"

    def _generation(self) -> None:
        missing = self._missing()
        present = [n for n in self.fx.cases if n in self.run.rows]
        miss = f"missing cases {missing}" if missing else ""

        # Greedy ids.
        greedy = self._key_check(
            "dotnet-ids.greedy_ids", "dotnet-ids", "Greedy ids against the reference", "discrete.greedy_ids"
        )
        rule = greedy.bound
        margin_bound: float | None = None
        if rule == "exact":
            margin_bound = 0.0
        elif isinstance(rule, str) and (m := _GREEDY_RULE.match(rule)):
            margin_bound = float(m.group(1))
        exact = 0
        bad = []
        for name in present:
            ref, cand = self.fx.cases[name]["output_ids"], self.run.rows[name]["output_ids"]
            step = _first_divergence(ref, cand)
            if step is None:
                exact += 1
                continue
            margins = self.fx.cases[name]["step_margin"]
            margin = margins[step] if step < len(margins) else None
            self.divergences.append({"id": name, "step": step, "reference_margin": margin})
            if margin_bound is None or margin is None or not margin < margin_bound:
                bad.append(f"{name} at step {step} (reference margin {margin})")
        greedy.metrics = {"exact": exact, "compared": len(present), "divergences": self.divergences}
        if "discrete.greedy_ids" in self.tol:
            if margin_bound is None:
                greedy.detail = f"rule {rule!r} is not implemented"
            elif bad or missing:
                greedy.status, greedy.detail = "failed", "; ".join([*bad[:8], *([miss] if miss else [])])
            else:
                greedy.status = "passed"
                if self.divergences:
                    greedy.detail = "allowed near-tie divergences: " + ", ".join(
                        f"{d['id']}@{d['step']} ({d['reference_margin']:.3g})" for d in self.divergences
                    )

        # Finish reason.
        finish = self._key_check("dotnet-ids.finish_reason", "dotnet-ids", "Finish reason", "discrete.finish_reason")
        diverged = {d["id"] for d in self.divergences}
        wrong = [n for n in present if self.run.rows[n].get("finish_reason") != self.fx.cases[n]["finish_reason"]]
        finish.metrics = {"equal": len(present) - len(wrong), "compared": len(present)}
        if "discrete.finish_reason" in self.tol:
            if finish.bound == "equal":
                bad_finish = wrong
            elif finish.bound == "equal_when_ids_equal":
                bad_finish = [n for n in wrong if n not in diverged]
            else:
                bad_finish = None
                finish.detail = f"rule {finish.bound!r} is not implemented"
            if bad_finish is not None:
                if bad_finish or missing:
                    finish.status, finish.detail = "failed", f"differs: {bad_finish} {miss}".strip()
                else:
                    finish.status = "passed"

        # Step-0 vectors.
        nan = self._key_check(
            "dotnet-ids.nan_or_inf", "dotnet-ids", "Step-0 log-probabilities finite", "float.nan_or_inf"
        )
        argmax = self._key_check("dotnet-ids.step0_argmax", "dotnet-ids", "Step-0 argmax", "discrete.step0_argmax")
        top5 = self._key_check(
            "dotnet-ids.step0_top5_logprob",
            "dotnet-ids",
            "Step-0 log-probability, reference top 5",
            "float.step0_top5_logprob_max_abs",
        )
        full = self._key_check(
            "dotnet-ids.step0_logprob",
            "dotnet-ids",
            "Step-0 log-probability, full vocabulary",
            "float.step0_logprob_max_abs",
        )
        no_vector, non_finite, argmax_bad = [], [], []
        top5_max = top64_max = full_max = 0.0
        top5_worst = full_worst = None
        for name in present:
            vec = self._candidate_step0(name)
            if vec is None:
                no_vector.append(name)
                continue
            if not all(math.isfinite(v) for v in vec):
                non_finite.append(name)
                continue
            top = self.fx.cases[name]["step0_top"]
            best = max(range(len(vec)), key=vec.__getitem__)
            if best != top[0][0]:
                argmax_bad.append(name)
            for rank, (token, lp) in enumerate(top):
                d = abs(vec[token] - lp)
                if rank < 5 and d > top5_max:
                    top5_max, top5_worst = d, name
                top64_max = max(top64_max, d)
            ref = self._reference_step0(name)
            if ref is not None:
                d = max(abs(a - b) for a, b in zip(ref, vec, strict=True))
                if d > full_max:
                    full_max, full_worst = d, name
        lacking = sorted(set(missing) | set(no_vector))
        lacking_detail = f"missing step-0 vectors or cases {lacking}" if lacking else ""
        nan.metrics = {"non_finite_cases": non_finite}
        if "float.nan_or_inf" in self.tol:
            nan.status = "failed" if non_finite or lacking else "passed"
            nan.detail = f"non-finite: {non_finite} {lacking_detail}".strip() if nan.status == "failed" else ""
        argmax.metrics = {"equal": len(present) - len(no_vector) - len(non_finite) - len(argmax_bad)}
        if "discrete.step0_argmax" in self.tol:
            if argmax.bound != "equal":
                argmax.detail = f"rule {argmax.bound!r} is not implemented"
            elif argmax_bad or lacking or non_finite:
                argmax.status, argmax.detail = "failed", f"differs: {argmax_bad} {lacking_detail}".strip()
            else:
                argmax.status = "passed"
        top5.metrics = {"max_abs": top5_max, "worst_case": top5_worst}
        b = _bound(self.tol, "float.step0_top5_logprob_max_abs")
        if b is not None:
            if top5_max > b or lacking or non_finite:
                top5.status = "failed"
                top5.detail = f"max {top5_max:.3g} ({top5_worst}) > {b:g}" if top5_max > b else lacking_detail
            else:
                top5.status = "passed"
        full.metrics = {
            "top64_max_abs": top64_max,
            "full_vocabulary_max_abs": full_max if self.reference_tensors else None,
            "worst_case": full_worst,
        }
        b = _bound(self.tol, "float.step0_logprob_max_abs")
        if b is not None:
            if top64_max > b:
                full.status, full.detail = "failed", f"already the reference top-64 ids give {top64_max:.3g} > {b:g}"
            elif self.reference_tensors is None:
                full.detail = (
                    f"reference full step-0 vectors not supplied (--reference-tensors); the reference top-64 ids give "
                    f"{top64_max:.3g}, a lower bound of the full-vocabulary maximum"
                )
            elif full_max > b or lacking or non_finite:
                full.status = "failed"
                full.detail = f"max {full_max:.3g} ({full_worst}) > {b:g}" if full_max > b else lacking_detail
            else:
                full.status = "passed"

        # Streaming reassembly and usage.
        stream = self._key_check(
            "dotnet-ids.streaming_reassembly", "dotnet-ids", "Streaming reassembly", "discrete.streaming_reassembly"
        )
        usage = self._key_check("dotnet-ids.usage", "dotnet-ids", "Usage counts", "discrete.usage")
        stream_bad, usage_bad = [], []
        for name in present:
            row = self.run.rows[name]
            if not all(
                row.get(k) is True
                for k in ("stream_text_equals_result", "stream_ids_equal_result", "stream_text_equals_decode")
            ):
                stream_bad.append(name)
            completion = len(row["output_ids"]) - (1 if row.get("finish_reason") == "eos" else 0)
            if (
                row.get("usage_consistent") is not True
                or row.get("usage_prompt_tokens") != len(row["prompt_ids"])
                or row.get("usage_completion_tokens") != completion
            ):
                usage_bad.append(name)
        for check, failed in ((stream, stream_bad), (usage, usage_bad)):
            check.metrics = {"failed": failed}
            if check.tolerance_key in self.tol:
                if check.bound != "exact":
                    check.detail = f"rule {check.bound!r} is not implemented"
                elif failed or missing:
                    check.status, check.detail = "failed", f"failed: {failed} {miss}".strip()
                else:
                    check.status = "passed"

        # Provider options.
        prov = Check(
            "dotnet-ids.provider_options",
            "dotnet-ids",
            "Provider options equal the tolerances",
            "provider_options",
            self.raw_tol.get("provider_options"),
        )
        self._add(prov)
        env = self.run.environment
        if "provider_options" not in self.raw_tol:
            prov.detail = "provider_options is not set in the protected tolerances file"
        elif env is None or not isinstance(env.get("provider_options"), dict):
            prov.detail = "no environment.json with provider_options in the results directory"
        else:
            want = self.raw_tol.get("provider_options") or {}
            got = env["provider_options"]
            wrong_opts = {k: got.get(k) for k, v in want.items() if str(got.get(k)) != str(v)}
            prov.metrics = {"runner": got}
            prov.status = "failed" if wrong_opts else "passed"
            prov.detail = f"differs: {wrong_opts}" if wrong_opts else ""

    def _semantics(self) -> None:
        sem = self.run.semantics
        cached = self._key_check(
            "dotnet-text.cached_vs_uncached",
            "dotnet-text",
            "Cached vs uncached decoding logits",
            "float.cached_vs_uncached_logprob_max_abs",
        )
        chunked = self._key_check(
            "dotnet-text.chunked_prefill",
            "dotnet-text",
            "Chunked vs one-call prefill, step 0",
            "float.chunked_vs_one_call_prefill_logprob_max_abs",
        )
        conc = self._key_check(
            "dotnet-text.concurrent_requests",
            "dotnet-text",
            "Concurrent and interleaved requests equal the reference",
            "discrete.concurrent_requests",
        )
        if sem is None:
            for c in (cached, chunked, conc):
                c.detail = (c.detail + "; " if c.detail else "") + "no semantics.json (runner --modes semantics)"
            return
        b = _bound(self.tol, "float.cached_vs_uncached_logprob_max_abs")
        part = sem.get("cached_vs_uncached")
        if isinstance(part, dict) and isinstance(part.get("max_abs_logprob"), int | float):
            value = float(part["max_abs_logprob"])
            cached.metrics = {
                "max_abs": value,
                "steps": part.get("steps"),
                "generated_equals_reference_prefix": part.get("generated_equals_reference_prefix"),
            }
            if b is not None:
                ok = value <= b and part.get("generated_equals_reference_prefix") is True and math.isfinite(value)
                cached.status = "passed" if ok else "failed"
                cached.detail = (
                    ""
                    if ok
                    else f"max {value:.3g} vs {b:g}, prefix equal {part.get('generated_equals_reference_prefix')}"
                )
        elif b is not None:
            cached.detail = "semantics.json has no cached_vs_uncached.max_abs_logprob"

        b = _bound(self.tol, "float.chunked_vs_one_call_prefill_logprob_max_abs")
        part = sem.get("chunked_prefill")
        if isinstance(part, dict):
            measured = {
                k: v
                for k, v in part.items()
                if isinstance(v, dict) and isinstance(v.get("step0_max_abs_logprob_vs_whole"), int | float)
            }
            unmeasured = {k: (v or {}).get("whole_prefill_error") for k, v in part.items() if k not in measured}
            worst = max((float(v["step0_max_abs_logprob_vs_whole"]) for v in measured.values()), default=None)
            tokens_bad = [k for k, v in measured.items() if v.get("tokens_equal_whole") is False]
            chunked.metrics = {"max_abs": worst, "measured": sorted(measured), "not_measured": unmeasured}
            if b is not None:
                if not measured:
                    chunked.detail = f"no one-call prefill measured: {unmeasured}"
                elif worst > b or tokens_bad:
                    chunked.status, chunked.detail = "failed", f"max {worst:.3g} vs {b:g}; tokens differ {tokens_bad}"
                else:
                    chunked.status = "passed"
                    if unmeasured:
                        chunked.detail = f"not measured (one-call prefill failed): {sorted(unmeasured)}"
        elif b is not None:
            chunked.detail = "semantics.json has no chunked_prefill"

        part = sem.get("concurrency")
        if isinstance(part, dict):
            flags = {}
            for group in ("parallel_two_threads_prefix_equal", "interleaved_streams_prefix_equal"):
                for k, v in (part.get(group) or {}).items():
                    flags[f"{group}/{k}"] = v
            conc.metrics = {"results": flags}
            if "discrete.concurrent_requests" in self.tol:
                if conc.bound != "equal_to_sequential":
                    conc.detail = f"rule {conc.bound!r} is not implemented"
                elif not flags:
                    conc.detail = "semantics.json concurrency has no results"
                else:
                    bad = [k for k, v in flags.items() if v is not True]
                    conc.status = "failed" if bad else "passed"
                    conc.detail = f"differs: {bad}" if bad else ""
        elif "discrete.concurrent_requests" in self.tol:
            conc.detail = "semantics.json has no concurrency section"


def diagnose(checks: list[Check]) -> tuple[str | None, list[dict[str, str]]]:
    status = {s: substage_status(checks, s) for s in SUBSTAGES}
    first = next((s for s in SUBSTAGES if status[s] == "failed"), None)
    findings = []
    for c in checks:
        if c.status == "invalid":
            findings.append({"stage": c.substage, "taxonomy": "packaging/harness", "finding": f"{c.id}: {c.detail}"})
        elif c.status == "failed":
            taxonomy = {
                "tokenizer": "template or tokenizer",
                "dotnet-ids": "numerics" if (c.tolerance_key or "").startswith("float.") else ".NET decoding",
                "dotnet-text": "runtime semantics",
            }.get(c.substage, "other")
            findings.append({"stage": c.substage, "taxonomy": taxonomy, "finding": f"{c.id}: {c.detail}"})
    return first, findings


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    if value is None:
        return ""
    return str(value)


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Fidelity report: {report['profile']} ({report['task']})",
        "",
        f"Date {report['date']}. Evaluator {report['evaluator']['name']} {report['evaluator']['version']} at "
        f"`{report['evaluator']['repository_commit']}`. Exit code {report['exit_code']}.",
        "",
        f"Protected evidence: **{report['protected_evidence']['status']}** ({report['protected_evidence']['detail']}).",
        "",
        "| Gate | Status | Reason |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {g} | {v['status']} | {v['reason']} |" for g, v in report["gates"].items()]
    lines += ["", f"First divergence: **{report['first_divergence'] or 'none'}**.", ""]
    if report["findings"]:
        lines += ["| Stage | Taxonomy | Finding |", "| --- | --- | --- |"]
        lines += [f"| {f['stage']} | {f['taxonomy']} | {f['finding']} |" for f in report["findings"]]
        lines.append("")
    for substage in SUBSTAGES:
        mine = [c for c in report["checks"] if c["substage"] == substage]
        lines += [f"## {SUBSTAGE_TITLES[substage]}: {report['stages'][substage]}", ""]
        if not mine:
            lines += ["No checks.", ""]
            continue
        lines += ["| Check | Key | Bound | Status | Metrics | Detail |", "| --- | --- | --- | --- | --- | --- |"]
        for c in mine:
            metrics = ", ".join(
                f"{k} {_fmt(v)}" for k, v in c["metrics"].items() if not isinstance(v, list | dict) or not v
            )
            lines.append(
                f"| {c['id']} | {c['tolerance_key'] or ''} | {_fmt(c['bound'])} | {c['status']} | {metrics} | "
                f"{c['detail'].replace('|', '/')} |"
            )
        lines.append("")
    if report.get("greedy_divergences"):
        lines += ["## Greedy divergences", "", "| Case | Step | Reference margin |", "| --- | ---: | ---: |"]
        lines += [f"| {d['id']} | {d['step']} | {_fmt(d['reference_margin'])} |" for d in report["greedy_divergences"]]
        lines.append("")
    if report.get("excluded_by_contract"):
        lines += ["## Excluded by contract", ""]
        lines += [f"- `{k}`: {v}" for k, v in report["excluded_by_contract"].items()]
        lines.append("")
    return "\n".join(lines)
