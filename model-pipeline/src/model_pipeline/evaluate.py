"""Staged fidelity evaluator (plan sections 8 and 10, backlog P-007).

Compares candidate outputs, written by runners in the stage results format (`results.py`),
against a profile's protected fixtures, using only the profile's protected tolerances:

    stage 1  tokenizer  owned tokenizer ids and masks against the protected tokenizer fixtures (G1)
    stage 2  export     Python ONNX Runtime on the exported graph against the PyTorch reference (G2)
    stage 3  dotnet     the C# implementation from fixture ids, then from text (G2)

Because stage 2 runs the same artifact as stage 3 in Python, a failure names where the outputs
first diverge: Python ONNX Runtime failing means the export or backend is at fault, upstream of
.NET; Python passing and C# failing against Python on identical ids means a .NET bug; ids path
passing and text path failing with an exact tokenizer means the orchestration between them.

The evaluator is deterministic and pure Python. It writes a JSON report and a Markdown summary
with max, mean and p99 absolute errors and pass/fail per tolerance key, and gate statuses in the
manifest's vocabulary. It never edits manifests, fixtures or tolerances.

Implemented for the `embedding` task (encoder profiles, this module), the `entity-extraction`
task (manifest task `extraction`, GLiNER2 span models: `evaluate_extraction.py`), the image and video
embedding tasks, and the `generation` task (Qwen3.5: `evaluate_generation.py`). Other tasks are
rejected explicitly. A profile without a protected tolerances file is measured but never qualified:
every bound-dependent check is not-run, no gate can pass, and the exit code is 4.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from model_pipeline import fixtures, manifest, protection, results

REPORT_FORMAT = "starlings.fidelity-report"
REPORT_FORMAT_VERSION = 1

# Order in which stages are walked to find the first divergence.
SUBSTAGES = ("tokenizer", "export", "dotnet-ids", "dotnet-text")
SUBSTAGE_TITLES = {
    "tokenizer": "Stage 1, tokenizer (owned tokenizer vs protected tokenizer fixtures)",
    "export": "Stage 2, export (Python ONNX Runtime vs PyTorch reference)",
    "dotnet-ids": "Stage 3a, .NET from fixture ids",
    "dotnet-text": "Stage 3b, .NET from text (owned tokenizer + encoder)",
}
GATE_OF_SUBSTAGE = {"tokenizer": "G1", "export": "G2", "dotnet-ids": "G2", "dotnet-text": "G2"}
STAGE_OF_SUBSTAGE = {"tokenizer": "tokenizer", "export": "export", "dotnet-ids": "dotnet", "dotnet-text": "dotnet"}

# Tolerance keys this evaluator implements. Any other key is reported as not evaluated, and the
# gate it belongs to cannot pass: a bound nobody checks is not silently satisfied.
KNOWN_KEYS = {
    "profile",
    "reference",
    "provider_options",
    "discrete.token_ids",
    "discrete.attention_mask",
    "discrete.offsets",
    "float.last_hidden_state_max_abs",
    "float.token_embeddings_relative",
    "float.embedding_max_abs",
    "float.min_cosine",
    "float.padded_batch_vs_single_embedding_max_abs",
    "float.nan_or_inf",
    "float.report_only",
    "csharp_vs_python_same_machine_and_provider.embedding_max_abs",
}
GATE_OF_KEY_PREFIX = {"discrete": "G1"}

# Case id of an embedding fixture row computed with a named prompt: "<id>@<prompt>" (rows without a prompt, or with
# prompt "none", keep their id).
PROMPT_SEPARATOR = "@"

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INPUT = 2
EXIT_PROTECTED = 3
EXIT_GATE_NOT_PASSED = 4


class EvaluationError(Exception):
    """Inputs cannot be evaluated (malformed manifest, fixtures or results)."""


# --------------------------------------------------------------------------------------------
# Statistics


@dataclass
class ErrorStats:
    """Absolute-error summary over many values, as the M0 reports compute it (p99 linear, like numpy)."""

    values: list[float] = field(default_factory=list)
    per_case_max: dict[str, float] = field(default_factory=dict)
    cosines: dict[str, float] = field(default_factory=dict)
    non_finite: list[str] = field(default_factory=list)

    def add(self, case_id: str, actual: list[float], expected: list[float], cosine: bool) -> None:
        if len(actual) != len(expected):
            raise ValueError(f"{case_id}: {len(actual)} values, expected {len(expected)}")
        finite = all(math.isfinite(a) for a in actual)
        if not finite:
            self.non_finite.append(case_id)
        diffs = [abs(a - e) if math.isfinite(a) else math.inf for a, e in zip(actual, expected, strict=True)]
        self.values.extend(diffs)
        self.per_case_max[case_id] = max(diffs, default=0.0)
        if cosine and finite:
            dot = sum(a * e for a, e in zip(actual, expected, strict=True))
            na = math.sqrt(sum(a * a for a in actual))
            ne = math.sqrt(sum(e * e for e in expected))
            self.cosines[case_id] = dot / (na * ne) if na and ne else 0.0

    def summary(self) -> dict[str, Any]:
        a = sorted(self.values)
        n = len(a)
        if n == 0:
            return {"values": 0}
        pos = (n - 1) * 0.99
        lo, hi = math.floor(pos), math.ceil(pos)
        p99 = a[lo] + (a[hi] - a[lo]) * (pos - lo) if math.isfinite(a[hi]) else math.inf
        worst = max(self.per_case_max, key=lambda k: self.per_case_max[k]) if self.per_case_max else None
        out: dict[str, Any] = {
            "max_abs": a[-1],
            "mean_abs": math.fsum(a) / n,
            "p99_abs": p99,
            "values": n,
            "cases": len(self.per_case_max),
            "worst_case": worst,
            "non_finite_cases": self.non_finite,
        }
        if self.cosines:
            out["min_cosine"] = min(self.cosines.values())
            out["min_cosine_case"] = min(self.cosines, key=lambda k: self.cosines[k])
        return out


# --------------------------------------------------------------------------------------------
# Inputs


@dataclass
class Fixtures:
    """Protected expectations for an encoder profile."""

    ids: dict[str, list[int]]
    attention_mask: dict[str, list[int]]
    embedding: dict[str, list[float]]
    files: list[Path]

    @property
    def case_ids(self) -> list[str]:
        return list(self.embedding)


def load_fixtures(directory: Path) -> Fixtures:
    tok_path = directory / "tokenizer.jsonl"
    emb_path = directory / "embeddings.jsonl"
    for p in (tok_path, emb_path):
        if not p.is_file():
            raise EvaluationError(f"protected fixture {p} is missing")
    ids, masks = {}, {}
    for row in fixtures.read_jsonl(tok_path):
        if "pair" in row:
            continue  # pair encoding is not an embedding input for this task (manifest unsupported list)
        enc = row.get("truncated", row)["encoding"]
        ids[row["id"]] = enc["ids"]
        masks[row["id"]] = enc["attention_mask"]
    embedding = {}
    prompted: dict[str, list[int]] = {}
    for row in fixtures.read_jsonl(emb_path):
        if row["id"] not in ids:
            raise EvaluationError(f"{emb_path}: case {row['id']!r} has no single-text tokenizer fixture")
        prompt = row.get("prompt", "none")
        if prompt == "none":
            if row["input_ids"] != ids[row["id"]]:
                raise EvaluationError(f"{emb_path}: case {row['id']!r} input_ids differ from tokenizer.jsonl")
            cid = row["id"]
        else:
            # A row computed with a named prompt (EmbeddingGemma 2: prompt_text + text tokenized as one string) is its
            # own case "<id>@<prompt>"; its ids come from the row, which the reference runner tokenized.
            cid = f"{row['id']}{PROMPT_SEPARATOR}{prompt}"
            prompted[cid] = row["input_ids"]
        if cid in embedding:
            raise EvaluationError(f"{emb_path}: case {cid!r} appears twice")
        embedding[cid] = row["embedding"]
    for cid, row_ids in prompted.items():
        ids[cid] = row_ids
        masks[cid] = [1] * len(row_ids)
    missing = sorted(set(ids) - set(embedding))
    if missing:
        raise EvaluationError(f"{emb_path}: no embedding for single-text cases {missing}")
    return Fixtures(ids=ids, attention_mask=masks, embedding=embedding, files=[tok_path, emb_path])


def load_tolerances(path: Path, profile_id: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise EvaluationError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise EvaluationError(f"{path}: expected a mapping")
    if data.get("profile") != profile_id:
        raise protection.ProtectedEvidenceError(
            f"{path} declares profile {data.get('profile')!r}, not {profile_id!r}; a profile is evaluated only "
            "against its own protected tolerances"
        )
    return data


def flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and key != "provider_options":
            out |= flatten(value, f"{name}.")
        else:
            out[name] = value
    return out


# --------------------------------------------------------------------------------------------
# Checks


@dataclass
class Check:
    id: str
    substage: str
    title: str
    tolerance_key: str | None = None
    bound: Any = None
    status: str = "not-run"  # passed, failed, not-run, not-applicable, invalid
    metrics: dict[str, Any] = field(default_factory=dict)
    detail: str = ""

    @property
    def gate(self) -> str:
        return GATE_OF_SUBSTAGE[self.substage]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "stage": STAGE_OF_SUBSTAGE[self.substage],
            "substage": self.substage,
            "gate": self.gate,
            "title": self.title,
            "tolerance_key": self.tolerance_key,
            "bound": self.bound,
            "status": self.status,
            "metrics": self.metrics,
            "detail": self.detail,
        }


def _bound(tol: dict[str, Any], key: str) -> float | None:
    value = tol.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


class EncoderEvaluator:
    """Checks for an `embedding` profile. One instance per evaluation."""

    def __init__(self, man: dict[str, Any], fx: Fixtures, tol: dict[str, Any], stages: dict[str, results.StageResults]):
        self.man = man
        self.fx = fx
        self.tol = flatten(tol)
        self.stages = stages
        self.checks: list[Check] = []

    # -- helpers ------------------------------------------------------------------------------

    def _add(self, check: Check) -> Check:
        self.checks.append(check)
        return check

    def _missing_stage(self, check: Check, stage: str) -> Check:
        check.status = "not-run"
        check.detail = f"no results for stage {stage!r}"
        return self._add(check)

    def _case_set_problem(self, have: set[str], what: str) -> str | None:
        expected = set(self.fx.case_ids)
        missing, extra = sorted(expected - have), sorted(have - expected)
        parts = []
        if missing:
            parts.append(f"{what}: missing cases {missing} (every protected case is required)")
        if extra:
            parts.append(f"{what}: cases not in the protected fixtures {extra}")
        return "; ".join(parts) or None

    def _compare_vectors(
        self,
        check: Check,
        actual: dict[str, list[float]],
        expected: dict[str, list[float]],
        bound: float | None,
        cosine: bool = True,
        expected_stage: str | None = None,
    ) -> Check:
        """Compare per-case vectors. Both case sets are validated before any case is indexed.

        `expected_stage` names the stage that produced `expected` when it is another stage's results
        (for example Python ONNX Runtime for the C# same-machine bound). Cases missing there are that
        stage's failure, reported by its own checks; this comparison is then incomplete (not-run).
        """
        problem = self._case_set_problem(set(actual), check.id)
        if problem:
            check.status, check.detail = "failed", problem
            return self._add(check)
        missing_expected = sorted(set(self.fx.case_ids) - set(expected))
        if missing_expected:
            if expected_stage:
                check.status = "not-run"
                check.detail = (
                    f"the {expected_stage} results lack cases {missing_expected}; that stage's checks fail, "
                    "and this comparison cannot be completed"
                )
            else:
                check.status = "failed"
                check.detail = f"reference lacks cases {missing_expected}"
            return self._add(check)
        stats = ErrorStats()
        try:
            for case_id in self.fx.case_ids:
                stats.add(case_id, actual[case_id], expected[case_id], cosine)
        except ValueError as exc:
            check.status, check.detail = "failed", f"shape mismatch: {exc}"
            return self._add(check)
        check.metrics = stats.summary()
        return self._judge(check, bound)

    def _judge(self, check: Check, bound: float | None) -> Check:
        check.bound = bound
        if check.metrics.get("non_finite_cases"):
            check.status = "failed"
            check.detail = f"NaN or Inf in cases {check.metrics['non_finite_cases']}"
        elif bound is None:
            check.status = "not-run"
            check.detail = f"no numeric bound for {check.tolerance_key} in the protected tolerances"
        else:
            check.status = "passed" if check.metrics["max_abs"] <= bound else "failed"
            if check.status == "failed":
                check.detail = (
                    f"max abs {check.metrics['max_abs']:.3g} > {bound:.3g} (worst case {check.metrics['worst_case']})"
                )
        return self._add(check)

    def _case_output(self, stage: results.StageResults, output: str) -> dict[str, list]:
        return {cid: outs[output] for cid, outs in stage.cases.items() if output in outs}

    # -- stage 0: identity of what each stage ran ----------------------------------------------

    def check_identity(self) -> None:
        manifest_files = {f["path"]: f["sha256"] for f in self.man["origin"]["files"]}
        required = {"tokenizer": ["tokenizer.json"], "export": [], "dotnet": []}
        graph = [f["path"] for f in self.man["origin"]["files"] if f["path"].endswith(".onnx")]
        required["export"] = graph[:1]
        required["dotnet"] = graph[:1] + (["tokenizer.json"] if self._has_text_path(self.stages.get("dotnet")) else [])
        for name, stage in self.stages.items():
            substage = {"tokenizer": "tokenizer", "export": "export", "dotnet": "dotnet-ids"}[name]
            check = Check(f"{name}.identity", substage, f"{name}: profile and artifact digests match the manifest")
            problems = []
            if stage.profile != self.man["id"]:
                problems.append(f"results are for profile {stage.profile!r}, not {self.man['id']!r}")
            declared = {a["path"]: a["sha256"] for a in stage.assets}
            for path, sha in declared.items():
                if path not in manifest_files:
                    problems.append(f"asset {path} is not in the manifest's origin.files")
                elif manifest_files[path] != sha:
                    problems.append(f"asset {path} sha256 {sha} differs from the manifest ({manifest_files[path]})")
            for path in required[name]:
                if path not in declared:
                    problems.append(f"asset {path} not declared; the stage must name the artifact it ran")
            check.status = "invalid" if problems else "passed"
            check.detail = "; ".join(problems)
            self._add(check)

    @staticmethod
    def _has_text_path(stage: results.StageResults | None) -> bool:
        """True when a .NET stage ran the text path (any text_embedding output); then it must name tokenizer.json."""
        if stage is None:
            return False
        return any("text_embedding" in outs for outs in stage.cases.values()) or any(
            b["output"] == "text_embedding" for b in stage.batches.values()
        )

    # -- stage 1: tokenizer ----------------------------------------------------------------------

    def check_tokenizer(self) -> None:
        stage = self.stages.get("tokenizer")
        for output, key, expected in (
            ("input_ids", "discrete.token_ids", self.fx.ids),
            ("attention_mask", "discrete.attention_mask", self.fx.attention_mask),
        ):
            check = Check(f"tokenizer.{output}", "tokenizer", f"tokenizer {output} exact", key, self.tol.get(key))
            if key not in self.tol:
                check.detail = f"{key} is not in the protected tolerances"
                self._add(check)
                continue
            if self.tol[key] != "exact":
                check.detail = f"{key}: {self.tol[key]!r} has no implemented meaning (only 'exact')"
                self._add(check)
                continue
            if stage is None:
                self._missing_stage(check, "tokenizer")
                continue
            actual = self._case_output(stage, output)
            problem = self._case_set_problem(set(actual), check.id)
            mismatched = [cid for cid in self.fx.case_ids if cid in actual and actual[cid] != expected[cid]]
            check.metrics = {"cases": len(actual), "mismatched_cases": mismatched}
            if mismatched:
                first = mismatched[0]
                a, e = actual[first], expected[first]
                pos = next((i for i, (x, y) in enumerate(zip(a, e, strict=False)) if x != y), min(len(a), len(e)))
                check.metrics["first_mismatch"] = {
                    "case": first,
                    "position": pos,
                    "actual": a[max(0, pos - 2) : pos + 3],
                    "expected": e[max(0, pos - 2) : pos + 3],
                    "actual_length": len(a),
                    "expected_length": len(e),
                }
            check.status = "failed" if problem or mismatched else "passed"
            check.detail = problem or (f"{len(mismatched)} case(s) differ: {mismatched}" if mismatched else "")
            self._add(check)

        key = "discrete.offsets"
        if key in self.tol:
            check = Check("tokenizer.offsets", "tokenizer", "tokenizer offsets exact", key, self.tol[key])
            if self.man["behavior"]["offsets_unit"] == "none":
                check.status = "not-applicable"
                check.detail = "manifest behavior.offsets_unit is 'none': the embedding task exposes no offsets"
            else:
                check.detail = "offset comparison is not implemented in this evaluator"
            self._add(check)

    # -- shared numeric checks for runtime stages ------------------------------------------------

    def _input_integrity(self, stage: results.StageResults, substage: str) -> bool:
        check = Check(f"{stage.stage}.input_ids", substage, f"{stage.stage}: ran on the protected fixture ids")
        # Missing cases are failed by each output check; here only provenance of the inputs matters.
        actual = self._case_output(stage, "input_ids")
        unrecorded = sorted(set(stage.cases) - set(actual))
        unknown = sorted(set(actual) - set(self.fx.case_ids))
        wrong = [cid for cid in self.fx.case_ids if cid in actual and actual[cid] != self.fx.ids[cid]]
        if unrecorded or unknown or wrong:
            check.status = "invalid"
            check.detail = (
                f"cases without recorded input_ids {unrecorded}, unknown cases {unknown}, ids differing from the "
                f"fixtures {wrong}; the comparison would be invalid"
            )
        else:
            check.status = "passed"
        self._add(check)
        return check.status == "passed"

    def _non_finite(self, stage: results.StageResults, substage: str) -> None:
        key = "float.nan_or_inf"
        check = Check(f"{stage.stage}.finite", substage, f"{stage.stage}: no NaN or Inf in any output", key)
        check.bound = self.tol.get(key)
        if check.bound != "fail":
            check.detail = f"{key}: {check.bound!r} has no implemented meaning (only 'fail')"
            self._add(check)
            return
        bad = []
        for cid, outs in stage.cases.items():
            for name, values in outs.items():
                if any(not math.isfinite(v) for v in values):
                    bad.append(f"{cid}/{name}")
        for bname, batch in stage.batches.items():
            for cid, row in zip(batch["case_ids"], batch["values"], strict=True):
                if any(not math.isfinite(v) for v in row):
                    bad.append(f"batch {bname}/{cid}")
        for tname in stage.tensor_specs:
            for cid, values in stage.tensor(tname).values.items():
                if any(not math.isfinite(v) for v in values):
                    bad.append(f"tensor {tname}/{cid}")
        check.metrics = {"non_finite": bad}
        check.status = "failed" if bad else "passed"
        check.detail = f"NaN or Inf in {bad}" if bad else ""
        self._add(check)

    def _batches(self, stage: results.StageResults, substage: str, outputs: tuple[str, ...]) -> None:
        emb_key = "float.embedding_max_abs"
        pad_key = "float.padded_batch_vs_single_embedding_max_abs"
        relevant = {n: b for n, b in stage.batches.items() if b["output"] in outputs}
        # The padded-batch bound is binding for every output path the gate covers, so each path needs its own
        # batch that actually pads: more than one row, with input lengths that differ.
        for output in outputs:
            sub = "dotnet-text" if output == "text_embedding" else substage
            evidence = Check(
                f"{stage.stage}.batch_evidence.{output}",
                sub,
                f"{stage.stage}: a padded batch exercises '{output}'",
                pad_key,
                _bound(self.tol, pad_key),
            )
            mine = {n: b for n, b in relevant.items() if b["output"] == output}
            padded = sorted(n for n, b in mine.items() if self._pads(b["case_ids"]))
            evidence.metrics = {"batches": sorted(mine), "padding_batches": padded}
            if padded:
                evidence.status = "passed"
            else:
                evidence.status = "failed"
                evidence.detail = (
                    f"no batch for output '{output}'"
                    if not mine
                    else f"batches {sorted(mine)} never pad (each needs more than one row with differing lengths)"
                ) + "; the padded-batch bound is binding and needs evidence"
            self._add(evidence)
        for name, batch in relevant.items():
            singles = self._case_output(stage, batch["output"])
            sub = substage if batch["output"] != "text_embedding" else "dotnet-text"
            vs_single = Check(
                f"{stage.stage}.batch.{name}.vs_single",
                sub,
                f"{stage.stage}: batch '{name}' vs own single rows",
                pad_key,
            )
            vs_ref = Check(
                f"{stage.stage}.batch.{name}.vs_reference", sub, f"{stage.stage}: batch '{name}' vs reference", emb_key
            )
            unknown = sorted(set(batch["case_ids"]) - set(self.fx.case_ids))
            missing_single = sorted(set(batch["case_ids"]) - set(singles))
            if unknown or missing_single:
                for c in (vs_single, vs_ref):
                    c.status = "failed"
                    c.detail = f"batch rows without fixture case {unknown} or single output {missing_single}"
                    self._add(c)
                continue
            s1, s2 = ErrorStats(), ErrorStats()
            try:
                for cid, row in zip(batch["case_ids"], batch["values"], strict=True):
                    s1.add(cid, row, singles[cid], True)
                    s2.add(cid, row, self.fx.embedding[cid], True)
            except ValueError as exc:
                for c in (vs_single, vs_ref):
                    c.status, c.detail = "failed", f"shape mismatch: {exc}"
                    self._add(c)
                continue
            vs_single.metrics, vs_ref.metrics = s1.summary(), s2.summary()
            self._judge(vs_single, _bound(self.tol, pad_key))
            self._min_cosine(self._judge(vs_ref, _bound(self.tol, emb_key)))

    def _pads(self, case_ids: list[str]) -> bool:
        lengths = {len(self.fx.ids[c]) for c in case_ids if c in self.fx.ids}
        return len(case_ids) > 1 and len(lengths) > 1

    def _provider_options(self, stage: results.StageResults, substage: str) -> None:
        if "provider_options" not in self.tol:
            return
        want = self.tol["provider_options"]
        have = stage.environment.get("provider_options", {})
        check = Check(f"{stage.stage}.provider_options", substage, f"{stage.stage}: provider options of the profile")
        check.bound = want
        diff = {k: have.get(k) for k, v in want.items() if str(have.get(k)) != str(v)}
        check.status = "failed" if diff else "passed"
        check.detail = f"stage ran with {diff}, the profile requires {want}" if diff else ""
        check.metrics = {"declared": have}
        self._add(check)

    # -- stage 2: export ---------------------------------------------------------------------------

    def check_export(self) -> None:
        stage = self.stages.get("export")
        emb_key, hid_key = "float.embedding_max_abs", "float.last_hidden_state_max_abs"
        hidden = Check(
            "export.last_hidden_state", "export", "Python ORT hidden states vs PyTorch hidden states", hid_key
        )
        emb = Check("export.embedding", "export", "Python ORT embeddings vs protected reference embeddings", emb_key)
        if stage is None:
            for c in (hidden, emb):
                self._missing_stage(c, "export")
            self._missing_stage(Check("export.finite", "export", "export: no NaN or Inf", "float.nan_or_inf"), "export")
            return
        if not self._input_integrity(stage, "export"):
            return
        self._provider_options(stage, "export")
        self._non_finite(stage, "export")
        if {"last_hidden_state", "reference_last_hidden_state"} <= set(stage.tensor_specs):
            cand, ref = stage.tensor("last_hidden_state"), stage.tensor("reference_last_hidden_state")
            if cand.shapes != ref.shapes:
                hidden.status, hidden.detail = "failed", "candidate and reference hidden-state shapes differ"
                self._add(hidden)
            else:
                self._compare_vectors(hidden, cand.values, ref.values, _bound(self.tol, hid_key), cosine=False)
        else:
            hidden.bound = _bound(self.tol, hid_key)
            hidden.detail = "export results need tensors 'last_hidden_state' and 'reference_last_hidden_state'"
            self._add(hidden)
        if self._report_only(hid_key) and hidden.status != "failed":
            # The protected tolerances deliberately bind no hidden-state key (EmbeddingGemma 2: the 512-wide state is
            # diagnostic only): measured when present, never gating.
            hidden.status = "not-applicable"
            hidden.detail = f"report only: {hid_key} is not in the protected tolerances" + (
                f" ({hidden.detail})" if hidden.detail and not hidden.metrics else ""
            )
        self._token_embeddings(stage)
        self._min_cosine(
            self._compare_vectors(
                emb, self._case_output(stage, "embedding"), self.fx.embedding, _bound(self.tol, emb_key)
            )
        )
        self._batches(stage, "export", ("embedding",))

    def _report_only(self, key: str) -> bool:
        """True when the profile has protected tolerances that do not bind `key` (a measurement, not a gate)."""
        return bool(self.tol) and key not in self.tol

    def _min_cosine(self, base: Check) -> None:
        """Minimum cosine (float64) of a comparison against the reference, when the tolerances bind float.min_cosine."""
        key = "float.min_cosine"
        if key not in self.tol:
            return
        check = Check(
            f"{base.id}.min_cosine", base.substage, f"{base.title}: minimum cosine", key, _bound(self.tol, key)
        )
        cosine = base.metrics.get("min_cosine")
        if check.bound is None:
            check.detail = f"no numeric bound for {key} in the protected tolerances"
        elif cosine is None or base.metrics.get("non_finite_cases"):
            check.detail = f"{base.id} did not produce a complete comparison ({base.status})"
        else:
            check.metrics = {"min_cosine": cosine, "cases": base.metrics.get("cases")}
            worst = base.metrics.get("min_cosine_case")
            check.metrics["worst_case"] = worst
            check.status = "passed" if cosine >= check.bound else "failed"
            if check.status == "failed":
                check.detail = f"min cosine {cosine:.15g} < {check.bound:.15g} (worst case {worst})"
        self._add(check)

    def _token_embeddings(self, stage: results.StageResults) -> None:
        """Token embeddings vs PyTorch over valid positions, judged per case (token_embeddings_relative).

        For each case, max_i |d_i| / max(1, max_i |ref_i|) <= r: the largest absolute error relative to the case's
        largest reference magnitude (maintainer decision, 2026-10-11). The per-value reading, max_i |d_i| / max(1,
        |ref_i|), is reported for transparency and never binds.
        """
        key = "float.token_embeddings_relative"
        if key not in self.tol:
            return
        check = Check(
            "export.token_embeddings",
            "export",
            "Python ORT token embeddings vs PyTorch, per case max |d| <= r * max(1, max |ref|)",
            key,
            _bound(self.tol, key),
        )
        if not {"token_embeddings", "reference_token_embeddings"} <= set(stage.tensor_specs):
            check.detail = "export results need tensors 'token_embeddings' and 'reference_token_embeddings'"
            self._add(check)
            return
        cand, ref = stage.tensor("token_embeddings"), stage.tensor("reference_token_embeddings")
        problem = self._case_set_problem(set(cand.values), check.id) or self._case_set_problem(
            set(ref.values), f"{check.id} reference"
        )
        if problem:
            check.status, check.detail = "failed", problem
            self._add(check)
            return
        if cand.shapes != ref.shapes:
            check.status, check.detail = "failed", "candidate and reference token-embedding shapes differ"
            self._add(check)
            return
        worst_case, max_rel, max_abs, non_finite, count = None, 0.0, 0.0, [], 0
        value_worst_case, value_rel = None, 0.0
        for cid in self.fx.case_ids:
            case_value_rel, case_abs, case_ref_max, finite = 0.0, 0.0, 0.0, True
            for a, e in zip(cand.values[cid], ref.values[cid], strict=True):
                count += 1
                if not math.isfinite(a):
                    non_finite.append(cid)
                    finite, case_value_rel = False, math.inf
                    break
                d = abs(a - e)
                case_abs = max(case_abs, d)
                case_ref_max = max(case_ref_max, abs(e))
                case_value_rel = max(case_value_rel, d / max(1.0, abs(e)))
            max_abs = max(max_abs, case_abs)
            case_rel = case_abs / max(1.0, case_ref_max) if finite else math.inf
            if worst_case is None or case_rel > max_rel:
                worst_case, max_rel = cid, case_rel
            if value_worst_case is None or case_value_rel > value_rel:
                value_worst_case, value_rel = cid, case_value_rel
        check.metrics = {
            # Binding: the worst case's max |d| / max(1, max |ref|).
            "max_relative": max_rel,
            "worst_case": worst_case,
            "max_abs": max_abs,
            "values": count,
            "cases": len(self.fx.case_ids),
            "non_finite_cases": non_finite,
            # Report only: the per-value reading, max |d| / max(1, |ref|) over every value. Never binds.
            "per_value_max_relative": value_rel,
            "per_value_worst_case": value_worst_case,
        }
        if non_finite:
            check.status, check.detail = "failed", f"NaN or Inf in cases {non_finite}"
        elif check.bound is None:
            check.detail = f"no numeric bound for {key} in the protected tolerances"
        else:
            check.status = "passed" if max_rel <= check.bound else "failed"
            relation = "<=" if check.status == "passed" else ">"
            check.detail = (
                f"per case max |d| / max(1, max |ref|) {max_rel:.3g} {relation} {check.bound:.3g} "
                f"(worst case {worst_case}); report only, per value |d| / max(1, |ref|) {value_rel:.3g} "
                f"(worst case {value_worst_case})"
            )
        self._add(check)

    # -- stage 3: dotnet ----------------------------------------------------------------------------

    def check_dotnet(self) -> None:
        stage = self.stages.get("dotnet")
        export = self.stages.get("export")
        emb_key = "float.embedding_max_abs"
        cs_key = "csharp_vs_python_same_machine_and_provider.embedding_max_abs"
        ids_ref = Check("dotnet.embedding", "dotnet-ids", "C# embeddings from ids vs protected reference", emb_key)
        ids_py = Check("dotnet.embedding.vs_python", "dotnet-ids", "C# vs Python ORT embeddings, same ids", cs_key)
        text = Check("dotnet.text_embedding", "dotnet-text", "C# embeddings from text vs protected reference", emb_key)
        if stage is None:
            for c in (ids_ref, ids_py, text):
                self._missing_stage(c, "dotnet")
            return
        if not self._input_integrity(stage, "dotnet-ids"):
            return
        self._provider_options(stage, "dotnet-ids")
        self._non_finite(stage, "dotnet-ids")
        cs_emb = self._case_output(stage, "embedding")
        self._min_cosine(self._compare_vectors(ids_ref, cs_emb, self.fx.embedding, _bound(self.tol, emb_key)))

        if cs_key not in self.tol:
            ids_py.detail = f"{cs_key} is not in the protected tolerances"
            self._add(ids_py)
        elif export is None:
            self._missing_stage(ids_py, "export")
        elif not self._same_machine(stage, export):
            ids_py.bound = _bound(self.tol, cs_key)
            ids_py.detail = (
                "the bound applies to the same machine and provider; the stages declare "
                f"{stage.environment.get('machine')!r}/{stage.environment.get('provider')!r} and "
                f"{export.environment.get('machine')!r}/{export.environment.get('provider')!r}"
            )
            self._add(ids_py)
        else:
            self._compare_vectors(
                ids_py,
                cs_emb,
                self._case_output(export, "embedding"),
                _bound(self.tol, cs_key),
                expected_stage="export",
            )

        # Hidden states from C#, if dumped, against Python ORT: report-only (no protected key).
        if (
            export is not None
            and "last_hidden_state" in stage.tensor_specs
            and "last_hidden_state" in export.tensor_specs
        ):
            cs_h, py_h = stage.tensor("last_hidden_state"), export.tensor("last_hidden_state")
            info = Check(
                "dotnet.last_hidden_state.vs_python", "dotnet-ids", "C# vs Python ORT hidden states (report only)"
            )
            if cs_h.shapes == py_h.shapes and set(cs_h.values) == set(self.fx.case_ids):
                stats = ErrorStats()
                for cid in self.fx.case_ids:
                    stats.add(cid, cs_h.values[cid], py_h.values[cid], False)
                info.metrics = stats.summary()
                info.status = "not-applicable"
                info.detail = "report only: no protected bound for C# hidden states"
                info.metrics["bit_identical_cases"] = sum(cs_h.values[c] == py_h.values[c] for c in self.fx.case_ids)
            else:
                info.status, info.detail = "not-applicable", "report only: shapes or case sets differ"
            self._add(info)

        if self._has_text_path(stage):
            self._min_cosine(
                self._compare_vectors(
                    text, self._case_output(stage, "text_embedding"), self.fx.embedding, _bound(self.tol, emb_key)
                )
            )
            self._batches(stage, "dotnet-ids", ("embedding", "text_embedding"))
        else:
            # No text path in this stage (for example, no owned tokenizer for the profile yet): the text checks are not
            # run, so G2 cannot pass, but nothing that was not run is reported as failed.
            text.bound = _bound(self.tol, emb_key)
            text.detail = "the .NET stage ran no text path (no text_embedding outputs); text to embedding is untested"
            self._add(text)
            self._add(
                Check(
                    "dotnet.batch_evidence.text_embedding",
                    "dotnet-text",
                    "dotnet: a padded batch exercises 'text_embedding'",
                    "float.padded_batch_vs_single_embedding_max_abs",
                    _bound(self.tol, "float.padded_batch_vs_single_embedding_max_abs"),
                    detail="the .NET stage ran no text path",
                )
            )
            self._batches(stage, "dotnet-ids", ("embedding",))

    @staticmethod
    def _same_machine(a: results.StageResults, b: results.StageResults) -> bool:
        keys = ("machine", "provider")
        return all(a.environment.get(k) and a.environment.get(k) == b.environment.get(k) for k in keys)

    # -- coverage of the tolerance file ---------------------------------------------------------

    def check_unknown_keys(self) -> None:
        for key in sorted(self.tol):
            if key in KNOWN_KEYS:
                continue
            substage = "tokenizer" if GATE_OF_KEY_PREFIX.get(key.split(".")[0]) == "G1" else "export"
            self._add(
                Check(
                    f"tolerances.{key}",
                    substage,
                    f"protected tolerance {key}",
                    key,
                    self.tol[key],
                    "not-run",
                    detail="no check implemented for this key; the gate cannot pass until one is",
                )
            )

    def run(self) -> list[Check]:
        self.check_identity()
        self.check_tokenizer()
        self.check_export()
        self.check_dotnet()
        self.check_unknown_keys()
        return self.checks


# --------------------------------------------------------------------------------------------
# Attribution and gates


def substage_status(checks: list[Check], substage: str) -> str:
    mine = [c for c in checks if c.substage == substage]
    if any(c.status in ("failed", "invalid") for c in mine):
        return "failed"
    if any(c.status == "not-run" for c in mine) or not mine:
        return "incomplete"
    return "passed"


def _failed(checks: list[Check], check_id: str) -> bool:
    return any(c.id == check_id and c.status in ("failed", "invalid") for c in checks)


def _status(checks: list[Check], check_id: str) -> str | None:
    return next((c.status for c in checks if c.id == check_id), None)


def diagnose(checks: list[Check]) -> tuple[str | None, list[dict[str, str]]]:
    """First diverging stage and one finding per failing stage, naming the likely fault."""
    status = {s: substage_status(checks, s) for s in SUBSTAGES}
    first = next((s for s in SUBSTAGES if status[s] == "failed"), None)
    findings: list[dict[str, str]] = []

    def add(stage: str, taxonomy: str, text: str) -> None:
        findings.append({"stage": stage, "taxonomy": taxonomy, "finding": text})

    invalid = [c for c in checks if c.status == "invalid"]
    for c in invalid:
        add(c.substage, "packaging/harness", f"{c.id}: {c.detail}. Outputs of this stage are not comparable.")

    # Missing evidence (a protected case absent, or no padded batch for an output path) is its own finding;
    # the fault diagnosis below looks only at checks that compared values and found them wrong.
    for c in checks:
        if c.status == "failed" and _is_missing_evidence(c):
            add(c.substage, "missing evidence", f"{c.id}: {c.detail}.")
    real = {
        s: any(c.substage == s and c.status == "failed" and not _is_missing_evidence(c) for c in checks)
        for s in SUBSTAGES
    }

    if real["tokenizer"] and not any(c.substage == "tokenizer" for c in invalid):
        add("tokenizer", "tokenizer", "Owned tokenizer ids or masks differ from the protected tokenizer fixtures.")

    export_failed = status["export"] == "failed"
    if real["export"] and not any(c.substage == "export" for c in invalid):
        only_batch = all(
            ".batch." in c.id
            for c in checks
            if c.substage == "export" and c.status in ("failed", "invalid") and not _is_missing_evidence(c)
        )
        if only_batch:
            add(
                "export",
                "export/operator",
                "Python ONNX Runtime single rows match the reference but padded batches do not.",
            )
        else:
            add(
                "export",
                "export/operator or precision",
                "Python ONNX Runtime on the exported graph differs from the PyTorch reference: the fault is in the "
                "exported artifact or the backend, upstream of .NET.",
            )

    if real["dotnet-ids"] and not any(c.substage == "dotnet-ids" for c in invalid):
        vs_py = _status(checks, "dotnet.embedding.vs_python")
        vs_ref_failed = _failed(checks, "dotnet.embedding")
        batch_only = not vs_ref_failed and vs_py != "failed"
        if vs_py == "failed":
            add(
                "dotnet-ids",
                ".NET runtime binding, preprocessing or pooling",
                "C# differs from Python ONNX Runtime on identical ids and the same artifact: a .NET fault.",
            )
        elif batch_only:
            add("dotnet-ids", ".NET batching", "C# single rows pass but padded batches do not (padding or masking).")
        elif export_failed:
            add(
                "dotnet-ids",
                "consequence of export",
                "C# misses the reference bound, consistent with the export-stage failure (C# agrees with Python ORT).",
            )
        elif vs_py == "passed":
            add(
                "dotnet-ids",
                "precision",
                "C# is within the same-backend bound of Python ONNX Runtime but beyond the reference bound: "
                "accumulated error; inspect the export-stage margin.",
            )
        else:
            add(
                "dotnet-ids",
                "unattributed",
                "C# misses the reference bound, and without a same-machine export stage the export and .NET "
                "cannot be separated. Run stage 2 on the same machine.",
            )

    if real["dotnet-text"]:
        if real["tokenizer"]:
            add("dotnet-text", "consequence of tokenizer", "Text path fails; the tokenizer stage already fails.")
        elif real["dotnet-ids"]:
            ids_taxonomy = next((f["taxonomy"] for f in findings if f["stage"] == "dotnet-ids"), "")
            if ids_taxonomy == "consequence of export":
                add(
                    "dotnet-text",
                    "consequence of export",
                    "Text path fails like the ids path, which follows the export.",
                )
            else:
                add("dotnet-text", "consequence of .NET runtime", "Text path fails; the ids path already fails.")
        elif status["tokenizer"] != "incomplete" and status["dotnet-ids"] != "incomplete":
            add(
                "dotnet-text",
                "processor",
                "Tokenizer ids are exact and the ids path passes, but text to embedding fails: the orchestration "
                "between tokenizer and encoder (prefix, truncation, batching of text) is at fault.",
            )
        else:
            add("dotnet-text", "unattributed", "Text path fails; earlier stages are incomplete.")
    return first, findings


def _is_missing_evidence(check: Check) -> bool:
    return ".batch_evidence." in check.id or "missing cases" in check.detail


def gates(
    checks: list[Check], verification: protection.Verification, tolerances_missing: bool = False
) -> dict[str, dict[str, Any]]:
    out = {}
    for gate in ("G1", "G2"):
        mine = [c for c in checks if c.gate == gate]
        failed = [c.id for c in mine if c.status in ("failed", "invalid")]
        open_ = [c.id for c in mine if c.status == "not-run"]
        if verification.status == "changed":
            status, reason = "not-run", "protected evidence differs from the base commit; this run is not gate evidence"
        elif tolerances_missing:
            status = "failed" if failed else "not-run"
            reason = "the profile has no protected tolerances file; no bound is set, so no gate can pass" + (
                f" (failing checks: {', '.join(failed)})" if failed else ""
            )
        elif failed:
            status, reason = "failed", f"failing checks: {', '.join(failed)}"
        elif open_ or not mine:
            status, reason = "not-run", f"checks not run: {', '.join(open_) or 'none defined'}"
        elif verification.status != "verified":
            status, reason = "not-run", f"protected evidence not verified against Git ({verification.detail})"
        else:
            status, reason = "passed", "every check passed or is not applicable by the manifest"
        out[gate] = {"status": status, "reason": reason}
    return out


# --------------------------------------------------------------------------------------------
# Entry point


@dataclass
class Evaluation:
    report: dict[str, Any]
    exit_code: int


# Manifest task (behavior.tasks) -> evaluator task. Tasks without checks are rejected explicitly.
TASK_OF_MANIFEST_TASK = {
    "embedding": "embedding",
    "extraction": "entity-extraction",
    "image-embedding": "image-embedding",
    "video-embedding": "video-embedding",
    "generation": "generation",
}


def task_of(man: dict[str, Any]) -> str:
    for name in man["behavior"]["tasks"]:
        if name in TASK_OF_MANIFEST_TASK:
            return TASK_OF_MANIFEST_TASK[name]
    raise EvaluationError(f"{man['id']}: no staged evaluator for tasks {man['behavior']['tasks']} yet")


def expected_tolerance_keys(task: str) -> dict[str, Any]:
    """The keys a protected tolerances file for `task` is evaluated on, with the documented proposal values."""
    if task == "embedding":
        return {k: None for k in sorted(KNOWN_KEYS)}
    if task == "image-embedding":
        from model_pipeline import evaluate_image

        return dict(evaluate_image.EXPECTED_KEYS)
    if task == "video-embedding":
        from model_pipeline import evaluate_video

        return dict(evaluate_video.EXPECTED_KEYS)
    if task == "generation":
        from model_pipeline import evaluate_generation

        return dict(evaluate_generation.EXPECTED_KEYS)
    from model_pipeline import evaluate_extraction

    return dict(evaluate_extraction.EXPECTED_KEYS)


def evaluate(
    manifest_path: Path,
    results_dir: Path,
    *,
    repo_root: Path | None = None,
    protected_base: str = "auto",
    require_gates: tuple[str, ...] = (),
    now: dt.datetime | None = None,
    reference_tensors: Path | None = None,
) -> Evaluation:
    problems = manifest.validate_file(manifest_path)
    if problems:
        raise EvaluationError("manifest is invalid: " + "; ".join(str(p) for p in problems))
    man = manifest.load(manifest_path)
    task = task_of(man)
    root = repo_root or protection.find_repository_root(manifest_path.parent)
    qual = man["qualification"]
    fixtures_dir = protection.resolve_protected(root, qual["fixtures"], kind="fixtures")
    tol_path = protection.resolve_protected(root, qual["tolerances"], kind="tolerances", must_exist=False)
    if not tol_path.name.startswith("tolerances.") or (tol_path.exists() and not tol_path.is_file()):
        raise protection.ProtectedEvidenceError(f"{qual['tolerances']} is not a protected tolerances.* file")
    # A profile whose protected tolerances file does not exist yet is measured, never qualified: every bound is
    # unset, so every bound-dependent check is not-run and no gate can pass (exit code 4).
    tolerances_missing = not tol_path.exists()
    tol = {} if tolerances_missing else load_tolerances(tol_path, man["id"])
    run = None
    if task == "generation":
        from model_pipeline import evaluate_generation as ge

        # The generation task reads the C# runner's output directory, not stage results files.
        stages = {}
        run = ge.load_results(results_dir)
    else:
        stages = results.load_directory(results_dir)
        if not stages:
            raise EvaluationError(f"{results_dir}: no *{results.RESULTS_SUFFIX} files")
    for s in stages.values():
        if s.task != task:
            raise EvaluationError(f"{s.path}: results are for task {s.task!r}; the profile is evaluated as {task!r}")

    generation = None
    if task == "generation":
        fx = ge.load_fixtures(fixtures_dir)
        verification = protection.verify_unchanged(root, [tol_path, *fx.files], protected_base)
        generation = ge.GenerationEvaluator(man, fx, tol, run, reference_tensors)
        checks = generation.run_checks()
        first, findings = ge.diagnose(checks)
        exceptions = None
    elif task == "embedding":
        fx = load_fixtures(fixtures_dir)
        verification = protection.verify_unchanged(root, [tol_path, *fx.files], protected_base)
        checks = EncoderEvaluator(man, fx, tol, stages).run()
        first, findings = diagnose(checks)
        exceptions = None
    elif task == "image-embedding":
        from model_pipeline import evaluate_image as im

        fx = im.load_fixtures(fixtures_dir)
        verification = protection.verify_unchanged(root, [tol_path, *fx.files], protected_base)
        checks = im.ImageEvaluator(man, fx, tol, stages, reference_tensors).run()
        first, findings = im.diagnose(checks)
        exceptions = None
    elif task == "video-embedding":
        from model_pipeline import evaluate_video as vi

        fx = vi.load_fixtures(fixtures_dir)
        verification = protection.verify_unchanged(root, [tol_path, *fx.files], protected_base)
        checks = vi.VideoEvaluator(man, fx, tol, stages, reference_tensors).run()
        first, findings = vi.diagnose(checks)
        exceptions = None
    else:
        from model_pipeline import evaluate_extraction as ex

        fx = ex.load_fixtures(fixtures_dir)
        verification = protection.verify_unchanged(root, [tol_path, *fx.files], protected_base)
        evaluator = ex.ExtractionEvaluator(man, fx, tol, stages)
        checks = evaluator.run()
        first, findings = ex.diagnose(checks)
        exceptions = evaluator.exceptions
    substages = SUBSTAGES
    gate_status = gates(checks, verification, tolerances_missing)
    date = (now or dt.datetime.now(dt.UTC)).date().isoformat()

    def rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(root.resolve())).replace("\\", "/")
        except ValueError:
            return str(p)

    tolerances_input: dict[str, Any] = {"path": rel(tol_path), "sha256": None, "missing": True}
    if not tolerances_missing:
        tolerances_input = {"path": rel(tol_path), "sha256": fixtures.sha256_file(tol_path)}
    head: dict[str, Any] = {"format": REPORT_FORMAT, "format_version": REPORT_FORMAT_VERSION}
    if task != "embedding":
        head = {"format": REPORT_FORMAT, "format_version": 2, "task": task}
    report = {
        **head,
        "profile": man["id"],
        "date": date,
        "evaluator": {
            "name": "model-pipeline evaluate",
            "version": _version(),
            "repository_commit": fixtures.git_head(root),
        },
        "inputs": {
            "manifest": {"path": rel(manifest_path), "sha256": fixtures.sha256_file(manifest_path)},
            "tolerances": tolerances_input,
            "fixtures": [{"path": rel(p), "sha256": fixtures.sha256_file(p)} for p in fx.files],
            "results": run.describe(rel)
            if run is not None
            else {
                name: {
                    "path": rel(s.path),
                    "sha256": s.sha256,
                    "candidate": s.candidate,
                    "producer": s.producer,
                    "environment": s.environment,
                    "assets": s.assets,
                }
                for name, s in sorted(stages.items(), key=lambda kv: results.STAGES.index(kv[0]))
            },
        },
        "protected_evidence": verification.as_dict(),
        "report_only_metrics": (tol.get("float") or {}).get("report_only", []),
        "stages": {s: substage_status(checks, s) for s in substages},
        "first_divergence": first,
        "findings": findings,
        "gates": gate_status,
        "manifest_gates": {
            g: {"status": v["status"], "date": date} for g, v in gate_status.items() if v["status"] != "not-run"
        },
        "checks": [c.as_dict() for c in checks],
    }
    if exceptions is not None:
        report["entity_exceptions"] = exceptions
    if task in ("image-embedding", "video-embedding"):
        report["excluded_by_contract"] = fx.excluded
        if task == "video-embedding":
            report["evaluated_although_excluded"] = sorted(fx.optional_included)
        report["reference_tensors"] = None if reference_tensors is None else {"directory": str(reference_tensors)}
    if generation is not None:
        report["greedy_divergences"] = generation.divergences
        report["excluded_by_contract"] = generation.excluded
        report["reference_tensors"] = None if reference_tensors is None else {"directory": str(reference_tensors)}
    if tolerances_missing:
        report["expected_tolerance_keys"] = expected_tolerance_keys(task)
    if verification.status == "changed":
        code = EXIT_PROTECTED
    elif any(c.status in ("failed", "invalid") for c in checks):
        code = EXIT_FAILED
    elif tolerances_missing or any(gate_status[g]["status"] != "passed" for g in require_gates):
        code = EXIT_GATE_NOT_PASSED
    else:
        code = EXIT_OK
    report["exit_code"] = code
    return Evaluation(report, code)


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("model-pipeline")
    except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
        return "unknown"


# --------------------------------------------------------------------------------------------
# Markdown


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.3g}"
    if value is None:
        return ""
    return str(value)


def _cos(value: Any) -> str:
    return f"{value:.12f}" if isinstance(value, float) else ""


def markdown(report: dict[str, Any]) -> str:
    if report.get("task") == "image-embedding":
        from model_pipeline import evaluate_image

        return evaluate_image.markdown(report)
    if report.get("task") == "video-embedding":
        from model_pipeline import evaluate_video

        return evaluate_video.markdown(report)
    if report.get("task") == "generation":
        from model_pipeline import evaluate_generation

        return evaluate_generation.markdown(report)
    if report.get("task", "embedding") != "embedding":
        from model_pipeline import evaluate_extraction

        return evaluate_extraction.markdown(report)
    lines = [
        f"# Fidelity report: {report['profile']}",
        "",
        f"Date {report['date']}. Evaluator {report['evaluator']['name']} {report['evaluator']['version']} at "
        f"`{report['evaluator']['repository_commit']}`. Exit code {report['exit_code']}.",
        "",
        f"First divergence: **{report['first_divergence'] or 'none'}**.",
        "",
    ]
    if report["findings"]:
        lines += ["| Stage | Taxonomy | Finding |", "| --- | --- | --- |"]
        lines += [f"| {f['stage']} | {f['taxonomy']} | {f['finding']} |" for f in report["findings"]]
        lines.append("")
    lines += ["## Gates", "", "| Gate | Status | Reason |", "| --- | --- | --- |"]
    lines += [f"| {g} | {v['status']} | {v['reason']} |" for g, v in report["gates"].items()]
    lines += ["", "## Stages", "", "| Stage | Status |", "| --- | --- |"]
    lines += [f"| {SUBSTAGE_TITLES[s]} | {v} |" for s, v in report["stages"].items()]
    lines += [
        "",
        "## Checks",
        "",
        "Absolute errors. Bounds come only from the protected tolerances file; "
        f"report-only metrics: {', '.join(report['report_only_metrics']) or 'none'}.",
        "",
        "| Check | Tolerance key | Bound | Max abs | Mean abs | p99 abs | Min cosine | Cases | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for c in report["checks"]:
        m = c["metrics"]
        lines.append(
            f"| `{c['id']}` | {c['tolerance_key'] or ''} | {_fmt(c['bound'])} | {_fmt(m.get('max_abs'))} | "
            f"{_fmt(m.get('mean_abs'))} | {_fmt(m.get('p99_abs'))} | {_cos(m.get('min_cosine'))} | "
            f"{_fmt(m.get('cases'))} | {c['status']} |"
        )
    details = [c for c in report["checks"] if c["detail"]]
    if details:
        lines += ["", "Details:", ""]
        lines += [f"- `{c['id']}`: {c['detail']}" for c in details]
    pe = report["protected_evidence"]
    lines += [
        "",
        "## Protected evidence",
        "",
        f"Status **{pe['status']}** ({pe['detail']}; base `{pe['base']}`).",
        "",
        "| File | SHA-256 |",
        "| --- | --- |",
    ]
    lines += [f"| `{p}` | `{d}` |" for p, d in pe["sha256"].items()]
    if pe["changed"]:
        lines += ["", "Changed against the base: " + ", ".join(f"`{p}`" for p in pe["changed"])]
    lines += ["", "## Candidate inputs", "", "| Stage | Candidate | Producer | Machine | Results SHA-256 |"]
    lines += ["| --- | --- | --- | --- | --- |"]
    for name, r in report["inputs"]["results"].items():
        producer = r["producer"].get("name", "")
        commit = r["producer"].get("repository_commit")
        lines.append(
            f"| {name} | {r['candidate']} | {producer}{f' `{commit}`' if commit else ''} | "
            f"{r['environment'].get('machine', '')} | `{r['sha256'][:16]}` |"
        )
    return "\n".join(lines) + "\n"


def write_report(evaluation: Evaluation, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / "report.json", out_dir / "report.md"
    fixtures.write_json(json_path, evaluation.report)
    md_path.write_text(markdown(evaluation.report), encoding="utf-8", newline="\n")
    return json_path, md_path
