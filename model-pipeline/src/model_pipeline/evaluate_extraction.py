"""Staged fidelity evaluator, task `entity-extraction` (GLiNER2 span models; backlog P-015, extends P-007).

Stages, walked in this order to find the first divergence (same substage names as the embedding task):

    tokenizer    stage 1, processor: normalized text, words, character maps, input ids, attention mask,
                 word and schema routing positions, exact against `extraction.jsonl`              (G1)
    export       stage 2: Python ONNX Runtime on the exported graph, fed the fixture inputs, against
                 the PyTorch reference: hidden state, count logits and the gate decision, span logits
                 (relative bound), scores, decoded entities                                       (G2)
    dotnet-ids   stage 3: the C# implementation from the fixture inputs; same comparisons, plus
                 C# against Python ONNX Runtime on the same machine and provider                  (G2)
    dotnet-text  stage 4: the C# implementation end to end from text: entities                   (G2)

Entities compare per the rule in docs/m0/gliner2-p014.md, section 6: label, text, start, end and
order exactly, with two kinds of allowed exception, each listed in the report:

- order swap: two items of one label in a different relative order, allowed only when their
  reference confidences differ by at most `entities.order_swap_max_confidence_delta`;
- threshold flip: an item present on one side only, allowed only when its reference score is
  within `entities.threshold_flip_max_score_margin` of the case's threshold.

Anything else fails: a different gate decision, an item that is neither (including an item that
appears because a flipped item no longer suppresses it), items out of label order, duplicates,
offsets outside the normalized text, or a surface that is not `normalized_text[start:end].strip()`.
Reference scores of spans the reference did not return are recomputed from the fixture's span
logits (float32 sigmoid); for returned spans the fixture's own confidence is used.

Tolerance keys (flattened YAML paths) this module reads, from the profile's protected tolerances
file only; see `EXPECTED_KEYS` and model-pipeline/README.md.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_pipeline import fixtures, results
from model_pipeline.evaluate import (
    Check,
    ErrorStats,
    EvaluationError,
    _bound,
    _failed,
    _is_missing_evidence,
    _status,
    flatten,
    substage_status,
)

TASK = "entity-extraction"
SUBSTAGES = ("tokenizer", "export", "dotnet-ids", "dotnet-text")
SUBSTAGE_TITLES = {
    "tokenizer": "Stage 1, processor (normalized text, words, maps, ids, positions vs protected fixtures)",
    "export": "Stage 2, export (Python ONNX Runtime vs PyTorch reference)",
    "dotnet-ids": "Stage 3, .NET from fixture inputs",
    "dotnet-text": "Stage 4, .NET end to end from text",
}

# Every key this evaluator implements, with the proposed value from docs/m0/gliner2-p014.md section 6. The
# values are documentation for the maintainer who writes the protected file; the evaluator never uses them.
EXPECTED_KEYS: dict[str, Any] = {
    "profile": "<family>/<profile>, must equal the manifest id",
    "reference": "free text naming the reference outputs",
    "discrete.normalized_text": "exact",
    "discrete.words": "exact",
    "discrete.character_maps": "exact",
    "discrete.token_ids": "exact",
    "discrete.attention_mask": "exact",
    "discrete.word_positions": "exact",
    "discrete.schema_positions": "exact",
    "discrete.gate_decision": "exact",
    "float.last_hidden_state_max_abs": 1.0e-4,
    "float.count_logits_max_abs": 5.0e-4,
    "float.span_logits_relative": 1.0e-4,
    "float.scores_max_abs": 2.0e-5,
    "float.confidence_max_abs": 2.0e-5,
    "float.padded_batch_vs_single_span_logits_relative": 1.0e-4,
    "float.nan_or_inf": "fail",
    "float.report_only": ["mean_abs", "p99_abs"],
    "entities.identity": "exact",
    "entities.order_swap_max_confidence_delta": 2.0e-5,
    "entities.threshold_flip_max_score_margin": 2.0e-5,
    "provider_options": "{} on CPU; {use_tf32: 0} on CUDA",
    "csharp_vs_python_same_machine_and_provider.span_logits_relative": "optional; not in the P-014 proposal",
}
KNOWN_KEYS = set(EXPECTED_KEYS)
GATE_OF_KEY_PREFIX = {"discrete": "G1"}

# (output, tolerance key) for stage 1. Character maps share one key.
PROCESSOR_OUTPUTS = (
    ("normalized_text", "discrete.normalized_text"),
    ("words", "discrete.words"),
    ("start_map", "discrete.character_maps"),
    ("end_map", "discrete.character_maps"),
    ("input_ids", "discrete.token_ids"),
    ("attention_mask", "discrete.attention_mask"),
    ("word_index", "discrete.word_positions"),
    ("schema_index", "discrete.schema_positions"),
)
INPUTS = ("input_ids", "word_index", "schema_index")
ENTITY_KEYS = (
    "entities.identity",
    "entities.order_swap_max_confidence_delta",
    "entities.threshold_flip_max_score_margin",
)


# --------------------------------------------------------------------------------------------
# float32 arithmetic without numpy


def f32(x: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def sigmoid32(x: float) -> float:
    """torch.sigmoid on a float32 input, rounded to float32 (may differ from torch by one ulp)."""
    x = f32(x)
    if math.isnan(x):
        return x
    if x >= 0:
        return f32(1.0 / (1.0 + math.exp(-x)))
    e = math.exp(x)
    return f32(e / (1.0 + e))


def argmax(values: list[float]) -> int:
    """First index of the maximum, as numpy and torch return it."""
    return max(range(len(values)), key=lambda i: (values[i], -i)) if values else -1


# --------------------------------------------------------------------------------------------
# Fixtures


@dataclass
class Case:
    id: str
    normalized_text: str
    words: list[str]
    start_map: list[int]
    end_map: list[int]
    input_ids: list[int]
    word_index: list[int]
    schema_index: list[int]
    labels: list[str]
    threshold: float
    count_logits: list[float]
    gate_open: bool
    max_width: int
    span_logits: list[float]
    entities: list[dict[str, Any]]
    tensor_sha256: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n, k = len(self.start_map), self.max_width
        self.offsets = []
        total = 0
        for s in range(n):
            self.offsets.append(total)
            total += min(k, n - s)
        self.valid_per_label = total
        self.start_index = {c: i for i, c in enumerate(self.start_map)}
        self.label_order = list(OrderedDict.fromkeys(self.labels))
        self.confidence = {(e["label"], e["start"], e["end"]): e["confidence"] for e in self.entities}

    @property
    def threshold32(self) -> float:
        return f32(self.threshold)

    @property
    def span_logit_count(self) -> int:
        return len(self.labels) * self.valid_per_label

    def span(self, label: str, start: int, end: int) -> tuple[int, int, int] | None:
        """(label index, start word, width) of a character span, or None if no word span of the case has it."""
        if label not in self.labels or start not in self.start_index:
            return None
        s = self.start_index[start]
        e = next((i for i in range(s, len(self.end_map)) if self.end_map[i] == end), None)
        if e is None or e - s >= self.max_width:
            return None
        return self.labels.index(label), s, e - s

    def reference_score(self, label: str, start: int, end: int) -> float | None:
        """The reference's confidence for a returned span, else sigmoid of the fixture span logit."""
        if (label, start, end) in self.confidence:
            return self.confidence[(label, start, end)]
        where = self.span(label, start, end)
        if where is None:
            return None
        m, s, w = where
        return sigmoid32(self.span_logits[m * self.valid_per_label + self.offsets[s] + w])


@dataclass
class Fixtures:
    cases: dict[str, Case]
    files: list[Path]

    @property
    def case_ids(self) -> list[str]:
        return list(self.cases)


def flatten_entities(output: dict[str, Any]) -> list[dict[str, Any]]:
    """GLiNER2's public `{"entities": {label: [items]}}` as one list in output order."""
    items = []
    for label, entries in output.get("entities", {}).items():
        for e in entries:
            items.append(
                {"label": label, "text": e["text"], "start": e["start"], "end": e["end"], "confidence": e["confidence"]}
            )
    return items


def load_fixtures(directory: Path) -> Fixtures:
    path = directory / "extraction.jsonl"
    if not path.is_file():
        raise EvaluationError(f"protected fixture {path} is missing")
    cases = {}
    for row in fixtures.read_jsonl(path):
        try:
            case = Case(
                id=row["id"],
                normalized_text=row["normalized_text"],
                words=row["words"],
                start_map=row["start_map"],
                end_map=row["end_map"],
                input_ids=row["input_ids"],
                word_index=row["word_index"],
                schema_index=row["schema_index"],
                labels=row["labels"],
                threshold=float(row["threshold"]),
                count_logits=row["count_logits"],
                gate_open=bool(row["gate_open"]),
                max_width=int(row["span_logits_shape"][2]),
                span_logits=row["span_logits_valid"],
                entities=flatten_entities(row["output"]),
                tensor_sha256=row.get("tensor_sha256", {}),
            )
        except (KeyError, TypeError, IndexError) as exc:
            raise EvaluationError(f"{path}: case {row.get('id')!r} lacks a field: {exc}") from exc
        if len(case.span_logits) != case.span_logit_count:
            raise EvaluationError(
                f"{path}: case {case.id!r} has {len(case.span_logits)} span logits, expected {case.span_logit_count}"
            )
        if case.id in cases:
            raise EvaluationError(f"{path}: duplicate case {case.id!r}")
        cases[case.id] = case
    if not cases:
        raise EvaluationError(f"{path}: no cases")
    return Fixtures(cases, [path])


# --------------------------------------------------------------------------------------------
# The entity rule


@dataclass
class EntityComparison:
    violations: list[str] = field(default_factory=list)
    exceptions: list[dict[str, Any]] = field(default_factory=list)
    confidence_diffs: list[float] = field(default_factory=list)


def _key(e: dict[str, Any]) -> tuple[str, int, int]:
    return e["label"], e["start"], e["end"]


def compare_entities(
    case: Case,
    expected: dict[str, Any],
    candidate: dict[str, Any],
    swap_bound: float,
    flip_bound: float,
) -> EntityComparison:
    """Compare a candidate entity result with an expected one (the reference, or the candidate's own single run).

    Exceptions are decided on reference scores from the fixture (`Case.reference_score`), never on the
    expected side's confidences, so comparing a batch row with its single row uses the same evidence.
    """
    out = EntityComparison()
    text = case.normalized_text
    if candidate["gate_open"] != expected["gate_open"]:
        out.violations.append(
            f"gate decision differs: candidate {'open' if candidate['gate_open'] else 'closed'}, "
            f"expected {'open' if expected['gate_open'] else 'closed'}"
        )
        return out
    items = candidate["entities"]
    if not candidate["gate_open"] and items:
        out.violations.append(f"gate closed but {len(items)} entities returned")
        return out
    for e in items:
        where = f"{e['label']} [{e['start']}, {e['end']})"
        if e["label"] not in case.label_order:
            out.violations.append(f"{where}: label not in the case's labels {case.label_order}")
        if not 0 <= e["start"] < e["end"] <= len(text):
            out.violations.append(f"{where}: offsets outside the normalized text (length {len(text)})")
        elif e["text"] != text[e["start"] : e["end"]].strip():
            out.violations.append(f"{where}: text {e['text']!r} is not normalized_text[start:end].strip()")
        if not math.isfinite(e["confidence"]):
            out.violations.append(f"{where}: confidence {e['confidence']} is not finite")
    if out.violations:
        return out

    ranks = [case.label_order.index(e["label"]) for e in items]
    if any(b < a for a, b in zip(ranks, ranks[1:], strict=False)):
        out.violations.append(f"items are not grouped in schema label order {case.label_order}")
    for label in case.label_order:
        exp = [e for e in expected["entities"] if e["label"] == label]
        cand = [e for e in items if e["label"] == label]
        cand_keys = [_key(e) for e in cand]
        dupes = sorted({k for k in cand_keys if cand_keys.count(k) > 1})
        if dupes:
            out.violations.append(f"{label}: duplicate items {[(s, t) for _, s, t in dupes]}")
            continue
        exp_pos = {_key(e): i for i, e in enumerate(exp)}
        cand_set = set(cand_keys)
        score: dict[tuple[str, int, int], float | None] = {}
        for e in [*exp, *cand]:
            score[_key(e)] = case.reference_score(*_key(e))

        for e in exp:
            if _key(e) not in cand_set:
                _flip(out, case, e, score[_key(e)], "dropped", flip_bound)
        for e in cand:
            if _key(e) not in exp_pos:
                _flip(out, case, e, score[_key(e)], "added", flip_bound)
        exp_by_key = {_key(e): e for e in exp}
        for e in cand:
            if _key(e) in exp_by_key:
                out.confidence_diffs.append(abs(e["confidence"] - exp_by_key[_key(e)]["confidence"]))

        for i, a in enumerate(cand):
            for b in cand[i + 1 :]:
                ka, kb = _key(a), _key(b)
                sa, sb = score[ka], score[kb]
                if ka in exp_pos and kb in exp_pos:
                    inverted = exp_pos[ka] > exp_pos[kb]
                elif sa is None or sb is None:
                    continue  # already a violation
                else:
                    inverted = sa < sb
                if not inverted:
                    continue
                delta = abs(sa - sb) if sa is not None and sb is not None else math.inf
                pair = (
                    f"{label}: {a['text']!r} [{a['start']}, {a['end']}) before {b['text']!r} [{b['start']}, {b['end']})"
                )
                if delta <= swap_bound:
                    out.exceptions.append(
                        {
                            "kind": "order-swap",
                            "label": label,
                            "first": {"text": a["text"], "start": a["start"], "end": a["end"], "reference_score": sa},
                            "second": {"text": b["text"], "start": b["start"], "end": b["end"], "reference_score": sb},
                            "delta": delta,
                        }
                    )
                else:
                    out.violations.append(
                        f"{pair}: order differs from the reference and reference confidences differ by {delta:.3g} "
                        f"(allowed {swap_bound:.3g})"
                    )
    return out


def _flip(
    out: EntityComparison, case: Case, e: dict[str, Any], r: float | None, direction: str, flip_bound: float
) -> None:
    """An item on one side only: an allowed threshold flip, or a violation."""
    thr = case.threshold32
    span = f"{e['label']} {e['text']!r} [{e['start']}, {e['end']})"
    if r is None:
        out.violations.append(f"{span} {direction}: no word span of width <= {case.max_width} has these offsets")
        return
    margin = abs(r - thr)
    if margin <= flip_bound:
        out.exceptions.append(
            {
                "kind": "threshold-flip",
                "direction": direction,
                "label": e["label"],
                "text": e["text"],
                "start": e["start"],
                "end": e["end"],
                "reference_score": r,
                "threshold": thr,
                "margin": margin,
            }
        )
    else:
        out.violations.append(
            f"{span} {direction}: reference score {r:.9g} is {margin:.3g} from the threshold {thr:.9g} "
            f"(allowed {flip_bound:.3g})"
        )


def reference_result(case: Case) -> dict[str, Any]:
    return {"gate_open": case.gate_open, "entities": case.entities}


# --------------------------------------------------------------------------------------------
# Relative error statistics for span logits


@dataclass
class RelStats(ErrorStats):
    """Absolute errors as ErrorStats, plus |d| / max(1, |ref|) per value (the span-logit bound)."""

    relative: list[float] = field(default_factory=list)
    per_case_rel: dict[str, float] = field(default_factory=dict)

    def add(self, case_id: str, actual: list[float], expected: list[float], cosine: bool = False) -> None:
        super().add(case_id, actual, expected, False)
        rel = [
            abs(a - e) / max(1.0, abs(e)) if math.isfinite(a) else math.inf
            for a, e in zip(actual, expected, strict=True)
        ]
        self.relative.extend(rel)
        self.per_case_rel[case_id] = max(rel, default=0.0)

    def summary(self) -> dict[str, Any]:
        out = super().summary()
        if self.relative:
            out["max_rel"] = max(self.relative)
            out["worst_case_rel"] = max(self.per_case_rel, key=lambda k: self.per_case_rel[k])
        return out


# --------------------------------------------------------------------------------------------
# Checks


class ExtractionEvaluator:
    """Checks for an `entity-extraction` profile. One instance per evaluation."""

    def __init__(self, man: dict[str, Any], fx: Fixtures, tol: dict[str, Any], stages: dict[str, results.StageResults]):
        if man["behavior"]["offsets_unit"] != "unicode-scalars":
            raise EvaluationError(
                f"{man['id']}: behavior.offsets_unit is {man['behavior']['offsets_unit']!r}; the extraction fixtures "
                "record Unicode scalar offsets and no conversion is implemented"
            )
        self.man = man
        self.fx = fx
        self.tol = flatten(tol)
        self.stages = stages
        self.checks: list[Check] = []
        self.exceptions: list[dict[str, Any]] = []

    # -- helpers ------------------------------------------------------------------------------

    def _add(self, check: Check) -> Check:
        self.checks.append(check)
        return check

    def _missing_stage(self, check: Check, stage: str) -> Check:
        check.status = "not-run"
        check.detail = f"no results for stage {stage!r}"
        if check.tolerance_key and check.bound is None:
            check.bound = self.tol.get(check.tolerance_key)
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

    def _output(self, stage: results.StageResults, output: str) -> dict[str, Any]:
        return {cid: outs[output] for cid, outs in stage.cases.items() if output in outs}

    def _exact_key(self, check: Check, key: str) -> bool:
        """True if `key` is bound to 'exact'; otherwise the check is recorded as not-run."""
        check.bound = self.tol.get(key)
        if key not in self.tol:
            check.detail = f"{key} is not in the protected tolerances"
        elif self.tol[key] != "exact":
            check.detail = f"{key}: {self.tol[key]!r} has no implemented meaning (only 'exact')"
        else:
            return True
        self._add(check)
        return False

    def _judge(self, check: Check, bound: float | None, metric: str = "max_abs") -> Check:
        check.bound = bound
        if check.metrics.get("non_finite_cases"):
            check.status = "failed"
            check.detail = f"NaN or Inf in cases {check.metrics['non_finite_cases']}"
        elif bound is None:
            check.status = "not-run"
            check.detail = f"no numeric bound for {check.tolerance_key} in the protected tolerances"
        elif metric not in check.metrics:
            check.status, check.detail = "passed", "no values to compare"
        else:
            value = check.metrics[metric]
            check.status = "passed" if value <= bound else "failed"
            if check.status == "failed":
                worst = check.metrics.get("worst_case_rel" if metric == "max_rel" else "worst_case")
                check.detail = f"{metric.replace('_', ' ')} {value:.3g} > {bound:.3g} (worst case {worst})"
        return self._add(check)

    def _compare_numbers(
        self,
        check: Check,
        actual: dict[str, list[float]],
        expected: dict[str, list[float]],
        bound: float | None,
        *,
        relative: bool = False,
        transform=None,
        expected_stage: str | None = None,
    ) -> Check:
        problem = self._case_set_problem(set(actual), check.id)
        if problem:
            check.bound = bound
            check.status, check.detail = "failed", problem
            return self._add(check)
        missing_expected = sorted(set(self.fx.case_ids) - set(expected))
        if missing_expected:
            check.bound = bound
            if expected_stage:
                check.status = "not-run"
                check.detail = (
                    f"the {expected_stage} results lack cases {missing_expected}; that stage's checks fail, "
                    "and this comparison cannot be completed"
                )
            else:
                check.status, check.detail = "failed", f"reference lacks cases {missing_expected}"
            return self._add(check)
        stats = RelStats() if relative else ErrorStats()
        try:
            for cid in self.fx.case_ids:
                a, e = actual[cid], expected[cid]
                if transform:
                    a, e = [transform(v) for v in a], [transform(v) for v in e]
                stats.add(cid, a, e, False)
        except ValueError as exc:
            check.bound = bound
            check.status, check.detail = "failed", f"shape mismatch: {exc}"
            return self._add(check)
        check.metrics = stats.summary()
        return self._judge(check, bound, "max_rel" if relative else "max_abs")

    def _entity_bounds(self) -> tuple[float, float] | str:
        missing = [k for k in ENTITY_KEYS if k not in self.tol]
        if missing:
            return f"{', '.join(missing)} not in the protected tolerances"
        if self.tol["entities.identity"] != "exact":
            return f"entities.identity: {self.tol['entities.identity']!r} has no implemented meaning (only 'exact')"
        swap, flip = _bound(self.tol, ENTITY_KEYS[1]), _bound(self.tol, ENTITY_KEYS[2])
        if swap is None or flip is None or swap < 0 or flip < 0:
            return "entities order-swap and threshold-flip bounds must be non-negative numbers"
        return swap, flip

    def _compare_entity_results(
        self,
        check: Check,
        conf: Check | None,
        actual: dict[str, dict[str, Any]],
        expected: dict[str, dict[str, Any]] | None = None,
        case_ids: list[str] | None = None,
    ) -> None:
        """Entity rule over every case (or `case_ids`); `expected` defaults to the reference."""
        bounds = self._entity_bounds()
        check.bound = {k: self.tol.get(k) for k in ENTITY_KEYS}
        if isinstance(bounds, str):
            check.detail = bounds
            self._add(check)
            if conf is not None:
                conf.bound = _bound(self.tol, "float.confidence_max_abs")
                conf.detail = "entities were not compared: " + bounds
                self._add(conf)
            return
        swap, flip = bounds
        ids = case_ids if case_ids is not None else self.fx.case_ids
        problem = None if case_ids is not None else self._case_set_problem(set(actual), check.id)
        violations: dict[str, list[str]] = {}
        exceptions: list[dict[str, Any]] = []
        diffs: dict[str, float] = {}
        exact = 0
        for cid in ids:
            if cid not in actual:
                continue
            case = self.fx.cases[cid]
            exp = expected[cid] if expected is not None else reference_result(case)
            cmp = compare_entities(case, exp, actual[cid], swap, flip)
            if cmp.violations:
                violations[cid] = cmp.violations
            exceptions += [dict(x, case=cid, check=check.id) for x in cmp.exceptions]
            if not cmp.violations and not cmp.exceptions:
                exact += 1
            if cmp.confidence_diffs:
                diffs[cid] = max(cmp.confidence_diffs)
        self.exceptions += exceptions
        check.metrics = {
            "cases": len([c for c in ids if c in actual]),
            "exact_cases": exact,
            "cases_with_allowed_exceptions": sorted({x["case"] for x in exceptions}),
            "allowed_exceptions": exceptions,
            "violations": violations,
        }
        if problem:
            check.status, check.detail = "failed", problem
        elif violations:
            first = next(iter(violations))
            check.status = "failed"
            check.detail = (
                f"{len(violations)} case(s) differ beyond the allowed exceptions: {sorted(violations)}; "
                f"first: {first}: {violations[first][0]}"
            )
        else:
            check.status = "passed"
            if exceptions:
                check.detail = f"{len(exceptions)} allowed exception(s), listed in metrics.allowed_exceptions"
        self._add(check)
        if conf is not None:
            stats = ErrorStats()
            for cid, d in diffs.items():
                stats.add(cid, [d], [0.0], False)
            conf.metrics = stats.summary()
            if problem:
                conf.bound = _bound(self.tol, "float.confidence_max_abs")
                conf.status, conf.detail = "failed", problem
                self._add(conf)
            else:
                self._judge(conf, _bound(self.tol, "float.confidence_max_abs"))

    # -- stage 0: identity ------------------------------------------------------------------------

    def check_identity(self) -> None:
        manifest_files = {f["path"]: f["sha256"] for f in self.man["origin"]["files"]}
        graph = [p for p in manifest_files if ".onnx" in p]
        required = {"tokenizer": ["tokenizer.json"], "export": graph, "dotnet": [*graph, "tokenizer.json"]}
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
            if name != "tokenizer" and not graph:
                problems.append(
                    "the manifest's origin.files pins no ONNX graph, so the artifact this stage ran cannot be "
                    "identified"
                )
            for path in required[name]:
                if path not in declared:
                    problems.append(f"asset {path} not declared; the stage must name the artifact it ran")
            check.status = "invalid" if problems else "passed"
            check.detail = "; ".join(problems)
            self._add(check)

    # -- stage 1: processor --------------------------------------------------------------------------

    def _expected(self, output: str) -> dict[str, Any]:
        if output == "attention_mask":
            return {cid: [1] * len(c.input_ids) for cid, c in self.fx.cases.items()}
        return {cid: getattr(c, output) for cid, c in self.fx.cases.items()}

    def check_processor(self) -> None:
        stage = self.stages.get("tokenizer")
        for output, key in PROCESSOR_OUTPUTS:
            check = Check(f"tokenizer.{output}", "tokenizer", f"processor {output} exact", key)
            if not self._exact_key(check, key):
                continue
            if stage is None:
                self._missing_stage(check, "tokenizer")
                continue
            actual, expected = self._output(stage, output), self._expected(output)
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
        if stage is not None:
            self._maps_well_formed(stage)

    def _maps_well_formed(self, stage: results.StageResults) -> None:
        """Structural: character maps index the stage's own normalized text, one entry per word, in order."""
        check = Check("tokenizer.character_maps.well_formed", "tokenizer", "processor character maps are well formed")
        bad = {}
        for cid, outs in stage.cases.items():
            if not {"normalized_text", "words", "start_map", "end_map"} <= set(outs):
                continue
            text, words, starts, ends = outs["normalized_text"], outs["words"], outs["start_map"], outs["end_map"]
            if not (len(words) == len(starts) == len(ends)):
                bad[cid] = f"{len(words)} words, {len(starts)} starts, {len(ends)} ends"
                continue
            for i, (s, e) in enumerate(zip(starts, ends, strict=True)):
                if not (0 <= s < e <= len(text)):
                    bad[cid] = f"word {i} [{s}, {e}) outside the normalized text (length {len(text)})"
                    break
                if i and s < ends[i - 1]:
                    bad[cid] = f"word {i} starts at {s}, before the previous word ends ({ends[i - 1]})"
                    break
        check.metrics = {"malformed_cases": bad}
        check.status = "failed" if bad else "passed"
        check.detail = f"malformed maps in {sorted(bad)}: {next(iter(bad.values()))}" if bad else ""
        self._add(check)

    # -- shared stage checks ---------------------------------------------------------------------------

    def _input_integrity(self, stage: results.StageResults, substage: str) -> bool:
        check = Check(f"{stage.stage}.inputs", substage, f"{stage.stage}: ran on the protected fixture inputs")
        problems = []
        for output in INPUTS:
            actual = self._output(stage, output)
            unrecorded = sorted(set(stage.cases) - set(actual))
            unknown = sorted(set(actual) - set(self.fx.case_ids))
            wrong = [
                cid for cid in self.fx.case_ids if cid in actual and actual[cid] != getattr(self.fx.cases[cid], output)
            ]
            if unrecorded or unknown or wrong:
                problems.append(f"{output}: unrecorded {unrecorded}, unknown cases {unknown}, differing {wrong}")
        if problems:
            check.status = "invalid"
            check.detail = "; ".join(problems) + "; the comparison would be invalid"
        else:
            check.status = "passed"
        self._add(check)
        return check.status == "passed"

    def _provider_options(self, stage: results.StageResults, substage: str) -> None:
        if "provider_options" not in self.tol:
            return
        want = self.tol["provider_options"] or {}
        have = stage.environment.get("provider_options", {})
        check = Check(f"{stage.stage}.provider_options", substage, f"{stage.stage}: provider options of the profile")
        check.bound = want
        diff = {k: have.get(k) for k, v in want.items() if str(have.get(k)) != str(v)}
        check.status = "failed" if diff else "passed"
        check.detail = f"stage ran with {diff}, the profile requires {want}" if diff else ""
        check.metrics = {"declared": have}
        self._add(check)

    @staticmethod
    def _non_finite_values(value: Any) -> bool:
        if isinstance(value, dict):
            return any(not math.isfinite(e["confidence"]) for e in value.get("entities", []))
        if isinstance(value, list):
            return any(isinstance(v, int | float) and not isinstance(v, bool) and not math.isfinite(v) for v in value)
        return False

    def _finite(self, stage: results.StageResults, substage: str) -> None:
        key = "float.nan_or_inf"
        check = Check(f"{stage.stage}.finite", substage, f"{stage.stage}: no NaN or Inf in any output", key)
        check.bound = self.tol.get(key)
        if check.bound != "fail":
            check.detail = f"{key}: {check.bound!r} has no implemented meaning (only 'fail')"
            self._add(check)
            return
        bad = [
            f"{cid}/{name}"
            for cid, outs in stage.cases.items()
            for name, v in outs.items()
            if self._non_finite_values(v)
        ]
        for bname, batch in stage.batches.items():
            bad += [
                f"batch {bname}/{cid}"
                for cid, row in zip(batch["case_ids"], batch["values"], strict=True)
                if self._non_finite_values(row)
            ]
        for tname in stage.tensor_specs:
            bad += [
                f"tensor {tname}/{cid}"
                for cid, values in stage.tensor(tname).values.items()
                if any(not math.isfinite(v) for v in values)
            ]
        check.metrics = {"non_finite": bad}
        check.status = "failed" if bad else "passed"
        check.detail = f"NaN or Inf in {bad}" if bad else ""
        self._add(check)

    def _gate(self, stage: results.StageResults, substage: str) -> None:
        key = "discrete.gate_decision"
        check = Check(f"{stage.stage}.gate_decision", substage, f"{stage.stage}: count gate decision exact", key)
        if not self._exact_key(check, key):
            return
        counts = self._output(stage, "count_logits")
        problem = self._case_set_problem(set(counts), check.id)
        mismatched, inconsistent = [], []
        for cid in self.fx.case_ids:
            if cid not in counts:
                continue
            gate = argmax(counts[cid]) > 0
            if gate != self.fx.cases[cid].gate_open:
                mismatched.append(cid)
            ent = stage.cases[cid].get("entities")
            if isinstance(ent, dict) and ent.get("gate_open") != gate:
                inconsistent.append(cid)
        check.metrics = {
            "cases": len(counts),
            "mismatched_cases": mismatched,
            "entities_gate_inconsistent_with_count_logits": inconsistent,
        }
        parts = [problem] if problem else []
        if mismatched:
            parts.append(f"gate decision differs from the reference in {mismatched}")
        if inconsistent:
            parts.append(f"the entity result's gate_open disagrees with the stage's own count logits in {inconsistent}")
        check.status = "failed" if parts else "passed"
        check.detail = "; ".join(parts)
        self._add(check)

    def _numeric(self, stage: results.StageResults, substage: str) -> None:
        name = stage.stage
        ref_counts = {cid: c.count_logits for cid, c in self.fx.cases.items()}
        ref_logits = {cid: c.span_logits for cid, c in self.fx.cases.items()}
        count_key, span_key, score_key = (
            "float.count_logits_max_abs",
            "float.span_logits_relative",
            "float.scores_max_abs",
        )
        self._compare_numbers(
            Check(f"{name}.count_logits", substage, f"{name}: count logits vs reference", count_key),
            self._output(stage, "count_logits"),
            ref_counts,
            _bound(self.tol, count_key),
        )
        self._gate(stage, substage)
        logits = self._output(stage, "span_logits")
        self._compare_numbers(
            Check(
                f"{name}.span_logits", substage, f"{name}: span logits (valid spans) vs reference, relative", span_key
            ),
            logits,
            ref_logits,
            _bound(self.tol, span_key),
            relative=True,
        )
        self._compare_numbers(
            Check(
                f"{name}.scores", substage, f"{name}: scores (float32 sigmoid of span logits) vs reference", score_key
            ),
            logits,
            ref_logits,
            _bound(self.tol, score_key),
            transform=sigmoid32,
        )

    def _entities(self, stage: results.StageResults, output: str, substage: str, title: str) -> None:
        check = Check(f"{stage.stage}.{output}", substage, title, "entities.identity")
        conf = Check(
            f"{stage.stage}.{output}.confidence",
            substage,
            f"{stage.stage}: {output} confidence vs reference (matched items)",
            "float.confidence_max_abs",
        )
        actual = {cid: v for cid, v in self._output(stage, output).items() if isinstance(v, dict)}
        self._compare_entity_results(check, conf, actual)

    def _batches(self, stage: results.StageResults, substage: str, required: tuple[str, ...]) -> None:
        pad_key, span_key, count_key = (
            "float.padded_batch_vs_single_span_logits_relative",
            "float.span_logits_relative",
            "float.count_logits_max_abs",
        )

        def sub_of(output: str) -> str:
            return "dotnet-text" if output.startswith("text_") else substage

        for output in required:
            key = pad_key if output == "span_logits" else "entities.identity"
            evidence = Check(
                f"{stage.stage}.batch_evidence.{output}",
                sub_of(output),
                f"{stage.stage}: a padded batch exercises '{output}'",
                key,
                self.tol.get(key),
            )
            mine = {n: b for n, b in stage.batches.items() if b["output"] == output}
            padded = sorted(n for n, b in mine.items() if self._pads(b["case_ids"]))
            evidence.metrics = {"batches": sorted(mine), "padding_batches": padded}
            if padded:
                evidence.status = "passed"
            else:
                evidence.status = "failed"
                evidence.detail = (
                    f"no batch for output '{output}'"
                    if not mine
                    else f"batches {sorted(mine)} never pad (each needs more than one row with differing input lengths)"
                ) + "; the padded-batch bounds are binding and need evidence"
            self._add(evidence)

        for bname, batch in stage.batches.items():
            output = batch["output"]
            sub = sub_of(output)
            prefix = f"{stage.stage}.batch.{bname}"
            singles = self._output(stage, output)
            unknown = sorted(set(batch["case_ids"]) - set(self.fx.case_ids))
            missing_single = sorted(set(batch["case_ids"]) - set(singles))
            rows = dict(zip(batch["case_ids"], batch["values"], strict=True))
            if output in ("span_logits", "count_logits"):
                key = span_key if output == "span_logits" else count_key
                vs_single = Check(
                    f"{prefix}.vs_single",
                    sub,
                    f"{stage.stage}: batch '{bname}' vs own single rows",
                    pad_key if output == "span_logits" else count_key,
                )
                vs_ref = Check(f"{prefix}.vs_reference", sub, f"{stage.stage}: batch '{bname}' vs reference", key)
                if unknown or missing_single or len(rows) != len(batch["case_ids"]):
                    for c in (vs_single, vs_ref):
                        c.status = "failed"
                        c.detail = (
                            f"batch rows without fixture case {unknown} or single output {missing_single}, "
                            "or repeated rows"
                        )
                        self._add(c)
                    continue
                ref = {
                    cid: (
                        self.fx.cases[cid].span_logits if output == "span_logits" else self.fx.cases[cid].count_logits
                    )
                    for cid in rows
                }
                relative = output == "span_logits"
                s1, s2 = (RelStats(), RelStats()) if relative else (ErrorStats(), ErrorStats())
                try:
                    for cid, row in rows.items():
                        s1.add(cid, row, singles[cid], False)
                        s2.add(cid, row, ref[cid], False)
                except (ValueError, TypeError) as exc:
                    for c in (vs_single, vs_ref):
                        c.status, c.detail = "failed", f"shape mismatch: {exc}"
                        self._add(c)
                    continue
                metric = "max_rel" if relative else "max_abs"
                vs_single.metrics, vs_ref.metrics = s1.summary(), s2.summary()
                self._judge(vs_single, _bound(self.tol, vs_single.tolerance_key), metric)
                self._judge(vs_ref, _bound(self.tol, key), metric)
            elif output in ("entities", "text_entities"):
                vs_single = Check(
                    f"{prefix}.vs_single",
                    sub,
                    f"{stage.stage}: batch '{bname}' entities vs own single rows",
                    "entities.identity",
                )
                vs_ref = Check(
                    f"{prefix}.vs_reference",
                    sub,
                    f"{stage.stage}: batch '{bname}' entities vs reference",
                    "entities.identity",
                )
                if unknown or missing_single or not all(isinstance(v, dict) for v in rows.values()):
                    for c in (vs_single, vs_ref):
                        c.status = "failed"
                        c.detail = f"batch rows without fixture case {unknown} or single output {missing_single}"
                        self._add(c)
                    continue
                self._compare_entity_results(vs_single, None, rows, singles, list(rows))
                self._compare_entity_results(vs_ref, None, rows, None, list(rows))
            else:
                self._add(
                    Check(
                        f"{prefix}.unknown_output",
                        sub,
                        f"{stage.stage}: batch '{bname}' of output '{output}'",
                        status="not-applicable",
                        detail="no batch comparison is defined for this output; not evaluated",
                    )
                )

    def _pads(self, case_ids: list[str]) -> bool:
        lengths = {len(self.fx.cases[c].input_ids) for c in case_ids if c in self.fx.cases}
        return len(case_ids) > 1 and len(lengths) > 1

    # -- stage 2: export ---------------------------------------------------------------------------------

    STAGE2_CHECKS = (
        ("last_hidden_state", "float.last_hidden_state_max_abs"),
        ("count_logits", "float.count_logits_max_abs"),
        ("gate_decision", "discrete.gate_decision"),
        ("span_logits", "float.span_logits_relative"),
        ("scores", "float.scores_max_abs"),
        ("entities", "entities.identity"),
        ("finite", "float.nan_or_inf"),
    )

    def check_export(self) -> None:
        stage = self.stages.get("export")
        if stage is None:
            for name, key in self.STAGE2_CHECKS:
                self._missing_stage(Check(f"export.{name}", "export", f"export: {name}", key), "export")
            return
        if not self._input_integrity(stage, "export"):
            return
        self._provider_options(stage, "export")
        self._finite(stage, "export")
        self._hidden(stage)
        self._numeric(stage, "export")
        self._entities(stage, "entities", "export", "export: entities decoded from Python ORT logits vs reference")
        self._batches(stage, "export", ("span_logits",))

    def _hidden(self, stage: results.StageResults) -> None:
        key = "float.last_hidden_state_max_abs"
        check = Check("export.last_hidden_state", "export", "Python ORT hidden states vs PyTorch hidden states", key)
        if not {"last_hidden_state", "reference_last_hidden_state"} <= set(stage.tensor_specs):
            check.bound = _bound(self.tol, key)
            check.status = "failed"
            check.detail = (
                "missing cases (all): export results need tensors 'last_hidden_state' and 'reference_last_hidden_state'"
            )
            self._add(check)
            return
        cand, ref = stage.tensor("last_hidden_state"), stage.tensor("reference_last_hidden_state")
        if cand.shapes != ref.shapes:
            check.bound = _bound(self.tol, key)
            check.status, check.detail = "failed", "candidate and reference hidden-state shapes differ"
            self._add(check)
            return
        self._compare_numbers(check, cand.values, ref.values, _bound(self.tol, key))
        # Report only: whether the stage's PyTorch rerun reproduces the fixture's hidden state bit for bit.
        same = 0
        for cid, values in ref.values.items():
            want = self.fx.cases.get(cid, None)
            if want is None or "last_hidden_state" not in want.tensor_sha256:
                continue
            data = struct.pack(f"<{len(values)}f", *values)
            same += hashlib.sha256(data).hexdigest() == want.tensor_sha256["last_hidden_state"]
        check.metrics["reference_rerun_bit_identical_to_fixture_cases"] = same

    # -- stages 3 and 4: dotnet ------------------------------------------------------------------------

    STAGE3_CHECKS = (
        ("count_logits", "dotnet-ids", "float.count_logits_max_abs"),
        ("gate_decision", "dotnet-ids", "discrete.gate_decision"),
        ("span_logits", "dotnet-ids", "float.span_logits_relative"),
        ("scores", "dotnet-ids", "float.scores_max_abs"),
        ("entities", "dotnet-ids", "entities.identity"),
        ("finite", "dotnet-ids", "float.nan_or_inf"),
        ("text_entities", "dotnet-text", "entities.identity"),
    )

    def check_dotnet(self) -> None:
        stage = self.stages.get("dotnet")
        export = self.stages.get("export")
        if stage is None:
            for name, sub, key in self.STAGE3_CHECKS:
                self._missing_stage(Check(f"dotnet.{name}", sub, f"dotnet: {name}", key), "dotnet")
            return
        if not self._input_integrity(stage, "dotnet-ids"):
            # The text path does not use the fixture inputs; it is still evaluated.
            self._entities(stage, "text_entities", "dotnet-text", "C# entities from text vs reference")
            return
        self._provider_options(stage, "dotnet-ids")
        self._finite(stage, "dotnet-ids")
        self._numeric(stage, "dotnet-ids")
        self._entities(stage, "entities", "dotnet-ids", "C# entities from fixture inputs vs reference")
        self._vs_python(stage, export)
        self._entities(stage, "text_entities", "dotnet-text", "C# entities from text vs reference")
        self._batches(stage, "dotnet-ids", ("span_logits", "text_entities"))

    def _vs_python(self, stage: results.StageResults, export: results.StageResults | None) -> None:
        key = "csharp_vs_python_same_machine_and_provider.span_logits_relative"
        check = Check(
            "dotnet.span_logits.vs_python", "dotnet-ids", "C# vs Python ORT span logits, same inputs, relative", key
        )
        binding = key in self.tol
        if export is None:
            if binding:
                self._missing_stage(check, "export")
            return
        if not self._same_machine(stage, export):
            if binding:
                check.bound = _bound(self.tol, key)
                check.detail = (
                    "the bound applies to the same machine and provider; the stages declare "
                    f"{stage.environment.get('machine')!r}/{stage.environment.get('provider')!r} and "
                    f"{export.environment.get('machine')!r}/{export.environment.get('provider')!r}"
                )
                self._add(check)
            return
        self._compare_numbers(
            check,
            self._output(stage, "span_logits"),
            self._output(export, "span_logits"),
            _bound(self.tol, key),
            relative=True,
            expected_stage="export",
        )
        if not binding and check.status in ("passed", "failed", "not-run"):
            check.status = "not-applicable"
            check.detail = f"report only: {key} is not in the protected tolerances"
            check.metrics["bit_identical_cases"] = sum(
                stage.cases[c].get("span_logits") == export.cases[c].get("span_logits")
                for c in self.fx.case_ids
                if c in stage.cases and c in export.cases
            )

    @staticmethod
    def _same_machine(a: results.StageResults, b: results.StageResults) -> bool:
        keys = ("machine", "provider")
        return all(a.environment.get(k) and a.environment.get(k) == b.environment.get(k) for k in keys)

    # -- coverage of the tolerance file ------------------------------------------------------------------

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
        self.check_processor()
        self.check_export()
        self.check_dotnet()
        self.check_unknown_keys()
        return self.checks


# --------------------------------------------------------------------------------------------
# Attribution


def _real(checks: list[Check], substage: str, pred=lambda c: True) -> list[Check]:
    return [
        c for c in checks if c.substage == substage and c.status == "failed" and not _is_missing_evidence(c) and pred(c)
    ]


NUMERIC = (".count_logits", ".gate_decision", ".span_logits", ".scores", ".last_hidden_state", ".finite")


def _is_numeric(c: Check) -> bool:
    return ".batch." not in c.id and c.id.endswith(NUMERIC)


def diagnose(checks: list[Check]) -> tuple[str | None, list[dict[str, str]]]:
    status = {s: substage_status(checks, s) for s in SUBSTAGES}
    first = next((s for s in SUBSTAGES if status[s] == "failed"), None)
    findings: list[dict[str, str]] = []

    def add(stage: str, taxonomy: str, text: str) -> None:
        findings.append({"stage": stage, "taxonomy": taxonomy, "finding": text})

    invalid = [c for c in checks if c.status == "invalid"]
    for c in invalid:
        add(c.substage, "packaging/harness", f"{c.id}: {c.detail}. Outputs of this stage are not comparable.")
    for c in checks:
        if c.status == "failed" and _is_missing_evidence(c):
            add(c.substage, "missing evidence", f"{c.id}: {c.detail}.")
    has_invalid = {s: any(c.substage == s for c in invalid) for s in SUBSTAGES}

    tok = _real(checks, "tokenizer")
    if tok and not has_invalid["tokenizer"]:
        names = sorted({c.id.split(".")[1] for c in tok})
        if set(names) & {"normalized_text", "words", "start_map", "end_map", "character_maps"}:
            add(
                "tokenizer",
                "processor",
                f"Text normalization, word splitting or character maps differ from the fixtures ({', '.join(names)}).",
            )
        elif set(names) & {"word_index", "schema_index"}:
            add(
                "tokenizer",
                "processor",
                f"Schema prefix or routing positions differ from the fixtures ({', '.join(names)}).",
            )
        else:
            add("tokenizer", "tokenizer", f"Subword ids or masks differ from the fixtures ({', '.join(names)}).")

    exp = _real(checks, "export")
    export_failed = status["export"] == "failed"
    if exp and not has_invalid["export"]:
        if all(".batch." in c.id for c in exp):
            add(
                "export",
                "export/operator",
                "Python ONNX Runtime single rows match the reference but padded batches do not.",
            )
        elif any(_is_numeric(c) for c in exp):
            add(
                "export",
                "export/operator or precision",
                "Python ONNX Runtime on the exported graph differs from the PyTorch reference (logits, gate or hidden "
                "state): the fault is in the exported artifact or the backend, upstream of .NET.",
            )
        else:
            add(
                "export",
                "threshold or ordering boundary",
                "Logits are within their bounds but the decoded entities differ beyond the allowed exceptions: a "
                "threshold-boundary or ordering failure (plan section 10, G2), or a fault in the stage's decoder.",
            )

    ids = _real(checks, "dotnet-ids")
    if ids and not has_invalid["dotnet-ids"]:
        vs_py = _status(checks, "dotnet.span_logits.vs_python")
        numeric_failed = any(_is_numeric(c) for c in ids)
        if vs_py == "failed":
            add(
                "dotnet-ids",
                ".NET runtime binding or preprocessing",
                "C# span logits differ from Python ONNX Runtime on identical inputs and the same artifact: "
                "a .NET fault.",
            )
        elif all(".batch." in c.id for c in ids):
            add("dotnet-ids", ".NET batching", "C# single rows pass but padded batches do not (padding or masking).")
        elif not numeric_failed:
            add(
                "dotnet-ids",
                ".NET decoding",
                "C# logits are within their bounds but its entities differ beyond the allowed exceptions: the C# "
                "decoder (gate, threshold, span to characters, overlap suppression or order) is at fault.",
            )
        elif export_failed:
            add(
                "dotnet-ids",
                "consequence of export",
                "C# misses the reference bound, consistent with the export-stage failure.",
            )
        elif vs_py in ("passed", "not-applicable") and _failed(checks, "dotnet.span_logits"):
            add(
                "dotnet-ids",
                "precision" if vs_py == "passed" else "unattributed",
                "C# misses the reference bound while Python ONNX Runtime passes; "
                + (
                    "C# is within the same-backend bound of Python ORT: accumulated error."
                    if vs_py == "passed"
                    else "no protected same-machine bound separates export from .NET (see the report-only "
                    "dotnet.span_logits.vs_python metrics)."
                ),
            )
        else:
            add(
                "dotnet-ids",
                "unattributed",
                "C# misses a reference bound, and without a same-machine export stage the export and .NET cannot be "
                "separated. Run stage 2 on the same machine.",
            )

    text = _real(checks, "dotnet-text")
    if text:
        if tok:
            add("dotnet-text", "consequence of processor", "Text path fails; the processor stage already fails.")
        elif ids:
            ids_tax = next((f["taxonomy"] for f in findings if f["stage"] == "dotnet-ids"), "")
            if ids_tax == "consequence of export":
                add(
                    "dotnet-text",
                    "consequence of export",
                    "Text path fails like the inputs path, which follows the export.",
                )
            else:
                add("dotnet-text", "consequence of .NET runtime", "Text path fails; the inputs path already fails.")
        elif status["tokenizer"] != "incomplete" and status["dotnet-ids"] != "incomplete":
            add(
                "dotnet-text",
                "processor",
                "Processor outputs are exact and the inputs path passes, but text to entities fails: the orchestration "
                "in the C# text path (normalization, splitting, schema prefix, positions, offsets) is at fault.",
            )
        else:
            add("dotnet-text", "unattributed", "Text path fails; earlier stages are incomplete.")
    return first, findings


# --------------------------------------------------------------------------------------------
# Markdown


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.3g}"
    if value is None:
        return ""
    if isinstance(value, dict):
        return ", ".join(f"{k.split('.')[-1]} {_fmt(v)}" for k, v in value.items())
    return str(value)


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Fidelity report: {report['profile']} ({report['task']})",
        "",
        f"Date {report['date']}. Evaluator {report['evaluator']['name']} {report['evaluator']['version']} at "
        f"`{report['evaluator']['repository_commit']}`. Exit code {report['exit_code']}.",
        "",
    ]
    if report["inputs"]["tolerances"].get("missing"):
        lines += [
            f"**No protected tolerances file** at `{report['inputs']['tolerances']['path']}`: every bound is "
            "unset, so no gate can pass. Metrics below are measurements only. Expected keys are listed under "
            "`expected_tolerance_keys` in report.json.",
            "",
        ]
    lines += [f"First divergence: **{report['first_divergence'] or 'none'}**.", ""]
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
        "Absolute errors; `Max rel` is |d| / max(1, |ref|). Bounds come only from the protected tolerances file.",
        "",
        "| Check | Tolerance key | Bound | Max abs | Mean abs | p99 abs | Max rel | Cases | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for c in report["checks"]:
        m = c["metrics"]
        lines.append(
            f"| `{c['id']}` | {c['tolerance_key'] or ''} | {_fmt(c['bound'])} | {_fmt(m.get('max_abs'))} | "
            f"{_fmt(m.get('mean_abs'))} | {_fmt(m.get('p99_abs'))} | {_fmt(m.get('max_rel'))} | "
            f"{_fmt(m.get('cases'))} | {c['status']} |"
        )
    details = [c for c in report["checks"] if c["detail"]]
    if details:
        lines += ["", "Details:", ""]
        lines += [f"- `{c['id']}`: {c['detail']}" for c in details]
    exc = report["entity_exceptions"]
    lines += ["", "## Allowed entity exceptions", ""]
    if not exc:
        lines.append("None: every compared entity list matched exactly, in order.")
    else:
        lines += [
            "| Check | Case | Kind | Label | Items | Reference scores | Delta or margin |",
            "| --- | --- | --- | --- | --- | --- | ---: |",
        ]
        for x in exc:
            if x["kind"] == "order-swap":
                items = f"{x['first']['text']!r} before {x['second']['text']!r}"
                scores = f"{x['first']['reference_score']:.9g}, {x['second']['reference_score']:.9g}"
                value = x["delta"]
            else:
                items = f"{x['text']!r} [{x['start']}, {x['end']}) {x['direction']}"
                scores = f"{x['reference_score']:.9g} (threshold {x['threshold']:.9g})"
                value = x["margin"]
            lines.append(
                f"| `{x['check']}` | {x['case']} | {x['kind']} | {x['label']} | {items} | {scores} | {value:.3g} |"
            )
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
