"""Evaluate an intervention profile over a strength sweep and write an evidence report (experimental).

    model-pipeline interventions evaluate <spec.yaml> --snapshot <model snapshot> --cache <cache root> --out <dir>
        [--provider cpu|cuda] [--device-id N] [--threads N]

The spec (``schemas/intervention-evaluation-spec.schema.json``, ``docs/interventions/evaluation.md``) names the
profile, the strength sweep, three prompt sets and the recommendation thresholds. The command

1. compiles (or reuses) the profile's derived graph and, for the control, a derived graph of a seeded random unit
   direction at the same sites (``cache.get_or_compile``);
2. for each arm (profile, control) and strength, on one ONNX Runtime session of that derived graph with the strength
   passed as the graph's strength input:

   - greedy-decodes every target prompt (the behavioural effect: generated tokens before ``<|im_end|>``, whether the
     answer ended before the cap, a keyword check and a repetition check);
   - greedy-decodes every capability item and scores it programmatically (exact match, last integer, format checks);
   - teacher-forces every neutral (prompt, response) text and records the next-token distributions at the response
     positions, to measure drift as KL(strength 0 || strength s) against strength 0 *of the same session*;

3. runs the generated answers, and the target set's reference answers, through the base graph's capture graph
   (``capture.get_or_compile_capture``) and projects their pooled residuals on the profile directions (the internal
   effect: where the produced text sits between the two reference styles in activation space);
4. on a non-CPU provider, greedy-decodes a few target prompts on the CPU provider too and compares the ids;
5. applies the spec's recommendation rules and writes ``evaluation-report.json``, ``evaluation-report.md`` (rendered
   from the JSON) and ``generations.jsonl``.

The decode loop is a Python ONNX Runtime loop over the derived graph (prefill, then one token per call, the caches fed
back), the loop whose greedy ids the LFM3 onboarding evidence found equal to the .NET implementation's on 19 of
19 cases.
Everything but the session is plain arrays, so the harness is tested without a model.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from .contract import (
    OPERATOR_DIRECTIONAL_ADDITION,
    OPERATOR_DIRECTIONAL_PROJECTION,
    load_profile,
    sha256_bytes,
    sha256_file,
)
from .extract import Encoder, Example, Tokenized, auc, cohens_d, pool, render_lfm3, tokenize

SPEC_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "intervention-evaluation-spec.schema.json"
EVALUATOR_VERSION = "0.1.0"

# What a strength does, per operator, as the report states it. The evaluator measures only the operators listed here:
# the control's ``matched_rms`` scaling compares projections removed or amplified along a direction, and the
# recommendation reads a strength's sign through this convention. ``directional_addition`` (profile schema 2) adds
# ``strength * scale`` times the direction whatever the residual's component, so neither carries over; it is rejected.
STRENGTH_SEMANTICS: dict[str, str] = {
    OPERATOR_DIRECTIONAL_PROJECTION: (
        "negative strengths amplify the component along the profile directions, positive strengths remove it "
        "(1 removes it fully where the scale is 1)"
    ),
}
REPORT_JSON = "evaluation-report.json"
REPORT_MD = "evaluation-report.md"
GENERATIONS = "generations.jsonl"
ARM_PROFILE = "profile"
ARM_CONTROL = "control"
LFM3_IM_END_ID = 124900

# A response is degenerate when one word n-gram of this length occurs at least this many times.
DEGENERATE_NGRAM = 6
DEGENERATE_REPEATS = 3


class EvaluationSpecError(ValueError):
    """The spec or a prompt set is invalid."""


# --------------------------------------------------------------------------------------------------------------------
# Spec and prompt sets


@dataclass(frozen=True)
class Recommendation:
    min_length_reduction: float
    max_quality_drop: float
    max_capability_drop: int
    max_mean_kl: float
    min_specificity: float


@dataclass(frozen=True)
class EvaluationSpec:
    path: Path
    sha256: str
    evaluation_id: str
    profile: Path
    model: str
    graph: str
    chat_template: str
    strengths: tuple[float, ...]
    max_new_tokens: int
    capability_max_new_tokens: int
    effect: str
    target_prompts: Path
    positive_label: str
    negative_label: str
    pooling: str
    neutral: Path
    capability: Path
    control_seed: int
    control_strengths: tuple[float, ...]
    control_scaling: str
    cross_check_prompts: int
    cross_check_strengths: tuple[float, ...]
    cross_check_max_new_tokens: int
    recommendation: Recommendation


def _schema() -> dict[str, Any]:
    return json.loads(SPEC_SCHEMA_PATH.read_text(encoding="utf-8"))


def _path(base: Path, relative: str) -> Path:
    return Path(os.path.normpath(base.absolute() / relative))


def parse_spec(data: bytes, path: Path) -> EvaluationSpec:
    import jsonschema

    try:
        document = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as e:
        raise EvaluationSpecError(f"{path}: not valid YAML: {e}") from e
    errors = sorted(jsonschema.Draft202012Validator(_schema()).iter_errors(document), key=lambda e: list(e.path))
    if errors:
        e = errors[0]
        where = "/".join(str(p) for p in e.path) or "(root)"
        raise EvaluationSpecError(f"{path}: {where}: {e.message}")
    base = path.parent
    g, t, c, x, r = (document[k] for k in ("generation", "target", "control", "cross_check", "recommendation"))
    if t["labels"]["positive"] == t["labels"]["negative"]:
        raise EvaluationSpecError(f"{path}: the two target labels must differ.")
    strengths = tuple(float(s) for s in document["strengths"])
    if len(set(strengths)) != len(strengths):
        raise EvaluationSpecError(f"{path}: duplicate strengths.")
    return EvaluationSpec(
        path=path.absolute(),
        sha256=sha256_bytes(data),
        evaluation_id=document["evaluation_id"],
        profile=_path(base, document["profile"]),
        model=document["model"],
        graph=document["graph"],
        chat_template=document["chat_template"],
        strengths=tuple(sorted(strengths)),
        max_new_tokens=g["max_new_tokens"],
        capability_max_new_tokens=g["capability_max_new_tokens"],
        effect=t["effect"],
        target_prompts=_path(base, t["prompts"]),
        positive_label=t["labels"]["positive"],
        negative_label=t["labels"]["negative"],
        pooling=t["pooling"],
        neutral=_path(base, document["neutral"]),
        capability=_path(base, document["capability"]),
        control_seed=c["seed"],
        control_strengths=tuple(sorted(float(s) for s in c["strengths"])),
        control_scaling=c["scaling"],
        cross_check_prompts=x["prompts"],
        cross_check_strengths=tuple(float(s) for s in x["strengths"]),
        cross_check_max_new_tokens=x["max_new_tokens"],
        recommendation=Recommendation(
            min_length_reduction=float(r["min_length_reduction"]),
            max_quality_drop=float(r["max_quality_drop"]),
            max_capability_drop=int(r["max_capability_drop"]),
            max_mean_kl=float(r["max_mean_kl"]),
            min_specificity=float(r["min_specificity"]),
        ),
    )


def load_spec(path: str | Path) -> EvaluationSpec:
    path = Path(path)
    if not path.is_file():
        raise EvaluationSpecError(f"spec '{path}' does not exist.")
    return parse_spec(path.read_bytes(), path)


@dataclass(frozen=True)
class TargetPrompt:
    id: str
    prompt: str
    keywords: tuple[tuple[str, ...], ...]  # every group must match; a group matches when any of its words occurs
    reference_positive: str
    reference_negative: str


@dataclass(frozen=True)
class CapabilityItem:
    id: str
    kind: str
    prompt: str
    check: Mapping[str, Any]


CHECK_TYPES = {
    "exact",
    "last_integer",
    "uppercase_contains",
    "bullets",
    "one_word",
    "json_object",
    "starts_with",
    "max_words",
}


def _jsonl(path: Path, keys: set[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EvaluationSpecError(f"prompt set '{path}' does not exist.")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            raise EvaluationSpecError(f"{path}:{number}: invalid JSON: {e}") from e
        if not isinstance(item, dict) or set(item) != keys:
            raise EvaluationSpecError(f"{path}:{number}: expected exactly the keys {', '.join(sorted(keys))}.")
        if (
            not isinstance(item["id"], str)
            or not item["id"]
            or not isinstance(item["prompt"], str)
            or not item["prompt"]
        ):
            raise EvaluationSpecError(f"{path}:{number}: id and prompt must be non-empty strings.")
        rows.append(item)
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise EvaluationSpecError(f"{path}: duplicate ids.")
    if not rows:
        raise EvaluationSpecError(f"{path}: empty prompt set.")
    return rows


def load_target(path: Path) -> list[TargetPrompt]:
    out = []
    for r in _jsonl(path, {"id", "prompt", "keywords", "reference_positive", "reference_negative"}):
        groups = r["keywords"]
        if (
            not isinstance(groups, list)
            or not groups
            or not all(isinstance(g, list) and g and all(isinstance(w, str) and w for w in g) for g in groups)
        ):
            raise EvaluationSpecError(f"{path}: '{r['id']}': keywords must be a non-empty list of non-empty lists.")
        for key in ("reference_positive", "reference_negative"):
            if not isinstance(r[key], str) or not r[key]:
                raise EvaluationSpecError(f"{path}: '{r['id']}': {key} must be a non-empty string.")
        out.append(
            TargetPrompt(
                r["id"],
                r["prompt"],
                tuple(tuple(w.lower() for w in g) for g in groups),
                r["reference_positive"],
                r["reference_negative"],
            )
        )
    return out


def load_neutral(path: Path) -> list[Example]:
    out = []
    for r in _jsonl(path, {"id", "prompt", "response"}):
        if not isinstance(r["response"], str) or not r["response"]:
            raise EvaluationSpecError(f"{path}: '{r['id']}': response must be a non-empty string.")
        out.append(Example(r["id"], r["prompt"], r["response"]))
    return out


def load_capability(path: Path) -> list[CapabilityItem]:
    out = []
    for r in _jsonl(path, {"id", "kind", "prompt", "check"}):
        check = r["check"]
        if not isinstance(r["kind"], str) or not r["kind"]:
            raise EvaluationSpecError(f"{path}: '{r['id']}': kind must be a non-empty string.")
        if not isinstance(check, dict) or check.get("type") not in CHECK_TYPES:
            raise EvaluationSpecError(f"{path}: '{r['id']}': check.type must be one of {sorted(CHECK_TYPES)}.")
        out.append(CapabilityItem(r["id"], r["kind"], r["prompt"], check))
    return out


# --------------------------------------------------------------------------------------------------------------------
# Scoring (plain text and arrays)

_FENCE = re.compile(r"^```[a-zA-Z0-9]*\n?|\n?```$")


def _clean(text: str) -> str:
    """Trim whitespace, a surrounding code fence or backticks, surrounding quotes and one trailing period."""
    t = _FENCE.sub("", text.strip()).strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "`'\"":
        t = t[1:-1].strip()
    if t.endswith(".") and not t.endswith(".."):
        t = t[:-1].rstrip()
    return t


def score_capability(check: Mapping[str, Any], text: str) -> bool:
    kind = check["type"]
    cleaned = _clean(text)
    if kind == "exact":
        return cleaned.lower() == str(check["value"]).lower()
    if kind == "last_integer":
        numbers = re.findall(r"-?\d+", re.sub(r"(?<=\d),(?=\d{3})", "", text))
        return bool(numbers) and int(numbers[-1]) == int(check["value"])
    if kind == "uppercase_contains":
        letters = [c for c in cleaned if c.isalpha()]
        return bool(letters) and all(c.isupper() for c in letters) and str(check["value"]) in cleaned
    if kind == "bullets":
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        return len(lines) == int(check["count"]) and all(line.startswith(("- ", "* ", "• ")) for line in lines)
    if kind == "one_word":
        return re.fullmatch(r"[A-Za-z-]+", cleaned) is not None and cleaned.lower() == str(check["value"]).lower()
    if kind == "json_object":
        try:
            value = json.loads(cleaned)
        except json.JSONDecodeError:
            return False
        return value == check["value"]
    if kind == "starts_with":
        return text.strip().startswith(str(check["value"]))
    if kind == "max_words":
        words = re.findall(r"[A-Za-z0-9']+", text)
        return 0 < len(words) <= int(check["value"])
    raise ValueError(f"unknown check type '{kind}'.")


def keyword_hits(groups: Sequence[Sequence[str]], text: str) -> bool:
    lower = text.lower()
    return all(any(word in lower for word in group) for group in groups)


def is_degenerate(text: str, n: int = DEGENERATE_NGRAM, repeats: int = DEGENERATE_REPEATS) -> bool:
    words = re.findall(r"\w+", text.lower())
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - n + 1):
        key = tuple(words[i : i + n])
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= repeats:
            return True
    return False


def log_softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def kl_per_position(base_logits: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """KL(base || other) in nats per row of ``[positions, vocab]`` logits, computed in float64."""
    lp, lq = log_softmax(base_logits), log_softmax(logits)
    return (np.exp(lp) * (lp - lq)).sum(axis=-1)


# --------------------------------------------------------------------------------------------------------------------
# The model under test


class LanguageModel(Protocol):
    """One session of a derived graph; the strength is passed on every call."""

    def generate(self, prompt_ids: Sequence[int], strength: float, max_new_tokens: int) -> tuple[list[int], bool]:
        """Greedy ids after the prompt, without the stop token, and whether the stop token came before the cap."""
        ...

    def logits(self, ids: Sequence[int], strength: float) -> np.ndarray:
        """``[len(ids), vocab]`` float32 next-token logits of one prefill."""
        ...


Decode = Callable[[Sequence[int]], str]
CaptureRunner = Callable[[Sequence[int]], Mapping[int, np.ndarray]]


@dataclass(frozen=True)
class Prepared:
    target: list[TargetPrompt]
    target_ids: list[list[int]]
    references: dict[str, list[Tokenized]]  # label -> one per target prompt
    neutral: list[Example]
    neutral_tokens: list[Tokenized]
    capability: list[CapabilityItem]
    capability_ids: list[list[int]]


def prepare(spec: EvaluationSpec, encoder: Encoder) -> Prepared:
    target = load_target(spec.target_prompts)
    neutral = load_neutral(spec.neutral)
    capability = load_capability(spec.capability)
    return Prepared(
        target=target,
        target_ids=[list(encoder(render_lfm3(t.prompt, "")[0])) for t in target],
        references={
            spec.positive_label: [tokenize(encoder, Example(t.id, t.prompt, t.reference_positive)) for t in target],
            spec.negative_label: [tokenize(encoder, Example(t.id, t.prompt, t.reference_negative)) for t in target],
        },
        neutral=neutral,
        neutral_tokens=[tokenize(encoder, e) for e in neutral],
        capability=capability,
        capability_ids=[list(encoder(render_lfm3(c.prompt, "")[0])) for c in capability],
    )


def run_arm(
    lm: LanguageModel,
    prepared: Prepared,
    strengths: Sequence[float],
    *,
    max_new_tokens: int,
    capability_max_new_tokens: int,
    decode: Decode,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Generations and drift of one derived graph over ``strengths`` (strength 0 is always run for the drift base)."""
    base = [lm.logits(t.ids, 0.0)[t.response_start - 1 : t.response_end] for t in prepared.neutral_tokens]
    rows: list[dict[str, Any]] = []
    drift: dict[float, list[float]] = {}
    for s in strengths:
        for t, ids in zip(prepared.target, prepared.target_ids, strict=True):
            out, finished = lm.generate(ids, s, max_new_tokens)
            text = decode(out)
            rows.append(
                {
                    "set": "target",
                    "id": t.id,
                    "strength": s,
                    "ids": out,
                    "tokens": len(out),
                    "finished": finished,
                    "text": text,
                    "keywords": keyword_hits(t.keywords, text),
                    "degenerate": is_degenerate(text),
                }
            )
        for c, ids in zip(prepared.capability, prepared.capability_ids, strict=True):
            out, finished = lm.generate(ids, s, capability_max_new_tokens)
            text = decode(out)
            rows.append(
                {
                    "set": "capability",
                    "id": c.id,
                    "kind": c.kind,
                    "strength": s,
                    "tokens": len(out),
                    "finished": finished,
                    "text": text,
                    "correct": score_capability(c.check, text),
                }
            )
        drift[s] = []
        for t, b in zip(prepared.neutral_tokens, base, strict=True):
            other = b if s == 0 else lm.logits(t.ids, s)[t.response_start - 1 : t.response_end]
            drift[s].append(float(kl_per_position(b, other).mean()))
        target_tokens = [r["tokens"] for r in rows if r["set"] == "target" and r["strength"] == s]
        log(
            f"strength {s:g}: median {float(np.median(target_tokens)):g} target tokens, "
            f"mean neutral KL {float(np.mean(drift[s])):.4g}"
        )
    return {"rows": rows, "drift": drift}


def projections(
    runner: CaptureRunner, tokens: Sequence[Tokenized], directions: Mapping[int, np.ndarray], pooling: str
) -> np.ndarray:
    """``[examples, layers]`` projection of each pooled residual on direction 1 of each layer (layers sorted)."""
    layers = sorted(directions)
    out = np.zeros((len(tokens), len(layers)))
    for i, token in enumerate(tokens):
        acts = runner(token.ids)
        for j, layer in enumerate(layers):
            out[i, j] = float(pool(np.asarray(acts[layer]), token, pooling) @ directions[layer][0].astype(np.float64))
    return out


def token_rms(
    runner: CaptureRunner, tokens: Sequence[Tokenized], directions: Mapping[int, np.ndarray]
) -> dict[int, np.ndarray]:
    """Per layer, ``[rank]`` root mean square of the per-token projection on each direction over all response tokens."""
    sums = {layer: np.zeros(directions[layer].shape[0]) for layer in directions}
    count = 0
    for token in tokens:
        acts = runner(token.ids)
        for layer, q in directions.items():
            y = np.asarray(acts[layer])[token.response_start : token.response_end].astype(np.float64)
            sums[layer] += ((y @ q.astype(np.float64).T) ** 2).sum(axis=0)
        count += token.response_tokens
    if count == 0:
        raise ValueError("no response tokens.")
    return {layer: np.sqrt(v / count) for layer, v in sums.items()}


def generated_tokens(prompt_ids: Sequence[int], out: Sequence[int], end_id: int = LFM3_IM_END_ID) -> Tokenized:
    """The prompt, the generated ids and ``<|im_end|>`` (appended even when the answer hit the cap)."""
    return Tokenized(tuple(prompt_ids) + tuple(out) + (end_id,), len(prompt_ids), len(prompt_ids) + len(out))


# --------------------------------------------------------------------------------------------------------------------
# Summary and recommendation


def _r(value: float) -> float:
    return float(f"{value:.6g}") if math.isfinite(value) else value


def summarize_arm(arm: Mapping[str, Any], cap: int) -> dict[float, dict[str, Any]]:
    rows = arm["rows"]
    out: dict[float, dict[str, Any]] = {}
    for s in sorted(arm["drift"]):
        target = [r for r in rows if r["set"] == "target" and r["strength"] == s]
        capability = [r for r in rows if r["set"] == "capability" and r["strength"] == s]
        tokens = np.asarray([r["tokens"] for r in target])
        finished = sum(r["finished"] for r in target)
        quality = [r["keywords"] and not r["degenerate"] for r in target]
        kinds = sorted({r["kind"] for r in capability})
        kl = np.asarray(arm["drift"][s])
        out[s] = {
            "strength": s,
            "target": {
                "prompts": len(target),
                "median_tokens": _r(float(np.median(tokens))),
                "mean_tokens": _r(float(tokens.mean())),
                "median_censored": bool(np.median(tokens) >= cap),
                "finished": int(finished),
                "keywords": int(sum(r["keywords"] for r in target)),
                "degenerate": int(sum(r["degenerate"] for r in target)),
                "quality_pass": int(sum(quality)),
                "quality_pass_rate": _r(sum(quality) / len(target)),
            },
            "capability": {
                "items": len(capability),
                "correct": int(sum(r["correct"] for r in capability)),
                "by_kind": {
                    k: [
                        int(sum(r["correct"] for r in capability if r["kind"] == k)),
                        sum(1 for r in capability if r["kind"] == k),
                    ]
                    for k in kinds
                },
            },
            "drift": {"mean_kl": _r(float(kl.mean())), "max_kl": _r(float(kl.max())), "texts": len(kl)},
        }
    return out


def _paired_shorter(arm: Mapping[str, Any], s: float) -> tuple[int, int]:
    rows = [r for r in arm["rows"] if r["set"] == "target"]
    base = {r["id"]: r["tokens"] for r in rows if r["strength"] == 0}
    at = {r["id"]: r["tokens"] for r in rows if r["strength"] == s}
    return sum(at[i] < base[i] for i in at), sum(at[i] > base[i] for i in at)


def add_effects(summary: dict[float, dict[str, Any]], arm: Mapping[str, Any]) -> None:
    base = summary[0.0]["target"]["median_tokens"]
    for s, entry in summary.items():
        shorter, longer = _paired_shorter(arm, s)
        entry["target"]["length_reduction"] = _r(1 - entry["target"]["median_tokens"] / base) if base else 0.0
        entry["target"]["shorter_than_strength_0"] = shorter
        entry["target"]["longer_than_strength_0"] = longer


def recommend(
    rule: Recommendation, profile: Mapping[float, Mapping[str, Any]], control: Mapping[float, Mapping[str, Any]]
) -> dict[str, Any]:
    base = profile[0.0]
    per: list[dict[str, Any]] = []
    for s, e in profile.items():
        if s == 0:
            continue
        t, b = e["target"], base["target"]
        c = control.get(s)
        specificity = None if c is None else _r(t["length_reduction"] - c["target"]["length_reduction"])
        checks = {
            "length_reduction": t["length_reduction"] >= rule.min_length_reduction,
            "finished": t["finished"] >= b["finished"],
            "quality": t["quality_pass_rate"] >= b["quality_pass_rate"] - rule.max_quality_drop - 1e-9,
            "capability": e["capability"]["correct"] >= base["capability"]["correct"] - rule.max_capability_drop,
            "drift": e["drift"]["mean_kl"] <= rule.max_mean_kl,
            "specificity": specificity is not None and specificity >= rule.min_specificity,
        }
        per.append({"strength": s, "checks": checks, "specificity": specificity, "recommended": all(checks.values())})
    chosen = [p["strength"] for p in per if p["recommended"]]
    sweep = sorted(profile)
    contiguous = (
        bool(chosen)
        and all(s in chosen for s in sweep if min(chosen) <= s <= max(chosen) and s != 0)
        and not (min(chosen) < 0 < max(chosen))
    )
    return {
        "rules": {k: getattr(rule, k) for k in rule.__dataclass_fields__},
        "per_strength": per,
        "recommended_strengths": chosen,
        "recommended_range": [min(chosen), max(chosen)] if contiguous else None,
    }


# --------------------------------------------------------------------------------------------------------------------
# Report


def _relative(path: Path, root: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), root.resolve()).replace(os.sep, "/")
    except ValueError:
        return path.name


def build_report(
    spec: EvaluationSpec,
    prepared: Prepared,
    arms: Mapping[str, Mapping[str, Any]],
    internal: Mapping[str, Any],
    cross_check: Mapping[str, Any] | None,
    *,
    out_dir: Path,
    profile_sha256: str,
    profile_id: str,
    layers: Sequence[int],
    environment: Mapping[str, Any],
    operator: str = OPERATOR_DIRECTIONAL_PROJECTION,
) -> dict[str, Any]:
    summaries = {}
    for name, arm in arms.items():
        summary = summarize_arm(arm, spec.max_new_tokens)
        add_effects(summary, arm)
        summaries[name] = summary
    control = summaries.get(ARM_CONTROL, {})
    rec = recommend(spec.recommendation, summaries[ARM_PROFILE], control)
    return {
        "evaluator_version": EVALUATOR_VERSION,
        "evaluation_id": spec.evaluation_id,
        "spec": {"path": _relative(spec.path, out_dir), "sha256": spec.sha256},
        "profile": {
            "id": profile_id,
            "path": _relative(spec.profile, out_dir),
            "sha256": profile_sha256,
            "layers": list(layers),
            "operator": operator,
        },
        "prompt_sets": {
            name: {"path": _relative(p, out_dir), "sha256": sha256_file(p), "items": n}
            for name, p, n in (
                ("target", spec.target_prompts, len(prepared.target)),
                ("neutral", spec.neutral, len(prepared.neutral)),
                ("capability", spec.capability, len(prepared.capability)),
            )
        },
        "method": {
            "effect": spec.effect,
            "labels": {"positive": spec.positive_label, "negative": spec.negative_label},
            "pooling": spec.pooling,
            "decoding": "greedy",
            "max_new_tokens": spec.max_new_tokens,
            "capability_max_new_tokens": spec.capability_max_new_tokens,
            "control_seed": spec.control_seed,
            "control_scaling": spec.control_scaling,
            "degenerate": {"ngram": DEGENERATE_NGRAM, "repeats": DEGENERATE_REPEATS},
        },
        "arms": {name: [summary[s] for s in sorted(summary)] for name, summary in summaries.items()},
        "internal": dict(internal),
        "recommendation": rec,
        "cross_check": None if cross_check is None else dict(cross_check),
        "environment": dict(environment),
    }


def summarize_internal(
    spec: EvaluationSpec,
    layers: Sequence[int],
    refs: Mapping[str, np.ndarray],
    generated: Mapping[str, Mapping[float, np.ndarray]],
) -> dict[str, Any]:
    """Reference separability per layer and, per arm and strength, the mean projection of the generated answers.

    ``position`` maps a projection onto the reference scale per layer (0 = the negative references' mean, 1 = the
    positive references' mean) and averages over layers.
    """
    pos, neg = refs[spec.positive_label], refs[spec.negative_label]
    mp, mn = pos.mean(axis=0), neg.mean(axis=0)
    per_layer = [
        {
            "layer": layer,
            "auc": _r(auc(pos[:, j], neg[:, j])),
            "cohens_d": _r(cohens_d(pos[:, j], neg[:, j])),
            f"mean_{spec.positive_label}": _r(float(mp[j])),
            f"mean_{spec.negative_label}": _r(float(mn[j])),
        }
        for j, layer in enumerate(layers)
    ]

    def position(values: np.ndarray) -> float:
        return float(np.mean((values.mean(axis=0) - mn) / (mp - mn)))

    return {
        "references": per_layer,
        "reference_position": {spec.positive_label: _r(position(pos)), spec.negative_label: _r(position(neg))},
        "generated": {
            arm: [
                {
                    "strength": s,
                    "position": _r(position(v)),
                    "mean_projection": [_r(float(x)) for x in v.mean(axis=0)],
                    "answers": int(v.shape[0]),
                }
                for s, v in sorted(by_strength.items())
            ]
            for arm, by_strength in generated.items()
        },
    }


def _fmt(value: float) -> str:
    return f"{value:g}"


def strength_semantics(operator: str) -> str:
    """The report's sentence on what a strength does under ``operator``; unsupported operators raise."""
    try:
        return STRENGTH_SEMANTICS[operator]
    except KeyError:
        raise EvaluationSpecError(
            f"the evaluator has no strength convention for the '{operator}' operator; it evaluates "
            + ", ".join(f"'{o}'" for o in sorted(STRENGTH_SEMANTICS))
            + " profiles only."
        ) from None


def render_report_markdown(report: Mapping[str, Any]) -> str:
    m, rec = report["method"], report["recommendation"]
    # Reports written before the operator was recorded evaluated directional_projection profiles only.
    semantics = strength_semantics(report["profile"].get("operator", OPERATOR_DIRECTIONAL_PROJECTION))
    labels = m["labels"]
    cap = m["max_new_tokens"]
    lines = [
        f"# Evaluation report: `{report['profile']['id']}`",
        "",
        "Status: **experimental**. Generated by `model-pipeline interventions evaluate` "
        f"(evaluator {report['evaluator_version']}); `{REPORT_JSON}` holds the same numbers and `{GENERATIONS}` every "
        f"answer. Strength 0 is the original model; {semantics}.",
        "",
        "| Input | Value |",
        "| --- | --- |",
        f"| Spec | `{report['spec']['path']}` (`{report['spec']['sha256'][:12]}…`) |",
        f"| Profile | `{report['profile']['path']}` (`{report['profile']['sha256'][:12]}…`), layers "
        f"{report['profile']['layers']} |",
    ]
    for name, p in report["prompt_sets"].items():
        lines.append(f"| {name.capitalize()} set | `{p['path']}` ({p['items']} items, `{p['sha256'][:12]}…`) |")
    lines += [
        f"| Decoding | greedy, at most {cap} new tokens (target), {m['capability_max_new_tokens']} (capability) |",
        f"| Control | a seeded random unit direction per site (seed {m['control_seed']}), same layers; scales "
        + (
            "matched so the per-token RMS projection on the reference answers equals the profile's (`matched_rms`)"
            if m["control_scaling"] == "matched_rms"
            else "the profile's (`profile`)"
        )
        + " |",
    ]
    for key, value in report["environment"].items():
        lines.append(f"| {key} | {value} |")

    lines += [
        "",
        "## Target effect: behaviour",
        "",
        f'Generated tokens before `<|im_end|>` on the target prompts. "Ended" counts answers that stopped before the '
        f"{cap}-token cap; a capped answer's length is a lower bound, and a median marked ≥ is censored. Quality: the "
        f"answer contains every keyword group of its prompt and has no word {m['degenerate']['ngram']}-gram repeated "
        f"{m['degenerate']['repeats']} or more times. Shorter / longer: prompts whose answer is shorter / longer than "
        "at strength 0.",
        "",
    ]
    for arm, entries in report["arms"].items():
        lines += [
            f"**{arm.capitalize()}**",
            "",
            "| Strength | Median tokens | Mean tokens | Reduction | Ended | Shorter / longer | Quality | Keywords | "
            "Degenerate |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for e in entries:
            t = e["target"]
            median = ("≥" if t["median_censored"] else "") + _fmt(t["median_tokens"])
            lines.append(
                f"| {_fmt(e['strength'])} | {median} | {t['mean_tokens']:.1f} | {t['length_reduction']:+.2f} | "
                f"{t['finished']}/{t['prompts']} | {t['shorter_than_strength_0']} / {t['longer_than_strength_0']} | "
                f"{t['quality_pass']}/{t['prompts']} | {t['keywords']}/{t['prompts']} | {t['degenerate']} |"
            )
        lines.append("")

    internal = report["internal"]
    lines += [
        "## Target effect: internal",
        "",
        f"Each answer (and each reference answer of the target set) is run through the *base* graph's capture graph, "
        f"`{m['pooling']}`-pooled at each profile layer and projected on the layer's direction. Position: 0 is the "
        f"mean of the `{labels['negative']}` references, 1 the mean of the `{labels['positive']}` references, per "
        "layer, averaged over layers. At the intervention sites themselves the projection after the operator is "
        "(1 − strength·scale) times the projection before it by construction, so this measures the produced text, "
        "not the operator.",
        "",
        f"| Layer | Reference AUC ({labels['positive']} vs {labels['negative']}) | Cohen's d | "
        f"Mean {labels['positive']} | Mean {labels['negative']} |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in internal["references"]:
        lines.append(
            f"| {r['layer']} | {r['auc']:.4f} | {r['cohens_d']:.3f} | {r['mean_' + labels['positive']]:.4g} | "
            f"{r['mean_' + labels['negative']]:.4g} |"
        )
    lines += ["", "| Arm | Strength | Position | Mean projection per layer |", "| --- | --- | --- | --- |"]
    for arm, entries in internal["generated"].items():
        for e in entries:
            lines.append(
                f"| {arm} | {_fmt(e['strength'])} | {e['position']:.3f} | "
                f"{', '.join(f'{x:.4g}' for x in e['mean_projection'])} |"
            )

    lines += [
        "",
        "## Drift and capability",
        "",
        "Drift: KL(strength 0 ‖ strength s) of the next-token distributions at the response positions of the "
        "teacher-forced neutral texts (nats), mean per text, then mean and max over texts; strength 0 comes from the "
        "same session. Capability: items answered correctly (exact match, last integer or format checks).",
        "",
        "| Arm | Strength | Mean KL | Max KL | Capability | By kind |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm, entries in report["arms"].items():
        for e in entries:
            c, d = e["capability"], e["drift"]
            kinds = ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in c["by_kind"].items())
            lines.append(
                f"| {arm} | {_fmt(e['strength'])} | {d['mean_kl']:.4g} | {d['max_kl']:.4g} | "
                f"{c['correct']}/{c['items']} | {kinds} |"
            )

    rules = rec["rules"]
    lines += [
        "",
        "## Recommendation",
        "",
        "Rules (fixed in the spec before the run), each against strength 0 of the profile arm: length reduction of "
        f"the median ≥ {rules['min_length_reduction']}; no fewer answers ending before the cap; quality pass rate "
        f"drop ≤ {rules['max_quality_drop']}; capability drop ≤ {rules['max_capability_drop']} items; mean neutral "
        f"KL ≤ {rules['max_mean_kl']}; specificity (profile reduction minus control reduction at the same strength) "
        f"≥ {rules['min_specificity']}, which needs a control run at that strength.",
        "",
        "| Strength | Reduction | Ended | Quality | Capability | Drift | Specificity | Recommended |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for p in rec["per_strength"]:
        c = p["checks"]
        spec_text = "not run" if p["specificity"] is None else f"{p['specificity']:+.2f}"
        yes = "yes" if p["recommended"] else "no"
        lines.append(
            f"| {_fmt(p['strength'])} | {_ok(c['length_reduction'])} | {_ok(c['finished'])} | {_ok(c['quality'])} | "
            f"{_ok(c['capability'])} | {_ok(c['drift'])} | {spec_text} {_ok(c['specificity'])} | {yes} |"
        )
    lines.append("")
    if rec["recommended_range"] is not None:
        lo, hi = rec["recommended_range"]
        lines.append(
            f"Recommended strength range: **{_fmt(lo)} to {_fmt(hi)}** (strengths of this sweep that meet "
            "every rule; values between them were interpolated by no run)."
        )
    elif rec["recommended_strengths"]:
        lines.append(
            "Strengths meeting every rule: "
            + ", ".join(_fmt(s) for s in rec["recommended_strengths"])
            + " (not a contiguous range of the sweep)."
        )
    else:
        lines.append("No strength of the sweep meets every rule; no strength is recommended.")

    x = report["cross_check"]
    if x is not None:
        lines += [
            "",
            "## CPU cross-check",
            "",
            f"Greedy ids of the first {x['max_new_tokens']} new tokens on the CPU provider against the main run.",
            "",
            "| Prompt | Strength | Equal ids | First difference |",
            "| --- | --- | --- | --- |",
        ]
        for c in x["cases"]:
            first = "" if c["first_difference"] is None else str(c["first_difference"])
            lines.append(
                f"| {c['id']} | {_fmt(c['strength'])} | {c['compared']} {'yes' if c['equal'] else 'no'} | {first} |"
            )
    return "\n".join(lines) + "\n"


def _ok(flag: bool) -> str:
    return "pass" if flag else "fail"


def write_outputs(out_dir: Path, report: Mapping[str, Any], arms: Mapping[str, Mapping[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / REPORT_JSON).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / REPORT_MD).write_text(render_report_markdown(report), encoding="utf-8")
    with (out_dir / GENERATIONS).open("w", encoding="utf-8") as f:
        for name, arm in arms.items():
            for r in arm["rows"]:
                row = {"arm": name} | {k: v for k, v in r.items() if k != "ids"}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compare_ids(main: Sequence[int], other: Sequence[int], limit: int) -> dict[str, Any]:
    a, b = list(main)[:limit], list(other)[:limit]
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), None)
    if first is None and len(a) != len(b):
        first = n
    return {"compared": n, "equal": first is None, "first_difference": first}


def evaluate_with(
    spec: EvaluationSpec,
    prepared: Prepared,
    *,
    open_model: Callable[[str], LanguageModel],
    open_capture: Callable[[], CaptureRunner],
    directions: Mapping[int, np.ndarray],
    scales: Mapping[int, np.ndarray],
    control_directions: Mapping[int, np.ndarray],
    prepare_control: Callable[[Mapping[int, np.ndarray]], None],
    decode: Decode,
    open_cpu_model: Callable[[], LanguageModel] | None,
    out_dir: Path,
    profile_sha256: str,
    profile_id: str,
    environment: Mapping[str, Any],
    operator: str = OPERATOR_DIRECTIONAL_PROJECTION,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """The whole evaluation with injected sessions; one session is open at a time.

    ``prepare_control(scales)`` is called with the control's per-site scales before the control arm's model opens.
    """
    strength_semantics(operator)  # an operator the report cannot describe is rejected before any session opens
    layers = sorted(directions)
    runner = open_capture()
    refs = {label: projections(runner, toks, directions, spec.pooling) for label, toks in prepared.references.items()}
    reference_tokens = [t for toks in prepared.references.values() for t in toks]
    rms_profile = token_rms(runner, reference_tokens, directions)
    rms_control = token_rms(runner, reference_tokens, control_directions)
    del runner
    gc.collect()
    control_scales = {
        layer: np.asarray(scales[layer], dtype=np.float64) * (rms_profile[layer] / rms_control[layer])
        if spec.control_scaling == "matched_rms"
        else np.asarray(scales[layer], dtype=np.float64)
        for layer in layers
    }
    prepare_control(control_scales)

    arms: dict[str, dict[str, Any]] = {}
    for name, strengths in ((ARM_PROFILE, spec.strengths), (ARM_CONTROL, spec.control_strengths)):
        log(f"arm {name}: strengths {list(strengths)}")
        lm = open_model(name)
        arms[name] = run_arm(
            lm,
            prepared,
            strengths,
            max_new_tokens=spec.max_new_tokens,
            capability_max_new_tokens=spec.capability_max_new_tokens,
            decode=decode,
            log=log,
        )
        del lm
        gc.collect()

    runner = open_capture()
    generated: dict[str, dict[float, np.ndarray]] = {}
    for name, arm in arms.items():
        generated[name] = {}
        for s in sorted(arm["drift"]):
            toks = [
                generated_tokens(prepared.target_ids[i], r["ids"])
                for i, r in enumerate(r for r in arm["rows"] if r["set"] == "target" and r["strength"] == s)
                if r["ids"]
            ]
            generated[name][s] = projections(runner, toks, directions, spec.pooling)
    del runner
    gc.collect()
    internal = summarize_internal(spec, layers, refs, generated)
    internal["control"] = {
        "scaling": spec.control_scaling,
        "sites": [
            {
                "layer": layer,
                "rms_projection_profile": [_r(float(x)) for x in rms_profile[layer]],
                "rms_projection_control": [_r(float(x)) for x in rms_control[layer]],
                "control_scales": [_r(float(x)) for x in control_scales[layer]],
            }
            for layer in layers
        ],
    }

    cross = None
    if open_cpu_model is not None and spec.cross_check_prompts and spec.cross_check_strengths:
        lm = open_cpu_model()
        cases = []
        main = {(r["id"], r["strength"]): r["ids"] for r in arms[ARM_PROFILE]["rows"] if r["set"] == "target"}
        for t, ids in list(zip(prepared.target, prepared.target_ids, strict=True))[: spec.cross_check_prompts]:
            for s in spec.cross_check_strengths:
                if (t.id, s) not in main:
                    continue
                out, _ = lm.generate(ids, s, spec.cross_check_max_new_tokens)
                cases.append(
                    {"id": t.id, "strength": s} | compare_ids(main[(t.id, s)], out, spec.cross_check_max_new_tokens)
                )
        del lm
        gc.collect()
        cross = {"max_new_tokens": spec.cross_check_max_new_tokens, "cases": cases}

    report = build_report(
        spec,
        prepared,
        arms,
        internal,
        cross,
        out_dir=out_dir,
        profile_sha256=profile_sha256,
        profile_id=profile_id,
        layers=layers,
        environment=environment,
        operator=operator,
    )
    write_outputs(out_dir, report, arms)
    return report


# --------------------------------------------------------------------------------------------------------------------
# The real model (LFM3, ONNX Runtime)


def _providers(provider: str, device_id: int) -> list[Any]:
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    if provider == "cuda":
        return [("CUDAExecutionProvider", {"device_id": device_id, "use_tf32": 0}), "CPUExecutionProvider"]
    raise ValueError(f"unknown provider '{provider}'.")


def _session(graph_path: Path, provider: str, device_id: int, threads: int | None) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
    return ort.InferenceSession(str(graph_path), options, providers=_providers(provider, device_id))


class Lfm3OrtModel:
    """Greedy decoding and prefill logits on one session of an LFM3 derived graph (one fp32 session at a time).

    The loop is the one ``Lfm3Sequence`` runs: a prefill call with empty caches, then one call per token with the
    previous call's ``present*`` outputs fed back as ``past*`` inputs, positions counting on, ``logits_to_keep`` 1,
    and the strength as a ``[1]`` float32 on every call. It stops at ``<|im_end|>``.
    """

    def __init__(
        self,
        graph_path: Path,
        strength_input: str,
        *,
        provider: str = "cpu",
        device_id: int = 0,
        threads: int | None = None,
        stop_ids: Sequence[int] = (LFM3_IM_END_ID,),
    ) -> None:
        self._session = _session(graph_path, provider, device_id, threads)
        self._inputs = self._session.get_inputs()
        self._outputs = [o.name for o in self._session.get_outputs()]
        self._strength = strength_input
        if strength_input not in {i.name for i in self._inputs}:
            raise ValueError(f"the graph has no input '{strength_input}'.")
        self._stop = set(stop_ids)

    def _empty_cache(self) -> dict[str, np.ndarray]:
        feeds: dict[str, np.ndarray] = {}
        for item in self._inputs:
            if item.name.startswith("past_conv."):
                feeds[item.name] = np.zeros((1, item.shape[1], item.shape[2]), dtype=np.float32)
            elif item.name.startswith("past_key_values."):
                feeds[item.name] = np.zeros((1, item.shape[1], 0, item.shape[3]), dtype=np.float32)
        return feeds

    def _run(
        self,
        feeds: dict[str, np.ndarray],
        ids: Sequence[int],
        start: int,
        keep: int,
        strength: float,
        names: Sequence[str],
    ) -> list[np.ndarray]:
        n = len(ids)
        feeds["input_ids"] = np.asarray([ids], dtype=np.int64)
        feeds["position_ids"] = np.arange(start, start + n, dtype=np.int64)[None, :]
        feeds["attention_mask"] = np.ones((1, start + n), dtype=np.int64)
        feeds["logits_to_keep"] = np.asarray([keep], dtype=np.int64)
        feeds[self._strength] = np.asarray([strength], dtype=np.float32)
        return self._session.run(list(names), feeds)

    def generate(self, prompt_ids: Sequence[int], strength: float, max_new_tokens: int) -> tuple[list[int], bool]:
        feeds = self._empty_cache()
        current, total, out = list(prompt_ids), 0, []
        for _ in range(max_new_tokens):
            values = dict(zip(self._outputs, self._run(feeds, current, total, 1, strength, self._outputs), strict=True))
            total += len(current)
            token = int(np.argmax(values["logits"][0, -1]))
            if token in self._stop:
                return out, True
            out.append(token)
            current = [token]
            for name, value in values.items():
                if name.startswith("present_conv."):
                    feeds["past_conv." + name.split(".", 1)[1]] = value
                elif name.startswith("present."):
                    feeds["past_key_values." + name.split(".", 1)[1]] = value
        return out, False

    def logits(self, ids: Sequence[int], strength: float) -> np.ndarray:
        return self._run(self._empty_cache(), ids, 0, 0, strength, ["logits"])[0][0]


class Lfm3OrtCapture:
    """Prefill runs of an LFM3 capture graph on any provider; returns ``{layer: [seq, hidden]}``."""

    def __init__(
        self,
        graph_path: Path,
        outputs: Mapping[int, str],
        *,
        provider: str = "cpu",
        device_id: int = 0,
        threads: int | None = None,
    ) -> None:
        self._session = _session(graph_path, provider, device_id, threads)
        self._outputs = dict(outputs)
        self._inputs = self._session.get_inputs()

    def __call__(self, ids: Sequence[int]) -> dict[int, np.ndarray]:
        n = len(ids)
        feeds: dict[str, np.ndarray] = {
            "input_ids": np.asarray([ids], dtype=np.int64),
            "attention_mask": np.ones((1, n), dtype=np.int64),
            "position_ids": np.arange(n, dtype=np.int64)[None, :],
            "logits_to_keep": np.asarray([1], dtype=np.int64),
        }
        for item in self._inputs:
            if item.name.startswith("past_conv."):
                feeds[item.name] = np.zeros((1, item.shape[1], item.shape[2]), dtype=np.float32)
            elif item.name.startswith("past_key_values."):
                feeds[item.name] = np.zeros((1, item.shape[1], 0, item.shape[3]), dtype=np.float32)
        names = list(self._outputs.values())
        by_name = dict(zip(names, self._session.run(names, feeds), strict=True))
        return {layer: by_name[name][0] for layer, name in self._outputs.items()}


def check_supported_profile(profile: Any) -> None:
    """Reject profiles the evaluator does not measure correctly.

    The control, the report's strength convention (``STRENGTH_SEMANTICS``) and the capture of a single target assume
    an operator listed there (``directional_projection``), one site per layer and one target for every site.
    """
    if profile.operator == OPERATOR_DIRECTIONAL_ADDITION:
        raise EvaluationSpecError(
            f"the evaluator does not support '{OPERATOR_DIRECTIONAL_ADDITION}' profiles: a positive strength adds the "
            "direction and a negative one subtracts it, while the control's matched_rms scaling and the report's "
            f"strength convention are written for '{OPERATOR_DIRECTIONAL_PROJECTION}' (negative amplifies, positive "
            "removes)."
        )
    strength_semantics(profile.operator)
    if len({s.layer for s in profile.sites}) != len(profile.sites):
        raise EvaluationSpecError("the evaluator supports one site per layer.")
    if len({s.target for s in profile.sites}) != 1:
        raise EvaluationSpecError("the evaluator supports profiles whose sites share one target.")


def control_directions(profile_file: Any, seed: int) -> dict[int, np.ndarray]:
    """The control's directions: ``random_orthonormal(rank, hidden, seed + i)`` for the i-th site by (layer, target)."""
    from .synthetic import random_orthonormal

    sites = sorted(profile_file.profile.sites, key=lambda s: (s.layer, s.target))
    return {s.layer: random_orthonormal(s.rank, s.hidden_size, seed + i) for i, s in enumerate(sites)}


def write_control_profile(directory: Path, profile_file: Any, seed: int, scales: Mapping[int, Sequence[float]]) -> Any:
    """A profile with the control's directions (``control_directions``) at the sites of ``profile_file``."""
    from .synthetic import SyntheticSite, write_synthetic_profile

    p = profile_file.profile
    return write_synthetic_profile(
        directory,
        profile_id=f"{p.id}-control-{seed}",
        base_model_sha256=p.base_model_sha256,
        sites=[
            SyntheticSite(s.layer, s.hidden_size, s.rank, [float(x) for x in scales[s.layer]], s.target)
            for s in p.sites
        ],
        seed=seed,
        description=f"Control for {p.id}: seeded random orthonormal directions at the same sites; no meaning.",
    )


def run_evaluation(
    spec: EvaluationSpec,
    *,
    snapshot: Path,
    cache_root: Path,
    out_dir: Path,
    provider: str = "cpu",
    device_id: int = 0,
    threads: int | None = None,
    external_sha256: Mapping[str, str] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    import shutil
    import tempfile

    import onnxruntime as ort
    from tokenizers import Tokenizer

    from .cache import get_or_compile
    from .capture import get_or_compile_capture

    profile_file = load_profile(spec.profile)
    check_supported_profile(profile_file.profile)
    graph_sha = sha256_bytes(snapshot.joinpath(*spec.graph.split("/")).read_bytes())
    profile_file.verify_base_model(graph_sha)
    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    prepared = prepare(spec, lambda text: tokenizer.encode(text, add_special_tokens=False).ids)

    def compile_entry(profile: Any) -> Any:
        entry = get_or_compile(
            cache_root=cache_root,
            model=spec.model,
            base_dir=snapshot,
            graph=spec.graph,
            profile=profile,
            external_sha256=external_sha256,
        )
        log(f"graph of {entry.profile.profile.id}: {entry.directory} ({'compiled' if entry.created else 'reused'})")
        return entry

    entries = {ARM_PROFILE: compile_entry(profile_file)}
    layers = sorted(s.layer for s in profile_file.profile.sites)
    target = profile_file.profile.sites[0].target
    capture = get_or_compile_capture(
        cache_root=cache_root,
        model=spec.model,
        base_dir=snapshot,
        graph=spec.graph,
        layers=layers,
        target=target,
        external_sha256=external_sha256,
    )
    directions = {s.layer: profile_file.load_directions(s) for s in profile_file.profile.sites}
    scales = {s.layer: np.asarray(s.scales, dtype=np.float64) for s in profile_file.profile.sites}
    environment: dict[str, Any] = {
        "Runtime": f"onnxruntime {ort.__version__}, "
        + ("CPUExecutionProvider" if provider == "cpu" else f"CUDAExecutionProvider (device {device_id}, TF32 off)")
        + ", fp32 graph",
        "Profile graph": f"`{entries[ARM_PROFILE].key[:12]}…` (site mapping "
        f"`{entries[ARM_PROFILE].manifest.site_mapping_version}`)",
    }
    control_dir = Path(tempfile.mkdtemp(prefix="starling-control-profile-"))

    def prepare_control(control_scales: Mapping[int, np.ndarray]) -> None:
        control = write_control_profile(control_dir, profile_file, spec.control_seed, control_scales)
        entries[ARM_CONTROL] = compile_entry(control)
        environment["Control graph"] = f"`{entries[ARM_CONTROL].key[:12]}…`, profile `{control.profile.id}`"
        environment["Capture graph"] = f"`{capture.key[:12]}…` (base graph, layers {layers})"
        environment["Tokenizer"] = "snapshot `tokenizer.json` via HF tokenizers"

    def open_model(name: str) -> LanguageModel:
        e = entries[name]
        return Lfm3OrtModel(
            e.graph_path, e.manifest.strength_input.name, provider=provider, device_id=device_id, threads=threads
        )

    def open_cpu() -> LanguageModel:
        e = entries[ARM_PROFILE]
        return Lfm3OrtModel(e.graph_path, e.manifest.strength_input.name, provider="cpu", threads=threads)

    def open_capture() -> CaptureRunner:
        return Lfm3OrtCapture(
            capture.graph_path,
            {layer: capture.output(layer, target) for layer in layers},
            provider=provider,
            device_id=device_id,
            threads=threads,
        )

    try:
        return evaluate_with(
            spec,
            prepared,
            open_model=open_model,
            open_capture=open_capture,
            directions=directions,
            scales=scales,
            control_directions=control_directions(profile_file, spec.control_seed),
            prepare_control=prepare_control,
            decode=lambda ids: tokenizer.decode(list(ids), skip_special_tokens=False),
            open_cpu_model=None if provider == "cpu" else open_cpu,
            out_dir=out_dir,
            profile_sha256=profile_file.sha256,
            profile_id=profile_file.profile.id,
            environment=environment,
            operator=profile_file.profile.operator,
            log=log,
        )
    finally:
        shutil.rmtree(control_dir, ignore_errors=True)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("spec", type=Path, help="evaluation spec (YAML)")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=os.environ.get("STARLING_LFM3_SNAPSHOT"),
        help="model snapshot root (default: $STARLING_LFM3_SNAPSHOT)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=os.environ.get("STARLING_LFM3_INTERVENTION_CACHE"),
        help="intervention cache root on the snapshot's file system (default: $STARLING_LFM3_INTERVENTION_CACHE)",
    )
    parser.add_argument("--out", type=Path, required=True, help="report directory")
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu", help="ONNX Runtime execution provider")
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device id")
    parser.add_argument("--threads", type=int, help="ONNX Runtime intra-op threads")


def run_cli(args: argparse.Namespace) -> int:
    if args.snapshot is None or args.cache is None:
        print(
            "error: --snapshot and --cache are required (or STARLING_LFM3_SNAPSHOT / STARLING_LFM3_INTERVENTION_CACHE)"
        )
        return 2
    try:
        spec = load_spec(args.spec)
    except EvaluationSpecError as e:
        print(f"error: {e}")
        return 2
    from .injection_sites import LFM3_FP32_GRAPH_SHA256

    external = None
    if load_profile(spec.profile).profile.base_model_sha256 == LFM3_FP32_GRAPH_SHA256:
        from .lfm3_synthetic import LFM3_FP32_EXTERNAL_SHA256

        external = LFM3_FP32_EXTERNAL_SHA256
    report = run_evaluation(
        spec,
        snapshot=Path(args.snapshot),
        cache_root=Path(args.cache),
        out_dir=args.out.absolute(),
        provider=args.provider,
        device_id=args.device_id,
        threads=args.threads,
        external_sha256=external,
    )
    print(args.out / REPORT_MD)
    rng = report["recommendation"]["recommended_range"]
    print(f"recommended range: {rng if rng else 'none'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m model_pipeline.interventions.evaluate", description=__doc__.split("\n")[0]
    )
    add_arguments(parser)
    return run_cli(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
