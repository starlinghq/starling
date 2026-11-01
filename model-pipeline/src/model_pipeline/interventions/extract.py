"""Extract a directional-projection profile from two contrasting prompt sets (experimental).

    model-pipeline interventions extract <spec.yaml> --snapshot <model snapshot> --cache <cache root> [--out <dir>]

The spec (``schemas/intervention-extraction-spec.schema.json``, ``docs/interventions/extraction.md``) names the base
graph, two paired prompt sets (``positive`` and ``negative``: the same prompts with contrasting responses), the
candidate layers, the rank, the pooling and the seed. The command

1. compiles (or reuses) the capture graph of the candidate layers (``capture.get_or_compile_capture``);
2. renders every (prompt, response) pair with the model's chat template, runs it through the capture graph once
   (prefill only, no generation) and pools each site's residual over the response tokens (``response_mean``) or takes
   the last response token (``last_token``);
3. per candidate layer, forms the paired differences ``d_i = positive_i - negative_i`` and measures how well the
   mean-difference direction separates the two sets **on held-out pairs** (``folds``-fold cross-validation over pairs,
   split by ``seed``): projection AUC, Cohen's d and paired accuracy;
4. selects up to ``max_layers`` layers whose held-out AUC is at least ``min_auc``, by held-out Cohen's d;
5. computes each selected layer's ``rank`` orthonormal directions from all pairs (direction 1: the normalized mean
   difference; directions 2..rank: top principal components of the paired differences with direction 1 projected
   out) and writes ``profile.json``, ``directions/*.f32`` (``docs/interventions/profile-format.md``) and
   ``extraction-report.json`` / ``extraction-report.md`` next to the spec (or into ``--out``).

The numerical part (``analyze``) takes plain arrays, so it is tested without a model. A profile written here describes
a direction in activation space that separates the two prompt sets; it is not evidence of any behavioural effect.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from .contract import (
    DTYPE_FLOAT32,
    OPERATOR_DIRECTIONAL_PROJECTION,
    SCHEMA_VERSION,
    InterventionProfileFile,
    encode_directions,
    format_float32,
    is_relative_path,
    load_profile,
    max_orthonormality_error,
    sha256_bytes,
    sha256_file,
)
from .synthetic import _dumps, _Raw

SPEC_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "intervention-extraction-spec.schema.json"
EXTRACTOR_VERSION = "0.1.0"
REPORT_JSON = "extraction-report.json"
REPORT_MD = "extraction-report.md"
PROFILE_NAME = "profile.json"

# The LFM3 chat template (Lfm3ChatTemplate in Starlings.Models.Lfm3) for one user turn and one assistant turn with
# thinking disabled. The generation prompt ends in an empty think block; the response follows it.
LFM3_BOS = "<|startoftext|>"
LFM3_IM_END = "<|im_end|>"


class ExtractionSpecError(ValueError):
    """The spec or a prompt set is invalid."""


class ExtractionError(RuntimeError):
    """Extraction ran but cannot produce a profile (for example no layer meets the selection threshold)."""


# --------------------------------------------------------------------------------------------------------------------
# Spec


@dataclass(frozen=True)
class PromptSetSpec:
    label: str
    path: Path


@dataclass(frozen=True)
class ExtractionSpec:
    path: Path
    sha256: str
    profile_id: str
    description: str | None
    model: str
    base_model_sha256: str
    graph: str
    chat_template: str
    operator: str
    target: str
    positive: PromptSetSpec
    negative: PromptSetSpec
    pooling: str
    candidate_layers: tuple[int, ...]
    rank: int
    direction_method: str
    scale: float
    folds: int
    max_layers: int
    min_auc: float
    seed: int

    @property
    def directory(self) -> Path:
        return self.path.parent


def _schema() -> dict[str, Any]:
    return json.loads(SPEC_SCHEMA_PATH.read_text(encoding="utf-8"))


def parse_spec(data: bytes, path: Path) -> ExtractionSpec:
    import jsonschema

    try:
        document = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as e:
        raise ExtractionSpecError(f"{path}: not valid UTF-8 YAML: {e}") from e
    validator = jsonschema.Draft202012Validator(_schema())
    errors = sorted(validator.iter_errors(document), key=lambda e: list(e.absolute_path))
    if errors:
        details = "; ".join(f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errors)
        raise ExtractionSpecError(f"{path}: {details}")
    sets = {}
    for role in ("positive", "negative"):
        entry = document["prompt_sets"][role]
        if not is_relative_path(entry["path"]):
            raise ExtractionSpecError(
                f"{path}: prompt_sets/{role}/path '{entry['path']}' is not a plain relative path."
            )
        sets[role] = PromptSetSpec(entry["label"], path.parent.joinpath(*entry["path"].split("/")))
    if sets["positive"].label == sets["negative"].label:
        raise ExtractionSpecError(f"{path}: the two prompt sets have the same label.")
    if sets["positive"].path.resolve() == sets["negative"].path.resolve():
        raise ExtractionSpecError(f"{path}: the two prompt sets are the same file.")
    if not is_relative_path(document["graph"]):
        raise ExtractionSpecError(f"{path}: graph '{document['graph']}' is not a plain relative path.")
    selection = document["selection"]
    if selection["max_layers"] > len(document["candidate_layers"]):
        raise ExtractionSpecError(f"{path}: selection/max_layers exceeds the number of candidate layers.")
    return ExtractionSpec(
        path=path,
        sha256=sha256_bytes(data),
        profile_id=document["profile_id"],
        description=document.get("description"),
        model=document["model"],
        base_model_sha256=document["base_model_sha256"],
        graph=document["graph"],
        chat_template=document["chat_template"],
        operator=document["operator"],
        target=document["target"],
        positive=sets["positive"],
        negative=sets["negative"],
        pooling=document["pooling"],
        candidate_layers=tuple(sorted(document["candidate_layers"])),
        rank=document["rank"],
        direction_method=document["direction_method"],
        scale=float(document["scale"]),
        folds=document["evaluation"]["folds"],
        max_layers=selection["max_layers"],
        min_auc=float(selection["min_auc"]),
        seed=document["seed"],
    )


def load_spec(path: str | Path) -> ExtractionSpec:
    path = Path(path).absolute()
    return parse_spec(path.read_bytes(), path)


# --------------------------------------------------------------------------------------------------------------------
# Prompt sets


@dataclass(frozen=True)
class Example:
    id: str
    prompt: str
    response: str


def load_prompt_set(path: Path) -> list[Example]:
    if not path.is_file():
        raise ExtractionSpecError(f"prompt set '{path}' does not exist.")
    examples: list[Example] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            raise ExtractionSpecError(f"{path}:{number}: invalid JSON: {e}") from e
        if not isinstance(item, dict) or set(item) != {"id", "prompt", "response"}:
            raise ExtractionSpecError(f"{path}:{number}: expected exactly the keys id, prompt, response.")
        if not all(isinstance(item[k], str) for k in item) or not item["id"] or not item["prompt"]:
            raise ExtractionSpecError(f"{path}:{number}: id and prompt must be non-empty strings, response a string.")
        examples.append(Example(item["id"], item["prompt"], item["response"]))
    ids = [e.id for e in examples]
    if len(set(ids)) != len(ids):
        raise ExtractionSpecError(f"{path}: duplicate ids.")
    return examples


def load_pairs(spec: ExtractionSpec) -> tuple[list[Example], list[Example]]:
    """Both prompt sets, paired by id: the same ids in the same order, the same prompts, different responses."""
    positive, negative = load_prompt_set(spec.positive.path), load_prompt_set(spec.negative.path)
    if [e.id for e in positive] != [e.id for e in negative]:
        raise ExtractionSpecError("the two prompt sets must list the same ids in the same order.")
    for p, n in zip(positive, negative, strict=True):
        if p.prompt != n.prompt:
            raise ExtractionSpecError(f"pair '{p.id}': the prompts differ; pairs must share the prompt.")
        if p.response == n.response:
            raise ExtractionSpecError(f"pair '{p.id}': the responses are identical.")
        if spec.pooling == "response_mean" and (not p.response or not n.response):
            raise ExtractionSpecError(f"pair '{p.id}': response_mean pooling needs non-empty responses.")
    if len(positive) < 2 * spec.folds:
        raise ExtractionSpecError(f"{len(positive)} pairs are too few for {spec.folds} folds (at least 2 per fold).")
    return positive, negative


# --------------------------------------------------------------------------------------------------------------------
# Rendering and tokenization


class Encoder(Protocol):
    def __call__(self, text: str) -> list[int]: ...


def render_lfm3(prompt: str, response: str) -> tuple[str, str]:
    """(generation prompt, full text) for one user turn and its assistant response, thinking disabled.

    The full text is what the LFM3 template renders for [user, assistant] (the assistant turn after the last user turn
    gets an empty think block), up to and including the assistant's ``<|im_end|>``.
    """
    prefix = f"{LFM3_BOS}<|im_start|>user\n{prompt}{LFM3_IM_END}\n<|im_start|>assistant\n<think></think>"
    return prefix, f"{prefix}{response}{LFM3_IM_END}"


@dataclass(frozen=True)
class Tokenized:
    ids: tuple[int, ...]
    response_start: int  # first response token
    response_end: int  # one past the last response token (the position of <|im_end|>)

    @property
    def response_tokens(self) -> int:
        return self.response_end - self.response_start


def tokenize(encoder: Encoder, example: Example, template: str = "lfm3") -> Tokenized:
    if template != "lfm3":
        raise ExtractionSpecError(f"unknown chat template '{template}'.")
    prefix, full = render_lfm3(example.prompt, example.response)
    prefix_ids = list(encoder(prefix))
    ids = list(encoder(full))
    end_ids = list(encoder(LFM3_IM_END))
    if ids[: len(prefix_ids)] != prefix_ids:
        raise ExtractionSpecError(f"'{example.id}': the generation prompt does not tokenize to a prefix of the turn.")
    if len(end_ids) != 1 or ids[-1] != end_ids[0]:
        raise ExtractionSpecError(f"'{example.id}': the turn does not end in the single <|im_end|> token.")
    return Tokenized(tuple(ids), len(prefix_ids), len(ids) - 1)


def pool(activations: np.ndarray, token: Tokenized, pooling: str) -> np.ndarray:
    """``activations`` is ``[sequence, hidden]`` for one example; returns ``[hidden]`` float64."""
    if activations.shape[0] != len(token.ids):
        raise ValueError(f"activations cover {activations.shape[0]} positions, the example has {len(token.ids)}.")
    if pooling == "response_mean":
        if token.response_tokens < 1:
            raise ValueError("response_mean pooling needs at least one response token.")
        return activations[token.response_start : token.response_end].astype(np.float64).mean(axis=0)
    if pooling == "last_token":
        return activations[token.response_end - 1].astype(np.float64)
    raise ValueError(f"unknown pooling '{pooling}'.")


Runner = Callable[[Sequence[int]], Mapping[int, np.ndarray]]


def collect(runner: Runner, tokens: Sequence[Tokenized], layers: Sequence[int], pooling: str) -> np.ndarray:
    """``[examples, layers, hidden]`` float64 pooled activations; ``runner(ids)`` returns ``{layer: [seq, hidden]}``."""
    rows = []
    for token in tokens:
        out = runner(token.ids)
        rows.append(np.stack([pool(np.asarray(out[layer]), token, pooling) for layer in layers]))
    return np.stack(rows)


# --------------------------------------------------------------------------------------------------------------------
# Analysis (plain arrays)


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Probability that a positive score exceeds a negative one (ties count half): the Mann-Whitney AUC."""
    positive, negative = np.asarray(positive, np.float64), np.asarray(negative, np.float64)
    scores = np.concatenate([positive, negative])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos, n_neg = len(positive), len(negative)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def cohens_d(positive: np.ndarray, negative: np.ndarray) -> float:
    positive, negative = np.asarray(positive, np.float64), np.asarray(negative, np.float64)
    n1, n2 = len(positive), len(negative)
    pooled = ((n1 - 1) * positive.var(ddof=1) + (n2 - 1) * negative.var(ddof=1)) / (n1 + n2 - 2)
    diff = positive.mean() - negative.mean()
    if pooled <= 0:
        return math.inf if diff > 0 else (-math.inf if diff < 0 else 0.0)
    return float(diff / math.sqrt(pooled))


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm == 0.0 or not math.isfinite(norm):
        raise ExtractionError("the mean paired difference is zero or not finite; the sets do not differ here.")
    return v / norm


def directions(positive: np.ndarray, negative: np.ndarray, rank: int) -> np.ndarray:
    """``[rank, hidden]`` float64 orthonormal rows from paired samples ``[pairs, hidden]`` (``mean_difference_pca``).

    Row 0 is the normalized mean of ``d_i = positive_i - negative_i`` (sign: towards ``positive``). Rows 1.. are the
    top right singular vectors of the centred differences after projecting row 0 out, each signed so that its largest
    absolute component is positive. Rows are re-orthonormalized with a QR step.
    """
    d = np.asarray(positive, np.float64) - np.asarray(negative, np.float64)
    if rank < 1 or rank > d.shape[1]:
        raise ValueError(f"rank {rank} is outside 1..{d.shape[1]}.")
    u = _unit(d.mean(axis=0))
    rows = [u]
    if rank > 1:
        residual = d - d.mean(axis=0)
        residual = residual - np.outer(residual @ u, u)
        if rank - 1 > min(residual.shape):
            raise ExtractionError(f"rank {rank} needs more pairs than {d.shape[0]}.")
        _, singular, vt = np.linalg.svd(residual, full_matrices=False)
        for k in range(rank - 1):
            if singular[k] <= 1e-12 * max(1.0, singular[0]):
                raise ExtractionError(f"the paired differences span fewer than {rank} directions.")
            v = vt[k]
            rows.append(v if v[np.argmax(np.abs(v))] > 0 else -v)
    q, r = np.linalg.qr(np.stack(rows).T)
    q = q.T * np.sign(np.diag(r))[:, None]  # keep each row's orientation
    return q


@dataclass
class LayerMetrics:
    layer: int
    heldout_auc: float
    heldout_cohens_d: float
    heldout_pair_accuracy: float
    insample_auc: float
    insample_cohens_d: float
    mean_difference_norm: float
    mean_activation_norm: float
    captured_difference_energy: float = 0.0
    orthonormality_error: float = 0.0

    def to_document(self) -> dict[str, Any]:
        return {k: (_round(v) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class Analysis:
    layers: list[LayerMetrics]
    selected: list[int]
    directions: dict[int, np.ndarray] = field(default_factory=dict)
    folds: list[list[int]] = field(default_factory=list)


def fold_indices(pairs: int, folds: int, seed: int) -> list[np.ndarray]:
    order = np.random.default_rng(seed).permutation(pairs)
    return [np.sort(part) for part in np.array_split(order, folds)]


def layer_metrics(positive: np.ndarray, negative: np.ndarray, layer: int, folds: list[np.ndarray]) -> LayerMetrics:
    """Separability of one layer's paired samples ``[pairs, hidden]`` along the mean-difference direction.

    Held-out: for each fold, the direction is fitted on the other folds and every held-out sample is projected on it;
    AUC, Cohen's d and paired accuracy are computed over the pooled out-of-fold projections. In-sample values use the
    direction fitted on all pairs (they are optimistic and reported for comparison only).
    """
    pairs = positive.shape[0]
    pos_scores, neg_scores = np.empty(pairs), np.empty(pairs)
    everything = np.arange(pairs)
    for held in folds:
        train = np.setdiff1d(everything, held)
        u = _unit((positive[train] - negative[train]).mean(axis=0))
        pos_scores[held] = positive[held] @ u
        neg_scores[held] = negative[held] @ u
    mean_difference = (positive - negative).mean(axis=0)
    u = _unit(mean_difference)
    return LayerMetrics(
        layer=layer,
        heldout_auc=auc(pos_scores, neg_scores),
        heldout_cohens_d=cohens_d(pos_scores, neg_scores),
        heldout_pair_accuracy=float(np.mean(pos_scores > neg_scores)),
        insample_auc=auc(positive @ u, negative @ u),
        insample_cohens_d=cohens_d(positive @ u, negative @ u),
        mean_difference_norm=float(np.linalg.norm(mean_difference)),
        mean_activation_norm=float(np.linalg.norm(np.concatenate([positive, negative]), axis=1).mean()),
    )


def analyze(
    positive: np.ndarray,
    negative: np.ndarray,
    layers: Sequence[int],
    *,
    rank: int,
    folds: int,
    seed: int,
    max_layers: int,
    min_auc: float,
) -> Analysis:
    """``positive`` and ``negative`` are ``[pairs, len(layers), hidden]``, row ``i`` of each from the same prompt.

    Layers with held-out AUC >= ``min_auc`` are ranked by held-out Cohen's d (ties: lower layer) and the first
    ``max_layers`` are selected; their directions are fitted on all pairs.
    """
    positive, negative = np.asarray(positive, np.float64), np.asarray(negative, np.float64)
    if positive.shape != negative.shape or positive.ndim != 3 or positive.shape[1] != len(layers):
        raise ValueError(f"expected two [pairs, {len(layers)}, hidden] arrays, got {positive.shape}, {negative.shape}.")
    if not np.all(np.isfinite(positive)) or not np.all(np.isfinite(negative)):
        raise ExtractionError("captured activations contain non-finite values.")
    parts = fold_indices(positive.shape[0], folds, seed)
    metrics = [layer_metrics(positive[:, i], negative[:, i], layer, parts) for i, layer in enumerate(layers)]
    eligible = [m for m in metrics if m.heldout_auc >= min_auc]
    ranked = sorted(eligible, key=lambda m: (-m.heldout_cohens_d, m.layer))
    selected = sorted(m.layer for m in ranked[:max_layers])
    found: dict[int, np.ndarray] = {}
    for i, layer in enumerate(layers):
        if layer not in selected:
            continue
        q = directions(positive[:, i], negative[:, i], rank)
        d = positive[:, i] - negative[:, i]
        m = metrics[i]
        m.captured_difference_energy = float(np.sum((d @ q.T) ** 2) / np.sum(d**2))
        m.orthonormality_error = max_orthonormality_error(q.astype(np.float32))
        found[layer] = q
    return Analysis(metrics, selected, found, [p.tolist() for p in parts])


def _round(value: float) -> float:
    return float(f"{value:.6g}") if math.isfinite(value) else value


# --------------------------------------------------------------------------------------------------------------------
# Output


def write_profile(
    directory: Path,
    *,
    profile_id: str,
    description: str | None,
    base_model_sha256: str,
    target: str,
    found: Mapping[int, np.ndarray],
    scale: float,
) -> InterventionProfileFile:
    """``profile.json`` (canonical form) and ``directions/layer<L>-<target>.f32``; returns the loaded profile."""
    (directory / "directions").mkdir(parents=True, exist_ok=True)
    sites = []
    for layer in sorted(found):
        q = np.ascontiguousarray(found[layer], dtype=np.float32)
        data = encode_directions(q)
        relative = f"directions/layer{layer}-{target}.f32"
        (directory / relative).write_bytes(data)
        sites.append(
            {
                "layer": layer,
                "target": target,
                "hidden_size": int(q.shape[1]),
                "rank": int(q.shape[0]),
                "dtype": DTYPE_FLOAT32,
                "directions_file": relative,
                "directions_sha256": sha256_bytes(data),
                "scales": [_Raw(format_float32(scale)) for _ in range(q.shape[0])],
            }
        )
    document: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "id": profile_id}
    if description is not None:
        document["description"] = description
    document |= {"base_model_sha256": base_model_sha256, "operator": OPERATOR_DIRECTIONAL_PROJECTION, "sites": sites}
    path = directory / PROFILE_NAME
    path.write_bytes(_dumps(document).encode("utf-8"))
    return load_profile(path)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def build_report(
    spec: ExtractionSpec,
    analysis: Analysis,
    *,
    pairs: int,
    response_tokens: Mapping[str, Sequence[int]],
    out_dir: Path,
    profile: InterventionProfileFile | None,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    def stats(values: Sequence[int]) -> dict[str, Any]:
        a = np.asarray(values)
        return {"min": int(a.min()), "median": _round(float(np.median(a))), "mean": _round(float(a.mean())),
                "max": int(a.max())}  # fmt: skip

    return {
        "extractor_version": EXTRACTOR_VERSION,
        "spec": {"path": _relative(spec.path, out_dir), "sha256": spec.sha256},
        "profile_id": spec.profile_id,
        "base_model_sha256": spec.base_model_sha256,
        "prompt_sets": {
            role: {"label": s.label, "path": _relative(s.path, out_dir), "sha256": sha256_file(s.path)}
            for role, s in (("positive", spec.positive), ("negative", spec.negative))
        },
        "pairs": pairs,
        "response_tokens": {role: stats(values) for role, values in response_tokens.items()},
        "method": {
            "pooling": spec.pooling,
            "direction_method": spec.direction_method,
            "rank": spec.rank,
            "scale": spec.scale,
            "folds": spec.folds,
            "seed": spec.seed,
            "selection": {"max_layers": spec.max_layers, "min_auc": spec.min_auc},
        },
        "layers": [m.to_document() for m in analysis.layers],
        "selected_layers": analysis.selected,
        "profile": None
        if profile is None
        else {
            "path": _relative(profile.path, out_dir),
            "sha256": profile.sha256,
            "sites": [
                {"layer": s.layer, "rank": s.rank, "directions_file": s.directions_file, "sha256": s.directions_sha256}
                for s in profile.profile.sites
            ],
        },
        "environment": dict(environment),
    }


def render_report_markdown(report: Mapping[str, Any]) -> str:
    method = report["method"]
    positive, negative = report["prompt_sets"]["positive"], report["prompt_sets"]["negative"]
    lines = [
        f"# Extraction report: `{report['profile_id']}`",
        "",
        "Status: **experimental**. Generated by `model-pipeline interventions extract` "
        f"(extractor {report['extractor_version']}); `{REPORT_JSON}` holds the same numbers. The profile describes a "
        "direction in activation space that separates the two prompt sets on held-out pairs. It is not evidence of a "
        "behavioural effect.",
        "",
        "| Input | Value |",
        "| --- | --- |",
        f"| Spec | `{report['spec']['path']}` (SHA-256 `{report['spec']['sha256'][:12]}…`) |",
        f"| Base graph | SHA-256 `{report['base_model_sha256']}` |",
        f"| Positive set | `{positive['label']}`: `{positive['path']}` (`{positive['sha256'][:12]}…`) |",
        f"| Negative set | `{negative['label']}`: `{negative['path']}` (`{negative['sha256'][:12]}…`) |",
        f"| Pairs | {report['pairs']} |",
    ]
    for role, s in report["response_tokens"].items():
        lines.append(
            f"| Response tokens, {role} | min {s['min']}, median {s['median']}, mean {s['mean']}, max {s['max']} |"
        )
    lines += [
        f"| Pooling | `{method['pooling']}` |",
        f"| Directions | `{method['direction_method']}`, rank {method['rank']}, scale {method['scale']} |",
        f"| Held-out evaluation | {method['folds']}-fold cross-validation over pairs, seed {method['seed']} |",
        f"| Selection | up to {method['selection']['max_layers']} layers with held-out AUC >= "
        f"{method['selection']['min_auc']}, ranked by held-out Cohen's d |",
    ]
    for key, value in report["environment"].items():
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "## Per-layer separability",
        "",
        "Held-out columns project each pair on the mean-difference direction fitted without its fold. "
        "In-sample columns "
        "fit and score on all pairs and are optimistic. `|mean diff| / |x|` is the norm of the mean paired difference "
        "over the mean residual norm.",
        "",
        "| Layer | Held-out AUC | Held-out d | Held-out pair acc. | In-sample AUC | In-sample d | |mean diff| / |x| | "
        "Selected |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for m in report["layers"]:
        ratio = m["mean_difference_norm"] / m["mean_activation_norm"] if m["mean_activation_norm"] else float("nan")
        lines.append(
            f"| {m['layer']} | {m['heldout_auc']:.4f} | {m['heldout_cohens_d']:.3f} | {m['heldout_pair_accuracy']:.3f} "
            f"| {m['insample_auc']:.4f} | {m['insample_cohens_d']:.3f} | {ratio:.4f} | "
            f"{'yes' if m['layer'] in report['selected_layers'] else ''} |"
        )
    lines += ["", "## Result", ""]
    profile = report["profile"]
    if profile is None:
        lines.append("No layer met the selection threshold; no profile was written.")
    else:
        lines.append(
            f"`{profile['path']}` (SHA-256 `{profile['sha256']}`), selected layers {report['selected_layers']}:"
        )
        lines.append("")
        by_layer = {m["layer"]: m for m in report["layers"]}
        for site in profile["sites"]:
            m = by_layer[site["layer"]]
            lines.append(
                f"- layer {site['layer']}: rank {site['rank']}, `{site['directions_file']}` "
                f"(`{site['sha256'][:12]}…`), "
                f"captures {m['captured_difference_energy']:.3f} of the paired-difference energy, orthonormality "
                f"error {m['orthonormality_error']:.2g}"
            )
    return "\n".join(lines) + "\n"


def write_report(out_dir: Path, report: Mapping[str, Any]) -> None:
    (out_dir / REPORT_JSON).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / REPORT_MD).write_text(render_report_markdown(report), encoding="utf-8")


def extract_from_activations(
    spec: ExtractionSpec,
    positive: np.ndarray,
    negative: np.ndarray,
    *,
    response_tokens: Mapping[str, Sequence[int]],
    out_dir: Path,
    environment: Mapping[str, Any],
) -> tuple[Analysis, InterventionProfileFile | None]:
    """Analyze pooled activations ``[pairs, candidate layers, hidden]``, write the profile (if any) and the report."""
    analysis = analyze(
        positive,
        negative,
        spec.candidate_layers,
        rank=spec.rank,
        folds=spec.folds,
        seed=spec.seed,
        max_layers=spec.max_layers,
        min_auc=spec.min_auc,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = None
    if analysis.selected:
        profile = write_profile(
            out_dir,
            profile_id=spec.profile_id,
            description=spec.description,
            base_model_sha256=spec.base_model_sha256,
            target=spec.target,
            found=analysis.directions,
            scale=spec.scale,
        )
    report = build_report(
        spec,
        analysis,
        pairs=positive.shape[0],
        response_tokens=response_tokens,
        out_dir=out_dir,
        profile=profile,
        environment=environment,
    )
    write_report(out_dir, report)
    if profile is None:
        raise ExtractionError(
            f"no candidate layer reached held-out AUC {spec.min_auc}; see {out_dir / REPORT_MD}. "
            "No profile was written."
        )
    return analysis, profile


# --------------------------------------------------------------------------------------------------------------------
# The real model (LFM3, ONNX Runtime CPU)


class Lfm3CaptureRunner:
    """Prefill runs of an LFM3 capture graph: empty caches, ``logits_to_keep = 1``; returns ``{layer: [seq, hidden]}``.

    One session for every run (load at most one fp32 LFM3 session at a time).
    """

    def __init__(self, graph_path: Path, outputs: Mapping[int, str], threads: int | None = None) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self._session = ort.InferenceSession(str(graph_path), options, providers=["CPUExecutionProvider"])
        self._outputs = dict(outputs)
        self._names = list(self._outputs.values())
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
        values = self._session.run(self._names, feeds)
        by_name = dict(zip(self._names, values, strict=True))
        return {layer: by_name[name][0] for layer, name in self._outputs.items()}


def lfm3_encoder(snapshot: Path) -> Encoder:
    """HF ``tokenizers`` over the snapshot's ``tokenizer.json`` (no special tokens added: the template renders them).

    The renderer plus this encoder reproduce every case of the LFM3 template corpus
    (``ai-dotnet/tests/Starlings.Models.Lfm3.Tests/Data/template-reference.jsonl``); ``test_intervention_extract_lfm3``
    checks it.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    return lambda text: tokenizer.encode(text, add_special_tokens=False).ids


def run_extraction(
    spec: ExtractionSpec,
    *,
    snapshot: Path,
    cache_root: Path,
    out_dir: Path | None = None,
    threads: int | None = None,
    external_sha256: Mapping[str, str] | None = None,
    log: Callable[[str], None] = print,
) -> tuple[Analysis, InterventionProfileFile | None]:
    import onnxruntime as ort

    from .capture import get_or_compile_capture
    from .contract import sha256_bytes as _sha

    out_dir = (out_dir or spec.directory).absolute()
    graph_sha = _sha(snapshot.joinpath(*spec.graph.split("/")).read_bytes())
    if graph_sha != spec.base_model_sha256:
        raise ExtractionSpecError(f"the snapshot's {spec.graph} has SHA-256 {graph_sha}, the spec says "
                                  f"{spec.base_model_sha256}.")  # fmt: skip
    positive, negative = load_pairs(spec)
    encoder = lfm3_encoder(snapshot)
    pos_tokens = [tokenize(encoder, e, spec.chat_template) for e in positive]
    neg_tokens = [tokenize(encoder, e, spec.chat_template) for e in negative]
    entry = get_or_compile_capture(
        cache_root=cache_root,
        model=spec.model,
        base_dir=snapshot,
        graph=spec.graph,
        layers=spec.candidate_layers,
        target=spec.target,
        external_sha256=external_sha256,
    )
    log(f"capture graph: {entry.directory} ({'compiled' if entry.created else 'reused'})")
    runner = Lfm3CaptureRunner(
        entry.graph_path, {layer: entry.output(layer, spec.target) for layer in spec.candidate_layers}, threads
    )
    pos = collect(runner, pos_tokens, spec.candidate_layers, spec.pooling)
    log(f"{spec.positive.label}: {len(pos_tokens)} runs")
    neg = collect(runner, neg_tokens, spec.candidate_layers, spec.pooling)
    log(f"{spec.negative.label}: {len(neg_tokens)} runs")
    del runner
    environment = {
        "Runtime": f"onnxruntime {ort.__version__}, CPUExecutionProvider, fp32 graph, prefill only",
        "Capture entry": f"`{entry.key[:12]}…` (capture {entry.manifest.capture_version}, site mapping "
        f"`{entry.manifest.site_mapping_version}`)",
        "Tokenizer": "snapshot `tokenizer.json` via HF tokenizers",
    }
    return extract_from_activations(
        spec,
        pos,
        neg,
        response_tokens={
            spec.positive.label: [t.response_tokens for t in pos_tokens],
            spec.negative.label: [t.response_tokens for t in neg_tokens],
        },
        out_dir=out_dir,
        environment=environment,
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("spec", type=Path, help="extraction spec (YAML)")
    parser.add_argument("--snapshot", type=Path, default=os.environ.get("STARLING_LFM3_SNAPSHOT"),
                        help="model snapshot root (default: $STARLING_LFM3_SNAPSHOT)")  # fmt: skip
    parser.add_argument("--cache", type=Path, default=os.environ.get("STARLING_LFM3_INTERVENTION_CACHE"),
                        help="intervention cache root on the snapshot's file system "
                        "(default: $STARLING_LFM3_INTERVENTION_CACHE)")  # fmt: skip
    parser.add_argument("--out", type=Path, help="output directory (default: the spec's directory)")
    parser.add_argument("--threads", type=int, help="ONNX Runtime intra-op threads")


def run_cli(args: argparse.Namespace) -> int:
    if args.snapshot is None or args.cache is None:
        print(
            "error: --snapshot and --cache are required (or STARLING_LFM3_SNAPSHOT / STARLING_LFM3_INTERVENTION_CACHE)"
        )
        return 2
    spec = load_spec(args.spec)
    external = None
    if spec.base_model_sha256 == _lfm3_graph_sha():
        from .lfm3_synthetic import LFM3_FP32_EXTERNAL_SHA256

        external = LFM3_FP32_EXTERNAL_SHA256
    try:
        _, profile = run_extraction(
            spec,
            snapshot=Path(args.snapshot),
            cache_root=Path(args.cache),
            out_dir=args.out,
            threads=args.threads,
            external_sha256=external,
        )
    except ExtractionError as e:
        print(f"error: {e}")
        return 1
    print(profile.path if profile else "")
    return 0


def _lfm3_graph_sha() -> str:
    from .injection_sites import LFM3_FP32_GRAPH_SHA256

    return LFM3_FP32_GRAPH_SHA256


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m model_pipeline.interventions.extract", description=__doc__.split("\n")[0]
    )
    add_arguments(parser)
    return run_cli(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
