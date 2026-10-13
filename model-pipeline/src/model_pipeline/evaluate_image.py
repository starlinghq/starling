"""Staged fidelity evaluator, task `image-embedding` (EmbeddingGemma 2 images; extends P-007).

Stages (same substage names as the other tasks, so gates and reports line up):

    tokenizer    stage 1, rendering and preprocessing from decoded RGB: rendered ids (placeholder expansion and
                 the owned tokenizer), and per image the SHA-256 of the decoded RGB, the resized bytes, the pixel
                 values and the positions, exact against `image-embeddings.jsonl`                          (G1)
    export       not applicable: the image tolerances bind the delivered C# path; the Python ONNX Runtime
                 parity of the graphs is protected evidence of its own (`parity-cpu.json`)
    dotnet-ids   stage 3, the C# implementation from decoded RGB: embeddings (max abs, min cosine), soft tokens
                 and token embeddings (row-relative: per image or row, max |d| / max |ref|), batches against
                 single rows, NaN/Inf, provider options and the assets it loaded                          (G2)
    dotnet-text  not applicable (text-only rows are part of stage 3: they run on the released text graph)

Rows whose image has more than 8 bits per channel (PIL mode `I;16` in the protected `images.json`) are excluded by
contract: upstream turns such images white and the C# profile rejects them (manifest `behavior.unsupported`). Every
other row is required.

Soft tokens and token embeddings are compared with the reference tensors, which are too large for Git: the `.npz`
files of `run_embeddinggemma2_image.py --tensors`, each array checked against the SHA-256 in the protected
`tensors.json` before use (`reference_tensors`, CLI `--reference-tensors`). Without them those two checks are
not-run, so G2 cannot pass.

Tolerance keys (flattened YAML paths) read from the profile's protected tolerances file only: `EXPECTED_KEYS`.
"""

from __future__ import annotations

import json
import math
import struct
import zipfile
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from model_pipeline import fixtures, results
from model_pipeline.evaluate import Check, ErrorStats, EvaluationError, _bound, flatten, substage_status

TASK = "image-embedding"
SUBSTAGES = ("tokenizer", "export", "dotnet-ids", "dotnet-text")
SUBSTAGE_TITLES = {
    "tokenizer": "Stage 1, rendering and preprocessing from decoded RGB (ids, resized bytes, pixel values, positions)",
    "export": "Stage 2, export (not applicable: Python ORT parity is protected evidence of its own)",
    "dotnet-ids": "Stage 3, .NET from decoded RGB (embeddings, soft tokens, token embeddings, batches)",
    "dotnet-text": "Stage 4 (not applicable: text-only rows run in stage 3 on the released text graph)",
}

EXPECTED_KEYS: dict[str, Any] = {
    "profile": "<family>/<profile>, must equal the manifest id",
    "provider_options": "{} on CPU; {use_tf32: 0} on CUDA",
    "discrete.token_ids": "exact",
    "discrete.resized_pixels": "exact",
    "discrete.pixel_values": "exact",
    "discrete.image_positions": "exact",
    "float.soft_tokens_row_relative": 1.0e-3,
    "float.token_embeddings_row_relative": 1.0e-3,
    "float.embedding_max_abs": 1.0e-5,
    "float.min_cosine": 0.9999995,
    "float.padded_batch_vs_single_embedding_max_abs": 1.0e-5,
    "float.nan_or_inf": "fail",
    "float.report_only": ["mean_abs", "p99_abs", "per_value_relative"],
    "decoder_skiasharp.decoded_rgb_vs_pil": "exact",
}
KNOWN_KEYS = set(EXPECTED_KEYS)
EXCLUDED_MODES = ("I;16",)

# (stage-1 output suffix, fixture digest name, tolerance key)
DIGESTS = (
    ("rgb_sha256", "rgb_sha256", "decoder_skiasharp.decoded_rgb_vs_pil"),
    ("resized_sha256", "resized_uint8", "discrete.resized_pixels"),
    ("pixel_values_sha256", "pixel_values", "discrete.pixel_values"),
    ("positions_sha256", "positions", "discrete.image_positions"),
)


@dataclass
class Row:
    id: str
    input_ids: list[int]
    embedding: list[float]
    images: list[dict[str, Any]]


@dataclass
class Fixtures:
    rows: dict[str, Row]
    excluded: dict[str, str]
    tensor_digests: dict[str, dict[str, Any]]
    files: list[Path] = field(default_factory=list)

    @property
    def case_ids(self) -> list[str]:
        return [r for r in self.rows if r not in self.excluded]


def load_fixtures(directory: Path) -> Fixtures:
    emb, imgs, tens = (directory / n for n in ("image-embeddings.jsonl", "images.json", "tensors.json"))
    for p in (emb, imgs, tens):
        if not p.is_file():
            raise EvaluationError(f"{p}: missing image fixture file")
    modes = {i["id"]: i["mode"] for i in json.loads(imgs.read_text(encoding="utf-8"))["images"]}
    rows: dict[str, Row] = {}
    excluded: dict[str, str] = {}
    for line in emb.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        rows[r["id"]] = Row(r["id"], r["input_ids"], r["embedding"], r["images"])
        bad = [i["image"] for i in r["images"] if modes.get(i["image"]) in EXCLUDED_MODES]
        if bad:
            excluded[r["id"]] = f"image {', '.join(bad)} has more than 8 bits per channel (rejected by contract)"
    return Fixtures(rows, excluded, json.loads(tens.read_text(encoding="utf-8")), [emb, imgs, tens])


def read_npz_float32(path: Path, name: str, expected_sha256: str) -> tuple[list[int], list[float]]:
    """One float32 array of an uncompressed `.npz`, checked against its protected digest."""
    with zipfile.ZipFile(path) as z:
        data = z.read(name + ".npy")
    if data[:6] != b"\x93NUMPY":
        raise EvaluationError(f"{path}/{name}: not an .npy array")
    major = data[6]
    header_len = struct.unpack_from("<H" if major == 1 else "<I", data, 8)[0]
    start = (10 if major == 1 else 12) + header_len
    header = data[(10 if major == 1 else 12) : start].decode("latin1")
    if "'descr': '<f4'" not in header or "'fortran_order': True" in header:
        raise EvaluationError(f"{path}/{name}: expected C-ordered little-endian float32 ({header.strip()})")
    payload = data[start:]
    digest = sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise EvaluationError(f"{path}/{name}: SHA-256 {digest} differs from the protected tensors.json")
    shape_text = header.split("'shape': (")[1].split(")")[0]
    shape = [int(x) for x in shape_text.split(",") if x.strip()]
    return shape, list(struct.unpack(f"<{len(payload) // 4}f", payload))


def _row_relative(actual: list[float], expected: list[float]) -> float:
    if len(actual) != len(expected):
        return math.inf
    max_d = max((abs(a - e) for a, e in zip(actual, expected, strict=True)), default=0.0)
    max_r = max((abs(e) for e in expected), default=0.0)
    return max_d / max_r if max_r else (0.0 if max_d == 0 else math.inf)


class ImageEvaluator:
    """Checks for an `image-embedding` profile. One instance per evaluation."""

    def __init__(
        self,
        man: dict[str, Any],
        fx: Fixtures,
        tol: dict[str, Any],
        stages: dict[str, results.StageResults],
        reference_tensors: Path | None = None,
    ):
        self.man = man
        self.fx = fx
        self.tol = flatten(tol)
        self.stages = stages
        self.reference_tensors = reference_tensors
        self.checks: list[Check] = []

    def _add(self, check: Check) -> Check:
        self.checks.append(check)
        return check

    def _coverage(self, have: set[str]) -> str | None:
        expected = set(self.fx.case_ids)
        missing = sorted(expected - have)
        extra = sorted(have - set(self.fx.rows))
        parts = []
        if missing:
            parts.append(f"missing cases {missing} (every protected case is required)")
        if extra:
            parts.append(f"cases not in the protected fixtures {extra}")
        return "; ".join(parts) or None

    def _exact_key(self, check: Check, key: str) -> bool:
        check.tolerance_key, check.bound = key, self.tol.get(key)
        if self.tol.get(key) != "exact":
            check.status = "not-run"
            check.detail = f"{key} is not 'exact' in the protected tolerances"
            self._add(check)
            return False
        return True

    # -- identity -----------------------------------------------------------------------------

    def check_identity(self) -> None:
        manifest_files = {f["path"]: f["sha256"] for f in self.man["origin"]["files"]}
        required = {
            "tokenizer": ["tokenizer.json"],
            "dotnet": [p for p in manifest_files if p.endswith(".onnx")] + ["tokenizer.json"],
        }
        for stage_name, substage in (("tokenizer", "tokenizer"), ("dotnet", "dotnet-ids")):
            check = Check(f"{substage}.assets", substage, "Assets the stage loaded equal the manifest's origin.files")
            stage = self.stages.get(stage_name)
            if stage is None:
                check.detail = f"no results for stage {stage_name!r}"
                self._add(check)
                continue
            loaded = {a["path"]: a["sha256"] for a in stage.assets}
            wrong = [p for p, d in loaded.items() if manifest_files.get(p) != d]
            missing = [p for p in required[stage_name] if p not in loaded]
            check.metrics = {"assets": sorted(loaded)}
            if wrong or missing:
                check.status = "invalid"
                check.detail = (
                    f"assets not in origin.files or with another digest: {wrong}; required but absent: {missing}"
                )
            else:
                check.status = "passed"
            self._add(check)

    # -- stage 1 ------------------------------------------------------------------------------

    def check_preprocessing(self) -> None:
        stage = self.stages.get("tokenizer")
        ids = Check("tokenizer.token_ids", "tokenizer", "Rendered ids (placeholder expansion + tokenizer) exact")
        if stage is None:
            ids.detail = "no results for stage 'tokenizer'"
            self._add(ids)
            for suffix, _, key in DIGESTS:
                self._add(
                    Check(f"tokenizer.{suffix}", "tokenizer", key, key, detail="no results for stage 'tokenizer'")
                )
            return
        if self._exact_key(ids, "discrete.token_ids"):
            have = {c for c, o in stage.cases.items() if "input_ids" in o}
            problem = self._coverage(have)
            bad = [
                c for c in self.fx.case_ids if c in have and stage.cases[c]["input_ids"] != self.fx.rows[c].input_ids
            ]
            ids.metrics = {"cases": len(have & set(self.fx.case_ids)), "excluded_by_contract": self.fx.excluded}
            if problem or bad:
                ids.status = "failed"
                ids.detail = "; ".join(x for x in (problem, f"ids differ in {bad}" if bad else None) if x)
            else:
                ids.status = "passed"
            self._add(ids)
        for suffix, digest, key in DIGESTS:
            check = Check(f"tokenizer.{suffix}", "tokenizer", f"Per image {digest} digest exact")
            if not self._exact_key(check, key):
                continue
            images = mismatched = 0
            missing, differ = [], []
            for cid in self.fx.case_ids:
                for k, image in enumerate(self.fx.rows[cid].images):
                    images += 1
                    expected = image["rgb_sha256"] if digest == "rgb_sha256" else image["digests"][digest]
                    got = stage.cases.get(cid, {}).get(f"image{k}_{suffix}")
                    if got is None:
                        missing.append(f"{cid}/image{k}")
                    elif got != expected:
                        mismatched += 1
                        differ.append(f"{cid}/image{k}")
            check.metrics = {"images": images, "mismatched": mismatched}
            if missing or differ:
                check.status = "failed"
                check.detail = "; ".join(
                    x for x in (f"missing {missing}" if missing else None, f"differ {differ}" if differ else None) if x
                )
            else:
                check.status = "passed"
            self._add(check)

    # -- stage 3 ------------------------------------------------------------------------------

    def _judge(self, check: Check, key: str, value: float, worst: str, larger_is_worse: bool = True) -> Check:
        check.tolerance_key = key
        bound = _bound(self.tol, key)
        check.bound = bound
        if bound is None:
            check.status, check.detail = "not-run", f"no numeric bound for {key} in the protected tolerances"
        elif (value <= bound) if larger_is_worse else (value >= bound):
            check.status = "passed"
        else:
            check.status, check.detail = "failed", f"{value:.6g} against bound {bound:.6g} (worst {worst})"
        return self._add(check)

    def check_dotnet(self) -> None:
        stage = self.stages.get("dotnet")
        if stage is None:
            for name in ("embedding", "min_cosine", "soft_tokens", "token_embeddings", "finite", "provider_options"):
                self._add(Check(f"dotnet-ids.{name}", "dotnet-ids", name, detail="no results for stage 'dotnet'"))
            return
        emb = {c: o["embedding"] for c, o in stage.cases.items() if "embedding" in o}
        problem = self._coverage(set(emb))
        stats = ErrorStats()
        min_cos, worst_cos = 1.0, ""
        non_finite = []
        for cid in self.fx.case_ids:
            if cid not in emb:
                continue
            a, e = emb[cid], self.fx.rows[cid].embedding
            if any(not math.isfinite(x) for x in a):
                non_finite.append(cid)
                continue
            stats.add(cid, a, e, cosine=False)
            dot = sum(x * y for x, y in zip(a, e, strict=True))
            na, ne = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in e))
            cos = dot / (na * ne) if na and ne else 0.0
            if cos < min_cos:
                min_cos, worst_cos = cos, cid
        summary = stats.summary()
        check = Check("dotnet-ids.embedding", "dotnet-ids", "Embedding vs the reference, max abs")
        check.metrics = {**summary, "excluded_by_contract": self.fx.excluded}
        if problem:
            check.tolerance_key, check.bound = "float.embedding_max_abs", _bound(self.tol, "float.embedding_max_abs")
            check.status, check.detail = "failed", problem
            self._add(check)
        else:
            self._judge(
                check, "float.embedding_max_abs", summary.get("max_abs", math.inf), summary.get("worst_case", "")
            )
        cos_check = Check("dotnet-ids.min_cosine", "dotnet-ids", "Embedding vs the reference, minimum cosine (float64)")
        cos_check.metrics = {"min_cosine": min_cos, "cases": summary.get("cases", 0)}
        self._judge(cos_check, "float.min_cosine", min_cos, worst_cos, larger_is_worse=False)

        finite = Check("dotnet-ids.finite", "dotnet-ids", "No NaN or Inf in embeddings or tensors", "float.nan_or_inf")
        finite.bound = self.tol.get("float.nan_or_inf")
        self._tensors(stage, non_finite)
        finite.status = "failed" if non_finite else "passed"
        finite.detail = f"non-finite values in {non_finite}" if non_finite else ""
        self._add(finite)
        self._batches(stage, emb)
        self._provider_options(stage)

    def _tensors(self, stage: results.StageResults, non_finite: list[str]) -> None:
        for name, key, ref_name in (
            ("soft_tokens", "float.soft_tokens_row_relative", "image{k}_soft_tokens"),
            ("token_embeddings", "float.token_embeddings_row_relative", "token_embeddings"),
        ):
            check = Check(f"dotnet-ids.{name}", "dotnet-ids", f"{name}, row-relative max |d| / max |ref|")
            if name not in stage.tensor_specs:
                check.tolerance_key, check.detail = key, f"stage 'dotnet' has no {name!r} tensor"
                self._add(check)
                continue
            tensor = stage.tensor(name)
            if self.reference_tensors is None:
                check.tolerance_key = key
                check.detail = (
                    "reference tensors not given (--reference-tensors <dir of run_embeddinggemma2_image.py .npz>)"
                )
                self._add(check)
                continue
            worst, worst_id, compared, missing = 0.0, "", 0, []
            for cid in self.fx.case_ids:
                row = self.fx.rows[cid]
                keys = [f"{cid}/image{k}" for k in range(len(row.images))] if name == "soft_tokens" else [cid]
                if name == "token_embeddings" and not row.images:
                    continue  # text-only rows run on the released text graph (onnx-fp32-cpu evidence)
                for k, tid in enumerate(keys):
                    if tid not in tensor.values:
                        missing.append(tid)
                        continue
                    values = tensor.values[tid]
                    if any(not math.isfinite(v) for v in values):
                        non_finite.append(tid)
                        continue
                    ref = ref_name.format(k=k)
                    digest = self.fx.tensor_digests[cid][ref]["sha256"]
                    shape, expected = read_npz_float32(self.reference_tensors / f"{cid}.npz", ref, digest)
                    if list(tensor.shapes[tid]) != shape:
                        missing.append(f"{tid} (shape {tensor.shapes[tid]} vs {shape})")
                        continue
                    rel = _row_relative(values, expected)
                    compared += 1
                    if rel > worst:
                        worst, worst_id = rel, tid
            check.metrics = {"max_row_relative": worst, "worst_case": worst_id, "cases": compared}
            if missing:
                check.tolerance_key, check.bound = key, _bound(self.tol, key)
                check.status, check.detail = "failed", f"missing or misshapen: {missing}"
                self._add(check)
            else:
                self._judge(check, key, worst, worst_id)

    def _batches(self, stage: results.StageResults, singles: dict[str, list[float]]) -> None:
        key = "float.padded_batch_vs_single_embedding_max_abs"
        if not stage.batches:
            self._add(
                Check("dotnet-ids.batch_evidence", "dotnet-ids", "Batches vs single rows", key, detail="no batches")
            )
            return
        for name, batch in stage.batches.items():
            check = Check(f"dotnet-ids.batch.{name}", "dotnet-ids", f"Batch {name!r} vs single rows")
            stats = ErrorStats()
            missing = [c for c in batch["case_ids"] if c not in singles]
            for cid, values in zip(batch["case_ids"], batch["values"], strict=True):
                if cid in singles:
                    stats.add(cid, values, singles[cid], cosine=False)
            summary = stats.summary()
            check.metrics = summary
            if missing or batch["output"] != "embedding":
                check.tolerance_key, check.bound = key, _bound(self.tol, key)
                check.status = "invalid"
                check.detail = f"batch output {batch['output']!r} or cases without a single embedding {missing}"
                self._add(check)
            else:
                self._judge(check, key, summary.get("max_abs", math.inf), summary.get("worst_case", ""))

    def _provider_options(self, stage: results.StageResults) -> None:
        check = Check("dotnet-ids.provider_options", "dotnet-ids", "Provider options of the run", "provider_options")
        want = self.tol.get("provider_options") or {}
        have = stage.environment.get("provider_options") or {}
        check.bound = want
        wrong = {k: have.get(k) for k, v in want.items() if str(have.get(k)) != str(v)}
        check.status = "failed" if wrong else "passed"
        check.detail = f"run used {wrong}, the profile pins {want}" if wrong else ""
        self._add(check)

    def check_not_applicable(self) -> None:
        for substage, why in (
            ("export", "the image tolerances bind the C# path; Python ORT parity: protected parity-cpu.json"),
            ("dotnet-text", "text-only rows are stage 3 rows (released text graph)"),
        ):
            self._add(
                Check(
                    f"{substage}.not_applicable",
                    substage,
                    SUBSTAGE_TITLES[substage],
                    status="not-applicable",
                    detail=why,
                )
            )

    def check_unknown_keys(self) -> None:
        unknown = sorted(k for k in self.tol if k not in KNOWN_KEYS and k != "reference")
        if unknown:
            gate_sub = "tokenizer" if all(k.startswith("discrete.") for k in unknown) else "dotnet-ids"
            self._add(
                Check(
                    f"{gate_sub}.unknown_tolerance_keys",
                    gate_sub,
                    "Tolerance keys this evaluator does not implement",
                    detail=f"not evaluated: {unknown}",
                )
            )

    def run(self) -> list[Check]:
        self.check_identity()
        self.check_preprocessing()
        self.check_dotnet()
        self.check_not_applicable()
        self.check_unknown_keys()
        return self.checks


def diagnose(checks: list[Check]) -> tuple[str | None, list[dict[str, str]]]:
    status = {s: substage_status(checks, s) for s in SUBSTAGES}
    first = next((s for s in SUBSTAGES if status[s] == "failed"), None)
    findings = []
    for c in checks:
        if c.status == "invalid":
            findings.append({"stage": c.substage, "taxonomy": "packaging/harness", "finding": f"{c.id}: {c.detail}"})
        elif c.status == "failed":
            taxonomy = {
                "tokenizer": "rendering or preprocessing",
                "dotnet-ids": "numerics" if c.tolerance_key and c.tolerance_key.startswith("float.") else "integration",
            }.get(c.substage, "other")
            findings.append({"stage": c.substage, "taxonomy": taxonomy, "finding": f"{c.id}: {c.detail}"})
    return first, findings


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    if value is None:
        return ""
    return str(value)


def _fmt_bound(value: Any) -> str:
    return f"{value:.10g}" if isinstance(value, float) else _fmt(value)


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Fidelity report: {report['profile']} ({report['task']})",
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
        "Bounds come only from the protected tolerances file. Row-relative: per image (soft tokens) or row (token "
        "embeddings), max |d| / max |ref|.",
        "",
        "| Check | Tolerance key | Bound | Max abs | Row-relative | Min cosine | Cases | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for c in report["checks"]:
        m = c["metrics"]
        cos = m.get("min_cosine")
        lines.append(
            f"| `{c['id']}` | {c['tolerance_key'] or ''} | {_fmt_bound(c['bound'])} | {_fmt(m.get('max_abs'))} | "
            f"{_fmt(m.get('max_row_relative'))} | {f'{cos:.12f}' if isinstance(cos, float) else ''} | "
            f"{_fmt(m.get('cases', m.get('images')))} | {c['status']} |"
        )
    details = [c for c in report["checks"] if c["detail"]]
    if details:
        lines += ["", "Details:", ""]
        lines += [f"- `{c['id']}`: {c['detail']}" for c in details]
    pe = report["protected_evidence"]
    lines += ["", "## Protected evidence", "", f"Status **{pe['status']}** ({pe['detail']}; base `{pe['base']}`).", ""]
    lines += ["| File | SHA-256 |", "| --- | --- |"]
    lines += [f"| `{p}` | `{d}` |" for p, d in pe["sha256"].items()]
    lines += ["", "## Candidate inputs", "", "| Stage | Candidate | Producer | Machine | Results SHA-256 |"]
    lines += ["| --- | --- | --- | --- | --- |"]
    for name, r in report["inputs"]["results"].items():
        commit = r["producer"].get("repository_commit")
        lines.append(
            f"| {name} | {r['candidate']} | {r['producer'].get('name', '')}{f' `{commit}`' if commit else ''} | "
            f"{r['environment'].get('machine', '')} | `{r['sha256'][:16]}` |"
        )
    if report.get("reference_tensors"):
        rt = report["reference_tensors"]
        lines += [
            "",
            f"Reference tensors: `{rt['directory']}` (each array checked against the protected tensors.json).",
        ]
    return "\n".join(lines) + "\n"


def sha256_file(path: Path) -> str:
    return fixtures.sha256_file(path)
