"""G1 on a tokenizer fixture file: a tokenizer holdout or release set (P-030).

`model-pipeline evaluate <manifest> --tokenizer-fixtures <protected .jsonl> --results <dir> --out <dir>` compares the
owned tokenizer's outputs on every row of one protected tokenizer fixture file with the reference rows, exactly. The
candidate directory is the output of `Starlings.Fidelity.Tool -- tokenizer` (`tokenizer.candidate.jsonl`,
`environment.json`).

Two fixture formats:

- `encoding` rows (reference `tokenizers` on the pinned tokenizer.json: e5, EmbeddingGemma 2, Qwen3.5): ids, attention
  mask, offsets, tokens, type ids, special-tokens mask, decode with and without special tokens.
- `whitespace_splitter` rows (GLiNER2's processor): normalized text, words, word character offsets, schema tokens,
  input ids, word and schema positions.

Which fields bind G1 comes only from the profile's protected tolerances (`discrete.*`, all exact); the other fields
are measured and reported, never gating. Offsets bind only in the manifest's `behavior.offsets_unit` (none: not
applicable). Every fixture row is required; a candidate row for another id is invalid. The candidate's tokenizer
digest must be one of the manifest's `processing.tokenizer.assets`. The fixture file and the tolerances must equal
the protected base in Git, as for the staged evaluator.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_pipeline import fixtures, manifest, protection, results
from model_pipeline.evaluate import (
    EXIT_FAILED,
    EXIT_GATE_NOT_PASSED,
    EXIT_OK,
    EXIT_PROTECTED,
    Evaluation,
    EvaluationError,
    load_tolerances,
)

REPORT_FORMAT = "starlings.tokenizer-fixtures-report"
REPORT_FORMAT_VERSION = 1
CANDIDATE_FILE = "tokenizer.candidate.jsonl"
ENVIRONMENT_FILE = "environment.json"
MAX_LISTED = 20

# (check id, path into the row, tolerance keys that bind it). Offsets are added per unit below.
ENCODING_FIELDS: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    ("ids", ("encoding", "ids"), ("token_ids", "prompt_ids")),
    ("attention_mask", ("encoding", "attention_mask"), ("attention_mask",)),
    ("tokens", ("encoding", "tokens"), ()),
    ("type_ids", ("encoding", "type_ids"), ()),
    ("special_tokens_mask", ("encoding", "special_tokens_mask"), ()),
    ("decode.skip_special_tokens", ("decode", "skip_special_tokens"), ()),
    ("decode.keep_special_tokens", ("decode", "keep_special_tokens"), ()),
]
OFFSET_UNITS = ("unicode-scalars", "utf16-code-units", "utf8-bytes")

SPLITTER_FIELDS: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    ("normalized_text", ("whitespace_splitter", "normalized_text"), ("normalized_text",)),
    ("words", ("whitespace_splitter", "words", "*word"), ("words",)),
    ("word_offsets", ("whitespace_splitter", "words", "*offsets.unicode-scalars"), ("character_maps",)),
    ("schema_tokens", ("whitespace_splitter", "schema_tokens"), ()),
    ("input_ids", ("whitespace_splitter", "input_ids"), ("token_ids",)),
    ("word_positions", ("whitespace_splitter", "text_word_first_positions"), ("word_positions",)),
    ("schema_positions", ("whitespace_splitter", "schema_special_positions"), ("schema_positions",)),
]


@dataclass
class FieldCheck:
    id: str
    tolerance_key: str | None
    binding: bool
    status: str = "not-run"
    cases: int = 0
    mismatched: list[str] = field(default_factory=list)
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "gate": "G1" if self.binding else None,
            "tolerance_key": self.tolerance_key,
            "binding": self.binding,
            "status": self.status,
            "cases": self.cases,
            "mismatches": len(self.mismatched),
            "mismatched_ids": self.mismatched[:MAX_LISTED],
            "detail": self.detail,
        }


def _get(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = row
    for part in path:
        if part.startswith("*"):
            # A list of objects: project each item on a dotted sub-path.
            sub = part[1:].split(".")
            if not isinstance(value, list):
                return None
            return [_get(item, tuple(sub)) for item in value]
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _read_rows(path: Path, what: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in fixtures.read_jsonl(path):
        rid = row.get("id")
        if not isinstance(rid, str):
            raise EvaluationError(f"{path}: a {what} row has no string id")
        if rid in rows:
            raise EvaluationError(f"{path}: {what} id {rid!r} appears twice")
        rows[rid] = row
    return rows


def _fixture_format(rows: dict[str, dict[str, Any]], path: Path) -> str:
    kinds = {
        "encoding" if "encoding" in r else "whitespace_splitter" if "whitespace_splitter" in r else "?"
        for r in rows.values()
    }
    if len(kinds) != 1 or "?" in kinds:
        raise EvaluationError(
            f"{path}: rows must all carry 'encoding' or all 'whitespace_splitter' (found {sorted(kinds)})"
        )
    return kinds.pop()


def _field_specs(fmt: str, offsets_unit: str) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    if fmt == "whitespace_splitter":
        return list(SPLITTER_FIELDS)
    specs = list(ENCODING_FIELDS)
    for unit in OFFSET_UNITS:
        specs.append((f"offsets.{unit}", ("encoding", "offsets", unit), ("offsets",) if unit == offsets_unit else ()))
    return specs


def evaluate(
    manifest_path: Path,
    fixture_file: str,
    results_dir: Path,
    *,
    repo_root: Path | None = None,
    protected_base: str = "auto",
    now: dt.datetime | None = None,
) -> Evaluation:
    problems = manifest.validate_file(manifest_path)
    if problems:
        raise EvaluationError("manifest is invalid: " + "; ".join(str(p) for p in problems))
    man = manifest.load(manifest_path)
    root = repo_root or protection.find_repository_root(manifest_path.parent)
    qual = man["qualification"]
    fx_path = protection.resolve_protected(root, fixture_file, kind="fixtures")
    if not fx_path.is_file() or fx_path.suffix != ".jsonl":
        raise EvaluationError(f"tokenizer fixtures {fixture_file!r} must be a .jsonl file")
    tol_path = protection.resolve_protected(root, qual["tolerances"], kind="tolerances", must_exist=False)
    if not tol_path.name.startswith("tolerances.") or not tol_path.is_file():
        raise protection.ProtectedEvidenceError(f"{qual['tolerances']} is not an existing protected tolerances.* file")
    tol = load_tolerances(tol_path, man["id"])
    discrete = tol.get("discrete") or {}
    verification = protection.verify_unchanged(root, [tol_path, fx_path], protected_base)

    cand_path = results_dir / CANDIDATE_FILE
    env_path = results_dir / ENVIRONMENT_FILE
    for p in (cand_path, env_path):
        if not p.is_file():
            raise EvaluationError(f"{p} is missing (Starlings.Fidelity.Tool -- tokenizer writes it)")
    env = json.loads(env_path.read_text(encoding="utf-8"))
    forbidden = results._find_forbidden(env)
    expected = _read_rows(fx_path, "fixture")
    actual = _read_rows(cand_path, "candidate")
    for row in actual.values():
        forbidden += results._find_forbidden(row)
    if forbidden:
        raise results.ToleranceOverrideError(
            f"{results_dir}: threshold-like keys are refused: {sorted(set(forbidden))}"
        )
    fmt = _fixture_format(expected, fx_path)
    offsets_unit = man["behavior"].get("offsets_unit", "none")

    checks: list[FieldCheck] = []
    # Identity: the candidate ran the manifest's tokenizer asset on this fixture file.
    ident = FieldCheck("identity", None, True)
    pinned = {a["sha256"] for a in man["processing"]["tokenizer"].get("assets", [])}
    cand_sha = (env.get("tokenizer") or {}).get("sha256")
    fx_sha = fixtures.sha256_file(fx_path)
    problems_ = []
    if cand_sha not in pinned:
        problems_.append(f"candidate tokenizer SHA-256 {cand_sha} is not a processing.tokenizer asset of the manifest")
    if (env.get("fixtures") or {}).get("sha256") != fx_sha:
        problems_.append("the candidate ran on another fixture file (SHA-256 differs)")
    ident.status = "invalid" if problems_ else "passed"
    ident.detail = "; ".join(problems_)
    checks.append(ident)

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    cover = FieldCheck("cases", None, True, cases=len(expected))
    cover.mismatched = missing + extra
    cover.status = "failed" if missing else "invalid" if extra else "passed"
    cover.detail = "; ".join(
        x
        for x in (f"missing cases {missing[:MAX_LISTED]}" if missing else "", f"unknown cases {extra}" if extra else "")
        if x
    )
    checks.append(cover)

    for cid, path, keys in _field_specs(fmt, offsets_unit):
        key = next((k for k in keys if k in discrete), None)
        binding = key is not None
        c = FieldCheck(cid, f"discrete.{key}" if key else None, binding)
        if key is not None and str(discrete[key]).lower() != "exact":
            c.status, c.detail = "not-run", f"discrete.{key} is {discrete[key]!r}; only 'exact' is implemented"
            checks.append(c)
            continue
        compared = 0
        absent = 0
        for rid, ref_row in expected.items():
            ref = _get(ref_row, path)
            if ref is None:
                continue
            if rid not in actual:
                continue
            got = _get(actual[rid], path)
            if got is None:
                absent += 1
                continue
            compared += 1
            if got != ref:
                c.mismatched.append(rid)
        c.cases = compared
        if compared == 0 and absent == 0:
            c.status, c.detail = "not-applicable", "the fixture rows do not carry this field"
        elif absent:
            c.status = "not-run"
            c.detail = f"the candidate does not carry this field on {absent} rows"
        else:
            c.status = "failed" if c.mismatched else "passed"
        if not binding and c.status == "failed":
            c.status = "differs"
        checks.append(c)
    # Offsets in a unit the profile does not expose: say why they do not bind.
    if fmt == "encoding" and offsets_unit == "none" and "offsets" in discrete:
        for c in checks:
            if c.id.startswith("offsets."):
                c.detail = (c.detail + "; " if c.detail else "") + (
                    "behavior.offsets_unit is none: the profile exposes no offsets; discrete.offsets does not apply"
                )

    gate = _gate(checks, verification)
    date = (now or dt.datetime.now(dt.UTC)).date().isoformat()

    def rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(root.resolve())).replace("\\", "/")
        except ValueError:
            return str(p)

    report = {
        "format": REPORT_FORMAT,
        "format_version": REPORT_FORMAT_VERSION,
        "task": "tokenizer-fixtures",
        "profile": man["id"],
        "date": date,
        "evaluator": {
            "name": "model-pipeline evaluate --tokenizer-fixtures",
            "repository_commit": fixtures.git_head(root),
        },
        "inputs": {
            "manifest": {"path": rel(manifest_path), "sha256": fixtures.sha256_file(manifest_path)},
            "tolerances": {"path": rel(tol_path), "sha256": fixtures.sha256_file(tol_path)},
            "fixtures": {"path": rel(fx_path), "sha256": fx_sha, "format": fmt, "rows": len(expected)},
            "results": {
                "path": rel(cand_path),
                "sha256": fixtures.sha256_file(cand_path),
                "environment": env,
            },
        },
        "offsets_unit": offsets_unit,
        "protected_evidence": verification.as_dict(),
        "gates": {"G1": gate},
        "manifest_gates": {"G1": {"status": gate["status"], "date": date}} if gate["status"] != "not-run" else {},
        "checks": [c.as_dict() for c in checks],
    }
    if verification.status == "changed":
        code = EXIT_PROTECTED
    elif any(c.binding and c.status in ("failed", "invalid") for c in checks):
        code = EXIT_FAILED
    elif gate["status"] != "passed":
        code = EXIT_GATE_NOT_PASSED
    else:
        code = EXIT_OK
    report["exit_code"] = code
    return Evaluation(report, code)


def _gate(checks: list[FieldCheck], verification: protection.Verification) -> dict[str, str]:
    mine = [c for c in checks if c.binding]
    failed = [c.id for c in mine if c.status in ("failed", "invalid")]
    open_ = [c.id for c in mine if c.status == "not-run"]
    bound = [c for c in mine if c.tolerance_key]
    if verification.status == "changed":
        return {
            "status": "not-run",
            "reason": "protected evidence differs from the base commit; this run is not gate evidence",
        }
    if failed:
        return {"status": "failed", "reason": f"failing checks: {', '.join(failed)}"}
    if open_:
        return {"status": "not-run", "reason": f"checks not run: {', '.join(open_)}"}
    if not any(c.status == "passed" for c in bound):
        return {"status": "not-run", "reason": "no field bound by the protected tolerances was compared"}
    if verification.status != "verified":
        return {"status": "not-run", "reason": f"protected evidence not verified against Git ({verification.detail})"}
    return {"status": "passed", "reason": "every bound field is exact on every fixture row"}


def markdown(report: dict[str, Any]) -> str:
    fx = report["inputs"]["fixtures"]
    env = report["inputs"]["results"]["environment"]
    lines = [
        f"# Tokenizer fixtures report: {report['profile']}",
        "",
        f"Date {report['date']}. Evaluator {report['evaluator']['name']} at "
        f"`{report['evaluator']['repository_commit']}`. Exit code {report['exit_code']}.",
        "",
        f"Fixtures: `{fx['path']}` ({fx['rows']} rows, format `{fx['format']}`, SHA-256 `{fx['sha256']}`). "
        f"Candidate tokenizer SHA-256 `{(env.get('tokenizer') or {}).get('sha256')}`; producer "
        f"{(env.get('producer') or {}).get('name')} `{(env.get('producer') or {}).get('repository_commit')}`. "
        f"Manifest offsets unit: `{report['offsets_unit']}`.",
        "",
        "## Gate",
        "",
        "| Gate | Status | Reason |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {g} | {v['status']} | {v['reason']} |" for g, v in report["gates"].items()]
    lines += [
        "",
        "## Checks",
        "",
        "Exact comparison per row. Binding fields are the protected tolerances' `discrete` keys; the others are "
        "measured and never gate.",
        "",
        "| Check | Tolerance key | Binding | Rows compared | Mismatches | Status |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for c in report["checks"]:
        lines.append(
            f"| `{c['id']}` | {c['tolerance_key'] or ''} | {'yes' if c['binding'] else 'no'} | {c['cases']} | "
            f"{c['mismatches']} | {c['status']} |"
        )
    details = [c for c in report["checks"] if c["detail"] or c["mismatched_ids"]]
    if details:
        lines += ["", "Details:", ""]
        for c in details:
            ids = (
                f" Mismatched rows (first {MAX_LISTED}): {', '.join(c['mismatched_ids'])}."
                if c["mismatched_ids"]
                else ""
            )
            lines.append(f"- `{c['id']}`: {c['detail']}{ids}".rstrip())
    pe = report["protected_evidence"]
    lines += ["", "## Protected evidence", "", f"Status **{pe['status']}** ({pe['detail']}; base `{pe['base']}`).", ""]
    lines += ["| File | SHA-256 |", "| --- | --- |"]
    lines += [f"| `{p}` | `{d}` |" for p, d in pe["sha256"].items()]
    return "\n".join(lines) + "\n"


def write_report(evaluation: Evaluation, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / "report.json", out_dir / "report.md"
    fixtures.write_json(json_path, evaluation.report)
    md_path.write_text(markdown(evaluation.report), encoding="utf-8", newline="\n")
    return json_path, md_path
