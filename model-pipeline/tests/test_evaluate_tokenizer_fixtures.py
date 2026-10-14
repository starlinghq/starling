"""`model-pipeline evaluate --tokenizer-fixtures`: G1 on one protected tokenizer fixture file."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml
from synthetic_profile import PROTECTED, SyntheticRepo, git

from model_pipeline import evaluate, evaluate_tokenizer_fixtures, fixtures
from model_pipeline.cli import main

TOKENIZER_SHA = "0000000000000000000000000000000000000000000000000000000000000002"
HOLDOUT = "model-pipeline/protected/synthetic-encoder/release/tokenizer-release/tokenizer.jsonl"
SPLITTER = "model-pipeline/protected/synthetic-encoder/release/tokenizer-release/splitter.jsonl"


def _encoding_row(rid: str, text: str, ids: list[int]) -> dict:
    n = len(ids)
    spans = [[0, 0]] + [[i, i + 1] for i in range(n - 2)] + [[0, 0]]
    return {
        "id": rid,
        "category": "x",
        "text": text,
        "encoding": {
            "ids": ids,
            "tokens": [f"t{i}" for i in ids],
            "type_ids": [0] * n,
            "attention_mask": [1] * n,
            "special_tokens_mask": [1] + [0] * (n - 2) + [1],
            "offsets": {"unicode-scalars": spans, "utf16-code-units": spans, "utf8-bytes": spans},
        },
        "decode": {"skip_special_tokens": text, "keep_special_tokens": f"<s>{text}</s>"},
    }


ROWS = [_encoding_row("a", "ab", [0, 5, 6, 2]), _encoding_row("b", "xyz", [0, 7, 8, 9, 2])]
SPLITTER_ROWS = [
    {
        "id": "s1",
        "category": "x",
        "text": "Ann went",
        "whitespace_splitter": {
            "normalized_text": "Ann went.",
            "words": [
                {"word": "Ann", "subwords": ["▁Ann"], "offsets": {"unicode-scalars": [0, 3], "utf8-bytes": [0, 3]}},
                {"word": "went", "subwords": ["▁went"], "offsets": {"unicode-scalars": [4, 8], "utf8-bytes": [4, 8]}},
            ],
            "schema_tokens": [["(", "[P]", "entities", "(", "[E]", "person", ")", ")"]],
            "input_ids": [1, 2, 3, 1, 4, 5, 6, 6, 7, 8, 9],
            "text_word_first_positions": [9, 10],
            "schema_special_positions": [[1, 4]],
        },
    }
]


def _candidate(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        c = {k: copy.deepcopy(v) for k, v in r.items() if k in ("id", "encoding", "decode", "whitespace_splitter")}
        if "whitespace_splitter" in c:
            for w in c["whitespace_splitter"]["words"]:
                w.pop("subwords")
                w["offsets"] = {"unicode-scalars": w["offsets"]["unicode-scalars"]}
        out.append(c)
    return out


class Repo(SyntheticRepo):
    def __init__(self, root: Path, offsets_unit: str = "none", gliner_tolerances: bool = False):
        super().__init__(root)
        if offsets_unit != "none" or gliner_tolerances:

            def edit(m):
                m["behavior"]["offsets_unit"] = offsets_unit

            self.write_manifest(edit)
        if gliner_tolerances:
            tol = yaml.safe_load((self.protected / "tolerances.yaml").read_text(encoding="utf-8"))
            tol["discrete"] = {
                k: "exact"
                for k in (
                    "normalized_text",
                    "words",
                    "character_maps",
                    "token_ids",
                    "attention_mask",
                    "word_positions",
                    "schema_positions",
                )
            }
            (self.protected / "tolerances.yaml").write_text(yaml.safe_dump(tol), encoding="utf-8")
        (root / HOLDOUT).parent.mkdir(parents=True)
        fixtures.write_jsonl(root / HOLDOUT, ROWS)
        fixtures.write_jsonl(root / SPLITTER, SPLITTER_ROWS)
        git(root, "add", "-A")
        git(root, "commit", "-q", "-m", "tokenizer release set")
        self.out = root / "tok-results"

    def candidate(self, rows: list[dict], fixture: str = HOLDOUT, sha: str = TOKENIZER_SHA, env_extra=None) -> Path:
        self.out.mkdir(exist_ok=True)
        fixtures.write_jsonl(self.out / evaluate_tokenizer_fixtures.CANDIDATE_FILE, rows)
        env = {
            "format": "starlings.tokenizer-candidate",
            "format_version": 1,
            "producer": {"name": "tests", "repository_commit": None},
            "tokenizer": {"path": "tokenizer.json", "sha256": sha},
            "fixtures": {"path": fixture, "sha256": fixtures.sha256_file(self.root / fixture)},
            **(env_extra or {}),
        }
        fixtures.write_json(self.out / evaluate_tokenizer_fixtures.ENVIRONMENT_FILE, env)
        return self.out


@pytest.fixture
def repo(tmp_path):
    return Repo(tmp_path / "repo")


def run(repo: Repo, fixture: str = HOLDOUT, base: str = "HEAD") -> evaluate.Evaluation:
    return evaluate_tokenizer_fixtures.evaluate(
        repo.manifest, fixture, repo.out, repo_root=repo.root, protected_base=base
    )


def check(ev: evaluate.Evaluation, cid: str) -> dict:
    return next(c for c in ev.report["checks"] if c["id"] == cid)


def test_exact_candidate_passes_g1(repo):
    repo.candidate(_candidate(ROWS))
    ev = run(repo)
    assert ev.exit_code == evaluate.EXIT_OK, ev.report["checks"]
    assert ev.report["gates"]["G1"]["status"] == "passed"
    assert ev.report["manifest_gates"]["G1"]["status"] == "passed"
    assert check(ev, "ids")["binding"] and check(ev, "ids")["tolerance_key"] == "discrete.token_ids"
    assert check(ev, "attention_mask")["binding"]
    # offsets_unit none: offsets are measured, never binding.
    assert not check(ev, "offsets.unicode-scalars")["binding"]
    assert check(ev, "offsets.unicode-scalars")["status"] == "passed"
    assert ev.report["inputs"]["fixtures"]["rows"] == 2
    assert "# Tokenizer fixtures report" in evaluate_tokenizer_fixtures.markdown(ev.report)


def test_id_mismatch_fails_g1(repo):
    rows = _candidate(ROWS)
    rows[1]["encoding"]["ids"][2] = 99
    repo.candidate(rows)
    ev = run(repo)
    assert ev.exit_code == evaluate.EXIT_FAILED
    assert check(ev, "ids")["mismatched_ids"] == ["b"]
    assert ev.report["gates"]["G1"]["status"] == "failed"


def test_unbound_field_difference_is_reported_not_gating(repo):
    rows = _candidate(ROWS)
    rows[0]["decode"]["skip_special_tokens"] = "a b"
    repo.candidate(rows)
    ev = run(repo)
    assert ev.exit_code == evaluate.EXIT_OK
    assert check(ev, "decode.skip_special_tokens")["status"] == "differs"
    assert ev.report["gates"]["G1"]["status"] == "passed"


def test_offsets_bind_in_the_manifest_unit(tmp_path):
    repo = Repo(tmp_path / "repo", offsets_unit="unicode-scalars")
    rows = _candidate(ROWS)
    rows[0]["encoding"]["offsets"]["unicode-scalars"][1] = [0, 2]
    repo.candidate(rows)
    ev = run(repo)
    assert check(ev, "offsets.unicode-scalars")["binding"]
    assert not check(ev, "offsets.utf8-bytes")["binding"]
    assert ev.report["gates"]["G1"]["status"] == "failed"


def test_missing_row_fails_and_unknown_row_is_invalid(repo):
    repo.candidate(_candidate(ROWS)[:1])
    ev = run(repo)
    assert check(ev, "cases")["status"] == "failed"
    assert ev.report["gates"]["G1"]["status"] == "failed"
    extra = _candidate(ROWS) + [_candidate([_encoding_row("zzz", "q", [0, 2])])[0]]
    repo.candidate(extra)
    assert check(run(repo), "cases")["status"] == "invalid"


def test_missing_binding_field_leaves_g1_not_run(repo):
    rows = _candidate(ROWS)
    for r in rows:
        del r["encoding"]["attention_mask"]
    repo.candidate(rows)
    ev = run(repo)
    assert check(ev, "attention_mask")["status"] == "not-run"
    assert ev.report["gates"]["G1"]["status"] == "not-run"
    assert ev.exit_code == evaluate.EXIT_GATE_NOT_PASSED


def test_other_tokenizer_or_fixture_file_is_invalid(repo):
    repo.candidate(_candidate(ROWS), sha="f" * 64)
    ev = run(repo)
    assert check(ev, "identity")["status"] == "invalid"
    assert ev.report["gates"]["G1"]["status"] == "failed"
    repo.candidate(_candidate(ROWS), env_extra={"fixtures": {"path": "x", "sha256": "0" * 64}})
    assert check(run(repo), "identity")["status"] == "invalid"


def test_splitter_format(tmp_path):
    repo = Repo(tmp_path / "repo", offsets_unit="unicode-scalars", gliner_tolerances=True)
    repo.candidate(_candidate(SPLITTER_ROWS), fixture=SPLITTER)
    ev = run(repo, SPLITTER)
    assert ev.exit_code == evaluate.EXIT_OK, ev.report["checks"]
    assert ev.report["inputs"]["fixtures"]["format"] == "whitespace_splitter"
    assert check(ev, "word_offsets")["tolerance_key"] == "discrete.character_maps"
    rows = _candidate(SPLITTER_ROWS)
    rows[0]["whitespace_splitter"]["text_word_first_positions"] = [9, 11]
    repo.candidate(rows, fixture=SPLITTER)
    ev = run(repo, SPLITTER)
    assert check(ev, "word_positions")["status"] == "failed"
    assert ev.report["gates"]["G1"]["status"] == "failed"


def test_uncommitted_fixture_change_is_not_gate_evidence(repo):
    path = repo.root / HOLDOUT
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    repo.candidate(_candidate(ROWS))
    ev = run(repo)
    assert ev.exit_code == evaluate.EXIT_PROTECTED
    assert ev.report["gates"]["G1"]["status"] == "not-run"


def test_cli(repo, tmp_path):
    repo.candidate(_candidate(ROWS))
    out = tmp_path / "o"
    args = ["evaluate", str(repo.manifest), "--results", str(repo.out), "--out", str(out)]
    args += ["--protected-base", "HEAD", "--repo-root", str(repo.root)]
    assert main([*args, "--tokenizer-fixtures", HOLDOUT]) == evaluate.EXIT_OK
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["gates"]["G1"]["status"] == "passed"
    assert (out / "report.md").is_file()


@pytest.mark.parametrize("path", ["tok-results/tokenizer.candidate.jsonl", "/etc/hostname", f"{PROTECTED}/../x.jsonl"])
def test_cli_refuses_paths_outside_protected(repo, tmp_path, path):
    repo.candidate(_candidate(ROWS))
    args = ["evaluate", str(repo.manifest), "--results", str(repo.out), "--out", str(tmp_path / "o")]
    args += ["--protected-base", "HEAD", "--repo-root", str(repo.root)]
    assert main([*args, "--tokenizer-fixtures", path]) == evaluate.EXIT_PROTECTED


def test_cli_refuses_threshold_keys(repo, tmp_path):
    repo.candidate(_candidate(ROWS), env_extra={"threshold": 0.5})
    args = ["evaluate", str(repo.manifest), "--results", str(repo.out), "--out", str(tmp_path / "o")]
    args += ["--protected-base", "HEAD", "--repo-root", str(repo.root)]
    assert main([*args, "--tokenizer-fixtures", HOLDOUT]) == evaluate.EXIT_PROTECTED


def test_cli_rejects_other_modes(repo, tmp_path):
    args = ["evaluate", str(repo.manifest), "--results", str(repo.out), "--out", str(tmp_path / "o")]
    assert main([*args, "--tokenizer-fixtures", HOLDOUT, "--g3", "retrieval"]) == evaluate.EXIT_INPUT
    assert main([*args, "--tokenizer-fixtures", HOLDOUT, "--fixtures", PROTECTED]) == evaluate.EXIT_INPUT
