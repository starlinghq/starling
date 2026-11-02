"""Generate the hand-maintained onboarding inputs of a bert-onnx-cpu-v1 candidate from its profile.json.

Adding a repository to the BERT route used to mean editing the recipe, both allow-lists, the fetch script, the
package README and the contract tests by hand. This module derives every one of them from the candidate contract
(``candidates/<name>/profile.json``), so a contributor writes one file and reviews the rest. It changes no gate,
tolerance, protected file or maturity: the route still accepts the repository only through the contract table it
writes, and the evidence still comes from a full ``model-pipeline run``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from model_pipeline import onboarding_bert as bert
from model_pipeline.onboarding_candidate import load_candidate
from model_pipeline.onboarding_runner import PipelineError

CONTRACTS = "model-pipeline/routes/bert-onnx-cpu-v1/contracts.json"
FETCH_SCRIPT = "tools/fetch-tokenizer-assets.sh"
README = "ai-dotnet/src/Starlings.Models.Bert/README.md"
CONTRACT_TESTS = "ai-dotnet/tests/Starlings.Models.Bert.Tests/ContractTests.cs"
REFERENCE_ENVIRONMENT = "model-pipeline/reference/encoder/requirements.txt"
CORPORA = {
    "embeddings": "model-pipeline/corpora/five-models/embeddings.jsonl",
    "reranking": "model-pipeline/corpora/five-models/reranking.jsonl",
}
# Special ids and segment-id input of each tokenizer profile the route implements (BertModelProfile, run_bert.py).
TOKENIZER_DEFAULTS = {
    "bert-wordpiece-uncased": {"cls_token_id": 101, "sep_token_id": 102, "token_type_ids": True},
    "xlmr-unigram": {"cls_token_id": 0, "sep_token_id": 2, "token_type_ids": False},
}
_NAME = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
_TEST_ANCHOR = "    public void ParsesOnlyDeclaredTaskContracts("
_README_HEADER = "| Candidate | Profile id | Task | Max tokens | Pooling | Width | Normalize | Prompts |"
_FETCH_START = "assets=("


def _candidate_path(root: Path, candidate: str) -> tuple[str, Path]:
    path = Path(candidate)
    if path.suffix == ".json":
        resolved = path.resolve()
        name = resolved.parent.name
    else:
        name = candidate
        resolved = root / "model-pipeline" / "candidates" / name / "profile.json"
    if not _NAME.match(name):
        raise PipelineError(f"candidate name {name!r} must be lowercase letters, digits, '.' and '-'")
    if resolved != (root / "model-pipeline" / "candidates" / name / "profile.json").resolve():
        raise PipelineError("the candidate must be model-pipeline/candidates/<name>/profile.json in this repository")
    if not resolved.is_file():
        raise PipelineError(f"candidate contract not found: {resolved}")
    return name, resolved


def derive_contract(candidate: dict, *, token_type_ids: bool | None = None) -> dict[str, Any]:
    """The route contract table entry a BERT candidate implies (tokenizer-profile defaults for ids not in it).

    The candidate does not record whether the graph takes a segment-id input: ``token_type_ids`` states it, and
    ``None`` takes the tokenizer profile's default (a BERT encoder can pair an XLM-R tokenizer with segment ids).
    """
    behavior = candidate["behavior"]
    if behavior.get("architecture") != "bert":
        raise PipelineError("scaffold supports the bert-onnx-cpu-v1 route only (behavior.architecture must be bert)")
    profile = behavior.get("tokenizer_profile")
    if profile not in TOKENIZER_DEFAULTS:
        raise PipelineError(f"tokenizer_profile must be one of {sorted(TOKENIZER_DEFAULTS)}")
    return {
        "task": candidate["task"],
        "max_tokens": behavior["max_tokens"],
        "pooling": behavior["pooling"],
        "hidden_size": behavior["hidden_size"],
        "normalize": behavior["normalize"],
        "vocabulary_size": behavior["vocabulary_size"],
        "cls_token_id": TOKENIZER_DEFAULTS[profile]["cls_token_id"],
        "sep_token_id": TOKENIZER_DEFAULTS[profile]["sep_token_id"],
        "pad_token_id": behavior["pad_token_id"],
        "output_name": behavior["output_name"],
        "tokenizer_profile": profile,
        "token_type_ids": TOKENIZER_DEFAULTS[profile]["token_type_ids"] if token_type_ids is None else token_type_ids,
        "prompts": behavior["prompts"],
    }


def _file(candidate: dict, path: str) -> dict:
    for entry in candidate["origin"]["files"]:
        if entry["path"] == path:
            return entry
    raise PipelineError(f"the candidate pins no {path}")


def _pascal(name: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in re.split(r"[-.]", name) if part)


def _csharp_lock(name: str, candidate: dict) -> str:
    origin = candidate["origin"]
    paths = ["onnx/model.onnx"]
    if any(entry["path"] == "onnx/model.onnx_data" for entry in origin["files"]):
        paths.append("onnx/model.onnx_data")
    paths.append("tokenizer.json")
    files = ",\n".join(
        f'        new ModelAssetFile("{p}", {_file(candidate, p)["size"]}, "{_file(candidate, p)["sha256"]}")'
        for p in paths
    )
    return (
        f"/// <summary>Runtime files of {candidate['id']} (generated by model-pipeline scaffold {name}).</summary>\n"
        f"public static ModelAssetLock {_pascal(name)}RuntimeLock {{ get; }} = new(\n"
        f'    "{candidate["id"]}", "{origin["repo"]}", "{origin["revision"]}",\n'
        f"    [\n{files},\n    ]);\n"
    )


def _readme_row(name: str, candidate: dict, contract: dict) -> str:
    width = str(contract["hidden_size"]) + (" (one logit out)" if contract["task"] == "reranking" else "")
    prompts = ", ".join(f"`{key}`" for key in contract["prompts"])
    return (
        f"| `{name}` | `{candidate['id']}` | {contract['task']} | {contract['max_tokens']} | {contract['pooling']} | "
        f"{width} | {contract['normalize']} | {prompts} |"
    )


def _test_row(repo: str, contract: dict) -> str:
    task = "BertTask.Embeddings" if contract["task"] == "embeddings" else "BertTask.Reranking"
    return (
        f'    [InlineData("{repo}", {contract["max_tokens"]}, "{contract["pooling"]}", {task}, '
        f"{contract['hidden_size']})]"
    )


def _fetch_line(candidate: dict) -> str:
    origin = candidate["origin"]
    digest = _file(candidate, "tokenizer.json")["sha256"]
    return f'    "{origin["repo"]}|{origin["revision"]}|tokenizer.json|{digest}"'


def _recipe(name: str, corpus: str) -> str:
    return (
        "recipe_version: 1\n"
        f"id: {name}-cpu\n"
        f"plugin: {bert.PLUGIN}\n"
        f"manifest: model-pipeline/candidates/{name}/profile.json\n"
        f"reference_corpus: {corpus}\n"
        f"reference_environment: {REFERENCE_ENVIRONMENT}\n"
    )


def _insert_line(text: str, line: str, *, before: str | None = None, after_block_of: str | None = None) -> str:
    """Insert a line right before an anchor line, or after the last line of the block that follows a header."""
    lines = text.split("\n")
    if before is not None:
        index = next((i for i, current in enumerate(lines) if current.startswith(before)), None)
        if index is None:
            raise PipelineError(f"anchor not found: {before!r}")
        # The anchor's [InlineData] rows end right above it; the new row becomes the last one.
        lines.insert(index, line)
        return "\n".join(lines)
    assert after_block_of is not None
    index = next((i for i, current in enumerate(lines) if current.startswith(after_block_of)), None)
    if index is None:
        raise PipelineError(f"anchor not found: {after_block_of!r}")
    end = index + 1
    while end < len(lines) and lines[end].startswith(("|", '    "')):
        end += 1
    lines.insert(end, line)
    return "\n".join(lines)


def scaffold(
    root: Path,
    candidate_name: str,
    *,
    corpus: str | None = None,
    write: bool = False,
    token_type_ids: bool | None = None,
) -> dict[str, Any]:
    """Describe (and with ``write``, apply) every generated input of one BERT-route candidate.

    ``token_type_ids`` states whether the graph takes segment ids; ``None`` keeps an existing table entry's value
    (the candidate cannot contradict it) or, for a new entry, takes the tokenizer profile's default.
    """
    root = root.resolve()
    name, path = _candidate_path(root, candidate_name)
    candidate = load_candidate(path)
    repo = candidate["origin"]["repo"]
    contracts_path = root / CONTRACTS
    document = json.loads(contracts_path.read_text(encoding="utf-8"))
    existing = document["contracts"].get(repo)
    if token_type_ids is None and existing is not None and type(existing.get("token_type_ids")) is bool:
        token_type_ids = existing["token_type_ids"]
    contract = derive_contract(candidate, token_type_ids=token_type_ids)
    if existing is not None:
        differing = sorted(key for key, value in contract.items() if existing.get(key) != value)
        if differing:
            raise PipelineError(f"{repo} is in {CONTRACTS} with different {differing}; change one of the two")
        contract = existing
    else:
        document["contracts"][repo] = contract
    contracts = bert.validate_contracts(document)
    bert.validate_candidate_behavior(candidate, contracts)

    corpus = corpus or CORPORA[contract["task"]]
    if not (root / corpus).is_file():
        raise PipelineError(f"reference corpus not found: {corpus}")
    recipe_path = f"model-pipeline/recipes/{name}-cpu.yaml"
    pieces: list[dict[str, Any]] = []

    def plan(target: str, content: str, present: bool, *, kind: str = "insert", new_text: str | None = None) -> None:
        pieces.append(
            {
                "path": target,
                "action": "unchanged" if present else kind,
                "content": content,
                **({"_text": new_text} if new_text is not None and not present else {}),
            }
        )

    recipe = _recipe(name, corpus)
    recipe_file = root / recipe_path
    if recipe_file.exists():
        # An existing recipe is kept (its corpus may be a deliberate choice) if it runs this candidate on this route.
        recipe = recipe_file.read_text(encoding="utf-8")
        existing_recipe = yaml.safe_load(recipe)
        expected = {"plugin": bert.PLUGIN, "manifest": f"model-pipeline/candidates/{name}/profile.json"}
        if not isinstance(existing_recipe, dict) or any(existing_recipe.get(k) != v for k, v in expected.items()):
            raise PipelineError(f"{recipe_path} exists but does not run this candidate on {bert.PLUGIN}")
    plan(recipe_path, recipe, recipe_file.exists(), kind="create", new_text=recipe)

    table = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    plan(CONTRACTS, json.dumps({repo: contract}, indent=2, ensure_ascii=False), existing is not None, new_text=table)

    fetch = _fetch_line(candidate)
    fetch_text = (root / FETCH_SCRIPT).read_text(encoding="utf-8")
    plan(
        FETCH_SCRIPT,
        fetch,
        fetch in fetch_text.split("\n"),
        new_text=_insert_line(fetch_text, fetch, after_block_of=_FETCH_START),
    )

    row = _readme_row(name, candidate, contract)
    readme_text = (root / README).read_text(encoding="utf-8")
    plan(
        README,
        row,
        f"| `{name}` |" in readme_text,
        new_text=_insert_line(readme_text, row, after_block_of=_README_HEADER),
    )

    test = _test_row(repo, contract)
    tests_text = (root / CONTRACT_TESTS).read_text(encoding="utf-8")
    plan(
        CONTRACT_TESTS,
        test,
        f'[InlineData("{repo}",' in tests_text,
        new_text=_insert_line(tests_text, test, before=_TEST_ANCHOR),
    )

    pieces.append(
        {
            "path": None,
            "action": "print",
            "content": _csharp_lock(name, candidate),
            "note": "Starlings.Models.Bert derives its locks from profile.json at run time; paste this static lock "
            "where a test or another package needs the runtime files without the candidate file.",
        }
    )
    if write:
        for piece in pieces:
            text = piece.pop("_text", None)
            if text is not None:
                target = root / piece["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
                piece["action"] = {"create": "created", "insert": "inserted"}[piece["action"]]
    else:
        for piece in pieces:
            piece.pop("_text", None)
    return {
        "candidate": f"model-pipeline/candidates/{name}/profile.json",
        "plugin": bert.PLUGIN,
        "repository": repo,
        "written": write,
        "files": pieces,
        "python_tests": "test_onboarding_bert.py is parametrized over the contract table; no row to add.",
        "next": [
            f"model-pipeline plan {recipe_path}",
            f"model-pipeline run {recipe_path} --snapshot <pinned snapshot> --reference-python <venv python> "
            f"--out artifacts/onboarding/{name}",
        ],
    }
