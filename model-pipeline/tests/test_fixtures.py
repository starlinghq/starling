import json
import struct
from pathlib import Path

import pytest

from model_pipeline import fixtures
from model_pipeline.manifest import find_manifests, validate_file

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "reference" / "corpus" / "tokenizer-corpus-v1.jsonl"
PROTECTED = ROOT / "protected"
E5 = PROTECTED / "multilingual-e5-small" / "onnx-fp32-cpu"
GLINER2 = PROTECTED / "gliner2-base-v1" / "tokenizer"

# Plan section 6.2 corpus coverage that a tokenizer profile must exercise.
REQUIRED_CATEGORIES = {
    "ascii",
    "multilingual",
    "combining",
    "astral",
    "whitespace",
    "control",
    "empty",
    "special-lookalike",
    "truncation",
    "pair",
}


def test_offsets_convert_between_units():
    text = "a😀é"  # 1 + astral + 2-byte
    offsets = [(0, 1), (1, 2), (2, 3), (0, 3)]
    assert fixtures.convert_offsets(text, offsets, "unicode-scalars") == [[0, 1], [1, 2], [2, 3], [0, 3]]
    assert fixtures.convert_offsets(text, offsets, "utf16-code-units") == [[0, 1], [1, 3], [3, 4], [0, 4]]
    assert fixtures.convert_offsets(text, offsets, "utf8-bytes") == [[0, 1], [1, 5], [5, 7], [0, 7]]
    with pytest.raises(ValueError):
        fixtures.convert_offsets(text, offsets, "graphemes")


@pytest.mark.parametrize("value", [0.1, 1 / 3, -2.5e-8, 123456.789, 0.0])
def test_float32_list_round_trips(value):
    (out,) = fixtures.float32_list([value])
    assert struct.pack("<f", out) == struct.pack("<f", value)


def test_corpus_loader_keeps_line_separators_inside_text(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({"id": "a", "category": "x", "text": "x\u0085y z"}, ensure_ascii=False) + "\n")
    assert fixtures.load_corpus(path)[0]["text"] == "x\u0085y z"


def test_corpus_loader_rejects_duplicates(tmp_path):
    path = tmp_path / "c.jsonl"
    row = json.dumps({"id": "a", "category": "x", "text": ""})
    path.write_text(row + "\n" + row + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        fixtures.load_corpus(path)


def test_corpus_covers_plan_categories():
    cases = fixtures.load_corpus(CORPUS)
    assert {c["category"] for c in cases} >= REQUIRED_CATEGORIES
    assert any(c["text"] == "" for c in cases)
    assert any(any(ord(ch) > 0xFFFF for ch in c["text"]) for c in cases)


def test_registry_manifests_validate():
    manifests = find_manifests(ROOT / "registry")
    assert manifests
    for path in manifests:
        assert validate_file(path) == [], path


def _check_encoding(text: str, enc: dict, pair: str | None = None):
    n = len(enc["ids"])
    for key in ("tokens", "type_ids", "attention_mask", "special_tokens_mask"):
        assert len(enc[key]) == n, key
    for unit in fixtures.OFFSET_UNITS:
        assert len(enc["offsets"][unit]) == n
    sources = [pair if s == 1 else text for s in enc.get("sequence_ids", [0] * n)]
    for i, (start, end) in enumerate(enc["offsets"]["unicode-scalars"]):
        source = sources[i]
        assert 0 <= start <= end <= len(source)
        for unit in ("utf16-code-units", "utf8-bytes"):
            assert enc["offsets"][unit][i] == fixtures.convert_offsets(source, [(start, end)], unit)[0]


@pytest.mark.skipif(not E5.exists(), reason="e5 fixtures not present")
def test_e5_fixtures_are_internally_consistent():
    env = json.loads((E5 / "environment.json").read_text(encoding="utf-8"))
    assert env["offset_units"]["recorded"] == list(fixtures.OFFSET_UNITS)
    corpus = {c["id"]: c for c in fixtures.load_corpus(CORPUS)}
    rows = fixtures.read_jsonl(E5 / "tokenizer.jsonl")
    assert {r["id"] for r in rows} == set(corpus)
    for row in rows:
        assert row["text"] == corpus[row["id"]]["text"]
        _check_encoding(row["text"], row["encoding"], row.get("pair"))
        if "truncated" in row:
            assert len(row["truncated"]["encoding"]["ids"]) == 512
            _check_encoding(row["text"], row["truncated"]["encoding"], row.get("pair"))
    for emb in fixtures.read_jsonl(E5 / "embeddings.jsonl"):
        assert len(emb["embedding"]) == 384
        assert abs(sum(x * x for x in emb["embedding"]) ** 0.5 - 1) < 1e-5
        assert 2 <= len(emb["input_ids"]) <= 512


@pytest.mark.skipif(not GLINER2.exists(), reason="GLiNER2 fixtures not present")
def test_gliner2_word_offsets_index_the_normalized_text():
    for row in fixtures.read_jsonl(GLINER2 / "gliner2-tokenizer.jsonl"):
        for key in ("whitespace_splitter", "char_splitter"):
            if key not in row:
                continue
            rec = row[key]
            text = rec["normalized_text"]
            assert len(rec["text_word_first_positions"]) == len(rec["words"])
            for word in rec["words"]:
                start, end = word["offsets"]["unicode-scalars"]
                assert text[start:end].lower() == word["word"]
