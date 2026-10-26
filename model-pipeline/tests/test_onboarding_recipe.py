from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_pipeline.onboarding_recipe import _QWEN2_SPLIT, RecipeError, inspect_snapshot, load_recipe


def _recipe(root: Path, **overrides: object) -> Path:
    data = {
        "recipe_version": 1,
        "id": "demo",
        "plugin": "e5-onnx-cpu-v1",
        "manifest": "manifest.yaml",
        "reference_corpus": "corpus.jsonl",
        "reference_environment": "requirements.txt",
    }
    data.update(overrides)
    path = root / "recipe.yaml"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "overrides",
    [
        {"recipe_version": 2},
        {"plugin": "arbitrary-python"},
        {"shell": "echo unsafe"},
        {"id": "../escape"},
        {"manifest": "../outside.yaml"},
        {"reference_corpus": "/etc/passwd"},
    ],
)
def test_rejects_invalid_recipe_or_path(tmp_path: Path, overrides: dict[str, object]) -> None:
    for name in ("manifest.yaml", "corpus.jsonl", "requirements.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    recipe = _recipe(tmp_path, **overrides)
    with pytest.raises(RecipeError):
        load_recipe(recipe, tmp_path)


def test_loads_known_recipe_without_rewriting_paths(tmp_path: Path) -> None:
    for name in ("manifest.yaml", "corpus.jsonl", "requirements.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    loaded = load_recipe(_recipe(tmp_path), tmp_path)
    assert loaded["manifest"] == "manifest.yaml"
    assert loaded["plugin"] == "e5-onnx-cpu-v1"


def test_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    for name in ("manifest.yaml", "corpus.jsonl", "requirements.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    path = tmp_path / "duplicate.yaml"
    path.write_text("recipe_version: 1\nrecipe_version: 1\n", encoding="utf-8")
    with pytest.raises(RecipeError, match="duplicate key"):
        load_recipe(path, tmp_path)


@pytest.mark.parametrize(
    "text",
    [
        "recipe_version: 1\nitems: &anchor [x]\ncopy: *anchor\n",
        "recipe_version: 1\n? [unhashable, key]\n: value\n",
        "recipe_version: 1\nvalue: " + "[" * 40 + "x" + "]" * 40 + "\n",
    ],
)
def test_rejects_aliases_unhashable_keys_and_deep_yaml(tmp_path: Path, text: str) -> None:
    for name in ("manifest.yaml", "corpus.jsonl", "requirements.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    path = tmp_path / "malformed.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(RecipeError):
        load_recipe(path, tmp_path)


def test_inspect_snapshot_reports_supported_and_unknown_parameters(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    (snapshot / "1_Pooling").mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bert",
                "architectures": ["BertModel"],
                "hidden_size": 384,
                "max_position_embeddings": 512,
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "modules.json").write_text(
        json.dumps(
            [
                {"path": "", "type": "sentence_transformers.models.Transformer"},
                {"path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
                {"path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
            ]
        ),
        encoding="utf-8",
    )
    (snapshot / "1_Pooling/config.json").write_text(
        json.dumps(
            {
                "pooling_mode_mean_tokens": True,
                "pooling_mode_cls_token": False,
                "pooling_mode_max_tokens": False,
                "pooling_mode_mean_sqrt_len_tokens": False,
            }
        ),
        encoding="utf-8",
    )
    tokenizer = {
        "version": "1.0",
        "model": {"type": "Unigram", "unk_id": 3, "vocab": [], "byte_fallback": False},
        "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True},
    }
    (snapshot / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")

    report = inspect_snapshot(snapshot)
    assert report["model_config"]["model_type"] == "bert"
    assert report["sentence_transformers"]["pooling"]["facts"]["pooling_mode_mean_tokens"] is True
    assert report["plugin_route"]["status"] == "candidate"
    tokenizer_report = report["tokenizer"]
    model = next(item for item in tokenizer_report["components"] if item["path"] == "model")
    pretokenizer = next(item for item in tokenizer_report["components"] if item["path"] == "pre_tokenizer")
    assert model["status"] == "supported"
    assert pretokenizer["status"] == "supported"
    json.dumps(report)


def test_inspection_does_not_call_unknown_operations_supported(tmp_path: Path) -> None:
    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "2.0",
                "model": {"type": "WordPiece", "vocab": {}, "merges": []},
                "normalizer": {"type": "Lowercase", "ignored": True},
            }
        ),
        encoding="utf-8",
    )
    report = inspect_snapshot(tmp_path)
    components = {item["path"]: item for item in report["tokenizer"]["components"]}
    assert components["model"]["status"] == "unsupported"
    assert components["normalizer"]["status"] == "unsupported"
    assert components["version"]["status"] == "unsupported"
    assert report["plugin_route"]["status"] == "unknown"


def test_inspector_accepts_only_implemented_tokenizer_features(tmp_path: Path) -> None:
    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "normalizer": {"type": "NFC"},
                "pre_tokenizer": {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
                "model": {
                    "type": "BPE",
                    "unk_token": None,
                    "continuing_subword_prefix": None,
                    "end_of_word_suffix": "",
                    "fuse_unk": True,
                    "byte_fallback": False,
                    "ignore_merges": False,
                    "vocab": {"a": 0, "b": 1, "ab": 2},
                    "merges": [["a", "b"]],
                },
                "post_processor": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True},
                "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True},
            }
        ),
        encoding="utf-8",
    )
    components = {item["path"]: item for item in inspect_snapshot(tmp_path)["tokenizer"]["components"]}
    for path in ("normalizer", "pre_tokenizer", "model", "post_processor", "decoder"):
        assert components[path]["status"] == "supported", (path, components[path])
    assert components["post_processor"]["ignored_parameters"] == []
    assert "add_prefix_space" in components["decoder"]["ignored_parameters"]


def test_bpe_byte_fallback_and_vocabulary_unknown_token_are_supported(tmp_path: Path) -> None:
    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "model": {
                    "type": "BPE",
                    "unk_token": "<unk>",
                    "byte_fallback": True,
                    "fuse_unk": True,
                    "vocab": {"a": 0, "<unk>": 1},
                    "merges": [],
                },
            }
        ),
        encoding="utf-8",
    )
    model = next(item for item in inspect_snapshot(tmp_path)["tokenizer"]["components"] if item["path"] == "model")
    assert model["status"] == "supported"
    assert model["unsupported_parameters"] == []


def test_bpe_rejects_unknown_token_absent_from_vocabulary(tmp_path: Path) -> None:
    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "model": {
                    "type": "BPE",
                    "unk_token": "<missing>",
                    "byte_fallback": True,
                    "vocab": {"a": 0},
                    "merges": [],
                },
            }
        ),
        encoding="utf-8",
    )
    model = next(item for item in inspect_snapshot(tmp_path)["tokenizer"]["components"] if item["path"] == "model")
    assert model["status"] == "unsupported"
    assert "unk_token (must be present in the vocabulary)" in model["unsupported_parameters"]


def test_inspector_handles_only_exact_qwen_split_pattern(tmp_path: Path) -> None:
    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "pre_tokenizer": {
                    "type": "Split",
                    "pattern": {"Regex": _QWEN2_SPLIT},
                    "behavior": "Isolated",
                    "invert": False,
                },
            }
        ),
        encoding="utf-8",
    )
    split = next(
        item for item in inspect_snapshot(tmp_path)["tokenizer"]["components"] if item["path"] == "pre_tokenizer"
    )
    assert split["status"] == "supported"
    data = json.loads((tmp_path / "tokenizer.json").read_text(encoding="utf-8"))
    data["pre_tokenizer"]["behavior"] = "MergedWithPrevious"
    (tmp_path / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
    split = next(
        item for item in inspect_snapshot(tmp_path)["tokenizer"]["components"] if item["path"] == "pre_tokenizer"
    )
    assert split["status"] == "unsupported"


def test_inspector_supports_literal_split_with_all_implemented_behaviors(tmp_path: Path) -> None:
    data = {
        "version": "1.0",
        "pre_tokenizer": {
            "type": "Split",
            "pattern": {"String": " "},
            "behavior": "MergedWithPrevious",
            "invert": False,
        },
    }
    (tmp_path / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
    split = next(
        item for item in inspect_snapshot(tmp_path)["tokenizer"]["components"] if item["path"] == "pre_tokenizer"
    )
    assert split["status"] == "supported"

    data["pre_tokenizer"]["behavior"] = "Contiguous"
    data["pre_tokenizer"]["invert"] = True
    (tmp_path / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
    split = next(
        item for item in inspect_snapshot(tmp_path)["tokenizer"]["components"] if item["path"] == "pre_tokenizer"
    )
    assert split["status"] == "supported"


def test_inspector_supports_gemma_decoder_sequence_operations(tmp_path: Path) -> None:
    (tmp_path / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "decoder": {
                    "type": "Sequence",
                    "decoders": [
                        {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                        {"type": "ByteFallback"},
                        {"type": "Fuse"},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    components = {item["path"]: item for item in inspect_snapshot(tmp_path)["tokenizer"]["components"]}
    assert components["decoder"]["status"] == "supported"
    assert components["decoder.decoders[0]"]["status"] == "supported"
    assert components["decoder.decoders[1]"]["status"] == "supported"
    assert components["decoder.decoders[2]"]["status"] == "supported"


def test_inspector_reads_standard_hugging_face_blob_symlinks(tmp_path: Path) -> None:
    model_cache = tmp_path / "models--sample--model"
    snapshot = model_cache / "snapshots" / "abcdef"
    blobs = model_cache / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    config_blob = blobs / "config-hash"
    config_blob.write_text(json.dumps({"model_type": "sample"}), encoding="utf-8")
    (snapshot / "config.json").symlink_to("../../blobs/config-hash")

    report = inspect_snapshot(snapshot)
    assert report["model_config"]["status"] == "observed"
    assert report["model_config"]["model_type"] == "sample"


def test_inspector_rejects_arbitrary_snapshot_symlink_escape(tmp_path: Path) -> None:
    snapshot = tmp_path / "ordinary-snapshot"
    snapshot.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"model_type": "outside"}), encoding="utf-8")
    (snapshot / "config.json").symlink_to(outside)

    report = inspect_snapshot(snapshot)
    assert report["model_config"]["status"] == "unsafe_path"
    assert report["model_config"]["model_type"] is None


def test_missing_metadata_stays_unknown(tmp_path: Path) -> None:
    report = inspect_snapshot(tmp_path)
    assert report["model_config"]["status"] == "missing"
    assert report["model_config"]["hidden_size"] == "unknown"
    assert report["tokenizer"]["status"] == "missing"
    assert report["sentence_transformers"]["pooling"]["status"] == "unknown"


_WORDPIECE = {
    "version": "1.0",
    "normalizer": {
        "type": "BertNormalizer",
        "clean_text": True,
        "handle_chinese_chars": True,
        "strip_accents": None,
        "lowercase": True,
    },
    "pre_tokenizer": {"type": "BertPreTokenizer"},
    "model": {
        "type": "WordPiece",
        "unk_token": "[UNK]",
        "continuing_subword_prefix": "##",
        "max_input_chars_per_word": 100,
        "vocab": {"[UNK]": 0},
    },
}


def _write_snapshot(
    root: Path, config: dict, tokenizer: dict, modules: list | None = None, pooling: dict | None = None
):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
    if modules is not None:
        (root / "modules.json").write_text(json.dumps(modules), encoding="utf-8")
    if pooling is not None:
        (root / "1_Pooling").mkdir(exist_ok=True)
        (root / "1_Pooling/config.json").write_text(json.dumps(pooling), encoding="utf-8")
    return root


def _modules(normalize: bool) -> list:
    modules = [
        {"path": "", "type": "sentence_transformers.models.Transformer"},
        {"path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    ]
    return modules + ([{"path": "2_Normalize", "type": "sentence_transformers.models.Normalize"}] if normalize else [])


def _pooling(mode: str) -> dict:
    keys = ("cls_token", "mean_tokens", "max_tokens", "mean_sqrt_len_tokens")
    return {f"pooling_mode_{key}": key == mode for key in keys}


_BERT = {"model_type": "bert", "architectures": ["BertModel"], "hidden_size": 768, "max_position_embeddings": 512}


@pytest.mark.parametrize(
    ("mode", "normalize", "expected"), [("cls_token", True, "cls"), ("mean_tokens", False, "mean")]
)
def test_inspect_recommends_the_bert_route_for_sentence_transformers_encoders(
    tmp_path: Path, mode: str, normalize: bool, expected: str
) -> None:
    snapshot = _write_snapshot(tmp_path / "s", _BERT, _WORDPIECE, _modules(normalize), _pooling(mode))
    route = inspect_snapshot(snapshot)["plugin_route"]
    assert (route["id"], route["status"], route["task"]) == ("bert-onnx-cpu-v1", "candidate", "embeddings")
    assert route["facts"] == {
        "hidden_size": 768,
        "max_position_embeddings": 512,
        "pooling": expected,
        "normalize": "l2" if normalize else "none",
    }


def test_inspect_refuses_the_bert_route_for_other_pooling_modes(tmp_path: Path) -> None:
    snapshot = _write_snapshot(tmp_path / "s", _BERT, _WORDPIECE, _modules(True), _pooling("max_tokens"))
    route = inspect_snapshot(snapshot)["plugin_route"]
    assert (route["id"], route["status"]) == ("unknown", "unknown")
    assert "mean or CLS" in route["reason"]


def test_inspect_recommends_the_bert_route_for_single_logit_cross_encoders(tmp_path: Path) -> None:
    config = {**_BERT, "architectures": ["BertForSequenceClassification"], "id2label": {"0": "LABEL_0"}}
    route = inspect_snapshot(_write_snapshot(tmp_path / "s", config, _WORDPIECE))["plugin_route"]
    assert (route["id"], route["task"], route["facts"]["pooling"]) == ("bert-onnx-cpu-v1", "reranking", "none")
    two_labels = {**config, "id2label": {"0": "a", "1": "b"}}
    assert inspect_snapshot(_write_snapshot(tmp_path / "t", two_labels, _WORDPIECE))["plugin_route"]["id"] == "unknown"


def test_inspect_needs_a_supported_tokenizer_for_the_bert_route(tmp_path: Path) -> None:
    tokenizer = {**_WORDPIECE, "model": {**_WORDPIECE["model"], "merges": []}}
    snapshot = _write_snapshot(tmp_path / "s", _BERT, tokenizer, _modules(True), _pooling("mean_tokens"))
    assert inspect_snapshot(snapshot)["plugin_route"]["status"] == "unknown"


def _qwen_tokenizer() -> dict:
    byte_level = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False, "use_regex": False}
    return {
        "version": "1.0",
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": _QWEN2_SPLIT}, "behavior": "Isolated", "invert": False},
                byte_level,
            ],
        },
        "post_processor": byte_level,
        "decoder": byte_level,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": {},
            "merges": [],
        },
    }


@pytest.mark.parametrize(("model_type", "architecture"), [("qwen3", "Qwen3ForCausalLM"), ("qwen2", "Qwen2ForCausalLM")])
def test_inspect_recommends_the_qwen_route_for_qwen_decoders(
    tmp_path: Path, model_type: str, architecture: str
) -> None:
    config = {"model_type": model_type, "architectures": [architecture], "hidden_size": 1024}
    route = inspect_snapshot(_write_snapshot(tmp_path / "s", config, _qwen_tokenizer()))["plugin_route"]
    assert (route["id"], route["status"], route["task"]) == ("qwen-genai-cpu-v1", "candidate", "generation")


def test_inspect_refuses_the_qwen_route_for_another_head_or_tokenizer(tmp_path: Path) -> None:
    head = {"model_type": "qwen3", "architectures": ["Qwen3ForSequenceClassification"]}
    assert inspect_snapshot(_write_snapshot(tmp_path / "s", head, _qwen_tokenizer()))["plugin_route"]["id"] == "unknown"
    config = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}
    route = inspect_snapshot(_write_snapshot(tmp_path / "t", config, _WORDPIECE))["plugin_route"]
    assert (route["id"], route["status"]) == ("unknown", "unknown")
