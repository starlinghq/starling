"""Strict recipe loading and read-only model snapshot capability inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

RECIPE_SCHEMA = Path(__file__).resolve().parents[2] / "schemas" / "onboarding-recipe.schema.json"
MAX_CONFIG_BYTES = 4 * 1024 * 1024
MAX_TOKENIZER_BYTES = 64 * 1024 * 1024
MAX_RECIPE_BYTES = 1024 * 1024
_QWEN2_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
_QWEN35_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
_QWEN_SPLIT_PATTERNS = {_QWEN2_SPLIT, _QWEN35_SPLIT}


class RecipeError(ValueError):
    """Invalid recipe or unreadable/unsafe model snapshot."""


class _UniqueSafeLoader(yaml.SafeLoader):
    def __init__(self, stream: Any):
        super().__init__(stream)
        self._recipe_depth = 0

    def compose_node(self, parent: Any, index: Any) -> yaml.Node:
        if self.check_event(yaml.AliasEvent):
            raise RecipeError("YAML aliases are not permitted in recipes")
        self._recipe_depth += 1
        try:
            if self._recipe_depth > 32:
                raise RecipeError("recipe nesting exceeds the 32-level limit")
            return super().compose_node(parent, index)
        finally:
            self._recipe_depth -= 1


def _construct_mapping(loader: _UniqueSafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise RecipeError(f"duplicate key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _repository_file(value: str, root: Path, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or not value or "\\" in value:
        raise RecipeError(f"{label} must be a repository-relative path")
    try:
        resolved = (root / candidate).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RecipeError(f"{label} cannot be resolved: {value}") from exc
    if not _within(resolved, root):
        raise RecipeError(f"{label} escapes repository root")
    if not resolved.is_file():
        raise RecipeError(f"{label} is not a file: {value}")
    return resolved


def load_recipe(path: Path, repo_root: Path) -> dict[str, Any]:
    """Load a strict YAML/JSON recipe and resolve its declared source files safely."""
    try:
        root = repo_root.resolve(strict=True)
        recipe_path = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RecipeError(f"recipe or repository path cannot be resolved: {exc}") from exc
    if not root.is_dir():
        raise RecipeError("repo_root must be a directory")
    if not _within(recipe_path, root):
        raise RecipeError("recipe path must be inside repo_root")
    if not recipe_path.is_file():
        raise RecipeError("recipe path must be a file")
    try:
        if recipe_path.stat().st_size > MAX_RECIPE_BYTES:
            raise RecipeError("recipe exceeds the 1 MiB size limit")
        recipe = yaml.load(recipe_path.read_text(encoding="utf-8"), Loader=_UniqueSafeLoader)
    except (OSError, UnicodeError, yaml.YAMLError, TypeError, RecursionError) as exc:
        raise RecipeError(f"cannot read recipe: {exc}") from exc
    except RecipeError:
        raise
    if not isinstance(recipe, dict):
        raise RecipeError("recipe must be a mapping")

    try:
        schema = json.loads(RECIPE_SCHEMA.read_text(encoding="utf-8"))
        errors = sorted(Draft202012Validator(schema).iter_errors(recipe), key=lambda e: list(e.absolute_path))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecipeError(f"cannot load recipe schema: {exc}") from exc
    if errors:
        first = errors[0]
        location = "/".join(str(part) for part in first.absolute_path) or "<root>"
        raise RecipeError(f"invalid recipe at {location}: {first.message}")

    from model_pipeline.onboarding_plugins import PLUGINS

    if recipe["plugin"] not in PLUGINS:
        raise RecipeError(f"unsupported plugin: {recipe['plugin']}")
    resolved = dict(recipe)
    for key in ("manifest", "reference_corpus", "reference_environment"):
        _repository_file(recipe[key], root, key)
    # Keep recipe paths repository-relative in the public value. This makes recipe IDs portable.
    return resolved


def _read_json_file(root: Path, relative: str, limit: int) -> tuple[dict[str, Any] | list[Any] | None, str]:
    path = root / relative
    try:
        resolved = path.resolve(strict=True)
        safe_in_snapshot = _within(resolved, root)
        if not safe_in_snapshot and path.is_symlink() and root.parent.name == "snapshots":
            model_cache = root.parent.parent
            blobs = model_cache / "blobs"
            try:
                safe_in_snapshot = model_cache.name.startswith("models--") and resolved.is_relative_to(
                    blobs.resolve(strict=True)
                )
            except (OSError, RuntimeError):
                safe_in_snapshot = False
        if not safe_in_snapshot or not resolved.is_file():
            return None, "unsafe_path"
        if resolved.stat().st_size > limit:
            return None, "too_large"
        value = json.loads(resolved.read_text(encoding="utf-8"))
        if not isinstance(value, (dict, list)):
            return None, "invalid_shape"
        return value, "observed"
    except FileNotFoundError:
        return None, "missing"
    except RuntimeError:
        return None, "unsafe_path"
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        return None, "invalid"


def _component(path: str, value: Any, kind: str) -> dict[str, Any]:
    """Classify a tokenizer component from the parser's concrete type and parameter branches."""
    if not isinstance(value, dict):
        return {
            "path": path,
            "type": "unknown",
            "status": "unknown",
            "supported_parameters": [],
            "unsupported_parameters": [],
            "unknown_parameters": ["<component-not-object>"],
            "missing_parameters": [],
            "ignored_parameters": [],
        }

    type_name = value.get("type") if isinstance(value.get("type"), str) else "unknown"
    fields: dict[str, set[str]] = {
        "normalizer": {
            "Sequence": {"type", "normalizers"},
            "Precompiled": {"type", "precompiled_charsmap"},
            "Replace": {"type", "pattern", "content"},
            "Strip": {"type", "strip_left", "strip_right"},
            "NFC": {"type"},
            "BertNormalizer": {"type", "clean_text", "handle_chinese_chars", "strip_accents", "lowercase"},
        },
        "pre_tokenizer": {
            "Sequence": {"type", "pretokenizers"},
            "Metaspace": {"type", "replacement", "add_prefix_space", "prepend_scheme", "split"},
            "Split": {"type", "pattern", "behavior", "invert"},
            "ByteLevel": {"type", "add_prefix_space", "trim_offsets", "use_regex"},
            "BertPreTokenizer": {"type"},
        },
        "model": {
            "Unigram": {"type", "unk_id", "vocab", "byte_fallback"},
            "WordPiece": {"type", "unk_token", "continuing_subword_prefix", "max_input_chars_per_word", "vocab"},
            "BPE": {
                "type",
                "dropout",
                "unk_token",
                "continuing_subword_prefix",
                "end_of_word_suffix",
                "fuse_unk",
                "byte_fallback",
                "ignore_merges",
                "vocab",
                "merges",
            },
        },
        "post_processor": {
            "TemplateProcessing": {"type", "single", "pair", "special_tokens"},
            "ByteLevel": {"type", "add_prefix_space", "trim_offsets", "use_regex"},
        },
        "decoder": {
            "Metaspace": {"type", "replacement", "add_prefix_space", "prepend_scheme", "split"},
            "ByteLevel": {"type", "add_prefix_space", "trim_offsets", "use_regex"},
            "Sequence": {"type", "decoders"},
            "Replace": {"type", "pattern", "content"},
            "ByteFallback": {"type"},
            "Fuse": {"type"},
            "WordPiece": {"type", "prefix", "cleanup"},
        },
    }
    known = type_name in fields[kind]
    unknown = sorted(set(value) - fields[kind].get(type_name, {"type"}))
    unsupported: list[str] = []
    supported: list[str] = []
    ignored: list[str] = []
    if not known:
        status = "unsupported" if type_name != "unknown" else "unknown"
    else:
        for key in sorted(set(value) - {"type"} - set(unknown)):
            supported.append(key)
        status = "supported"

    if known:
        required: dict[tuple[str, str], set[str]] = {
            ("normalizer", "Sequence"): {"normalizers"},
            ("normalizer", "Precompiled"): {"precompiled_charsmap"},
            ("normalizer", "Replace"): {"pattern", "content"},
            ("decoder", "Replace"): {"pattern", "content"},
            ("normalizer", "Strip"): {"strip_left", "strip_right"},
            ("pre_tokenizer", "Split"): {"pattern", "behavior", "invert"},
            ("pre_tokenizer", "Sequence"): {"pretokenizers"},
            ("pre_tokenizer", "Metaspace"): {"replacement"},
            ("pre_tokenizer", "ByteLevel"): {"add_prefix_space", "trim_offsets"},
            ("model", "Unigram"): {"vocab"},
            ("model", "WordPiece"): {"vocab"},
            ("model", "BPE"): {"vocab", "merges"},
            ("post_processor", "TemplateProcessing"): {"single", "pair", "special_tokens"},
            ("post_processor", "ByteLevel"): {"add_prefix_space", "trim_offsets"},
            ("decoder", "Metaspace"): {"replacement"},
            ("decoder", "ByteLevel"): {"add_prefix_space", "trim_offsets"},
            ("decoder", "Sequence"): {"decoders"},
        }
        missing = sorted(required.get((kind, type_name), set()) - set(value))
        unsupported.extend(f"{parameter} (required parameter missing)" for parameter in missing)
        if kind == "model" and type_name != "BPE" and value.get("byte_fallback") is True:
            unsupported.append("byte_fallback=true")
        if (
            kind == "model"
            and "byte_fallback" in value
            and not isinstance(value["byte_fallback"], bool)
            and not (type_name == "BPE" and value["byte_fallback"] is None)
        ):
            unsupported.append("byte_fallback (must be a boolean)")
        if kind in {"pre_tokenizer", "decoder"} and type_name == "Metaspace":
            scheme = value.get("prepend_scheme", "always")
            if not isinstance(scheme, str) or scheme not in {"always", "first", "never"}:
                unsupported.append(f"prepend_scheme={scheme!r}")
            if value.get("add_prefix_space") is not None and not isinstance(value.get("add_prefix_space"), bool):
                unsupported.append("add_prefix_space (must be a boolean or null)")
            if value.get("add_prefix_space") is False and scheme != "never":
                unsupported.append("add_prefix_space=false (inconsistent prepend_scheme)")
            replacement = value.get("replacement")
            if not isinstance(replacement, str) or len(replacement) != 1 or 0xD800 <= ord(replacement[0]) <= 0xDFFF:
                unsupported.append("replacement (must be one Unicode scalar)")
            if "split" in value and not isinstance(value["split"], bool):
                unsupported.append("split (must be a boolean)")
        if kind in {"normalizer", "decoder"} and type_name == "Replace":
            pattern = value.get("pattern")
            accepted = isinstance(pattern, dict) and (
                (set(pattern) == {"String"} and isinstance(pattern["String"], str) and bool(pattern["String"]))
                or (set(pattern) == {"Regex"} and pattern["Regex"] == " {2,}")
            )
            if not accepted:
                unsupported.append("pattern (only nonempty String or Regex ' {2,}' is implemented)")
            if not isinstance(value.get("content"), str):
                unsupported.append("content (must be a string)")
        if kind == "model" and type_name == "Unigram" and not isinstance(value.get("vocab"), list):
            unsupported.append("vocab (required Unigram vocabulary is missing or invalid)")
        if kind == "model" and type_name == "Unigram":
            if value.get("unk_id") is not None and type(value.get("unk_id")) is not int:
                unsupported.append("unk_id (must be an integer or null)")
            if "byte_fallback" in value and type(value["byte_fallback"]) is not bool:
                unsupported.append("byte_fallback (must be a boolean)")
            vocab = value.get("vocab")
            if isinstance(vocab, list) and any(
                not isinstance(entry, list)
                or len(entry) != 2
                or not isinstance(entry[0], str)
                or type(entry[1]) not in {int, float}
                for entry in vocab
            ):
                unsupported.append("vocab (entries must be [string, number])")
        if kind == "pre_tokenizer" and type_name == "Split":
            pattern = value.get("pattern")
            string_pattern = isinstance(pattern, dict) and set(pattern) == {"String"}
            regex_pattern = isinstance(pattern, dict) and set(pattern) == {"Regex"}
            if string_pattern:
                if not isinstance(pattern["String"], str) or not pattern["String"]:
                    unsupported.append("pattern.String (must be a nonempty string)")
                if value.get("behavior") not in {
                    "Removed",
                    "Isolated",
                    "MergedWithPrevious",
                    "MergedWithNext",
                    "Contiguous",
                }:
                    unsupported.append(f"behavior={value.get('behavior')!r} (unknown split behavior)")
                if type(value.get("invert")) is not bool:
                    unsupported.append("invert (must be a boolean)")
            elif regex_pattern:
                if not isinstance(pattern["Regex"], str) or pattern["Regex"] not in _QWEN_SPLIT_PATTERNS:
                    unsupported.append("pattern (only the exact Qwen2 and Qwen3.5 regexes are implemented)")
                if value.get("behavior") != "Isolated":
                    unsupported.append(f"behavior={value.get('behavior')!r} (only Isolated is implemented for Regex)")
                if value.get("invert") is not False:
                    unsupported.append("invert (only false is implemented for Regex)")
            else:
                unsupported.append("pattern (must be a supported String or Regex pattern)")
        if kind in {"pre_tokenizer", "post_processor", "decoder"} and type_name == "ByteLevel":
            if type(value.get("add_prefix_space")) is not bool or type(value.get("trim_offsets")) is not bool:
                unsupported.append("add_prefix_space/trim_offsets (required booleans)")
            if "use_regex" in value and type(value["use_regex"]) is not bool:
                unsupported.append("use_regex (must be a boolean)")
            elif kind == "pre_tokenizer" and value.get("use_regex", True) is not False:
                unsupported.append("use_regex=true (the GPT-2 split is not implemented)")
            elif kind == "post_processor" and "use_regex" in value:
                ignored.append("use_regex (accepted but not used by the post-processor)")
            elif kind == "decoder":
                ignored.extend(["add_prefix_space", "trim_offsets", "use_regex"])
        if kind == "model" and type_name == "BPE":
            if not isinstance(value.get("vocab"), dict):
                unsupported.append("vocab (required object is missing or invalid)")
            else:
                vocab = value["vocab"]
                if any(
                    type(token_id) is not int or token_id < 0 or token_id > 2**31 - 1 for token_id in vocab.values()
                ):
                    unsupported.append("vocab (ids must be nonnegative signed 32-bit integers)")
            if not isinstance(value.get("merges"), list):
                unsupported.append("merges (required array is missing or invalid)")
            else:
                merges = value["merges"]
                pair_form = not merges or isinstance(merges[0], list)
                if pair_form:
                    valid_merges = all(
                        isinstance(merge, list) and len(merge) == 2 and all(isinstance(piece, str) for piece in merge)
                        for merge in merges
                    )
                else:
                    valid_merges = all(
                        isinstance(merge, str) and (merge.startswith("#version") or len(merge.split(" ")) == 2)
                        for merge in merges
                    )
                if not valid_merges:
                    unsupported.append("merges (must be string pairs or legacy two-token lines)")
                elif isinstance(value.get("vocab"), dict):
                    vocab_tokens = set(value["vocab"])
                    for merge in merges:
                        if isinstance(merge, str):
                            if merge.startswith("#version"):
                                continue
                            left, right = merge.split(" ")
                        else:
                            left, right = merge
                        if left not in vocab_tokens or right not in vocab_tokens or left + right not in vocab_tokens:
                            unsupported.append("merges (a merge token is absent from the vocabulary)")
                            break
            if value.get("dropout") is not None:
                unsupported.append("dropout (only null is implemented)")
            unk_token = value.get("unk_token")
            if unk_token is not None:
                if not isinstance(unk_token, str):
                    unsupported.append("unk_token (must be a string or null)")
                elif not isinstance(value.get("vocab"), dict) or unk_token not in value["vocab"]:
                    unsupported.append("unk_token (must be present in the vocabulary)")
            for affix in ("continuing_subword_prefix", "end_of_word_suffix"):
                if value.get(affix) not in (None, ""):
                    unsupported.append(f"{affix} (only null or empty is implemented)")
            for flag in ("byte_fallback", "fuse_unk", "ignore_merges"):
                if flag in value and value[flag] is not None and type(value[flag]) is not bool:
                    unsupported.append(f"{flag} (must be a boolean)")
        if kind == "decoder" and type_name == "Sequence" and not isinstance(value.get("decoders"), list):
            unsupported.append("decoders (must be an array)")
        if kind == "normalizer" and type_name == "Precompiled":
            import base64

            encoded = value.get("precompiled_charsmap")
            try:
                if not isinstance(encoded, str):
                    raise ValueError
                charsmap = base64.b64decode(encoded, validate=True)
            except ValueError:
                charsmap = b""
                unsupported.append("precompiled_charsmap (invalid base64)")
            if charsmap:
                if len(charsmap) < 4:
                    unsupported.append("precompiled_charsmap (too short)")
                else:
                    trie_size = int.from_bytes(charsmap[:4], "little")
                    if trie_size == 0 or trie_size % 4 != 0 or trie_size > len(charsmap) - 4:
                        unsupported.append("precompiled_charsmap (invalid trie size)")
                    else:
                        try:
                            charsmap[4 + trie_size :].decode("utf-8", errors="strict")
                        except UnicodeDecodeError:
                            unsupported.append("precompiled_charsmap (invalid UTF-8 replacements)")
        if (
            kind == "normalizer"
            and type_name == "Strip"
            and any(not isinstance(value.get(key), bool) for key in ("strip_left", "strip_right"))
        ):
            unsupported.append("strip_left/strip_right (must be booleans)")
        if kind == "normalizer" and type_name == "BertNormalizer":
            for parameter in ("clean_text", "handle_chinese_chars", "lowercase"):
                if parameter in value and type(value[parameter]) is not bool:
                    unsupported.append(f"{parameter} (must be a boolean)")
            if (
                "strip_accents" in value
                and value["strip_accents"] is not None
                and type(value["strip_accents"]) is not bool
            ):
                unsupported.append("strip_accents (must be a boolean or null)")
            elif (
                value.get("clean_text", True) is not True
                or value.get("handle_chinese_chars", True) is not True
                or value.get("strip_accents") is not None
                or value.get("lowercase", True) is not True
            ):
                unsupported.append(
                    "only clean_text=true, handle_chinese_chars=true, strip_accents=null, lowercase=true are supported"
                )
        if kind == "model" and type_name == "WordPiece":
            vocab = value.get("vocab")
            if not isinstance(vocab, dict):
                unsupported.append("vocab (required object is missing or invalid)")
            else:
                ids = list(vocab.values())
                if any(type(token_id) is not int or token_id < 0 or token_id > 2**31 - 1 for token_id in ids):
                    unsupported.append("vocab (ids must be nonnegative signed 32-bit integers)")
                elif len(set(ids)) != len(ids):
                    unsupported.append("vocab (ids must be unique)")
            unknown_token = value.get("unk_token", "[UNK]")
            if not isinstance(unknown_token, str):
                unsupported.append("unk_token (must be a string)")
            elif isinstance(vocab, dict) and unknown_token not in vocab:
                unsupported.append("unk_token (must be present in the vocabulary)")
            prefix = value.get("continuing_subword_prefix", "##")
            if not isinstance(prefix, str):
                unsupported.append("continuing_subword_prefix (must be a string)")
            max_chars = value.get("max_input_chars_per_word", 100)
            if type(max_chars) is not int or max_chars <= 0 or max_chars > 2**31 - 1:
                unsupported.append("max_input_chars_per_word (must be a positive 32-bit integer)")
        if kind == "decoder" and type_name == "WordPiece":
            if "prefix" in value and not isinstance(value["prefix"], str):
                unsupported.append("prefix (must be a string)")
            if "cleanup" in value and type(value["cleanup"]) is not bool:
                unsupported.append("cleanup (must be a boolean)")
        if kind == "normalizer" and type_name == "Sequence" and not isinstance(value.get("normalizers"), list):
            unsupported.append("normalizers (must be an array)")
        if kind == "pre_tokenizer" and type_name == "Sequence" and not isinstance(value.get("pretokenizers"), list):
            unsupported.append("pretokenizers (must be an array)")
        if (
            kind == "post_processor"
            and type_name == "TemplateProcessing"
            and not all(
                isinstance(value.get(key), list if key in {"single", "pair"} else dict)
                for key in ("single", "pair", "special_tokens")
            )
        ):
            unsupported.append("single/pair/special_tokens (invalid structure)")
        if (
            kind == "post_processor"
            and type_name == "TemplateProcessing"
            and all(
                isinstance(value.get(key), list if key in {"single", "pair"} else dict)
                for key in ("single", "pair", "special_tokens")
            )
        ):
            specials = value["special_tokens"]
            special_ids: set[str] = set()
            for name, special in specials.items():
                if not isinstance(special, dict) or set(special) - {"id", "ids", "tokens"}:
                    unsupported.append(f"special_tokens.{name} (invalid fields)")
                    continue
                ids, tokens = special.get("ids"), special.get("tokens")
                if (
                    not isinstance(special.get("id"), str)
                    or not isinstance(ids, list)
                    or not isinstance(tokens, list)
                    or len(ids) != len(tokens)
                    or any(not isinstance(token_id, int) for token_id in ids)
                    or any(not isinstance(token, str) for token in tokens)
                ):
                    unsupported.append(f"special_tokens.{name} (invalid id/ids/tokens)")
                else:
                    special_ids.add(special["id"])
            for template_name, template in (("single", value["single"]), ("pair", value["pair"])):
                sequences: list[str] = []
                for index, part in enumerate(template):
                    if not isinstance(part, dict) or len(part) != 1:
                        unsupported.append(f"{template_name}[{index}] (invalid template piece)")
                        continue
                    kind_name, payload = next(iter(part.items()))
                    if not isinstance(payload, dict) or set(payload) - {"id", "type_id"}:
                        unsupported.append(f"{template_name}[{index}] (invalid template parameters)")
                        continue
                    if type(payload.get("type_id")) is not int:
                        unsupported.append(f"{template_name}[{index}].type_id (must be integer)")
                    if (
                        kind_name == "Sequence"
                        and isinstance(payload.get("id"), str)
                        and payload.get("id") in {"A", "B"}
                    ):
                        sequences.append(payload["id"])
                    elif (
                        kind_name == "SpecialToken"
                        and isinstance(payload.get("id"), str)
                        and payload.get("id") in special_ids
                    ):
                        pass
                    else:
                        unsupported.append(f"{template_name}[{index}] (unknown sequence or special token)")
                expected = ["A"] if template_name == "single" else ["A", "B"]
                if sorted(sequences) != expected:
                    unsupported.append(f"{template_name} (must contain required sequence exactly once)")
        unsupported = list(dict.fromkeys(unsupported))
        if unknown or unsupported:
            status = "unsupported"
    return {
        "path": path,
        "type": type_name,
        "status": status,
        "supported_parameters": supported,
        "unsupported_parameters": unsupported,
        "unknown_parameters": unknown,
        "missing_parameters": [
            part.partition(" ")[0] for part in unsupported if part.endswith("(required parameter missing)")
        ],
        "ignored_parameters": ignored,
    }


def _tokenizer_components(tokenizer: dict[str, Any]) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []

    def walk(value: Any, path: str, kind: str) -> None:
        if value is None:
            return
        item = _component(path, value, kind)
        components.append(item)
        if isinstance(value, dict) and item["status"] in {"supported", "unsupported"}:
            child_key = {
                "normalizer": "normalizers",
                "pre_tokenizer": "pretokenizers",
                "decoder": "decoders",
            }.get(kind)
            if value.get("type") == "Sequence" and child_key:
                children = value.get(child_key)
                if isinstance(children, list):
                    start = len(components)
                    for index, child in enumerate(children):
                        walk(child, f"{path}.{child_key}[{index}]", kind)
                    if any(child["status"] != "supported" for child in components[start:]):
                        item["status"] = "unsupported"

    for key, kind in (
        ("normalizer", "normalizer"),
        ("pre_tokenizer", "pre_tokenizer"),
        ("model", "model"),
        ("post_processor", "post_processor"),
        ("decoder", "decoder"),
    ):
        if key in tokenizer:
            walk(tokenizer[key], key, kind)
        elif key == "model":
            components.append(
                {
                    "path": key,
                    "type": "unknown",
                    "status": "unknown",
                    "supported_parameters": [],
                    "unsupported_parameters": [],
                    "unknown_parameters": ["<missing-required-component>"],
                    "missing_parameters": ["model"],
                    "ignored_parameters": [],
                }
            )
    version = tokenizer.get("version")
    components.append(
        {
            "path": "version",
            "type": str(version) if version is not None else "unknown",
            "status": "supported" if version == "1.0" else ("unsupported" if version is not None else "unknown"),
            "supported_parameters": [],
            "unsupported_parameters": [],
            "unknown_parameters": [],
            "missing_parameters": [],
            "ignored_parameters": [],
        }
    )
    for key in ("added_tokens", "truncation", "padding"):
        value = tokenizer.get(key)
        if value is None:
            continue
        if key == "added_tokens":
            items = value if isinstance(value, list) else []
            unsupported = []
            unknown = []
            required = {"id", "content", "single_word", "lstrip", "rstrip", "normalized", "special"}
            for i, token in enumerate(items):
                if not isinstance(token, dict):
                    unknown.append(f"{i}:not-object")
                    continue
                for missing in sorted(required - set(token)):
                    unsupported.append(f"{i}.{missing}=missing")
                for param in ("single_word", "lstrip", "rstrip"):
                    if token.get(param) is True:
                        unsupported.append(f"{i}.{param}=true")
                    elif param in token and not isinstance(token[param], bool):
                        unsupported.append(f"{i}.{param}=invalid")
                if any(not isinstance(token.get(param), bool) for param in ("normalized", "special")):
                    unsupported.append(f"{i}.normalized/special=invalid")
                if type(token.get("id")) is not int or not isinstance(token.get("content"), str):
                    unsupported.append(f"{i}.id/content=invalid")
                extra = set(token) - {"id", "content", "single_word", "lstrip", "rstrip", "normalized", "special"}
                unknown.extend(f"{i}.{item}" for item in sorted(extra))
            status = "unsupported" if unsupported or unknown or not isinstance(value, list) else "supported"
            supported = ["id", "content", "normalized", "special"]
        elif key == "truncation":
            obj = value if isinstance(value, dict) else {}
            unknown = sorted(set(obj) - {"direction", "max_length", "strategy", "stride"})
            unsupported = []
            for required in ("max_length", "strategy", "stride"):
                if required not in obj:
                    unsupported.append(f"{required}=missing")
            if obj.get("stride") not in (0, None):
                unsupported.append("stride!=0")
            if obj.get("strategy") is not None and (
                not isinstance(obj.get("strategy"), str)
                or obj.get("strategy") not in {"LongestFirst", "OnlyFirst", "OnlySecond"}
            ):
                unsupported.append(f"strategy={obj.get('strategy')!r}")
            if "strategy" in obj and obj["strategy"] is None:
                unsupported.append("strategy=null")
            if not isinstance(obj.get("direction", "Right"), str) or obj.get("direction", "Right") not in {
                "Right",
                "Left",
            }:
                unsupported.append(f"direction={obj.get('direction')!r}")
            if type(obj.get("max_length")) is not int or type(obj.get("stride")) is not int:
                unsupported.append("max_length/stride (must be integers)")
            status = "unsupported" if unknown or unsupported else "supported"
            supported = sorted(set(obj) - set(unknown) - set(unsupported))
        else:
            obj = value if isinstance(value, dict) else {}
            unknown = sorted(
                set(obj) - {"strategy", "direction", "pad_to_multiple_of", "pad_id", "pad_type_id", "pad_token"}
            )
            strategy = obj.get("strategy")
            valid_strategy = strategy == "BatchLongest" or (isinstance(strategy, dict) and set(strategy) == {"Fixed"})
            required = {"strategy", "direction", "pad_id", "pad_type_id", "pad_token"}
            unsupported = [f"{field}=missing" for field in sorted(required - set(obj))]
            if (
                not valid_strategy
                or not isinstance(obj.get("direction"), str)
                or obj.get("direction") not in {"Left", "Right"}
            ):
                unsupported.append("strategy/direction")
            if (
                type(obj.get("pad_id")) is not int
                or type(obj.get("pad_type_id")) is not int
                or not isinstance(obj.get("pad_token"), str)
            ):
                unsupported.append("pad_id/pad_type_id/pad_token (invalid type)")
            status = "unsupported" if unknown or unsupported else "supported"
            supported = sorted(set(obj) - set(unknown))
        components.append(
            {
                "path": key,
                "type": key,
                "status": status,
                "supported_parameters": supported,
                "unsupported_parameters": unsupported,
                "unknown_parameters": unknown,
                "missing_parameters": [],
                "ignored_parameters": [],
            }
        )
    extra = sorted(
        set(tokenizer)
        - {
            "version",
            "truncation",
            "padding",
            "added_tokens",
            "normalizer",
            "pre_tokenizer",
            "post_processor",
            "decoder",
            "model",
        }
    )
    if extra:
        components.append(
            {
                "path": "<root>",
                "type": "tokenizer.json",
                "status": "unsupported",
                "supported_parameters": [],
                "unsupported_parameters": [],
                "unknown_parameters": extra,
                "missing_parameters": [],
                "ignored_parameters": [],
            }
        )
    return components


def inspect_snapshot(snapshot: Path) -> dict[str, Any]:
    """Inspect local JSON metadata only. Never imports, executes, or downloads snapshot code."""
    try:
        root = snapshot.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RecipeError(f"snapshot does not exist: {snapshot}") from exc
    if not root.is_dir():
        raise RecipeError("snapshot must be a directory")

    config, config_status = _read_json_file(root, "config.json", MAX_CONFIG_BYTES)
    if config_status == "observed" and not isinstance(config, dict):
        config_status = "invalid_shape"
    config_obj = config if isinstance(config, dict) else {}
    model_facts = {
        "status": config_status,
        "path": "config.json",
        "model_type": config_obj.get("model_type"),
        "architectures": config_obj.get("architectures", "unknown"),
        "hidden_size": config_obj.get("hidden_size", "unknown"),
        "max_position_embeddings": config_obj.get("max_position_embeddings", "unknown"),
        "auto_map": config_obj.get("auto_map", "unknown"),
    }
    tokenizer, tokenizer_status = _read_json_file(root, "tokenizer.json", MAX_TOKENIZER_BYTES)
    if tokenizer_status == "observed" and not isinstance(tokenizer, dict):
        tokenizer_status = "invalid_shape"
    tokenizer_obj = tokenizer if isinstance(tokenizer, dict) else {}
    tokenizer_report = {
        "status": tokenizer_status,
        "path": "tokenizer.json",
        "format_version": tokenizer_obj.get("version", "unknown"),
        "model_type": (tokenizer_obj.get("model") or {}).get("type", "unknown")
        if isinstance(tokenizer_obj.get("model"), dict)
        else "unknown",
        "components": _tokenizer_components(tokenizer_obj) if tokenizer_status == "observed" else [],
    }
    tokenizer_config, tokenizer_config_status = _read_json_file(root, "tokenizer_config.json", MAX_CONFIG_BYTES)
    if tokenizer_config_status == "observed" and not isinstance(tokenizer_config, dict):
        tokenizer_config_status = "invalid_shape"
    special_tokens_map, special_tokens_status = _read_json_file(root, "special_tokens_map.json", MAX_CONFIG_BYTES)
    if special_tokens_status == "observed" and not isinstance(special_tokens_map, dict):
        special_tokens_status = "invalid_shape"
    tokenizer_report["config"] = {
        "status": tokenizer_config_status,
        "facts": tokenizer_config if isinstance(tokenizer_config, dict) else {},
    }
    tokenizer_report["special_tokens_map"] = {
        "status": special_tokens_status,
        "facts": special_tokens_map if isinstance(special_tokens_map, dict) else {},
    }

    modules_raw, modules_status = _read_json_file(root, "modules.json", MAX_CONFIG_BYTES)
    if modules_status == "observed" and not isinstance(modules_raw, list):
        modules_status = "invalid_shape"
    modules: list[dict[str, Any]] = []
    pooling: dict[str, Any] = {"status": "unknown", "facts": {}}
    if modules_status == "observed" and isinstance(modules_raw, list):
        for item in modules_raw:
            if not isinstance(item, dict):
                modules.append({"status": "unknown", "observed": item})
                continue
            module_path = item.get("path", "")
            module_type = item.get("type", "unknown")
            if not isinstance(module_path, str) or not isinstance(module_type, str):
                modules.append({"status": "unknown", "observed": item})
                continue
            config_path = str(Path(module_path) / "config.json") if module_path else "config.json"
            if module_type.endswith("Pooling"):
                raw, status = _read_json_file(root, config_path, MAX_CONFIG_BYTES)
                if status == "observed" and not isinstance(raw, dict):
                    status = "invalid_shape"
                facts = raw if isinstance(raw, dict) else {}
                pooling = {"status": status, "path": config_path, "facts": facts}
                modules.append(
                    {
                        "status": status,
                        "path": module_path,
                        "type": module_type,
                        "config_path": config_path,
                        "facts": facts,
                    }
                )
            else:
                modules.append(
                    {"status": "observed", "path": module_path, "type": module_type, "config_path": config_path}
                )
    elif modules_status != "observed":
        modules.append({"status": modules_status})

    module_types = {str(m.get("type", "")).rsplit(".", maxsplit=1)[-1] for m in modules}
    pool_facts = pooling.get("facts", {})
    tokenizer_pipeline_known = bool(tokenizer_report["components"]) and all(
        component["status"] == "supported" for component in tokenizer_report["components"]
    )
    route_match = (
        config_status == "observed"
        and modules_status == "observed"
        and pooling["status"] == "observed"
        and model_facts["model_type"] == "bert"
        and model_facts["architectures"] == ["BertModel"]
        and model_facts["hidden_size"] == 384
        and model_facts["max_position_embeddings"] == 512
        and tokenizer_report["model_type"] == "Unigram"
        and tokenizer_pipeline_known
        and {"Transformer", "Pooling", "Normalize"}.issubset(module_types)
        and pool_facts.get("pooling_mode_mean_tokens") is True
        and all(
            pool_facts.get(key) is False
            for key in ("pooling_mode_cls_token", "pooling_mode_max_tokens", "pooling_mode_mean_sqrt_len_tokens")
        )
    )
    result = {
        "snapshot": str(root),
        "model_config": model_facts,
        "tokenizer": tokenizer_report,
        "sentence_transformers": {
            "modules_file": {"path": "modules.json", "status": modules_status},
            "modules": modules,
            "pooling": pooling,
        },
        "plugin_route": {
            "id": "e5-onnx-cpu-v1" if route_match else "unknown",
            "status": "candidate" if route_match else "unknown",
            "reason": (
                "Observed the E5-sized BERT, Unigram tokenizer, mean-pooling, normalization module contract; "
                "plugin compatibility still requires its own validation."
                if route_match
                else "No plugin route can be inferred from these observed facts."
            ),
        },
    }
    result["missing"] = sorted(
        path
        for path, status in (
            ("config.json", config_status),
            ("tokenizer.json", tokenizer_status),
            ("modules.json", modules_status),
            ("tokenizer_config.json", tokenizer_config_status),
            ("special_tokens_map.json", special_tokens_status),
            (pooling.get("path", "1_Pooling/config.json"), pooling["status"]),
        )
        if status == "missing"
    )
    result["unknown"] = sorted(
        component["path"] for component in tokenizer_report["components"] if component["status"] == "unknown"
    )
    return result
