"""Python side of the intervention file contract (profile schema versions 1 and 2, manifest schema versions 1 and 2).

The normative specification is ``docs/interventions/profile-format.md``; the C# reference implementation is
``ai-dotnet/src/Starlings.Interventions``. This module reads ``profile.json`` and directions files as strictly as the
C# reader does, and writes ``compiler-manifest.json`` in the canonical form the C# serializer produces, so a manifest
written here is accepted by ``InterventionManifest.Load``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
# Profile schema 2 adds the operator ``directional_addition``; manifest schema 2 holds several profiles and a strength
# vector. Version 1 documents are still read and written unchanged.
SCHEMA_VERSION_2 = 2
PROFILE_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION, SCHEMA_VERSION_2})
MANIFEST_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION, SCHEMA_VERSION_2})
MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
MAX_SITES = 1024
MAX_PROFILES = 64  # profiles in one schema 2 manifest (strength-vector length)
MAX_HIDDEN_SIZE = 1 << 20
MAX_INPUT_RANK = 8
MAX_INPUT_ELEMENTS = 2**31 - 1

OPERATOR_DIRECTIONAL_PROJECTION = "directional_projection"
OPERATOR_DIRECTIONAL_ADDITION = "directional_addition"
TARGET_RESIDUAL_POST = "residual_post"
DTYPE_FLOAT32 = "float32"
ROLE_STRENGTH = "strength"
ARTIFACT_ROLE_GRAPH = "graph"
ARTIFACT_ROLE_EXTERNAL_DATA = "external_data"

OPERATORS = frozenset({OPERATOR_DIRECTIONAL_PROJECTION})  # schema version 1 (profiles and manifests)
OPERATORS_V2 = frozenset({OPERATOR_DIRECTIONAL_PROJECTION, OPERATOR_DIRECTIONAL_ADDITION})
TARGETS = frozenset({TARGET_RESIDUAL_POST})
DTYPES = frozenset({DTYPE_FLOAT32})
INPUT_ROLES = frozenset({ROLE_STRENGTH})
ARTIFACT_ROLES = frozenset({ARTIFACT_ROLE_GRAPH, ARTIFACT_ROLE_EXTERNAL_DATA})
# What onnx.checker said about a derived graph (manifest "validation.onnx_checker").
CHECKER_PASSED = "passed"
CHECKER_SKIPPED_BASE_REJECTED = "skipped_base_rejected"
CHECKER_RESULTS = frozenset({CHECKER_PASSED, CHECKER_SKIPPED_BASE_REJECTED})
MAX_VALIDATION_DETAIL = 512

_PROFILE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_VERSION_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+/-]{0,63}")
_PRECISION = re.compile(r"[a-z0-9][a-z0-9_]{0,31}")
_SHA256 = re.compile(r"[0-9a-f]{64}")

_PROFILE_KEYS = ("schema_version", "id", "description", "base_model_sha256", "operator", "sites")
_PROFILE_OPTIONAL = frozenset({"description"})
_SITE_KEYS = ("layer", "target", "hidden_size", "rank", "dtype", "directions_file", "directions_sha256", "scales")


class InterventionContractError(ValueError):
    """A profile, directions file or manifest breaks the schema; ``path`` is the JSON path when there is one."""

    def __init__(self, message: str, path: str | None = None) -> None:
        super().__init__(f"{path}: {message}" if path else message)
        self.json_path = path


class InterventionMismatchError(ValueError):
    """An identity or hash differs from the one expected (wrong base graph, changed file)."""


# --------------------------------------------------------------------------------------------------------------------
# Shared rules


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def is_profile_id(value: object) -> bool:
    return isinstance(value, str) and _PROFILE_ID.fullmatch(value) is not None


def is_version_label(value: object) -> bool:
    return isinstance(value, str) and _VERSION_LABEL.fullmatch(value) is not None


def is_precision(value: object) -> bool:
    return isinstance(value, str) and _PRECISION.fullmatch(value) is not None


def utf16_length(value: str) -> int:
    """Length in UTF-16 code units, as C# ``string.Length`` counts it."""
    return len(value.encode("utf-16-le")) // 2


def _has_control(value: str) -> bool:
    return any(unicodedata.category(c) == "Cc" for c in value)


def is_tensor_name(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 1024 and not _has_control(value)


def is_relative_path(value: object) -> bool:
    """Contract-relative path: forward slashes, not rooted, no empty/'.'/'..' segment, no backslash/colon/control."""
    if not isinstance(value, str) or not value or len(value) > 512 or value[0] == "/":
        return False
    if any(c in "\\:" for c in value) or _has_control(value):
        return False
    return all(segment not in ("", ".", "..") for segment in value.split("/"))


def resolve_relative(directory: Path, relative: str) -> Path:
    """Resolve a contract-relative path under ``directory``; the result never leaves it (text check, as in C#)."""
    if not is_relative_path(relative):
        raise InterventionContractError(f"'{relative}' is not a valid relative artifact path.")
    root = Path(directory).absolute()
    return root.joinpath(*relative.split("/"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------------------------------------------------------------
# Strict JSON


def _reject_constant(name: str) -> Any:
    raise InterventionContractError(f"'{name}' is not standard JSON.")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InterventionContractError(f"property '{key}' appears more than once.")
        result[key] = value
    return result


def parse_strict_json(data: bytes, what: str) -> Any:
    """Parse UTF-8 JSON as the C# reader does: no BOM, no NaN/Infinity, no duplicate keys, at most 16 MiB."""
    if len(data) > MAX_DOCUMENT_BYTES:
        raise InterventionContractError(f"{what} is {len(data)} bytes; the limit is {MAX_DOCUMENT_BYTES}.")
    if data.startswith(b"\xef\xbb\xbf"):
        raise InterventionContractError(f"{what} starts with a UTF-8 byte order mark; write UTF-8 without one.")
    try:
        text = data.decode("utf-8")
        return json.loads(text, object_pairs_hook=_pairs, parse_constant=_reject_constant)
    except InterventionContractError as e:
        raise InterventionContractError(f"{what} is not valid: {e}") from e
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise InterventionContractError(f"{what} is not valid JSON: {e}") from e


def _object(value: Any, path: str, keys: tuple[str, ...], optional: frozenset[str] = frozenset()) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InterventionContractError(f"expected an object, found {_kind(value)}.", path)
    for key in value:
        if key not in keys:
            raise InterventionContractError(f"unknown property '{key}'.", path)
    for key in keys:
        if key not in value and key not in optional:
            raise InterventionContractError(f"missing required property '{key}'.", path)
    return value


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, list):
        return "an array"
    return "an object"


def _int(value: Any, path: str, minimum: int, maximum: int) -> int:
    # bool is an int subclass, and 1.0 / 1e0 parse as float: both are rejected, as in C#.
    if type(value) is not int:
        raise InterventionContractError("expected an integer without fraction or exponent.", path)
    if not minimum <= value <= maximum:
        raise InterventionContractError(f"expected an integer in [{minimum}, {maximum}], found {value}.", path)
    return value


def _str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise InterventionContractError(f"expected a string, found {_kind(value)}.", path)
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise InterventionContractError(f"expected an array, found {_kind(value)}.", path)
    return value


def _sha(value: Any, path: str) -> str:
    if not is_sha256(value):
        raise InterventionContractError("expected a SHA-256 as 64 lower-case hexadecimal characters.", path)
    return value


def _vocab(value: Any, path: str, allowed: frozenset[str], what: str) -> str:
    text = _str(value, path)
    if text not in allowed:
        raise InterventionContractError(f"unknown {what} '{text}'; expected one of {sorted(allowed)}.", path)
    return text


def _schema_version(
    document: Any, path: str = "schema_version", allowed: frozenset[int] = frozenset({SCHEMA_VERSION})
) -> int:
    if not isinstance(document, dict):
        raise InterventionContractError(f"expected an object, found {_kind(document)}.", "$")
    if "schema_version" not in document:
        raise InterventionContractError("missing required property 'schema_version'.", "$")
    version = document["schema_version"]
    if type(version) is not int or version not in allowed:
        expected = " or ".join(str(v) for v in sorted(allowed))
        raise InterventionContractError(f"unsupported schema version {version!r}; expected {expected}.", path)
    return version


# --------------------------------------------------------------------------------------------------------------------
# Profile


@dataclass(frozen=True)
class InterventionSite:
    layer: int
    target: str
    hidden_size: int
    rank: int
    dtype: str
    directions_file: str
    directions_sha256: str
    scales: tuple[float, ...]  # float32 values, held as Python floats that are exactly representable in float32

    @property
    def key(self) -> tuple[int, str]:
        return (self.layer, self.target)


@dataclass(frozen=True)
class InterventionProfile:
    id: str
    description: str | None
    base_model_sha256: str
    operator: str
    sites: tuple[InterventionSite, ...]
    # 1 or 2. Version 1 allows only ``directional_projection``; version 2 adds ``directional_addition``.
    schema_version: int = SCHEMA_VERSION


@dataclass(frozen=True)
class InterventionProfileFile:
    """A profile read from disk, with the SHA-256 of its bytes as stored and its directory."""

    path: Path
    sha256: str
    profile: InterventionProfile

    @property
    def directory(self) -> Path:
        return self.path.parent

    def load_directions(self, site: InterventionSite) -> np.ndarray:
        return load_directions(resolve_relative(self.directory, site.directions_file), site)

    def verify_base_model(self, sha256: str) -> None:
        if sha256 != self.profile.base_model_sha256:
            raise InterventionMismatchError(
                f"profile '{self.profile.id}' targets base graph {self.profile.base_model_sha256}, "
                f"but the graph has SHA-256 {sha256}."
            )


def _float32(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InterventionContractError("expected a finite float32 number.", path)
    with np.errstate(over="ignore"):
        single = np.float32(value)
    if not math.isfinite(float(single)):
        raise InterventionContractError("expected a finite float32 number.", path)
    return float(single)


def parse_profile(data: bytes) -> InterventionProfile:
    document = parse_strict_json(data, "profile.json")
    version = _schema_version(document, allowed=PROFILE_SCHEMA_VERSIONS)
    root = _object(document, "$", _PROFILE_KEYS, _PROFILE_OPTIONAL)
    profile_id = _str(root["id"], "id")
    if not is_profile_id(profile_id):
        raise InterventionContractError(f"'{profile_id}' is not a valid profile id.", "id")
    description = _str(root["description"], "description") if "description" in root else None
    base = _sha(root["base_model_sha256"], "base_model_sha256")
    operators = OPERATORS if version == SCHEMA_VERSION else OPERATORS_V2
    operator = _vocab(root["operator"], "operator", operators, "operator")
    raw_sites = _list(root["sites"], "sites")
    if not 1 <= len(raw_sites) <= MAX_SITES:
        raise InterventionContractError(f"expected 1 to {MAX_SITES} sites, found {len(raw_sites)}.", "sites")
    sites: list[InterventionSite] = []
    seen: set[tuple[int, str]] = set()
    for i, raw in enumerate(raw_sites):
        path = f"sites[{i}]"
        site = _object(raw, path, _SITE_KEYS)
        layer = _int(site["layer"], f"{path}.layer", 0, 2**31 - 1)
        target = _vocab(site["target"], f"{path}.target", TARGETS, "target")
        hidden = _int(site["hidden_size"], f"{path}.hidden_size", 1, MAX_HIDDEN_SIZE)
        rank = _int(site["rank"], f"{path}.rank", 1, hidden)
        dtype = _vocab(site["dtype"], f"{path}.dtype", DTYPES, "dtype")
        directions_file = _str(site["directions_file"], f"{path}.directions_file")
        if not is_relative_path(directions_file):
            raise InterventionContractError(
                f"'{directions_file}' is not a valid relative path.", f"{path}.directions_file"
            )
        directions_sha = _sha(site["directions_sha256"], f"{path}.directions_sha256")
        raw_scales = _list(site["scales"], f"{path}.scales")
        if len(raw_scales) != rank:
            raise InterventionContractError(
                f"expected exactly rank={rank} scales, found {len(raw_scales)}.", f"{path}.scales"
            )
        scales = tuple(_float32(v, f"{path}.scales[{j}]") for j, v in enumerate(raw_scales))
        key = (layer, target)
        if key in seen:
            raise InterventionContractError(f"two sites have layer {layer} and target '{target}'.", "sites")
        seen.add(key)
        sites.append(InterventionSite(layer, target, hidden, rank, dtype, directions_file, directions_sha, scales))
    sites.sort(key=lambda s: s.key)
    return InterventionProfile(profile_id, description, base, operator, tuple(sites), version)


def load_profile(path: str | Path) -> InterventionProfileFile:
    path = Path(path)
    data = path.read_bytes()
    return InterventionProfileFile(path.absolute(), sha256_bytes(data), parse_profile(data))


# --------------------------------------------------------------------------------------------------------------------
# Directions


def decode_directions(data: bytes, site: InterventionSite) -> np.ndarray:
    """Decode raw little-endian float32 ``[rank, hidden_size]``: size, then SHA-256, then finite values."""
    expected = site.rank * site.hidden_size * 4
    if len(data) != expected:
        raise InterventionContractError(
            f"directions file '{site.directions_file}' is {len(data)} bytes; expected rank*hidden*4 = {expected}."
        )
    actual = sha256_bytes(data)
    if actual != site.directions_sha256:
        raise InterventionMismatchError(
            f"directions file '{site.directions_file}' has SHA-256 {actual}; the profile records "
            f"{site.directions_sha256}."
        )
    q = np.frombuffer(data, dtype="<f4").reshape(site.rank, site.hidden_size).astype(np.float32)
    if not np.all(np.isfinite(q)):
        raise InterventionContractError(f"directions file '{site.directions_file}' holds a NaN or infinite value.")
    return q


def load_directions(path: Path, site: InterventionSite) -> np.ndarray:
    return decode_directions(Path(path).read_bytes(), site)


def encode_directions(q: np.ndarray) -> bytes:
    q = np.asarray(q)
    if q.ndim != 2:
        raise InterventionContractError(f"directions must be a 2-D [rank, hidden] array, got shape {q.shape}.")
    return np.ascontiguousarray(q, dtype="<f4").tobytes(order="C")


def max_orthonormality_error(q: np.ndarray) -> float:
    """Largest absolute deviation of ``Q Q^T`` from the identity (a diagnostic; orthonormality is not enforced)."""
    q64 = np.asarray(q, dtype=np.float64)
    return float(np.max(np.abs(q64 @ q64.T - np.eye(q64.shape[0]))))


def format_float32(value: float) -> str:
    """Shortest decimal text that round-trips to the same float32 (``0.1``, ``0.33333334``, ``1``)."""
    single = np.float32(value)
    if not math.isfinite(float(single)):
        raise InterventionContractError(f"{value!r} is not a finite float32.")
    return np.format_float_positional(single, unique=True, trim="-")


# --------------------------------------------------------------------------------------------------------------------
# Compiler manifest


@dataclass(frozen=True)
class FileIdentity:
    """Where a derived file's bytes live on this machine: ``st_dev``, ``st_ino`` and ``st_mtime_ns`` (stat, following
    symlinks) when the entry was written. A hard link shares these with its base file. Machine-local: a copied or
    restored entry no longer matches, and Fast verification then fails (Full, which hashes, is unaffected)."""

    device: int
    inode: int
    mtime_ns: int

    @staticmethod
    def of(path: str | Path) -> FileIdentity:
        st = os.stat(path)
        return FileIdentity(st.st_dev, st.st_ino, st.st_mtime_ns)


@dataclass(frozen=True)
class ArtifactFile:
    path: str
    sha256: str
    size: int
    # Only on ``derived_files`` entries (optional in the schema; the compiler records it for every file it links).
    identity: FileIdentity | None = None


@dataclass(frozen=True)
class ManifestValidation:
    """How the compiler validated the derived graph (manifest ``validation``, optional in the schema).

    ``onnx_checker`` is ``passed``, or ``skipped_base_rejected`` when onnx.checker already rejects the base graph (for
    example operators only ONNX Runtime registers in the default domain); ``detail`` is then the first line of the
    checker's message about the base graph. The structural checks of ``validate_derived_model`` run either way.
    """

    onnx_checker: str
    detail: str | None = None


@dataclass(frozen=True)
class BaseArtifact:
    role: str
    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class GraphInput:
    name: str
    role: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class SiteBinding:
    layer: int
    target: str
    tensor: str


@dataclass(frozen=True)
class InterventionManifest:
    compiler_version: str
    site_mapping_version: str
    base_artifacts: tuple[BaseArtifact, ...]
    profile_id: str
    profile_sha256: str
    operator: str
    opset: int
    precision: str
    inputs: tuple[GraphInput, ...]
    sites: tuple[SiteBinding, ...]
    derived_graph: ArtifactFile
    derived_files: tuple[ArtifactFile, ...]
    validation: ManifestValidation | None = None

    @property
    def base_graph(self) -> BaseArtifact:
        return next(a for a in self.base_artifacts if a.role == ARTIFACT_ROLE_GRAPH)

    @property
    def strength_input(self) -> GraphInput:
        return next(i for i in self.inputs if i.role == ROLE_STRENGTH)

    def validate(self) -> InterventionManifest:
        """Apply every rule the C# reader applies; raise ``InterventionContractError`` on the first violation."""
        _manifest_from_document(manifest_to_document(self))
        return self

    def serialize(self) -> bytes:
        self.validate()
        return serialize_manifest_document(manifest_to_document(self))


_MANIFEST_KEYS = (
    "schema_version",
    "compiler_version",
    "site_mapping_version",
    "base_artifacts",
    "profile_id",
    "profile_sha256",
    "operator",
    "opset",
    "precision",
    "inputs",
    "sites",
    "derived_graph",
    "derived_files",
    "validation",
)
_MANIFEST_OPTIONAL = frozenset({"validation"})
_BASE_ARTIFACT_KEYS = ("role", "path", "sha256", "size")
_FILE_KEYS = ("path", "sha256", "size")
_DERIVED_FILE_KEYS = ("path", "sha256", "size", "identity")
_IDENTITY_KEYS = ("device", "inode", "mtime_ns")
_VALIDATION_KEYS = ("onnx_checker", "detail")
_INPUT_KEYS = ("name", "role", "dtype", "shape")
_BINDING_KEYS = ("layer", "target", "tensor")
_MAX_SIZE = 2**63 - 1


def manifest_to_document(manifest: InterventionManifest) -> dict[str, Any]:
    """Plain dict in canonical key order (sites sorted by layer, then target)."""

    def file(f: ArtifactFile) -> dict[str, Any]:
        document: dict[str, Any] = {"path": f.path, "sha256": f.sha256, "size": f.size}
        if f.identity is not None:
            document["identity"] = {
                "device": f.identity.device,
                "inode": f.identity.inode,
                "mtime_ns": f.identity.mtime_ns,
            }
        return document

    document = {
        "schema_version": SCHEMA_VERSION,
        "compiler_version": manifest.compiler_version,
        "site_mapping_version": manifest.site_mapping_version,
        "base_artifacts": [
            {"role": a.role, "path": a.path, "sha256": a.sha256, "size": a.size} for a in manifest.base_artifacts
        ],
        "profile_id": manifest.profile_id,
        "profile_sha256": manifest.profile_sha256,
        "operator": manifest.operator,
        "opset": manifest.opset,
        "precision": manifest.precision,
        "inputs": [{"name": i.name, "role": i.role, "dtype": i.dtype, "shape": list(i.shape)} for i in manifest.inputs],
        "sites": [
            {"layer": s.layer, "target": s.target, "tensor": s.tensor}
            for s in sorted(manifest.sites, key=lambda s: (s.layer, s.target))
        ],
        "derived_graph": file(manifest.derived_graph),
        "derived_files": [file(f) for f in manifest.derived_files],
    }
    if manifest.validation is not None:
        validation: dict[str, Any] = {"onnx_checker": manifest.validation.onnx_checker}
        if manifest.validation.detail is not None:
            validation["detail"] = manifest.validation.detail
        document["validation"] = validation
    return document


def serialize_manifest_document(document: dict[str, Any]) -> bytes:
    """Canonical JSON: two-space indent, LF, final LF, no BOM, non-ASCII kept as UTF-8."""
    return (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _file(value: Any, path: str, keys: tuple[str, ...] = _FILE_KEYS) -> ArtifactFile:
    obj = _object(value, path, keys, frozenset({"identity"}))
    rel = _str(obj["path"], f"{path}.path")
    if not is_relative_path(rel):
        raise InterventionContractError(f"'{rel}' is not a valid relative path.", f"{path}.path")
    identity = None
    if "identity" in obj:
        raw = _object(obj["identity"], f"{path}.identity", _IDENTITY_KEYS)
        identity = FileIdentity(
            _int(raw["device"], f"{path}.identity.device", 0, _MAX_SIZE),
            _int(raw["inode"], f"{path}.identity.inode", 0, _MAX_SIZE),
            _int(raw["mtime_ns"], f"{path}.identity.mtime_ns", -_MAX_SIZE - 1, _MAX_SIZE),
        )
    return ArtifactFile(
        rel, _sha(obj["sha256"], f"{path}.sha256"), _int(obj["size"], f"{path}.size", 0, _MAX_SIZE), identity
    )


def _validation(value: Any, path: str) -> ManifestValidation:
    obj = _object(value, path, _VALIDATION_KEYS, frozenset({"detail"}))
    result = _vocab(obj["onnx_checker"], f"{path}.onnx_checker", CHECKER_RESULTS, "onnx_checker result")
    detail = None
    if "detail" in obj:
        detail = _str(obj["detail"], f"{path}.detail")
        if not detail or utf16_length(detail) > MAX_VALIDATION_DETAIL or _has_control(detail):
            raise InterventionContractError(
                f"expected 1 to {MAX_VALIDATION_DETAIL} characters without control characters.", f"{path}.detail"
            )
    return ManifestValidation(result, detail)


def _manifest_from_document(document: Any) -> InterventionManifest | MultiProfileManifest:
    if _schema_version(document, allowed=MANIFEST_SCHEMA_VERSIONS) == SCHEMA_VERSION_2:
        return _multi_manifest_from_document(document)
    root = _object(document, "$", _MANIFEST_KEYS, _MANIFEST_OPTIONAL)
    compiler_version = _str(root["compiler_version"], "compiler_version")
    if not is_version_label(compiler_version):
        raise InterventionContractError(f"'{compiler_version}' is not a valid version label.", "compiler_version")
    mapping_version = _str(root["site_mapping_version"], "site_mapping_version")
    if not is_version_label(mapping_version):
        raise InterventionContractError(f"'{mapping_version}' is not a valid version label.", "site_mapping_version")

    artifacts: list[BaseArtifact] = []
    for i, raw in enumerate(_list(root["base_artifacts"], "base_artifacts")):
        path = f"base_artifacts[{i}]"
        obj = _object(raw, path, _BASE_ARTIFACT_KEYS)
        role = _vocab(obj["role"], f"{path}.role", ARTIFACT_ROLES, "base artifact role")
        rel = _str(obj["path"], f"{path}.path")
        if not is_relative_path(rel):
            raise InterventionContractError(f"'{rel}' is not a valid relative path.", f"{path}.path")
        artifacts.append(
            BaseArtifact(
                role, rel, _sha(obj["sha256"], f"{path}.sha256"), _int(obj["size"], f"{path}.size", 0, _MAX_SIZE)
            )
        )
    if sum(a.role == ARTIFACT_ROLE_GRAPH for a in artifacts) != 1:
        raise InterventionContractError("expected exactly one base artifact with role 'graph'.", "base_artifacts")
    if len({a.path for a in artifacts}) != len(artifacts):
        raise InterventionContractError("base artifact paths must be unique.", "base_artifacts")

    profile_id = _str(root["profile_id"], "profile_id")
    if not is_profile_id(profile_id):
        raise InterventionContractError(f"'{profile_id}' is not a valid profile id.", "profile_id")
    profile_sha = _sha(root["profile_sha256"], "profile_sha256")
    operator = _vocab(root["operator"], "operator", OPERATORS, "operator")
    opset = _int(root["opset"], "opset", 1, 1000)
    precision = _str(root["precision"], "precision")
    if not is_precision(precision):
        raise InterventionContractError(f"'{precision}' is not a valid precision.", "precision")

    inputs: list[GraphInput] = []
    for i, raw in enumerate(_list(root["inputs"], "inputs")):
        path = f"inputs[{i}]"
        obj = _object(raw, path, _INPUT_KEYS)
        name = _str(obj["name"], f"{path}.name")
        if not is_tensor_name(name):
            raise InterventionContractError("not a valid ONNX input name.", f"{path}.name")
        role = _vocab(obj["role"], f"{path}.role", INPUT_ROLES, "input role")
        dtype = _vocab(obj["dtype"], f"{path}.dtype", DTYPES, "dtype")
        dims = _list(obj["shape"], f"{path}.shape")
        if len(dims) > MAX_INPUT_RANK:
            raise InterventionContractError(f"at most {MAX_INPUT_RANK} dimensions.", f"{path}.shape")
        shape = tuple(_int(d, f"{path}.shape[{j}]", 0, _MAX_SIZE) for j, d in enumerate(dims))
        if math.prod(shape) > MAX_INPUT_ELEMENTS:
            raise InterventionContractError(f"more than {MAX_INPUT_ELEMENTS} elements.", f"{path}.shape")
        inputs.append(GraphInput(name, role, dtype, shape))
    strengths = [g for g in inputs if g.role == ROLE_STRENGTH]
    if len(strengths) != 1:
        raise InterventionContractError("expected exactly one input with role 'strength'.", "inputs")
    if strengths[0].shape not in ((), (1,)):
        raise InterventionContractError("the strength input's shape must be [] or [1].", "inputs")
    if len({g.name for g in inputs}) != len(inputs):
        raise InterventionContractError("input names must be unique.", "inputs")

    raw_sites = _list(root["sites"], "sites")
    if not 1 <= len(raw_sites) <= MAX_SITES:
        raise InterventionContractError(f"expected 1 to {MAX_SITES} sites, found {len(raw_sites)}.", "sites")
    sites: list[SiteBinding] = []
    for i, raw in enumerate(raw_sites):
        path = f"sites[{i}]"
        obj = _object(raw, path, _BINDING_KEYS)
        tensor = _str(obj["tensor"], f"{path}.tensor")
        if not is_tensor_name(tensor):
            raise InterventionContractError("not a valid ONNX tensor name.", f"{path}.tensor")
        sites.append(
            SiteBinding(
                _int(obj["layer"], f"{path}.layer", 0, 2**31 - 1),
                _vocab(obj["target"], f"{path}.target", TARGETS, "target"),
                tensor,
            )
        )
    if len({(s.layer, s.target) for s in sites}) != len(sites):
        raise InterventionContractError("two site bindings have the same layer and target.", "sites")
    if len({s.tensor for s in sites}) != len(sites):
        raise InterventionContractError("two site bindings name the same tensor.", "sites")

    derived_graph = _file(root["derived_graph"], "derived_graph")
    derived_files = tuple(
        _file(v, f"derived_files[{i}]", _DERIVED_FILE_KEYS)
        for i, v in enumerate(_list(root["derived_files"], "derived_files"))
    )
    paths = [f.path for f in derived_files]
    if len(set(paths)) != len(paths) or derived_graph.path in paths:
        raise InterventionContractError(
            "derived file paths must be unique and differ from derived_graph.", "derived_files"
        )

    return InterventionManifest(
        compiler_version,
        mapping_version,
        tuple(artifacts),
        profile_id,
        profile_sha,
        operator,
        opset,
        precision,
        tuple(inputs),
        tuple(sorted(sites, key=lambda s: (s.layer, s.target))),
        derived_graph,
        derived_files,
        _validation(root["validation"], "validation") if "validation" in root else None,
    )


def parse_manifest(data: bytes) -> InterventionManifest | MultiProfileManifest:
    """A schema 1 document gives an ``InterventionManifest``, a schema 2 document a ``MultiProfileManifest``."""
    return _manifest_from_document(parse_strict_json(data, "compiler-manifest.json"))


def load_manifest(path: str | Path) -> InterventionManifest | MultiProfileManifest:
    return parse_manifest(Path(path).read_bytes())


def verify_manifest_profile(manifest: InterventionManifest, profile: InterventionProfileFile) -> None:
    """The checks of C# ``InterventionManifest.VerifyProfile``."""
    if isinstance(manifest, MultiProfileManifest):
        verify_manifest_profiles(manifest, [profile])
        return
    p = profile.profile
    if manifest.profile_id != p.id:
        raise InterventionMismatchError(f"manifest is for profile '{manifest.profile_id}', not '{p.id}'.")
    if manifest.profile_sha256 != profile.sha256:
        raise InterventionMismatchError(
            f"manifest records profile SHA-256 {manifest.profile_sha256}; profile.json has {profile.sha256}."
        )
    if manifest.operator != p.operator:
        raise InterventionMismatchError(f"manifest operator '{manifest.operator}' differs from '{p.operator}'.")
    if manifest.base_graph.sha256 != p.base_model_sha256:
        raise InterventionMismatchError(
            f"manifest base graph {manifest.base_graph.sha256} differs from the profile's {p.base_model_sha256}."
        )
    if sorted((s.layer, s.target) for s in manifest.sites) != sorted(s.key for s in p.sites):
        raise InterventionMismatchError("manifest sites differ from the profile's sites.")


def verify_artifact_file(path: Path, sha256: str, size: int) -> None:
    """Size first, then SHA-256 (the order of C# ``InterventionHashes.VerifyFile``)."""
    path = Path(path)
    if not path.is_file():
        raise InterventionMismatchError(f"'{path}' does not exist.")
    actual_size = path.stat().st_size
    if actual_size != size:
        raise InterventionMismatchError(f"'{path}' is {actual_size} bytes; expected {size}.")
    actual = sha256_file(path)
    if actual != sha256:
        raise InterventionMismatchError(f"'{path}' has SHA-256 {actual}; expected {sha256}.")


# --------------------------------------------------------------------------------------------------------------------
# Compiler manifest, schema version 2: several profiles and a strength vector


@dataclass(frozen=True)
class ManifestProfile:
    """One compiled profile of a schema 2 manifest; its position in ``profiles`` is its strength-vector index."""

    id: str
    sha256: str
    operator: str


@dataclass(frozen=True)
class MultiSiteBinding:
    """A resolved site of a schema 2 manifest and the profiles applied there, by strength index, in application order
    (ascending): profile ``profiles[0]`` reads the base tensor, each later one reads the previous one's output."""

    layer: int
    target: str
    tensor: str
    profiles: tuple[int, ...]


@dataclass(frozen=True)
class MultiProfileManifest:
    """``compiler-manifest.json`` schema version 2 (``docs/interventions/profile-format.md``)."""

    compiler_version: str
    site_mapping_version: str
    base_artifacts: tuple[BaseArtifact, ...]
    profiles: tuple[ManifestProfile, ...]
    opset: int
    precision: str
    inputs: tuple[GraphInput, ...]
    sites: tuple[MultiSiteBinding, ...]
    derived_graph: ArtifactFile
    derived_files: tuple[ArtifactFile, ...]
    validation: ManifestValidation | None = None

    schema_version = SCHEMA_VERSION_2

    @property
    def base_graph(self) -> BaseArtifact:
        return next(a for a in self.base_artifacts if a.role == ARTIFACT_ROLE_GRAPH)

    @property
    def strength_input(self) -> GraphInput:
        return next(i for i in self.inputs if i.role == ROLE_STRENGTH)

    @property
    def profile_ids(self) -> tuple[str, ...]:
        return tuple(p.id for p in self.profiles)

    def validate(self) -> MultiProfileManifest:
        _multi_manifest_from_document(multi_manifest_to_document(self))
        return self

    def serialize(self) -> bytes:
        self.validate()
        return serialize_manifest_document(multi_manifest_to_document(self))


_MULTI_MANIFEST_KEYS = (
    "schema_version",
    "compiler_version",
    "site_mapping_version",
    "base_artifacts",
    "profiles",
    "opset",
    "precision",
    "inputs",
    "sites",
    "derived_graph",
    "derived_files",
    "validation",
)
_MANIFEST_PROFILE_KEYS = ("id", "sha256", "operator")
_MULTI_BINDING_KEYS = ("layer", "target", "tensor", "profiles")


def _file_document(f: ArtifactFile) -> dict[str, Any]:
    document: dict[str, Any] = {"path": f.path, "sha256": f.sha256, "size": f.size}
    if f.identity is not None:
        document["identity"] = {"device": f.identity.device, "inode": f.identity.inode, "mtime_ns": f.identity.mtime_ns}
    return document


def multi_manifest_to_document(manifest: MultiProfileManifest) -> dict[str, Any]:
    """Plain dict in canonical key order (sites sorted by layer, then target; profiles in strength-vector order)."""
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION_2,
        "compiler_version": manifest.compiler_version,
        "site_mapping_version": manifest.site_mapping_version,
        "base_artifacts": [
            {"role": a.role, "path": a.path, "sha256": a.sha256, "size": a.size} for a in manifest.base_artifacts
        ],
        "profiles": [{"id": p.id, "sha256": p.sha256, "operator": p.operator} for p in manifest.profiles],
        "opset": manifest.opset,
        "precision": manifest.precision,
        "inputs": [{"name": i.name, "role": i.role, "dtype": i.dtype, "shape": list(i.shape)} for i in manifest.inputs],
        "sites": [
            {"layer": s.layer, "target": s.target, "tensor": s.tensor, "profiles": list(s.profiles)}
            for s in sorted(manifest.sites, key=lambda s: (s.layer, s.target))
        ],
        "derived_graph": _file_document(manifest.derived_graph),
        "derived_files": [_file_document(f) for f in manifest.derived_files],
    }
    if manifest.validation is not None:
        validation: dict[str, Any] = {"onnx_checker": manifest.validation.onnx_checker}
        if manifest.validation.detail is not None:
            validation["detail"] = manifest.validation.detail
        document["validation"] = validation
    return document


def _multi_manifest_from_document(document: Any) -> MultiProfileManifest:
    """Schema 2. The parts shared with schema 1 follow the same rules; the differences are ``profiles`` (1 to 64, ids
    unique, operators from the schema 2 vocabulary), a strength input of shape exactly ``[len(profiles)]``, and a
    ``profiles`` list on each site binding (strength indices, strictly ascending, each profile at one site at least)."""
    _schema_version(document, allowed=frozenset({SCHEMA_VERSION_2}))
    root = _object(document, "$", _MULTI_MANIFEST_KEYS, _MANIFEST_OPTIONAL)
    # The schema 1 parser checks the shared fields; feed it a schema 1 view with a placeholder single profile.
    shared = {k: v for k, v in root.items() if k not in ("profiles", "sites", "inputs")}
    shared |= {
        "schema_version": SCHEMA_VERSION,
        "profile_id": "placeholder",
        "profile_sha256": "0" * 64,
        "operator": OPERATOR_DIRECTIONAL_PROJECTION,
        "inputs": [],
        "sites": [],
    }
    raw_profiles = _list(root["profiles"], "profiles")
    if not 1 <= len(raw_profiles) <= MAX_PROFILES:
        raise InterventionContractError(
            f"expected 1 to {MAX_PROFILES} profiles, found {len(raw_profiles)}.", "profiles"
        )
    profiles: list[ManifestProfile] = []
    for i, raw in enumerate(raw_profiles):
        path = f"profiles[{i}]"
        obj = _object(raw, path, _MANIFEST_PROFILE_KEYS)
        profile_id = _str(obj["id"], f"{path}.id")
        if not is_profile_id(profile_id):
            raise InterventionContractError(f"'{profile_id}' is not a valid profile id.", f"{path}.id")
        profiles.append(
            ManifestProfile(
                profile_id,
                _sha(obj["sha256"], f"{path}.sha256"),
                _vocab(obj["operator"], f"{path}.operator", OPERATORS_V2, "operator"),
            )
        )
    if len({p.id for p in profiles}) != len(profiles):
        raise InterventionContractError("profile ids must be unique.", "profiles")

    inputs = _inputs(root["inputs"])
    strengths = [g for g in inputs if g.role == ROLE_STRENGTH]
    if len(strengths) != 1:
        raise InterventionContractError("expected exactly one input with role 'strength'.", "inputs")
    if strengths[0].shape != (len(profiles),):
        raise InterventionContractError(
            f"the strength input's shape must be [{len(profiles)}], one element per profile.", "inputs"
        )

    raw_sites = _list(root["sites"], "sites")
    if not 1 <= len(raw_sites) <= MAX_SITES:
        raise InterventionContractError(f"expected 1 to {MAX_SITES} sites, found {len(raw_sites)}.", "sites")
    sites: list[MultiSiteBinding] = []
    used: set[int] = set()
    for i, raw in enumerate(raw_sites):
        path = f"sites[{i}]"
        obj = _object(raw, path, _MULTI_BINDING_KEYS)
        tensor = _str(obj["tensor"], f"{path}.tensor")
        if not is_tensor_name(tensor):
            raise InterventionContractError("not a valid ONNX tensor name.", f"{path}.tensor")
        raw_indices = _list(obj["profiles"], f"{path}.profiles")
        indices = tuple(_int(v, f"{path}.profiles[{j}]", 0, len(profiles) - 1) for j, v in enumerate(raw_indices))
        if not indices or any(a >= b for a, b in zip(indices, indices[1:], strict=False)):
            raise InterventionContractError(
                "expected one or more strength indices in strictly ascending order.", f"{path}.profiles"
            )
        used.update(indices)
        sites.append(
            MultiSiteBinding(
                _int(obj["layer"], f"{path}.layer", 0, 2**31 - 1),
                _vocab(obj["target"], f"{path}.target", TARGETS, "target"),
                tensor,
                indices,
            )
        )
    if len({(s.layer, s.target) for s in sites}) != len(sites):
        raise InterventionContractError("two site bindings have the same layer and target.", "sites")
    if len({s.tensor for s in sites}) != len(sites):
        raise InterventionContractError("two site bindings name the same tensor.", "sites")
    unused = sorted(set(range(len(profiles))) - used)
    if unused:
        raise InterventionContractError(f"profiles {unused} are applied at no site.", "sites")

    # Shared fields (versions, base artifacts, opset, precision, derived files, validation) by the schema 1 rules.
    shared["inputs"] = [{"name": "placeholder", "role": ROLE_STRENGTH, "dtype": DTYPE_FLOAT32, "shape": [1]}]
    shared["sites"] = [{"layer": 0, "target": TARGET_RESIDUAL_POST, "tensor": "placeholder"}]
    v1 = _manifest_from_document(shared)
    return MultiProfileManifest(
        v1.compiler_version,
        v1.site_mapping_version,
        v1.base_artifacts,
        tuple(profiles),
        v1.opset,
        v1.precision,
        inputs,
        tuple(sorted(sites, key=lambda s: (s.layer, s.target))),
        v1.derived_graph,
        v1.derived_files,
        v1.validation,
    )


def _inputs(value: Any) -> tuple[GraphInput, ...]:
    inputs: list[GraphInput] = []
    for i, raw in enumerate(_list(value, "inputs")):
        path = f"inputs[{i}]"
        obj = _object(raw, path, _INPUT_KEYS)
        name = _str(obj["name"], f"{path}.name")
        if not is_tensor_name(name):
            raise InterventionContractError("not a valid ONNX input name.", f"{path}.name")
        role = _vocab(obj["role"], f"{path}.role", INPUT_ROLES, "input role")
        dtype = _vocab(obj["dtype"], f"{path}.dtype", DTYPES, "dtype")
        dims = _list(obj["shape"], f"{path}.shape")
        if len(dims) > MAX_INPUT_RANK:
            raise InterventionContractError(f"at most {MAX_INPUT_RANK} dimensions.", f"{path}.shape")
        shape = tuple(_int(d, f"{path}.shape[{j}]", 0, _MAX_SIZE) for j, d in enumerate(dims))
        if math.prod(shape) > MAX_INPUT_ELEMENTS:
            raise InterventionContractError(f"more than {MAX_INPUT_ELEMENTS} elements.", f"{path}.shape")
        inputs.append(GraphInput(name, role, dtype, shape))
    if len({g.name for g in inputs}) != len(inputs):
        raise InterventionContractError("input names must be unique.", "inputs")
    return tuple(inputs)


def verify_manifest_profiles(manifest: MultiProfileManifest, profiles: list[InterventionProfileFile]) -> None:
    """The checks of C# ``InterventionManifest.VerifyProfiles``: the profiles in strength-vector order, each with the
    recorded id, SHA-256 and operator, the manifest's base graph, and exactly the sites whose bindings list it."""
    if len(profiles) != len(manifest.profiles):
        raise InterventionMismatchError(
            f"manifest records {len(manifest.profiles)} profiles; {len(profiles)} were given."
        )
    for index, (recorded, profile) in enumerate(zip(manifest.profiles, profiles, strict=True)):
        p = profile.profile
        if recorded.id != p.id:
            raise InterventionMismatchError(f"manifest profile {index} is '{recorded.id}', not '{p.id}'.")
        if recorded.sha256 != profile.sha256:
            raise InterventionMismatchError(
                f"manifest records SHA-256 {recorded.sha256} for profile '{p.id}'; its profile.json has "
                f"{profile.sha256}."
            )
        if recorded.operator != p.operator:
            raise InterventionMismatchError(
                f"manifest operator '{recorded.operator}' of profile '{p.id}' differs from '{p.operator}'."
            )
        if manifest.base_graph.sha256 != p.base_model_sha256:
            raise InterventionMismatchError(
                f"manifest base graph {manifest.base_graph.sha256} differs from profile '{p.id}''s "
                f"{p.base_model_sha256}."
            )
        bound = sorted((s.layer, s.target) for s in manifest.sites if index in s.profiles)
        if bound != sorted(s.key for s in p.sites):
            raise InterventionMismatchError(f"manifest sites of profile '{p.id}' differ from its sites.")
