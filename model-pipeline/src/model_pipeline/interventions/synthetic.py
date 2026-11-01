"""Synthetic profiles: seeded random orthonormal directions, the only V1 evidence.

These profiles test the mechanism (strength 0 leaves outputs unchanged, other strengths match a NumPy reference). They
carry no behavioural meaning.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contract import (
    DTYPE_FLOAT32,
    OPERATOR_DIRECTIONAL_PROJECTION,
    SCHEMA_VERSION,
    SCHEMA_VERSION_2,
    TARGET_RESIDUAL_POST,
    InterventionProfileFile,
    encode_directions,
    format_float32,
    load_profile,
    sha256_bytes,
)


@dataclass(frozen=True)
class SyntheticSite:
    layer: int
    hidden_size: int
    rank: int
    scales: Sequence[float] | None = None  # default: all 1
    target: str = TARGET_RESIDUAL_POST


def random_orthonormal(rank: int, hidden_size: int, seed: int) -> np.ndarray:
    """``[rank, hidden_size]`` float32 with orthonormal rows (QR of a seeded Gaussian matrix)."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((hidden_size, rank)))
    return np.ascontiguousarray(q.T, dtype=np.float32)


def write_synthetic_profile(
    directory: str | Path,
    *,
    profile_id: str,
    base_model_sha256: str,
    sites: Sequence[SyntheticSite],
    seed: int = 0,
    description: str | None = "Synthetic random orthonormal directions; no behavioural meaning.",
    operator: str = OPERATOR_DIRECTIONAL_PROJECTION,
) -> InterventionProfileFile:
    """Write ``profile.json`` and ``directions/layer<N>.f32`` files in canonical form and load the result back.

    A ``directional_projection`` profile is written as schema version 1, any other operator as schema version 2."""
    directory = Path(directory)
    (directory / "directions").mkdir(parents=True, exist_ok=True)
    entries = []
    for index, site in enumerate(sorted(sites, key=lambda s: (s.layer, s.target))):
        q = random_orthonormal(site.rank, site.hidden_size, seed + index)
        data = encode_directions(q)
        relative = f"directions/layer{site.layer}-{site.target}.f32"
        (directory / relative).write_bytes(data)
        scales = list(site.scales) if site.scales is not None else [1.0] * site.rank
        entries.append(
            {
                "layer": site.layer,
                "target": site.target,
                "hidden_size": site.hidden_size,
                "rank": site.rank,
                "dtype": DTYPE_FLOAT32,
                "directions_file": relative,
                "directions_sha256": sha256_bytes(data),
                "scales": [_Raw(format_float32(v)) for v in scales],
            }
        )
    version = SCHEMA_VERSION if operator == OPERATOR_DIRECTIONAL_PROJECTION else SCHEMA_VERSION_2
    document: dict = {"schema_version": version, "id": profile_id}
    if description is not None:
        document["description"] = description
    document |= {"base_model_sha256": base_model_sha256, "operator": operator, "sites": entries}
    path = directory / "profile.json"
    path.write_bytes(_dumps(document).encode("utf-8"))
    return load_profile(path)


class _Raw(str):
    """A number already formatted as JSON text (float32 shortest form)."""


def _dumps(document: dict) -> str:
    # json cannot emit pre-formatted numbers, so format with placeholders and substitute the float32 texts.
    raws: list[str] = []

    def walk(value):
        if isinstance(value, _Raw):
            raws.append(str(value))
            return f"\u0000{len(raws) - 1}\u0000"
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value

    text = json.dumps(walk(document), indent=2, ensure_ascii=False)
    for index, raw in enumerate(raws):
        text = text.replace(f'"\\u0000{index}\\u0000"', raw)
    return text + "\n"
