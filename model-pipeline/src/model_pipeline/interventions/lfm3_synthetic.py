"""Compile a synthetic intervention profile into the LFM3-1.7B fp32 graph (evidence and test setup, not a real profile).

Usage::

    python -m model_pipeline.interventions.lfm3_synthetic \\
        --snapshot <LFM3-1.7B-ONNX snapshot at c680759a> --cache <cache root> [--profile-dir <dir>] [--multi | --fp16]

The snapshot is the directory holding ``onnx/model.onnx``, its four ``model.onnx_data*`` files and ``tokenizer.json``
(for example ``$STARLING_MODELS/LiquidAI/LFM3-1.7B-ONNX/c680759afee1ce8310c90d6c621f4558200dee77``). The cache root must
be on the snapshot's file system: the derived graph reaches the weights through hard links. The command writes a
profile of seeded random orthonormal directions (no behavioural meaning), compiles it with ``get_or_compile`` and
prints the cache entry directory, which ``STARLING_LFM3_INTERVENTION`` takes in the .NET tests.

With ``--multi`` it compiles two profiles into one graph with a strength vector of length 2 (compiler manifest schema
2): the same ``lfm3-synthetic-test`` profile as index 0, and ``lfm3-synthetic-add`` (``directional_addition``, rank 2,
after layers 12 and 16) as index 1; layer 16 is shared, so there the projection runs first and the addition second. The
printed entry is what ``STARLING_LFM3_INTERVENTION_MULTI`` takes in the .NET tests.

With ``--fp16`` it compiles the single synthetic profile against ``onnx/model_fp16.onnx`` (site mapping
``lfm3-1.7b/fp16/1``) instead; the snapshot must then hold the fp16 graph and its two ``model_fp16.onnx_data*`` files.
The .NET integration does not load fp16 entries yet.

The external data SHA-256 values are the fp32 asset lock's, so a cold compile hashes the 7 GB of weights once (inside
the compiler) instead of twice.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .cache import CacheEntry, get_or_compile, get_or_compile_profiles
from .contract import OPERATOR_DIRECTIONAL_ADDITION
from .injection_sites import LFM3_FP16_GRAPH_SHA256, LFM3_FP32_GRAPH_SHA256, LFM3_HIDDEN_SIZE
from .synthetic import SyntheticSite, write_synthetic_profile

LFM3_FP32_GRAPH = "onnx/model.onnx"

# The fp32 asset lock's external data files (Lfm3ModelAssets.Fp32Lock), relative to the snapshot root.
LFM3_FP32_EXTERNAL_SHA256 = {
    "onnx/model.onnx_data": "bf44fe98d130ad2bca747e9b62f28b18a3277d45bf1367da15a478538c849057",
    "onnx/model.onnx_data_1": "bb631a02237e4f8a491e12b25c333849046332b6db60fe0a2120f278c982fa76",
    "onnx/model.onnx_data_2": "ad39ed4b378c9c1b34c1a3cdb879880c7e110061070ff29fca118e8dd0b48010",
    "onnx/model.onnx_data_3": "1538827aa1c279442457028c4e3d784b7ea82bcf03abd4b55e45fa4facf9fe62",
}

LFM3_FP16_GRAPH = "onnx/model_fp16.onnx"

# The fp16 asset lock's external data files (Lfm3ModelAssets.Fp16Lock), relative to the snapshot root.
LFM3_FP16_EXTERNAL_SHA256 = {
    "onnx/model_fp16.onnx_data": "38205c6368e442fea49f52f2e66bdcca65fbdb635938ed69efb6ecb91a0bf908",
    "onnx/model_fp16.onnx_data_1": "9c7dbadee6ca54f41125b8a3641b1368a02cddb3b394f9d09e06a72f359135bb",
}

SYNTHETIC_PROFILE_ID = "lfm3-synthetic-test"
SYNTHETIC_LAYERS = (8, 16)
SYNTHETIC_RANK = 8
SYNTHETIC_SEED = 20261031

ADDITION_PROFILE_ID = "lfm3-synthetic-add"
ADDITION_LAYERS = (12, 16)
ADDITION_RANK = 2
ADDITION_SCALES = (1.0, 0.5)
ADDITION_SEED = 20261101


def write_lfm3_synthetic_profile(
    directory: str | Path, *, seed: int = SYNTHETIC_SEED, base_model_sha256: str = LFM3_FP32_GRAPH_SHA256
):
    """``profile.json`` plus directions for residual_post after layers 8 and 16, rank 8, scales 1."""
    return write_synthetic_profile(
        directory,
        profile_id=SYNTHETIC_PROFILE_ID,
        base_model_sha256=base_model_sha256,
        sites=[SyntheticSite(layer, LFM3_HIDDEN_SIZE, SYNTHETIC_RANK) for layer in SYNTHETIC_LAYERS],
        seed=seed,
    )


def compile_lfm3_synthetic(snapshot: str | Path, cache_root: str | Path, profile_dir: str | Path) -> CacheEntry:
    profile = write_lfm3_synthetic_profile(profile_dir)
    return get_or_compile(
        cache_root=cache_root,
        model="lfm3",
        base_dir=snapshot,
        graph=LFM3_FP32_GRAPH,
        profile=profile,
        external_sha256=LFM3_FP32_EXTERNAL_SHA256,
    )


def compile_lfm3_synthetic_fp16(snapshot: str | Path, cache_root: str | Path, profile_dir: str | Path) -> CacheEntry:
    """The same synthetic profile (same seed, so the same directions) against the fp16 graph (site mapping
    ``lfm3-1.7b/fp16/1``). Only ``base_model_sha256`` differs from the fp32 profile, so its SHA-256 differs too."""
    profile = write_lfm3_synthetic_profile(profile_dir, base_model_sha256=LFM3_FP16_GRAPH_SHA256)
    return get_or_compile(
        cache_root=cache_root,
        model="lfm3",
        base_dir=snapshot,
        graph=LFM3_FP16_GRAPH,
        profile=profile,
        external_sha256=LFM3_FP16_EXTERNAL_SHA256,
    )


def write_lfm3_addition_profile(directory: str | Path, *, seed: int = ADDITION_SEED):
    """A schema 2 ``directional_addition`` profile after layers 12 and 16, rank 2, scales (1, 0.5)."""
    return write_synthetic_profile(
        directory,
        profile_id=ADDITION_PROFILE_ID,
        base_model_sha256=LFM3_FP32_GRAPH_SHA256,
        sites=[SyntheticSite(layer, LFM3_HIDDEN_SIZE, ADDITION_RANK, ADDITION_SCALES) for layer in ADDITION_LAYERS],
        seed=seed,
        operator=OPERATOR_DIRECTIONAL_ADDITION,
    )


def compile_lfm3_synthetic_multi(snapshot: str | Path, cache_root: str | Path, profile_dir: str | Path) -> CacheEntry:
    """Both synthetic profiles in one graph: index 0 ``lfm3-synthetic-test``, index 1 ``lfm3-synthetic-add``."""
    profile_dir = Path(profile_dir)
    profiles = [
        write_lfm3_synthetic_profile(profile_dir / SYNTHETIC_PROFILE_ID),
        write_lfm3_addition_profile(profile_dir / ADDITION_PROFILE_ID),
    ]
    return get_or_compile_profiles(
        cache_root=cache_root,
        model="lfm3",
        base_dir=snapshot,
        graph=LFM3_FP32_GRAPH,
        profiles=profiles,
        external_sha256=LFM3_FP32_EXTERNAL_SHA256,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--snapshot", required=True, type=Path, help="LFM3-1.7B-ONNX snapshot root (c680759a)")
    parser.add_argument("--cache", required=True, type=Path, help="intervention cache root (same file system)")
    parser.add_argument(
        "--profile-dir", type=Path, help="where to write the synthetic profile (default: <cache>/profiles/...)"
    )
    parser.add_argument(
        "--multi", action="store_true", help="compile both synthetic profiles into one graph with a strength vector"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="compile the synthetic profile against the fp16 graph (onnx/model_fp16.onnx)",
    )
    args = parser.parse_args(argv)
    if args.fp16 and args.multi:
        parser.error("--fp16 and --multi cannot be combined")
    if args.fp16:
        profile_dir = args.profile_dir or args.cache / "profiles" / f"{SYNTHETIC_PROFILE_ID}-fp16"
        print(compile_lfm3_synthetic_fp16(args.snapshot, args.cache, profile_dir).directory)
        return 0
    if args.multi:
        entry = compile_lfm3_synthetic_multi(args.snapshot, args.cache, args.profile_dir or args.cache / "profiles")
        print(entry.directory)
        return 0
    profile_dir = args.profile_dir or args.cache / "profiles" / SYNTHETIC_PROFILE_ID
    entry = compile_lfm3_synthetic(args.snapshot, args.cache, profile_dir)
    print(entry.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
