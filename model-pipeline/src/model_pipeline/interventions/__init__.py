"""ONNX Runtime interventions: compile intervention profiles into a derived ONNX graph (experimental).

Requires the ``interventions`` extra (numpy, onnx, onnxruntime). Specification: ``docs/interventions/``.
"""

from .cache import (
    CacheEntry,
    InterventionCacheError,
    cache_key_text,
    entry_directory,
    evict,
    get_or_compile,
    get_or_compile_profiles,
    linked_base_artifacts,
    manifest_cache_key,
    multi_cache_key_text,
    open_entry,
    profile_path_in_entry,
)
from .compiler import (
    COMPILER_VERSION,
    STRENGTH_INPUT_NAME,
    CompileResult,
    InterventionCompileError,
    compile_intervention,
    compile_interventions,
)
from .contract import (
    FileIdentity,
    InterventionContractError,
    InterventionManifest,
    InterventionMismatchError,
    InterventionProfile,
    InterventionProfileFile,
    ManifestProfile,
    MultiProfileManifest,
    MultiSiteBinding,
    load_manifest,
    load_profile,
)
from .external_data import ExternalDataLinkError
from .injection_sites import SITE_MAPPINGS, MappedSite, SiteMapping, UnsupportedGraphError, lookup_site_mapping
from .validate import DerivedGraphError

__all__ = [
    "CacheEntry",
    "InterventionCacheError",
    "cache_key_text",
    "entry_directory",
    "evict",
    "get_or_compile",
    "get_or_compile_profiles",
    "multi_cache_key_text",
    "profile_path_in_entry",
    "compile_interventions",
    "ManifestProfile",
    "MultiProfileManifest",
    "MultiSiteBinding",
    "linked_base_artifacts",
    "manifest_cache_key",
    "open_entry",
    "COMPILER_VERSION",
    "SITE_MAPPINGS",
    "STRENGTH_INPUT_NAME",
    "CompileResult",
    "DerivedGraphError",
    "ExternalDataLinkError",
    "FileIdentity",
    "InterventionCompileError",
    "InterventionContractError",
    "InterventionManifest",
    "InterventionMismatchError",
    "InterventionProfile",
    "InterventionProfileFile",
    "MappedSite",
    "SiteMapping",
    "UnsupportedGraphError",
    "compile_intervention",
    "load_manifest",
    "load_profile",
    "lookup_site_mapping",
]
