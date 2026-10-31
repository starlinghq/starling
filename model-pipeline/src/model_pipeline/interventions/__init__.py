"""ONNX Runtime interventions: compile a directional-projection profile into a derived ONNX graph (experimental).

Requires the ``interventions`` extra (numpy, onnx, onnxruntime). Specification: ``docs/interventions/``.
"""

from .compiler import (
    COMPILER_VERSION,
    STRENGTH_INPUT_NAME,
    CompileResult,
    InterventionCompileError,
    compile_intervention,
)
from .contract import (
    InterventionContractError,
    InterventionManifest,
    InterventionMismatchError,
    InterventionProfile,
    InterventionProfileFile,
    load_manifest,
    load_profile,
)
from .external_data import ExternalDataLinkError
from .injection_sites import SITE_MAPPINGS, MappedSite, SiteMapping, UnsupportedGraphError, lookup_site_mapping
from .validate import DerivedGraphError

__all__ = [
    "COMPILER_VERSION",
    "SITE_MAPPINGS",
    "STRENGTH_INPUT_NAME",
    "CompileResult",
    "DerivedGraphError",
    "ExternalDataLinkError",
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
