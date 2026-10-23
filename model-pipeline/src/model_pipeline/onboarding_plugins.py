"""Explicit registry of trusted, repository-owned onboarding plugins."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

from model_pipeline.onboarding_runner import PipelineError

PLUGINS = {
    "e5-onnx-cpu-v1": "model_pipeline.onboarding_e5",
    "bert-onnx-cpu-v1": "model_pipeline.onboarding_bert",
    "qwen-genai-cpu-v1": "model_pipeline.onboarding_generation",
    "kev-onnx-cpu-v1": "model_pipeline.onboarding_kev",
    "whisper-onnx-cpu-v1": "model_pipeline.onboarding_whisper",
}


def uses_shared_distribution(plugin) -> bool:
    """Whether the optional distribution stage comes from onboarding_distribution rather than the plugin."""
    value = plugin.SHARED_DISTRIBUTION
    if type(value) is not bool:
        raise PipelineError("onboarding plugin must declare SHARED_DISTRIBUTION as a bool")
    return value


def get_plugin(recipe: dict):
    try:
        module = PLUGINS[recipe["plugin"]]
    except (KeyError, TypeError) as exc:
        raise PipelineError("unsupported onboarding plugin") from exc
    return importlib.import_module(module)


def build_stages(recipe: dict, *, recipe_path: Path | None = None, plugin=None, **options):
    """Compose optional distribution without changing expensive plugin identities."""
    plugin = plugin or get_plugin(recipe)
    if "base_snapshot" in options and "base_snapshot" not in inspect.signature(plugin.build_stages).parameters:
        raise PipelineError("this onboarding plugin takes no base snapshot")
    candidate_distribution = options.get("distribution", False) and uses_shared_distribution(plugin)
    stages = plugin.build_stages(recipe, **{**options, "distribution": False} if candidate_distribution else options)
    if candidate_distribution:
        from model_pipeline.onboarding_distribution import append_distribution

        distribution_recipe = dict(recipe)
        if recipe_path is not None:
            distribution_recipe["recipe_path"] = (
                recipe_path.resolve().relative_to(options["repo_root"].resolve()).as_posix()
            )
        stages = append_distribution(
            stages,
            recipe=distribution_recipe,
            **{key: options[key] for key in ("repo_root", "snapshot", "threads", "timeout")},
        )
    return stages
