"""Configuration loading, validation, and normalization utilities."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

PathLike = Union[str, Path]


class ConfigError(RuntimeError):
    """Raised when configuration is missing, invalid, or inconsistent."""


def load_yaml(path: PathLike) -> Dict[str, Any]:
    """
    Load a YAML file and ensure the top-level object is a dictionary.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise ConfigError(f"Config file not found: {file_path}")

    try:
        data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in file: {file_path}\n{exc}") from exc

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ConfigError(
            f"Top-level YAML structure must be a mapping/dictionary: {file_path}"
        )

    return data


def deep_merge(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Recursively merge override config into base config.

    Values in `override` replace values in `base`.
    Nested dictionaries are merged recursively.
    """
    merged = deepcopy(base)

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged


def _require_section(config: Dict[str, Any], section_name: str) -> None:
    """
    Ensure a required top-level config section exists and is a dictionary.
    """
    if section_name not in config or not isinstance(config[section_name], dict):
        raise ConfigError(f"Missing or invalid config section: '{section_name}'")


def validate_base_config(config: Dict[str, Any]) -> None:
    """
    Validate minimal required sections for Phase 0 and later phases.
    """
    required_sections = [
        "project",
        "paths",
        "audit",
        "task",
    ]

    for section in required_sections:
        _require_section(config, section)

    if "raw_dir" not in config["paths"]:
        raise ConfigError("Missing required config value: paths.raw_dir")


def normalize_dataset_files(raw_datasets: Any) -> List[Dict[str, Any]]:
    """
    Normalize datasets.yaml into a flat list of dataset dictionaries.

    Supports both structures:

    1. List structure:
        files:
          - key: clinical_patient
            filename: data_clinical_patient.txt

    2. Nested structure:
        tcga:
          clinical:
            path: datasets/data_clinical_patient.txt
        external:
          gse131907:
            path: datasets/GSE...txt
    """
    if raw_datasets is None:
        return []

    if isinstance(raw_datasets, list):
        return [dict(item) for item in raw_datasets if isinstance(item, dict)]

    if not isinstance(raw_datasets, dict):
        raise ConfigError(
            "datasets.yaml must contain a mapping/dictionary or a list of datasets."
        )

    # Preferred explicit structure: files: [...]
    if "files" in raw_datasets and isinstance(raw_datasets["files"], list):
        return [
            dict(item)
            for item in raw_datasets["files"]
            if isinstance(item, dict)
        ]

    files: List[Dict[str, Any]] = []

    for top_key, top_value in raw_datasets.items():
        if not isinstance(top_value, dict):
            continue

        # Case where top_value itself looks like one dataset entry.
        if any(key in top_value for key in ("path", "filename", "modality")):
            item = dict(top_value)
            item.setdefault("key", top_key)
            files.append(item)
            continue

        # Case where top_value contains multiple dataset entries.
        for sub_key, sub_value in top_value.items():
            if not isinstance(sub_value, dict):
                continue

            item = dict(sub_value)
            item.setdefault("key", sub_key)
            files.append(item)

    return files


def _resolve_dataset_path(
    item: Dict[str, Any],
    raw_dir: Path,
    project_root: Path,
) -> Path:
    """
    Resolve dataset path from either `filename` or `path`.

    Rules:
    - If item has `filename`, it is interpreted relative to raw_dir,
      unless it is absolute.
    - If item has `path`, it is interpreted as project-relative if relative,
      or absolute if absolute.
    - For safety, several candidates are checked and the first existing one is returned.
    - If none exists, the first candidate is returned so later audit can report it.
    """
    candidates: List[Path] = []

    filename = item.get("filename")
    if filename:
        filename_path = Path(str(filename))

        if filename_path.is_absolute():
            candidates.append(filename_path)
        else:
            candidates.append(raw_dir / filename_path)

    path_value = item.get("path")
    if path_value:
        path_obj = Path(str(path_value))

        if path_obj.is_absolute():
            candidates.append(path_obj)
        else:
            candidates.append(project_root / path_obj)
            candidates.append(raw_dir / path_obj)

    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue

    if candidates:
        return candidates[0]

    return Path("")


def resolve_dataset_files(
    files: List[Dict[str, Any]],
    raw_dir: Path,
    project_root: Path,
) -> List[Dict[str, Any]]:
    """
    Add resolved_path to each dataset entry.
    """
    resolved_files: List[Dict[str, Any]] = []

    for file_config in files:
        item = dict(file_config)
        item.setdefault("audit", True)

        resolved_path = _resolve_dataset_path(
            item=item,
            raw_dir=raw_dir,
            project_root=project_root,
        )

        item["resolved_path"] = str(resolved_path)
        resolved_files.append(item)

    return resolved_files


def load_config(
    config_path: PathLike,
    datasets_path: Optional[PathLike] = None,
    models_path: Optional[PathLike] = None,
    experiment_path: Optional[PathLike] = None,
    project_root: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """
    Load main config and optionally merge datasets/models/experiment configs.

    The returned config contains:
    - original config sections
    - datasets_files: normalized and resolved dataset list
    - models_search_space: loaded models.yaml if available
    - runtime: resolved paths used for loading
    """
    config_path_obj = Path(config_path).resolve()

    if project_root is None:
        if len(config_path_obj.parents) >= 2:
            project_root_obj = config_path_obj.parents[1]
        else:
            project_root_obj = Path.cwd()
    else:
        project_root_obj = Path(project_root).resolve()

    config = load_yaml(config_path_obj)

    if experiment_path is not None:
        experiment_config = load_yaml(experiment_path)
        config = deep_merge(config, experiment_config)

    validate_base_config(config)

    config_dir = config_path_obj.parent

    # Load datasets.yaml if available.
    if datasets_path is None:
        default_datasets_path = config_dir / "datasets.yaml"
        datasets_path_obj = (
            default_datasets_path if default_datasets_path.exists() else None
        )
    else:
        datasets_path_obj = Path(datasets_path)

    if datasets_path_obj is not None:
        datasets_raw = load_yaml(datasets_path_obj)
        config["datasets_raw"] = datasets_raw

        dataset_files = normalize_dataset_files(datasets_raw)

        raw_dir_value = config["paths"]["raw_dir"]
        raw_dir_obj = Path(str(raw_dir_value))

        if not raw_dir_obj.is_absolute():
            raw_dir_obj = project_root_obj / raw_dir_obj

        config["datasets_files"] = resolve_dataset_files(
            files=dataset_files,
            raw_dir=raw_dir_obj,
            project_root=project_root_obj,
        )
    else:
        config["datasets_raw"] = {}
        config["datasets_files"] = []

    # Load models.yaml if available.
    if models_path is None:
        default_models_path = config_dir / "models.yaml"
        models_path_obj = (
            default_models_path if default_models_path.exists() else None
        )
    else:
        models_path_obj = Path(models_path)

    if models_path_obj is not None:
        config["models_search_space"] = load_yaml(models_path_obj)
    else:
        config["models_search_space"] = {}

    config["runtime"] = {
        "config_path": str(config_path_obj),
        "project_root": str(project_root_obj),
        "datasets_path": str(datasets_path_obj) if datasets_path_obj else None,
        "models_path": str(models_path_obj) if models_path_obj else None,
        "experiment_path": str(experiment_path) if experiment_path else None,
    }

    return config