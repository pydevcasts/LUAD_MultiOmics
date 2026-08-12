"""Small IO helpers for saving/loading artifacts and ensuring directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import yaml

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """
    Create directory if it does not exist and return it as Path.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_json(path: PathLike, data: Any, indent: int = 2) -> Path:
    """
    Save Python object as JSON with UTF-8 encoding.
    """
    file_path = Path(path)
    ensure_dir(file_path.parent)

    text = json.dumps(data, ensure_ascii=False, indent=indent)
    file_path.write_text(text, encoding="utf-8")

    return file_path


def load_json(path: PathLike) -> Any:
    """
    Load JSON file.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    return json.loads(file_path.read_text(encoding="utf-8"))


def save_yaml(path: PathLike, data: Any) -> Path:
    """
    Save Python object as YAML with UTF-8 encoding.
    """
    file_path = Path(path)
    ensure_dir(file_path.parent)

    with file_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)

    return file_path


def load_yaml(path: PathLike) -> Any:
    """
    Load YAML file.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_text(path: PathLike, text: str) -> Path:
    """
    Save plain text file with UTF-8 encoding.
    """
    file_path = Path(path)
    ensure_dir(file_path.parent)

    file_path.write_text(text, encoding="utf-8")

    return file_path


def load_text(path: PathLike) -> str:
    """
    Load plain text file.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    return file_path.read_text(encoding="utf-8")