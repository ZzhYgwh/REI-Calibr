from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_scalar(text: str) -> Any:
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if "." not in text and "e" not in lowered:
            return int(text)
        return float(text)
    except ValueError:
        return text


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    out = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key.path=value, got: {item}")
        key_path, raw_value = item.split("=", 1)
        keys = key_path.split(".")
        cursor = out
        for key in keys[:-1]:
            if isinstance(cursor, list):
                cursor = cursor[int(key)]
            elif isinstance(cursor, dict):
                cursor = cursor.setdefault(key, {})
            else:
                raise ValueError(f"Cannot set nested override through non-container key: {key_path}")
        last = keys[-1]
        if isinstance(cursor, list):
            cursor[int(last)] = parse_scalar(raw_value)
        elif isinstance(cursor, dict):
            cursor[last] = parse_scalar(raw_value)
        else:
            raise ValueError(f"Cannot set override on non-container key: {key_path}")
    return out


def resolve_path(value: Any, base_dir: Path) -> Any:
    if value is None:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
