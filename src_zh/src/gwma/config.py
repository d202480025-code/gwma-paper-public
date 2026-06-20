from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """加载 YAML，并解析可选的相对路径父配置。"""
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    parent = config.pop("defaults", None)
    if parent is None:
        return config

    parents = [parent] if isinstance(parent, str) else parent
    resolved: dict[str, Any] = {}
    for parent_path in parents:
        candidate = (path.parent / parent_path).resolve()
        resolved = deep_merge(resolved, load_config(candidate))
    return deep_merge(resolved, config)


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged
