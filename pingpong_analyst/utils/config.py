"""配置加载工具"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import yaml
from loguru import logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value: str | Path | None) -> Path | None:
    """Resolve a configured asset path against the repository root.

    The server convention stores assets under ``models/`` while the local
    checkout historically kept the YOLO weight at the repository root.  Keep
    both layouts valid and return the canonical configured path when the asset
    is not present yet (for example, before downloading a server engine).
    """
    if value is None or str(value).strip() == "":
        return None

    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    candidates = [PROJECT_ROOT / path]
    if path.parts and path.parts[0] == "models":
        candidates.append(PROJECT_ROOT / path.name)
    else:
        candidates.append(PROJECT_ROOT / "models" / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@dataclass
class Config:
    """全局配置"""
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path = None) -> "Config":
        if path is None:
            path = Path(__file__).parent.parent.parent / "config.yaml"
        path = Path(path)

        if not path.exists():
            logger.warning(f"配置文件不存在: {path}, 使用默认配置")
            return cls(raw={})

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        logger.info(f"配置加载完成: {path}")
        return cls(raw=data)

    def get(self, *keys: str, default: Any = None) -> Any:
        """嵌套获取配置: config.get("device", "mode", default="auto")"""
        node = self.raw
        for key in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(key, default)
            if node is None:
                return default
        return node

    def resolve_path(self, value: str | Path | None) -> Path | None:
        """Resolve a file path using the same rules as model loaders."""
        return resolve_project_path(value)

    @property
    def device_mode(self) -> str:
        return self.get("device", "mode", default="auto")

    @property
    def gpu_memory_fraction(self) -> float:
        return self.get("device", "gpu_memory_fraction", default=0.8)


# 全局单例
_config: Config | None = None


def get_config(path: str | Path = None) -> Config:
    global _config
    if _config is None or path is not None:
        _config = Config.load(path)
    return _config
