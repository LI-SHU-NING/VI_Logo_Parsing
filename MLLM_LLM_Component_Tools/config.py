# -*- coding: utf-8 -*-
"""模型配置加载与管理 —— 支持 JSON 配置文件 + auto 自动探测。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .exceptions import BackendNotConfiguredError, NoAvailableBackendError

logger = logging.getLogger("MLLM_LLM_Tools.config")

_DEFAULT_CONFIG_PATHS = [
    Path(__file__).resolve().parent / "model_config.json",
    Path.cwd() / "model_config.json",
]


class ModelConfig:
    """模型配置管理器。从 JSON 文件加载配置，支持 auto 自动探测可用后端。

    active_backend 值:
        "auto"          — 按 priority 顺序探测，选第一个通过健康检查的后端
        "ollama"        — 强制使用 Ollama 后端
        "remote_gateway"— 强制使用远程网关后端
        其他自定义名称   — 强制使用对应后端
    """

    def __init__(self, config: Dict[str, Any]):
        self._raw_config = config
        self._active_backend: str = config.get("active_backend", "auto")
        self._backends: Dict[str, Dict[str, Any]] = config.get("backends", {})

        if self._active_backend == "auto":
            self._active_backend = self._auto_detect()

    @property
    def active_backend(self) -> str:
        return self._active_backend

    @property
    def active_backend_config(self) -> Dict[str, Any]:
        cfg = self._backends.get(self._active_backend)
        if cfg is None:
            raise BackendNotConfiguredError(
                f"活跃后端 '{self._active_backend}' 不在配置中"
            )
        return cfg

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "ModelConfig":
        """加载配置文件。优先级：显式路径 > 模块目录 > 当前工作目录。"""
        path = cls._resolve_config_path(config_path)
        if path is None:
            logger.warning("未找到 model_config.json，使用内置默认配置（Ollama 本地）")
            return cls(cls._default_config())

        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        logger.info("已加载模型配置: %s", path)
        return cls(config)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ModelConfig":
        """直接从字典创建配置（用于编程方式注入）。"""
        return cls(config)

    def _auto_detect(self) -> str:
        """按 priority 顺序探测可用后端，返回第一个通过健康检查的后端名。"""
        from .backends import create_backend

        sorted_backends = sorted(
            self._backends.items(),
            key=lambda item: item[1].get("priority", 999),
        )

        for name, backend_config in sorted_backends:
            if not backend_config.get("enabled", True):
                continue
            try:
                backend = create_backend(name, backend_config)
                if backend.health_check():
                    logger.info("[auto] 后端 '%s' 健康检查通过，已选中", name)
                    return name
                logger.debug("[auto] 后端 '%s' 健康检查未通过", name)
            except Exception as e:
                logger.debug("[auto] 后端 '%s' 探测失败: %s", name, e)

        raise NoAvailableBackendError(
            f"自动探测未找到可用后端，已尝试: {list(self._backends.keys())}"
        )

    @staticmethod
    def _resolve_config_path(config_path: Optional[str]) -> Optional[Path]:
        if config_path:
            p = Path(config_path)
            if p.exists():
                return p
            raise FileNotFoundError(f"指定的配置文件不存在: {config_path}")

        for candidate in _DEFAULT_CONFIG_PATHS:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """内置默认配置（Ollama 本地）。"""
        return {
            "active_backend": "ollama",
            "backends": {
                "ollama": {
                    "type": "ollama",
                    "enabled": True,
                    "priority": 1,
                    "base_url": "http://127.0.0.1:11434",
                    "llm_model": "deepseek-r1:32b",
                    "mllm_model": "qwen3-vl:8b",
                    "timeout": 7200,
                }
            },
        }
