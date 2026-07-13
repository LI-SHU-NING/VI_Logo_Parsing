# -*- coding: utf-8 -*-
"""后端工厂 —— 根据配置创建对应的后端实例。"""

from __future__ import annotations

from typing import Any, Dict, Type

from .base_backend import BaseBackend
from .ollama_backend import OllamaBackend
from .remote_gateway_backend import RemoteGatewayBackend

_BACKEND_REGISTRY: Dict[str, Type[BaseBackend]] = {
    "ollama": OllamaBackend,
    "remote_gateway": RemoteGatewayBackend,
}


def register_backend(backend_type: str, backend_class: Type[BaseBackend]) -> None:
    """注册自定义后端类型（供外部扩展新的模型平台）。"""
    _BACKEND_REGISTRY[backend_type] = backend_class


def create_backend(backend_name: str, backend_config: Dict[str, Any]) -> BaseBackend:
    """根据配置创建后端实例。

    参数:
        backend_name:    后端名称（用于日志/报错）
        backend_config:  后端配置字典，必须包含 "type" 字段

    返回:
        BaseBackend 实例
    """
    backend_type = backend_config.get("type", backend_name)
    backend_class = _BACKEND_REGISTRY.get(backend_type)
    if backend_class is None:
        raise ValueError(
            f"未知的后端类型: {backend_type}，已注册: {list(_BACKEND_REGISTRY.keys())}"
        )
    return backend_class(backend_config)


__all__ = [
    "BaseBackend",
    "OllamaBackend",
    "RemoteGatewayBackend",
    "create_backend",
    "register_backend",
]
