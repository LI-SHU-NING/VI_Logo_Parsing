# -*- coding: utf-8 -*-
"""
MLLM / LLM 组件工具包

提供统一的大语言模型（LLM）和多模态大语言模型（MLLM）调用能力。
支持多后端（Ollama 本地 / 远程网关 / 可扩展），通过 JSON 配置文件切换。

架构:
    BaseModelClient (base.py)
        ├── LLMClient  (llm_client.py)   — 纯文本 LLM
        └── MLLMClient (mllm_client.py)  — 多模态 MLLM

    BaseBackend (backends/base_backend.py)
        ├── OllamaBackend         — 本地 Ollama（/api/chat 原生接口）
        └── RemoteGatewayBackend  — 远程 AI 网关（HMAC 认证）
    可通过 register_backend() 注册新后端

配置:
    model_config.json 定义后端列表，active_backend="auto" 时自动探测

快速使用:
    from MLLM_LLM_Component_Tools import LLMClient, MLLMClient

    # LLM（纯文本）
    llm = LLMClient()
    text = llm.chat_simple("什么是品牌标识？")

    # MLLM（多模态，支持多张图片）
    mllm = MLLMClient()
    text = mllm.chat_simple("分析这张图的品牌标识", image_paths=["logo.png"])
"""

from .llm_client import LLMClient
from .mllm_client import MLLMClient
from .config import ModelConfig
from .base import BaseModelClient
from .exceptions import (
    ModelToolError,
    NoAvailableBackendError,
    BackendNotConfiguredError,
    ModelCallError,
)
from .backends import (
    BaseBackend,
    OllamaBackend,
    RemoteGatewayBackend,
    create_backend,
    register_backend,
)

__all__ = [
    # 客户端
    "LLMClient",
    "MLLMClient",
    "BaseModelClient",
    # 配置
    "ModelConfig",
    # 异常
    "ModelToolError",
    "NoAvailableBackendError",
    "BackendNotConfiguredError",
    "ModelCallError",
    # 后端
    "BaseBackend",
    "OllamaBackend",
    "RemoteGatewayBackend",
    "create_backend",
    "register_backend",
]
