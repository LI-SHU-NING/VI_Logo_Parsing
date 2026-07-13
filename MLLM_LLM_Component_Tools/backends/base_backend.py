# -*- coding: utf-8 -*-
"""模型后端抽象基类。所有后端（Ollama、远程网关等）继承此类，实现统一的调用接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union


class BaseBackend(ABC):
    """模型后端抽象基类。

    子类需实现:
        call_llm  — 纯文本 LLM 调用
        call_mllm — 多模态 MLLM 调用（含图片）
        health_check — 快速探测后端是否可用
    """

    backend_type: str = "base"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.timeout: int = config.get("timeout", 7200)

    @abstractmethod
    def call_llm(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.5,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """调用纯文本 LLM。"""
        ...

    @abstractmethod
    def call_mllm(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        image_paths: Optional[List[str]] = None,
        temperature: float = 0.5,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """调用多模态 MLLM（支持图片输入）。"""
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """检查后端是否可用（快速探测，不发起完整推理）。"""
        ...
