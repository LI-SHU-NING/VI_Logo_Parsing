# -*- coding: utf-8 -*-
"""LLM 客户端 —— 纯文本大语言模型调用工具。"""

from __future__ import annotations

import logging
from typing import Any, Dict, Generator, List, Optional, Union

from .base import BaseModelClient

logger = logging.getLogger("MLLM_LLM_Tools.llm")


class LLMClient(BaseModelClient):
    """纯文本 LLM 客户端。

    用法:
        client = LLMClient()                          # 自动加载配置 + auto 探测后端
        result = client.chat(prompt="你好")            # 完整响应（dict）
        text = client.chat_simple(prompt="你好")       # 直接返回文本字符串
    """

    def chat(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.5,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """调用 LLM 进行文本对话。

        参数:
            prompt:      简单文本提示（与 messages 二选一）
            messages:    OpenAI 格式消息列表（高级用法，可带 system/assistant 历史轮次）
            temperature: 采样温度
            tools:       工具定义列表
            stream:      是否流式输出（True 时返回 generator，逐块 yield 文本）

        返回:
            stream=False: dict（OpenAI 兼容格式，含 choices/message/content）
            stream=True:  Generator[str, None, None]（逐块 yield 文本片段）
        """
        return self._backend.call_llm(
            prompt=prompt,
            messages=messages,
            temperature=temperature,
            tools=tools,
            stream=stream,
        )

    def chat_simple(self, prompt: str, temperature: float = 0.5) -> str:
        """简化调用 —— 非流式，直接返回文本内容字符串。"""
        result = self.chat(prompt=prompt, temperature=temperature)
        if isinstance(result, dict):
            choices = result.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return result.get("content", str(result))
        return str(result)
