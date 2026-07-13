# -*- coding: utf-8 -*-
"""MLLM 客户端 —— 多模态大语言模型调用工具（支持图片输入）。"""

from __future__ import annotations

import logging
from typing import Any, Dict, Generator, List, Optional, Union

from .base import BaseModelClient

logger = logging.getLogger("MLLM_LLM_Tools.mllm")


class MLLMClient(BaseModelClient):
    """多模态 MLLM 客户端，支持传入多张图片。

    混合接口:
        简单场景: 传 prompt + image_paths，内部自动编码图片
        高级场景: 传 messages（含图片 content），image_paths 可同时传入
                  （图片会追加到最后一条 user 消息）

    用法:
        client = MLLMClient()

        # 简单用法：prompt + image_paths
        result = client.chat(
            prompt="这张图里有什么品牌标识？",
            image_paths=["logo1.png", "logo2.png"],
        )

        # 高级用法：直接传 OpenAI 格式 messages
        result = client.chat(messages=[
            {"role": "system", "content": "你是品牌标识分析专家"},
            {"role": "user", "content": [
                {"type": "text", "text": "分析这张图"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            ]},
        ])

        # 简化调用：直接返回文本
        text = client.chat_simple("分析这张图", image_paths=["logo.png"])
    """

    def chat(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        image_paths: Optional[List[str]] = None,
        temperature: float = 0.5,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        """调用 MLLM 进行多模态对话。

        参数:
            prompt:      文本提示（简单用法，与 messages 二选一）
            messages:    OpenAI 格式消息列表（高级用法）
            image_paths: 本地图片路径列表（简单用法时自动编码注入）
            temperature: 采样温度
            tools:       工具定义列表
            stream:      是否流式输出

        返回:
            stream=False: dict（OpenAI 兼容格式）
            stream=True:  Generator[str, None, None]
        """
        return self._backend.call_mllm(
            prompt=prompt,
            messages=messages,
            image_paths=image_paths,
            temperature=temperature,
            tools=tools,
            stream=stream,
        )

    def chat_simple(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        temperature: float = 0.5,
    ) -> str:
        """简化调用 —— 非流式，直接返回文本内容字符串。"""
        result = self.chat(
            prompt=prompt, image_paths=image_paths, temperature=temperature
        )
        if isinstance(result, dict):
            choices = result.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return result.get("content", str(result))
        return str(result)
