# -*- coding: utf-8 -*-
"""Ollama 后端 —— 通过原生 /api/chat 接口调用本地模型。"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import requests

from .base_backend import BaseBackend

logger = logging.getLogger("MLLM_LLM_Tools.ollama")


def _encode_image(path: str) -> str:
    """读取图片文件并返回 base64 编码字符串。"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _extract_text_from_content(content: Any) -> str:
    """从 message content 中提取纯文本（兼容 str 和 list 格式）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content)


class OllamaBackend(BaseBackend):
    """Ollama 本地模型后端，使用原生 /api/chat 接口。

    图片格式: Ollama 原生方式，在 message 中添加 "images": [base64_str, ...]
    """

    backend_type = "ollama"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url: str = config["base_url"].rstrip("/")
        self.llm_model: str = config.get("llm_model", "")
        self.mllm_model: str = config.get("mllm_model", "")

    def call_llm(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.5,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        built_messages = self._build_text_messages(prompt, messages)
        payload: Dict[str, Any] = {
            "model": self.llm_model,
            "messages": built_messages,
            "stream": stream,
            "options": {"temperature": temperature},
        }
        if tools:
            payload["tools"] = tools
        return self._request(payload, stream, self.llm_model)

    def call_mllm(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        image_paths: Optional[List[str]] = None,
        temperature: float = 0.5,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        built_messages = self._build_mllm_messages(prompt, messages, image_paths)
        payload: Dict[str, Any] = {
            "model": self.mllm_model,
            "messages": built_messages,
            "stream": stream,
            "options": {"temperature": temperature},
        }
        if tools:
            payload["tools"] = tools
        return self._request(payload, stream, self.mllm_model)

    def health_check(self) -> bool:
        """GET /api/tags，200 即视为可用。"""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def _request(
        self,
        payload: Dict[str, Any],
        stream: bool,
        model: str,
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        url = f"{self.base_url}/api/chat"
        headers = {"Content-Type": "application/json"}
        logger.info("[Ollama] POST %s (model=%s, stream=%s)", url, model, stream)

        if stream:
            return self._stream_request(url, payload, headers)

        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return self._normalize_response(resp.json(), model)

    def _stream_request(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
    ) -> Generator[str, None, None]:
        with requests.post(url, json=payload, headers=headers, stream=True, timeout=self.timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

    @staticmethod
    def _normalize_response(data: Dict[str, Any], model: str) -> Dict[str, Any]:
        """将 Ollama 响应统一为 OpenAI 兼容格式。"""
        message = data.get("message", {})
        return {
            "choices": [
                {
                    "message": {
                        "role": message.get("role", "assistant"),
                        "content": message.get("content", ""),
                    },
                    "finish_reason": "stop" if data.get("done", True) else None,
                }
            ],
            "model": model,
            "usage": {
                "prompt_eval_count": data.get("prompt_eval_count", 0),
                "eval_count": data.get("eval_count", 0),
            },
        }

    @staticmethod
    def _build_text_messages(
        prompt: Optional[str],
        messages: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if messages is not None:
            return [
                {"role": m.get("role", "user"), "content": _extract_text_from_content(m.get("content", ""))}
                for m in messages
            ]
        return [{"role": "user", "content": prompt or ""}]

    @staticmethod
    def _build_mllm_messages(
        prompt: Optional[str],
        messages: Optional[List[Dict[str, Any]]],
        image_paths: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        images = [_encode_image(p) for p in (image_paths or []) if Path(p).exists()]

        if messages is None:
            msg: Dict[str, Any] = {"role": "user", "content": prompt or ""}
            if images:
                msg["images"] = images
            return [msg]

        result = []
        for m in messages:
            msg = {
                "role": m.get("role", "user"),
                "content": _extract_text_from_content(m.get("content", "")),
            }
            result.append(msg)
        if images and result:
            result[-1]["images"] = images
        return result
