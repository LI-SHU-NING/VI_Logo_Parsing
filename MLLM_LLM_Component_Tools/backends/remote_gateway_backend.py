# -*- coding: utf-8 -*-
"""远程 AI 网关后端 —— 通过 HMAC 认证调用远程模型推理服务。"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import requests
import urllib3

from .base_backend import BaseBackend

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logger = logging.getLogger("MLLM_LLM_Tools.remote_gateway")


def _generate_hmac_auth(username: str, secret: str) -> tuple[str, str]:
    """生成 HMAC 认证头和日期。"""
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%a, %d %b %Y %H:%M:%S GMT")
    string_to_sign = f"x-date: {date_str}"
    hmac_code = hmac.new(secret.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha256).digest()
    signature = base64.b64encode(hmac_code).decode("utf-8")
    auth_header = (
        f'hmac username="{username}", '
        f'algorithm="hmac-sha256", '
        f'headers="x-date", '
        f'signature="{signature}"'
    )
    return auth_header, date_str


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class RemoteGatewayBackend(BaseBackend):
    """远程 AI 网关后端（HMAC 认证 + componentCode）。

    图片格式: OpenAI 兼容的 image_url content（base64 编码，不带 data: 前缀）
    """

    backend_type = "remote_gateway"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url: str = config["base_url"]
        self.verify_ssl: bool = config.get("verify_ssl", False)
        self.llm_cfg: Dict[str, Any] = config.get("llm", {})
        self.mllm_cfg: Dict[str, Any] = config.get("mllm", {})

    def call_llm(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.5,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]

        payload: Dict[str, Any] = {
            "componentCode": self.llm_cfg.get("component_code", ""),
            "model": self.llm_cfg.get("model", ""),
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools

        return self._request(payload, self.llm_cfg, stream)

    def call_mllm(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        image_paths: Optional[List[str]] = None,
        temperature: float = 0.5,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        image_contents = [
            {"type": "image_url", "image_url": _encode_image(p)}
            for p in (image_paths or []) if Path(p).exists()
        ]

        if messages is None:
            content = [{"type": "text", "text": prompt or ""}, *image_contents]
            messages = [{"role": "user", "content": content}]
        elif image_contents:
            last = messages[-1]
            if isinstance(last.get("content"), str):
                last["content"] = [{"type": "text", "text": last["content"]}, *image_contents]
            elif isinstance(last.get("content"), list):
                last["content"].extend(image_contents)

        payload: Dict[str, Any] = {
            "componentCode": self.mllm_cfg.get("component_code", ""),
            "model": self.mllm_cfg.get("model", ""),
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools

        return self._request(payload, self.mllm_cfg, stream)

    def health_check(self) -> bool:
        """远程网关健康检查：发起极简请求，HTTP < 500 即视为可达。"""
        try:
            auth_header, date_str = _generate_hmac_auth(
                self.mllm_cfg.get("username", ""), self.mllm_cfg.get("secret", "")
            )
            headers = {
                "Content-Type": "application/json",
                "x-date": date_str,
                "Authorization": auth_header,
            }
            resp = requests.post(
                self.base_url,
                headers=headers,
                json={
                    "componentCode": self.mllm_cfg.get("component_code", ""),
                    "model": self.mllm_cfg.get("model", ""),
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                },
                verify=self.verify_ssl,
                timeout=10,
            )
            return resp.status_code < 500
        except Exception:
            return False

    def _request(
        self,
        payload: Dict[str, Any],
        model_cfg: Dict[str, Any],
        stream: bool,
    ) -> Union[Dict[str, Any], Generator[str, None, None]]:
        auth_header, date_str = _generate_hmac_auth(
            model_cfg.get("username", ""), model_cfg.get("secret", "")
        )
        headers = {
            "Content-Type": "application/json",
            "x-date": date_str,
            "Authorization": auth_header,
            "User-Agent": "PostmanRuntime/7.32.3",
            "Accept": "*/*",
        }

        logger.info(
            "[RemoteGateway] POST %s (model=%s, stream=%s)",
            self.base_url, model_cfg.get("model", ""), stream,
        )

        resp = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            verify=self.verify_ssl,
            timeout=self.timeout,
            stream=stream,
        )
        resp.raise_for_status()

        if stream:
            return self._stream_response(resp)

        try:
            return resp.json()
        except json.JSONDecodeError:
            return self._parse_stream_text(resp.text)

    @staticmethod
    def _stream_response(resp: requests.Response) -> Generator[str, None, None]:
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            yield data_str

    @staticmethod
    def _parse_stream_text(raw_text: str) -> Dict[str, Any]:
        """尝试从流式格式的纯文本中提取内容（兼容远程 API 返回格式异常）。"""
        full_content = ""
        for line in raw_text.split("\n"):
            line_str = line.strip()
            if not line_str or not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if "content" in delta:
                        full_content += delta["content"]
            except json.JSONDecodeError:
                continue
        if full_content:
            return {
                "choices": [{"message": {"role": "assistant", "content": full_content}}],
            }
        return {"error": "Failed to parse response", "raw_response": raw_text[:500]}
