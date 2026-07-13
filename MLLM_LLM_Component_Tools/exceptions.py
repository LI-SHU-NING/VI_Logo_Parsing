# -*- coding: utf-8 -*-
"""MLLM/LLM 组件工具自定义异常。"""


class ModelToolError(Exception):
    """所有模型工具异常的基类。"""


class NoAvailableBackendError(ModelToolError):
    """自动探测时未找到任何可用后端。"""


class BackendNotConfiguredError(ModelToolError):
    """后端配置缺失或无效。"""


class ModelCallError(ModelToolError):
    """模型调用失败（网络、认证、服务端错误等）。"""
