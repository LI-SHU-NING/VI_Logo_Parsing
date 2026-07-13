# -*- coding: utf-8 -*-
"""模型客户端基类 —— 公共初始化逻辑，子类实现具体的 chat 方法。"""

from __future__ import annotations

import logging
from typing import Optional

from .config import ModelConfig
from .backends import create_backend
from .backends.base_backend import BaseBackend

logger = logging.getLogger("MLLM_LLM_Tools")


class BaseModelClient:
    """模型客户端基类。负责配置加载、后端创建、健康检查等公共逻辑。

    子类（LLMClient / MLLMClient）继承此类，添加具体的 chat 方法。
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[ModelConfig] = None,
    ):
        if config is not None:
            self._config = config
        else:
            self._config = ModelConfig.load(config_path)

        backend_name = self._config.active_backend
        backend_config = self._config.active_backend_config
        self._backend: BaseBackend = create_backend(backend_name, backend_config)
        logger.info(
            "%s 已初始化，后端: %s (%s)",
            self.__class__.__name__, backend_name, self._backend.backend_type,
        )

    @property
    def backend_name(self) -> str:
        """当前使用的后端名称。"""
        return self._config.active_backend

    @property
    def backend(self) -> BaseBackend:
        """底层后端实例（高级用法，可直接调用后端方法）。"""
        return self._backend

    def health_check(self) -> bool:
        """检查当前后端是否可用。"""
        return self._backend.health_check()
