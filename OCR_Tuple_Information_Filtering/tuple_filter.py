# -*- coding: utf-8 -*-
"""
OCR 元组信息过滤工具

根据白名单关键字库，对 OCR 输出的文字元组进行正向过滤，
只保留包含品牌相关关键字的元组，过滤掉无关噪声。

过滤规则:
    1. 空文本过滤：文本为空或仅含空白 → 过滤
    2. 最小长度过滤：文本长度 < min_text_length → 过滤
    3. 白名单关键字匹配：文本中必须包含至少一个白名单关键字才保留
       英文关键字去除空格后匹配（兼容 OCR 无空格输出）
"""

import json
from pathlib import Path
from typing import Optional

# 白名单默认路径
_DEFAULT_WHITELIST = Path(__file__).resolve().parent.parent / "Brand_Resource_Libraries" / "OCR_Filter_Whitelist.json"


class TupleFilter:
    """OCR 元组信息过滤器"""

    def __init__(self, whitelist_path: Optional[str] = None):
        path = Path(whitelist_path) if whitelist_path else _DEFAULT_WHITELIST
        with open(path, "r", encoding="utf-8") as f:
            self._data = json.load(f)

        # 加载过滤规则
        rules = self._data.get("filter_rules", {})
        self._min_length = rules.get("min_text_length", 2)
        self._filter_empty = rules.get("empty_text_filter", True)

        # 构建白名单关键字列表（全部大写用于英文匹配）
        self._keywords = []
        whitelist = self._data.get("whitelist", {})
        for category in whitelist.values():
            for kw in category.get("keywords", []):
                self._keywords.append(kw)

        # 预处理：英文关键字去空格版本
        self._keywords_no_space = [kw.upper().replace(" ", "") for kw in self._keywords]

    def filter(self, text_tuples: list[tuple]) -> list[tuple]:
        """
        对 OCR 输出的文字元组进行过滤，只保留包含白名单关键字的元组。

        参数:
            text_tuples: [(text, poly, orientation), ...] 格式的元组列表

        返回:
            list[tuple]: 过滤后的元组列表
        """
        result = []
        for tup in text_tuples:
            if self._should_keep(tup):
                result.append(tup)
        return result

    def _should_keep(self, tup: tuple) -> bool:
        """判断一个元组是否应该保留"""
        text = tup[0] if len(tup) > 0 else ""

        # 规则1：空文本过滤
        if self._filter_empty and not text.strip():
            return False

        # 规则2：最小长度过滤
        if len(text.strip()) < self._min_length:
            return False

        # 规则3：白名单关键字匹配
        return self._match_whitelist(text)

    def _match_whitelist(self, text: str) -> bool:
        """
        检查文本是否包含白名单中的至少一个关键字。
        英文关键字用大写+去空格匹配，中文关键字原文匹配。
        """
        text_upper = text.upper()
        text_no_space = text_upper.replace(" ", "")

        for i, kw in enumerate(self._keywords):
            # 中文关键字直接匹配
            if kw in text:
                return True
            # 英文关键字大写匹配
            if kw.upper() in text_upper:
                return True
            # 去空格后匹配（OCR 可能丢失空格）
            if self._keywords_no_space[i] in text_no_space:
                return True

        return False


def filter_tuples(text_tuples: list[tuple], whitelist_path: Optional[str] = None) -> list[tuple]:
    """
    便捷函数：对 OCR 输出的文字元组进行过滤。

    参数:
        text_tuples: [(text, poly, orientation), ...] 格式的元组列表
        whitelist_path: 白名单 JSON 文件路径（可选）

    返回:
        list[tuple]: 过滤后的元组列表
    """
    f = TupleFilter(whitelist_path)
    return f.filter(text_tuples)
