# -*- coding: utf-8 -*-
"""
品牌标识规范分类工具

根据 OCR 识别的文字元组，结合 Company_Name_Rule_Library.json 规则库，
按照流程图的决策树逻辑判断品牌标识类型，为后续布局定位算法提供分类依据。

分类决策流程（基于 1-vi_logo规则识别流程图）：
    1. 判断有无"中国南方电网"文字
       ├─ 有 → 判断有无中国南方电网英文文字
       │   ├─ 无英文 → 不同组合形式品牌标识-纯中文
       │   └─ 有英文 → 判断有无使用"公司"、"局"、"中心"等文字
       │       ├─ 有 → 品牌标识组合使用规范
       │       └─ 无 → 判断有无使用"万家灯火南网情深"
       │           ├─ 无 → 不同组合形式品牌标识
       │           └─ 有 → 有无"Light up Every Home, Warm up Every Heart"
       │               ├─ 有 → 品牌标识与品牌口号组合规范-英文
       │               └─ 无 → 品牌标识与品牌口号组合规范-中文
       └─ 无 → 判断有无"CSG"文字
           ├─ 有 → 不同组合形式品牌标识（海外版）-简称
           └─ 无 → 判断有无"CHINA SOUTHERN POWER GRID"文字
               ├─ 有 → 不同组合形式品牌标识（海外版）-全称
               └─ 无 → 无logo文字描述（需进一步判断是否有logo图形）
"""

import json
from pathlib import Path
from typing import Optional

# 规则库默认路径
_DEFAULT_RULE_LIBRARY = Path(__file__).resolve().parent.parent / "Brand_Resource_Libraries" / "Company_Name_Rule_Library.json"


class BrandLogoClassifier:
    """品牌标识规范分类器 - 基于流程图决策树逻辑"""

    def __init__(self, rule_library_path: Optional[str] = None):
        path = Path(rule_library_path) if rule_library_path else _DEFAULT_RULE_LIBRARY
        with open(path, "r", encoding="utf-8") as f:
            self._rules_data = json.load(f)
        self._rules = self._rules_data.get("rules", [])
        self._rules.sort(key=lambda r: r.get("priority", 99))

    def classify(self, text_tuples: list[tuple]) -> dict:
        """
        对 OCR 输出的文字元组进行品牌标识分类。
        按照流程图的决策树逻辑逐步判断。

        参数:
            text_tuples: [(text, poly, orientation), ...] 格式的元组列表

        返回:
            dict: {
                "brand_type": str,
                "rule_id": str | None,
                "layout_types": list[str],
                "matched_keywords": list,
                "description": str,
                "has_logo_text": bool,
                "is_logo_alone": bool,
            }
        """
        all_texts = [t[0] for t in text_tuples]
        combined_upper = " ".join(all_texts).upper()

        # ===== 决策树第1层：判断有无"中国南方电网"文字 =====
        has_chinese_name = self._keyword_in_text("中国南方电网", combined_upper, all_texts)

        if has_chinese_name:
            return self._classify_with_chinese_name(combined_upper, all_texts)
        else:
            return self._classify_without_chinese_name(combined_upper, all_texts, text_tuples)

    def _classify_with_chinese_name(self, combined_upper: str, all_texts: list[str]) -> dict:
        """有"中国南方电网"中文时的分类逻辑"""

        # ===== 决策树第2层：判断有无中国南方电网英文文字 =====
        has_english = (
            self._keyword_in_text("CHINA SOUTHERN", combined_upper, all_texts)
            or self._keyword_in_text("POWER GRID", combined_upper, all_texts)
            or self._keyword_in_text("CSG", combined_upper, all_texts)
        )

        if not has_english:
            # → 不同组合形式品牌标识-纯中文
            return self._build_result(rule_id="R01", matched_keywords=["中国南方电网"])

        # ===== 决策树第3层：判断有无使用"公司"、"局"、"中心"等文字 =====
        subsidiary_keywords = [
            "公司", "局", "厅", "营业厅", "中心", "研究院", "院", "所",
            "广东电网", "广西电网", "云南电网", "贵州电网",
            "海南电网", "深圳供电局", "广州供电局",
            "南网数字", "南网科研院", "南网物资", "南网资本",
            "鼎信信息", "鼎元信息", "南网储能", "南网科技",
        ]
        matched_subsidiary = []
        for kw in subsidiary_keywords:
            if self._keyword_in_text(kw, combined_upper, all_texts):
                matched_subsidiary.append(kw)

        if matched_subsidiary:
            # → 品牌标识组合使用规范
            return self._build_result(rule_id="R02", matched_keywords=["中国南方电网"] + matched_subsidiary)

        # ===== 决策树第4层：判断有无使用"万家灯火南网情深" =====
        slogan_cn_keywords = ["万家灯火南网情深", "万家灯火", "南网情深"]
        matched_slogan_cn = []
        for kw in slogan_cn_keywords:
            if self._keyword_in_text(kw, combined_upper, all_texts):
                matched_slogan_cn.append(kw)

        if not matched_slogan_cn:
            # 无中文口号 → 不同组合形式品牌标识
            return self._build_result(rule_id="R03", matched_keywords=["中国南方电网"])

        # ===== 决策树第5层：有无"Light up Every Home, Warm up Every Heart" =====
        slogan_en_keywords = ["LIGHT UP EVERY HOME", "WARM UP EVERY HEART", "LIGHT UP", "WARM UP"]
        matched_slogan_en = []
        for kw in slogan_en_keywords:
            if self._keyword_in_text(kw, combined_upper, all_texts):
                matched_slogan_en.append(kw)

        if matched_slogan_en:
            # 有中文口号+英文口号 → 品牌标识与品牌口号组合规范-英文
            return self._build_result(rule_id="R04", matched_keywords=["中国南方电网"] + matched_slogan_cn + matched_slogan_en)

        # 有中文口号无英文口号 → 品牌标识与品牌口号组合规范-中文
        return self._build_result(rule_id="R05", matched_keywords=["中国南方电网"] + matched_slogan_cn)

    def _classify_without_chinese_name(self, combined_upper: str, all_texts: list[str], text_tuples: list[tuple]) -> dict:
        """无"中国南方电网"中文时的分类逻辑"""

        # ===== 决策树第2层：判断有无"CSG"文字 =====
        has_csg = self._keyword_in_text("CSG", combined_upper, all_texts)

        if has_csg:
            # → 不同组合形式品牌标识（海外版）-简称
            return self._build_result(rule_id="R06", matched_keywords=["CSG"])

        # ===== 决策树第3层：判断有无"CHINA SOUTHERN POWER GRID"文字 =====
        has_full_english = (
            self._keyword_in_text("CHINA SOUTHERN", combined_upper, all_texts)
            and self._keyword_in_text("POWER GRID", combined_upper, all_texts)
        )

        if has_full_english:
            # → 不同组合形式品牌标识（海外版）-全称
            return self._build_result(rule_id="R07", matched_keywords=["CHINA SOUTHERN", "POWER GRID"])

        # ===== 无任何品牌标识文字 =====
        # 判断是否有logo图形（非文字区域）
        non_empty_texts = [t for t in text_tuples if t[0].strip()]
        if non_empty_texts:
            # 有文字但不是品牌相关文字
            no_logo_text = self._rules_data.get("no_logo_text_result", {})
            return {
                "brand_type": no_logo_text.get("brand_type", "无logo文字描述"),
                "rule_id": None,
                "layout_types": no_logo_text.get("layout_types", []),
                "matched_keywords": [],
                "description": no_logo_text.get("description", "未检测到品牌标识相关文字"),
                "has_logo_text": False,
                "is_logo_alone": False,
            }
        else:
            # 完全没有文字，可能有logo图形
            logo_alone = self._rules_data.get("logo_alone_result", {})
            return {
                "brand_type": logo_alone.get("brand_type", "Logo单独使用错误"),
                "rule_id": None,
                "layout_types": logo_alone.get("layout_types", []),
                "matched_keywords": [],
                "description": logo_alone.get("description", "Logo不得单独使用"),
                "has_logo_text": False,
                "is_logo_alone": True,
            }

    def _build_result(self, rule_id: str, matched_keywords: list[str]) -> dict:
        """根据规则ID构建分类结果"""
        rule = next((r for r in self._rules if r["id"] == rule_id), None)
        if rule is None:
            return {
                "brand_type": "未识别的品牌标识",
                "rule_id": None,
                "layout_types": [],
                "matched_keywords": matched_keywords,
                "description": "规则未找到",
                "has_logo_text": True,
                "is_logo_alone": False,
            }
        return {
            "brand_type": rule["brand_type"],
            "rule_id": rule["id"],
            "layout_types": rule.get("layout_types", []),
            "matched_keywords": matched_keywords,
            "description": rule.get("description", ""),
            "has_logo_text": True,
            "is_logo_alone": False,
        }

    @staticmethod
    def _keyword_in_text(keyword_upper: str, combined_upper: str, all_texts: list[str]) -> bool:
        """
        检查关键字是否在文本中。
        英文关键字用大写匹配，中文关键字逐条原文匹配。
        支持无空格的 OCR 结果（如 CHINASOUTHERNPOWERGRID）。
        """
        if keyword_upper in combined_upper:
            return True
        # 去除关键字中的空格后匹配（OCR 可能丢失空格）
        keyword_no_space = keyword_upper.replace(" ", "")
        if keyword_no_space in combined_upper.replace(" ", ""):
            return True
        # 中文可能在合并时丢失，逐条原文匹配
        for text in all_texts:
            text_upper = text.upper()
            if keyword_upper in text_upper or keyword_upper in text:
                return True
            if keyword_no_space in text_upper.replace(" ", ""):
                return True
        return False


def classify_brand(text_tuples: list[tuple], rule_library_path: Optional[str] = None) -> dict:
    """
    便捷函数：对 OCR 输出的文字元组进行品牌标识分类。

    参数:
        text_tuples: [(text, poly, orientation), ...] 格式的元组列表
        rule_library_path: 规则库 JSON 文件路径（可选）

    返回:
        dict: 分类结果
    """
    classifier = BrandLogoClassifier(rule_library_path)
    return classifier.classify(text_tuples)
