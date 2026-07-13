# -*- coding: utf-8 -*-
"""
品牌标识合规检测模块。

功能：
    根据 Logo_Compliance_Rule_Template.json 规则模板，对已检测出的品牌标识
    元素（Logo、文字框等）进行间距/比例合规判定，输出违规报告。

用法：
    from Logo_Compliance_Check import check_compliance, print_report

    result = check_compliance(
        filtered_tuples=filtered_tuples,
        best_logo=roi_result["best_logo"],
        layout_type=roi_result["layout_type"],
        rule_id=brand_result["rule_id"],
        brand_type=brand_result["brand_type"],
        img_bgr=img_bgr,
    )
    print_report(result)
"""

from .compliance_checker import (
    check_compliance,
    print_report,
    load_annotations,
    save_annotation,
    get_annotation,
    RATIO_MAP,
)
from .visualizer import show_windows, save_gaps_image

__all__ = [
    "check_compliance",
    "print_report",
    "load_annotations",
    "save_annotation",
    "get_annotation",
    "RATIO_MAP",
    "show_windows",
    "save_gaps_image",
]
