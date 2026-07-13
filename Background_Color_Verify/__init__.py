# -*- coding: utf-8 -*-
"""
背景颜色校验模块。

功能：
    对 ROI 内排除 logo 和文字后的背景区域进行颜色检测和大模型分析，
    判断背景颜色是否符合品牌标识规范。

用法：
    from Background_Color_Verify import BackgroundVerifier

    verifier = BackgroundVerifier()
    result = verifier.verify_background(
        roi_image=roi_result["roi_image"],
        roi_elements=roi_result["roi_elements"],
        roi_logo_bbox=roi_result["roi_logo_bbox"],
    )
"""

from .background_verifier import BackgroundVerifier

__all__ = ["BackgroundVerifier"]
