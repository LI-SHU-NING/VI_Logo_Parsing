# -*- coding: utf-8 -*-
"""
Logo 字体验证模块。

功能：
    南网品牌标识字体规范校验（verify_brand_font）：
        使用南网标准字体样例图（Brand_Resource_Libraries/Fonts_Image/）校验，
        不识别具体字体类型，只判断是否符合南网字体规范，以 JSON 格式输出。
        3 次 MLLM 调用并发执行（中文、英文全称、英文简称各一次）。
        - 若图中有"中国南方电网" → 对比中文字体标准样例图
        - 若图中有"CHINA SOUTHERN POWER GRID" → 对比英文全称字体标准样例图
        - 若图中有"CSG" → 对比英文简称字体标准样例图

用法：
    from Logo_Font_Verify import FontVerifier

    verifier = FontVerifier()
    result = verifier.verify_brand_font("logo.png")
"""

from .font_verifier import FontVerifier

__all__ = ["FontVerifier"]
