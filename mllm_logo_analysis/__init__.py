"""Logo 分析模块。

使用多模态大模型进行 Logo 分析，包括：
- Logo 存在性检测
- Logo 颜色分析

注：字体识别与验证已迁移到独立的 Logo_Font_Verify 模块。
"""

from .router import router
from .analyzer import LogoAnalyzer

__all__ = ["router", "LogoAnalyzer"]
