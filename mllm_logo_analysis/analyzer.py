"""Logo 分析核心逻辑。

使用多模态大模型判断图像中是否存在指定 logo。
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import re

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from MLLM_LLM_Component_Tools import MLLMClient


# Brand_Resource_Libraries/logo 目录
LOGO_DIR = _PROJECT_ROOT / "Brand_Resource_Libraries" / "logo"
DEFAULT_LOGO_PATH = LOGO_DIR / "白色logo.png"

WHITE_LOGO_PATH = LOGO_DIR / "白色logo.png"  # 白色 logo
BLACK_LOGO_PATH = LOGO_DIR / "黑色logo.png"  # 黑色 logo
BLUE_LOGO_PATH = LOGO_DIR / "蓝色logo.png"   # 蓝色 logo


class LogoAnalyzer:
    """Logo 分析器"""

    def __init__(self, reference_logo_path: Optional[str] = None):
        """
        初始化 Logo 分析器。

        Args:
            reference_logo_path: 参考图片路径，默认使用 Brand_Resource_Libraries/logo/白色logo.png
        """
        self.reference_logo_path = Path(reference_logo_path) if reference_logo_path else DEFAULT_LOGO_PATH

        if not self.reference_logo_path.exists():
            raise FileNotFoundError(f"参考图片不存在: {self.reference_logo_path}")

        self._mllm = MLLMClient()

    def check_logo_exists(
        self,
        user_image_path: str,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        检查用户图片中是否存在与参考图片相同的 logo。

        Args:
            user_image_path: 用户上传的图片路径
            temperature: 采样温度，默认 0.3（较低温度保证结果稳定）

        Returns:
            包含分析结果的字典
        """
        user_path = Path(user_image_path)
        if not user_path.exists():
            return {
                "success": False,
                "error": f"用户图片不存在: {user_image_path}",
                "logo_exists": False,
            }

        # 构建 prompt，要求模型对比两张图片
        prompt = """请仔细对比这两张图片：

第一张图片是参考图片，包含一个标准的 logo。
第二张图片是用户提供的图片。

请判断：用户图片中是否存在与参考图片相同的 logo？

请按以下格式回答：
1. 是否存在相同的 logo：是/否
2. 置信度：高/中/低
3. 详细说明：简要描述你的判断依据

注意：
- 如果 logo 的颜色、形状、字体等关键特征一致，即使大小、位置不同，也应判定为相同
- 如果 logo 有轻微的变形、旋转或遮挡，但仍可识别为同一 logo，也应判定为相同
- 如果完全不存在 logo 或 logo 明显不同，则判定为不同"""

        try:
            # 调用 MLLM API，传入两张图片
            result = self._mllm.chat(
                prompt=prompt,
                image_paths=[str(self.reference_logo_path), str(user_path)],
                temperature=temperature,
                stream=False,
            )

            # 解析结果
            parsed = self._parse_result(result)
            parsed["raw_response"] = result
            return parsed

        except Exception as e:
            return {
                "success": False,
                "error": f"调用 MLLM API 失败: {str(e)}",
                "logo_exists": False,
                "raw_response": None,
            }

    def _parse_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析 MLLM 返回的结果。

        Args:
            result: MLLM API 返回的结果

        Returns:
            标准化的结果字典
        """
        try:
            # 提取响应内容
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0].get("message", {}).get("content", "")
            elif "data" in result and "choices" in result["data"]:
                content = result["data"]["choices"][0].get("message", {}).get("content", "")
            else:
                content = str(result)

            # 简单的结果解析（基于关键词）
            logo_exists = False
            confidence = "未知"

            # 精确匹配 prompt 要求的回答格式："是否存在相同的 logo：是/否"
            # 兼容中英文冒号，以及"是/否"后可能紧跟的标点或换行
            match = re.search(r"是否存在相同的\s*logo\s*[：:]\s*(是|否)", content)
            if match:
                logo_exists = match.group(1) == "是"
            else:
                # 兜底逻辑：去除问句中的"是否"后再判断
                cleaned = content.replace("是否", "")
                if "否" in cleaned:
                    logo_exists = False
                elif "是" in cleaned or "存在" in cleaned:
                    logo_exists = True

            # 判断置信度（限定在"置信度："之后的片段中查找，避免误匹配）
            conf_match = re.search(r"置信度\s*[：:]\s*(高|中|低)", content)
            if conf_match:
                confidence = conf_match.group(1)

            return {
                "success": True,
                "logo_exists": logo_exists,
                "confidence": confidence,
                "raw_content": content,
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"解析结果失败: {str(e)}",
                "logo_exists": False,
                "raw_content": None,
            }

    def analyze_logo_color(
        self,
        user_image_path: str,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        分析用户图片中 logo 的颜色，判断是否属于白色、黑色或蓝色。

        Args:
            user_image_path: 用户上传的图片路径
            temperature: 采样温度

        Returns:
            包含颜色分析结果的字典，color 字段为 "白色"、"黑色"、"蓝色" 或 None
        """
        user_path = Path(user_image_path)
        if not user_path.exists():
            return {
                "success": False,
                "error": f"用户图片不存在: {user_image_path}",
                "color": None,
            }

        # 检查参考图片是否存在
        if not WHITE_LOGO_PATH.exists():
            return {
                "success": False,
                "error": f"白色参考图片不存在: {WHITE_LOGO_PATH}",
                "color": None,
            }
        if not BLACK_LOGO_PATH.exists():
            return {
                "success": False,
                "error": f"黑色参考图片不存在: {BLACK_LOGO_PATH}",
                "color": None,
            }
        if not BLUE_LOGO_PATH.exists():
            return {
                "success": False,
                "error": f"蓝色参考图片不存在: {BLUE_LOGO_PATH}",
                "color": None,
            }

        prompt = """请仔细对比这四张图片：

第一张图片：白色 logo 参考图片
第二张图片：黑色 logo 参考图片
第三张图片：蓝色 logo 参考图片
第四张图片：用户提供的图片

请判断：用户图片中的 logo 颜色是否与前三张参考图片中的某一张颜色相同？

请严格按照以下格式回答（不要添加任何其他内容）：
颜色：白色/黑色/蓝色/None

判断标准：
- 如果用户图片中的 logo 颜色与白色参考图片一致，输出"颜色：白色"
- 如果用户图片中的 logo 颜色与黑色参考图片一致，输出"颜色：黑色"
- 如果用户图片中的 logo 颜色与蓝色参考图片一致，输出"颜色：蓝色"
- 如果用户图片中的 logo 颜色与以上三种都不一致，输出"颜色：None"

注意：
- 只考虑 logo 本身的颜色，不考虑背景色
- 颜色应该完全一致或非常接近才判定为相同
- 必须严格按照指定格式输出"""

        try:
            result = self._mllm.chat(
                prompt=prompt,
                image_paths=[
                    str(WHITE_LOGO_PATH),
                    str(BLACK_LOGO_PATH),
                    str(BLUE_LOGO_PATH),
                    str(user_path),
                ],
                temperature=temperature,
                stream=False,
            )

            # 提取响应内容
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0].get("message", {}).get("content", "")
            else:
                content = str(result)

            # 解析颜色结果
            color = None
            if "白色" in content and "颜色：白色" in content:
                color = "白色"
            elif "黑色" in content and "颜色：黑色" in content:
                color = "黑色"
            elif "蓝色" in content and "颜色：蓝色" in content:
                color = "蓝色"

            return {
                "success": True,
                "color": color,
                "raw_response": result,
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"颜色分析失败: {str(e)}",
                "color": None,
                "raw_response": None,
            }

    def analyze_logo_font(
        self,
        user_image_path: str,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        分析用户图片中 logo 的字体。

        Args:
            user_image_path: 用户上传的图片路径
            temperature: 采样温度

        Returns:
            包含字体分析结果的字典
        """
        user_path = Path(user_image_path)
        if not user_path.exists():
            return {
                "success": False,
                "error": f"用户图片不存在: {user_image_path}",
            }

        prompt = """请分析这张图片中 logo 使用的字体。

请描述：
1. 字体风格（如：黑体、宋体、艺术字体等）
2. 字体特点（如：粗细、倾斜、间距等）
3. 是否符合 VI 品牌规范（如果能判断的话）

请用简洁清晰的语言回答。"""

        try:
            result = self._mllm.chat(
                prompt=prompt,
                image_paths=[str(user_path)],
                temperature=temperature,
                stream=False,
            )

            # 提取响应内容
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0].get("message", {}).get("content", "")
            else:
                content = str(result)

            return {
                "success": True,
                "font_analysis": content,
                "raw_response": result,
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"字体分析失败: {str(e)}",
                "raw_response": None,
            }
