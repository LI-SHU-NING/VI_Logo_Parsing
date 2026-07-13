# -*- coding: utf-8 -*-
"""
背景颜色校验模块。

功能：
    对 ROI 内排除 logo 和文字后的背景区域进行颜色检测和大模型分析，
    判断背景颜色是否符合品牌标识规范。

实现逻辑：
    1. 从主流程接收 ROI 图像、ROI 内文字区域、ROI 内 logo 框
    2. 创建背景掩码（排除 logo + 文字区域），提取背景像素
    3. K-Means 聚类提取背景主导色
    4. 将 3 张颜色版本模板 + ROI 图 + 颜色数据送入 MLLM，判断：
       - Logo 颜色版本（标准彩色/反白/墨稿）
       - 背景色号
       - 是否符合要求
       - 是否干净
       - 描述解释
    5. 解析 MLLM 结构化文本输出，返回 dict

使用标准样例图对比比凭经验识别更准确，因为 MLLM 可以直接视觉比较
用户图片与标准样例的笔画粗细、字形结构、衬线特征。
"""

import sys
import re
import json
import tempfile
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from MLLM_LLM_Component_Tools import MLLMClient

# =========================== 模板配置 ===========================

COLOR_TEMPLATE_DIR = _PROJECT_ROOT / "Brand_Resource_Libraries" / "color_template"
TEMPLATE_PATHS = [
    str(COLOR_TEMPLATE_DIR / "标准彩色版本.png"),
    str(COLOR_TEMPLATE_DIR / "反白版本.png"),
    str(COLOR_TEMPLATE_DIR / "墨稿版本.png"),
]
TEMPLATE_LABELS = ["标准彩色版本", "反白版本", "墨稿版本"]


class BackgroundVerifier:
    """背景颜色校验器。

    使用 MLLM 对品牌标识 ROI 的背景区域进行颜色检测和规范校验。
    不重复 OCR/Logo 检测/ROI 处理，直接接收主流程的结果。
    """

    def __init__(self):
        self._mllm = MLLMClient()

    def verify_background(
        self,
        roi_image: np.ndarray,
        roi_elements: List[Tuple[str, Tuple[int, int, int, int], int]],
        roi_logo_bbox: Optional[Tuple[int, int, int, int]],
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """背景颜色校验。

        从 ROI 图像中排除 logo 和文字区域，提取背景像素，
        用 K-Means 聚类提取主导色，再通过 MLLM 判断背景是否符合品牌规范。

        参数:
            roi_image:     ROI 图像 (BGR numpy array)
            roi_elements:  ROI 内文字区域 [(text, bbox, orientation), ...]
            roi_logo_bbox: ROI 内 logo 框 (x1, y1, x2, y2) 或 None
            temperature:   MLLM 采样温度

        返回:
            {
                "success": bool,
                "logo_version": str,       # 标准彩色版本/反白版本/墨稿版本
                "bg_color": {tone, hex, rgb},
                "is_compliant": bool,      # 是否符合要求
                "is_clean": bool,          # 是否干净
                "has_clutter": bool,       # 是否包含杂乱背景
                "clutter_location": str,
                "description": str,        # 描述解释
                "dominant_colors": [{"bgr": [...], "hex": "#XXXXXX", "ratio": float}, ...],
                "raw_response": dict|None,
                "error": str|None,
            }
        """
        if roi_image is None or roi_image.size == 0:
            return self._error_result("ROI 图像为空")

        roi_h, roi_w = roi_image.shape[:2]

        # Step 1: 创建背景掩码（排除 logo + 文字区域）
        text_bboxes = [b for _, b, _ in roi_elements]
        bg_mask = self._create_background_mask(
            roi_h, roi_w,
            [roi_logo_bbox] if roi_logo_bbox else [],
            text_bboxes,
        )
        bg_pixels = roi_image[bg_mask == 1]

        if len(bg_pixels) == 0:
            return self._error_result("无背景像素，无法进行颜色检测")

        # Step 2: K-Means 颜色检测
        dominant_colors = self._kmeans_colors(bg_pixels, k=5)

        # Step 3: MLLM 分析
        llm_result = self._llm_analyze(roi_image, dominant_colors, temperature)

        if llm_result is None:
            return self._error_result("大模型分析失败或未响应")

        return {
            "success": True,
            "logo_version": llm_result.get("logo_version"),
            "bg_color": {
                "tone": llm_result.get("bg_color_tone"),
                "hex": llm_result.get("bg_color_hex"),
                "rgb": llm_result.get("bg_color_rgb"),
            },
            "is_compliant": llm_result.get("is_compliant"),
            "is_clean": llm_result.get("is_clean"),
            "has_clutter": llm_result.get("has_clutter"),
            "clutter_location": llm_result.get("clutter_location", ""),
            "description": llm_result.get("description", ""),
            "dominant_colors": [
                {"bgr": list(bgr), "hex": self._bgr_to_hex(bgr), "ratio": ratio}
                for bgr, ratio in dominant_colors
            ],
            "raw_response": llm_result.get("_raw"),
            "error": None,
        }

    # ------------------------------------------------------------------
    #  内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _create_background_mask(
        h: int, w: int,
        logo_bboxes: List[Tuple[int, int, int, int]],
        text_bboxes: List[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """在 ROI 图像上创建背景掩码：背景=1，logo/文字区域=0。"""
        mask = np.ones((h, w), dtype=np.uint8)
        for (x1, y1, x2, y2) in list(logo_bboxes) + list(text_bboxes):
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 0
        return mask

    @staticmethod
    def _kmeans_colors(pixels: np.ndarray, k: int = 5) -> List[Tuple[Tuple, float]]:
        """OpenCV K-Means 提取主导色，返回 [(BGR, ratio), ...] 按频率降序。"""
        n = len(pixels)
        if n == 0:
            return []
        if n > 10000:
            idx = np.random.choice(n, 10000, replace=False)
            pixels = pixels[idx]
        pixels_f32 = pixels.astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        k = min(k, len(pixels_f32))
        _, labels, centers = cv2.kmeans(
            pixels_f32, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        centers = centers.astype(np.uint8)
        cnt = Counter(labels.flatten())
        total = sum(cnt.values())
        result = []
        for cluster_id, count in cnt.most_common():
            result.append((tuple(centers[cluster_id].tolist()), count / total))
        return result

    @staticmethod
    def _bgr_to_hex(bgr: Tuple[int, int, int]) -> str:
        """BGR tuple → '#RRGGBB'。"""
        return "#{:02X}{:02X}{:02X}".format(bgr[2], bgr[1], bgr[0])

    @staticmethod
    def _extract_content(result: Any) -> str:
        """从 MLLM 响应提取文本（兼容 OpenAI choices 格式）。"""
        if isinstance(result, dict):
            choices = result.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
            return result.get("content", "")
        return str(result) if result else ""

    def _llm_analyze(
        self,
        roi_bgr: np.ndarray,
        dominant_colors: List[Tuple[Tuple, float]],
        temperature: float,
    ) -> Optional[Dict[str, Any]]:
        """一次 MLLM 调用完成：Logo 颜色版本判别 + 背景质量评估。

        发送 3 模板 + ROI + 颜色数据，返回结构化解析结果，失败返回 None。
        """
        # 检查模板文件
        for tp in TEMPLATE_PATHS:
            if not Path(tp).exists():
                print(f"  [背景校验] 模板不存在: {tp}")
                return None

        # 颜色数据文本
        color_lines = []
        for bgr, ratio in dominant_colors:
            hex_c = self._bgr_to_hex(bgr)
            color_lines.append(f"  {hex_c}  占比 {ratio*100:.1f}%")
        color_text = "\n".join(color_lines) if color_lines else "  (无)"

        prompt = (
            "前面3张是模板logo，按顺序：1标准彩色版本、2反白版本、3墨稿版本。"
            "第4张是待判别的输入ROI图像。\n"
            "\n"
            "墨稿版本logo为黑色；彩色版本logo颜色与反白参考蓝色 #00377A 一致。\n"
            "反白版本参考蓝色: RGB(0,55,122) #00377A（仅作参考色调，不要求严格匹配）\n"
            "\n"
            "背景主导颜色（K-Means聚类，已筛除logo和文字检测框区域，仅统计背景像素）:\n"
            f"{color_text}\n"
            "\n"
            "判断规则:\n"
            "- 根据模板判别输入logo属于哪个颜色版本\n"
            "- 标准彩色/墨稿版本: 要求背景颜色为同一色调即可，可为任意色调，可不为白色（重点）\n"
            "- 反白版本: 背景应为蓝色系（参考 #00377A 色调），此为唯一硬性要求\n"
            "- logo/文字上的瑕疵、脏污、以及背景中的水印、污渍均判定为不干净\n"
            "- logo/文字边缘、正常排版线条、阴影，反光不算杂乱\n"
            "- 色号数据是ROI图像过滤logo与文字检测框后得到的，输出背景色号时，需结合图像中logo附近区域综合作出判断\n"
            "- 掉色判定: 当背景色调符合版本要求，但色号与预期偏差较大，或观察到褪色、氧化变色迹象时，判定为掉色\n"
            "\n"
            "请严格按以下格式逐行输出，不要多余内容:\n"
            "类型: 标准彩色版本/反白版本/墨稿版本\n"
            "背景色号: 色调名称(如蓝色色调)；#XXXXXX RGB(x,x,x)\n"
            "是否符合要求: 是/否\n"
            "是否干净: 是/否\n"
            "描述解释: 详细描述背景情况，包含是否存在掉色、反光、阴影、残缺、脏污等问题...\n"
        )

        # ROI 图像保存为临时文件（MLLMClient 需要文件路径）
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                tmp_path = f.name
            cv2.imwrite(tmp_path, roi_bgr)

            image_paths = list(TEMPLATE_PATHS) + [tmp_path]

            result = self._mllm.chat(
                prompt=prompt,
                image_paths=image_paths,
                temperature=temperature,
                stream=False,
            )

            content = self._extract_content(result)
            parsed = self._parse_llm_output(content)
            if parsed:
                parsed["_raw"] = result
            return parsed

        except Exception as e:
            print(f"  [背景校验] MLLM 请求失败: {e}")
            return None
        finally:
            if tmp_path and Path(tmp_path).exists():
                Path(tmp_path).unlink()

    @staticmethod
    def _parse_llm_output(raw: str) -> Optional[Dict[str, Any]]:
        """解析 LLM 输出的结构化文本，提取为 dict。"""
        result = {}
        for line in raw.split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if key == "类型":
                result["logo_version"] = val
            elif key == "背景色号":
                m = re.match(
                    r"(.+?)\s+(#[0-9A-Fa-f]{6})\s+RGB\((\d+),\s*(\d+),\s*(\d+)\)",
                    val,
                )
                if m:
                    result["bg_color_tone"] = m.group(1).strip()
                    result["bg_color_hex"] = m.group(2).upper()
                    result["bg_color_rgb"] = [
                        int(m.group(3)), int(m.group(4)), int(m.group(5))
                    ]
                else:
                    result["bg_color_tone"] = val
                    result["bg_color_hex"] = ""
                    result["bg_color_rgb"] = []
            elif key == "是否符合要求":
                result["is_compliant"] = val.startswith("是")
            elif key == "是否干净":
                result["is_clean"] = val.startswith("是")
            elif key == "是否包含杂乱背景":
                has = val.startswith("是")
                result["has_clutter"] = has
                result["clutter_location"] = (
                    val.split("，", 1)[-1].strip()
                    if "，" in val and has
                    else ("" if not has else val)
                )
            elif key == "描述解释":
                result["description"] = val
        return result if result else None

    @staticmethod
    def _error_result(error: str) -> Dict[str, Any]:
        """构建错误返回值。"""
        return {
            "success": False,
            "logo_version": None,
            "bg_color": {"tone": None, "hex": None, "rgb": None},
            "is_compliant": None,
            "is_clean": None,
            "has_clutter": None,
            "clutter_location": "",
            "description": "",
            "dominant_colors": [],
            "raw_response": None,
            "error": error,
        }
